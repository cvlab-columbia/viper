"""
Adding a new functionality is easy. Just implement your new model as a subclass of BaseModel.
The code will make the rest: it will make it available for the processes to call by using
process(name, *args, **kwargs), where *args and **kwargs are the arguments of the models process() method.
"""

import abc
import backoff
import contextlib
import openai
import os
import re
import timeit
import torch
import torchvision
import warnings
from PIL import Image
from collections import Counter
from contextlib import redirect_stdout
from functools import partial
from itertools import chain
from joblib import Memory
from rich.console import Console
from torch import hub
from torch.nn import functional as F
from torchvision import transforms
from typing import List, Union

from configs import config
from utils import HiddenPrints

with open('api.key') as f:
    openai.api_key = f.read().strip()

cache = Memory('cache/' if config.use_cache else None, verbose=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
console = Console(highlight=False)
HiddenPrints = partial(HiddenPrints, console=console, use_newline=config.multiprocessing)


# --------------------------- Base abstract model --------------------------- #

class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]

# ------------------------------ Specific models ---------------------------- #


class ObjectDetector(BaseModel):
    name = 'object_detector'

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number)

        with HiddenPrints('ObjectDetector'):
            detection_model = hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).to(self.dev)
            detection_model.eval()

        self.detection_model = detection_model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """get_object_detection_bboxes"""
        input_batch = image.to(self.dev).unsqueeze(0)  # create a mini-batch as expected by the model
        detections = self.detection_model(input_batch)
        p = detections['pred_boxes']
        p = torch.stack([p[..., 0], 1 - p[..., 3], p[..., 2], 1 - p[..., 1]], -1)  # [left, lower, right, upper]
        detections['pred_boxes'] = p
        return detections


class DepthEstimationModel(BaseModel):
    name = 'depth'

    def __init__(self, gpu_number=0, model_type='DPT_Large'):
        super().__init__(gpu_number)
        with HiddenPrints('DepthEstimation'):
            warnings.simplefilter("ignore")
            # Model options: MiDaS_small, DPT_Hybrid, DPT_Large
            depth_estimation_model = hub.load('intel-isl/MiDaS', model_type, pretrained=True).to(self.dev)
            depth_estimation_model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.depth_estimation_model = depth_estimation_model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """Estimate depth map"""
        image_numpy = image.cpu().permute(1, 2, 0).numpy() * 255
        input_batch = self.transform(image_numpy).to(self.dev)
        prediction = self.depth_estimation_model(input_batch)
        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_numpy.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        # We compute the inverse because the model returns inverse depth
        to_return = 1 / prediction
        to_return = to_return.cpu()
        return to_return  # To save: plt.imsave(path_save, prediction.cpu().numpy())


class CLIPModel(BaseModel):
    name = 'clip'

    def __init__(self, gpu_number=0, version="ViT-L/14@336px"):  # @336px
        super().__init__(gpu_number)

        import clip
        self.clip = clip

        with HiddenPrints('CLIP'):
            model, preprocess = clip.load(version, device=self.dev)
            model.eval()
            model.requires_grad_ = False
        self.model = model
        self.negative_text_features = None
        self.transform = self.get_clip_transforms_from_tensor(336 if "336" in version else 224)

    # @staticmethod
    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    # @staticmethod
    def get_clip_transforms_from_tensor(self, n_px=336):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            self._convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def binary_score(self, image: torch.Tensor, prompt, negative_categories=None):
        is_video = isinstance(image, torch.Tensor) and image.ndim == 4
        if is_video:  # video
            image = torch.stack([self.transform(image[i]) for i in range(image.shape[0])], dim=0)
        else:
            image = self.transform(image).unsqueeze(0).to(self.dev)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        if negative_categories is None:
            if self.negative_text_features is None:
                self.negative_text_features = self.clip_negatives(prompt_prefix)
            negative_text_features = self.negative_text_features
        else:
            negative_text_features = self.clip_negatives(prompt_prefix, negative_categories)

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(image.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        pos_text_features = self.model.encode_text(text)
        pos_text_features = F.normalize(pos_text_features, dim=-1)

        text_features = torch.concat([pos_text_features, negative_text_features], axis=0)

        # run competition where we do a binary classification
        # between the positive and all the negatives, then take the mean
        sim = (100.0 * image_features @ text_features.T).squeeze(dim=0)
        if is_video:
            query = sim[..., 0].unsqueeze(-1).broadcast_to(sim.shape[0], sim.shape[-1] - 1)
            others = sim[..., 1:]
            res = F.softmax(torch.stack([query, others], dim=-1), dim=-1)[..., 0].mean(-1)
        else:
            res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                       sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    @torch.no_grad()
    def clip_negatives(self, prompt_prefix, negative_categories=None):
        if negative_categories is None:
            with open('useful_lists/random_negatives.txt') as f:
                negative_categories = [x.strip() for x in f.read().split()]
        # negative_categories = negative_categories[:1000]
        # negative_categories = ["a cat", "a lamp"]
        negative_categories = [prompt_prefix + x for x in negative_categories]
        negative_tokens = self.clip.tokenize(negative_categories).to(self.dev)

        negative_text_features = self.model.encode_text(negative_tokens)
        negative_text_features = F.normalize(negative_text_features, dim=-1)

        return negative_text_features

    @torch.no_grad()
    def classify(self, image: Union[torch.Tensor, list], categories: list[str], return_index=True):
        is_list = isinstance(image, list)
        if is_list:
            assert len(image) == len(categories)
            image = [self.transform(x).unsqueeze(0) for x in image]
            image_clip = torch.cat(image, dim=0).to(self.dev)
        elif len(image.shape) == 3:
            image_clip = self.transform(image).to(self.dev).unsqueeze(0)
        else:  # Video (process images separately)
            image_clip = torch.stack([self.transform(x) for x in image], dim=0).to(self.dev)

        # if len(image_clip.shape) == 3:
        #     image_clip = image_clip.unsqueeze(0)

        prompt_prefix = "photo of "
        categories = [prompt_prefix + x for x in categories]
        categories = self.clip.tokenize(categories).to(self.dev)

        text_features = self.model.encode_text(categories)
        text_features = F.normalize(text_features, dim=-1)

        image_features = self.model.encode_image(image_clip)
        image_features = F.normalize(image_features, dim=-1)

        if image_clip.shape[0] == 1:
            # get category from image
            softmax_arg = image_features @ text_features.T  # 1 x n
        else:
            if is_list:
                # get highest category-image match with n images and n corresponding categories
                softmax_arg = (image_features @ text_features.T).diag().unsqueeze(0)  # n x n -> 1 x n
            else:
                softmax_arg = (image_features @ text_features.T)

        similarity = (100.0 * softmax_arg).softmax(dim=-1).squeeze(0)
        if not return_index:
            return similarity
        else:
            result = torch.argmax(similarity, dim=-1)
            if result.shape == ():
                result = result.item()
            return result

    @torch.no_grad()
    def compare(self, images: list[torch.Tensor], prompt, return_scores=False):
        images = [self.transform(im).unsqueeze(0).to(self.dev) for im in images]
        images = torch.cat(images, dim=0)

        prompt_prefix = "photo of "
        prompt = prompt_prefix + prompt

        text = self.clip.tokenize([prompt]).to(self.dev)

        image_features = self.model.encode_image(images.to(self.dev))
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        sim = (image_features @ text_features.T).squeeze(dim=-1)  # Only one text, so squeeze

        if return_scores:
            return sim
        res = sim.argmax()
        return res

    def forward(self, image, prompt, task='score', return_index=True, negative_categories=None, return_scores=False):
        if task == 'classify':
            categories = prompt
            clip_sim = self.classify(image, categories, return_index=return_index)
            out = clip_sim
        elif task == 'score':
            clip_score = self.binary_score(image, prompt, negative_categories=negative_categories)
            out = clip_score
        else:  # task == 'compare'
            idx = self.compare(image, prompt, return_scores)
            out = idx
        if not isinstance(out, int):
            out = out.cpu()
        return out


class MaskRCNNModel(BaseModel):
    name = 'maskrcnn'

    def __init__(self, gpu_number=0, threshold=config.detect_thresholds.maskrcnn):
        super().__init__(gpu_number)
        with HiddenPrints('MaskRCNN'):
            obj_detect = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1').to(self.dev)
            obj_detect.eval()
            obj_detect.requires_grad_(False)
        self.categories = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
        self.obj_detect = obj_detect
        self.threshold = threshold

    def prepare_image(self, image):
        image = image.to(self.dev)
        return image

    @torch.no_grad()
    def detect(self, images: torch.Tensor, return_labels=True):
        if type(images) != list:
            images = [images]
        images = [self.prepare_image(im) for im in images]
        detections = self.obj_detect(images)
        for i in range(len(images)):
            height = detections[i]['masks'].shape[-2]
            # Just return boxes (no labels no masks, no scores) with scores > threshold
            if return_labels:  # In the current implementation, we only return labels
                d_i = detections[i]['labels'][detections[i]['scores'] > self.threshold]
                detections[i] = set([self.categories[d] for d in d_i])
            else:
                d_i = detections[i]['boxes'][detections[i]['scores'] > self.threshold]
                # Return [left, lower, right, upper] instead of [left, upper, right, lower]
                detections[i] = torch.stack([d_i[:, 0], height - d_i[:, 3], d_i[:, 2], height - d_i[:, 1]], dim=1)

        return detections

    def forward(self, image, return_labels=False):
        obj_detections = self.detect(image, return_labels)
        # Move to CPU before sharing. Alternatively we can try cloning tensors in CUDA, but may not work
        obj_detections = [(v.to('cpu') if isinstance(v, torch.Tensor) else list(v)) for v in obj_detections]
        return obj_detections


class OwlViTModel(BaseModel):
    name = 'owlvit'

    def __init__(self, gpu_number=0, threshold=config.detect_thresholds.owlvit):
        super().__init__(gpu_number)

        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        with HiddenPrints("OwlViT"):
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            model.eval()
            model.requires_grad_(False)
        self.model = model.to(self.dev)
        self.processor = processor
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, image: torch.Tensor, text: List[str], return_labels: bool = False):
        if isinstance(image, list):
            raise TypeError("image has to be a torch tensor, not a list")
        if isinstance(text, str):
            text = [text]
        text_original = text
        text = ['a photo of a ' + t for t in text]
        inputs = self.processor(text=text, images=image, return_tensors="pt") # padding="longest",
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([image.shape[1:]]).to(self.dev)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        indices_good = scores > self.threshold
        boxes = boxes[indices_good]

        # Change to format where large "upper"/"lower" means more up
        left, upper, right, lower = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        height = image.shape[-2]
        boxes = torch.stack([left, height - lower, right, height - upper], -1)

        if return_labels:
            labels = labels[indices_good]
            labels = [text_original[lab].re('a photo of a ') for lab in labels]
            return boxes, labels

        return boxes.cpu()  # [x_min, y_min, x_max, y_max]


class GLIPModel(BaseModel):
    name = 'glip'

    def __init__(self, model_size='large', gpu_number=0, *args):
        BaseModel.__init__(self, gpu_number)

        with contextlib.redirect_stderr(open(os.devnull, "w")):  # Do not print nltk_data messages when importing
            from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
                create_positive_map_label_to_token_from_positive_map

        working_dir = f'{config.path_pretrained_models}/GLIP/'
        if model_size == 'tiny':
            config_file = working_dir + "configs/glip_Swin_T_O365_GoldG.yaml"
            weight_file = working_dir + "checkpoints/glip_tiny_model_o365_goldg_cc_sbu.pth"
        else:  # large
            config_file = working_dir + "configs/glip_Swin_L.yaml"
            weight_file = working_dir + "checkpoints/glip_large_model.pth"

        class OurGLIPDemo(GLIPDemo):

            def __init__(self, dev, *args_demo):

                kwargs = {
                    'min_image_size': 800,
                    'confidence_threshold': config.detect_thresholds.glip,
                    'show_mask_heatmaps': False
                }

                self.dev = dev

                from maskrcnn_benchmark.config import cfg

                # manual override some options
                cfg.local_rank = 0
                cfg.num_gpus = 1
                cfg.merge_from_file(config_file)
                cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
                cfg.merge_from_list(["MODEL.DEVICE", self.dev])

                with HiddenPrints("GLIP"), torch.cuda.device(self.dev):
                    from transformers.utils import logging
                    logging.set_verbosity_error()
                    GLIPDemo.__init__(self, cfg, *args_demo, **kwargs)
                if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                    plus = 1
                else:
                    plus = 0
                self.plus = plus
                self.color = 255

            @torch.no_grad()
            def compute_prediction(self, original_image, original_caption, custom_entity=None):
                image = self.transforms(original_image)
                # image = [image, image.permute(0, 2, 1)]
                image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
                image_list = image_list.to(self.dev)
                # caption
                if isinstance(original_caption, list):

                    if len(original_caption) > 40:
                        all_predictions = None
                        for loop_num, i in enumerate(range(0, len(original_caption), 40)):
                            list_step = original_caption[i:i + 40]
                            prediction_step = self.compute_prediction(original_image, list_step, custom_entity=None)
                            if all_predictions is None:
                                all_predictions = prediction_step
                            else:
                                # Aggregate predictions
                                all_predictions.bbox = torch.cat((all_predictions.bbox, prediction_step.bbox), dim=0)
                                for k in all_predictions.extra_fields:
                                    all_predictions.extra_fields[k] = \
                                        torch.cat((all_predictions.extra_fields[k],
                                                   prediction_step.extra_fields[k] + loop_num), dim=0)
                        return all_predictions

                    # we directly provided a list of category names
                    caption_string = ""
                    tokens_positive = []
                    seperation_tokens = " . "
                    for word in original_caption:
                        tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                        caption_string += word
                        caption_string += seperation_tokens

                    tokenized = self.tokenizer([caption_string], return_tensors="pt")
                    # tokens_positive = [tokens_positive]  # This was wrong
                    tokens_positive = [[v] for v in tokens_positive]

                    original_caption = caption_string
                    # print(tokens_positive)
                else:
                    tokenized = self.tokenizer([original_caption], return_tensors="pt")
                    if custom_entity is None:
                        tokens_positive = self.run_ner(original_caption)
                    # print(tokens_positive)
                # process positive map
                positive_map = create_positive_map(tokenized, tokens_positive)

                positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map,
                                                                                                   plus=self.plus)
                self.positive_map_label_to_token = positive_map_label_to_token
                tic = timeit.time.perf_counter()

                # compute predictions
                with HiddenPrints():   # Hide some deprecated notices
                    predictions = self.model(image_list, captions=[original_caption],
                                             positive_map=positive_map_label_to_token)
                predictions = [o.to(self.cpu_device) for o in predictions]
                # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

                # always single image is passed at a time
                prediction = predictions[0]

                # reshape prediction (a BoxList) into the original image size
                height, width = original_image.shape[-2:]
                # if self.tensor_inputs:
                # else:
                #     height, width = original_image.shape[:-1]
                prediction = prediction.resize((width, height))

                if prediction.has_field("mask"):
                    # if we have masks, paste the masks in the right position
                    # in the image, as defined by the bounding boxes
                    masks = prediction.get_field("mask")
                    # always single image is passed at a time
                    masks = self.masker([masks], [prediction])[0]
                    prediction.add_field("mask", masks)

                return prediction

            @staticmethod
            def to_left_right_upper_lower(bboxes):
                return [(bbox[1], bbox[3], bbox[0], bbox[2]) for bbox in bboxes]

            @staticmethod
            def to_xmin_ymin_xmax_ymax(bboxes):
                # invert the previous method
                return [(bbox[2], bbox[0], bbox[3], bbox[1]) for bbox in bboxes]

            @staticmethod
            def prepare_image(image):
                image = image[[2, 1, 0]]  # convert to bgr for opencv-format for glip
                return image

            @torch.no_grad()
            def forward(self, image: torch.Tensor, obj: Union[str, list], return_labels: bool = False,
                        confidence_threshold=None):

                if confidence_threshold is not None:
                    original_confidence_threshold = self.confidence_threshold
                    self.confidence_threshold = confidence_threshold

                # if isinstance(object, list):
                #     object = ' . '.join(object) + ' .' # add separation tokens
                image = self.prepare_image(image)

                # Avoid the resizing creating a huge image in a pathological case
                ratio = image.shape[1] / image.shape[2]
                ratio = max(ratio, 1 / ratio)
                original_min_image_size = self.min_image_size
                if ratio > 10:
                    self.min_image_size = int(original_min_image_size * 10 / ratio)
                    self.transforms = self.build_transform()

                with torch.cuda.device(self.dev):
                    inference_output = self.inference(image, obj)

                bboxes = inference_output.bbox.cpu().numpy().astype(int)
                # bboxes = self.to_left_right_upper_lower(bboxes)

                if ratio > 10:
                    self.min_image_size = original_min_image_size
                    self.transforms = self.build_transform()

                bboxes = torch.tensor(bboxes)

                # Convert to [left, lower, right, upper] instead of [left, upper, right, lower]
                height = image.shape[-2]
                bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)

                if confidence_threshold is not None:
                    self.confidence_threshold = original_confidence_threshold
                if return_labels:
                    # subtract 1 because it's 1-indexed for some reason
                    return bboxes, inference_output.get_field("labels").cpu().numpy() - 1
                return bboxes

        self.glip_demo = OurGLIPDemo(*args, dev=self.dev)

    def forward(self, *args, **kwargs):
        return self.glip_demo.forward(*args, **kwargs)


class TCLModel(BaseModel):
    name = 'tcl'

    def __init__(self, gpu_number=0):

        from base_models.tcl.tcl_model_pretrain import ALBEF
        from base_models.tcl.tcl_vit import interpolate_pos_embed
        from base_models.tcl.tcl_tokenization_bert import BertTokenizer

        super().__init__(gpu_number)
        config = {
            'image_res': 384,
            'mlm_probability': 0.15,
            'embed_dim': 256,
            'vision_width': 768,
            'bert_config': 'base_models/tcl_config_bert.json',
            'temp': 0.07,
            'queue_size': 65536,
            'momentum': 0.995,
        }

        text_encoder = 'bert-base-uncased'
        checkpoint_path = f'{config.path_pretrained_models}/TCL_4M.pth'

        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)

        with warnings.catch_warnings(), HiddenPrints("TCL"):
            model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=self.tokenizer)

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model']

            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
            model.load_state_dict(state_dict, strict=False)

        self.model = model.to(self.dev)
        self.model.eval()

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        self.negative_text_features = None

    def transform(self, image):
        image = transforms.ToPILImage()(image)
        image = self.test_transform(image)
        return image

    def prepare_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.dev)
        return image

    @torch.no_grad()
    def binary_score(self, images: Union[list[torch.Tensor], torch.Tensor], prompt):
        single_image = False
        if isinstance(images, torch.Tensor):
            single_image = True
            images = [images]
        images = [self.prepare_image(im) for im in images]
        images = torch.cat(images, dim=0)

        first_words = ['description', 'caption', 'alt text']
        second_words = ['photo', 'image', 'picture']
        options = [f'{fw}: {sw} of a' for fw in first_words for sw in second_words]

        prompts = [f'{option} {prompt}' for option in options]

        text_input = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=30, return_tensors="pt") \
            .to(self.dev)
        text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                              mode='text')
        text_feats = text_output  # .last_hidden_state
        text_atts = text_input.attention_mask

        image_feats = self.model.visual_encoder(images)

        img_len = image_feats.shape[0]
        text_len = text_feats.shape[0]
        image_feats = image_feats.unsqueeze(1).repeat(1, text_len, 1, 1).view(-1, *image_feats.shape[-2:])
        text_feats = text_feats.unsqueeze(0).repeat(img_len, 1, 1, 1).view(-1, *text_feats.shape[-2:])
        text_atts = text_atts.unsqueeze(0).repeat(img_len, 1, 1).view(-1, *text_atts.shape[-1:])

        image_feats_att = torch.ones(image_feats.size()[:-1], dtype=torch.long).to(self.dev)
        output = self.model.text_encoder(encoder_embeds=text_feats, attention_mask=text_atts,
                                         encoder_hidden_states=image_feats, encoder_attention_mask=image_feats_att,
                                         return_dict=True, mode='fusion')

        scores = self.model.itm_head(output[:, 0, :])[:, 1]
        scores = scores.view(img_len, text_len)
        score = scores.sigmoid().max(-1)[0]

        if single_image:
            score = score.item()

        return score

    @torch.no_grad()
    def classify(self, image, texts, return_index=True):
        if isinstance(image, list):
            assert len(image) == len(texts)
            image = [self.transform(x).unsqueeze(0) for x in image]
            image_tcl = torch.cat(image, dim=0).to(self.dev)
        else:
            image_tcl = self.prepare_image(image)

        text_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt") \
            .to(self.dev)
        text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                              mode='text')
        text_feats = text_output  # .last_hidden_state
        text_embeds = F.normalize(self.model.text_proj(text_feats[:, 0, :]))
        text_atts = text_input.attention_mask

        image_feats = self.model.visual_encoder(image_tcl)
        image_embeds = self.model.vision_proj(image_feats[:, 0, :])
        image_embeds = F.normalize(image_embeds, dim=-1)

        # In the original code, this is only used to select the topk pairs, to not compute ITM head on all pairs.
        # But other than that, not used
        sims_matrix = image_embeds @ text_embeds.t()
        sims_matrix_t = sims_matrix.t()

        # Image-Text Matching (ITM): Binary classifier for every image-text pair
        # Only one direction, because we do not filter bet t2i, i2t, and do all pairs

        image_feats_att = torch.ones(image_feats.size()[:-1], dtype=torch.long).to(self.dev)
        output = self.model.text_encoder(encoder_embeds=text_feats, attention_mask=text_atts,
                                         encoder_hidden_states=image_feats, encoder_attention_mask=image_feats_att,
                                         return_dict=True, mode='fusion')

        score_matrix = self.model.itm_head(output[:, 0, :])[:, 1]

        if not return_index:
            return score_matrix
        else:
            return torch.argmax(score_matrix).item()

    def forward(self, image, texts, task='classify', return_index=True):
        if task == 'classify':
            best_text = self.classify(image, texts, return_index=return_index)
            out = best_text
        else:  # task == 'score':  # binary_score
            score = self.binary_score(image, texts)
            out = score
        if isinstance(out, torch.Tensor):
            out = out.cpu()
        return out


@cache.cache(ignore=['result'])
def gpt3_cache_aux(fn_name, prompts, temperature, n_votes, result):
    """
    This is a trick to manually cache results from GPT-3. We want to do it manually because the queries to GPT-3 are
    batched, and caching doesn't make sense for batches. With this we can separate individual samples in the batch
    """
    return result


class GPT3Model(BaseModel):
    name = 'gpt3'
    to_batch = False
    requires_gpu = False

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=gpu_number)
        with open(config.gpt3.qa_prompt) as f:
            self.qa_prompt = f.read().strip()
        self.temperature = config.gpt3.temperature
        self.n_votes = config.gpt3.n_votes
        self.model = config.gpt3.model

    # initial cleaning for reference QA results
    @staticmethod
    def process_answer(answer):
        answer = answer.lstrip()  # remove leading spaces (our addition)
        answer = answer.replace('.', '').replace(',', '').lower()
        to_be_removed = {'a', 'an', 'the', 'to', ''}
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return ' '.join(answer_list)

    @staticmethod
    def get_union(lists):
        return list(set(chain.from_iterable(lists)))

    @staticmethod
    def most_frequent(answers):
        answer_counts = Counter(answers)
        return answer_counts.most_common(1)[0][0]

    def get_qa(self, prompts, prompt_base: str=None) -> list[str]:
        if prompt_base is None:
            prompt_base = self.qa_prompt
        prompts_total = []
        for p in prompts:
            question = p
            prompts_total.append(prompt_base.format(question))
        response = self.get_qa_fn(prompts_total)
        if self.n_votes > 1:
            response_ = []
            for i in range(len(prompts)):
                if self.model == 'chatgpt':
                    resp_i = [r['message']['content']
                              for r in response['choices'][i * self.n_votes:(i + 1) * self.n_votes]]
                else:
                    resp_i = [r['text'] for r in response['choices'][i * self.n_votes:(i + 1) * self.n_votes]]
                response_.append(self.most_frequent(resp_i))
            response = response_
        else:
            if self.model == 'chatgpt':
                response = [r['message']['content'] for r in response['choices']]
            else:
                response = [self.process_answer(r["text"]) for r in response['choices']]
        return response

    def get_qa_fn(self, prompt):
        response = self.query_gpt3(prompt, model=self.model, max_tokens=5, logprobs=1, stream=False,
                                   stop=["\n", "<|endoftext|>"])
        return response

    def get_general(self, prompts) -> list[str]:
        if self.model == "chatgpt":
            raise NotImplementedError
        response = self.query_gpt3(prompts, model=self.model, max_tokens=256, top_p=1, frequency_penalty=0,
                                   presence_penalty=0)
        response = [r["text"] for r in response['choices']]
        return response

    def query_gpt3(self, prompt, model="text-davinci-003", max_tokens=16, logprobs=None, stream=False,
                   stop=None, top_p=1, frequency_penalty=0, presence_penalty=0):
        if model == "chatgpt":
            messages = [{"role": "user", "content": p} for p in prompt]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                logprobs=logprobs,
                temperature=self.temperature,
                stream=stream,
                stop=stop,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=self.n_votes,
            )
        return response

    def forward(self, prompt, process_name):
        if not self.to_batch:
            prompt = [prompt]
        
        if process_name == 'gpt3_qa':
            # if items in prompt are tuples, then we assume it is a question and context
            if isinstance(prompt[0], tuple) or isinstance(prompt[0], list):
                prompt = [question.format(context) for question, context in prompt]

        to_compute = None
        results = []
        # Check if in cache
        if config.use_cache:
            for p in prompt:
                # This is not ideal, because if not found, later it will have to re-hash the arguments.
                # But I could not find a better way to do it.
                result = gpt3_cache_aux(process_name, p, self.temperature, self.n_votes, None)
                results.append(result)  # If in cache, will be actual result, otherwise None
            to_compute = [i for i, r in enumerate(results) if r is None]
            prompt = [prompt[i] for i in to_compute]

        if len(prompt) > 0:
            if process_name == 'gpt3_qa':
                response = self.get_qa(prompt)
            else:  # 'gpt3_general', general prompt, has to be given all of it
                response = self.get_general(prompt)
        else:
            response = []  # All previously cached

        if config.use_cache:
            for p, r in zip(prompt, response):
                # "call" forces the overwrite of the cache
                gpt3_cache_aux.call(process_name, p, self.temperature, self.n_votes, r)
            for i, idx in enumerate(to_compute):
                results[idx] = response[i]
        else:
            results = response

        if not self.to_batch:
            results = results[0]
        return results

    @classmethod
    def list_processes(cls):
        return ['gpt3_' + n for n in ['qa', 'general']]


# @cache.cache
@backoff.on_exception(backoff.expo, Exception, max_tries=10)
def codex_helper(extended_prompt):
    assert 0 <= config.codex.temperature <= 1
    assert 1 <= config.codex.best_of <= 20

    if config.codex.model in ("gpt-4", "gpt-3.5-turbo"):
        if not isinstance(extended_prompt, list):
            extended_prompt = [extended_prompt]
        responses = [openai.ChatCompletion.create(
                model=config.codex.model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": "Only answer with a function starting def execute_command."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.codex.temperature,
                max_tokens=config.codex.max_tokens,
                top_p = 1.,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=config.codex.best_of,
                stop=["\n\n"],
                )
                    for prompt in extended_prompt]
        resp = [r['choices'][0]['message']['content'] for r in responses]
        if len(resp) == 1:
            resp = resp[0]
    else:
        response = openai.Completion.create(
            model="code-davinci-002",
            temperature=config.codex.temperature,
            prompt=extended_prompt,
            max_tokens=config.codex.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=config.codex.best_of,
            stop=["\n\n"],
        )

        if isinstance(extended_prompt, list):
            resp = [r['text'] for r in response['choices']]
        else:
            resp = response['choices'][0]['text']

    return resp


class CodexModel(BaseModel):
    name = 'codex'
    requires_gpu = False
    max_batch_size = 5

    # Not batched, but every call will probably be a batch (coming from the same process)

    def __init__(self, gpu_number=0):
        super().__init__(gpu_number=0)
        with open(config.codex.prompt) as f:
            self.base_prompt = f.read().strip()
        self.fixed_code = None
        if config.use_fixed_code:
            with open(config.fixed_code_file) as f:
                self.fixed_code = f.read()

    def forward(self, prompt, input_type='image', prompt_file=None, base_prompt=None):
        if config.use_fixed_code:  # Use the same program for every sample, like in socratic models
            return [self.fixed_code] * len(prompt) if isinstance(prompt, list) else self.fixed_code

        if prompt_file is not None and base_prompt is None:  # base_prompt takes priority
            with open(prompt_file) as f:
                base_prompt = f.read().strip()
        elif base_prompt is None:
            base_prompt = self.base_prompt

        if isinstance(prompt, list):
            extended_prompt = [base_prompt.replace("INSERT_QUERY_HERE", p).replace('INSERT_TYPE_HERE', input_type)
                               for p in prompt]
        elif isinstance(prompt, str):
            extended_prompt = [base_prompt.replace("INSERT_QUERY_HERE", prompt).
                               replace('INSERT_TYPE_HERE', input_type)]
        else:
            raise TypeError("prompt must be a string or a list of strings")

        result = self.forward_(extended_prompt)
        if not isinstance(prompt, list):
            result = result[0]

        return result

    def forward_(self, extended_prompt):
        if len(extended_prompt) > self.max_batch_size:
            response = []
            for i in range(0, len(extended_prompt), self.max_batch_size):
                response += self.forward_(extended_prompt[i:i + self.max_batch_size])
        try:
            response = codex_helper(extended_prompt)
        except openai.error.RateLimitError as e:
            print("Retrying Codex, splitting batch")
            if len(extended_prompt) == 1:
                warnings.warn("This is taking too long, maybe OpenAI is down? (status.openai.com/)")
            # Will only be here after the number of retries in the backoff decorator.
            # It probably means a single batch takes up the entire rate limit.
            sub_batch_1 = extended_prompt[:len(extended_prompt) // 2]
            sub_batch_2 = extended_prompt[len(extended_prompt) // 2:]
            if len(sub_batch_1) > 0:
                response_1 = self.forward_(sub_batch_1)
            else:
                response_1 = []
            if len(sub_batch_2) > 0:
                response_2 = self.forward_(sub_batch_2)
            else:
                response_2 = []
            response = response_1 + response_2
        except Exception as e:
            # Some other error like an internal OpenAI error
            print("Retrying Codex")
            print(e)
            response = self.forward_(extended_prompt)
        return response


class BLIPModel(BaseModel):
    name = 'blip'
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2  # The queue has additionally the time it is executing the previous forward pass

    def __init__(self, gpu_number=0, half_precision=config.blip_half_precision,
                 blip_v2_model_type=config.blip_v2_model_type):
        super().__init__(gpu_number)

        # from lavis.models import load_model_and_preprocess
        from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, \
            Blip2ForConditionalGeneration

        # https://huggingface.co/models?sort=downloads&search=Salesforce%2Fblip2-
        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']

        with warnings.catch_warnings(), HiddenPrints("BLIP"), torch.cuda.device(self.dev):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}

            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
            # Device_map must be sequential for manual GPU selection
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    f"Salesforce/{blip_v2_model_type}", load_in_8bit=half_precision,
                    torch_dtype=torch.float16 if half_precision else "auto",
                    device_map="sequential", max_memory=max_memory
                )
            except Exception as e:
                # Clarify error message. The problem is that it tries to load part of the model to disk.
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e

        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.half_precision = half_precision
        self.max_words = 50

    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.dev, torch.float16)
        generated_ids = self.model.generate(**inputs, length_penalty=1., num_beams=5, max_length=30, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = [cap.strip() for cap in
                          self.processor.batch_decode(generated_ids, skip_special_tokens=True)]
        return generated_text
    
    def pre_question(self, question):
        # from LAVIS blip_processors
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]

        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']

        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []

        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))

        if not self.to_batch:
            response = response[0]
        return response


class SaliencyModel(BaseModel):
    name = 'saliency'

    def __init__(self, gpu_number=0,
                 path_checkpoint=f'{config.path_pretrained_models}/saliency_inspyrenet_plus_ultra'):

        from base_models.inspyrenet.saliency_transforms import get_transform
        from base_models.inspyrenet.InSPyReNet import InSPyReNet
        from base_models.inspyrenet.backbones.SwinTransformer import SwinB

        # These parameters are for the Plus Ultra LR model
        super().__init__(gpu_number)
        depth = 64
        pretrained = True
        base_size = [384, 384]
        kwargs = {'name': 'InSPyReNet_SwinB', 'threshold': 512}
        with HiddenPrints("Saliency"):
            model = InSPyReNet(SwinB(pretrained=pretrained, path_pretrained_models=config.path_pretrained_models),
                               [128, 128, 256, 512, 1024], depth, base_size, **kwargs)
            model.load_state_dict(torch.load(os.path.join(path_checkpoint, 'latest.pth'),
                                             map_location=torch.device('cpu')), strict=True)
        model = model.to(self.dev)
        model.eval()

        self.model = model
        self.transform_pil = transforms.ToPILImage()
        self.transform = get_transform({
            'static_resize': {'size': [384, 384]},
            'dynamic_resize': {'L': 1280},
            'tonumpy': None,
            'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'totensor': None
        })

    @torch.no_grad()
    def forward(self, image):
        image_t = self.transform({'image': self.transform_pil(image)})
        image_t['image_resized'] = image_t['image_resized'].unsqueeze(0).to(self.dev)
        image_t['image'] = image_t['image'].unsqueeze(0).to(self.dev)
        pred = self.model(image_t)['pred']
        pred_resized = F.interpolate(pred, image.shape[1:], mode='bilinear', align_corners=True)[0, 0]
        mask_foreground = pred_resized < 0.5
        image_masked = image.clone()
        image_masked[:, mask_foreground] = 0

        return image_masked


class XVLMModel(BaseModel):
    name = 'xvlm'

    def __init__(self, gpu_number=0,
                 path_checkpoint=f'{config.path_pretrained_models}/xvlm/retrieval_mscoco_checkpoint_9.pth'):

        from base_models.xvlm.xvlm import XVLMBase
        from transformers import BertTokenizer

        super().__init__(gpu_number)

        image_res = 384
        self.max_words = 30
        config_xvlm = {
            'image_res': image_res,
            'patch_size': 32,
            'text_encoder': 'bert-base-uncased',
            'block_num': 9,
            'max_tokens': 40,
            'embed_dim': 256,
        }

        vision_config = {
            'vision_width': 1024,
            'image_res': 384,
            'window_size': 12,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32]
        }
        with warnings.catch_warnings(), HiddenPrints("XVLM"):
            model = XVLMBase(config_xvlm, use_contrastive_loss=True, vision_config=vision_config)

            checkpoint = torch.load(path_checkpoint, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) > 0:
            print('XVLM Missing keys: ', msg.missing_keys)

        model = model.to(self.dev)
        model.eval()

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        with open('useful_lists/random_negatives.txt') as f:
            self.negative_categories = [x.strip() for x in f.read().split()]

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption

    @torch.no_grad()
    def score(self, images, texts):

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(images, list):
            images = [images]

        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).to(self.dev)

        texts = [self.pre_caption(text, self.max_words) for text in texts]
        text_input = self.tokenizer(texts, padding='longest', return_tensors="pt").to(self.dev)

        image_embeds, image_atts = self.model.get_vision_embeds(images)
        text_ids, text_atts = text_input.input_ids, text_input.attention_mask
        text_embeds = self.model.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.model.get_features(image_embeds, text_embeds)
        logits = image_feat @ text_feat.t()

        return logits

    @torch.no_grad()
    def binary_score(self, image, text, negative_categories):
        # Compare with a pre-defined set of negatives
        texts = [text] + negative_categories
        sim = 100 * self.score(image, texts)[0]
        res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                   sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
        return res

    def forward(self, image, text, task='score', negative_categories=None):
        if task == 'score':
            score = self.score(image, text)
        else:  # binary
            score = self.binary_score(image, text, negative_categories=negative_categories)
        return score.cpu()
