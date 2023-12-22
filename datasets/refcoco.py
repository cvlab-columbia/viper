import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import json
import pickle
import itertools
import torch
from torchvision.ops import box_iou


class RefCOCODataset(Dataset):
    """
    Used code from https://github.com/lichengunc/refer/blob/master/refer.py
    """
    def __init__(self, split, data_path="", image_transforms=None, question_transforms=None, tokenize=None,
                 max_samples=None, version='refcoco', split_by='unc', **kwargs):

        self.split = split
        self.data_path = data_path
        self.max_samples = max_samples
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        self.input_type = 'image'

        assert version in ['refcoco', 'refcoco+', 'refcocog']

        # load refs from data/dataset/refs(dataset).json
        ref_file = os.path.join(data_path, version, 'refs(' + split_by + ').p')
        with open(ref_file, 'rb') as f:
            self.refs = pickle.load(f)

        # load annotations from data/dataset/instances.json

        instances_file = os.path.join(data_path, version, 'instances.json')
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        self.images = instances['images']
        self.annotations = instances['annotations']
        self.categories = instances['categories']

        self.create_index()

        ref_ids = self.get_ref_ids(split=split)
        self.samples = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            for i in range(len(ref['sent_ids'])):
                self.samples.append((ref_id, i))

        np.random.seed(4)
        np.random.shuffle(self.samples)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def create_index(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}

        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for img in self.images:
            Imgs[img['id']] = img
        for ann in self.annotations:
            Anns[ann['id']] = ann
            height = Imgs[ann['image_id']]['height']
            ann['bbox'] = [ann['bbox'][0], height-(ann['bbox'][1]+ann['bbox'][3]), ann['bbox'][2]+ann['bbox'][0],
                           height-ann['bbox'][1]]
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for cat in self.categories:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.refs:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens

    def get_ref_ids(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.refs
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if
                            split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise KeyError(f'No split {split}')
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def get_ann_ids(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.annotations]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.annotations
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def get_img_ids(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def load_refs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def get_index_from_sample_id(self, sample_id):
        return sample_id

    def get_sample_path(self, index=None, ref=None):
        if ref is None:
            assert index is not None
            ref_id, i = self.samples[index]
            ref = self.load_refs(ref_id)[0]

        file_name = '_'.join(ref['file_name'].split('_')[:-1]) + '.' + ref['file_name'].split('.')[-1]
        coco_split = file_name.split('_')[1]

        img_path = os.path.join(self.data_path, 'mscoco', coco_split, file_name)
        return img_path

    def __getitem__(self, index):
        ref_id, i = self.samples[index]
        ref = self.load_refs(ref_id)[0]

        img_path = self.get_sample_path(ref=ref)

        with open(img_path, "rb") as f:
            pil_img = Image.open(f).convert("RGB")
        if self.image_transforms:
            img = self.image_transforms(pil_img)
        else:
            img = pil_img

        # There are different texts associated to every image
        text = ref['sentences'][i]['sent']

        answer = self.refToAnn[ref_id]['bbox']

        return {'query': text, 'image': img, 'sample_id': index, 'answer': answer, 'index': index,
                'possible_answers': [], 'info_to_prompt': text, "query_type": -1, 'extra_context': ''}

    def __len__(self):
        return len(self.samples)

    @classmethod
    def accuracy(cls, prediction, ground_truth, *args):
        """
        Compute IoU score
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
        Returns:
            score (float): Score of the prediction. It is an IoU score
        """
        assert len(prediction) == len(ground_truth)
        num_samples = 0
        iou = 0
        acc = 0
        for p, g in zip(prediction, ground_truth):
            try:
                if p is None:
                    # Average bounding box
                    p = torch.tensor([50.9,  39.1, 493.5, 356.5])[None]  # Mean IoU is 22.64%
                else:
                    if type(p) == list:
                        p = torch.tensor(p)[None]
                    elif type(p) == str:
                        p = torch.tensor([float(x) for x in p.split('(')[1].split(')')[0].split(',')])[None]
                    else:
                        p = torch.tensor([p.left, p.lower, p.right, p.upper])[None]
                if type(g) == str:
                    g = [float(x) for x in g.split('[')[1].split(']')[0].split(',')]
                g = torch.tensor([g[0], g[1], g[2], g[3]])[None]
                iou_ = box_iou(p, g).item()  # Expects (x1, y1, x2, y2) format. So (left, lower, right, upper)
                iou += iou_
                if iou_ > 0.7:
                    acc += 1
            except Exception as e:
                pass  # If the prediction is not a box, we consider iou = 0
            num_samples += 1
        return iou / max(num_samples, 1), acc / max(num_samples, 1)

