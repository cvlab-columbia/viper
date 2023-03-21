import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import sys
import time
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as tv_draw_bounding_boxes
from torchvision.utils import make_grid
from typing import Union

clip_stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter


def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_batch(batch, stats=clip_stats):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    denorm_images = denormalize(batch, *stats)
    ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))


def show_batch_from_dl(dl):
    for images, labels in dl:
        show_batch(images)
        print(labels[:64])
        break


def show_single_image(image, denormalize_stats=None, bgr_image=False, save_path=None, size='small', bbox_info=None):
    if not is_interactive():
        import matplotlib
        matplotlib.use("module://imgcat")
    if size == 'size_img':
        figsize = (image.shape[2] / 100, image.shape[1] / 100)  # The default dpi of plt.savefig is 100
    elif size == 'small':
        figsize = (4, 4)
    else:
        figsize = (12, 12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])

    if bbox_info is not None:
        image = draw_bounding_boxes(image, bbox_info['bboxes'], labels=bbox_info['labels'], colors=bbox_info['colors'],
                                    width=5)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if denormalize_stats is not None:
            image = denormalize(image.unsqueeze(0), *denormalize_stats)
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        ax.imshow(image.squeeze(0).permute(1, 2, 0))
    else:
        if bgr_image:
            image = image[..., ::-1]
        ax.imshow(image)

    if save_path is None:
        plt.show()
    # save image if save_path is provided
    if save_path is not None:
        # make path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)


def draw_bounding_boxes(
        image: Union[torch.Tensor, Image.Image],
        bboxes: Union[list, torch.Tensor],
        width: int = 5,
        **kwargs
):
    """
    Wrapper around torchvision.utils.draw_bounding_boxes
    bboxes: [xmin, ymin, xmax, ymax]
    :return:
    """
    if isinstance(image, Image.Image):
        if type(image) == Image.Image:
            image = transforms.ToTensor()(image)
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)

    image = (image * 255).to(torch.uint8).cpu()
    height = image.shape[1]
    bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)
    return tv_draw_bounding_boxes(image, bboxes, width=width, **kwargs)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_index_from_sample_id(sample_id, dataset):
    df = dataset.df
    return np.arange(df.shape[0])[df.index == sample_id]


def save_json(data: dict, path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def make_print_safe(string: str) -> str:
    return string.replace(r'[', r'\[')


def sprint(string: str):
    print(make_print_safe(string))


def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        if is_interactive():
            display(df)
        else:
            print(df)


def code_to_paste(code):
    print('\n'.join([c[4:] for c in code.split('\n')[1:]]).replace('image', 'ip').replace('return ', ''))


class HiddenPrints:
    def __init__(self, model_name=None, console=None, use_newline=True):
        self.model_name = model_name
        self.console = console
        self.use_newline = use_newline
        self.tqdm_aux = None

    def __enter__(self):
        import tqdm  # We need to do an extra step to hide tqdm outputs. Does not work in Jupyter Notebooks.

        def nop(it, *a, **k):
            return it

        self.tqdm_aux = tqdm.tqdm
        tqdm.tqdm = nop

        if self.model_name is not None:
            self.console.print(f'Loading {self.model_name}...')
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        # May not be what we always want, but some annoying warnings end up to stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stdout = self._original_stderr
        if self.model_name is not None:
            self.console.print(f'{self.model_name} loaded ')
        import tqdm
        tqdm.tqdm = self.tqdm_aux
