from pathlib import Path

import decord
from decord import cpu, gpu
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, split, data_path="", input_type='image', image_transforms=None, fps=30, max_num_frames=30,
                 max_samples=None, start_sample=0, **kwargs):
        """
        Args:
            split (str): Data split. One of ["challenge", "submission", "test", "testdev", "train", "val"]
            data_path (str): Path to the data folder
            input_type (str): Type of input. One of ["image", "video"]
            image_transforms (callable, optional): Optional transform to be applied on an image. Only used if input_type
                is "image".
            fps (int): Frames per second. Only used if input_type is "video".
            max_num_frames (int): Maximum number of frames to use. Only used if input_type is "video".
            max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
            start_sample (int, optional): Index of the first sample to load. If None, start from the beginning.
        """

        self.split = split
        self.data_path = Path(data_path)
        self.input_type = input_type
        self.image_transforms = image_transforms
        self.fps = fps
        self.max_num_frames = max_num_frames

        # Load questions, answers, and image ids
        with open(self.data_path / self.split / 'queries.csv', 'r') as f:
            # The csv has the rows [query, answer, image_name or video_name]
            self.df = pd.read_csv(f, index_col=None, keep_default_na=False)

        if max_samples is not None:
            self.df = self.df.iloc[start_sample:start_sample + max_samples]

        self.n_samples = len(self.df)

    def get_sample_path(self, index):
        sample_name = self.df.iloc[index][f"{self.input_type}_name"]
        sample_path = self.data_path / f"{self.input_type}s" / sample_name
        return sample_path

    def get_image(self, image_path):
        with open(image_path, "rb") as f:
            pil_image = Image.open(f).convert("RGB")
        if self.image_transforms:
            image = self.image_transforms(pil_image)[:3]
        else:
            image = pil_image
        return image

    def get_video(self, video_path):
        # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        num_frames = int(vlen * self.fps / original_fps)
        num_frames = min(self.max_num_frames, num_frames)
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int)
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)
        return video

    def __getitem__(self, index):

        out_dict = self.df.iloc[index].to_dict()

        sample_path = self.get_sample_path(index)

        # Load and transform image
        image = self.get_image(sample_path) if self.input_type == "image" else self.get_video(sample_path)

        out_dict["image"] = image
        out_dict["index"] = index
        
        return out_dict

    def __len__(self):
        return self.n_samples
