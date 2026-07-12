"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, group-aware splitting,
and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import glob
import re
from tqdm import tqdm
import numpy as np


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a list of ``(video_directory, label)`` tuples.

    Args:
        vid_dataset (list): List where each item contains a video directory path and integer label.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = sorted(glob.glob(os.path.join(self.dataset[idx][0], '*.jpg')))
        fr_paths = fr_paths[:self.fpv]

        # Open images using PIL
        fr_imgs = [Image.open(fr_path).convert('RGB') for fr_path in fr_paths]

        # Get the label associated with the video
        fr_label = self.dataset[idx][1]

        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label, len(fr_imgs_trans)


def extract_group_id(video_path):
    """
    Extract a leakage-safe group id from a video/frame-folder name.

    UCF-style clips commonly encode related clips as gXX_cYY. Clips from the
    same group share subject/background/viewpoint and must stay in one split.
    If no explicit group token exists, the video folder itself is used.
    """
    video_name = os.path.basename(video_path)
    match = re.search(r'(?:^|[_-])(g\d+)(?:[_-]|$)', video_name, flags=re.IGNORECASE)
    if match:
        class_name = os.path.basename(os.path.dirname(video_path))
        return '{}:{}'.format(class_name, match.group(1).lower())
    return video_path


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a sample list with video paths, integer labels, and split groups.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (list): List of (video_path, label, group_id) tuples.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    vid_cats = [vid_cat for vid_cat in sorted(os.listdir(frame_dir))
                if os.path.isdir(os.path.join(frame_dir, vid_cat))]
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(vid_cats)}
    vid_dataset = []
    print('Loading video dataset....')
    for vid_cat in tqdm(vid_cats):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in sorted(os.listdir(vid_cat_path)):
            vid_path = os.path.join(vid_cat_path, vid)
            if os.path.isdir(vid_path):
                vid_dataset.append((vid_path, label_dict[vid_cat], extract_group_id(vid_path)))
    return vid_dataset, label_dict


def _split_group_keys(group_keys, train_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    group_keys = np.array(sorted(group_keys))
    rng.shuffle(group_keys)
    n_groups = len(group_keys)
    if n_groups < 3:
        raise ValueError('At least three groups per class are required for train/val/test splitting.')

    n_test = max(1, int(round(n_groups * test_ratio)))
    n_val = max(1, int(round(n_groups * (1 - train_ratio - test_ratio))))
    if n_test + n_val >= n_groups:
        n_test = max(1, min(n_test, n_groups - 2))
        n_val = max(1, min(n_val, n_groups - n_test - 1))

    test_keys = set(group_keys[:n_test])
    val_keys = set(group_keys[n_test:n_test + n_val])
    train_keys = set(group_keys[n_test + n_val:])
    return train_keys, val_keys, test_keys


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets by leakage-safe group.

    Groups are split independently within each class so related clips do not cross
    train, validation, and test boundaries.

    Args:
        vid_dataset (list): List of (video_path, label, group_id) tuples.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    if tr_ratio + ts_ratio >= 1:
        raise ValueError('train_size + test_size must be less than 1.0.')

    print('Splitting train/validation/test datasets by group....')
    by_label_group = {}
    for video_path, label, group_id in vid_dataset:
        by_label_group.setdefault(label, {}).setdefault(group_id, []).append((video_path, label, group_id))

    tr_dataset, val_dataset, ts_dataset = [], [], []
    for label, groups in sorted(by_label_group.items()):
        train_keys, val_keys, test_keys = _split_group_keys(groups.keys(), tr_ratio, ts_ratio, seed + int(label))
        for group_key, samples in groups.items():
            cleaned_samples = [(sample[0], sample[1]) for sample in samples]
            if group_key in train_keys:
                tr_dataset.extend(cleaned_samples)
            elif group_key in val_keys:
                val_dataset.extend(cleaned_samples)
            elif group_key in test_keys:
                ts_dataset.extend(cleaned_samples)

    rng = np.random.default_rng(seed)
    rng.shuffle(tr_dataset)
    rng.shuffle(val_dataset)
    rng.shuffle(ts_dataset)

    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).

    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch, length_batch = list(zip(*batch))
    valid_samples = [(imgs, label, length) for imgs, label, length in zip(imgs_batch, label_batch, length_batch)
                     if len(imgs) > 0]
    if not valid_samples:
        return None, None, None
    imgs_batch, label_batch, length_batch = zip(*valid_samples)
    label_batch = [torch.tensor(l) for l in label_batch]
    imgs_tensor = pad_sequence(imgs_batch, batch_first=True)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    lengths_tensor = torch.tensor(length_batch)
    return imgs_tensor, labels_tensor, lengths_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch, length_batch = list(zip(*batch))

    # Filter out any samples that have no frames
    valid_samples = [(imgs, label, length) for imgs, label, length in zip(imgs_batch, label_batch, length_batch)
                     if len(imgs) > 0]
    if not valid_samples:
        return None, None, None
    imgs_batch, label_batch, length_batch = zip(*valid_samples)

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    lengths_tensor = torch.tensor(length_batch)

    return padded_imgs, labels_tensor, lengths_tensor
