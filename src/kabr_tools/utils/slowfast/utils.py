# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# get_sequence, pack_pathway_output, tensor_normalize from https://github.com/facebookresearch/SlowFast/blob/bac7b672f40d44166a84e8c51d1a5ba367ace816/slowfast/datasets/utils.py
# scale from https://github.com/facebookresearch/SlowFast/blob/bac7b672f40d44166a84e8c51d1a5ba367ace816/slowfast/datasets/cv2_transform.py
# process_cv2_inputs from https://github.com/facebookresearch/SlowFast/blob/bac7b672f40d44166a84e8c51d1a5ba367ace816/slowfast/visualization/utils.py
# get_input_clip from https://github.com/facebookresearch/SlowFast/blob/bac7b672f40d44166a84e8c51d1a5ba367ace816/slowfast/visualization/ava_demo_precomputed_boxes.py

import math
import cv2
import torch
import numpy as np
from torch import Tensor


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def scale(size, image):
    """
    Scale the short side of the image to size.
    Args:
        size (int): size to scale the image.
        image (array): image to perform short side scale. Dimension is
            `height` x `width` x `channel`.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
    """
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (height <= width and height == size):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(image, (new_width, new_height),
                     interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32)


def process_cv2_inputs(frames, cfg):
    """
    Normalize and prepare inputs as a list of tensors. Each tensor
    correspond to a unique pathway.
    Args:
        frames (list of array): list of input images (correspond to one clip) in range [0, 255].
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    inputs = torch.from_numpy(np.array(frames)).float() / 255
    inputs = tensor_normalize(inputs, cfg.DATA.MEAN, cfg.DATA.STD)
    # T H W C -> C T H W.
    inputs = inputs.permute(3, 0, 1, 2)
    # Sample frames for num_frames specified.
    index = torch.linspace(0, inputs.shape[1] - 1, cfg.DATA.NUM_FRAMES).long()
    inputs = torch.index_select(inputs, 1, index)
    inputs = pack_pathway_output(cfg, inputs)
    inputs = [inp.unsqueeze(0) for inp in inputs]
    return inputs


def tensor_normalize(tensor, mean, std, func=None):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    if func is not None:
        tensor = func(tensor)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def get_input_clip(cap: cv2.VideoCapture, cfg, keyframe_idx: int) -> list[Tensor]:
    """
    Get input clip from the video/folder of images for a given
    keyframe index.
    Args:
        keyframe_idx (int): index of the current keyframe.
    Returns:
        clip (list of tensors): formatted input clip(s) corresponding to
            the current keyframe.
    """
    seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq = get_sequence(
        keyframe_idx,
        seq_length // 2,
        cfg.DATA.SAMPLING_RATE,
        total_frames,
    )
    clip = []
    for frame_idx in seq:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        was_read, frame = cap.read()
        if was_read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = scale(cfg.DATA.TEST_CROP_SIZE, frame)
            clip.append(frame)
        else:
            print("Unable to read frame. Duplicating previous frame.")
            clip.append(clip[-1])

    clip = process_cv2_inputs(clip, cfg)
    return clip
