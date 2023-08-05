"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from skimage.metrics import structural_similarity
from math import sqrt
import h5py
import numpy as np
import torch
import random
from .loss_function import ConsistencyLoss
import utils.model.fastmri as fastmri

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

def ssim_loss(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
       
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]
    return 1 - ssim

def consistency_loss(gt, pred, multi=False):
    if not multi:
        gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(1)
        pred = torch.tensor(pred, dtype=torch.float32).unsqueeze(1)
    else:
        gt = torch.tensor(gt, dtype=torch.float32)
        pred = torch.tensor(pred, dtype=torch.float32)
    cons_loss = ConsistencyLoss()
    gt = torch.tensor(gt).unsqueeze(1)
    pred = torch.tensor(pred).unsqueeze(1)
    cons_loss = ConsistencyLoss(win_size = win_size)
    loss_cons = 0.
    for slice_num in range(gt.shape[0]):
        loss_cons += cons_loss(gt[slice_num], pred[slice_num])
    loss_cons = loss_cons / gt.shape[0]
    return loss_cons

def fftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data, axes=axes), 
                    axes=axes, 
                    norm=norm), 
        axes=axes
    )


def ifftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered inverse fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(data, axes=axes), 
                     axes=axes, 
                     norm=norm), 
        axes=axes
    )

def kspace2image(kspace_pred):
    image_pred = fastmri.ifft2c(kspace_pred)
    image_pred = fastmri.complex_abs(image_pred)
    result = fastmri.rss(image_pred, dim=1)
    height = result.shape[-2]
    width = result.shape[-1]
    return result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]

def rss_combine(data, axis, keepdims=False):
    return np.sqrt(np.sum(np.square(np.abs(data)), axis, keepdims=keepdims))

def seed_fix(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)
    random.seed(n)