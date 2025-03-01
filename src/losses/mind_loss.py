# Implementation of MINDLoss for SC-GAN from https://github.com/HeranYang/sc-cyclegan/tree/master
# Unpaired brain MR-to-CT synthesis using a structure-constrained CycleGAN, Yang et al., 2018 MICCAI
# 1. SC-GAN : GAN_loss + lam1 * Cycle_loss + lam2 * MIND_L1 Loss
# MIND : sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7, lam2=5

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def get_gausian_filter(sigma, sz):
    xpos, ypos = torch.meshgrid(torch.arange(sz), torch.arange(sz))
    output = torch.ones([sz, sz, 1, 1])
    midpos = sz // 2
    d = (xpos - midpos) ** 2 + (ypos - midpos) ** 2
    gauss = torch.exp(-d / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return gauss


def gaussian_filter(img, n, sigma):
    """
    img: image tensor of size (1, 1, height, width)
    n: size of the Gaussian filter (n, n)
    sigma: standard deviation of the Gaussian distribution
    """
    # Create a Gaussian filter
    gaussian_filter = get_gausian_filter(sigma, n)
    # Add extra dimensions for the color channels and batch size
    gaussian_filter = gaussian_filter.view(1, 1, n, n)
    gaussian_filter = gaussian_filter.to(img.device)
    # Perform 2D convolution
    filtered_img = F.conv2d(img, gaussian_filter, padding=n // 2)
    return filtered_img


def Dp(image, sigma, patch_size, xshift, yshift):
    shift_image = torch.roll(image, shifts=(xshift, yshift), dims=(-1, -2))
    diff = image - shift_image
    diff_square = diff**2
    res = gaussian_filter(diff_square, patch_size, sigma)
    return res


def mind(image, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
    reduce_size = (patch_size + neigh_size - 2) // 2
    # estimate the local variance of each pixel within the input image.
    Vimg = (
        Dp(image, sigma, patch_size, -1, 0)
        + Dp(image, sigma, patch_size, 1, 0)
        + Dp(image, sigma, patch_size, 0, -1)
        + Dp(image, sigma, patch_size, 0, 1)
    )
    Vimg = Vimg / 4 + eps * torch.ones_like(Vimg)

    # estimate the (R*R)-length MIND feature by shifting the input image by R*R times.
    xshift_vec = np.arange(-neigh_size // 2, neigh_size - neigh_size // 2)
    yshift_vec = np.arange(-neigh_size // 2, neigh_size - neigh_size // 2)

    # print(xshift_vec, yshift_vec)

    iter_pos = 0
    for xshift in xshift_vec:
        for yshift in yshift_vec:
            if (xshift, yshift) == (0, 0):
                continue
            MIND_tmp = torch.exp(
                -Dp(image, sigma, patch_size, xshift, yshift) / Vimg
            )  # MIND_tmp : 1x1x256x256
            tmp = MIND_tmp[
                ..., reduce_size:-reduce_size, reduce_size:-reduce_size, None
            ]  # 1x1x250x250x1
            output = tmp if iter_pos == 0 else torch.cat((output, tmp), -1)
            iter_pos += 1

    # normalization.
    output = torch.divide(output, torch.max(output, dim=-1, keepdim=True)[0])

    return output


class MINDLoss(nn.Module):
    def __init__(self, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
        super(MINDLoss, self).__init__()
        self.sigma = sigma
        self.eps = eps
        self.neigh_size = neigh_size
        self.patch_size = patch_size

    def forward(self, pred, gt):
        pred_mind = mind(pred, self.sigma, self.eps, self.neigh_size, self.patch_size)
        gt_mind = mind(gt, self.sigma, self.eps, self.neigh_size, self.patch_size)
        mind_loss = F.l1_loss(pred_mind, gt_mind)
        return mind_loss
