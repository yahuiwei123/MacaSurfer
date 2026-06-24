# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


'''
author: rohitrango
'''
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union
from fireants.registration.deformation.abstract import AbstractDeformation
from fireants.io.image import Image, BatchedImages
from fireants.utils.imageutils import scaling_and_squaring, _find_integrator_n
from fireants.types import devicetype
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.util import grad_smoothing_hook
from fireants.utils.imageutils import jacobian
from fireants.registration.optimizers.sgd import WarpSGD
from fireants.registration.optimizers.adam import WarpAdam
from fireants.utils.globals import MIN_IMG_SIZE

from logging import getLogger
from copy import deepcopy

class CompositiveWarp(nn.Module, AbstractDeformation):
    '''
    Class for compositive warp function (collects gradients of dL/dp)
    The image is computed as M \circ (\phi + u)
    '''
    def __init__(self, fixed_images: torch.Tensor, moving_images: torch.Tensor,
                optimizer: str = 'Adam', optimizer_lr: float = 1e-2, optimizer_params: dict = {},
                init_scale: int = 1, 
                smoothing_grad_sigma: float = 0.5, smoothing_warp_sigma: float = 0.5, 
                freeform: bool = False,
                dtype: torch.dtype = torch.float32,
                ) -> None:
        super().__init__()
        self.num_images = num_images = max(fixed_images.shape[0], moving_images.shape[0])
        spatial_dims = list(fixed_images.shape[2:])  # [H, W, [D]]
        self.n_dims = len(spatial_dims)  # number of spatial dimensions
        self.freeform = freeform
        # permute indices
        self.permute_imgtov = (0, *range(2, self.n_dims+2), 1)  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (0, self.n_dims+1, *range(1, self.n_dims+1))  # [N, dims, HWD] -> [N, HWD, dims]
        self.device = fixed_images.device
        if optimizer_lr > 1:
            getLogger("CompositiveWarp").warning(f'optimizer_lr is {optimizer_lr}, which is very high. Unexpected registration may occur.')

        # define warp and register it as a parameter
        # set size
        if init_scale > 1:
            spatial_dims = [max(int(s / init_scale), MIN_IMG_SIZE) for s in spatial_dims]
        
        warp = torch.zeros([num_images, *list(fixed_images.shape[2:]), self.n_dims], dtype=dtype, device=fixed_images.device)  # [N, HWD, dims]
        self.register_parameter('warp', nn.Parameter(warp))
        
        inv_warp = torch.zeros([num_images, *list(moving_images.shape[2:]), self.n_dims], dtype=dtype, device=fixed_images.device)  # [N, HWD, dims]
        self.register_buffer('inv_warp', nn.Parameter(inv_warp))

        # attach grad hook if smooothing of the gradient is required
        self.smoothing_grad_sigma = smoothing_grad_sigma
        if smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.n_dims, device=fixed_images.device, dtype=dtype) + smoothing_grad_sigma)]
        self.attach_grad_hook()

        # if the warp is also to be smoothed, add this constraint to the optimizer (in the optimizer_params dict)
        oparams = deepcopy(optimizer_params)
        self.smoothing_warp_sigma = smoothing_warp_sigma
        if self.smoothing_warp_sigma > 0:
            smoothing_warp_gaussians = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.n_dims, device=fixed_images.device, dtype=dtype) + smoothing_warp_sigma)]
            oparams['smoothing_gaussians'] = smoothing_warp_gaussians

        if oparams.get('freeform') is None:
            oparams['freeform'] = freeform
        # add optimizer
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = WarpSGD(self.warp, lr=optimizer_lr, dtype=dtype, **oparams)
        elif optimizer == 'adam':
            self.optimizer = WarpAdam(self.warp, inv_warp=self.inv_warp, lr=optimizer_lr, dtype=dtype, **oparams)
        else:
            raise NotImplementedError(f'Optimizer {optimizer} not implemented')
    
    def attach_grad_hook(self):
        ''' attack the grad hook to the velocity field if needed '''
        if self.smoothing_grad_sigma > 0:
            hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
            self.warp.register_hook(hook)
            self.inv_warp.register_hook(hook)
    
    def initialize_grid(self):
        ''' initialize grid to a size 
        Simply use the grid from the optimizer, which should be initialized to the correct size
        '''
        self.grid = self.optimizer.grid
        self.inv_grid = self.optimizer.inv_grid

    def set_zero_grad(self):
        ''' set the gradient to zero (or None) '''
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

    def get_warp(self):
        ''' return warp function '''
        return self.warp
    
    def get_inverse_warp(self):
        return self.inv_warp
    
    def set_size(self, size_fixed, size_moving):
        mode = 'bilinear' if self.n_dims == 2 else 'trilinear'

        # 1) resize forward warp (fixed space)
        warp = F.interpolate(
            self.warp.detach().permute(*self.permute_vtoimg),
            size=size_fixed,
            mode=mode,
            align_corners=False,
        ).permute(*self.permute_imgtov)
        self.warp = nn.Parameter(warp)
        # self.register_parameter('warp', nn.Parameter(warp))
        self.warp.grad = None

        # 2) resize inverse warp (moving space)
        if hasattr(self, "inv_warp") and self.inv_warp is not None:
            inv_warp = F.interpolate(
                self.inv_warp.detach().permute(*self.permute_vtoimg),
                size=size_moving,
                mode=mode,
                align_corners=False,
            ).permute(*self.permute_imgtov)
            self.inv_warp = nn.Parameter(inv_warp)
            # self.register_parameter('inv_warp', nn.Parameter(inv_warp))
            self.inv_warp.grad = None

        self.attach_grad_hook()
        self.optimizer.set_data_and_size(self.warp, size_fixed, self.inv_warp, size_moving)
        self.initialize_grid()


if __name__ == '__main__':
    img1 = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    img2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00597/BraTS2021_00597_t1.nii.gz')
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    deformation = CompositiveWarp(fixed, moving)
    for i in range(100):
        deformation.set_zero_grad() 
        w = deformation.get_warp()
        loss = ((w-1/155)**2).mean()
        if i%10 == 0:
            print(loss)
        loss.backward()
        deformation.step()
    # w = deformation.get_inverse_warp(debug=True)
