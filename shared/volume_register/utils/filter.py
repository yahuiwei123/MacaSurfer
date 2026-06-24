"""Copyright (c) 2022, Muhammad Asad (masadcv@gmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import numbers

import torch
import torch.nn as nn


# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
def initialise_gaussian_weights(channels, kernel_size, sigma, dims):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dims
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dims

    if all(s == 0 for s in sigma):
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        center = [ (k - 1) // 2 for k in kernel_size ]
        kernel[tuple(center)] = 1.0

        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        return kernel

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


class GaussianFilter2d(nn.modules.Conv2d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        sigma,
        padding="same",
        stride=1,
        padding_mode="zeros",
    ):
        gausssian_weights = initialise_gaussian_weights(
            channels=in_channels, kernel_size=kernel_size, sigma=sigma, dims=2
        )

        out_channels = gausssian_weights.shape[0]

        super(GaussianFilter2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = gausssian_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)


class GaussianFilter3d(nn.modules.Conv3d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        sigma,
        padding="same",
        stride=1,
        padding_mode="zeros",
    ):
        gausssian_weights = initialise_gaussian_weights(
            channels=in_channels, kernel_size=kernel_size, sigma=sigma, dims=3
        )

        out_channels = gausssian_weights.shape[0]

        super(GaussianFilter3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = gausssian_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)