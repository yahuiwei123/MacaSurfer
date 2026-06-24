import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def get_conv_op(conv_op_str: str):
    # map string to actual class
    if conv_op_str.endswith('Conv3d'):
        return nn.Conv3d
    raise ValueError(f"Unsupported conv op: {conv_op_str}")


def get_norm_op(norm_op_str: str):
    if norm_op_str.endswith('InstanceNorm3d'):
        return nn.InstanceNorm3d
    raise ValueError(f"Unsupported norm op: {norm_op_str}")


def get_nonlinearity(nonlin_str: str):
    if nonlin_str.endswith('LeakyReLU'):
        return nn.LeakyReLU
    elif nonlin_str.endswith('Tanh'):
        return nn.Tanh
    raise ValueError(f"Unsupported nonlinearity: {nonlin_str}")


class PlainConvUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: str,
        kernel_sizes: List[List[int]],
        strides: List[List[int]],
        n_conv_per_stage: List[int],
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool,
        norm_op: str,
        norm_op_kwargs: dict,
        dropout_op: Optional[str],
        dropout_op_kwargs: Optional[dict],
        nonlin: str,
        nonlin_kwargs: dict,
    ):
        super().__init__()
        # parse ops
        Conv = get_conv_op(conv_op)
        Norm = get_norm_op(norm_op)
        Nonlin = get_nonlinearity(nonlin)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_in = in_channels
        # Encoder
        for stage in range(n_stages):
            feats = features_per_stage[stage]
            # conv layers per stage
            layers = []
            for _ in range(n_conv_per_stage[stage]):
                layers.append(Conv(current_in, feats, kernel_size=kernel_sizes[stage],
                                   stride=1, padding=[k//2 for k in kernel_sizes[stage]], bias=conv_bias))
                layers.append(Norm(feats, **norm_op_kwargs))
                layers.append(Nonlin(**nonlin_kwargs))
                current_in = feats
            self.encoder_blocks.append(nn.Sequential(*layers))
            # downsample (except last)
            if stage < n_stages - 1:
                ds_stride = strides[stage+1]
                self.downsamples.append(Conv(current_in, current_in,
                                             kernel_size=ds_stride, stride=ds_stride,
                                             bias=conv_bias))
        # Bottleneck conv at last stage if needed

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for idx, feats in enumerate(reversed(features_per_stage[:-1])):
            img_stage = n_stages - 2 - idx
            # upsample from features_per_stage[n_stages-1- idx]
            prev_feats = features_per_stage[n_stages-1-idx]
            us = nn.ConvTranspose3d(prev_feats, feats,
                                     kernel_size=strides[img_stage+1],
                                     stride=strides[img_stage+1], bias=False)
            self.upsamples.append(us)
            # convs after concat
            layers = []
            in_feats = feats * 2
            for _ in range(n_conv_per_stage_decoder[idx]):
                layers.append(Conv(in_feats, feats, kernel_size=kernel_sizes[img_stage],
                                   stride=1, padding=[k//2 for k in kernel_sizes[img_stage]], bias=conv_bias))
                layers.append(Norm(feats, **norm_op_kwargs))
                layers.append(Nonlin(**nonlin_kwargs))
                in_feats = feats
            self.decoder_blocks.append(nn.Sequential(*layers))

        # final conv map to out_channels
        self.conv_final = Conv(features_per_stage[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats = []
        # Encoder forward
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            enc_feats.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        # Decoder forward
        for us, dec_block, enc_feat in zip(self.upsamples, self.decoder_blocks, reversed(enc_feats[:-1])):
            x = us(x)
            # crop or pad x to match enc_feat spatial dims if needed
            if x.shape[2:] != enc_feat.shape[2:]:
                # center crop or pad
                diff = [e - d for e, d in zip(enc_feat.shape[2:], x.shape[2:])]
                pad = [(-d//2, -d+(-d//2)) for d in diff[::-1]]
                # pad format: (padW_left, padW_right, padH_left...)
                x = nn.functional.pad(x, sum(pad, ()))
            x = torch.cat((x, enc_feat), dim=1)
            x = dec_block(x)
        # Final conv
        x = self.conv_final(x)
        return x
