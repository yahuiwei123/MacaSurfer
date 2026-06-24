import torch
from deformable_attention_3d import DeformableAttention3D

attn = DeformableAttention3D(
    dim = 512,                          # feature dimensions
    dim_head = 64,                      # dimension per head
    heads = 8,                          # attention heads
    dropout = 0.,                       # dropout
    downsample_factor = (2, 8, 8),      # downsample factor (r in paper)
    offset_scale = (2, 8, 8),           # scale of offset, maximum offset
    offset_kernel_size = (4, 10, 10),   # offset kernel size
)

x = torch.randn(1, 512, 10, 32, 32) # (batch, dimension, frames, height, width)
attn(x) # (1, 512, 10, 32, 32)
print(x.shape)