import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from typing import Optional
import random

class RandInvertIntensityd(MapTransform):
    """
    以概率对影像灰度值进行线性反转，并保持原始的均值和标准差。

    Args:
        keys: 要处理的字段，如 ["image"]。
        prob: 执行反转的概率（默认0.2）。
        seed: 随机种子（可选）。
    """
    def __init__(self, keys: KeysCollection, prob: float = 0.2, seed: Optional[int] = None):
        super().__init__(keys)
        self.prob = prob
        if seed is not None:
            random.seed(seed)
            torch.random.seed(seed)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if random.random() < self.prob:
                img = d[key]
                orig_mean = torch.mean(img)
                orig_std = torch.std(img)

                # 灰度反转
                min_val = torch.min(img)
                max_val = torch.max(img)
                inverted = max_val + min_val - img

                # 新均值和方差
                new_mean = torch.mean(inverted)
                new_std = torch.std(inverted)

                # 避免除以0
                if new_std < 1e-8:
                    new_std = 1e-8

                # 线性调整，使反转后的图像恢复原均值和方差
                restored = (inverted - new_mean) * (orig_std / new_std) + orig_mean

                d[key] = restored.to(img.dtype)  # 保留原始类型
        return d