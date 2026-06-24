from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped,
    RandGaussianNoised, RandRicianNoised, RandGibbsNoised,
    RandAdjustContrastd, RandBiasFieldd, RandHistogramShiftd, 
    Spacingd, SpatialPadd, CropForegroundd, RandFlipd, 
    RandAffined, Rand3DElasticd, NormalizeIntensityd, Lambdad,
    ClipIntensityPercentilesd, RandRotate90d, RandSpatialCropd,
    RandGaussianSmoothd, RandScaleIntensityd, CastToTyped,
    RandCropByPosNegLabeld
)
from data.utils import RandInvertIntensityd
import numpy as np
import torch

class TrainTransform:
    def __init__(self, args = None):
        self.base_transforms = [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        
        self.fixed_transforms = [
            # crop
            RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=args['patch_size'], label_key='label', pos=1.0, neg=1.0, num_samples=1, allow_smaller=True),
        ]
        
        self.random_transforms = [
            # pad with mini intensity in current image, and with 0 in current label
            SpatialPadd(keys=["image", "label"], spatial_size=args['patch_size'], mode=["minimum", "constant"]),
            
            # random invert histo
            RandInvertIntensityd(keys="image", prob=0.25),
            
            # additive noise (local)
            RandRicianNoised(keys="image", mean=0.0, std=0.3, prob=0.2),
            RandGaussianNoised(keys="image", std=0.3, prob=0.2),
            RandGibbsNoised(keys="image", alpha=[0, 1], prob=0.2),
            
            # multiplicative noise (global)
            RandBiasFieldd(keys="image", coeff_range=(0.1, 0.5), degree=5, prob=0.40),
            RandAdjustContrastd(keys="image", gamma=(0.7, 1.4), prob=0.25),
            Lambdad(keys=["image", "label"], func=lambda x: x.contiguous()),
            
            # shift histo (global)
            RandHistogramShiftd(keys="image", num_control_points=(5, 10), prob=0.20),
            Lambdad(keys=["image", "label"], func=lambda x: x.contiguous()),
            
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.25,
            ),
            
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.25),
            Lambdad(keys=["image", "label"], func=lambda x: x.contiguous()),
            
            RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5),
            
            RandRotate90d(
            keys=["image", "label"],
            prob=0.25,
            spatial_axes=(2, 1),
            max_k=2
            ),
            
            RandAffined(
                keys=["image", "label"],
                rotate_range=(0.20, 0.20, 0.20),
                shear_range=(0.10, 0.10, 0.10),
                scale_range=(0.10, 0.10, 0.10),
                prob=0.15,
                mode=["trilinear", "nearest"],
            ),
            
            # Rand3DElasticd(
            #     keys=["image", "label"],
            #     sigma_range=(5, 7),
            #     magnitude_range=(50, 100),
            #     prob=0.3,
            #     mode=["trilinear", "nearest"]
            # ),
        ]
        
        self.post_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]

    def __call__(self, pair):
        pair = Compose(self.base_transforms)(pair)
        
        if torch.sum(pair['image']) == 0:
            print(pair['image_meta_dict']['filename_or_obj'], "may have no label!, please check!")
        
        pair = Compose(self.fixed_transforms)(pair)[0]
        pair = Compose(self.random_transforms)(pair)
        pair = Compose(self.post_transforms)(pair)
        return pair
    
class ValidTransform:
    def __init__(self, args = None):
        self.base_transforms = [
                LoadImaged(keys=["image", "label"], image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Lambdad(keys=["image", "label"], func=lambda x: x.contiguous()),
                # pad with mini intensity in current image, and with 0 in current label
                SpatialPadd(keys=["image", "label"], spatial_size=args['patch_size'], mode=["minimum", "constant"]),
        ]

        self.post_transforms = [
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
            EnsureTyped(keys=["image", "label"]),
        ]

    def __call__(self, pair):
        pair = Compose(self.base_transforms)(pair)
        # pair = Compose(self.fixed_transforms)(pair)
        pair = Compose(self.post_transforms)(pair)
        return pair
    
