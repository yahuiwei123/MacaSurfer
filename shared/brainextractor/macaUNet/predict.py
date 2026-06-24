import os
import torch
import torch.nn as nn
from monai.data import CacheDataset, decollate_batch
from monai.metrics import DiceMetric
from data.datasets import get_datasets, get_dataloaders
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.transforms import (
    Invertd,
    Activationsd,
    AsDiscreted,
    Compose,
)
from network.unet_dyn import DynUNet
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import label as scipy_label
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped,
    Spacingd, SpatialPadd, CropForegroundd, NormalizeIntensityd, Lambdad,
    ClipIntensityPercentilesd, CastToTyped,
    FillHolesd, 
)


def keep_largest_connected_component(mask):
    labeled, num_features = scipy_label(mask > 0)
    if num_features == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return (labeled == counts.argmax()).astype(np.uint8)


class TestTransform:
    def __init__(self, args = None):
        self.pre_transforms = [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            ClipIntensityPercentilesd(keys="image", lower=0.5, upper=99.5, sharpness_factor=5.),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            
            Spacingd(keys="image", pixdim=args["spacing"], mode=["trilinear"]),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            
            CropForegroundd(
                keys="image",
                source_key="image",
                select_fn=lambda x: x > 1e-8,
                margin=0,
                return_coords=False,
                allow_smaller=True
            ),
            SpatialPadd(keys="image", spatial_size=args['patch_size'], mode=["minimum"]),
            
            CastToTyped(keys="image", dtype=np.float32),
            EnsureTyped(keys="image"),
        ]

    def __call__(self, pair):
        pair = Compose(self.pre_transforms)(pair)
        return pair

class Predictor:
    def __init__(self, plans: dict = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plans = plans

        # Model
        deep_supr_num = 4
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2, 2],
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=deep_supr_num,
        ).to(self.device)
        
        if plans['model_pt']:
            state_dict = torch.load(plans['model_pt'], map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        # Data
        self.test_transforms = Compose([
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            ClipIntensityPercentilesd(keys="image", lower=0.5, upper=99.5, sharpness_factor=5.),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            
            Spacingd(keys="image", pixdim=plans["spacing"], mode=["trilinear"]),
            Lambdad(keys="image", func=lambda x: x.contiguous()),
            
            CropForegroundd(
                keys="image",
                source_key="image",
                select_fn=lambda x: x > 1e-8,
                margin=0,
                return_coords=False,
                allow_smaller=True
            ),
            SpatialPadd(keys="image", spatial_size=plans['patch_size'], mode=["minimum"]),
            
            CastToTyped(keys="image", dtype=np.float32),
            EnsureTyped(keys="image"),
        ])
        
        self.post_trans = Compose([
            EnsureTyped(keys="pred"),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            FillHolesd(keys="pred"),
            Invertd(
                keys="pred",
                transform=self.test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ), 
        ])

    def predict(self, data_path: str = None, overlap: float = 0.5, sigma_scale: float = 0.125):
        print("Testing...")
        self.model.eval()
        with torch.no_grad():
            data_pair = self.test_transforms({'image': data_path})
            test_img = data_pair['image'].unsqueeze(0).to(self.device, non_blocking=True)
            test_outputs = sliding_window_inference(test_img, self.plans['patch_size'], self.plans['batch_size'], self.model, overlap=overlap, mode='gaussian', sigma_scale=sigma_scale)
            
            # restore original space
            data_pair['pred'] = test_outputs.squeeze(0)
            
            # # restore original orientation
            # img_obj = nib.load(data_path)
            # original_axcodes = nib.aff2axcodes(img_obj.affine)
            # data_pair = Orientationd(keys="label", axcodes=original_axcodes)(data_pair)
            
            # restore original shape
            test_outputs = self.post_trans(data_pair)['pred'].squeeze(0).detach().cpu().numpy().astype(np.uint8)
            test_outputs = keep_largest_connected_component(test_outputs)
            
        return test_outputs, data_pair['image_meta_dict']['affine']
        
        
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default='', help="image to skullstrip")
    parser.add_argument("--out", type=str, default='', help="brainmask")
    args = parser.parse_args()
    
    plans = {
        "batch_size": 8,
        "patch_size": [96,96,96],
        "spacing": [0.4,0.4,0.4],
        "model_pt": os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/3_times_conv_on_downsample/fold_all/best_metric_model.pth")
    }
    
    predictor = Predictor(plans)
    pred_y, affine = predictor.predict(args.img)
    
    brainmask_img = nib.Nifti1Image(pred_y, affine)
    nib.save(brainmask_img, args.out)