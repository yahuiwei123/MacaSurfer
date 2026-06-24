import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import measure
from utils import detect_available_device
import argparse


def main(args):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    # available_device = detect_available_device(8)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_device)
    import nBEST_setenv
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


    # 创建文件夹
    dst_dit_list = [f"{args.workdir}/brain_mask",
                    f"{args.workdir}/brain_img"]
    for path in dst_dit_list:
        if not os.path.exists(path):
            os.makedirs(path)

    failed_list_be = []

    # nBEST获取brain mask
    print("Predicting brain mask")
    os.system(f'{args.python_env}/bin/nnUNet_predict -i {args.workdir}/ -o {args.workdir}/brain_mask/ -t 509 -m 3d_fullres -chk model_be -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER' )
    input_files = [f for f in os.listdir(f"{args.workdir}/brain_mask") if f.endswith('.nii.gz')]


    # 根据brain mask从原始T1获取脑
    t1_path = f'{args.workdir}/'
    mask_path = f'{args.workdir}/brain_mask/'
    files = os.listdir(t1_path)
    brain_path = f'{args.workdir}/brain_img/'

    for subject in files:
        if not subject.endswith('.nii.gz'):
            continue
        t1_name = subject
        mask_name = subject
        print(f"extracting brain from {t1_name}")
        t1_img = sitk.ReadImage(t1_path + t1_name)
        mask_img = sitk.ReadImage(mask_path + mask_name)
        t1_array = sitk.GetArrayFromImage(t1_img)
        mask_array = sitk.GetArrayFromImage(mask_img)
        arr = mask_array > 0 
        labelled = measure.label(arr)
        rp = measure.regionprops(labelled)
        if not rp:
            print("Failed to extract brain for scan ",t1_name, "generating empty mask and brain file")
            failed_list_be.append(t1_name)
            # 生成全0的mask和brain文件，确保文件存在
            cleaned = np.zeros_like(arr).astype(np.uint8)
            t1_array[cleaned != 1] = 0
            brain_img  = sitk.GetImageFromArray(t1_array.astype(np.float32))
            brain_img.CopyInformation(t1_img)
            sitk.WriteImage(brain_img, brain_path + t1_name)
            cleaned_img  = sitk.GetImageFromArray(cleaned.astype(np.uint8))
            cleaned_img.CopyInformation(mask_img)
            sitk.WriteImage(cleaned_img, mask_path + t1_name)
        else:
            size = max([i.area for i in rp])
            cleaned = morphology.remove_small_objects(arr, min_size=size-1).astype(np.uint8)
            labelled = measure.label(arr)
            rp = measure.regionprops(labelled)
            t1_array[cleaned != 1]=0
            brain_img  = sitk.GetImageFromArray(t1_array.astype(np.float32))
            brain_img.CopyInformation(t1_img)
            sitk.WriteImage(brain_img, brain_path + t1_name)
            cleaned_img  = sitk.GetImageFromArray(cleaned.astype(np.uint8))
            cleaned_img.CopyInformation(mask_img)
            sitk.WriteImage(cleaned_img, mask_path + t1_name)

    # recover the orientation
    # 定义输入，目标和输出目录
    input_folder = t1_path
    target_folder = f'{args.workdir}/brain_img/'
    brain_folder = f'{args.workdir}/brain_img/'
    for filename in os.listdir(target_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            target_path = os.path.join(target_folder, filename)
            brain_path = os.path.join(brain_folder, filename)
            input_img = sitk.ReadImage(input_path)
            input_direction = input_img.GetDirection()
            input_origin = input_img.GetOrigin()
            target_img = sitk.ReadImage(target_path)
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(input_direction)
            resampler.SetOutputOrigin(input_origin)
            resampler.SetSize(input_img.GetSize())
            resampler.SetOutputSpacing(input_img.GetSpacing())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_img = resampler.Execute(target_img)
            sitk.WriteImage(resampled_img, brain_path)

    if failed_list_be:
        print("Unable to extract brain for scan",failed_list_be,"and the rest have been processed.")
    os.chdir(cwd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_env", type=str, default='', help="python environment path")
    parser.add_argument("--workdir", type=str, default='', help="work directory (keep sure your image is in this dir)")
    args = parser.parse_args()
    
    main(args=args)