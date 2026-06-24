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
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    dst_dit_list = [f"{args.workdir}/brain_img_N4", 
                    f"{args.workdir}/brain_subcortical_region"]

    for path in dst_dit_list:
        if not os.path.exists(path):
            os.makedirs(path)

    failed_list_be = []
    failed_list_cere = []
        
    print("Bias field correction by bfc")
    brain_folder = f'{args.workdir}/brain_img/'
    subjects = os.listdir(brain_folder)
    corr_folder = f'{args.workdir}/brain_img_N4/'
    for subject in subjects:
        print("Processing", subject)
        cmd = '/workspace/BrainSuite21a/bin/bfc -i %s -o %s' % (brain_folder + subject, corr_folder + subject)
        os.system(cmd)
        
    # recover the orientation
    input_folder = f'{args.workdir}/'
    corr_folder = f'{args.workdir}/brain_img_N4/'
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            corr_path = os.path.join(corr_folder, filename)
            corr_path = os.path.join(corr_folder, filename)
            input_img = sitk.ReadImage(input_path)
            input_direction = input_img.GetDirection()
            input_origin = input_img.GetOrigin()
            target_img = sitk.ReadImage(corr_path)
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(input_direction)
            resampler.SetOutputOrigin(input_origin)
            resampler.SetSize(input_img.GetSize())
            resampler.SetOutputSpacing(input_img.GetSpacing())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_img = resampler.Execute(target_img)
            sitk.WriteImage(resampled_img, corr_path)

    print("Predicting subcortical region")
    # os.system('nnUNet_predict -i brain/  -o brain_cere_removal_mask/ -t 507 -m 3d_fullres -chk model_best -tr nnUNetTrainerV2_DA3_BN -f 0 ' )
    os.system(f'nnUNet_predict -i {args.workdir}/brain_img_N4/ -o {args.workdir}/brain_subcortical_region/ -t 509 -m 3d_fullres -chk model_subcortical_binary -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER')


    import nibabel as nib
    from scipy.ndimage import distance_transform_edt as distance
    from skimage import segmentation as skimage_seg

    def compute_sdf1_1(img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """

        img_gt = img_gt.astype(np.uint8)
        

        normalized_sdf = np.zeros(out_shape)

        
        posmask = img_gt
        negmask = 1 - posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        sdf = negdis - posdis
        sdf[boundary==1] = 0
                # normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

        return sdf




    import shutil
    input_folder = f'{args.workdir}/'
    corr_folder = f"{args.workdir}/brain_img_N4"
    # Loop through the directory to find and rename .nii.gz files
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            # Construct old and new file paths
            old_file = os.path.join(input_folder, filename)
            new_file = os.path.join(corr_folder, filename.replace(".nii.gz", "_0000.nii.gz"))

            # Rename the file
            shutil.move(old_file, new_file)
    # Listing the contents of the directory after renaming
    os.listdir(corr_folder)





    print("Obtaining brain subcortical SDM")
    # 设置输入和输出文件夹路径
    input_folder = f"{args.workdir}/brain_subcortical_region"
    corr_folder = f"{args.workdir}/brain_img_N4"
    # output_folder2 = "CHARM_SDM"
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            
            # 加载.nii.gz文件
            nii_file = nib.load(input_path)
            data = nii_file.get_fdata()
            
            # data[data == 2]=1
            # img_gt = data[np.newaxis,:,:,:]
            img_gt = data
            out_shape = img_gt.shape
            
            # 计算有符号距离图
            signed_distance_map1 = compute_sdf1_1(img_gt,out_shape)
            

            img_gt = data
            out_shape = img_gt.shape
            
            # 计算有符号距离图
            signed_distance_map2 = compute_sdf1_1(img_gt,out_shape)
            
            # 创建新的.nii.gz文件并保存有符号距离图
            new_nii1 = nib.Nifti1Image(signed_distance_map1, nii_file.affine, header=nii_file.header)
            # new_nii2 = nib.Nifti1Image(signed_distance_map2, nii_file.affine, header=nii_file.header)

            # 指定输出路径
            output_path1 = os.path.join(corr_folder, filename.replace(".nii.gz", '_0001.nii.gz'))
            # output_path2 = os.path.join(output_folder2, filename)
            nib.save(new_nii1, output_path1)
            # nib.save(new_nii2, output_path2)

    print("Predicting subcortical structures") 
    os.system(f'python ../nnunet/inference/predict_simple_SDMs.py  -i {args.workdir}/brain_img_N4/  -o {args.workdir}/brain_subcortical/  -t 509 -m 3d_fullres -chk model_subcortical_SDM -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER  ' )

    if failed_list_be:
        print("Unable to extract brain for scan",failed_list_be,"and the rest have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default='', help="work directory (keep sure your image is in this dir)")
    args = parser.parse_args()
    
    main(args=args)