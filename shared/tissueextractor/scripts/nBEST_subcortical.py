import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import measure
import argparse
import shutil
# license_path = './License/License.txt'
# if not os.path.isfile(license_path):
#     print('License.txt file not found!')
#     os._exit(0)
  
# with open(license_path, "r") as licFile:
#     Text = licFile.read().strip()
#     licFile.close()
    
# if Text[16:22] != '130313':
#     print('Please put the correct License.txt under this directory')
#     os._exit(0)
    

# #os.environ['CUDA_VISIBLE_DEVICES']='5'
import nBEST_setenv
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def main(args):
    failed_list_be = []
    parser = argparse.ArgumentParser(description="Brain processing script.")
    parser.add_argument('-skip_be', action='store_true', help="Skip brain extraction steps.")
    args = parser.parse_args()


    if not args.skip_be:
        dst_dit_list = ["brain_mask","T1w_img","brain_subcortical","T1w_T2w_img_brain"]

        for path in dst_dit_list:
            if not os.path.exists(path):
                os.makedirs(path)
        source_folder = 'T1w_T2w_img'
        target_folder = 'T1w_img'
        for file in os.listdir(source_folder):
            if file.endswith('0000.nii.gz'):
                source_file = os.path.join(source_folder, file)
                target_file = os.path.join(target_folder, file)
                shutil.copy(source_file, target_file)
            
        print("Predicting brain mask")
        os.system(f'{args.python_env}/bin/nnUNet_predict -i T1w_img/  -o brain_mask/ -t 509 -m 3d_fullres -chk model_be -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER ' )
        input_files = [f for f in os.listdir("brain_mask") if f.endswith('.nii.gz')]

        print("Obtaining brain img")
        T1_path = 'T1w_img/'
        both_path = "T1w_T2w_img/"
        Seg_path = 'brain_mask/'
        files = os.listdir(T1_path)
        output_path = 'T1w_T2w_img_brain/'
        for subject in files:
            T1_name = subject
            T2_name = subject[0:-8]+"1.nii.gz"
            seg_name = subject
            print(T1_name,T2_name)
            T1_img = sitk.ReadImage(both_path + T1_name)
            T2_img = sitk.ReadImage(both_path + T2_name)
            Seg_img = sitk.ReadImage(Seg_path + seg_name)
            T1_array = sitk.GetArrayFromImage(T1_img)
            T2_array = sitk.GetArrayFromImage(T2_img)
            Seg_array = sitk.GetArrayFromImage(Seg_img)
            arr = Seg_array > 0 
            labelled = measure.label(arr)
            rp = measure.regionprops(labelled)
            if not rp:
                print("Failed to extract brain for scan ",T1_name)
                failed_list_be.append(T1_name) 
            else:        
                size = max([i.area for i in rp])
                cleaned = morphology.remove_small_objects(arr, min_size=size-1).astype(np.uint8)
                labelled = measure.label(arr)
                rp = measure.regionprops(labelled)
                T1_array[cleaned!=1]=0
                T2_array[cleaned!=1]=0
                output_img  = sitk.GetImageFromArray(T1_array.astype(np.float32))
                output_img2  = sitk.GetImageFromArray(T2_array.astype(np.float32))
                output_img.CopyInformation(T1_img)
                output_img2.CopyInformation(T2_img)
                sitk.WriteImage(output_img, output_path + T1_name)
                sitk.WriteImage(output_img2, output_path + T2_name)
                cleaned_img  = sitk.GetImageFromArray(cleaned.astype(np.uint8))
                cleaned_img.CopyInformation(Seg_img)
                sitk.WriteImage(cleaned_img, Seg_path + T1_name)
            print("Predicting subcortical structures") 
            os.system('python ../nnunet/inference/predict_simple_SDMs.py  -i T1w_T2w_img_brain/  -o brain_subcortical/  -t 509 -m 3d_fullres -chk model_subcortical -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER --step_size 0.1 ' )

    else:
        dst_dit_list = ["brain_subcortical"]
        for path in dst_dit_list:
            if not os.path.exists(path):
                os.makedirs(path)
        print("Predicting subcortical structures.")    
        os.system('python ../nnunet/inference/predict_simple_SDMs.py  -i T1w_T2w_img/  -o brain_subcortical/  -t 509 -m 3d_fullres -chk model_subcortical -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER --step_size 0.1 ' )

    if failed_list_be:
        print("Unable to extract brain for scan",failed_list_be,"and the rest have been processed.") 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default='', help="work directory (keep sure your image is in this dir)")
    args = parser.parse_args()
    
    main(args=args)