import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import measure
import argparse
from utils import detect_available_device

cwd = os.getcwd()
os.chdir(os.path.dirname(__file__))

available_device = detect_available_device(8)
  
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_device)
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


dst_dit_list = ["data/brain_mask","data/brain_img","data/brain_img_N4","data/brain_cere_removal_mask",
                "data/brain_cerebrum","data/brain_cerebrum_mask","data/brain_tissue","data/brain_subcortical"]

for path in dst_dit_list:
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description="Brain processing script.")
parser.add_argument('-skip_be', action='store_true', help="Skip brain extraction steps.")
args = parser.parse_args()

failed_list_be = []
failed_list_cere = []

if not args.skip_be:
    print("Predicting brain mask")
    os.system('nnUNet_predict -i data/  -o data/brain_mask/ -t 509 -m 3d_fullres -chk model_be -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER ' )
    input_files = [f for f in os.listdir("data/brain_mask") if f.endswith('.nii.gz')]

    print("Obtaining brain img")
    T1_path = 'data/'
    Seg_path = 'data/brain_mask/'
    files = os.listdir(T1_path)
    output_path = 'data/brain_img/'
    for subject in files:
        if not subject.endswith('.nii.gz'):
                continue
        T1_name = subject
        seg_name = subject
        print(T1_name,seg_name)
        T1_img = sitk.ReadImage(T1_path + T1_name)
        Seg_img = sitk.ReadImage(Seg_path + seg_name)
        T1_array = sitk.GetArrayFromImage(T1_img)
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
            output_img  = sitk.GetImageFromArray(T1_array.astype(np.float32))
            output_img.CopyInformation(T1_img)
            sitk.WriteImage(output_img, output_path + T1_name)
            cleaned_img  = sitk.GetImageFromArray(cleaned.astype(np.uint8))
            cleaned_img.CopyInformation(Seg_img)
            sitk.WriteImage(cleaned_img, Seg_path + T1_name)


# print("Bias field correction by N4")
# corrector = sitk.N4BiasFieldCorrectionImageFilter()

# # corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
# # corrector.SetConvergenceThreshold(1e-6)
# # corrector.SetNumberOfHistogramBins(200)
# # corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
# # corrector.SetWienerFilterNoise(0.01)
# # corrector.SetNumberOfThreads(8)
# for subject in os.listdir("brain_img"):
#     origin_brain_img = sitk.ReadImage("brain_img/"+subject)
#     output_N4_brain_img = corrector.Execute(origin_brain_img)
#     sitk.WriteImage(output_N4_brain_img, "brain_img_N4/"+subject)


    
print("Bias field correction by bfc")
T1_path = 'data/brain_img/'
if args.skip_be:
    T1_path = 'data/'
    Brain_Seg_path = 'data/brain_mask/'
    files = os.listdir(T1_path)
    for subject in files:
        T1_name = subject
        T1_img = sitk.ReadImage(T1_path + T1_name)
        T1_array = sitk.GetArrayFromImage(T1_img)
        T1_array[T1_array > 0]=1
        output_img = sitk.GetImageFromArray(T1_array)
        output_img.CopyInformation(T1_img)
        sitk.WriteImage(output_img, Brain_Seg_path + T1_name)    
files = os.listdir(T1_path)
output_path = 'data/brain_img_N4/'  
for subject in files:
    print("Processing",subject)
    cmd = ' /workspace/BrainSuite21a/bin/bfc  -i  %s  -o  %s   ' %(T1_path + subject, output_path + subject)
    # print(cmd)
    os.system(cmd)
# recover the orientation
# 定义输入，目标和输出目录
input_folder = T1_path
target_folder = 'data/brain_img_N4/'
output_folder = 'data/brain_img_N4/'
for filename in os.listdir(target_folder):
    if filename.endswith('.nii.gz'):
        input_path = os.path.join(input_folder, filename)
        target_path = os.path.join(target_folder, filename)
        output_path = os.path.join(output_folder, filename)
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
        sitk.WriteImage(resampled_img, output_path)

    
    
print("Predicting brain cerebellum and brainstem mask")
# os.system('nnUNet_predict -i brain/  -o data/brain_cere_removal_mask/ -t 507 -m 3d_fullres -chk model_best -tr nnUNetTrainerV2_DA3_BN -f 0 ' )
os.system('nnUNet_predict -i data/brain_img_N4/  -o data/brain_cere_removal_mask/ -t 509 -m 3d_fullres -chk model_rmcere -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER ' )


print("Obtaining brain cerebrum img")
os.system('rm -rf data/brain_cere_removal_mask/*plans.pkl ' )
T1_path = 'data/brain_img_N4/'
Brain_Seg_path = 'data/brain_mask/'
Seg_path = 'data/brain_cere_removal_mask/'
files = os.listdir(Seg_path)
output_path = 'data/brain_cerebrum/'
output_path2 = 'data/brain_cerebrum_mask/'
for subject in files:
    # if os.path.splitext(subject)[1]  != '.nii.gz':
    #     continue
    T1_name = subject
    seg_name = subject
    T1_img = sitk.ReadImage(T1_path + T1_name)
    Seg_img = sitk.ReadImage(Seg_path + seg_name)
    Brain_Seg_img = sitk.ReadImage(Brain_Seg_path + seg_name)
    print("Processing", T1_name)
    T1_array = sitk.GetArrayFromImage(T1_img)
    Seg_array = sitk.GetArrayFromImage(Seg_img)
    Brain_Seg_array = sitk.GetArrayFromImage(Brain_Seg_img)
    Brain_rmcere_mask = Brain_Seg_array - Seg_array     # 脑掩码减去小脑+脑干掩码
    Brain_rmcere_mask = morphology.opening(Brain_rmcere_mask, morphology.ball(3))
    arr = Brain_rmcere_mask > 0
    labelled = measure.label(arr)
    rp = measure.regionprops(labelled)
    if not rp:
        print("Failed to extract cerebrum for scan ",T1_name)
        failed_list_cere.append(T1_name)
        
    else:
        size = max([i.area for i in rp])
        cleaned = morphology.remove_small_objects(arr, min_size=size-1).astype(np.uint8)
        cere_array = Brain_Seg_array - cleaned
        cere_array=morphology.opening(cere_array,morphology.ball(3))
        arr = cere_array > 0
        labelled = measure.label(arr)
        rp = measure.regionprops(labelled)

        T1_array[cleaned!=1]=0
        output_array=T1_array
    # output_array2=cleaned_cere
    
        output_img  = sitk.GetImageFromArray(T1_array)
        output_img2  = sitk.GetImageFromArray(cleaned)
        output_img.CopyInformation(T1_img)
        output_img2.CopyInformation(T1_img)

        sitk.WriteImage(output_img, output_path + T1_name)
        sitk.WriteImage(output_img2, output_path2 + seg_name)



print("Predicting brain tissue segmentation")    
os.system('nnUNet_predict -i data/brain_cerebrum/  -o data/brain_tissue/ -t 509 -m 3d_fullres -chk model_ts -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER --step_size 0.1 ' )


print("Predicting subcortical")
# os.system('nnUNet_predict -i brain/  -o data/brain_cere_removal_mask/ -t 507 -m 3d_fullres -chk model_best -tr nnUNetTrainerV2_DA3_BN -f 0 ' )
os.system('nnUNet_predict -i data/brain_img_N4/  -o data/brain_subcortical/ -t 509 -m 3d_fullres -chk model_subcortical_single -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER ' )





if failed_list_be:
    print("Unable to extract brain for scan",failed_list_be,"and the rest have been processed.")     
if failed_list_cere:
    print("Unable to extract cerebrum for scan",failed_list_cere,"and the rest have been processed.")  

os.chdir(cwd)
