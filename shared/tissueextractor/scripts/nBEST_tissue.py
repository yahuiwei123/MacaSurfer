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


    dst_dit_list = [f"{args.workdir}/brain_cerebellum_brainstem_mask",
                    f"{args.workdir}/brain_cerebrum",
                    f"{args.workdir}/brain_cerebrum_mask",
                    f"{args.workdir}/brain_tissue"]

    for path in dst_dit_list:
        if not os.path.exists(path):
            os.makedirs(path)

    failed_list_cere = []

    nbest_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pythonpath = f"PYTHONPATH={nbest_root}:$PYTHONPATH"

    # 获取小脑和脑干的mask
    print("Predicting brain cerebellum and brainstem mask")
    os.system(f'{pythonpath} {args.python_env}/bin/nnUNet_predict -i {args.workdir}/brain_img/ -o {args.workdir}/brain_cerebellum_brainstem_mask -t 509 -m 3d_fullres -chk model_rmcere -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER ' )
    rmcere_output = f'{args.workdir}/brain_cerebellum_brainstem_mask/T1w_conform.nii.gz'
    rmcere_output_modality = f'{args.workdir}/brain_cerebellum_brainstem_mask/T1w_conform_0000.nii.gz'
    if not os.path.exists(rmcere_output) and os.path.exists(rmcere_output_modality):
        os.rename(rmcere_output_modality, rmcere_output)
    if not os.path.exists(rmcere_output):
        raise FileNotFoundError(rmcere_output)

    # 提取大脑
    print("Obtaining brain cerebrum img")
    os.system(f'rm -rf {args.workdir}/brain_cerebellum_brainstem_mask/*plans.pkl ' )
    t1_path = f'{args.workdir}/brain_img/'
    brain_path = f'{args.workdir}/brain_mask/'
    mask_path = f'{args.workdir}/brain_cerebellum_brainstem_mask/'
    files = os.listdir(mask_path)
    cerebrum_path = f'{args.workdir}/brain_cerebrum/'
    cerebrum_mask_path = f'{args.workdir}/brain_cerebrum_mask/'
    for subject in files:
        if subject.endswith('plans.pkl'):
            continue
        t1_name = subject
        mask_name = subject
        t1_input_name = subject.replace('.nii.gz', '_0000.nii.gz')
        t1_img = sitk.ReadImage(t1_path + t1_input_name)
        mask_img = sitk.ReadImage(mask_path + mask_name)
        brain_mask_img = sitk.ReadImage(brain_path + mask_name)
        
        print(f"extracting brain from {t1_name}")
        t1_array = sitk.GetArrayFromImage(t1_img)
        mask_array = sitk.GetArrayFromImage(mask_img)
        brain_mask_array = sitk.GetArrayFromImage(brain_mask_img)
        
        # 脑掩码减去小脑+脑干掩码
        brain_rmcere_mask = brain_mask_array - mask_array
        brain_rmcere_mask = morphology.opening(brain_rmcere_mask, morphology.ball(3))
        arr = np.where(brain_rmcere_mask > 0, 1, 0)

        labelled = measure.label(arr) 
        rp = measure.regionprops(labelled)
        if not rp:
            print("Failed to extract cerebrum for scan ", t1_name)
            failed_list_cere.append(t1_name)
        else:
            size = max([i.area for i in rp])
            cleaned = morphology.remove_small_objects(arr, min_size=size-1).astype(np.uint8)
            cere_array = brain_mask_array - cleaned
            cere_array=morphology.opening(cere_array,morphology.ball(3))
            arr = cere_array > 0
            labelled = measure.label(arr)
            rp = measure.regionprops(labelled)

            t1_array[cleaned != 1]=0
            output_array = t1_array
        
            cerebrum_img  = sitk.GetImageFromArray(t1_array)
            cerebrum_mask  = sitk.GetImageFromArray(cleaned)
            cerebrum_img.CopyInformation(t1_img)
            cerebrum_mask.CopyInformation(t1_img)

            sitk.WriteImage(cerebrum_img, cerebrum_path + t1_input_name)
            sitk.WriteImage(cerebrum_mask, cerebrum_mask_path + mask_name)

    print("Predicting brain tissue segmentation")
    os.system(f'{pythonpath} {args.python_env}/bin/nnUNet_predict -i {args.workdir}/brain_cerebrum/ -o {args.workdir}/brain_tissue/ -t 509 -m 3d_fullres -chk model_ts -tr nnUNetTrainerV2_DA3_BN_UNeXt_axial_attn -f 1 -p nnUNetPlans_pretrained_IDENTIFIER --step_size 0.1' )
    tissue_output = f'{args.workdir}/brain_tissue/T1w_conform.nii.gz'
    tissue_output_modality = f'{args.workdir}/brain_tissue/T1w_conform_0000.nii.gz'
    if not os.path.exists(tissue_output) and os.path.exists(tissue_output_modality):
        os.rename(tissue_output_modality, tissue_output)
    if not os.path.exists(tissue_output):
        raise FileNotFoundError(tissue_output)

    if failed_list_cere:
        print("Unable to extract cerebrum for scan", failed_list_cere, "and the rest have been processed.")  

    os.chdir(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_env", type=str, default='', help="python environment path")
    parser.add_argument("--workdir", type=str, default='', help="work directory (keep sure your image is in this dir)")
    args = parser.parse_args()
    
    main(args=args)