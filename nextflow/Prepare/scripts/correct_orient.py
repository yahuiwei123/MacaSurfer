#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
from utils import *

def correct_orientation_single_run(run_name: str, run_dir: str,
                                   refer_path_path: str, is_brain: str = 'false'):
    """
    对单个run进行方向校正
    """
    print(f"\n=== Correcting orientation for {run_name} ===")
    
    # 创建重定向目录
    reorient_dir = os.path.join(run_dir, "reorient")
    make_dirs(reorient_dir)
    
    # 确定要校正的图像文件
    img_list = []
    out_list = []
    
    # 脑图像和脑掩码
    if os.path.exists(os.path.join(run_dir, "brain.nii.gz")):
        img_list.append(os.path.join(run_dir, "brain.nii.gz"))
        out_list.append(os.path.join(run_dir, "brain_reorient.nii.gz"))
        
        img_list.append(os.path.join(run_dir, "brainmask.nii.gz"))
        out_list.append(os.path.join(run_dir, "brainmask_reorient.nii.gz"))
    
    # 头图像（如果存在）
    if is_brain == 'false' and os.path.exists(os.path.join(run_dir, "head.nii.gz")):
        img_list.append(os.path.join(run_dir, "head.nii.gz"))
        out_list.append(os.path.join(run_dir, "head_reorient.nii.gz"))
    
    # 获取正确的方向
    if refer_path_path == "template":
        # 使用模板作为参考
        orientation = get_correct_orientation(
            mov=img_list[0],  # 使用第一个图像（通常是brain）
            ref=T1_TEMPLATE_BRAIN, 
            workdir=reorient_dir
        )
    else:
        # 使用其他run作为参考
        orientation = get_correct_orientation(
            mov=img_list[0],  # 使用第一个图像（通常是brain）
            ref=refer_path_path, 
            workdir=reorient_dir
        )
    
    # 执行方向校正
    correct_orientation(img_list, orientation, out_list)
    
    print(f"✅ Orientation correction completed for {run_name}")
    
    result = {
        'run_name': run_name,
        'orientation': orientation,
        'corrected_files': out_list
    }
    
    # 打印校正后的文件
    for i, out_file in enumerate(out_list):
        result[f'corrected_file_{i+1}'] = out_file
        print(f"   Corrected: {out_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Correct orientation for single run")
    parser.add_argument("--run_name", type=str, required=True, help="Path to the run NIfTI file")
    parser.add_argument("--is_brain", type=str, required=True, help="if run is brain")
    parser.add_argument("--run_dir", type=str, required=True, help="Output directory for this run (should contain brain/brainmask/head files)")
    parser.add_argument("--refer_path", type=str, required=True, help="Reference for orientation correction (path to refer_path image or 'template')")
    parser.add_argument("--python_inter", type=str, default="python", help="Python interpreter")
    parser.add_argument("--utils_path", type=str, required=True, help="Path to utils")
    parser.add_argument("--t1_template_brain", type=str, required=True, help="T1 template brain path")
    
    args = parser.parse_args()
    
    # 设置全局变量
    global PYTHON_INTER, UTILS_PATH, T1_TEMPLATE_BRAIN
    PYTHON_INTER = args.python_inter
    UTILS_PATH = args.utils_path
    T1_TEMPLATE_BRAIN = args.t1_template_brain
    
    # 校正单个run的方向
    result = correct_orientation_single_run(
        run_name=args.run_name,
        run_dir=args.run_dir,
        refer_path_path=args.refer_path,
        is_brain=args.is_brain
    )
    
    print(f"✅ Orientation correction completed for {result['run_name']}")
    print(f"   Target orientation: {result['orientation']}")

if __name__ == "__main__":
    main()