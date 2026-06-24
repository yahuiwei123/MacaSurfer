#!/usr/bin/env python3
import os
import numpy as np
import argparse
import nibabel as nib
import ast
from utils import *

def register_brain_to_reference(mov_brain, ref_brain, work_dir, utils_path, python_inter):
    """
    将移动图像配准到参考图像
    """
    make_dirs(work_dir)
    
    register_script = os.path.join(utils_path, "generalRegister.sh")
    run_cmd(["sh", register_script,
            "-i", mov_brain,
            "-o", work_dir,
            "-w", work_dir,
            "-r", ref_brain,
            "-m", "rigid", "-l", "MI", "-f", "aladin", "-d", "gpu"])
    
    reg_mat = os.path.join(work_dir, "linear.mat")
    inv_mat = os.path.join(work_dir, "linear_inv.mat")
    run_cmd(["convert_xfm", "-inverse", "-omat", inv_mat, reg_mat])
    
    # 配准脑图像
    final_brain = os.path.join(work_dir, "final_brain.nii.gz")
    run_cmd(["flirt", "-in", mov_brain, "-ref", ref_brain,
            "-applyxfm", "-init", reg_mat, "-interp", "trilinear",
            "-out", final_brain])
    
    return final_brain, reg_mat

def register_head_to_reference(mov_head, ref_brain, reg_mat, work_dir):
    """
    将移动头图像配准到参考图像
    """
    final_head = os.path.join(work_dir, "final_head.nii.gz")
    run_cmd(["flirt", "-in", mov_head, "-ref", ref_brain,
            "-applyxfm", "-init", reg_mat, "-interp", "trilinear",
            "-out", final_head])
    return final_head

def register_to_reference(ref_dir, run_dir):
    """
    对特定模态的所有runs进行平均
    """
    ref_name = os.path.basename(ref_dir)
    
    # 添加参考run的脑图像
    ref_brain = os.path.join(ref_dir, "brain_reorient.nii.gz")
    if os.path.exists(ref_brain):
        print(f"添加参考run脑图像: {ref_brain}")
    else:
        print(f"错误: 参考run脑图像不存在 {ref_brain}")
        return
    
    # 添加参考run的头图像（如果存在）
    ref_head = os.path.join(ref_dir, "head_reorient.nii.gz")
    if os.path.exists(ref_head):
        print(f"添加参考run头图像: {ref_head}")
    
    # 处理非参考run
    run_name = os.path.basename(run_dir)
    
    mov_brain = os.path.join(run_dir, "brain_reorient.nii.gz")
    if not os.path.exists(mov_brain):
        print(f"警告: 移动run脑图像不存在 {mov_brain}")
    
    # 创建配准工作目录
    work_dir = os.path.join(run_dir, f"{run_name}_to_{ref_name}")
    
    try:
        # 配准脑图像
        final_brain, reg_mat = register_brain_to_reference(
            mov_brain, ref_brain, work_dir, UTILS_PATH, PYTHON_INTER
        )
        safe_copy(final_brain, os.path.join(run_dir, f'{run_name}_to_{ref_name}_brain.nii.gz'))
        print(f"配准完成: {run_name} -> 脑图像")
        
        # 配准头图像（如果存在）
        mov_head = os.path.join(run_dir, "head_reorient.nii.gz")
        if os.path.exists(mov_head):
            final_head = register_head_to_reference(mov_head, ref_brain, reg_mat, work_dir)
            safe_copy(final_brain, os.path.join(run_dir, f'{run_name}_to_{ref_name}_head.nii.gz'))
            print(f"配准完成: {run_name} -> 头图像")
            
    except Exception as e:
        print(f"错误: 配准 {run_name} 失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="多张影像平均处理")
    parser.add_argument("--refer_dir", type=str, required=True, help="Reference subject path")
    parser.add_argument("--run_dir", type=str, required=True, help="All subject path")
    parser.add_argument("--python_inter", type=str, default="python", help="Python解释器")
    parser.add_argument("--utils_path", type=str, required=True, help="工具脚本路径")
    
    
    args = parser.parse_args()
    
    # 设置全局变量
    global PYTHON_INTER, UTILS_PATH
    PYTHON_INTER = args.python_inter
    UTILS_PATH = args.utils_path
    
    # 处理单个模态
    register_to_reference(args.refer_dir, args.run_dir)

if __name__ == "__main__":
    main()