#!/usr/bin/env python3
import os
import argparse
import nibabel as nib
from utils import *

def fix_brainmask_single_run(run_name: str, refer_name: str,
                             run_dir: str, refer_dir: str):
    """
    使用参考run修复目标run的脑掩码
    """
    print(f"\n=== Fixing brainmask for {run_name} using reference {refer_name} ===")
    
    # 创建输出目录
    make_dirs(run_dir)
    
    # 获取目标run的文件路径
    if os.path.exists(os.path.join(run_dir, "head.nii.gz")):
        target_head = os.path.join(run_dir, "head.nii.gz")
    else:
        target_head = os.path.join(run_dir, "brain.nii.gz")
    
    # 获取参考run的文件路径
    if os.path.exists(os.path.join(refer_dir, "head.nii.gz")):
        reference_head = os.path.join(refer_dir, "head.nii.gz")
    else:
        reference_head = os.path.join(refer_dir, "brain.nii.gz")
    
    reference_brainmask = os.path.join(refer_dir, "brainmask.nii.gz")
    
    # 创建配准目录
    fix_dir = os.path.join(run_dir, "register_brainmask")
    make_dirs(fix_dir)
    
    # 将目标图像转换到参考图像的空间
    backup_head = os.path.join(fix_dir, run_name + '.nii.gz')
    run_cmd([PYTHON_INTER, os.path.join(UTILS_PATH, 'conform.py'), 
             "--input", target_head, "--output", backup_head, 
             "--reorient", get_orient(nib.load(reference_head).affine)])
    
    # 配准
    reg_mat = os.path.join(fix_dir, "linear.mat")
    register_script = os.path.join(UTILS_PATH, "generalRegister.sh")
    run_cmd(["bash", register_script,
            "-i", backup_head,
            "-o", fix_dir,
            "-w", fix_dir,
            "-r", reference_head,
            "-m", "rigid", "-l", "MI", "-f", "aladin", "-d", "gpu"])
    
    # 计算逆变换
    inv_mat = os.path.join(fix_dir, "linear_inv.mat")
    run_cmd(["convert_xfm", "-inverse", "-omat", inv_mat, reg_mat])
    
    # 应用变换到参考脑掩码
    fixed_brainmask = os.path.join(run_dir, "brainmask_fixed.nii.gz")
    run_cmd(["flirt", "-in", reference_brainmask, "-ref", backup_head,
            "-applyxfm", "-init", inv_mat, "-interp", "nearestneighbour",
            "-out", fixed_brainmask])
    
    # 转换回原始方向
    run_cmd([PYTHON_INTER, os.path.join(UTILS_PATH, 'conform.py'), 
             "--input", fixed_brainmask, "--output", fixed_brainmask, 
             "--reorient", get_orient(nib.load(target_head).affine)])
    
    # 应用修复后的脑掩码
    run_cmd(['fslmaths', target_head, '-mas', fixed_brainmask,
            os.path.join(run_dir, "brain_fixed.nii.gz")])
    
    print(f"✅ Fixed brainmask for {run_name} saved as {fixed_brainmask}")
    
    safe_move(os.path.join(run_dir, "brainmask.nii.gz"), os.path.join(run_dir, "brainmask_nofix.nii.gz"))
    safe_move(os.path.join(run_dir, "brain.nii.gz"), os.path.join(run_dir, "brain_nofix.nii.gz"))
    safe_move(fixed_brainmask, os.path.join(run_dir, "brainmask.nii.gz"))
    safe_move(os.path.join(run_dir, "brain_fixed.nii.gz"), os.path.join(run_dir, "brain.nii.gz"))
    
    return {
        'fixed_brainmask': fixed_brainmask,
        'fixed_brain': os.path.join(run_dir, "brain_fixed.nii.gz"),
        'run_name': run_name,
        'refer_name': refer_name
    }

def main():
    parser = argparse.ArgumentParser(description="Fix brain mask for single run using reference run")
    parser.add_argument("--run_name", type=str, required=True, help="Path to target run (need fixing)")
    parser.add_argument("--refer_name", type=str, required=True, help="Path to reference run (good brainmask)")
    parser.add_argument("--run_dir", type=str, required=True, help="Output directory for target run")
    parser.add_argument("--refer_dir", type=str, required=True, help="Directory containing reference run's brainmask files")
    parser.add_argument("--python_inter", type=str, default="python", help="Python interpreter")
    parser.add_argument("--utils_path", type=str, required=True, help="Path to utils")
    
    args = parser.parse_args()
    
    # 设置全局变量
    global PYTHON_INTER, UTILS_PATH
    PYTHON_INTER = args.python_inter
    UTILS_PATH = args.utils_path
    
    # 修复单个run的脑掩码
    result = fix_brainmask_single_run(
        run_name=args.run_name,
        refer_name=args.refer_name,
        run_dir=args.run_dir,
        refer_dir=args.refer_dir
    )
    
    print(f"Brain mask fixing completed:")
    print(f"   Target run: {result['run_name']}")
    print(f"   Reference run: {result['refer_name']}")
    print(f"   Fixed brainmask: {result['fixed_brainmask']}")
    print(f"   Fixed brain: {result['fixed_brain']}")

if __name__ == "__main__":
    main()