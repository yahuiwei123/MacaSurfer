#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import ConvexHull
from utils import *

def generate_brain_mask_from_brain_img(img_path, mask_path, refine: bool = False):
    """                                                                                                                      
    【最保险版本】优先保留所有脑组织，完全避免偏置场导致的低信号丢失                                                         
    输入已经是去颅骨的脑图像，只需去掉极少背景即可                                                                           
    """                                                                                                                      
    img = nib.load(img_path)                                                                                                 
    data = img.get_fdata()                                                                                                   
    affine = img.affine                                                                                                      

    if refine:                                                                                                                         
        # 1. 用极低阈值去掉纯背景，保留几乎所有信号                                                                              
        # 只去掉信号最低的5%的体素（这些肯定是背景/噪声），剩下的全部保留                                                        
        # 百分位可以根据实际情况调到3%-7%，数值越小越保守                                                                        
        threshold = 0 # np.percentile(data, 0)                                                                                       
        binary = data > threshold                                                                                                
                                                                                                                                
        # 2. 提取最大连通分量，仅去除零星的噪声斑点                                                                              
        labeled, num_features = ndimage.label(binary)                                                                            
        if num_features > 0:                                                                                                     
            sizes = ndimage.sum(binary, labeled, range(num_features + 1))                                                        
            max_label = np.argmax(sizes[1:]) + 1                                                                                 
            largest_component = labeled == max_label                                                                             
        else:                                                                                                                    
            largest_component = binary                                                                                           
                                                                                                                                
        # 3. 填充内部孔洞                                                                                                        
        filled = ndimage.binary_fill_holes(largest_component)                                                                    
                                                                                                                                
        # 4. 充分膨胀扩展边界，确保所有边缘低信号区域都被包含                                                                    
        # 用二阶结构元，膨胀3次，即使带一点背景也没关系                                                                          
        struct = ndimage.generate_binary_structure(3, 2)                                                                         
        expanded = ndimage.binary_dilation(filled, structure=struct, iterations=3)                                               
                                                                                                                                
        # 5. 最终用凸包做保险，保证所有脑区都在mask内                                                                            
        coords = np.array(np.where(expanded)).T                                                                                  
        final_mask = expanded                                                                                                    
        if coords.shape[0] >= 4:                                                                                                 
            try:                                                                                                                 
                hull = ConvexHull(coords)                                                                                        
                x, y, z = np.indices(data.shape)                                                                                 
                points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T                                                          
                A = hull.equations[:, :-1]                                                                                       
                b = hull.equations[:, -1]                                                                                        
                inside = np.all(points @ A.T + b <= 1e-6, axis=1)                                                                
                convex_mask = inside.reshape(data.shape).astype(bool)                                                            
                final_mask = convex_mask  # 直接用凸包，最保险                                                                   
            except Exception as e:                                                                                               
                print(f"Warning: Convex hull failed, using expanded mask: {e}")
    else:
        threshold = 0
        final_mask = data > threshold
    # 保存mask                                                                                                               
    mask_img = nib.Nifti1Image(final_mask.astype(np.int16), affine)                                                          
    nib.save(mask_img, mask_path)                                                                                            
                                                                                                                            
    return threshold

def skullstrip_single_run(run_path: str, out_dir: str, is_brain: bool = False):
    """
    对单个run进行颅骨剥离处理
    """
    run_name = os.path.basename(run_path).replace('.nii.gz', '')
    
    print(f"\n=== Skullstriping single run: {run_name} ===")
    
    # 创建输出目录
    make_dirs(out_dir)
    
    head_img = os.path.join(out_dir, "head.nii.gz")
    brain_mask_img = os.path.join(out_dir, "brainmask.nii.gz")
    brain_img = os.path.join(out_dir, "brain.nii.gz")

    # 复制初始文件
    if is_brain == 'true':
        safe_copy(run_path, brain_img)
        print(f"Input is already brain image, copying to {brain_img}")
    else:
        safe_copy(run_path, head_img)
        print(f"Input is head image, copying to {head_img}")
    
    # 处理逻辑
    if is_brain == 'true':
        # 如果是已经剥离颅骨的图像，直接进行偏置场校正
        run_cmd([
            'N4BiasFieldCorrection', '-i', brain_img, '-o', brain_img,
            '-d', '3', '-b', '[2x2x2,3]', '-s', '3',
            '-c', '[100x50x25x10,0]', '-t', '[0.15,0.01,200]'
        ])

        # 使用Python实现的自动mask生成，应对背景非零情况
        threshold = generate_brain_mask_from_brain_img(brain_img, brain_mask_img)
        print(f"Brain mask created with auto-thresholding (threshold={threshold:.2f}), largest component and hole filling: {brain_mask_img}")
        run_cmd(['fslmaths', brain_img, '-mas', brain_mask_img, brain_img])
    else:
        # 偏置场校正
        run_cmd([
            'N4BiasFieldCorrection', '-i', head_img, '-o', head_img,
            '-d', '3', '-b', '[2x2x2,3]', '-s', '3',
            '-c', '[100x50x25x10,0]', '-t', '[0.15,0.01,200]'
        ])
        print("Bias field correction completed")
        
        # nBEST 分割
        if nBEST_UNET_MODEL:
            nbest_work_dir = os.path.join(out_dir, "nBEST")
            make_dirs(nbest_work_dir)
            safe_copy(head_img, os.path.join(nbest_work_dir, 'head.nii.gz'))
            run_cmd([PYTHON_INTER, os.path.join(nBEST_UNET_MODEL, 'scripts', 'nBEST_brain.py'), 
                    '--python_env', PYTHON_ENV, '--workdir', nbest_work_dir])
            safe_copy(os.path.join(nbest_work_dir, 'brain_mask', 'head.nii.gz'), brain_mask_img)
            safe_copy(os.path.join(nbest_work_dir, 'brain_img', 'head.nii.gz'), brain_img)
        else:
            # 使用 macaUNet 进行脑提取
            run_cmd([PYTHON_INTER, os.path.join(MACA_UNET_MODEL, 'predict.py'), 
                    '--img', head_img, '--out', brain_mask_img])
            
            # 形态学操作
            run_cmd(['fslmaths', brain_mask_img, '-kernel', 'sphere', '0.6', '-dilD', brain_mask_img])
            run_cmd(['fslmaths', head_img, '-mas', brain_mask_img, brain_img])
            print(f"Brain extraction completed: {brain_img}")
    
    return {
        'head': head_img if os.path.exists(head_img) else None,
        'brain': brain_img,
        'brainmask': brain_mask_img,
        'run_name': run_name
    }

def main():
    parser = argparse.ArgumentParser(description="Skull stripping for single run")
    parser.add_argument("--run_path", type=str, required=True, help="Path to the run NIfTI file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for this run")
    parser.add_argument("--is_brain", type=str, default='false', help="Image modality")
    parser.add_argument("--python_inter", type=str, default="python", help="Python interpreter")
    parser.add_argument("--utils_path", type=str, required=True, help="Path to utils")
    parser.add_argument("--maca_unet", type=str, required=True, help="Path to macaUNet model")
    parser.add_argument("--nbest_unet", type=str, default=None, help="Path to nBEST model")
    parser.add_argument("--python_env", type=str, default=None, help="Path to nBEST model")
    
    args = parser.parse_args()
    
    # 设置全局变量
    global PYTHON_INTER, UTILS_PATH, MACA_UNET_MODEL, nBEST_UNET_MODEL, PYTHON_ENV
    PYTHON_INTER = args.python_inter
    UTILS_PATH = args.utils_path
    MACA_UNET_MODEL = args.maca_unet
    nBEST_UNET_MODEL = args.nbest_unet
    PYTHON_ENV = args.python_env
    
    # 处理单个run
    result = skullstrip_single_run(
        run_path=args.run_path,
        out_dir=args.output_dir,
        is_brain=args.is_brain
    )
    
    print(f"✅ Skull stripping completed for {result['run_name']}")
    print(f"   Brain image: {result['brain']}")
    print(f"   Brain mask: {result['brainmask']}")
    if result['head']:
        print(f"   Head image: {result['head']}")

if __name__ == "__main__":
    main()