import numpy as np
import nibabel as nib
import os
import argparse
from collections import defaultdict

def main(args):
    # ==== 路径配置 ====
    root_dir = args.in_dir
    out_file = args.out_file
    label_index_file = "/home/yhwei/projects/Monkey_Surface/experiments/MBNA/level02/color_table.txt"     # 自定义 label-name 文件

    for hemi in ['L', 'R']:
        label_array_file = os.path.join(root_dir, f"{hemi}.aparc.label.gii")            # 顶点标签数组，值为整数 index
        thickness_file = os.path.join(root_dir, f"{hemi}.thickness.shape.gii")
        curvature_file = os.path.join(root_dir, f"{hemi}.curvature.shape.gii")
        sulc_file = os.path.join(root_dir, f"{hemi}.sulc.shape.gii")

        pial_file = os.path.join(root_dir, f"{hemi}.pial.surf.gii")
        white_file = os.path.join(root_dir, f"{hemi}.white.surf.gii")

        area_pial_file = os.path.join(root_dir, f"{hemi}.area.pial.shape.gii")
        area_white_file = os.path.join(root_dir, f"{hemi}.area.white.shape.gii")

        # ==== Step 1: 读取标签映射 ====
        label_map = {}  # index: label_name

        with open(label_index_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    name = parts[1]
                    
                    if name in ['FPOr', 'FPOri', 'FPOci', 'FPOc']:
                        name = name.replace('FPO', 'FPO.')

                    label_map[idx] = name

        # ==== Step 2: 读取每个顶点的标签 ====
        labels = nib.load(label_array_file).darrays[0].data.astype(int)


        # ==== Step 3: 读取 thickness curvature 和 sulc, 读入 pial 和 white 计算 area.pial 和 area.white ====
        os.system(f"wb_command -surface-vertex-areas {pial_file} {area_pial_file}")
        os.system(f"wb_command -surface-vertex-areas {white_file} {area_white_file}")
        
        thickness = nib.load(thickness_file).darrays[0].data
        sulc = nib.load(sulc_file).darrays[0].data
        curvature = nib.load(curvature_file).darrays[0].data
        
        area_pial = nib.load(area_pial_file).darrays[0].data
        area_white = nib.load(area_white_file).darrays[0].data
        


        # ==== Step 4: 检查长度 ====
        assert len(labels) == len(thickness) == len(sulc) == len(curvature), "数组长度不一致！"

        # ==== Step 5: 遍历每个 label index，计算平均值 ====
        with open(os.path.join(os.path.dirname(out_file), f'{hemi}.level2.' + os.path.basename(out_file)), "w") as f:
            f.write(f"{'Index':<6} {'Label':<20} {'MeanThickness':>15} {'MeanSulc':>15} {'MeanCurv':>15} {'PialArea':>15} {'WhiteArea':>15}\n")
            f.write("-" * 110 + '\n')

            # 计算所有二级脑区的指标
            for idx in sorted(label_map.keys()):
                mask = labels == idx
                if np.sum(mask) == 0:
                    continue
                avg_thick = np.mean(thickness[mask])
                avg_sulc = np.mean(sulc[mask])
                avg_curvature = np.mean(curvature[mask])
                avg_area_pial = np.sum(area_pial[mask])
                avg_area_white = np.sum(area_white[mask])
                label_name = label_map[idx]
                
                f.write(f"{idx:<6} {label_name:<20} {avg_thick:15.4f} {avg_sulc:15.4f} {avg_curvature:15.4f} {avg_area_pial:15.4f} {avg_area_white:15.4f}\n")
        
        with open(os.path.join(os.path.dirname(out_file), f'{hemi}.level1.' + os.path.basename(out_file)), "w") as f:
            f.write(f"{'Index':<6} {'Label':<20} {'MeanThickness':>15} {'MeanSulc':>15} {'MeanCurv':>15} {'PialArea':>15} {'WhiteArea':>15}\n")
            f.write("-" * 110 + '\n')
            
            # 计算所有一级脑区的指标
            coarse_to_indices = defaultdict(list)

            for idx, full_label in label_map.items():
                if '.' not in full_label:
                    continue  # 跳过非细致 label，比如“Unknown”
                coarse_name = full_label.split('.')[0]
                coarse_to_indices[coarse_name].append(idx)

            coarse_label_masks = dict()

            for coarse_name, label_indices in coarse_to_indices.items():
                # 构建一个 mask，选出 all_labels 中属于这些 label 的位置
                mask = np.isin(labels, label_indices)  # shape=(n_vertices,)
                coarse_label_masks[coarse_name] = mask
            
            for idx, (coarse_name, mask) in enumerate(coarse_label_masks.items()):
                if np.sum(mask) == 0:
                    continue
                avg_thick = np.mean(thickness[mask])
                avg_sulc = np.mean(sulc[mask])
                avg_curvature = np.mean(curvature[mask])
                avg_area_pial = np.sum(area_pial[mask])
                avg_area_white = np.sum(area_white[mask])
                f.write(f"{idx:<6} {coarse_name:<20} {avg_thick:15.4f} {avg_sulc:15.4f} {avg_curvature:15.4f} {avg_area_pial:15.4f} {avg_area_white:15.4f}\n")

        with open(os.path.join(os.path.dirname(out_file), f'{hemi}.level0.' + os.path.basename(out_file)), "w") as f:
            f.write(f"{'Index':<6} {'Label':<20} {'MeanThickness':>15} {'MeanSulc':>15} {'MeanCurv':>15} {'PialArea':>15} {'WhiteArea':>15}\n")
            f.write("-" * 110 + '\n')
            
            # 计算不划分脑区的总体指标
            f.write(f"{'None':<6} {'Total':<20} {np.mean(thickness):15.4f} {np.mean(sulc):15.4f} {np.mean(avg_curvature):15.4f} {np.sum(area_pial):15.4f} {np.sum(area_white):15.4f}\n")
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default='', help="input image")
    parser.add_argument("--out_file", type=str, default='', help="output image")
    args = parser.parse_args()
    
    main(args=args)