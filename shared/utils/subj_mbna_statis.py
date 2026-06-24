import os
import glob
import nibabel as nib
import pandas as pd
import argparse
from collections import defaultdict

def parse_label_names(label_array, label_table):
    """返回每个顶点的标签名列表"""
    label_dict = label_table.get_labels_as_dict()
    labels = [label_dict.get(int(idx), "Unknown") for idx in label_array]
    return labels

def split_label(label):
    """将CG.idr分成一级（CG）和二级（CG.idr）"""
    if '.' in label:
        parts = label.split('.')
        return parts[0], label
    else:
        return label, label

def compute_metrics(label_list, metric_array, metric = 'mean'):
    """返回每个一级、二级标签的平均值"""
    level1_dict = defaultdict(list)
    level2_dict = defaultdict(list)
    
    for label, value in zip(label_list, metric_array):
        if label == 'Unknown':
            continue
        lvl1, lvl2 = split_label(label)
        level1_dict[lvl1].append(value)
        level2_dict[lvl2].append(value)
    
    if metric == 'mean':
        level1_avg = {k: sum(v) / len(v) for k, v in level1_dict.items()}
        level2_avg = {k: sum(v) / len(v) for k, v in level2_dict.items()}
        total_avg = sum(metric_array) / len(metric_array)
        return total_avg, level1_avg, level2_avg
    elif metric == 'sum':
        level1_sum = {k: sum(v) for k, v in level1_dict.items()}
        level2_sum = {k: sum(v) for k, v in level2_dict.items()}
        total_sum = sum(metric_array)
        return total_sum, level1_sum, level2_sum
    

def process_subject(folder_path, subject_name, hemi = 'L'):
    label_file = os.path.join(folder_path, f'{hemi}.aparc.label.gii')
    curv_file = os.path.join(folder_path, f'{hemi}.curvature.shape.gii')
    sulc_file = os.path.join(folder_path, f'{hemi}.sulc.shape.gii')
    thick_file = os.path.join(folder_path, f'{hemi}.thickness.shape.gii')
    wmarea_file = os.path.join(folder_path, f'{hemi}.area.white.shape.gii')
    gmarea_file = os.path.join(folder_path, f'{hemi}.area.pial.shape.gii')
    
    # 创建顶点表面积文件
    os.system(f"wb_command -surface-vertex-areas {os.path.join(folder_path, f'{hemi}.white.surf.gii')} {wmarea_file}")
    os.system(f"wb_command -surface-vertex-areas {os.path.join(folder_path, f'{hemi}.pial.surf.gii')} {gmarea_file}")
    
    # 读取标签
    label_gii = nib.load(label_file)
    label_array = label_gii.darrays[0].data
    label_names = parse_label_names(label_array, label_gii.labeltable)

    # 读取指标
    curvature = nib.load(curv_file).darrays[0].data
    sulc = nib.load(sulc_file).darrays[0].data
    thickness = nib.load(thick_file).darrays[0].data
    wmarea = nib.load(wmarea_file).darrays[0].data
    gmarea = nib.load(gmarea_file).darrays[0].data

    # 计算每个指标的 total, level1, level2 平均值
    curv_total, curv_lvl1, curv_lvl2 = compute_metrics(label_names, curvature)
    sulc_total, sulc_lvl1, sulc_lvl2 = compute_metrics(label_names, sulc)
    thick_total, thick_lvl1, thick_lvl2 = compute_metrics(label_names, thickness)
    wmarea_total, wmarea_lvl1, wmarea_lvl2 = compute_metrics(label_names, wmarea, 'sum')
    gmarea_total, gmarea_lvl1, gmarea_lvl2 = compute_metrics(label_names, gmarea, 'sum')

    return {
        'subject': subject_name,
        'total': {'curvature': curv_total, 'sulc': sulc_total, 'thickness': thick_total, 'wmarea': wmarea_total, 'gmarea': gmarea_total},
        'level1': {'curvature': curv_lvl1, 'sulc': sulc_lvl1, 'thickness': thick_lvl1, 'wmarea': wmarea_lvl1, 'gmarea': gmarea_lvl1},
        'level2': {'curvature': curv_lvl2, 'sulc': sulc_lvl2, 'thickness': thick_lvl2, 'wmarea': wmarea_lvl2, 'gmarea': gmarea_lvl2},
    }

def save_to_excel(subj_data, out_dir, level='total', hemi='L'):
    rows = []

    subject = subj_data['subject']
    if level == 'total':
        row = {'subject': subject, 'region': 'total', **subj_data['total']}
        rows.append(row)
    else:
        keys = set()
        for metric in ['curvature', 'sulc', 'thickness', 'wmarea', 'gmarea']:
            keys.update(subj_data[level][metric].keys())
        for key in sorted(keys):
            row = {'subject': subject, 'region': key}
            for metric in ['curvature', 'sulc', 'thickness', 'wmarea', 'gmarea']:
                row[metric] = subj_data[level][metric].get(key, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    outname = os.path.join(out_dir, f'{hemi}_{level}_metrics.xlsx')
    df.to_excel(outname, index=False)
    print(f"Saved {level} metrics to {outname}")

def main(args):
    subj_dir = args.in_dir
    output_dir = args.out_dir
    subj_name = args.subj_name
    
    for hemi in ['L', 'R']:
        results = process_subject(subj_dir, subj_name, hemi)
        save_to_excel(results, output_dir, level='total', hemi=hemi)
        save_to_excel(results, output_dir, level='level1', hemi=hemi)
        save_to_excel(results, output_dir, level='level2', hemi=hemi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default='', help="input image")
    parser.add_argument("--out_dir", type=str, default='', help="output image")
    parser.add_argument("--subj_name", type=str, default='', help="output image")
    args = parser.parse_args()
    
    main(args=args)