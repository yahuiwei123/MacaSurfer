import numpy as np
import nibabel as nib

def extract_non_parenthesis_part(name):
    # 使用partition方法分割，返回(括号前部分, 分隔符, 括号后部分)
    non_parenthesis_part = name.partition('(')[0].strip()
    return non_parenthesis_part

def save_labels_to_gii_and_colortable(path: str, ignore_values=(0, -1)):
    """
    从 CIFTI 文件提取图谱（map），拆分到 L/R 半球，并保存为 .label.gii 文件。
    同时从数据中提取颜色表并保存成指定的格式。
    """
    ignore_values = set(ignore_values)
    img = nib.load(path)
    axes = [img.header.get_axis(i) for i in range(img.ndim)]
    
    if len(axes) != 2:
        raise ValueError(f"预期是 2D CIFTI (maps × grayordinates)，但 axes={len(axes)}")
    
    map_axis, bm_axis = axes[0], axes[1]
    n_maps = map_axis.size

    print(f"File: {path}")
    print(f"Number of maps: {n_maps}")
    print("-" * 60)

    dataobj = img.dataobj  # 懒加载对象

    # 遍历每个 map，提取并保存为 GIFTI
    for i in range(n_maps):
        vec = np.asarray(dataobj[i]).ravel()
        vec = vec[np.isfinite(vec)]
        
        # 转换为整数标签（四舍五入到最接近的整数）
        vec_int = np.rint(vec).astype(np.int64)

        map_name = map_axis.name[i] if hasattr(map_axis, "name") else f"map_{i}"
        map_name = extract_non_parenthesis_part(map_name)

        if vec_int.size == 0:
            print(f"[{i}] {map_name}: No valid labels found (after ignoring {sorted(ignore_values)})")
            continue
        
        # 创建左/右脑半球的 mask（假设对称性）
        left_mask = np.arange(0, len(vec_int)//2)  # 假设左脑区域索引
        right_mask = np.arange(len(vec_int)//2, len(vec_int))  # 假设右脑区域索引
        
        print(left_mask.shape, vec_int.shape)
        left_labels = vec_int[left_mask].astype(np.int32)  # 转为 int32
        right_labels = vec_int[right_mask].astype(np.int32)  # 转为 int32
        
        print(np.max(left_labels), np.min(left_labels))
        
        # 获取 label 信息并创建 LabelTable
        label_table = nib.gifti.GiftiLabelTable()
        if isinstance(map_axis, nib.cifti2.cifti2_axes.LabelAxis):
            for label_id, (label_name, rgba) in map_axis.label[i].items():
                # 创建 GiftiLabel 对象
                label = nib.gifti.GiftiLabel()
                label.key = int(label_id)
                label.label = label_name
                label.red = float(rgba[0])
                label.green = float(rgba[1])
                label.blue = float(rgba[2])
                label.alpha = float(rgba[3])
                # 添加到 label_table
                label_table.labels.append(label)
        
        # 创建左半球 GIFTI 文件
        left_darray = nib.gifti.GiftiDataArray(
            data=left_labels,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_LABEL']
        )
        left_gii = nib.gifti.GiftiImage()
        # 添加 label table
        left_gii.labeltable = label_table
        # 添加数据数组
        left_gii.add_gifti_data_array(left_darray)
        # 添加元数据
        left_gii_meta = nib.gifti.GiftiMetaData()
        left_gii_meta.data.append(nib.gifti.GiftiNVPairs(
            name='AnatomicalStructurePrimary', 
            value='CortexLeft'
        ))
        left_gii.meta = left_gii_meta
        
        # 创建右半球 GIFTI 文件
        right_darray = nib.gifti.GiftiDataArray(
            data=right_labels,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_LABEL']
        )
        right_gii = nib.gifti.GiftiImage()
        # 添加相同的 label table
        right_gii.labeltable = label_table
        # 添加数据数组
        right_gii.add_gifti_data_array(right_darray)
        # 添加元数据
        right_gii_meta = nib.gifti.GiftiMetaData()
        right_gii_meta.data.append(nib.gifti.GiftiNVPairs(
            name='AnatomicalStructurePrimary', 
            value='CortexRight'
        ))
        right_gii.meta = right_gii_meta
        
        # 使用GIFTI保存数据
        left_filename = f"L.{map_name}.label.gii"
        right_filename = f"R.{map_name}.label.gii"
        nib.save(left_gii, left_filename)
        nib.save(right_gii, right_filename)
        
        print(f"Saved left hemisphere labels to: {left_filename}")
        print(f"Saved right hemisphere labels to: {right_filename}")

        # 保存 color table 到文本文件（可选的额外输出）
        color_table_filename = f"{map_name}_color_table.txt"
        with open(color_table_filename, 'w') as f:
            if isinstance(map_axis, nib.cifti2.cifti2_axes.LabelAxis):
                for label_id, (label_name, rgba) in map_axis.label[i].items():
                    # 每个 label 都是：id name R G B A
                    f.write(f"{label_id}   {label_name:<30} {rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}\n")
                print(f"Saved color table to: {color_table_filename}")
            else:
                print(f"Warning: No label table available for map '{map_name}' to save color table.")


if __name__ == "__main__":
    path = "/home/weiyahui/projects/Monkey_Surface/experiments/statistic/atlas/fsaverage_LR32k/Yerkes19_Parcellations_v2.32k_fs_LR.dlabel.nii"  # 改成你的文件路径
    save_labels_to_gii_and_colortable(path)