import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from skimage import morphology, segmentation

def disconnect_thin_connections(mask, min_connection_strength=3):
    """
    断开细小连接而不改变主要区域形状
    使用距离变换来识别和断开细小连接
    
    Parameters:
    mask: 输入的二值掩码
    min_connection_strength: 最小连接强度阈值，值越大表示只断开越细的连接
    """
    # 计算距离变换（每个体素到背景的距离）
    distance = distance_transform_edt(mask)
    
    # 使用阈值来识别细小连接区域
    # 距离值小于阈值的区域被认为是细小连接
    thin_connections = distance < min_connection_strength
    
    # 如果几乎没有细小连接，直接返回原始掩码
    if np.sum(thin_connections) < 10:  # 小阈值，避免处理几乎没有细小连接的情况
        return mask, [mask]
    
    # 从原始掩码中移除细小连接区域
    disconnected_mask = mask & ~thin_connections
    
    # 在移除细小连接后的掩码上寻找连通域
    labeled_disconnected, num_disconnected = label(disconnected_mask)
    
    # 如果移除细小连接后区域完全断开，返回分割后的区域
    if num_disconnected > 1:
        disconnected_regions = []
        for region_label in range(1, num_disconnected + 1):
            region_mask = (labeled_disconnected == region_label)
            disconnected_regions.append(region_mask)
        return disconnected_mask, disconnected_regions
    
    # 如果移除细小连接后仍然是一个区域，尝试使用分水岭算法
    try:
        # 找到距离变换的局部最大值作为标记
        local_maxi = morphology.local_maxima(distance)
        
        # 如果找不到足够的局部最大值，直接返回原始掩码
        if np.sum(local_maxi) < 2:
            return mask, [mask]
        
        # 使用分水岭算法基于距离变换进行分割
        markers = label(local_maxi)[0]
        labels = segmentation.watershed(-distance, markers, mask=mask)
        
        # 获取所有分割后的区域
        disconnected_regions = []
        for region_label in range(1, np.max(labels) + 1):
            region_mask = (labels == region_label)
            if np.sum(region_mask) > 0:  # 确保区域不为空
                disconnected_regions.append(region_mask)
        
        # 如果没有成功分割，返回原始掩码
        if len(disconnected_regions) <= 1:
            return mask, [mask]
        
        return mask, disconnected_regions
        
    except Exception as e:
        print(f"分水岭算法出错: {e}")
        return mask, [mask]

def reconnect_adjacent_regions(regions, original_csf_mask):
    """
    重新连接原本相邻的区域
    检查哪些区域在原始CSF掩码中是连通的，将它们重新合并
    """
    if len(regions) <= 1:
        return regions
    
    # 创建所有区域的组合掩码
    combined_mask = np.zeros_like(original_csf_mask, dtype=bool)
    for region in regions:
        combined_mask = combined_mask | region
    
    # 在原始CSF掩码约束下寻找连通域
    constrained_combined = combined_mask & original_csf_mask
    labeled_combined, num_combined = label(constrained_combined)
    
    # 如果重新连接后区域数量减少，说明有区域原本是连通的
    if num_combined < len(regions):
        print(f"重新连接: 从 {len(regions)} 个区域合并为 {num_combined} 个区域")
        
        # 获取重新连接后的区域
        reconnected_regions = []
        for region_label in range(1, num_combined + 1):
            region_mask = (labeled_combined == region_label)
            reconnected_regions.append(region_mask)
        
        return reconnected_regions
    
    return regions

def identify_tumor_regions(label_file, output_file=None, min_connection_strength=3, 
                          max_regions_to_analyze=6, output_top_n=1):
    # 加载标签影像
    img = nib.load(label_file)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    # 创建一个掩码，标记所有脑脊液区域（标签1）
    csf_mask = (data == 1).astype(np.int32)
    
    # 先进行连通域分析，找到所有原始连通域
    original_labeled, num_original_features = label(csf_mask)
    print(f"原始图像中找到 {num_original_features} 个脑脊液连通域")
    
    # 计算每个连通域的大小并排序
    region_sizes = []
    for region_label in range(1, num_original_features + 1):
        region_mask = (original_labeled == region_label)
        region_sizes.append(np.sum(region_mask))
    
    # 按大小排序并只取前6个最大的区域
    sorted_indices = np.argsort(region_sizes)[::-1]  # 从大到小排序
    regions_to_analyze = min(max_regions_to_analyze, num_original_features)
    
    print(f"只分析前 {regions_to_analyze} 个最大的连通域:")
    for i, idx in enumerate(sorted_indices[:regions_to_analyze]):
        print(f"  区域 {idx + 1}: {region_sizes[idx]} 体素")
    
    # 收集所有满足条件的区域
    candidate_regions = []
    candidate_sizes = []
    candidate_original_indices = []  # 记录每个候选区域来自哪个原始区域
    
    # 只对前6个最大的连通域进行处理
    for i, idx in enumerate(sorted_indices[:regions_to_analyze]):
        region_label = idx + 1
        original_region_mask = (original_labeled == region_label)
        region_size = region_sizes[idx]
        
        print(f"\n处理第 {i+1} 大区域 (原始标签 {region_label})，体积: {region_size} 体素")
        
        # 尝试断开细小连接
        disconnected_mask, disconnected_regions = disconnect_thin_connections(
            original_region_mask, min_connection_strength
        )
        
        # 如果成功分割成多个区域
        if len(disconnected_regions) > 1:
            print(f"  区域被分割成 {len(disconnected_regions)} 个子区域")
            
            # 检查每个子区域是否满足条件
            for j, sub_region in enumerate(disconnected_regions):
                sub_region_size = np.sum(sub_region)
                print(f"    子区域 {j+1}，体积: {sub_region_size} 体素")
                if sub_region_size < 56: continue
                
                # 检查该子区域是否满足条件
                if check_region_conditions(sub_region, data):
                    candidate_regions.append(sub_region)
                    candidate_sizes.append(sub_region_size)
                    candidate_original_indices.append(region_label)  # 记录原始区域标签
                    print(f"    ✓ 子区域 {j+1} 满足条件")
                else:
                    print(f"    ✗ 子区域 {j+1} 不满足条件")
        else:
            # 如果没有被分割，检查原始区域
            if check_region_conditions(original_region_mask, data):
                candidate_regions.append(original_region_mask)
                candidate_sizes.append(region_size)
                candidate_original_indices.append(region_label)
                print(f"  ✓ 区域满足条件")
            else:
                print(f"  ✗ 区域不满足条件")
    
    # 跳过剩余的小区域分析
    if num_original_features > regions_to_analyze:
        print(f"\n跳过剩余的 {num_original_features - regions_to_analyze} 个较小区域的分析")
    
    if not candidate_regions:
        print("未找到任何满足条件的脑肿瘤区域")
        return None
    
    # 重新检查连通性：将原本相连的区域重新合并
    print("\n重新检查区域连通性...")
    
    # 按原始区域分组，分别重新连接
    final_candidate_regions = []
    final_candidate_sizes = []
    
    unique_original_indices = np.unique(candidate_original_indices)
    
    for orig_idx in unique_original_indices:
        # 找出来自同一个原始区域的所有候选区域
        same_origin_indices = [i for i, orig in enumerate(candidate_original_indices) if orig == orig_idx]
        same_origin_regions = [candidate_regions[i] for i in same_origin_indices]
        
        if len(same_origin_regions) > 1:
            # 重新连接来自同一个原始区域的区域
            reconnected = reconnect_adjacent_regions(same_origin_regions, csf_mask)
            
            for region in reconnected:
                region_size = np.sum(region)
                final_candidate_regions.append(region)
                final_candidate_sizes.append(region_size)
        else:
            # 只有一个区域，直接添加
            final_candidate_regions.append(same_origin_regions[0])
            final_candidate_sizes.append(candidate_sizes[same_origin_indices[0]])
    
    print(f"重新连接后，从 {len(candidate_regions)} 个区域变为 {len(final_candidate_regions)} 个区域")
    
    # 按体积大小排序最终候选区域
    final_candidate_sizes_arr = np.array(final_candidate_sizes)
    sorted_final_indices = np.argsort(final_candidate_sizes_arr)[::-1]
    
    # 获取前n个最大的区域
    output_top_n = min(output_top_n, len(final_candidate_regions))
    top_n_regions = [final_candidate_regions[i] for i in sorted_final_indices[:output_top_n]]
    top_n_sizes = [final_candidate_sizes[i] for i in sorted_final_indices[:output_top_n]]
    
    print(f"\n最终找到 {len(final_candidate_regions)} 个满足条件的区域")
    for i, size in enumerate(top_n_sizes):
        print(f"第 {i+1} 大区域的体积为: {size} 体素")
    
    # 创建输出影像（保存前n个最大的区域，用不同标签值区分）
    if output_top_n == 1:
        # 如果只输出1个区域，直接保存
        tumor_mask = top_n_regions[0].astype(np.int16)
        tumor_img = nib.Nifti1Image(tumor_mask, affine, header)
        
        if output_file:
            nib.save(tumor_img, output_file)
            print(f"最大肿瘤区域已保存到: {output_file}")
    else:
        # 如果输出多个区域，用不同标签值保存
        combined_mask = np.zeros_like(data, dtype=np.int16)
        for i, region in enumerate(top_n_regions, 1):
            combined_mask[region] = i
        
        combined_img = nib.Nifti1Image(combined_mask, affine, header)
        
        if output_file:
            nib.save(combined_img, output_file)
            print(f"前 {output_top_n} 大肿瘤区域已保存到: {output_file}")
            print(f"标签值说明: 1=最大区域, 2=第二大区域, ..., {output_top_n}=第{output_top_n}大区域")
    
    # 可选：保存所有满足条件的区域
    if len(final_candidate_regions) > output_top_n:
        save_all_candidates(final_candidate_regions, output_file, affine, header)
    
    return top_n_regions[0] if output_top_n == 1 else top_n_regions

def check_region_conditions(region_mask, original_data):
    """
    检查区域是否满足条件：不与背景0相连，被灰质/白质包围
    """
    # 检查该区域是否与背景0相邻
    dilated_region = ndimage.binary_dilation(region_mask, structure=np.ones((3,3,3)))
    bordering_voxels = dilated_region & ~region_mask
    
    # 获取边界体素的标签
    bordering_labels = original_data[bordering_voxels]
    
    # 如果边界中有背景0，则不满足条件
    if 0 in bordering_labels:
        return False
    
    # 检查边界中是否只有脑脊液(1)、灰质(2)和白质(3)
    unique_bordering = np.unique(bordering_labels)
    return all(label_val in [1, 2, 3] for label_val in unique_bordering)

def save_all_candidates(candidate_regions, output_file, affine, header):
    """
    保存所有候选区域
    """
    # 创建包含所有候选区域的影像
    all_regions_mask = np.zeros(candidate_regions[0].shape, dtype=np.int16)
    for i, region in enumerate(candidate_regions, 1):
        all_regions_mask[region] = i
    
    all_regions_img = nib.Nifti1Image(all_regions_mask, affine, header)
    base_name = output_file.split('.')[0] if output_file else "all_candidates"
    all_output_file = f"{base_name}_all_candidates.nii.gz"
    nib.save(all_regions_img, all_output_file)
    print(f"所有满足条件的区域已保存到: {all_output_file}")

def main():
    parser = argparse.ArgumentParser(description='识别被错误标记为脑脊液的脑肿瘤区域')
    parser.add_argument('--input', '-i', required=True, help='输入标签nii文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出肿瘤掩码nii文件路径')
    parser.add_argument('--min-connection', '-c', type=int, default=5, 
                       help='最小连接强度阈值')
    parser.add_argument('--max-regions', '-m', type=int, default=6, 
                       help='分析的最大连通域数量（默认: 6）')
    parser.add_argument('--output-top-n', '-n', type=int, default=1,
                       help='输出的前n大区域数量（默认: 1，只输出最大区域）')
    
    args = parser.parse_args()
    
    # 执行肿瘤区域识别
    identify_tumor_regions(args.input, args.output, args.min_connection, 
                          args.max_regions, args.output_top_n)

if __name__ == "__main__":
    main()