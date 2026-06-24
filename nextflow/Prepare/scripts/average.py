import argparse
import nibabel as nib
import numpy as np
import os

# 归一化函数，将数据归一化到 [0, 1] 范围
def normalize_to_01(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:  # 防止分母为0
        return np.zeros(data.shape)
    return (data - data_min) / (data_max - data_min)

# 归一化函数，将数据归一化到 [0, 255] 范围
def normalize_to_255(data):
    return np.float32(data * 255)

# 主程序
def average_nii_images(input_files, output_file):
    # 加载所有影像并归一化
    normalized_images = []
    valid_files = []

    for img_path in input_files:
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"Warning: File {img_path} does not exist and will be skipped.")
            continue

        # 如果文件存在，则读取影像
        img = nib.load(img_path)
        data = img.get_fdata()

        # 归一化到 [0, 1]
        norm_data = normalize_to_01(data)
        normalized_images.append(norm_data)
        valid_files.append(img_path)  # 记录有效文件

    # 如果没有有效文件，返回错误
    if len(normalized_images) == 0:
        print("Error: No valid image files to process.")
        return

    # 加和所有归一化的影像
    summed_image = np.sum(normalized_images, axis=0)

    # 归一化加和后的影像到 [0, 255]
    final_image = normalize_to_255(summed_image)

    # 保存最终影像
    final_img = nib.Nifti1Image(final_image, affine=img.affine)
    nib.save(final_img, output_file)

    print(f"Output saved to {output_file}")

# 使用 argparse 解析命令行输入
def parse_args():
    parser = argparse.ArgumentParser(description="Average multiple NIfTI images")
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help="List of NIfTI file paths to average")
    parser.add_argument('--output_file', type=str, required=True, help="Output file path for the averaged image")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析输入的命令行参数
    args = parse_args()

    # 执行平均操作
    average_nii_images(args.input_files, args.output_file)
