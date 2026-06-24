import nibabel as nib
import numpy as np
import subprocess
import argparse

def compute_brainmask_centroid(brainmask_path):
    img = nib.load(brainmask_path)
    mask = img.get_fdata() > 0  # binary mask

    # 获取非零点坐标
    coords = np.argwhere(mask)  # shape: [N, 3], in voxel indices

    # 计算坐标均值，即质心
    centroid_voxel = coords.mean(axis=0)
    
    return centroid_voxel

def run_bet2_with_centroid(input_nii, output_root, centroid, f_val=0.5):
    x, y, z = centroid
    print('find centroid:', x, y, z)
    cmd = [
        'bet2',
        input_nii,
        output_root,
        '-c', f'{x:.1f}', f'{y:.1f}', f'{z:.1f}',
        '-f', str(f_val),
        '-r', str(50),
        '-m',  # 输出 mask
        '-v'   # verbose
    ]
    print("Running BET2:", " ".join(cmd))
    subprocess.run(cmd)

def main(args):
    brainmask_path = args.in_msk
    input_nii = args.in_img
    output_root = args.out_msk

    centroid = compute_brainmask_centroid(brainmask_path)
    run_bet2_with_centroid(input_nii, output_root, centroid, f_val=0.5)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_img", type=str, default='', help="input image")
    parser.add_argument("--in_msk", type=str, default='', help="output image")
    parser.add_argument("--out_msk", type=str, default='', help="crop to nonzero area")
    args = parser.parse_args()
    
    main(args=args)