import numpy as np
from scipy.linalg import svd, polar
import argparse

def main(args):
    affine_path = args.affine
    output_rigid_path = args.output_rigid
    output_scaling_shearing_path = args.output_scaling_shearing
    
    affine_matrix = np.loadtxt(affine_path)
    
    # extract rotation scaling and shearing part R, then shift part T
    A = affine_matrix[:3, :3]
    T = affine_matrix[:3, 3]

    # polar decompose
    R_rigid, R_non_rigid = polar(A)

    # form rigid matrix
    rigid = np.eye(4)
    rigid[:3, :3] = R_rigid
    rigid[:3, 3] = T

    # form non-rigid matrix
    non_rigid = affine_matrix @ np.linalg.inv(rigid)
    
    with open(output_rigid_path, 'w') as f:
        for row in rigid:
            row_str = '  '.join([f'{e:.6f}' for e in row])
            f.write(row_str + '\n')
            
    with open(output_scaling_shearing_path, 'w') as f:
        for row in non_rigid:
            row_str = '  '.join([f'{e:.6f}' for e in row])
            f.write(row_str + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--affine", type=str, default='', help="affine matrix")
    parser.add_argument("--output_rigid", type=str, default='', help="output rigid transform matrix")
    parser.add_argument("--output_scaling_shearing", type=str, default='', help="output scaling and shearing transform matrix")
    args = parser.parse_args()
    
    main(args=args)