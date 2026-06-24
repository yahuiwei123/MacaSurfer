from fireants.io.image import Image, BatchedImages, FakeBatchedImages
from fireants.registration.greedy import GreedyRegistration
import matplotlib.pyplot as plt
import SimpleITK as sitk
from time import time
import argparse
import numpy as np
from torch import tensor


def main(args):
    # load the images
    fixed_img = Image.load_file(args.fixed)
    moving_img = Image.load_file(args.moving)
    
    # batchify them (we only have a single image per batch, but we can pass multiple images)
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    # load initial affine matrix
    init_aff = np.loadtxt(args.init_affine) if args.init_affine else np.eye(4)
    init_aff = tensor(init_aff).float().unsqueeze(0).to(fixed_batch.device)
    
    # nonlinear
    reg = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 150, 100], 
                fixed_images=fixed_batch, moving_images=moving_batch,
                cc_kernel_size=5, deformation_type='compositive', 
                smooth_grad_sigma=1, 
                optimizer='Adam', optimizer_lr=0.15, 
                init_affine=init_aff)
    
    start = time()
    reg.optimize()
    end = time()
    moved_batch = reg.evaluate(fixed_batch, moving_batch)
    
    # save image
    if args.moved:
        moved_img = FakeBatchedImages(moved_batch, fixed_batch)
        moved_img.write_image(args.moved)
    
    # save warp
    reg.save_as_ants_transforms(args.warp)
    # reg.get_inverse_warped_coordinates(fixed_batch, moving_batch)
    # reg.save_as_ants_transforms(args.inv_warp)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, default=None, help="fixed image")
    parser.add_argument("--moving", type=str, default=None, help="moving image")
    parser.add_argument("--init_affine", type=str, default=None, help="initial affine transform")
    parser.add_argument("--moved", type=str, default=None, help="moved image")
    parser.add_argument("--warp", type=str, default=None, help="warp field")
    parser.add_argument("--inv_warp", type=str, default=None, help="inv warp field")
    args = parser.parse_args()
    
    main(args=args)