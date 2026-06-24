#!/usr/bin/env python3

import os
import argparse
import nibabel as nib
import sys
sys.path.append("/home/yhwei/projects/Monkey_Surface/best_maca_v3/shared/brainextractor/")
from main import BrainExtractor


def main():
    # create command line parser
    parser = argparse.ArgumentParser(
        description="A Reimplementation of FSL's Brain Extraction Tool",
        epilog="Author: Andrew Van, vanandrew@wustl.edu, 12/15/2020",
    )
    parser.add_argument("--input_img", type=str, default=None, help="Input image to brain extract")
    parser.add_argument("--init_mask", type=str, default=None, help="Input mask to intialize brainmask")
    parser.add_argument("--output_img", type=str, default=None, help="Output image to write out")
    parser.add_argument("-w", "--write_surface_deform", help="Path to write out surface files at each deformation step")
    parser.add_argument(
        "-f",
        "--fractional_threshold",
        type=float,
        default=0.5,
        help="Main threshold parameter for controlling brain/background (Default: 0.5)",
    )
    parser.add_argument(
        "-n", "--iterations", type=int, default=250, help="Number of iterations to run (Default: 1000)"
    )
    parser.add_argument(
        "-t",
        "--histogram_threshold",
        nargs=2,
        type=float,
        default=[0.02, 0.98],
        help="Sets min/max of histogram (Default: 0.02, 0.98)",
    )
    parser.add_argument(
        "-d",
        "--search_distance",
        nargs=2,
        type=float,
        default=[10.0, 5.0],
        help="Sets search distance for max/min of image along vertex normals (Default: 20.0, 10.0)",
    )
    parser.add_argument(
        "-r",
        "--radius_of_curvatures",
        nargs=2,
        type=float,
        default=[3.33, 10.0],
        help="Sets min/max radius of curvature for surface (Default: 3.33, 10.0)",
    )

    # parse arguments
    args = parser.parse_args()

    # load input image
    input_img = os.path.abspath(args.input_img)
    img = nib.load(input_img)

    # load input mask
    if args.init_mask:
        input_msk = os.path.abspath(args.init_mask)
        msk = nib.load(input_msk)
    else:
        msk = None

    # create brain extractor
    bet = BrainExtractor(
        img=img,
        init_mask=msk, 
        t02t=args.histogram_threshold[0],
        t98t=args.histogram_threshold[1],
        bt=args.fractional_threshold,
        d1=args.search_distance[0],
        d2=args.search_distance[1],
        rmin=args.radius_of_curvatures[0],
        rmax=args.radius_of_curvatures[1],
    )

    # create output path for surface files if defined
    if args.write_surface_deform:
        deformation_path = os.path.abspath(args.write_surface_deform)
        try:
            os.remove(deformation_path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(deformation_path), exist_ok=True)

    # run brain extractor
    bet.run(iterations=args.iterations, deformation_path=deformation_path if args.write_surface_deform else None)

    # make dirs to output directory as needed
    output_img = os.path.abspath(args.output_img)
    os.makedirs(os.path.dirname(output_img), exist_ok=True)

    # write mask to file
    print("Saving mask...")
    bet.save_mask(output_img)
    print("Mask saved.")


if __name__ == '__main__':
    main()