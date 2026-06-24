import shutil
import argparse

from nitransforms.io.fsl import FSLLinearTransformArray
from nitransforms.io.itk import ITKLinearTransform

def main(args):
    mov = args.mov
    trg = args.trg
    
    fsl_file = args.fsl_mat
    ras_file = args.ras_mat
    
    fsl = FSLLinearTransformArray.from_filename(fsl_file)
    transform_matrix = fsl.to_ras(moving=mov, reference=trg)
    transform_matrix = transform_matrix.squeeze(0)
    itk = ITKLinearTransform.from_ras(transform_matrix, moving=mov, reference=trg)
    
    if ras_file != fsl_file:
        itk.to_filename(ras_file)
    else:
        shutil.move(fsl_file, fsl_file + '.fsl')
        itk.to_filename(ras_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsl_mat", type=str, default='', help="fsl transform mat to be converted")
    parser.add_argument("--ras_mat", type=str, default='', help="ras transform mat to be generated")
    parser.add_argument("--mov", type=str, default='', help="movable image")
    parser.add_argument("--trg", type=str, default='', help="target image")
    args = parser.parse_args()
    
    main(args=args)
