import argparse
import shutil

from nitransforms.io.fsl import FSLLinearTransformArray
from nitransforms.io.itk import ITKLinearTransform

def main(args):
    mov = args.mov
    trg = args.trg
    
    fsl_file = args.fsl_mat
    ras_file = args.ras_mat
    
    itk = ITKLinearTransform.from_filename(ras_file)
    transform_matrix = itk.to_ras(moving=mov, reference=trg)

    transform_matrix = transform_matrix[None, ...]
    fsl = FSLLinearTransformArray.from_ras(transform_matrix, moving=mov, reference=trg)
    
    if fsl_file != ras_file:
        fsl.to_filename(fsl_file)
    else:
        shutil.move(ras_file, ras_file + '.ras')
        fsl.to_filename(fsl_file)
    shutil.move(ras_file + '.000', ras_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ras_mat", type=str, default='', help="ras transform mat to be converted")
    parser.add_argument("--fsl_mat", type=str, default='', help="fsl transform mat to be generated")
    parser.add_argument("--mov", type=str, default='', help="movable image")
    parser.add_argument("--trg", type=str, default='', help="target image")
    args = parser.parse_args()
    
    main(args=args)
