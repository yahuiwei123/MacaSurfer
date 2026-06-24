import numpy as np
import nibabel as nib
import argparse

def main(args):
    omat_path = args.omat
    mov_path = args.mov
    trg_path = args.trg
    
    mov = nib.load(mov_path)
    mov2ras = mov.affine
    ras2mov = np.linalg.inv(mov2ras)

    trg = nib.load(trg_path)
    trg2ras = trg.affine
    
    mov2trg = ras2mov @ trg2ras
    
    with open(omat_path, 'w') as f:
        for row in mov2trg:
            row_str = '  '.join([f'{e:.6f}' for e in row])
            f.write(row_str + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mov", type=str, default='', help="movable image")
    parser.add_argument("--trg", type=str, default='', help="target image")
    parser.add_argument("--omat", type=str, default='', help="output matrix")
    args = parser.parse_args()
    
    main(args=args)