import argparse
import numpy as np
import nibabel as nib

def main(args):
    mov = args.mov
    trg = args.trg
    
    fsl_file = args.fsl_mat
    aladin_file = args.aladin_mat
    
    trg_aff = nib.load(trg).header.get_sform()
    mov_aff = nib.load(mov).header.get_sform()
    
    aladin_aff = np.loadtxt(aladin_file)
    
    S_mov = np.linalg.norm(mov_aff[:3, :3], axis=0)
    S_mov = np.diag(np.append(S_mov, 1))
    S_trg = np.linalg.norm(trg_aff[:3, :3], axis=0)
    S_trg = np.diag(np.append(S_trg, 1))
    
    tmp = np.linalg.inv(S_mov @ np.linalg.inv(mov_aff) @ aladin_aff @ trg_aff @ np.linalg.inv(S_trg))

    fsl_aff = tmp
    
    np.savetxt(fsl_file, fsl_aff)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--aladin_mat", type=str, default='', help="ras transform mat to be converted")
    parser.add_argument("--fsl_mat", type=str, default='', help="fsl transform mat to be generated")
    parser.add_argument("--mov", type=str, default='', help="movable image")
    parser.add_argument("--trg", type=str, default='', help="target image")
    args = parser.parse_args()
    
    main(args=args)
