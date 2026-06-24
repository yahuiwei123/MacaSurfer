"""
Apply an FSL affine matrix to a NIfTI header WITHOUT resampling the data.

This preserves the original voxel data matrix and only modifies the sform/qform
affine in the NIfTI header, so the image appears spatially aligned in ACPC space
while keeping the original voxel values intact.

FSL .mat files are stored in FSL's internal coordinate system (scaled-voxel
space), NOT in RAS-mm space.  To modify the NIfTI header correctly we must
first convert the FSL matrix to a RAS-mm transform.  This requires the
reference (target) image whose voxel sizes define the target-side scaling.
"""
import numpy as np
import nibabel as nib
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Apply FSL affine matrix to NIfTI header (no resampling)"
    )
    parser.add_argument("--input", required=True, help="Input NIfTI file")
    parser.add_argument("--matrix", required=True, help="FSL 4x4 affine matrix file")
    parser.add_argument("--output", required=True, help="Output NIfTI file")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference (target) NIfTI file, used to convert FSL matrix to RAS space",
    )
    args = parser.parse_args()

    M_fsl = np.loadtxt(args.matrix)
    img = nib.load(args.input)
    ref = nib.load(args.reference)

    orig_dtype = img.get_data_dtype()
    data = np.asanyarray(img.dataobj)

    mov_aff = img.affine
    ref_aff = ref.affine

    # Per-pipeline convention (see affine_fsl2niftyreg.py):
    #   aladin_aff = mov_aff @ S_mov^{-1} @ M_fsl^{-1} @ S_ref @ ref_aff^{-1}
    #   aladin_aff maps REF→MOV (NiftyReg convention).
    #
    # The inverse (MOV→REF) is:
    #   ras_mov2ref = ref_aff @ S_ref^{-1} @ M_fsl @ S_mov @ mov_aff^{-1}
    #
    # For header-only application:
    #   new_affine @ vox = ras_mov2ref @ mov_aff @ vox
    #                    = ref_aff @ S_ref^{-1} @ M_fsl @ S_mov @ vox
    #   =>  new_affine = ref_aff @ S_ref^{-1} @ M_fsl @ S_mov

    S_mov = np.linalg.norm(mov_aff[:3, :3], axis=0)
    S_mov = np.diag(np.append(S_mov, 1))
    S_ref = np.linalg.norm(ref_aff[:3, :3], axis=0)
    S_ref = np.diag(np.append(S_ref, 1))

    new_affine = ref_aff @ np.linalg.inv(S_ref) @ M_fsl @ S_mov

    new_img = nib.Nifti1Image(data.astype(orig_dtype), new_affine)
    new_img.set_sform(new_affine, code=2)
    new_img.set_qform(new_affine, code=2)

    nib.save(new_img, args.output)


if __name__ == "__main__":
    main()
