import nibabel as nib
import numpy as np
import argparse
from typing import Tuple
from scipy.optimize import minimize

def draw_close_intensity_v1(regis_mov: np.ndarray, trg: np.ndarray, orig_mov: np.ndarray) -> np.ndarray:
    """
    Draw close the grayscale of regis_mov to trg and then apply to orig_mov

    Parameters
    ----------
    regis_mov : np.ndarray
        movable image which is already reistered to target image
    trg : np.ndarray
        target image with a good contrast
    orig_mov : np.ndarray
        unregistered movable image
    
    Returns
    -------
    np.ndarray
    """

    def gamma_transform(image: np.ndarray, gamma: float, bias: float):
        transformed = np.clip(image - bias, 1e-3, None)
        return np.power(transformed / (1.0 - bias), gamma)

    def objective_function(x: np.ndarray[float], A: np.ndarray, B: np.ndarray, mask: np.ndarray):
        # Apply gamma transformation
        gamma, bias = x[0], x[1]
        A_transformed = gamma_transform(A, gamma, bias)
        # Compute Mean Squared Error (or another metric)
        print(np.mean((A_transformed - B) ** 2))
        print(gamma, bias)
        loss = (A_transformed - B) ** 2
        loss = np.mean(loss[mask])
        return loss

    # clip zero
    regis_mov_img = np.clip(regis_mov, 0, np.percentile(regis_mov, 99.5))
    trg_img = np.clip(trg, 0, np.percentile(trg, 99.9))

    # normalize
    regis_mov_img = (regis_mov_img - regis_mov_img.min()) / (regis_mov_img.max() - regis_mov_img.min())
    trg_img = (trg_img - trg_img.min()) / (trg_img.max() - trg_img.min())

    msk_img = trg_img > 1e-2

    # initial parameters
    initial_gamma = 1.0
    initial_bias = 0.0

    # optimize
    result = minimize(objective_function, np.array([initial_gamma, initial_bias]), args=(regis_mov_img, trg_img, msk_img), bounds=[(0.8, 4.0), [-0.30, 0.30]])

    # optimal parameters
    optimal_gamma = result.x[0]
    optimal_bias = result.x[1]
    print(f"Optimal gamma: {optimal_gamma} {optimal_bias}")

    # apply gamma transform
    corrected = gamma_transform(orig_mov, optimal_gamma, optimal_bias)
    corrected = np.clip(corrected, 0, np.percentile(corrected, 99.5))
    corrected = np.percentile(regis_mov, 99.5) * (corrected - corrected.min()) / (corrected.max() - corrected.min())
    
    return corrected

def draw_close_intensity_v2(regis_mov: np.ndarray, trg: np.ndarray, orig_mov: np.ndarray) -> np.ndarray:
    """
    Draw close the grayscale of regis_mov to trg and then apply to orig_mov

    Parameters
    ----------
    regis_mov : np.ndarray
        movable image which is already reistered to target image
    trg : np.ndarray
        target image with a good contrast
    orig_mov : np.ndarray
        unregistered movable image
    
    Returns
    -------
    np.ndarray
    """

    def gamma_transform(image: np.ndarray, gamma: float, alpha: float, bias: float):
        transformed = np.clip(image - bias, 1e-3, None)
        return alpha * np.power(transformed, gamma) + bias

    def objective_function(x: np.ndarray[float], A: np.ndarray, B: np.ndarray, mask: np.ndarray):
        # Apply gamma transformation
        gamma, alpha, bias = x[0], x[1], x[2]
        A_transformed = gamma_transform(A, gamma, alpha, bias)
        # Compute Mean Squared Error (or another metric)
        print(np.mean((A_transformed - B) ** 2))
        print(gamma, alpha, bias)
        loss = (A_transformed - B) ** 2
        loss = np.mean(loss[mask])
        return loss

    # clip zero
    regis_mov_img = np.clip(regis_mov, 0, np.percentile(regis_mov, 99.5))
    trg_img = np.clip(trg, 0, np.percentile(trg, 99.9))

    # normalize
    regis_mov_img = (regis_mov_img - regis_mov_img.min()) / (regis_mov_img.max() - regis_mov_img.min())
    trg_img = (trg_img - trg_img.min()) / (trg_img.max() - trg_img.min())

    msk_img = trg_img > 1e-2

    # initial parameters
    initial_gamma = 1.0
    initial_alpha = 1.0
    initial_bias = 0.0
    
    def constraint_function1(args):
        gamma, alpha, bias = args[0], args[1], args[2]
        # Ensure transformed image values do not exceed 1
        transformed_image = gamma_transform(regis_mov_img, gamma, alpha, bias)
        return 1.0 - np.max(transformed_image)
    
    def constraint_function2(args):
        gamma, alpha, bias = args[0], args[1], args[2]
        # Ensure transformed image values do not exceed 1
        transformed_image = gamma_transform(regis_mov_img, gamma, alpha, bias)
        return np.min(transformed_image)
    
    constraints = (
        {
            'type': 'ineq',  # >= 0
            'fun': constraint_function1
        },
        {
            'type': 'ineq',  # >= 0
            'fun': constraint_function2
        })

    # optimize
    result = minimize(objective_function, np.array([initial_gamma, initial_alpha, initial_bias]), args=(regis_mov_img, trg_img, msk_img), bounds=[(0.8, 4.0), (0.5, 2.0), [-0.20, 0.20]], constraints=constraints)

    # optimal parameters
    optimal_gamma = result.x[0]
    optimal_alpha = result.x[1]
    optimal_bias = result.x[2]
    print(f"Optimal gamma: {optimal_gamma} {optimal_alpha} {optimal_bias}")

    # apply gamma transform
    corrected = gamma_transform(orig_mov, optimal_gamma, optimal_alpha, optimal_bias)
    corrected = np.clip(corrected, 0, np.percentile(corrected, 99.5))
    corrected = np.percentile(regis_mov, 99.5) * (corrected - corrected.min()) / (corrected.max() - corrected.min())
    
    return corrected

def main(args):
    reg_mov_data = nib.load(args.reg_mov).get_fdata()
    reg_mov_aff = nib.load(args.reg_mov).affine
    trg_data = nib.load(args.target).get_fdata()
    trg_aff = nib.load(args.target).affine
    orig_mov_data = nib.load(args.orig_mov).get_fdata()
    orig_mov_aff = nib.load(args.orig_mov).affine

    # check valid
    if not np.isclose(reg_mov_aff, trg_aff).all():
        raise ValueError("movable and target offered maybe not in the same space")
    
    new_mov_data = draw_close_intensity_v1(reg_mov_data, trg_data, orig_mov_data)
    new_mov = nib.Nifti1Image(new_mov_data, orig_mov_aff)
    nib.save(new_mov, args.out_mov)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_mov", type=str, default='', help="movable image which is already reistered to target image")
    parser.add_argument("--target", type=str, default='', help="target image with a goog contrast")
    parser.add_argument("--orig_mov", type=str, default='', help="unregistered movable image")
    parser.add_argument("--out_mov", type=str, default='', help="intensity corrected movable image")
    args = parser.parse_args()
    
    main(args=args)