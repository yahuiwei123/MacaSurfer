import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os

class MonotonicBSpline(nn.Module):
    def __init__(self, num_control_points):
        super(MonotonicBSpline, self).__init__()
        # Initialize control points as learnable parameters
        self.control_points = nn.Parameter(torch.linspace(0, 1, num_control_points, device='cuda'))
        self.num_control_points = num_control_points
        self.degree = 3
        self.knots = self.create_knots(num_control_points).to('cuda')
        

    def create_knots(self, num_control_points):
        """ Create knots for B-spline """
        knots = [0] * self.degree + list(np.linspace(0, 1, num_control_points - self.degree + 1)) + [1] * self.degree
        return torch.tensor(knots, dtype=torch.float32)

    def forward(self, x):
        """ Apply monotonic B-spline transformation """
        # Ensure monotonicity by sorting the control points cumulatively
        control_points = torch.cumsum(torch.abs(self.control_points), dim=0)
        b_spline = torch.zeros_like(x, device='cuda')
        for i in range(self.num_control_points):
            b_spline += control_points[i] * self.basis_function(i, self.degree, x, self.knots)
        return b_spline

    def basis_function(self, i, degree, x, knots):
        """ Compute B-spline basis functions using Cox-de Boor recursion formula """
        if degree == 0:
            return ((knots[i] <= x) & (x < knots[i+1])).float()
        else:
            left = ((x - knots[i]) / (knots[i + degree] - knots[i] + 1e-8)) * self.basis_function(i, degree - 1, x, knots)
            right = ((knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1] + 1e-8)) * self.basis_function(i + 1, degree - 1, x, knots)
            return left + right

def basis_function_numpy(i, degree, x, knots):
    """ Compute B-spline basis functions in numpy for plotting """
    if degree == 0:
        return np.where((knots[i] <= x) & (x < knots[i+1]), 1.0, 0.0)
    else:
        left = ((x - knots[i]) / (knots[i + degree] - knots[i] + 1e-8)) * basis_function_numpy(i, degree - 1, x, knots)
        right = ((knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1] + 1e-8)) * basis_function_numpy(i + 1, degree - 1, x, knots)
        return left + right

def plot_bspline(control_points, knots, degree, epoch, outdir):
    """ Plot the B-spline curve """
    control_points = torch.cumsum(torch.abs(control_points), dim=0).cpu().detach().numpy()
    x = np.linspace(0, 1, 100)
    y = np.zeros_like(x)
    for i in range(len(control_points)):
        basis = basis_function_numpy(i, degree, x, knots)
        y += control_points[i] * basis
    plt.figure()
    plt.plot(x, y, label='B-spline Curve')
    plt.scatter(np.linspace(0, 1, len(control_points)), control_points, color='red', label='Control Points')
    plt.title(f'B-spline Curve at Epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(outdir, f'{epoch}_bspline.jpg'))
    

def train_bspline(model, original_data, target_data, mask_data, process_dir=None, epochs=400, lr=1e-2):
    """ Train the B-spline model to minimize intensity difference """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    original_tensor = torch.tensor(original_data, dtype=torch.float32, device='cuda')
    target_tensor = torch.tensor(target_data, dtype=torch.float32, device='cuda')
    mask_tensor = torch.tensor(mask_data, dtype=torch.bool, device='cuda')

    # Normalize original data
    original_tensor = (original_tensor - original_tensor.min()) / (original_tensor.max() - original_tensor.min())
    target_tensor = (target_tensor - target_tensor.min()) / (target_tensor.max() - target_tensor.min())

    for epoch in range(epochs):
        optimizer.zero_grad()
        corrected_data = model(original_tensor)
        loss = criterion(corrected_data[mask_tensor], target_tensor[mask_tensor])
        # loss = criterion(corrected_data, target_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            if process_dir:
                plot_bspline(model.control_points, model.knots.cpu().detach().numpy(), model.degree, epoch, process_dir)
    return model

def apply_bspline(model, original_data):
    """ Apply trained B-spline to original data """
    original_tensor = torch.tensor(original_data, dtype=torch.float32, device='cuda')
    original_tensor = (original_tensor - original_tensor.min()) / (original_tensor.max() - original_tensor.min())
    with torch.no_grad():
        corrected_data = model(original_tensor).cpu().numpy()
    # Rescale corrected data to original range
    corrected_data = corrected_data * (original_data.max() - original_data.min()) + original_data.min()
    return corrected_data

def save_corrected_image(data, affine, header, output_path):
    """ Save the corrected image to a NIfTI file """
    corrected_img = nib.Nifti1Image(data, affine, header)
    nib.save(corrected_img, output_path)

def main(args):
    reg_mov_data = nib.load(args.reg_mov).get_fdata()
    reg_mov_aff = nib.load(args.reg_mov).affine
    trg_data = nib.load(args.target).get_fdata()
    trg_aff = nib.load(args.target).affine
    mask_data = nib.load(args.mask).get_fdata()
    mask_aff =  nib.load(args.mask).affine
    orig_mov_data = nib.load(args.orig_mov).get_fdata()
    orig_mov_aff = nib.load(args.orig_mov).affine

    # check valid
    if not np.isclose(reg_mov_aff, trg_aff).all():
        raise ValueError("movable and target offered maybe not in the same space")
    
    # Define and train B-spline model
    num_control_points = int(args.knots) if args.knots else 16

    model = MonotonicBSpline(num_control_points).to('cuda')
    if args.process:
        model = train_bspline(model, reg_mov_data, trg_data, mask_data, args.process, epochs=100, lr=1e-1)
    else:
        model = train_bspline(model, reg_mov_data, trg_data, mask_data, epochs=100, lr=1e-1)

    # Apply trained B-spline to original data
    new_mov_data = apply_bspline(model, orig_mov_data)
    new_mov = nib.Nifti1Image(new_mov_data, orig_mov_aff)
    nib.save(new_mov, args.out_mov)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_mov", type=str, default='', help="movable image which is already reistered to target image")
    parser.add_argument("--target", type=str, default='', help="target image with a good contrast")
    parser.add_argument("--mask", type=str, default='', help="target image mask (only this region counts)")
    parser.add_argument("--orig_mov", type=str, default='', help="unregistered movable image")
    parser.add_argument("--out_mov", type=str, default='', help="intensity corrected movable image")
    parser.add_argument("--knots", type=str, default='', help="knots of B-spline")
    parser.add_argument("--process", type=str, default='', help="visually process directory")
    args = parser.parse_args()
    
    main(args=args)