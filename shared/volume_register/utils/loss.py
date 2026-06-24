import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class MultiLossWrapper(nn.Module):
    """
    Wrap multiple loss functions that share the same inputs into a single loss.

    - Each loss function is called with the same *args and **kwargs.
    - The final loss is a weighted sum of all individual losses.
    - You can optionally get the individual loss values for logging.

    Example:
        loss_dict = {
            "ce": nn.CrossEntropyLoss(),
            "dice": DiceLoss()
        }
        weights = {"ce": 1.0, "dice": 0.5}

        criterion = MultiLossWrapper(loss_dict, weights)

        total_loss, components = criterion(pred, target, return_components=True)
    """

    def __init__(
        self,
        loss_fns: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        strict_scalar: bool = True,
    ):
        """
        Args:
            loss_fns:
                A dict of loss modules, e.g. {"img": img_loss_fn, "reg": reg_loss_fn}.
                All loss functions must accept the same *args, **kwargs in forward().
            weights:
                Optional dict of scalar weights for each loss key.
                If None, all weights default to 1.0.
                If some keys are missing, they also default to 1.0.
            strict_scalar:
                If True, will assert that each loss returns a scalar tensor (0-dim).
        """
        super().__init__()
        self.loss_fns = nn.ModuleDict(loss_fns)
        self.weights = weights or {}
        self.strict_scalar = strict_scalar

    def forward(
        self,
        *args,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        Run all wrapped losses with the same inputs and return a weighted sum.

        Args:
            *args, **kwargs:
                Passed to every loss function in self.loss_fns.
            return_components:
                If True, also return a dict of per-loss values:
                    total_loss, {"name": loss_value, ...}
                If False, return only total_loss.

        Returns:
            total_loss (Tensor) or
            (total_loss, components_dict)
        """
        total_loss = 0.0
        components = {}

        for name, fn in self.loss_fns.items():
            weight = float(self.weights.get(name, 1.0))
            if weight == 0.0:
                loss_val = 0.0
                weighted = 0.0
            else:
                loss_val = fn(*args, **kwargs)
                if self.strict_scalar and loss_val.ndim != 0:
                    raise ValueError(
                        f"Loss '{name}' must return a scalar tensor, "
                        f"but got shape {tuple(loss_val.shape)}."
                    )
                weighted = weight * loss_val

            components[name] = loss_val
            total_loss = total_loss + weighted

        if return_components:
            return total_loss, components
        return total_loss


import torch
import torch.nn as nn


class JacobianDeterminantLoss(nn.Module):
    """
    Jacobian determinant regularization loss to discourage folding
    (i.e., negative or too small determinant) in a 3D displacement field.

    Expected input:
        flow: [B, 3, D, H, W]
            Displacement field u(x), in the same coordinate order as your grid:
            channels correspond to (z, y, x) or (x, y, z). As long as you're
            consistent, the determinant sign is meaningful.

    We construct the deformation as:
        phi(x) = x + u(x)
    and approximate spatial derivatives using central finite differences
    on the interior region (excluding 1-voxel boundary).

    The loss penalizes det(J_phi) smaller than `jacobian_min`, especially
    negative determinants (folding).
    """

    def __init__(
        self,
        jacobian_min: float = 0.0,
        squared: bool = True,
        reduction: str = "mean",
    ):
        """
        Args:
            jacobian_min:
                Minimal allowed determinant. Values below this will be penalized.
                Typically 0.0 (discourage det <= 0). You can set e.g. 0.3 to also
                discourage strong local compression.
            squared:
                If True, use squared penalty: (relu(jacobian_min - detJ))^2.
                If False, use linear penalty: relu(jacobian_min - detJ).
            reduction:
                'mean', 'sum' or 'none' over all voxels and batch.
        """
        super().__init__()
        self.jacobian_min = jacobian_min
        self.squared = squared
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: displacement field u, shape [B, 3, D, H, W]

        Returns:
            loss: scalar tensor (if reduction != 'none') or per-voxel tensor.
        """
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError(
                f"`flow` must be of shape [B, 3, D, H, W], got {tuple(flow.shape)}"
            )

        B, C, D, H, W = flow.shape
        if D < 3 or H < 3 or W < 3:
            raise ValueError(
                f"Spatial size too small for central differences: D={D}, H={H}, W={W}"
            )

        # We'll compute central differences on the interior region
        # indices: z in [1, D-2], y in [1, H-2], x in [1, W-2]
        # so interior shape is [D-2, H-2, W-2]
        u = flow  # [B, 3, D, H, W]

        # For convenience, assume channel order = (z, y, x)
        # If your field is (x, y, z), the determinant is still meaningful
        # as long as you're consistent across code.
        u_z = u[:, 0]  # [B, D, H, W]
        u_y = u[:, 1]
        u_x = u[:, 2]

        # Central finite differences:
        # du/dz at interior voxel (z,y,x):
        #   (u[z+1,y,x] - u[z-1,y,x]) / 2
        # Shapes will all be [B, D-2, H-2, W-2] after consistent slicing.

        # d(u_z)/dz, d(u_z)/dy, d(u_z)/dx
        duz_dz = (u_z[:, 2:, 1:-1, 1:-1] - u_z[:, :-2, 1:-1, 1:-1]) / 2.0
        duz_dy = (u_z[:, 1:-1, 2:, 1:-1] - u_z[:, 1:-1, :-2, 1:-1]) / 2.0
        duz_dx = (u_z[:, 1:-1, 1:-1, 2:] - u_z[:, 1:-1, 1:-1, :-2]) / 2.0

        # d(u_y)/dz, d(u_y)/dy, d(u_y)/dx
        duy_dz = (u_y[:, 2:, 1:-1, 1:-1] - u_y[:, :-2, 1:-1, 1:-1]) / 2.0
        duy_dy = (u_y[:, 1:-1, 2:, 1:-1] - u_y[:, 1:-1, :-2, 1:-1]) / 2.0
        duy_dx = (u_y[:, 1:-1, 1:-1, 2:] - u_y[:, 1:-1, 1:-1, :-2]) / 2.0

        # d(u_x)/dz, d(u_x)/dy, d(u_x)/dx
        dux_dz = (u_x[:, 2:, 1:-1, 1:-1] - u_x[:, :-2, 1:-1, 1:-1]) / 2.0
        dux_dy = (u_x[:, 1:-1, 2:, 1:-1] - u_x[:, 1:-1, :-2, 1:-1]) / 2.0
        dux_dx = (u_x[:, 1:-1, 1:-1, 2:] - u_x[:, 1:-1, 1:-1, :-2]) / 2.0

        # Build Jacobian J_phi(x) = I + grad(u)
        # For each voxel we have a 3x3 matrix:
        #   J = [[1+duz_dz, duz_dy,   duz_dx  ],
        #        [duy_dz,   1+duy_dy, duy_dx  ],
        #        [dux_dz,   dux_dy,   1+dux_dx]]

        a11 = 1.0 + duz_dz
        a12 = duz_dy
        a13 = duz_dx

        a21 = duy_dz
        a22 = 1.0 + duy_dy
        a23 = duy_dx

        a31 = dux_dz
        a32 = dux_dy
        a33 = 1.0 + dux_dx

        # Determinant of 3x3
        detJ = (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )  # [B, D-2, H-2, W-2]

        # Penalize detJ < jacobian_min
        # folding_penalty = relu(jacobian_min - detJ)
        penalty = torch.relu(self.jacobian_min - detJ)

        if self.squared:
            penalty = penalty ** 2

        if self.reduction == "mean":
            return penalty.mean()
        elif self.reduction == "sum":
            return penalty.sum()
        else:  # 'none'
            return penalty
