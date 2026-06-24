import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from utils.filter import GaussianFilter3d
align_corners = False

# ==========================
#   Base WarpField
# ==========================
class BaseWarpField(nn.Module):
    """
    Base class for 3D warp fields operating in normalized coordinates [-1, 1].

    Common features:
    - Stores batch size, spatial shape, number of dimensions, device.
    - Provides an identity sampling grid `base_grid` at the reference shape.
    - Provides helper methods to create identity grids at arbitrary shapes and
      to warp images with a given sampling grid.
    """

    def __init__(
        self,
        batch_size: int,
        shape: List[int],
        n_dims: int,
        device: torch.device,
    ):
        super().__init__()
        assert n_dims == 3, "Current implementation assumes 3D: [B, D, H, W, 3]."
        self.batch_size = batch_size
        self.shape = tuple(shape)  # reference (typically fixed image) shape
        self.n_dims = n_dims
        self.device = device

        # Identity grid at the reference shape in normalized coordinates [-1, 1]
        theta = torch.eye(n_dims, n_dims + 1, device=device).unsqueeze(0)   # [1,3,4]
        theta = theta.expand(batch_size, -1, -1)                            # [B,3,4]
        size = (batch_size, 1, *self.shape)                                 # [B,1,D,H,W]
        base_grid = F.affine_grid(theta, size=size, align_corners=align_corners)  # [B,D,H,W,3]
        self.register_buffer("base_grid", base_grid)

    # ------------------------------------------------------------------ #
    # Identity grids and image warping helpers
    # ------------------------------------------------------------------ #
    def _make_identity_grid(self, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Create an identity sampling grid (normalized [-1, 1]) for a given
        spatial shape.

        Args:
            spatial_shape: (D, H, W)

        Returns:
            grid: [B, D, H, W, 3]
        """
        theta = torch.eye(self.n_dims, self.n_dims + 1,
                          device=self.base_grid.device,
                          dtype=self.base_grid.dtype).unsqueeze(0)  # [1,3,4]
        theta = theta.expand(self.batch_size, -1, -1)             # [B,3,4]
        size = (self.batch_size, 1, *spatial_shape)               # [B,1,D,H,W]
        grid = F.affine_grid(theta, size=size, align_corners=align_corners)
        return grid

    @staticmethod
    def _warp_image(
        x: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> torch.Tensor:
        """
        Warp an image `x` with a given sampling grid.

        Args:
            x:    [B, C, D, H, W]
            grid: [B, D, H, W, 3]  normalized in [-1, 1]

        Returns:
            moved: [B, C, D, H, W]
        """
        moved = F.grid_sample(
            x,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return moved

    # ------------------------------------------------------------------ #
    # Interfaces to be implemented / extended by subclasses
    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        """
        Subclasses should override this method.

        Recommended contract:
            forward(...) -> (disp, warp)
        where
            disp: [B, D, H, W, 3], displacement field
            warp: [B, D, H, W, 3], sampling grid (phi(x))
        """
        raise NotImplementedError

    def get_inv(self, *args, **kwargs):
        """
        Optional method for subclasses to override, returning an approximate or
        exact inverse displacement and warp.

        Recommended contract:
            get_inv(out_shape: Optional[Tuple[int,int,int]] = None)
            -> (inv_disp, inv_warp)
        """
        raise NotImplementedError


# ==========================
#   Composite (scaling & squaring) module
# ==========================
class CompositeDiffeoWarpField(BaseWarpField):
    """
    Diffeomorphic warp field using scaling-and-squaring of a stationary
    velocity field:

      - Trainable parameter: velocity: [B, D, H, W, 3]
      - Compute exp(v) via scaling-and-squaring to get sampling grid phi
      - Theoretically diffeomorphic (in continuous setting), numerically approximate

    This class is now aligned with VelocityDiffeoWarpField:
      - forward() returns (disp, warp)
      - get_inv() computes the flow of -velocity via the same scheme
    """

    def __init__(
        self,
        batch_size: int,
        shape: List[int],
        n_dims: int,
        device: torch.device,
        init_velocity: Optional[torch.Tensor] = None,  # optional: init velocity from an existing displacement
        n_steps: int = 8,  # number of squaring steps, exp(v / 2^n_steps)
    ):
        super().__init__(batch_size=batch_size, shape=shape, n_dims=n_dims, device=device)
        self.n_steps = n_steps

        # Trainable stationary velocity field v(x)
        if init_velocity is None:
            v = torch.zeros(
                (batch_size, *self.shape, n_dims),
                dtype=torch.float32,
                device=device,
            )
        else:
            assert init_velocity.shape == (batch_size, *self.shape, n_dims), \
                f"init_velocity shape {init_velocity.shape} != {(batch_size, *self.shape, n_dims)}"
            v = init_velocity.to(device=device, dtype=torch.float32)

        self.velocity = nn.Parameter(v)  # [B, D, H, W, 3]

    def _exp_velocity(
        self,
        v: torch.Tensor,
        base_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaling-and-squaring of a stationary velocity field:

            v -> disp -> phi(x) = x + disp(x)

        Args:
            v:          [B, D, H, W, 3] stationary velocity
            base_grid:  identity grid where the flow is evaluated.
                        If None, uses self.base_grid (reference shape).

        Returns:
            disp: [B, D, H, W, 3] displacement field
            phi:  [B, D, H, W, 3] sampling grid = base_grid + disp
        """
        base = self.base_grid if base_grid is None else base_grid  # [B, D, H, W, 3]

        # Initial small-step displacement: v / 2^n_steps
        disp = v / (2.0 ** self.n_steps)  # [B, D, H, W, 3]

        for _ in range(self.n_steps):
            # disp encodes: x -> x + disp(x)
            # Compose g ∘ g: x -> x + disp(x) + disp(x + disp(x))
            disp_img = disp.permute(0, 4, 1, 2, 3)   # [B, 3, D, H, W]
            sampled = F.grid_sample(
                disp_img,
                base + disp,        # x + disp(x)
                mode="bilinear",
                padding_mode="border",
                align_corners=align_corners,
            )  # [B, 3, D, H, W]
            sampled = sampled.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3] = disp(x + disp(x))

            disp = disp + sampled

        phi = base + disp
        return disp, phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the forward displacement and warp (sampling grid) at the
        reference shape.

        Returns:
            disp: [B, D, H, W, 3]
            warp: [B, D, H, W, 3]
        """
        disp, warp = self._exp_velocity(self.velocity, self.base_grid)
        return disp, warp

    def get_inv(
        self,
        out_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse flow of the current velocity field by exponentiating
        -velocity via scaling-and-squaring.

        Args:
            out_shape: If None or equal to self.shape, use self.base_grid.
                       Otherwise, build an identity grid at out_shape to evaluate
                       the inverse flow.

        Returns:
            inv_disp: [B, D, H, W, 3]
            inv_warp: [B, D, H, W, 3]
        """
        if out_shape is None or tuple(out_shape) == self.shape:
            base = self.base_grid
        else:
            base = self._make_identity_grid(tuple(out_shape))

        inv_disp, inv_warp = self._exp_velocity(-self.velocity, base_grid=base)
        return inv_disp, inv_warp


# ==========================
#   Velocity (time integration) module
# ==========================
class VelocityDiffeoWarpField(BaseWarpField):
    """
    Diffeomorphic warp field via explicit time integration of a stationary
    velocity field:

      - Trainable parameter: velocity: [B, D, H, W, 3]
      - Use n_steps time steps of forward Euler:
            phi_0(x) = x
            phi_{k+1}(x) = phi_k(x) + dt * v(phi_k(x)),  dt = 1 / n_steps
      - In continuous theory, this corresponds to the flow of v, i.e., the
        solution of dφ/dt = v(φ), and φ_1 is a diffeomorphism.
      - This is a discrete approximation, conceptually similar to integrating
        the stationary velocity used in scaling-and-squaring.
    """

    def __init__(
        self,
        batch_size: int,
        shape: List[int],
        n_dims: int,
        device: torch.device,
        init_velocity: Optional[torch.Tensor] = None,  # optional: init velocity from displacement
        n_steps: int = 5,  # number of time steps
        smooth_sigma: float = 2e-1,
        smooth_kernel: int = 3,
    ):
        super().__init__(batch_size=batch_size, shape=shape, n_dims=n_dims, device=device)
        self.n_steps = n_steps
        self.gauss_filter = GaussianFilter3d(
            in_channels=n_dims,
            kernel_size=smooth_kernel,
            sigma=smooth_sigma
        ).to(device)

        # Trainable stationary velocity field v(x)
        if init_velocity is None:
            v = torch.zeros(
                (batch_size, *self.shape, n_dims),
                dtype=torch.float32,
                device=device,
            )
        else:
            assert init_velocity.shape == (batch_size, *self.shape, n_dims), \
                f"init_velocity shape {init_velocity.shape} != {(batch_size, *self.shape, n_dims)}"
            v = init_velocity.to(device=device, dtype=torch.float32)

        self.velocity = nn.Parameter(v)  # [B, D, H, W, 3]

    def _integrate_velocity(
        self,
        v: torch.Tensor,
        base_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Explicit time integration (forward Euler) of a stationary velocity field:

            phi_0(x) = base_grid
            for k in 0..n_steps-1:
                phi_{k+1}(x) = phi_k(x) + dt * v(phi_k(x))

        Args:
            v:         [B, Dv, Hv, Wv, 3] stationary velocity field
            base_grid: Identity grid where the flow is evaluated. If None, use
                       self.base_grid (reference shape).

        Returns:
            disp: [B, D, H, W, 3] displacement field (phi - base)
            phi:  [B, D, H, W, 3] sampling grid phi(x)
        """
        base = self.base_grid if base_grid is None else base_grid  # [B,D,H,W,3]
        v_img = v.permute(0, 4, 1, 2, 3)   # [B, 3, Dv, Hv, Wv]

        if torch.isinf(v_img).any() or torch.isnan(v_img).any():
            raise ValueError("Velocity field contains NaN or Inf!")
        if torch.max(v_img) > 2 ** self.n_steps:
            print(f"Warning: velocity magnitude exceeds 2 ** {self.n_steps} and I will clip it.")
            v_img = torch.clip(v_img, None, 2 ** self.n_steps)

        dt = 1.0 / float(self.n_steps)
        phi = base  # [B, D, H, W, 3]

        for _ in range(self.n_steps):
            v_phi = F.grid_sample(
                v_img,
                phi,
                mode="bilinear",
                padding_mode="border",
                align_corners=align_corners,
            )  # [B,3,D,H,W]
            v_phi = v_phi.permute(0, 2, 3, 4, 1)  # [B,D,H,W,3]

            phi = phi + dt * v_phi

        disp = phi - base
        # Optional: Gaussian smoothing on disp (currently disabled)
        # disp = self.gauss_filter(
        #     disp.permute(0, 4, 1, 2, 3)
        # ).permute(0, 2, 3, 4, 1).contiguous()
        # phi = base + disp

        return disp, phi

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the forward displacement and warp at the reference shape.

        Returns:
            disp: [B, D, H, W, 3]
            warp: [B, D, H, W, 3]
        """
        disp, warp = self._integrate_velocity(self.velocity, self.base_grid)
        return disp, warp

    def get_inv(
        self,
        out_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse flow by integrating the negative velocity field.

        Args:
            out_shape: If None or equal to self.shape, use self.base_grid.
                       Otherwise, build an identity grid at out_shape to evaluate
                       the inverse flow.

        Returns:
            inv_disp: [B, D, H, W, 3]
            inv_warp: [B, D, H, W, 3]
        """
        if out_shape is None or tuple(out_shape) == self.shape:
            base = self.base_grid
        else:
            base = self._make_identity_grid(tuple(out_shape))

        inv_disp, inv_warp = self._integrate_velocity(-self.velocity, base_grid=base)
        return inv_disp, inv_warp


# ==========================
#   Free-form (FFD-style) module
# ==========================
class FreeFormWarpField(BaseWarpField):
    """
    Free-form deformation (FFD-style) warp field:

      - Parameters are a low-resolution control-point displacement field:
            control: [B, d, h, w, 3]
      - Forward:
            1) Upsample control to full resolution -> disp(x)
            2) Construct phi(x) = x + disp(x)
      - This does not guarantee diffeomorphism, but is very flexible and
        captures large-scale deformations efficiently.

    The API is aligned with VelocityDiffeoWarpField:
      - forward()  -> (disp, warp)
      - get_inv()  -> approximate inverse using -disp (not strictly correct!)
    """

    def __init__(
        self,
        batch_size: int,
        shape: List[int],
        n_dims: int,
        device: torch.device,
        init_warp: Optional[torch.Tensor] = None,  # optional: init control points from an existing displacement
        downsample_factor: int = 4,                # roughly shape / factor for control grid
    ):
        super().__init__(batch_size=batch_size, shape=shape, n_dims=n_dims, device=device)

        # Heuristic control grid resolution: original / factor, at least 2
        ctrl_D = max(2, self.shape[0] // downsample_factor)
        ctrl_H = max(2, self.shape[1] // downsample_factor)
        ctrl_W = max(2, self.shape[2] // downsample_factor)
        self.ctrl_shape = (ctrl_D, ctrl_H, ctrl_W)

        if init_warp is None:
            control = torch.zeros(
                (batch_size, ctrl_D, ctrl_H, ctrl_W, n_dims),
                dtype=torch.float32,
                device=device,
            )
        else:
            # Initialize control points by downsampling an existing displacement
            assert init_warp.shape == (batch_size, *self.shape, n_dims), \
                f"init_warp shape {init_warp.shape} != {(batch_size, *self.shape, n_dims)}"
            init_img = init_warp.permute(0, 4, 1, 2, 3)  # [B,3,D,H,W]
            ctrl_img = F.interpolate(
                init_img,
                size=self.ctrl_shape,
                mode='trilinear',
                align_corners=align_corners,
            )  # [B,3,d,h,w]
            control = ctrl_img.permute(0, 2, 3, 4, 1)     # [B,d,h,w,3]

        self.control = nn.Parameter(control)  # [B, d, h, w, 3]

    def _upsample_to(
        self,
        spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Upsample control-point displacement field to a given full resolution.

        Args:
            spatial_shape: (D, H, W)

        Returns:
            disp: [B, D, H, W, 3]
        """
        ctrl = self.control                        # [B, d, h, w, 3]
        ctrl_img = ctrl.permute(0, 4, 1, 2, 3)     # [B, 3, d, h, w]

        full = F.interpolate(
            ctrl_img,
            size=spatial_shape,
            mode='trilinear',
            align_corners=align_corners,
        )  # [B,3,D,H,W]

        disp = full.permute(0, 2, 3, 4, 1)         # [B,D,H,W,3]
        return disp

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the forward displacement and warp at the reference shape.

        Returns:
            disp: [B, D, H, W, 3]
            warp: [B, D, H, W, 3]
        """
        disp = self._upsample_to(self.shape)      # [B,D,H,W,3]
        warp = self.base_grid + disp             # [B,D,H,W,3]
        return disp, warp

    def get_inv(
        self,
        out_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate inverse deformation by simply negating the displacement.

        NOTE:
            This is *not* a true inverse in the diffeomorphic sense. It is a
            first-order approximation, suitable for small deformations but can
            be inaccurate for large ones.

        Args:
            out_shape: If None or equal to self.shape, use self.base_grid.
                       Otherwise, build an identity grid at out_shape and
                       upsample control points to that shape.

        Returns:
            inv_disp: [B, D, H, W, 3]
            inv_warp: [B, D, H, W, 3]
        """
        if out_shape is None or tuple(out_shape) == self.shape:
            base = self.base_grid
            disp = self._upsample_to(self.shape)
        else:
            spatial_shape = tuple(out_shape)
            base = self._make_identity_grid(spatial_shape)
            disp = self._upsample_to(spatial_shape)

        inv_disp = -disp
        inv_warp = base + inv_disp
        return inv_disp, inv_warp
