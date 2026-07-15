"""
Teleparallel Simulation Utilities
=================================
Interpolation functions for bridging continuous particle coordinates
to the discrete FDTD torsion grid.

Used primarily by the pilot-wave (dBB) preset where particles are guided
by the spatial gradient of the Eulerian field at their exact (non-grid-
aligned) positions.
"""
import torch
import torch.nn.functional as F


def trilinear_interpolate(field, coords, grid_min, grid_max, grid_res):
    """
    Interpolate a 3-D scalar field at arbitrary continuous coordinates
    using PyTorch's grid_sample (trilinear mode).

    Parameters
    ----------
    field : torch.Tensor
        Shape ``(1, 1, D, H, W)`` – the Eulerian torsion grid.
    coords : torch.Tensor
        Shape ``(N, 3)`` – particle positions in *simulation* space
        (same units as ``grid_min`` / ``grid_max``).
    grid_min : float
        Lower bound of the grid in each axis.
    grid_max : float
        Upper bound of the grid in each axis.
    grid_res : int
        Number of grid cells per axis.

    Returns
    -------
    torch.Tensor
        Shape ``(N,)`` – interpolated scalar values at each coordinate.

    Notes
    -----
    ``grid_sample`` expects coordinates normalised to ``[-1, +1]``.
    Points outside the grid are clamped (``padding_mode='border'``),
    which matches the user's requested "clamp at boundary" behaviour.
    """
    # Normalise simulation coords → [-1, 1] for grid_sample
    norm = 2.0 * (coords - grid_min) / (grid_max - grid_min) - 1.0  # (N, 3)

    # grid_sample needs shape (1, 1, 1, N, 3) with (x→W, y→H, z→D) ordering.
    # Our convention: coords[:, 0]=x, coords[:, 1]=y, coords[:, 2]=z
    # grid_sample dim order is (x, y, z) already when input is (D, H, W)
    # and grid last dim is (W_idx, H_idx, D_idx), i.e. (x, y, z).
    grid = norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)

    sampled = F.grid_sample(
        field,
        grid,
        mode='bilinear',          # trilinear in 3-D
        padding_mode='border',    # clamp at walls
        align_corners=True,
    )
    # sampled shape: (1, 1, 1, 1, N)
    return sampled.squeeze()  # (N,)


def trilinear_interpolate_gradient(field, coords, grid_min, grid_max, grid_res, dx):
    """
    Compute the spatial gradient of *field* at arbitrary particle positions
    using central finite differences on the grid followed by trilinear
    interpolation.

    Parameters
    ----------
    field : torch.Tensor
        Shape ``(1, 1, D, H, W)``.
    coords : torch.Tensor
        Shape ``(N, 3)``.
    grid_min, grid_max : float
    grid_res : int
    dx : float
        Grid spacing.

    Returns
    -------
    torch.Tensor
        Shape ``(N, 3)`` – gradient vector (∂φ/∂x, ∂φ/∂y, ∂φ/∂z) at each
        particle position.
    """
    phi = field[0, 0]  # (D, H, W)

    # Central differences along each axis
    # X (dim 0): ∂φ/∂x
    grad_x = torch.zeros_like(phi)
    grad_x[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
    # Y (dim 1): ∂φ/∂y
    grad_y = torch.zeros_like(phi)
    grad_y[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dx)
    # Z (dim 2): ∂φ/∂z
    grad_z = torch.zeros_like(phi)
    grad_z[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dx)

    # Pack into (1, 3, D, H, W) for a single grid_sample call
    grad_field = torch.stack([grad_x, grad_y, grad_z], dim=0).unsqueeze(0)  # (1, 3, D, H, W)

    # Normalise coords to [-1, 1]
    norm = 2.0 * (coords - grid_min) / (grid_max - grid_min) - 1.0  # (N, 3)
    grid = norm.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, 3)

    sampled = F.grid_sample(
        grad_field,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return sampled[0, :, 0, 0, :].T  # (N, 3)


def cubic_interpolate(field, coords, grid_min, grid_max, grid_res):
    """
    Placeholder for future cubic (Catmull-Rom) interpolation.

    Falls back to trilinear for now.
    """
    # TODO: Implement cubic Catmull-Rom via manual kernel convolution
    return trilinear_interpolate(field, coords, grid_min, grid_max, grid_res)
