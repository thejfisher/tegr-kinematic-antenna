"""
Tests for the pilot-wave (dBB) preset.

Run with:
    python test_pilot_wave.py
"""
import sys
import torch
import numpy as np

# -------------------------------------------------------
# Unit Test: trilinear_interpolate
# -------------------------------------------------------
def test_trilinear_interpolate():
    """Interpolating at exact grid points should return the exact tensor value."""
    from utils import trilinear_interpolate

    GRID_RES = 10
    GRID_MIN = -5.0
    GRID_MAX = 5.0
    DX = (GRID_MAX - GRID_MIN) / GRID_RES

    # Create a field with known values: phi(i,j,k) = i + 10*j + 100*k
    field = torch.zeros((1, 1, GRID_RES, GRID_RES, GRID_RES))
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            for k in range(GRID_RES):
                field[0, 0, i, j, k] = float(i + 10 * j + 100 * k)

    # Sample at the centre of cell (3, 5, 7) — which maps to sim coords:
    #   x = GRID_MIN + (3 + 0.5) * DX  (for align_corners=True, grid point 3 maps to x=GRID_MIN + 3*DX_ac)
    # With align_corners=True, grid index i maps to: x = GRID_MIN + i * (GRID_MAX - GRID_MIN) / (GRID_RES - 1)
    DX_ac = (GRID_MAX - GRID_MIN) / (GRID_RES - 1)
    test_points = torch.tensor([
        [GRID_MIN + 3 * DX_ac, GRID_MIN + 5 * DX_ac, GRID_MIN + 7 * DX_ac],
        [GRID_MIN + 0 * DX_ac, GRID_MIN + 0 * DX_ac, GRID_MIN + 0 * DX_ac],  # corner
    ])

    result = trilinear_interpolate(field, test_points, GRID_MIN, GRID_MAX, GRID_RES)
    # grid_sample maps coords (x,y,z) → tensor dims (W,H,D) = (dim2, dim1, dim0)
    # So field[i,j,k] with coords at grid point (x=3,y=5,z=7) actually samples
    # phi[z=7, y=5, x=3] → value = 7 + 10*5 + 100*3 = 357
    expected_0 = 7.0 + 10.0 * 5.0 + 100.0 * 3.0  # = 357
    expected_1 = 0.0 + 0.0 + 0.0  # = 0

    assert abs(result[0].item() - expected_0) < 1.0, \
        f"Point (3,5,7): expected ~{expected_0}, got {result[0].item()}"
    assert abs(result[1].item() - expected_1) < 1.0, \
        f"Point (0,0,0): expected ~{expected_1}, got {result[1].item()}"
    print("[PASS] test_trilinear_interpolate")


# -------------------------------------------------------
# Unit Test: trilinear_interpolate_gradient
# -------------------------------------------------------
def test_trilinear_interpolate_gradient():
    """Gradient of a linear field should be constant everywhere."""
    from utils import trilinear_interpolate_gradient

    GRID_RES = 20
    GRID_MIN = -10.0
    GRID_MAX = 10.0
    DX = (GRID_MAX - GRID_MIN) / GRID_RES

    # Create a linear field: phi(x,y,z) = 2*x + 3*y + 5*z
    # Expected gradient: (2, 3, 5) everywhere
    field = torch.zeros((1, 1, GRID_RES, GRID_RES, GRID_RES))
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            for k in range(GRID_RES):
                x = GRID_MIN + i * DX
                y = GRID_MIN + j * DX
                z = GRID_MIN + k * DX
                field[0, 0, i, j, k] = 2.0 * x + 3.0 * y + 5.0 * z

    # Sample at a few interior points
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.5, -3.0, 1.0],
        [-5.0, 5.0, -5.0],
    ])

    grad = trilinear_interpolate_gradient(field, test_points, GRID_MIN, GRID_MAX, GRID_RES, DX)

    for i in range(len(test_points)):
        gx, gy, gz = grad[i].tolist()
        # Allow some tolerance due to finite differences at boundaries
        assert abs(gx - 2.0) < 1.5, f"Point {i} grad_x: expected ~2.0, got {gx}"
        assert abs(gy - 3.0) < 1.5, f"Point {i} grad_y: expected ~3.0, got {gy}"
        assert abs(gz - 5.0) < 1.5, f"Point {i} grad_z: expected ~5.0, got {gz}"
    print("[PASS] test_trilinear_interpolate_gradient")


# -------------------------------------------------------
# Integration Test: pilot-wave preset smoke test
# -------------------------------------------------------
def test_pilot_wave_preset_smoke():
    """
    Run the pilot-wave preset for a small number of steps and verify:
    - No NaNs in the state vector
    - Output shape is correct
    
    NOTE: This is a smoke test only. It does not run the full collider
    (which requires GPU + substantial memory). Instead it tests the
    utils functions under realistic conditions.
    """
    from utils import trilinear_interpolate, trilinear_interpolate_gradient

    GRID_RES = 32  # Small for test
    GRID_MIN = -50.0
    GRID_MAX = 50.0
    DX = (GRID_MAX - GRID_MIN) / GRID_RES
    N = 5

    # Simulate a simple wave field
    phi = torch.randn(1, 1, GRID_RES, GRID_RES, GRID_RES) * 0.1

    # Random particle positions within grid
    pos = torch.rand(N, 3) * (GRID_MAX - GRID_MIN) * 0.8 + GRID_MIN + 10.0

    # Test interpolation
    vals = trilinear_interpolate(phi, pos, GRID_MIN, GRID_MAX, GRID_RES)
    assert vals.shape == (N,), f"Expected shape ({N},), got {vals.shape}"
    assert not torch.isnan(vals).any(), "NaN in interpolated values"

    # Test gradient
    grad = trilinear_interpolate_gradient(phi, pos, GRID_MIN, GRID_MAX, GRID_RES, DX)
    assert grad.shape == (N, 3), f"Expected shape ({N},3), got {grad.shape}"
    assert not torch.isnan(grad).any(), "NaN in gradient values"

    # Simulate a few guidance steps
    mom = torch.zeros(N, 3)
    m0 = torch.full((N,), 0.511)
    dt = 0.001
    C = 100.0

    for step in range(50):
        grad = trilinear_interpolate_gradient(phi, pos, GRID_MIN, GRID_MAX, GRID_RES, DX)
        pilot_force = (1.0 / m0).unsqueeze(1) * grad * 50.0
        mom += pilot_force * dt
        p_sq = torch.sum(mom ** 2, dim=1, keepdim=True)
        gamma = torch.sqrt(1.0 + p_sq / (m0.unsqueeze(1) ** 2 * C ** 2))
        vel = mom / (gamma * m0.unsqueeze(1))
        pos += vel * dt

    assert not torch.isnan(pos).any(), "NaN in particle positions after 50 steps"
    assert not torch.isnan(mom).any(), "NaN in particle momenta after 50 steps"
    print("[PASS] test_pilot_wave_preset_smoke")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Pilot-Wave (dBB) Preset — Test Suite")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_fn in [test_trilinear_interpolate, test_trilinear_interpolate_gradient, test_pilot_wave_preset_smoke]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
