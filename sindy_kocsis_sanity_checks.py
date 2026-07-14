"""
sindy_kocsis_sanity_checks.py
=============================
Runs three critical sanity checks against the original SINDy result
to determine whether the theta - sin(theta) structure is real or a
mathematical mirage caused by collinearity, unit artifacts, and
hyperparameter sensitivity.

Test 1 (Null Test):     Polynomial-only library (no trig). Does R² match?
Test 2 (Unit Fix):      Normalize x to dimensionless x/L. Does sin() survive?
Test 3 (Threshold Sweep): Vary STLSQ threshold. When does sin(x) vanish?
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.interpolate import interp1d
import json

# ---------------------------------------------------------------------------
# Reuse the data loader from the main script
# ---------------------------------------------------------------------------
def load_and_reconstruct_trajectories(data_dir):
    pixelpitch = 0.026  # mm
    meas_coeff = 1.0 / 373.0
    num_files = 91

    H_arrays, V_arrays = [], []
    for i in range(num_files):
        filename = os.path.join(data_dir, f"Image{i}.txt")
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as f:
            content = f.read().strip()
            if not content:
                continue
            raw_data = np.array([float(x) for x in content.split(',') if x.strip()])
            if len(raw_data) < 2048:
                continue
            H_arrays.append(raw_data[0:1024])
            V_arrays.append(raw_data[1024:2048])

    H_arrays = np.array(H_arrays)
    V_arrays = np.array(V_arrays)

    bg_indices = np.concatenate((np.arange(399, 475), np.arange(749, 900)))
    roi_indices = np.arange(474, 725)
    xvals = pixelpitch * np.arange(1, 1025)
    x_roi = xvals[roi_indices] - np.mean(xvals[roi_indices])

    z_vals = np.linspace(2500, 8500, len(H_arrays))
    init_plane, end_plane = 10, min(50, len(H_arrays))
    z_range = z_vals[init_plane:end_plane]

    start_x = np.linspace(-1.5, 1.5, 20)
    current_x = start_x.copy()
    trajectory_points = [current_x.copy()]

    for i in range(init_plane, end_plane - 1):
        h = H_arrays[i][roi_indices]
        v = V_arrays[i][roi_indices]
        bg_h = np.mean(H_arrays[i][bg_indices])
        bg_v = np.mean(V_arrays[i][bg_indices])
        h_sub = np.maximum(h - bg_h, 1e-11)
        v_sub = np.maximum(v - bg_v, 1e-11)
        prob_h = h_sub / np.sum(h_sub)
        prob_v = v_sub / np.sum(v_sub)
        rmimag = np.clip(meas_coeff * (prob_v - prob_h) / (prob_h + prob_v), -0.5, 0.5)
        kx_k_weak = np.tan(np.arcsin(rmimag))
        f_kx = interp1d(x_roi, kx_k_weak, kind='cubic', fill_value="extrapolate")
        dz = z_vals[i+1] - z_vals[i]
        current_x = current_x + dz * f_kx(current_x)
        trajectory_points.append(current_x.copy())

    return z_range, np.array(trajectory_points)


def run_sindy(z_range, trajectories, feature_library, label, threshold=1e-4, alpha=0.01):
    """Run SINDy with a given library and return the model, equations, and score."""
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    x_train = [trajectories[:, i].reshape(-1, 1) for i in range(trajectories.shape[1])]
    model.fit(x_train, t=z_range)

    eqs = model.equations()
    score = float(model.score(x_train, t=z_range))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    model.print()
    print(f"  R² = {score:.6f}")
    print(f"{'='*70}")

    return {"label": label, "equations": list(eqs), "score": score,
            "features": model.get_feature_names()}


def main():
    data_dir = r"Z:\teleparallel_sim_photons\Kocsis_Data\OnlineArchive\Data\45-90 equiv both 0.05 15 sec\pics"
    out_dir = r"Z:\teleparallel_sim_photons\sessionsPapers"
    os.makedirs(out_dir, exist_ok=True)

    z_range, trajectories = load_and_reconstruct_trajectories(data_dir)
    print(f"Loaded trajectories: {trajectories.shape[0]} time steps x {trajectories.shape[1]} paths")
    print(f"x range: [{trajectories.min():.4f}, {trajectories.max():.4f}] mm")

    results = []

    # ==================================================================
    # TEST 1: NULL TEST — Polynomial-only (no trig, no 1/x)
    # If R² matches ~0.19, the sin(x) finding was a mirage.
    # ==================================================================
    print("\n" + "#"*70)
    print("# TEST 1: NULL TEST — Polynomial-only library")
    print("#"*70)

    poly_only = ps.PolynomialLibrary(degree=3)
    r = run_sindy(z_range, trajectories, poly_only, "TEST 1: Poly-only (degree 3)")
    results.append(r)

    # Also try degree 5 to give it more room
    poly5 = ps.PolynomialLibrary(degree=5)
    r = run_sindy(z_range, trajectories, poly5, "TEST 1b: Poly-only (degree 5)")
    results.append(r)

    # ==================================================================
    # TEST 2: DIMENSIONLESS UNITS — Normalize x by slit separation
    # The Kocsis experiment used slits separated by ~4.69 mm (from their
    # MATLAB code: rightmean - leftmean ≈ 19.12 - 14.44 = 4.68 mm).
    # We normalize x -> x/L so sin(x/L) is dimensionless.
    # ==================================================================
    print("\n" + "#"*70)
    print("# TEST 2: DIMENSIONLESS — x normalized by slit separation L")
    print("#"*70)

    L_slit = 4.68  # mm, slit separation from the MATLAB analysis
    traj_normed = trajectories / L_slit
    z_normed = z_range / L_slit  # Also normalize z for consistency

    print(f"  After normalization: x range [{traj_normed.min():.6f}, {traj_normed.max():.6f}]")

    # 2a: Poly-only on dimensionless data
    r = run_sindy(z_normed, traj_normed, ps.PolynomialLibrary(degree=3),
                  "TEST 2a: Dimensionless, Poly-only")
    results.append(r)

    # 2b: Full library (poly + trig + 1/x) on dimensionless data
    lib_funcs = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: 1.0/(x+1e-5)]
    lib_names = [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"1/({x})"]
    custom = ps.CustomLibrary(library_functions=lib_funcs, function_names=lib_names)
    full_lib = ps.PolynomialLibrary(degree=3) + custom
    r = run_sindy(z_normed, traj_normed, full_lib,
                  "TEST 2b: Dimensionless, Full library (poly+trig+1/x)")
    results.append(r)

    # ==================================================================
    # TEST 3: THRESHOLD SWEEP — At what threshold does sin(x) vanish?
    # Uses original (mm) data with the full library.
    # ==================================================================
    print("\n" + "#"*70)
    print("# TEST 3: THRESHOLD SWEEP — Varying STLSQ sparsity threshold")
    print("#"*70)

    full_lib_mm = ps.PolynomialLibrary(degree=3) + ps.CustomLibrary(
        library_functions=[lambda x: np.sin(x), lambda x: np.cos(x), lambda x: 1.0/(x+1e-5)],
        function_names=[lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"1/({x})"]
    )

    thresholds = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    sweep_results = []
    for thr in thresholds:
        r = run_sindy(z_range, trajectories, full_lib_mm,
                      f"Threshold = {thr:.0e}", threshold=thr)
        sweep_results.append(r)
        results.append(r)

    # ==================================================================
    # COLLINEARITY DIAGNOSTIC — Show how close x-sin(x) is to x^3/6
    # ==================================================================
    print("\n" + "#"*70)
    print("# COLLINEARITY DIAGNOSTIC")
    print("#"*70)
    x_test = np.linspace(-1.5, 1.5, 100)
    diff = x_test - np.sin(x_test)
    cubic = x_test**3 / 6.0
    correlation = np.corrcoef(diff, cubic)[0, 1]
    max_error = np.max(np.abs(diff - cubic))
    print(f"  Correlation between (x - sin(x)) and (x³/6): {correlation:.8f}")
    print(f"  Max absolute error over [-1.5, 1.5]:          {max_error:.6f}")
    print(f"  For x in [-0.32, 0.32] (dimensionless range):")
    x_small = np.linspace(-0.32, 0.32, 100)
    diff_s = x_small - np.sin(x_small)
    cubic_s = x_small**3 / 6.0
    corr_s = np.corrcoef(diff_s, cubic_s)[0, 1]
    max_err_s = np.max(np.abs(diff_s - cubic_s))
    print(f"  Correlation: {corr_s:.10f}")
    print(f"  Max error:   {max_err_s:.10f}")

    # ==================================================================
    # SAVE FULL REPORT
    # ==================================================================
    report = {
        "sanity_checks": results,
        "collinearity": {
            "correlation_mm_range": float(correlation),
            "max_error_mm_range": float(max_error),
            "correlation_dimensionless_range": float(corr_s),
            "max_error_dimensionless_range": float(max_err_s)
        }
    }

    report_path = os.path.join(out_dir, "sindy_kocsis_sanity_checks.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
