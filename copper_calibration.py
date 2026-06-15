#!/usr/bin/env python3
"""
COPPER BLOCK CALIBRATION SCRIPT
================================
Phase 1: Fire a single electron at copper walls of increasing depth.
Measures: deflection angle, penetration, throughput change.

Purpose: Determine if tunnel depth changes the landing pattern before
committing to expensive ensemble runs.

Usage (runs on GPU automatically):
  python copper_calibration.py

Outputs:
  - Console: deflection table for 1, 5, 10, 20 layer depths
  - File: copper_calibration_results.json with raw data
"""
import torch
import numpy as np
import time
import json
import math

# ================================================================
# PHYSICS ENGINE (identical to teleparallel_collider.py)
# ================================================================
DT = 0.001
C = 100.0
PAULI_SCALAR = 500.0
LAMBDA_VAC = 0.001
TORSION_G = 1.0
TOTAL_TICKS = 50000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Copper Calibration Engine on: {device}")

def build_wall(wall_depth, wall_spacing=0.5, wall_extent=15.0,
               slit_width=2.0, slit_separation=6.0, wall_z_layers=1):
    """Build 3D wall positions with slit openings."""
    sw, ss = slit_width, slit_separation
    s1_lo = ss/2.0 - sw/2.0
    s1_hi = ss/2.0 + sw/2.0
    s2_lo = -ss/2.0 - sw/2.0
    s2_hi = -ss/2.0 + sw/2.0

    # Y positions (skipping slits)
    y_positions = []
    y = -wall_extent
    while y <= wall_extent:
        in_s1 = s1_lo <= y <= s1_hi
        in_s2 = s2_lo <= y <= s2_hi
        if not in_s1 and not in_s2:
            y_positions.append(y)
        y += wall_spacing

    # Z offsets
    z_offsets = [(zl - (wall_z_layers - 1)/2.0) * wall_spacing
                 for zl in range(wall_z_layers)]

    # 3D grid
    positions = []
    for x_layer in range(wall_depth):
        wx = -x_layer * wall_spacing
        for wy in y_positions:
            for wz in z_offsets:
                positions.append((wx, wy, wz))

    return positions, y_positions

def run_single_electron(wall_positions, beam_y, beam_hue,
                        beam_px=15.33, mass_e=0.511, mass_cu=59193.0,
                        screen_x=20.0, wall_x=0.0):
    """Run one electron through the wall. Returns final state dict."""
    N_wall = len(wall_positions)
    N = 1 + N_wall

    state = torch.zeros((N, 10), device=device, dtype=torch.float64)

    # Beam electron
    state[0, 1] = -10.0   # x
    state[0, 2] = beam_y   # y
    state[0, 4] = beam_px  # px
    state[0, 7] = mass_e   # m0
    state[0, 8] = beam_hue # hue

    # Wall particles
    for j, (wx, wy, wz) in enumerate(wall_positions):
        idx = 1 + j
        state[idx, 1] = wx
        state[idx, 2] = wy
        state[idx, 3] = wz
        state[idx, 7] = mass_cu
        state[idx, 8] = 0.0

    sw, ss = 2.0, 6.0
    s1_lo, s1_hi = ss/2.0 - sw/2.0, ss/2.0 + sw/2.0
    s2_lo, s2_hi = -ss/2.0 - sw/2.0, -ss/2.0 + sw/2.0

    outcome = "in_flight"
    final_y = None
    trajectory_x = []
    trajectory_y = []

    for tick in range(TOTAL_TICKS):
        pos = state[:, 1:4]
        mom = state[:, 4:7]
        m0 = state[:, 7:8]
        hue = state[:, 8:9]

        # Gamma factor
        p_mag_sq = (mom * mom).sum(dim=1, keepdim=True)
        gamma = torch.sqrt(1.0 + p_mag_sq / (m0 * C) ** 2)

        # Velocities
        vel = mom / (m0 * gamma)

        # Pairwise forces
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2, keepdim=True).clamp(min=1e-12)
        dist = torch.sqrt(dist_sq)

        # Pauli exclusion: F = PAULI * r_hat / r^3
        hue_diff = hue.unsqueeze(1) - hue.unsqueeze(0)
        pauli_mod = torch.cos(hue_diff)
        inv_r3 = 1.0 / (dist_sq * dist)
        force_pauli = PAULI_SCALAR * pauli_mod * inv_r3 * diff

        # Torsion: cross product coupling
        vel_i = vel.unsqueeze(1).expand_as(diff)
        torsion_force = TORSION_G * torch.cross(vel_i, diff, dim=2) * inv_r3

        # Vacuum damping
        vacuum_drag = -LAMBDA_VAC * mom / gamma

        # Total
        total_force = force_pauli.sum(dim=1) + torsion_force.sum(dim=1) + vacuum_drag

        # Integration
        mom_new = mom + total_force * DT
        p_mag_sq_new = (mom_new * mom_new).sum(dim=1, keepdim=True)
        gamma_new = torch.sqrt(1.0 + p_mag_sq_new / (m0 * C) ** 2)
        vel_new = mom_new / (m0 * gamma_new)

        state[:, 4:7] = mom_new
        state[:, 1:4] = pos + vel_new * DT
        state[:, 8] = (state[:, 8] + (m0.squeeze() / gamma_new.squeeze()) * DT) % (2 * np.pi)
        state[:, 0] += DT

        # Track beam electron
        bx = state[0, 1].item()
        by = state[0, 2].item()
        bpx = state[0, 4].item()

        if tick % 100 == 0:
            trajectory_x.append(bx)
            trajectory_y.append(by)

        # Screen detection
        if bx > screen_x:
            outcome = "screen"
            final_y = by
            break

        # Wall reflection (hard backup)
        if bx > wall_x:
            in_s1 = s1_lo <= by <= s1_hi
            in_s2 = s2_lo <= by <= s2_hi
            if not in_s1 and not in_s2:
                state[0, 1] = wall_x - 0.01
                state[0, 4] = -abs(state[0, 4].item())

        # Reflected and leaving
        if bx < -30.0 and bpx < 0:
            outcome = "reflected"
            break

    if outcome == "in_flight":
        outcome = "timeout"

    return {
        "outcome": outcome,
        "final_y": final_y,
        "final_x": state[0, 1].item(),
        "final_px": state[0, 4].item(),
        "final_py": state[0, 5].item(),
        "ticks": tick + 1,
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
    }


# ================================================================
# CALIBRATION: Depth Scan
# ================================================================
print("\n" + "=" * 70)
print("COPPER CALIBRATION: Wall Depth Scan")
print("=" * 70)
print(f"  Electron: 0.511 MeV, Copper: 59193 MeV")
print(f"  Beam px=15.33 (~0.3c)")
print(f"  Testing depths: 1, 5, 10, 20 layers")
print(f"  Testing y offsets near slit edge for max sensitivity")
print()

depths = [1, 5, 10, 20]
# Test at y-positions that should go through slit 1 (centered at y=3.0)
test_ys = [3.0, 2.5, 3.5, 2.1, 3.9]  # center and near-edge positions
test_hue = 1.5  # Fixed hue for reproducibility

results = {}

for depth in depths:
    print(f"\n--- Depth = {depth} layers ({(depth-1)*0.5:.1f} sim units deep) ---")
    wall_pos, _ = build_wall(depth)
    N_wall = len(wall_pos)
    print(f"  Wall particles: {N_wall}")

    depth_results = []
    t0 = time.time()

    for yi, beam_y in enumerate(test_ys):
        result = run_single_electron(wall_pos, beam_y, test_hue)
        depth_results.append({
            "beam_y": beam_y,
            **result
        })
        status = result["outcome"]
        fy = f"y={result['final_y']:.4f}" if result['final_y'] else "N/A"
        print(f"  y={beam_y:.1f}: {status:10s} {fy} ({result['ticks']} ticks)")

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    results[f"depth_{depth}"] = {
        "n_wall": N_wall,
        "elapsed": elapsed,
        "trials": depth_results
    }

# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 70)
print("DEPTH SCAN SUMMARY")
print("=" * 70)
print(f"{'Depth':>6} {'N_wall':>7} {'Screen':>7} {'Reflected':>10} {'Mean y':>10} {'Time':>8}")
print("-" * 55)

for depth in depths:
    data = results[f"depth_{depth}"]
    trials = data["trials"]
    screen_hits = [t for t in trials if t["outcome"] == "screen"]
    reflected = [t for t in trials if t["outcome"] != "screen"]
    n_screen = len(screen_hits)
    n_reflected = len(reflected)
    mean_y = np.mean([t["final_y"] for t in screen_hits]) if screen_hits else float('nan')
    print(f"{depth:>6} {data['n_wall']:>7} {n_screen:>7} {n_reflected:>10} {mean_y:>10.4f} {data['elapsed']:>7.1f}s")

# Save results
out_path = "copper_calibration_results.json"
# Convert for JSON serialization
for k in results:
    for t in results[k]["trials"]:
        t["trajectory_x"] = t["trajectory_x"][-5:]  # Keep last 5 points only
        t["trajectory_y"] = t["trajectory_y"][-5:]
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print("Run this on your local GPU to calibrate tunnel depth response.")
