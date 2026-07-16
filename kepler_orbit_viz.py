"""
TEGR Kepler Orbit Visualization
================================
Load trajectory .npy from the Kepler Orbit test (Preset 36)
and generate 3D orbital trajectory + physics diagnostics.

State Vector: [tau, x, y, z, px, py, pz, m0, hue, gamma]

Usage:
  python kepler_orbit_viz.py                          # default: tegr_output.npy
  python kepler_orbit_viz.py --input my_trajectory.npy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='tegr_output.npy', help='Path to trajectory .npy file')
parser.add_argument('--dt', type=float, default=0.005, help='Timestep')
parser.add_argument('--C', type=float, default=100.0, help='Speed of light in sim units')
parser.add_argument('--GM', type=float, default=5.0, help='G*M for gravity equation check')
args = parser.parse_args()

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading {args.input}...")
states = np.load(args.input)
print(f"Shape: {states.shape}")

if states.ndim == 2:
    print("ERROR: This file contains final states only (2D), not a trajectory (3D).")
    print("Re-run the simulation — the collider now saves trajectories even when using ZMQ.")
    sys.exit(1)

T, N, D = states.shape
dt = args.dt
time = np.arange(T) * dt
C_LIGHT = args.C

# State Vector Indices
TAU, X, Y, Z, PX, PY, PZ, M0, HUE, GAMMA = range(10)

# Extract particle trajectories
A = states[:, 0, :]  # Particle A (orbiting)
if N > 1:
    B = states[:, 1, :]  # Particle B (spectator/freefall)
else:
    B = None

print(f"\nParticle A initial: pos=({A[0,X]:.2f}, {A[0,Y]:.2f}, {A[0,Z]:.2f})")
print(f"  momentum: ({A[0,PX]:.4f}, {A[0,PY]:.4f}, {A[0,PZ]:.4f}), m0={A[0,M0]:.4f}")
if B is not None:
    print(f"Particle B initial: pos=({B[0,X]:.2f}, {B[0,Y]:.2f}, {B[0,Z]:.2f})")
    print(f"  momentum: ({B[0,PX]:.4f}, {B[0,PY]:.4f}, {B[0,PZ]:.4f}), m0={B[0,M0]:.4f}")

# ============================================================
# DERIVED QUANTITIES
# ============================================================
r_A = np.sqrt(A[:, X]**2 + A[:, Y]**2 + A[:, Z]**2)
p_mag_A = np.sqrt(A[:, PX]**2 + A[:, PY]**2 + A[:, PZ]**2)
gamma_A = np.sqrt(1.0 + p_mag_A**2 / (A[:, M0]**2 * C_LIGHT**2 + 1e-12))
v_A = p_mag_A / (gamma_A * A[:, M0] + 1e-12)
Lz_A = A[:, X] * A[:, PY] - A[:, Y] * A[:, PX]
tau_A = A[:, TAU]

if B is not None:
    r_B = np.sqrt(B[:, X]**2 + B[:, Y]**2 + B[:, Z]**2)
    p_mag_B = np.sqrt(B[:, PX]**2 + B[:, PY]**2 + B[:, PZ]**2)
    gamma_B = np.sqrt(1.0 + p_mag_B**2 / (B[:, M0]**2 * C_LIGHT**2 + 1e-12))
    v_B = p_mag_B / (gamma_B * B[:, M0] + 1e-12)
    tau_B = B[:, TAU]

# ============================================================
# FIGURE: 5-PANEL DIAGNOSTIC
# ============================================================
fig = plt.figure(figsize=(18, 16))
fig.suptitle(f"TEGR Kepler Orbit — {T} ticks, dt={dt}, C={C_LIGHT}", 
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
skip = max(1, T // 3000)  # Downsample for plotting

# ----- 3D ORBIT -----
ax1 = fig.add_subplot(gs[0, :], projection='3d')
ax1.plot(A[::skip, X], A[::skip, Y], A[::skip, Z], 'b-', linewidth=0.5, alpha=0.8, label='Particle A (orbit)')
if B is not None:
    ax1.plot(B[::skip, X], B[::skip, Y], B[::skip, Z], 'r-', linewidth=0.5, alpha=0.6, label='Particle B (freefall)')
ax1.scatter([0], [0], [0], c='gold', s=300, marker='*', zorder=10, edgecolors='orange', label=f'Sink (GM={args.GM})')
ax1.scatter([A[0, X]], [A[0, Y]], [A[0, Z]], c='blue', s=80, marker='o', zorder=10, edgecolors='navy')
if B is not None:
    ax1.scatter([B[0, X]], [B[0, Y]], [B[0, Z]], c='red', s=80, marker='o', zorder=10, edgecolors='darkred')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.set_title('3D Orbital Trajectory', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)

# ----- r(t) -----
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time[::skip], r_A[::skip], 'b-', linewidth=0.8, label=f'A: r∈[{r_A.min():.2f}, {r_A.max():.2f}]')
if B is not None:
    ax2.plot(time[::skip], r_B[::skip], 'r-', linewidth=0.8, label=f'B: r∈[{r_B.min():.2f}, {r_B.max():.2f}]')
ax2.axhline(y=20.0, color='blue', linestyle='--', alpha=0.3)
ax2.set_xlabel('Time (sim units)', fontsize=11)
ax2.set_ylabel('r (distance from sink)', fontsize=11)
ax2.set_title('Orbital Radius r(t)', fontsize=13)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ----- gamma(t) -----
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time[::skip], gamma_A[::skip], 'b-', linewidth=0.8, label=f'γ_A max={gamma_A.max():.4f}')
if B is not None:
    ax3.plot(time[::skip], gamma_B[::skip], 'r-', linewidth=0.8, label=f'γ_B max={gamma_B.max():.4f}')
ax3.set_xlabel('Time (sim units)', fontsize=11)
ax3.set_ylabel('γ (Lorentz factor)', fontsize=11)
ax3.set_title('Lorentz Factor γ(t)', fontsize=13)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ----- Angular Momentum L_z(t) -----
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(time[::skip], Lz_A[::skip], 'b-', linewidth=0.8, label=f'L_z mean={Lz_A.mean():.4f}')
if np.abs(Lz_A.mean()) > 1e-10:
    Lz_var = Lz_A.std() / np.abs(Lz_A.mean()) * 100
    ax4.set_title(f'Angular Momentum L_z(t)  [σ/μ = {Lz_var:.4f}%]', fontsize=13)
else:
    ax4.set_title('Angular Momentum L_z(t)', fontsize=13)
ax4.set_xlabel('Time (sim units)', fontsize=11)
ax4.set_ylabel('L_z = x·py − y·px', fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ----- Proper Time τ(t) -----
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(time[::skip], tau_A[::skip], 'b-', linewidth=0.8, label=f'τ_A final={tau_A[-1]:.3f}')
if B is not None:
    ax5.plot(time[::skip], tau_B[::skip], 'r-', linewidth=0.8, label=f'τ_B final={tau_B[-1]:.3f}')
ax5.plot(time[::skip], time[::skip], 'k--', linewidth=0.5, alpha=0.4, label='τ = t (no dilation)')
ax5.set_xlabel('Coordinate Time t', fontsize=11)
ax5.set_ylabel('Proper Time τ', fontsize=11)
ax5.set_title('Proper Time τ(t) — Time Dilation', fontsize=13)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

OUTPUT = args.input.replace('.npy', '_analysis.png')
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to: {OUTPUT}")
plt.show()

# ============================================================
# GRAVITY EQUATION EXTRACTION
# ============================================================
print("\n" + "="*60)
print("GRAVITY EQUATION PROBE")
print("="*60)

# Compute acceleration from finite differences on momentum
dpx = np.diff(A[:, PX]) / dt
dpy = np.diff(A[:, PY]) / dt
dpz = np.diff(A[:, PZ]) / dt
# a = F/m = dp/dt / m (but dp/dt IS the force, and for gravity F = GMm/r², so dp/dt = GMm/r²)
# Actually dp/dt = F, and the acceleration felt by the particle is a = F/m * (1/gamma) for relativistic
# But for the power law fit we just care about |dp/dt| vs r

F_mag = np.sqrt(dpx**2 + dpy**2 + dpz**2)
r_mid = r_A[:-1]

# Fit log(|F|) vs log(r): F ∝ r^n → expect n = -2 for gravity
valid = (r_mid > 0.5) & (F_mag > 1e-12)
if valid.sum() > 100:
    log_r = np.log(r_mid[valid])
    log_F = np.log(F_mag[valid])
    
    # Robust fit (clip outliers)
    q05, q95 = np.percentile(log_F, [5, 95])
    mask = (log_F > q05) & (log_F < q95)
    if mask.sum() > 50:
        coeffs = np.polyfit(log_r[mask], log_F[mask], 1)
        n_power = coeffs[0]
        GM_extracted = np.exp(coeffs[1]) / A[0, M0]  # F = GMm/r^n → log(F) = log(GMm) + n*log(r)
        
        print(f"  Power law fit: |dp/dt| ∝ r^{n_power:.3f}")
        print(f"  Expected for Newtonian gravity: r^-2.000")
        print(f"  Deviation from -2: {abs(n_power + 2):.4f}")
        print(f"  GM extracted: {GM_extracted:.3f} (expected: {args.GM})")
        print(f"  Fit data points: {mask.sum()}")
    else:
        print(f"  Not enough filtered points for fit (only {mask.sum()})")
else:
    print(f"  Not enough valid data points for power law fit (only {valid.sum()})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"Total time: {T*dt:.1f} sim units ({T} ticks × dt={dt})")
print(f"C (speed of light): {C_LIGHT}")
print(f"\n--- Particle A ---")
print(f"  r: [{r_A.min():.3f}, {r_A.max():.3f}]")
print(f"  |v|: [{v_A.min():.4f}, {v_A.max():.4f}] (v/c: [{v_A.min()/C_LIGHT:.6f}, {v_A.max()/C_LIGHT:.6f}])")
print(f"  γ: [{gamma_A.min():.6f}, {gamma_A.max():.6f}]")
print(f"  L_z: mean={Lz_A.mean():.6f}, std={Lz_A.std():.6f}")
if np.abs(Lz_A.mean()) > 1e-10:
    print(f"  L_z conservation: {Lz_A.std()/np.abs(Lz_A.mean())*100:.6f}%")
print(f"  m0: [{A[:,M0].min():.6f}, {A[:,M0].max():.6f}]")
print(f"  τ final: {tau_A[-1]:.4f}")
if B is not None:
    print(f"\n--- Particle B ---")
    print(f"  r: [{r_B.min():.3f}, {r_B.max():.3f}]")
    print(f"  γ max: {gamma_B.max():.6f}")
    print(f"  τ final: {tau_B[-1]:.4f}")
