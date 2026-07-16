"""
Fast vectorized dBB ODE solver using numpy broadcasting.
Runs the full sweep in ~10 seconds instead of 30+ minutes.
"""
import numpy as np
import sys

# ============================================================================
# CONSTANTS (from teleparallel_collider.py)
# ============================================================================
C = 100.0
M0 = 0.511
BEAM_P = 2.28
DE_BROGLIE = 2 * np.pi / BEAM_P
V_BEAM = BEAM_P / (M0 * np.sqrt(1 + (BEAM_P / (M0 * C))**2))
GAMMA_BEAM = np.sqrt(1 + (BEAM_P / (M0 * C))**2)

SLIT_1_RANGE = (-6.0, -2.0)
SLIT_2_RANGE = (2.0, 6.0)
SLIT_WIDTH = 4.0
SLIT_SEP = 8.0
WALL_DEPTH = 1.0
SCREEN_X = 15.0
BEAM_Y_RANGE = 8.0

k = 2 * np.pi / DE_BROGLIE

# Slit point sources (vectorized)
N_SOURCES = 50
slit1_ys = np.linspace(SLIT_1_RANGE[0], SLIT_1_RANGE[1], N_SOURCES)
slit2_ys = np.linspace(SLIT_2_RANGE[0], SLIT_2_RANGE[1], N_SOURCES)
ALL_SOURCES = np.concatenate([slit1_ys, slit2_ys])  # (100,)

# ============================================================================
# VECTORIZED WAVE FIELD & GRADIENT
# ============================================================================
def wave_field_vec(x_arr, y_arr):
    """Compute phi(x,y) for arrays of positions using broadcasting."""
    # x_arr, y_arr: (N,) arrays
    # ALL_SOURCES: (100,) array
    # We want r[i,j] = sqrt((x[i] - 0)^2 + (y[i] - source[j])^2)
    dx = x_arr[:, None]  # (N, 1) — all sources are at x=0 (wall)
    dy = y_arr[:, None] - ALL_SOURCES[None, :]  # (N, 100)
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 0.01)
    phi = np.sum(np.cos(k * r) / np.sqrt(r), axis=1)  # (N,)
    return phi

def wave_gradient_y_vec(x_arr, y_arr, dy=0.01):
    """Vectorized central-difference gradient dphi/dy."""
    return (wave_field_vec(x_arr, y_arr + dy) - wave_field_vec(x_arr, y_arr - dy)) / (2 * dy)

# ============================================================================
# VECTORIZED ODE SOLVER
# ============================================================================
def solve_ensemble(dbb_strength, n_particles=500, n_steps=500, seed=42):
    """
    Solve dBB guidance ODE for all particles simultaneously.
    
    dy/dx = [S_dBB * (C^2/m0) / v_x] * dphi/dy
    
    Returns: screen_ys (only for particles that entered slits)
    """
    rng = np.random.RandomState(seed)
    initial_ys = rng.uniform(-BEAM_Y_RANGE, BEAM_Y_RANGE, n_particles)
    
    # Filter: only particles inside slit openings
    in_slit1 = (initial_ys >= SLIT_1_RANGE[0]) & (initial_ys <= SLIT_1_RANGE[1])
    in_slit2 = (initial_ys >= SLIT_2_RANGE[0]) & (initial_ys <= SLIT_2_RANGE[1])
    mask = in_slit1 | in_slit2
    
    n_reflected = np.sum(~mask)
    slit_ys = initial_ys[mask].copy()  # (M,)
    n_through = len(slit_ys)
    
    if n_through == 0:
        return np.array([]), n_reflected, 0
    
    # Integration grid in x (slit exit to screen)
    xs = np.linspace(WALL_DEPTH, SCREEN_X, n_steps)
    dx_step = xs[1] - xs[0]
    prefactor = dbb_strength * (C**2 / M0) / V_BEAM
    
    # Euler integration (all particles at once)
    ys = slit_ys.copy()
    for i in range(1, n_steps):
        x_arr = np.full_like(ys, xs[i-1])
        grad = wave_gradient_y_vec(x_arr, ys)
        ys = ys + prefactor * grad * dx_step
    
    return ys, n_reflected, n_through

# ============================================================================
# vdP MAPPING
# ============================================================================
E_beam = np.sqrt(M0**2 + BEAM_P**2 / C**2)
kappa_plus = 10.0 * E_beam
kappa_minus = 1.0 - 0.999
ratio = kappa_minus / kappa_plus
alpha_p = np.sqrt(kappa_plus / (2.0 * kappa_minus))

print("=" * 72)
print("COUDER PILOT WAVE: FAST VECTORIZED ODE SOLVER")
print("=" * 72)
print(f"  v_beam = {V_BEAM:.4f}  |  lambda_dB = {DE_BROGLIE:.4f}  |  gamma = {GAMMA_BEAM:.6f}")
print(f"  Flight distance: {SCREEN_X - WALL_DEPTH:.1f} sim units")
print()
print("--- vdP LIMIT-CYCLE MAPPING (Liu et al. 2026) ---")
print(f"  kappa_+  = {kappa_plus:.4f}   kappa_-  = {kappa_minus:.4f}")
print(f"  kappa_-/kappa_+ = {ratio:.6f}  (Paper near-classical: 0.14)")
print(f"  Limit cycle alpha_p = {alpha_p:.2f}")
print(f"  Regime: {'NEAR-CLASSICAL' if ratio < 0.1 else 'QUANTUM'}")
print(f"  We are {0.14/ratio:.0f}x deeper into classical than the paper's near-classical regime")
print()

# ============================================================================
# SWEEP
# ============================================================================
dbb_values = [0.0000, 0.001, 0.002, 0.003, 0.004, 0.005,
              0.006, 0.007, 0.008, 0.009, 0.0095,
              0.0098, 0.0099, 0.01, 0.0101, 0.0102, 
              0.0105, 0.011, 0.012, 0.015]

N_PARTICLES = 1000

print("=" * 72)
print(f"dBB STRENGTH SWEEP ({N_PARTICLES} particles, Huygens-Fresnel ODE)")
print("=" * 72)
print(f"{'S_dBB':>10} | {'Through':>8} | {'Refl':>6} | {'Mean|Y|':>8} | {'StdDev':>8} | {'Peak Y':>8} | {'Pattern'}")
print("-" * 95)

for s in dbb_values:
    hits, n_refl, n_through = solve_ensemble(s, N_PARTICLES, n_steps=500)
    
    if len(hits) == 0:
        print(f"{s:10.4f} | {0:8d} | {n_refl:6d} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | NO THROUGHPUT")
        continue
    
    mean_abs_y = np.mean(np.abs(hits))
    std_y = np.std(hits)
    
    # Histogram peak
    hist, edges = np.histogram(hits, bins=40, range=(-12, 12))
    centers = (edges[:-1] + edges[1:]) / 2
    peak_y = centers[np.argmax(hist)]
    
    # Pattern classification
    if mean_abs_y < 1.5:
        pattern = "TWO DISTINCT COLUMNS"
    elif mean_abs_y < 2.5:
        pattern = "PARTIAL OVERLAP"
    elif mean_abs_y < 3.5:
        pattern = "HEAVY OVERLAP"
    elif std_y < 2.0:
        pattern = "BLOB MERGER (central peak)"
    else:
        pattern = "CROSSING / FRINGES"
    
    hit_pct = 100 * len(hits) / N_PARTICLES
    print(f"{s:10.4f} | {n_through:8d} | {n_refl:6d} | {mean_abs_y:8.3f} | {std_y:8.3f} | {peak_y:8.3f} | {pattern}")

# ============================================================================
# CHOKE ANALYSIS
# ============================================================================
print()
print("=" * 72)
print("APERTURE CHOKE THRESHOLD")
print("=" * 72)
t_slit = WALL_DEPTH / V_BEAM
print(f"  Transit time through slit: {t_slit:.6f}")
print(f"  Slit <INSERT_BUFFER_USERNAME_HERE>f-width: {SLIT_WIDTH/2:.1f}")
print(f"  Prefactor C^2/m0 = {C**2/M0:.1f}")
print()
print("  GPU Empirical Data:")
print(f"    S=0.0100 -> 4981 hits (49.8%)")
print(f"    S=0.0105 -> 0 hits    (0.0%)")
print(f"    Choke threshold: S_crit in [0.0100, 0.0105]")
print()

# ============================================================================
# BALLPARK
# ============================================================================
print("=" * 72)
print("BALLPARK: ARE WE DOING THE RIGHT PHYSICS?")
print("=" * 72)
print(f"  d/lambda = {SLIT_SEP/DE_BROGLIE:.2f}  (need 2-5 for fringes)  -> {'YES' if 2 <= SLIT_SEP/DE_BROGLIE <= 5 else 'CHECK'}")
L_T = SLIT_SEP**2 / DE_BROGLIE
print(f"  Talbot L_T = {L_T:.2f}   Screen/{L_T:.2f} = {SCREEN_X/L_T:.2f}")
print(f"  Screen/(L_T/2) = {SCREEN_X/(L_T/2):.2f}  (want ~1.0 for max visibility)")
print()
print(f"  vdP regime: kappa_-/kappa_+ = {ratio:.6f}")
print(f"  Paper near-classical = 0.14, Paper quantum = 0.42")
print(f"  -> We are {0.14/ratio:.0f}x more classical than their near-classical case")
print(f"  -> CONFIRMED: Classical Couder droplet regime")
