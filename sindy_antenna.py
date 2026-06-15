import numpy as np
import pysindy as ps
import sys

# Load the trajectory
try:
    history = np.load("electron_trajectory.npy")
except FileNotFoundError:
    print("Error: electron_trajectory.npy not found.")
    sys.exit(1)

# history shape: (ticks, N, 10)
ticks, N, dims = history.shape
print(f"Loaded trajectory with {ticks} ticks, {N} particles, {dims} features.")

if N < 2:
    print("Need at least 2 particles (1 anchor, 1+ bath) to run antenna analysis.")
    sys.exit(1)

# We want to extract equations for a *surrounding* particle (e.g. index 1)
# to see if it synchronized to the anchor (index 0) amidst the spatial noise.

dt = 0.001

# The 10D State Vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
# We'll build a custom feature library including 1/r^2 or 1/r^3 if needed,
# but the prompt specifically mentioned extracting the equation for theta_hue' (hue velocity).

# Let's extract the dynamics of the first thermal bath particle
bath_idx = 15
state_0 = history[:, 0, :] # Still use anchor 0 as the reference for hue_diff
state_1 = history[:, bath_idx, :]
# Indices: 1: x, 2: y, 3: z, 4: px, 5: py, 6: pz, 7: m0, 8: hue, 9: gamma

# Features for SINDy:
hue_0 = state_0[:, 8:9]
hue_1 = state_1[:, 8:9]
hue_diff = hue_1 - hue_0

# Add debug to see the actual derivative of hue_1
dhue_dt = (hue_1[1:] - hue_1[:-1]) / dt
dhue_dt = np.where(dhue_dt > np.pi/dt, dhue_dt - 2*np.pi/dt, dhue_dt)
dhue_dt = np.where(dhue_dt < -np.pi/dt, dhue_dt + 2*np.pi/dt, dhue_dt)

print(f"Max hue_1 derivative: {np.max(dhue_dt):.3f}")
print(f"Min hue_1 derivative: {np.min(dhue_dt):.3f}")
print(f"Mean hue_1 derivative: {np.mean(dhue_dt):.3f}")
print(f"Sample dhue_dt vs sin(hue_diff):")
for i in range(10, 15):
    print(f"  t={i*dt:.3f}: dhue_dt={dhue_dt[i][0]:.3f}, sin(hue_diff)={np.sin(hue_diff[i][0]):.3f}, m0/gamma={state_1[i, 7]/state_1[i, 9]:.3f}")

X = np.hstack([state_1[:, 1:8], hue_1, state_1[:, 9:10], hue_diff]) # 10 variables
feature_names = ['x', 'y', 'z', 'px', 'py', 'pz', 'm0', 'hue', 'gamma', 'hue_diff']

# PySINDy setup
poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
fourier_lib = ps.FourierLibrary(n_frequencies=1)
combined_lib = poly_lib + fourier_lib

optimizer = ps.STLSQ(threshold=0.5, alpha=0.05)

model = ps.SINDy(feature_library=combined_lib, optimizer=optimizer)

print("\nRunning SINDy on Particle 1 (thermal bath particle) over the full 10,000 ticks...")
model.fit(X, t=dt, feature_names=feature_names)

print("\n--- SINDy Extraction Results ---")
model.print()

