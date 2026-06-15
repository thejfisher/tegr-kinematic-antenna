"""Quick diagnostic: what does the pulsar hue actually look like?"""
import numpy as np

trajectory = np.load("electron_trajectory.npy")
source_hue = trajectory[:, -1, 8]
pulsar_hue = trajectory[:, 0, 8]

# Sample at monthly positions
n_months = 90
n_ticks = trajectory.shape[0]
sample_ticks = np.linspace(0, n_ticks - 1, n_months).astype(int)

print("Source hue (first 12 months):", np.round(source_hue[sample_ticks[:12]], 4))
print("Pulsar hue (first 12 months):", np.round(pulsar_hue[sample_ticks[:12]], 4))
print()
print(f"Source hue range: [{source_hue.min():.4f}, {source_hue.max():.4f}]")
print(f"Pulsar hue range: [{pulsar_hue.min():.4f}, {pulsar_hue.max():.4f}]")
print()
print(f"Source hue std: {source_hue.std():.4f}")
print(f"Pulsar hue std: {pulsar_hue.std():.4f}")

# Check if it's wrapping around 2*pi
wraps = np.sum(np.abs(np.diff(source_hue)) > 3.0)
print(f"\nSource phase wraps (jumps > pi): {wraps}")
wraps_p = np.sum(np.abs(np.diff(pulsar_hue)) > 3.0)
print(f"Pulsar phase wraps (jumps > pi): {wraps_p}")

# Try unwrapping the phase
source_unwrapped = np.unwrap(source_hue)
pulsar_unwrapped = np.unwrap(pulsar_hue)
print(f"\nUnwrapped source range: [{source_unwrapped.min():.2f}, {source_unwrapped.max():.2f}]")
print(f"Unwrapped pulsar range: [{pulsar_unwrapped.min():.2f}, {pulsar_unwrapped.max():.2f}]")

# Compute the DIFFERENCE (phase lag) between source and pulsar
phase_diff = pulsar_unwrapped[sample_ticks] - source_unwrapped[sample_ticks]
print(f"\nPhase difference (pulsar - source) at monthly samples:")
print(f"  Mean: {phase_diff.mean():.4f}")
print(f"  Std:  {phase_diff.std():.4f}")
print(f"  Range: [{phase_diff.min():.4f}, {phase_diff.max():.4f}]")

# What about the rate of change of pulsar hue? That's what SINDy was fitting
pulsar_hue_dot = np.diff(pulsar_unwrapped) / 0.001  # dt = 0.001
print(f"\nPulsar hue' (derivative) stats:")
print(f"  Mean: {pulsar_hue_dot.mean():.2f}")
print(f"  Std:  {pulsar_hue_dot.std():.2f}")
print(f"  Range: [{pulsar_hue_dot.min():.2f}, {pulsar_hue_dot.max():.2f}]")
