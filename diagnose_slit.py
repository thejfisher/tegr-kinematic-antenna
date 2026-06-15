"""Diagnostic: trace beam particle trajectories to find why 0% reach screen."""
import numpy as np

history = np.load("electron_trajectory.npy")
print(f"Trajectory shape: {history.shape}")  # (ticks, N, 10)

N_beam = 50
WALL_X = 0.0
SCREEN_X = 20.0

# State: [t, x, y, z, px, py, pz, m0, hue, gamma]

# 1. Track when each beam particle's x is closest to the wall
print("\n=== BEAM PARTICLE WALL APPROACH ===")
print(f"{'Particle':<10} {'Start Y':>10} {'Start Hue':>10} {'Max X':>10} {'Max X Tick':>10} {'Y @ Max X':>10} {'In Slit?':>10}")
print("-" * 80)

slit_aimed = 0
actually_passed = 0

for i in range(N_beam):
    x_series = history[:, i, 1]
    y_series = history[:, i, 2]
    hue_start = history[0, i, 8]
    y_start = history[0, i, 2]
    
    # Find the tick where x is maximum (closest to/past wall)
    max_x_tick = np.argmax(x_series)
    max_x = x_series[max_x_tick]
    y_at_max_x = y_series[max_x_tick]
    
    in_s1 = 2.0 <= y_at_max_x <= 4.0
    in_s2 = -4.0 <= y_at_max_x <= -2.0
    in_slit = in_s1 or in_s2
    
    # Check if start y is in slit
    start_in_s1 = 2.0 <= y_start <= 4.0
    start_in_s2 = -4.0 <= y_start <= -2.0
    if start_in_s1 or start_in_s2:
        slit_aimed += 1
    
    if max_x > WALL_X:
        actually_passed += 1
    
    if i < 20 or in_slit or (start_in_s1 or start_in_s2):  # Print first 20 + any interesting ones
        print(f"{i:<10} {y_start:>10.2f} {hue_start:>10.2f} {max_x:>10.3f} {max_x_tick:>10} {y_at_max_x:>10.2f} {'YES' if in_slit else 'no':>10}")

print(f"\nParticles aimed at slits (start y in slit range): {slit_aimed}")
print(f"Particles that crossed x=0: {actually_passed}")

# 2. Look at the wall particle hue evolution
print("\n=== WALL PARTICLE HUE EVOLUTION ===")
wall_idx = N_beam  # First wall particle
print(f"Wall particle {wall_idx}:")
print(f"  Tick 0: hue={history[0, wall_idx, 8]:.4f}, x={history[0, wall_idx, 1]:.4f}")
print(f"  Tick 10: hue={history[10, wall_idx, 8]:.4f}, x={history[10, wall_idx, 1]:.4f}")
print(f"  Tick 100: hue={history[100, wall_idx, 8]:.4f}, x={history[100, wall_idx, 1]:.4f}")
print(f"  Tick 500: hue={history[500, wall_idx, 8]:.4f}, x={history[500, wall_idx, 1]:.4f}")
print(f"  Tick 1000: hue={history[1000, wall_idx, 8]:.4f}, x={history[1000, wall_idx, 1]:.4f}")

# 3. Trace particle 0 x-position and px over time
print("\n=== PARTICLE 0 TRAJECTORY (every 100 ticks) ===")
print(f"{'Tick':<8} {'x':>10} {'y':>10} {'px':>12} {'py':>12} {'hue':>8} {'gamma':>10}")
for t in range(0, min(3000, history.shape[0]), 100):
    p0 = history[t, 0]
    print(f"{t:<8} {p0[1]:>10.3f} {p0[2]:>10.3f} {p0[4]:>12.1f} {p0[5]:>12.1f} {p0[8]:>8.3f} {p0[9]:>10.3f}")

# 4. Final positions
print("\n=== FINAL BEAM POSITIONS ===")
final = history[-1, :N_beam, :]
x_final = final[:, 1]
y_final = final[:, 2]
print(f"X range: [{x_final.min():.1f}, {x_final.max():.1f}]")
print(f"Y range: [{y_final.min():.1f}, {y_final.max():.1f}]")
print(f"Particles with x > 0: {(x_final > 0).sum()}")
print(f"Particles with x > 20: {(x_final > 20).sum()}")
print(f"Particles with x < -50: {(x_final < -50).sum()}")
