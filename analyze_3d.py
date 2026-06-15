import numpy as np
import matplotlib.pyplot as plt
import os

print("--- Teleparallel 3D Phase Router Analysis ---")

# Ensure the aggregate file exists
file_path = "aggregate_states.npz"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please run a Tonomura simulation from the GUI first.")
    exit(1)

# Load the saved state from the latest run
data = np.load(file_path)
final_states = data['final_states']      # shape: (N, 10)
initial_states = data['initial_states']  # shape: (N, 2)  [initial_y, initial_hue]
outcomes = data['outcomes']              # shape: (N,)    [1=hit, -1=reflected]

# Filter for particles that hit the screen
hit_mask = outcomes == 1
hit_initial_states = initial_states[hit_mask]
hit_final_states = final_states[hit_mask]

total_trials = len(outcomes)
hits = np.sum(hit_mask)
print(f"Total Trials: {total_trials}")
print(f"Particles Reaching Screen: {hits} ({hits/total_trials*100:.2f}%)")

if hits == 0:
    print("Zero particles reached the screen. The tunnel is perfectly opaque.")
    print("Try widening the slit, or running a 2D tunnel to test.")
    exit(0)

# Extract Data
initial_hues = hit_initial_states[:, 1]
final_ys = hit_final_states[:, 2]
initial_ys = hit_initial_states[:, 0]

print(f"\nMath Extraction Complete. Plotting Deterministic Phase Routing...")

# Create the scatter plot showing the core discovery
plt.figure(figsize=(10, 6))

# Plot Hue vs Final Y Position
plt.scatter(initial_hues, final_ys, c=initial_ys, cmap='viridis', alpha=0.6, s=15)

plt.title("Deterministic Phase Routing (Hue vs Final Y Position)\nColor = Initial Y Position", fontsize=14, fontweight='bold')
plt.xlabel("Initial Hidden Phase (θ_hue) [Radians]", fontsize=12)
plt.ylabel("Final Screen Position (Y) [sim units]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Highlight standard diffraction fringes
plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='Center (y=0)')

cbar = plt.colorbar()
cbar.set_label("Initial Starting Y Position")

plt.tight_layout()
save_path = "phase_router_analysis.png"
plt.savefig(save_path, dpi=200)
print(f"\nSaved high-resolution plot to: {save_path}")
print("If the scatter plot forms distinct angled bands, it proves that Hue rigidly dictates the final quantum state!")

plt.show()
