import numpy as np
import matplotlib.pyplot as plt

# Load trajectory
history = np.load('electron_trajectory.npy')
state_0 = history[:, 0, :] # Electron
x = state_0[:, 1]
y = state_0[:, 2]
px = state_0[:, 4]
hue = state_0[:, 8]
ticks = np.arange(history.shape[0])

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0d1117')

# Plot 1: Trajectory
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('#0d1117')
ax1.plot(x, y, color='#58a6ff', linewidth=1.5, alpha=0.8)
ax1.set_title("QED Simulation: Electron Trajectory (x vs y)", color='#e6edf3', fontsize=12)
ax1.set_xlabel("x Position", color='#8b949e')
ax1.set_ylabel("y Position", color='#8b949e')
ax1.tick_params(colors='#8b949e')
ax1.grid(True, color='#30363d', alpha=0.5)

# Plot 2: Momentum (Compton Recoil)
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('#0d1117')
ax2.plot(ticks, px, color='#ff7b72', linewidth=1.5)
ax2.set_title("Compton Scattering: Momentum Recoil (px vs time)", color='#e6edf3', fontsize=12)
ax2.set_xlabel("Time (ticks)", color='#8b949e')
ax2.set_ylabel("Momentum (px)", color='#8b949e')
ax2.tick_params(colors='#8b949e')
ax2.grid(True, color='#30363d', alpha=0.5)

# Plot 3: U(1) Phase Evolution
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('#0d1117')
ax3.plot(ticks, hue, color='#d2a8ff', linewidth=1.5)
ax3.set_title("U(1) Gauge Phase Evolution (hue vs time)", color='#e6edf3', fontsize=12)
ax3.set_xlabel("Time (ticks)", color='#8b949e')
ax3.set_ylabel("Phase (radians)", color='#8b949e')
ax3.tick_params(colors='#8b949e')
ax3.grid(True, color='#30363d', alpha=0.5)

# Plot 4: SINDy Equations Text
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('#0d1117')
ax4.axis('off')

eq_text = """
SINDy Extracted QED Differential Equations:

(x)' =  0.974 px
(y)' = -0.016 x + 0.979 py - 0.001 x py

Compton Scattering (Radiation Pressure):
(px)' = ... + 0.078 sin(1 x) + 0.099 cos(1 x)

Vacuum Polarization (Uehling Potential):
(py)' = ... -0.060 sin(1 hue_diff) + 0.093 cos(1 hue_diff)

Lamb Shift (Self-Energy):
(hue)' = ... + 9.201 px hue - 2.707 px hue_diff
"""
ax4.text(0.05, 0.5, eq_text, color='#7ee787', fontsize=11, family='monospace', verticalalignment='center')
ax4.set_title("PySINDy Extraction Results", color='#e6edf3', fontsize=12)

plt.tight_layout()

# Save to the user's artifact directory so it pops up in their UI
out_path = r"C:\Users\Myna Bird\.gemini\antigravity\brain\627ddbc0-eb33-4d84-9d4c-abc9c787cfcf\qed_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"Saved artifact to {out_path}")
