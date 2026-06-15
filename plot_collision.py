import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot 3D particle collision trajectories.")
    parser.add_argument("--file", type=str, default="electron_trajectory.npy", help="The numpy trajectory file to plot.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: {args.file} not found. Please run the simulation first.")
        return

    print(f"Loading trajectory data from {args.file}...")
    history = np.load(args.file)
    
    # Shape should be (Ticks, Particles, 10D State)
    ticks = history.shape[0]
    num_particles = history.shape[1]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00'] # Cyberpunk/Sci-fi colors
    
    for i in range(num_particles):
        x = history[:, i, 1]
        y = history[:, i, 2]
        z = history[:, i, 3]
        
        c = colors[i % len(colors)]
        label = f"Particle {i}"
        
        # Plot the main trajectory line
        ax.plot(x, y, z, label=label, color=c, linewidth=2, alpha=0.8)
        
        # Mark the start and end points
        ax.scatter(x[0], y[0], z[0], color=c, marker='o', s=100, label=f"{label} Start")
        ax.scatter(x[-1], y[-1], z[-1], color=c, marker='X', s=150, edgecolors='white', label=f"{label} End")
        
        # Mark the point of closest approach (approximate impact zone)
        if ticks > 1000:
            ax.scatter(x[1000], y[1000], z[1000], color='white', marker='*', s=200, zorder=10)

    # Dark background for cool sci-fi aesthetic
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.grid(color='#333333', linestyle='--', linewidth=0.5)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')

    ax.tick_params(colors='white')
    ax.set_xlabel('X Axis', color='white', labelpad=10)
    ax.set_ylabel('Y Axis', color='white', labelpad=10)
    ax.set_zlabel('Z Axis', color='white', labelpad=10)
    
    ax.set_title('TEGR Particle Collider: High-Energy Scattering', color='white', pad=20, size=15)
    
    legend = ax.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white')
    
    output_filename = args.file.replace(".npy", "_plot.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot successfully saved to {output_filename}!")

if __name__ == "__main__":
    main()
