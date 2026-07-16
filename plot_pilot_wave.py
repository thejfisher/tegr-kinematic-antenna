import numpy as np
import matplotlib.pyplot as plt

def plot_slice():
    try:
        phi_slice = np.load("pilot_wave_slice.npy")
    except FileNotFoundError:
        print("pilot_wave_slice.npy not found.")
        return

    plt.figure(figsize=(10, 8))
    # We use a diverging colormap centered at zero
    vmax = np.max(np.abs(phi_slice))
    if vmax == 0:
        vmax = 1.0 # Prevent zero-division/warning if grid is empty
        
    plt.imshow(phi_slice, cmap='RdBu', vmin=-vmax, vmax=vmax, origin='lower')
    plt.colorbar(label="Torsional Amplitude ($\phi$)")
    plt.title("Eulerian Pilot Wave Field (z=0 Equatorial Slice)")
    plt.xlabel("X Index")
    plt.ylabel("Y Index")
    plt.tight_layout()
    plt.savefig("pilot_wave_render.png", dpi=150)
    print("Saved pilot_wave_render.png")

if __name__ == "__main__":
    plot_slice()
