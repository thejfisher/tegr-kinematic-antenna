import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_histogram(json_path="double_slit_results.json", output_path="y_histogram.png"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    y_hits = data.get("y_positions", [])
    if not y_hits:
        print("No screen hits found in JSON.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Use 80 bins across the +/- 10 range to get high-res visibility of the fringes
    bins = np.linspace(-10, 10, 80) 
    
    plt.hist(y_hits, bins=bins, color='cyan', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Calculate predicted fringe locations based on the geometry
    wavelength = 2.756
    screen_x = 15.0
    slit_sep = 8.0
    dy = (wavelength * screen_x) / slit_sep
    
    for n in [-2, -1, 0, 1, 2]:
        x_val = n * dy
        if abs(x_val) <= 10:
            plt.axvline(x=x_val, color='red', linestyle='--', alpha=0.6, 
                        label='Predicted Maxima (Equation)' if n==0 else "")
            
    plt.title(f"1D Y-Histogram: {len(y_hits)} Screen Hits (d={slit_sep}, D={screen_x})", fontsize=14, color='white')
    plt.xlabel("Screen Y Position", fontsize=12, color='white')
    plt.ylabel("Particle Count", fontsize=12, color='white')
    
    # Dark theme styling to match your GUI
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#1e1e1e')
    plt.gca().tick_params(colors='white')
    for spine in plt.gca().spines.values():
        spine.set_color('#444444')
    
    plt.grid(color='#444444', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#1e1e1e', edgecolor='#444444', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1e1e1e')
    print(f"Saved high-res histogram to {output_path} with predicted fringe overlays.")

if __name__ == "__main__":
    plot_histogram()
