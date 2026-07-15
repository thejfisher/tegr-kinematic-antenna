import numpy as np

def analyze_shockwave(file_path, label):
    data = np.load(file_path)
    T, N, _ = data.shape
    px = data[:, :, 4]
    
    shock_times = []
    positions = []
    
    threshold = 0.5  # Momentum threshold
    
    for i in range(1, N):
        moving = np.where(np.abs(px[:, i]) > threshold)[0]
        if len(moving) > 0:
            shock_times.append(moving[0])
            positions.append(i)
            
    if len(shock_times) < 2:
        print(f"{label}: Not enough particles moved! Try a different threshold or longer run.")
        return
        
    shock_times = np.array(shock_times)
    positions = np.array(positions)
    
    # Linear fit to get propagation speed in particles/snapshot
    slope, intercept = np.polyfit(shock_times, positions, 1)
    
    # physical speed = slope (particles/snapshot) * 0.1 (space units) / 0.001 (time units)
    speed = slope * 100.0
    
    print(f"[{label}] Shockwave Propagation Speed: {speed:.2f} c")
    print(f"[{label}] Penetration depth: {len(shock_times)} out of {N-1}")
    
    # Optional: calculate average time delay between adjacent particles
    if len(shock_times) > 1:
        delays = np.diff(shock_times)
        avg_delay = np.mean(delays)
        print(f"[{label}] Average snapshot delay per particle: {avg_delay:.2f} snapshots")

if __name__ == "__main__":
    try:
        analyze_shockwave("classical_trajectory.npy", "Classical Stick")
    except:
        pass
    try:
        analyze_shockwave("entangled_trajectory.npy", "Entangled Stick")
    except:
        pass
