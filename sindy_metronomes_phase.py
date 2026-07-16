import pandas as pd
import numpy as np
import pysindy as ps
import scipy.signal as signal
import sys
import os

def analyze_metronomes(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=2, header=0)
    x_cols = [c for c in df.columns if c.startswith('x')]
    
    t = df['t'].values
    
    # Extract phases
    Theta_list = []
    for col in x_cols[:5]: # 5 metronomes
        x_data = df[col].interpolate(method='linear', limit_direction='both').values
        x_data = x_data - np.mean(x_data)
        
        # Smooth position
        x_data = signal.savgol_filter(x_data, window_length=21, polyorder=3)
        
        # Extract analytic signal and phase
        analytic_signal = signal.hilbert(x_data)
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Smooth phase to avoid numerical derivative spikes
        phase = signal.savgol_filter(phase, window_length=51, polyorder=3)
        
        Theta_list.append(phase)
        
    Theta = np.stack(Theta_list, axis=-1)
    
    print(f"Extracted unwrapped phase for 5 metronomes.")
    
    # Let's just use PySINDy's CustomLibrary
    lib_functions = []
    lib_names = []
    
    # Constants and basic polynomials to test if SINDy correctly rejects them
    lib_functions.append(lambda x0, x1, x2, x3, x4: 1)
    lib_names.append(lambda x0, x1, x2, x3, x4: "1")
    
    # Linear and quadratic terms for the individual phases to see if they are pruned
    for k in range(5):
        def make_poly(idx, power):
            return lambda x0, x1, x2, x3, x4: [x0, x1, x2, x3, x4][idx] ** power
        def make_poly_name(idx, power):
            return lambda x0, x1, x2, x3, x4: f"t_{idx}^{power}" if power > 1 else f"t_{idx}"
            
        lib_functions.append(make_poly(k, 1))
        lib_names.append(make_poly_name(k, 1))
        lib_functions.append(make_poly(k, 2))
        lib_names.append(make_poly_name(k, 2))
    
    # For each metronome i, j
    for i in range(5):
        for j in range(5):
            if i != j:
                def make_func(i_idx, j_idx):
                    return lambda x0, x1, x2, x3, x4: np.sin([x0, x1, x2, x3, x4][j_idx] - [x0, x1, x2, x3, x4][i_idx])
                def make_name(i_idx, j_idx):
                    return lambda x0, x1, x2, x3, x4: f"sin(t_{j_idx} - t_{i_idx})"
                
                lib_functions.append(make_func(i, j))
                lib_names.append(make_name(i, j))
                
    custom_library = ps.CustomLibrary(library_functions=lib_functions, function_names=lib_names)
    
    optimizer = ps.STLSQ(threshold=0.01)
    
    model = ps.SINDy(
        feature_library=custom_library,
        optimizer=optimizer
    )
    
    model.fit(Theta, t=t)
    
    print("\n--- Discovered Governing Equations (Phase) ---")
    model.print()
    
    score = model.score(Theta, t=t)
    print(f"\nModel R^2 Score: {score:.4f}")

if __name__ == "__main__":
    analyze_metronomes("first test.txt")
