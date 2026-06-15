import numpy as np
import pysindy as ps
import requests
import json
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Extract universal physics math from simulation via PySINDy and Ollama.")
    parser.add_argument("--model", type=str, default="llama3.1:latest", help="The Ollama model to use for translation (e.g., llama3, mistral).")
    parser.add_argument("--file", type=str, default="tegr_trajectory.npy", help="The numpy trajectory file to analyze.")
    parser.add_argument("--context", type=str, default="galactic core collision", help="The physical context of the data (e.g., electron Møller scattering).")
    args = parser.parse_args()

    print(f"Loading Trajectory Data from {args.file}...")
    try:
        history = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: {args.file} not found. Please run the simulation first.")
        sys.exit(1)
        
    print(f"Data loaded successfully! Shape: {history.shape}")
    
    tracker_idx = 0
    
    # --- THE TIME-SLICER ---
    # Detect the AMPS Firewall singularity (when the core mass instantly triples)
    # We must truncate the data before the singularity so STLSQ doesn't try to calculate an infinite derivative
    m0_history = history[:, tracker_idx, 7]
    mass_diffs = np.abs(np.diff(m0_history))
    spike_indices = np.where(mass_diffs > 0.1)[0]
    
    if len(spike_indices) > 0:
        firewall_index = spike_indices[0]
        print(f"*** SINGULARITY DETECTED AT TICK {firewall_index} ***")
        print("Slicing history array to isolate the INFALL PHASE (Before the firewall)...")
        # Keep the trajectory up to the singularity to discover the pristine runaway tension loop
        history = history[:firewall_index]
    else:
        print("No topological yielding detected. Analyzing full continuous trajectory.")
        
    print(f"Extracting Force/Acceleration vectors for Tracker Particle {tracker_idx}...")
    
    # 1. Extract Positions
    x = history[:, tracker_idx, 1]
    y = history[:, tracker_idx, 2]
    z = history[:, tracker_idx, 3]
    
    # 2. Extract Velocities from Relativistic Momentum
    # 2. Extract Velocities from Relativistic Momentum
    x = history[:, tracker_idx, 1]
    y = history[:, tracker_idx, 2]
    z = history[:, tracker_idx, 3]
    r = history[:, tracker_idx, 6]
    
    hue = np.unwrap(history[:, tracker_idx, 8]) # Unwrap phase
    gamma = history[:, tracker_idx, 9]
    m0 = history[:, tracker_idx, 7]
    
    px = history[:, tracker_idx, 4]
    py = history[:, tracker_idx, 5]
    pz = history[:, tracker_idx, 10] if history.shape[2] > 10 else np.zeros_like(px)
    
    # --- PRUNED FEATURE LIBRARY ---
    # Only 7 physical variables. No exotic inverses.
    # With degree=2, this produces 36 library functions (1 + 7 + 28),
    # well within the data budget even for short infall windows (~22 frames).
    # The previous 11-feature library generated 78 functions, making the system
    # catastrophically underdetermined and forcing STLSQ to hallucinate.
    X_features = np.column_stack((x, y, z, r, hue, gamma, m0))
    feature_names = ['x', 'y', 'z', 'r', 'hue', 'gamma', 'm0']
    
    vx = px / (gamma * m0)
    vy = py / (gamma * m0)
    vz = pz / (gamma * m0)
    
    # 3. Calculate Radius and its derivative (r_dot)
    r_dot = (x * vx + y * vy + z * vz) / (r + 1e-6)
    
    # 4. Construct Phase Space Arrays
    X_dots = np.column_stack((vx, vy, vz, r_dot)) # We will compute the rest numerically
    
    # 5. Numerically Calculate Accelerations (Second Derivatives)
    # The timeline simulation saves state every SAVE_INTERVAL (1 tick)
    # The simulation DT is 0.001. So each history frame is 0.001 seconds apart.
    dt = 0.001
    time_array = np.arange(x.shape[0]) * dt
    fd = ps.FiniteDifference()
    
    # We need to differentiate hue, gamma, m0 as well
    internal_vars = np.column_stack((hue, gamma, m0))
    internal_dots = fd._differentiate(internal_vars, t=dt)
    
    X_ddots = fd._differentiate(np.column_stack((vx, vy, vz, r_dot)), t=dt)
    
    # Target values to predict: vx', vy', vz', r', hue', gamma', m0'
    X_targets = np.column_stack((X_ddots, internal_dots))
    print(f"Target m0' shape: {X_targets[:, 6].shape}, values: {X_targets[:, 6][:10]}")
    
    print("\nBuilding Universal Physics Library (Polynomials, Fourier Waves, Inverse Laws)...")
    
    # Universal Physics Library
    generalized_library = ps.PolynomialLibrary(degree=2, include_bias=True)
    
    # STLSQ threshold raised to 0.1 to kill noise-floor terms.
    # At 0.005, any coefficient > 0.5% survives — pure noise for short datasets.
    # At 0.1, only dominant physics with strong coefficients survive.
    optimizer = ps.STLSQ(threshold=0.1, alpha=0.0)
    
    model = ps.SINDy(
        feature_library=generalized_library, 
        optimizer=optimizer
    )
    
    print("Fitting SINDy Universal Model (Acceleration vs Phase Space). This might take 30-60 seconds...")
    model.fit(X_features, t=dt, x_dot=X_targets, feature_names=feature_names)
    
    print("\n========================================")
    print("PySINDy Extraction Complete!")
    print("Raw Governing Force Equations:")
    model.print()
    print("========================================\n")
    
    equations = [eq for eq in model.equations()]
    lhs = ["vx'", "vy'", "vz'", "r''", "hue'", "gamma'", "m0'"]
    raw_math = "\n".join([f"{lhs[i]} = {eq}" for i, eq in enumerate(equations)])
    
    print("Cleaned Math Equations:")
    print(raw_math)
    
    print(f"Piping Universal Physics output to local Ollama API ({args.model})...")
    
    prompt = f"""
Act as a world-class theoretical physicist analyzing mathematical data. 
I have built a simulation of a particle collision system in a Weitzenböck lattice (a Teleparallel equivalent to General Relativity). 

I used the PySINDy algorithm to extract the governing differential equations directly from the raw trajectory of a {args.context}.

CRITICAL SYSTEM KNOWLEDGE:
- This is a 10-dimensional Weitzenböck lattice simulation modeling fundamental particles not as point masses, but as resonant wave defects in a teleparallel vacuum.
- The connection medium has a strain limit (the "Schwinger Pair-Production Limit"). When the relativistic tension differential between particles (\\Delta\\gamma = |\\gamma_A - \\gamma_B|) exceeds 4.0, the topology structurally yields.
- Upon yielding, an AMPS Firewall event is triggered, snapping the entanglement bond and depositing extreme kinetic strain as a core rest-mass spike (m_0 -> 3m_0).
- NEVER guess or invent physics relationships. Only synthesize what is actually proven in the equations below. If an equation for a variable is `0.000`, the variable is dead/static.


EXACT VARIABLE MAPPINGS:
- `x` or `x0`: Spatial X coordinate.
- `y` or `x1`: Spatial Y coordinate.
- `z` or `x2`: Spatial Z coordinate.
- `r` or `x3`: Radial distance from origin.
- `hue` or `x4`: Internal phase angle / oscillation frequency (\\theta_hue).
- `gamma` or `x5`: Relativistic tension / Time dilation factor (\\gamma).
- `m0` or `x6`: Rest mass / Core energy density (m_0).

HERE ARE THE PYSINDY DIFFERENTIAL EQUATIONS FOR THE SYSTEM:
- The left-hand side of the equations (x', y', z', r', hue', gamma', m0') are NOT velocities. They are ACCELERATIONS (or the second derivatives of the structural state) extracted from the exact moment leading up to a Topological Yield.

Here are the raw mathematical rules PySINDy discovered governing the collision:

{raw_math}

INSTRUCTIONS FOR TRANSLATION:
1. Decode this mathematical output strictly using the TEGR / ER=EPR framework provided.
2. Specifically analyze how the internal variables (`hue`, `gamma`, `m0`) behave. How do phase and tension gradients distribute the shockwave before the topology snaps?
3. What do the specific extracted mathematical dependencies (like `1/r^2`, `x`, etc.) tell us about the geometry of the wormhole connection leading to the AMPS Firewall?
4. Do NOT use generic terms or spoon-fed answers. Synthesize a profound, original theoretical physics conclusion about what this specific mathematical matrix reveals regarding teleparallel gravity and topological yielding.
"""

    payload = {
        "model": "deepseek-r1:14b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        print("Waiting for Ollama to synthesize the laws of physics...")
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\n========================================")
            print("OLLAMA TRANSLATION:")
            print("========================================\n")
            print(result.get("response", "No response text found."))
            
            # Hardware & Token Usage Metrics
            eval_count = result.get("eval_count", 0)
            prompt_eval_count = result.get("prompt_eval_count", 0)
            print("\n--- LLM RESOURCE USAGE ---")
            print(f"Prompt Tokens    : {prompt_eval_count}")
            print(f"Generated Tokens : {eval_count}")
            print(f"Total Tokens     : {eval_count + prompt_eval_count}")
            
            print("\n========================================")
        else:
            print(f"Ollama API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Ollama.")

if __name__ == "__main__":
    main()
