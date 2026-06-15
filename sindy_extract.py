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
    parser.add_argument("--model", type=str, default="deepseek-r1:14b", help="The Ollama model to use for translation (e.g., llama3, mistral).")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="The Ollama API base URL.")
    parser.add_argument("--file", type=str, default="tegr_trajectory.npy", help="The numpy trajectory file to analyze.")
    parser.add_argument("--context", type=str, default="teleparallel particle system", help="The physical context of the data (e.g., electron Møller scattering).")
    parser.add_argument("--no_llm", action="store_true", help="Skip the LLM translation step")
    args = parser.parse_args()

    print(f"Loading Trajectory Data from {args.file}...")
    try:
        history = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: {args.file} not found. Please run the simulation first.")
        sys.exit(1)
        
    print(f"Data loaded successfully! Shape: {history.shape}")
    tracker_idx = -1  # Default: last particle (source in antenna mode, beam particle in collider modes)
    
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
    
    # Extract positions
    x = history[:, tracker_idx, 1]
    y = history[:, tracker_idx, 2]
    z = history[:, tracker_idx, 3]
    
    # Compute ACTUAL radial distance (was previously reading pz by mistake)
    # State vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
    r = np.sqrt(x**2 + y**2 + z**2 + 1e-12)  # Radial distance from origin
    
    hue = np.unwrap(history[:, tracker_idx, 8]) # Unwrap phase
    gamma = history[:, tracker_idx, 9]
    m0 = history[:, tracker_idx, 7]
    
    px = history[:, tracker_idx, 4]
    py = history[:, tracker_idx, 5]
    pz = history[:, tracker_idx, 6]  # Index 6 = pz (was incorrectly used as 'r' before)
    
    if np.all(gamma == 0.0):
        print("Warning: gamma channel is 0.0. Recomputing gamma from momentum...")
        p_sq = px**2 + py**2 + pz**2
        gamma = np.sqrt(1.0 + p_sq / (m0**2 * 1000.0**2))
    # --- BASE FEATURE LIBRARY ---
    inv_r3 = np.clip(1.0 / (r**3 + 1e-12), -1e6, 1e6)
    
    # [EMERGENT VALIDATION] Explicitly include the Proper Time kinematic vocabulary.
    # Without this division logic, PySINDy triggers massive Taylor expansion coefficients
    # attempting to fit the relativistic clock. This feature resolves collinearity at gamma ~ 1.
    m0_over_gamma = m0 / (gamma + 1e-12)
    
    sin_hue = np.sin(hue)
    cos_hue = np.cos(hue)
    X_features = np.column_stack((x, y, z, r, hue, gamma, m0, inv_r3, sin_hue, cos_hue, m0_over_gamma))
    feature_names = ['x', 'y', 'z', 'r', 'hue', 'gamma', 'm0', '1/r^3', 'sin(hue)', 'cos(hue)', 'm0/gamma']
    
    vx = px / (gamma * m0)
    vy = py / (gamma * m0)
    vz = pz / (gamma * m0)
    
    r_dot = (x * vx + y * vy + z * vz) / (r + 1e-6)
    
    # Numerically compute accelerations (second derivatives)
    dt = 0.001
    fd = ps.FiniteDifference()
    
    internal_vars = np.column_stack((hue, gamma, m0))
    internal_dots = fd._differentiate(internal_vars, t=dt)
    
    X_ddots = fd._differentiate(np.column_stack((vx, vy, vz, r_dot)), t=dt)
    
    # Target values to predict: vx', vy', vz', r', hue', gamma', m0'
    X_targets = np.column_stack((X_ddots, internal_dots))
    print(f"Target m0' shape: {X_targets[:, 6].shape}, values: {X_targets[:, 6][:10]}")
    
    # --- MULTI-BODY AGGREGATE FEATURES ---
    # When N > 2 (antenna, holographic, etc.), the tracker particle's own
    # coordinates are near-zero and useless. We must compute the NET FIELD
    # from all other particles — these are the actual physical inputs driving
    # the tracker's acceleration per the Paper 1 force law:
    #   F = chi * cos(d_hue) * (w_i . w_j) / r^3 * r_hat
    N_particles = history.shape[1]
    tracker_actual_idx = tracker_idx % N_particles  # Handle negative index
    
    # --- 2-BODY INTER-PARTICLE SEPARATION ---
    # For N=2 (head-on, gravity-sink), compute the true separation vector
    # between the two particles. This gives SINDy the actual inter-particle
    # distance to discover Pauli exchange (1/d^3), Coulomb (1/d^2), or
    # confinement (d) force laws — not just distance from coordinate origin.
    if N_particles == 2:
        other_idx = 0 if tracker_actual_idx == 1 else 1
        other_pos = history[:, other_idx, 1:4]  # (T, 3)
        tracker_pos_2b = history[:, tracker_actual_idx, 1:4]  # (T, 3)
        other_hue = history[:, other_idx, 8]
        
        sep_vec = tracker_pos_2b - other_pos  # separation vector
        d12 = np.sqrt(sep_vec[:, 0]**2 + sep_vec[:, 1]**2 + sep_vec[:, 2]**2 + 1e-12)
        inv_d12_2 = np.clip(1.0 / (d12**2 + 1e-12), -1e6, 1e6)
        inv_d12_3 = np.clip(1.0 / (d12**3 + 1e-12), -1e6, 1e6)
        d_hue = np.unwrap(history[:, tracker_actual_idx, 8]) - np.unwrap(other_hue)
        cos_dhue = np.cos(d_hue)
        
        X_features = np.column_stack((X_features, d12, inv_d12_2, inv_d12_3, cos_dhue))
        feature_names += ['d12', '1/d12^2', '1/d12^3', 'cos(d_hue)']
        print(f"2-body mode: added inter-particle features (d12, 1/d12^2, 1/d12^3, cos(d_hue))")
        print(f"  Separation range: [{d12.min():.4f}, {d12.max():.4f}]")

    if N_particles > 2:
        print(f"Multi-body mode detected (N={N_particles}). Computing net field at tracker...")
        tracker_pos = history[:, tracker_actual_idx, 1:4]  # (T, 3)
        tracker_hue_raw = history[:, tracker_actual_idx, 8]  # (T,)
        tracker_mom = history[:, tracker_actual_idx, 4:7]   # (T, 3)
        tracker_gamma = history[:, tracker_actual_idx, 9]    # (T,)
        tracker_m0_arr = history[:, tracker_actual_idx, 7]   # (T,)
        tracker_vel = tracker_mom / (tracker_gamma[:, np.newaxis] * tracker_m0_arr[:, np.newaxis] + 1e-12)
        
        SPIN_VAL = 0.5  # All particles spin=0.5 in antenna mode
        SPIN_DOT = SPIN_VAL * SPIN_VAL  # = 0.25
        
        net_sync = np.zeros(len(tracker_hue_raw))   # Kuramoto sync torque
        net_fx = np.zeros(len(tracker_hue_raw))      # Pauli force (engine-matched)
        net_fy = np.zeros(len(tracker_hue_raw))
        net_fz = np.zeros(len(tracker_hue_raw))
        net_tx = np.zeros(len(tracker_hue_raw))      # Torsion force
        net_ty = np.zeros(len(tracker_hue_raw))
        net_tz = np.zeros(len(tracker_hue_raw))
        
        for j in range(N_particles):
            if j == tracker_actual_idx:
                continue
            other_pos = history[:, j, 1:4]
            other_hue = history[:, j, 8]
            other_mom = history[:, j, 4:7]
            other_gamma = history[:, j, 9]
            other_m0_arr = history[:, j, 7]
            other_vel = other_mom / (other_gamma[:, np.newaxis] * other_m0_arr[:, np.newaxis] + 1e-12)
            
            dx = tracker_pos[:, 0] - other_pos[:, 0]
            dy = tracker_pos[:, 1] - other_pos[:, 1]
            dz = tracker_pos[:, 2] - other_pos[:, 2]
            d_hue = tracker_hue_raw - other_hue
            dist_sq = dx**2 + dy**2 + dz**2 + 1e-6  # Matches engine's 1e-6 softening
            
            # Kuramoto sync: sum of sin(d_hue)
            net_sync += np.sin(d_hue)
            
            # --- PAULI FORCE (Engine-matched: line 667) ---
            # F = PAULI_SCALAR * cos(dhue) * spin_dot * diff / dist_sq^1.5
            # power_exp = (2+1)/2 = 1.5, so dist_sq**1.5 = r^3
            pauli_coupling = SPIN_DOT * np.cos(d_hue) / (dist_sq**1.5)
            net_fx += pauli_coupling * dx
            net_fy += pauli_coupling * dy
            net_fz += pauli_coupling * dz
            
            # --- TORSION FORCE (Engine-matched: line 626) ---
            # F_torsion = TORSION_G * diff * cross(diff, v_diff) / dist_sq^2
            dvx = tracker_vel[:, 0] - other_vel[:, 0]
            dvy = tracker_vel[:, 1] - other_vel[:, 1]
            dvz = tracker_vel[:, 2] - other_vel[:, 2]
            # cross(diff, v_diff)
            cx = dy * dvz - dz * dvy
            cy = dz * dvx - dx * dvz
            cz = dx * dvy - dy * dvx
            # Element-wise: diff * cross / dist_sq^2
            tor_denom = dist_sq**2.0
            net_tx += dx * cx / tor_denom
            net_ty += dy * cy / tor_denom
            net_tz += dz * cz / tor_denom
        # Collapse torsion into scalar magnitude to break multicollinearity
        torsion_mag = np.sqrt(net_tx**2 + net_ty**2 + net_tz**2)
        
        X_features = np.column_stack((X_features, net_sync, net_fx, net_fy, net_fz, torsion_mag))
        feature_names += ['K_sync', 'F_pauli_x', 'F_pauli_y', 'F_pauli_z', 'F_torsion']
        print(f"  Max |K_sync|:    {np.max(np.abs(net_sync)):.4f}")
        pauli_mag = np.sqrt(net_fx**2 + net_fy**2 + net_fz**2)
        print(f"  Max |F_pauli|:   {np.max(pauli_mag):.6f}")
        print(f"  Max |F_torsion|: {np.max(torsion_mag):.6f}")
    
    # --- ALIGN ROW COUNTS ---
    # FiniteDifference drops boundary rows. Trim X_features to match X_targets.
    n_targets = X_targets.shape[0]
    n_features = X_features.shape[0]
    if n_features > n_targets:
        trim = n_features - n_targets
        front = trim // 2
        X_features = X_features[front:front + n_targets]
    
    print(f"\nBuilding Universal Physics Library (Polynomials, Fourier Waves, Inverse Laws)...")
    print(f"Max |vx'|: {np.max(np.abs(X_targets[:, 0])):.6f}")
    print(f"Max |vy'|: {np.max(np.abs(X_targets[:, 1])):.6f}")
    print(f"Max |vz'|: {np.max(np.abs(X_targets[:, 2])):.6f}")
    
    # Universal Physics Library
    generalized_library = ps.PolynomialLibrary(degree=1, include_bias=True)

    # STLSQ threshold at 0.005 — low enough to capture faint couplings,
    # high enough to suppress pure numerical noise
    optimizer = ps.STLSQ(threshold=0.005, alpha=0.001)
    
    model = ps.SINDy(
        feature_library=generalized_library, 
        optimizer=optimizer
    )
    
    print("Filtering NaNs from trajectory...")
    valid_mask = np.isfinite(X_features).all(axis=1) & np.isfinite(X_targets).all(axis=1)
    X_features = X_features[valid_mask]
    X_targets = X_targets[valid_mask]
    
    print(f"Fitting SINDy Universal Model ({valid_mask.sum()} valid frames). This might take 30-60 seconds...")
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
    
    try:
        r2_score = model.score(X_features, t=dt, x_dot=X_targets)
        print(f"\n[Validation] Overall R^2 Score: {r2_score:.6f}")
        
        # Per-equation R² to identify which equations drag the score down
        X_pred = model.predict(X_features)
        # Trim to match prediction length
        X_targets_trim = X_targets[:X_pred.shape[0]]
        for i, name in enumerate(lhs):
            ss_res = np.sum((X_targets_trim[:, i] - X_pred[:, i])**2)
            ss_tot = np.sum((X_targets_trim[:, i] - np.mean(X_targets_trim[:, i]))**2)
            r2_i = 1.0 - ss_res / (ss_tot + 1e-12)
            print(f"  {name:>8s} R^2 = {r2_i:.4f}")
    except Exception as e:
        print(f"\n[Validation] Could not compute R^2 Score: {e}")
    
    print(f"\nPiping Universal Physics output to local Ollama API ({args.model})...")
    
    prompt = f"""
Act as a world-class theoretical physicist analyzing mathematical data. 
I have built a simulation of a particle collision system in a Weitzenböck lattice (a Teleparallel equivalent to General Relativity). 

I used the PySINDy algorithm to extract the governing differential equations directly from the raw trajectory of a {args.context}.

CRITICAL SYSTEM KNOWLEDGE:
- This is a bare-metal test of a 10-dimensional Weitzenböck lattice simulation.
- All hardcoded phenomenological rules (like forced mass-spikes or forced phase synchronization) have been REMOVED.
- You are analyzing the PURE emergent dynamics from the spatial connection. 
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

    if args.no_llm:
        print("\n--- Skipping Ollama Translation (--no_llm flag active) ---")
        return

    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False
    }
    
    # Format the URL properly
    api_url = args.url.rstrip('/') + "/api/generate"
    
    try:
        print(f"Waiting for Ollama to synthesize the laws of physics from {api_url}...")
        response = requests.post(api_url, json=payload)
        
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
