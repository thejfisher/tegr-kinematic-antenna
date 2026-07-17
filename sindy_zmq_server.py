import zmq
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

import csv
import os

def run_sindy_and_ollama(history, ai_url, ai_model, no_llm=False, run_label="default", galactic_spin=0.0, dt=0.001):
    print(f"Data loaded from ZMQ Stream! Shape: {history.shape}")
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vram_gb = history.nbytes / 1e9
        if device.type == 'cuda':
            print(f"--- HARDWARE ACCELERATION ENABLED ---")
            print(f"Pushing {vram_gb:.2f} GB tensor directly into AMD r9700 VRAM...")
        else:
            print(f"Warning: CUDA/ROCm not detected. Using DRAM ({vram_gb:.2f} GB).")
    except ImportError:
        print("PyTorch not installed. Using standard DRAM numpy arrays.")
        device = "cpu"

    # Always track beam particle (index 0) — the active resonant wave defect.
    # In double-slit mode, index 0 = beam, indices 1..N = stationary wall.
    # In 2-body mode, index 0 = source particle.
    # galactic_spin mode also uses index 0 (anchor).
    tracker_idx = 0
    if "String-Sink" in run_label:
        tracker_idx = 1 # Track the first particle of the string, not the gravity sink
    
    m0_history = history[:, tracker_idx, 7]
    mass_diffs = np.abs(np.diff(m0_history))
    spike_indices = np.where(mass_diffs > 0.1)[0]
    
    if len(spike_indices) > 0:
        firewall_index = spike_indices[0]
        print(f"*** SINGULARITY DETECTED AT TICK {firewall_index} ***")
        history = history[:firewall_index]
    else:
        print("No topological yielding detected. Analyzing full continuous trajectory.")
        
    print(f"Extracting Force/Acceleration vectors for Tracker Particle {tracker_idx}...")
    
    # Raw lab frame kinematics
    x_raw = history[:, tracker_idx, 1]
    y_raw = history[:, tracker_idx, 2]
    z = history[:, tracker_idx, 3]
    px_raw = history[:, tracker_idx, 4]
    py_raw = history[:, tracker_idx, 5]
    pz = history[:, tracker_idx, 6]
    
    gamma = history[:, tracker_idx, 9]
    m0 = history[:, tracker_idx, 7]
    hue = np.unwrap(history[:, tracker_idx, 8])
    
    # dt is received as function parameter from collider via ZMQ
    N_ticks = len(x_raw)
    t = np.arange(N_ticks) * dt
    
    if galactic_spin > 0.0:
        omega = galactic_spin
        print(f"Applying co-rotating frame transformation (omega = {omega})...")
        cos_wt = np.cos(omega * t)
        sin_wt = np.sin(omega * t)
        
        # Transform spatial coordinates
        x = x_raw * cos_wt + y_raw * sin_wt
        y = -x_raw * sin_wt + y_raw * cos_wt
        
        # Transform momentum coordinates
        px = px_raw * cos_wt + py_raw * sin_wt
        py = -px_raw * sin_wt + py_raw * cos_wt
    else:
        x = x_raw
        y = y_raw
        px = px_raw
        py = py_raw

    r = np.sqrt(x**2 + y**2 + z**2 + 1e-12)
    
    if np.all(gamma == 0.0):
        p_sq = px_raw**2 + py_raw**2 + pz**2
        gamma = np.sqrt(1.0 + p_sq / (m0**2 * 1000.0**2))
        
    inv_r3 = np.clip(1.0 / (r**3 + 1e-12), -1e6, 1e6)
    inv_r2 = np.clip(1.0 / (r**2 + 1e-12), -1e6, 1e6)
    m0_over_gamma = m0 / (gamma + 1e-12)
    sin_hue = np.sin(hue)
    cos_hue = np.cos(hue)
    
    x_over_r3 = x * inv_r3
    y_over_r3 = y * inv_r3
    z_over_r3 = z * inv_r3
    
    X_features = np.column_stack((x, y, z, r, hue, gamma, m0, inv_r3, inv_r2, sin_hue, cos_hue, m0_over_gamma, x_over_r3, y_over_r3, z_over_r3))
    feature_names = ['x', 'y', 'z', 'r', 'hue', 'gamma', 'm0', '1/r^3', '1/r^2', 'sin(hue)', 'cos(hue)', 'm0/gamma', 'x/r^3', 'y/r^3', 'z/r^3']
    
    if history.shape[2] >= 12:
        grad_phi = history[:, tracker_idx, 10]
        kappa = history[:, tracker_idx, 11]
        X_features = np.column_stack((X_features, grad_phi, kappa))
        feature_names.extend(['gradPhi', 'kappa'])
    
    vx = px / (gamma * m0)
    vy = py / (gamma * m0)
    vz = pz / (gamma * m0)
    r_dot = (x * vx + y * vy + z * vz) / (r + 1e-6)
    
    # dt is now received from the collider via ZMQ DONE message.
    # Fallback to 0.001 if not provided (legacy compatibility).
    print(f"Using dt = {dt} for finite difference computation.")
    fd = ps.FiniteDifference()
    
    # Unwrap hue to prevent artificial numerical spikes in the derivative when crossing 2*pi
    unwrapped_hue = np.unwrap(hue)
    internal_vars = np.column_stack((unwrapped_hue, gamma, m0))
    internal_dots = fd._differentiate(internal_vars, t=dt)
    X_ddots = fd._differentiate(np.column_stack((vx, vy, vz, r_dot)), t=dt)
    X_targets = np.column_stack((X_ddots, internal_dots))
    
    N_particles = history.shape[1]
    # For N=2 (string-sink), we usually tracked particle 1 (sink).
    # For N>2 (double slit), the beam particle is always at index 0.
    if N_particles > 2:
        tracker_actual_idx = 0
    else:
        tracker_actual_idx = tracker_idx % N_particles
    
    if N_particles == 2:
        other_idx = 0 if tracker_actual_idx == 1 else 1
        other_pos = history[:, other_idx, 1:4]
        tracker_pos_2b = history[:, tracker_actual_idx, 1:4]
        other_hue = history[:, other_idx, 8]
        
        sep_vec = tracker_pos_2b - other_pos
        d12 = np.sqrt(sep_vec[:, 0]**2 + sep_vec[:, 1]**2 + sep_vec[:, 2]**2 + 1e-12)
        inv_d12_2 = np.clip(1.0 / (d12**2 + 1e-12), -1e6, 1e6)
        inv_d12_3 = np.clip(1.0 / (d12**3 + 1e-12), -1e6, 1e6)
        d_hue = np.unwrap(history[:, tracker_actual_idx, 8]) - np.unwrap(other_hue)
        cos_dhue = np.cos(d_hue)
        
        X_features = np.column_stack((X_features, d12, inv_d12_2, inv_d12_3, cos_dhue))
        feature_names += ['d12', '1/d12^2', '1/d12^3', 'cos(d_hue)']

    if N_particles > 2:
        tracker_pos = history[:, tracker_actual_idx, 1:4]
        tracker_hue_raw = history[:, tracker_actual_idx, 8]
        tracker_mom = history[:, tracker_actual_idx, 4:7]
        tracker_gamma = history[:, tracker_actual_idx, 9]
        tracker_m0_arr = history[:, tracker_actual_idx, 7]
        tracker_vel = tracker_mom / (tracker_gamma[:, np.newaxis] * tracker_m0_arr[:, np.newaxis] + 1e-12)
        
        SPIN_DOT = 0.25
        net_sync = np.zeros(len(tracker_hue_raw))
        net_fx = np.zeros(len(tracker_hue_raw))
        net_fy = np.zeros(len(tracker_hue_raw))
        net_fz = np.zeros(len(tracker_hue_raw))
        net_tx = np.zeros(len(tracker_hue_raw))
        net_ty = np.zeros(len(tracker_hue_raw))
        net_tz = np.zeros(len(tracker_hue_raw))
        
        # Use PyTorch VRAM if available for the massive O(N^2) tensor calculations
        if 'torch' in sys.modules and device.type == 'cuda':
            print("Accelerating N-body Pauli & Torsion feature construction in VRAM...")
            tracker_pos_t = torch.tensor(tracker_pos, device=device, dtype=torch.float32)
            tracker_hue_t = torch.tensor(tracker_hue_raw, device=device, dtype=torch.float32)
            tracker_vel_t = torch.tensor(tracker_vel, device=device, dtype=torch.float32)
            
            net_sync_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_fx_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_fy_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_fz_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_tx_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_ty_t = torch.zeros(len(tracker_hue_raw), device=device)
            net_tz_t = torch.zeros(len(tracker_hue_raw), device=device)
            
            for j in range(N_particles):
                if j == tracker_actual_idx:
                    continue
                other_pos_t = torch.tensor(history[:, j, 1:4], device=device, dtype=torch.float32)
                other_hue_t = torch.tensor(history[:, j, 8], device=device, dtype=torch.float32)
                
                other_mom = history[:, j, 4:7]
                other_gamma = history[:, j, 9]
                other_m0_arr = history[:, j, 7]
                other_vel = other_mom / (other_gamma[:, np.newaxis] * other_m0_arr[:, np.newaxis] + 1e-12)
                other_vel_t = torch.tensor(other_vel, device=device, dtype=torch.float32)
                
                dx = tracker_pos_t[:, 0] - other_pos_t[:, 0]
                dy = tracker_pos_t[:, 1] - other_pos_t[:, 1]
                dz = tracker_pos_t[:, 2] - other_pos_t[:, 2]
                d_hue = tracker_hue_t - other_hue_t
                dist_sq = dx**2 + dy**2 + dz**2 + 1e-6
                
                net_sync_t += torch.sin(d_hue)
                pauli_coupling = SPIN_DOT * torch.cos(d_hue) / (dist_sq**1.5)
                net_fx_t += pauli_coupling * dx
                net_fy_t += pauli_coupling * dy
                net_fz_t += pauli_coupling * dz
                
                dvx = tracker_vel_t[:, 0] - other_vel_t[:, 0]
                dvy = tracker_vel_t[:, 1] - other_vel_t[:, 1]
                dvz = tracker_vel_t[:, 2] - other_vel_t[:, 2]
                cx = dy * dvz - dz * dvy
                cy = dz * dvx - dx * dvz
                cz = dx * dvy - dy * dvx
                tor_denom = dist_sq**2.0
                net_tx_t += dx * cx / tor_denom
                net_ty_t += dy * cy / tor_denom
                net_tz_t += dz * cz / tor_denom
                
            net_sync = net_sync_t.cpu().numpy()
            net_fx = net_fx_t.cpu().numpy()
            net_fy = net_fy_t.cpu().numpy()
            net_fz = net_fz_t.cpu().numpy()
            net_tx = net_tx_t.cpu().numpy()
            net_ty = net_ty_t.cpu().numpy()
            net_tz = net_tz_t.cpu().numpy()
        else:
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
                dist_sq = dx**2 + dy**2 + dz**2 + 1e-6
                
                net_sync += np.sin(d_hue)
                pauli_coupling = SPIN_DOT * np.cos(d_hue) / (dist_sq**1.5)
                net_fx += pauli_coupling * dx
                net_fy += pauli_coupling * dy
                net_fz += pauli_coupling * dz
                
                dvx = tracker_vel[:, 0] - other_vel[:, 0]
                dvy = tracker_vel[:, 1] - other_vel[:, 1]
                dvz = tracker_vel[:, 2] - other_vel[:, 2]
                cx = dy * dvz - dz * dvy
                cy = dz * dvx - dx * dvz
                cz = dx * dvy - dy * dvx
                tor_denom = dist_sq**2.0
                net_tx += dx * cx / tor_denom
                net_ty += dy * cy / tor_denom
                net_tz += dz * cz / tor_denom
            
        torsion_mag = np.sqrt(net_tx**2 + net_ty**2 + net_tz**2)
        X_features = np.column_stack((X_features, net_sync, net_fx, net_fy, net_fz, torsion_mag))
        feature_names += ['K_sync', 'F_pauli_x', 'F_pauli_y', 'F_pauli_z', 'F_torsion']
    
    n_targets = X_targets.shape[0]
    n_features = X_features.shape[0]
    if n_features > n_targets:
        trim = n_features - n_targets
        print(f"Trimming {trim} elements to match targets. (Warning: mismatch implies skipped frames)")
        X_features = X_features[trim:]
    elif n_targets > n_features:
        trim = n_targets - n_features
        print(f"Trimming {trim} targets to match features.")
        X_targets = X_targets[trim:]
    
    generalized_library = ps.PolynomialLibrary(degree=1, include_bias=False)
    optimizer = ps.STLSQ(threshold=0.01, alpha=0.1, normalize_columns=True)
    model = ps.SINDy(feature_library=generalized_library, optimizer=optimizer)
    
    valid_mask = np.isfinite(X_features).all(axis=1) & np.isfinite(X_targets).all(axis=1)
    X_features = X_features[valid_mask]
    X_targets = X_targets[valid_mask]
    
    # Remove zero-variance columns (e.g., z=0, m0=const, z/r^3=0 in 2D orbits)
    # These cause SVD failure when normalize_columns=True divides by std=0
    col_var = np.var(X_features, axis=0)
    nonzero_cols = col_var > 1e-12
    if not np.all(nonzero_cols):
        removed = [feature_names[i] for i in range(len(feature_names)) if not nonzero_cols[i]]
        print(f"Removing {len(removed)} zero-variance features: {removed}")
        X_features = X_features[:, nonzero_cols]
        feature_names = [fn for i, fn in enumerate(feature_names) if nonzero_cols[i]]
    
    # Calculate Correlation Matrix
    print(f"Calculating feature correlation matrix...")
    try:
        corr_matrix = np.corrcoef(X_features.T)
        safe_label = "".join([c if c.isalnum() or c in " _-" else "_" for c in run_label])
        corr_file = f"sindy_correlations_{safe_label}.csv"
        with open(corr_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Feature"] + feature_names)
            for i, row in enumerate(corr_matrix):
                writer.writerow([feature_names[i]] + list(row))
        print(f"[BUFFER] Saved correlation matrix to {corr_file}")
        
        # Print Markdown table to console
        print("\nCorrelation Matrix:")
        header = "| Feature | " + " | ".join(feature_names) + " |"
        separator = "|---|" + "|".join(["---" for _ in feature_names]) + "|"
        print(header)
        print(separator)
        for i, row in enumerate(corr_matrix):
            row_str = " | ".join([f"{v:.4f}" for v in row])
            print(f"| {feature_names[i]} | {row_str} |")
        print("\n")
    except Exception as e:
        print(f"Failed to compute or save correlation matrix: {e}")
    
    print(f"Fitting SINDy Universal Model ({valid_mask.sum()} valid frames). This might take 30-60 seconds...")
    model.fit(X_features, t=dt, x_dot=X_targets, feature_names=feature_names)
    
    equations = [eq for eq in model.equations()]
    lhs = ["vx'", "vy'", "vz'", "r''", "hue'", "gamma'", "m0'"]
    # Calculate R^2 scores per variable
    try:
        from sklearn.metrics import r2_score as sklearn_r2
        predictions = model.predict(X_features)
        r2_per_var = sklearn_r2(X_targets, predictions, multioutput='raw_values')
        overall_r2 = model.score(X_features, t=dt, x_dot=X_targets)
    except Exception as e:
        r2_per_var = [0.0] * len(lhs)
        overall_r2 = 0.0
        print(f"Failed to compute R^2: {e}")

    # Build the math string with per-variable R^2
    formatted_eqs = []
    for i, eq in enumerate(equations):
        formatted_eqs.append(f"{lhs[i]} = {eq}   (R^2 = {r2_per_var[i]:.4f})")
    raw_math = "\n\n".join(formatted_eqs)
        
    print("\n========================================")
    print("Cleaned Math Equations:")
    print(raw_math)
    print(f"Overall R^2 Score: {overall_r2:.4f}")
    print("========================================\n")
    
    # Capture sindy_report
    sindy_report = {
        "run_label": run_label,
        "equations": dict(zip(lhs, equations)),
        "r2_score": float(overall_r2)
    }
    
    safe_label = "".join([c if c.isalnum() or c in " _-" else "_" for c in run_label])
    json_filename = f"sindy_report_{safe_label}.json"
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(sindy_report, f, indent=4)
        print(f"[BUFFER] Saved SINDy report to {json_filename}")
    except Exception as e:
        print(f"Failed to write to JSON: {e}")
        
    # Write to CSV
    csv_file = "sindy_results.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["run_label", "r2_score", "vx'", "vy'", "vz'", "r''", "hue'", "gamma'", "m0'"])
            writer.writerow([run_label, overall_r2] + equations)
        print(f"[BUFFER] Saved extracted equations to {csv_file}")
    except Exception as e:
        print(f"Failed to write to CSV: {e}")
    
    if no_llm:
        print("\n[BUFFER] --- SENDING TO OLLAMA ---")
        print("AI translation disabled via --no_llm. Returning raw equations to GUI.")
        return
        
    print(f"\n[BUFFER] --- SENDING TO OLLAMA ---")
    print(f"Piping output to Ollama API at {ai_url} ({ai_model})...")
    
    prompt = f"""
Act as a world-class theoretical physicist analyzing mathematical data. 
I have built a simulation of a particle collision system in a Weitzenböck lattice (a Teleparallel equivalent to General Relativity). 

I used the PySINDy algorithm to extract the governing differential equations directly from the raw trajectory.

CRITICAL SYSTEM KNOWLEDGE:
- This is a bare-metal test of a 10-dimensional Weitzenböck lattice simulation.
- All hardcoded phenomenological rules (like forced mass-spikes or forced phase synchronization) have been REMOVED.
- You are analyzing the PURE emergent dynamics from the spatial connection. 

EXACT VARIABLE MAPPINGS:
- `x` or `x0`: Spatial X coordinate.
- `y` or `x1`: Spatial Y coordinate.
- `z` or `x2`: Spatial Z coordinate.
- `r` or `x3`: Radial distance from origin.
- `hue` or `x4`: Internal phase angle / oscillation frequency (\\theta_hue).
- `gamma` or `x5`: Relativistic tension / Time dilation factor (\\gamma).
- `m0` or `x6`: Rest mass / Core energy density (m_0).

HERE ARE THE PYSINDY DIFFERENTIAL EQUATIONS FOR THE SYSTEM:
- The left-hand side of the equations (x', y', z', r', hue', gamma', m0') are NOT velocities. They are ACCELERATIONS (or the second derivatives of the structural state).

{raw_math}

INSTRUCTIONS FOR TRANSLATION:
1. Decode this mathematical output strictly using the TEGR / ER=EPR framework provided.
2. Specifically analyze how the internal variables (`hue`, `gamma`, `m0`) behave. 
3. Do NOT use generic terms or spoon-fed answers. Synthesize a profound, original theoretical physics conclusion.
"""

    base_url = ai_url.rstrip('/')
    is_openai = "chat/completions" in base_url or "openai" in base_url or "github" in base_url or "azure" in base_url
    
    try:
        if is_openai:
            if not base_url.endswith("/chat/completions") and not base_url.endswith("/completions"):
                api_url = base_url + "/chat/completions"
            else:
                api_url = base_url
                
            payload = {
                "model": ai_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            headers = {"Content-Type": "application/json"}
            
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
            response = requests.post(api_url, json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print("\n========================================")
                print("AI TRANSLATION:")
                print("========================================\n")
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "No response text found.")
                print(content)
            else:
                print(f"API Error: {response.status_code} - {response.text}")
        else:
            payload = {
                "model": ai_model,
                "prompt": prompt,
                "stream": False
            }
            api_url = base_url + "/api/generate"
            
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("\n========================================")
                print("OLLAMA TRANSLATION:")
                print("========================================\n")
                print(result.get("response", "No response text found."))
            else:
                print(f"Ollama API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to AI endpoint.")


def main():
    parser = argparse.ArgumentParser(description="ZeroMQ RAM Buffer & PySINDy Server")
    parser.add_argument("--port", type=int, default=7777, help="ZMQ Port to listen on")
    parser.add_argument("--ai_url", type=str, default="http://127.0.0.1:11434", help="Thejfisher Ollama API URL")
    parser.add_argument("--ai_model", type=str, default="gemma4:e4b", help="Ollama model name")
    parser.add_argument("--no_llm", action="store_true", help="Skip Ollama request and just print equations")
    parser.add_argument("--persistent", action="store_true", help="Do not exit after one run (for stress tests)")
    parser.add_argument("--galactic_spin", type=float, default=0.0, help="Angular velocity of the galactic rotation (co-rotating frame extraction)")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{args.port}")
    
    print(f"Buffer Node (HAL): Listening on port {args.port} for massive RAM tensor chunks...")
    
    chunks = []
    total_frames = 0
    
    while True:
        message = socket.recv_pyobj()
        status = message.get("status")
        
        if status == "STREAMING":
            data_chunk = message.get("data")
            chunks.append(data_chunk)
            total_frames += data_chunk.shape[0]
            if len(chunks) == 1 or len(chunks) % 100 == 0:
                mem_mb = sum(c.nbytes for c in chunks) / (1024**2)
                print(f"[BUFFER] Chunks: {len(chunks)} | Frames: {total_frames} | RAM: {mem_mb:.0f} MB | Latest shape: {data_chunk.shape}")
            
        elif status == "DONE":
            print("Received DONE signal. Stitching DRAM chunks together...")
            if not chunks:
                print("Error: Received DONE but no chunks were sent.")
                break
                
            full_history = np.concatenate(chunks, axis=0)
            print(f"Full trajectory stitched in RAM. Final Shape: {full_history.shape}")
            
            run_label = message.get("run_label", "default")
            actual_dt = message.get("dt", 0.001)
            print(f"[PIPELINE] Received dt = {actual_dt} from collider.")
            # Unleash SINDy
            run_sindy_and_ollama(full_history, args.ai_url, args.ai_model, args.no_llm, run_label=run_label, galactic_spin=args.galactic_spin, dt=actual_dt)
            
            print("\nSINDy extraction complete.")
            if not args.persistent:
                print("Server exiting.")
                break
            else:
                print("Persistent mode active. Waiting for next batch...")
                chunks = []
                total_frames = 0

if __name__ == "__main__":
    main()
