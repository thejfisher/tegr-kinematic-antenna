import torch
import numpy as np
import time
import os
import argparse
import psutil

# Create directory for high-res impact frames
os.makedirs("collider_frames", exist_ok=True)

# ---------------------------------------------------------
# Hyperparameters: The Dials of the Vacuum
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="2-body-collision")
parser.add_argument("--mass_a", type=float, default=1.0)
parser.add_argument("--mass_b", type=float, default=1.0)
parser.add_argument("--mass_c", type=float, default=1.0)
parser.add_argument("--pauli", type=float, default=500.0)
parser.add_argument("--vacuum", type=float, default=0.001)
parser.add_argument("--torsion", type=float, default=1.0)
parser.add_argument("--pauli_enabled", type=int, default=1)
parser.add_argument("--vacuum_enabled", type=int, default=1)
parser.add_argument("--torsion_enabled", type=int, default=1)
parser.add_argument("--entangled", type=int, default=0)
parser.add_argument("--t_symmetry", action="store_true", help="Run time-reversal stress test")
parser.add_argument("--velocity_clamp", type=int, default=-1,
                    help="0=off (free momentum, Paper 2), 1=on (0.99c cap, Paper 1), -1=auto")
parser.add_argument("--sink_mass", type=float, default=50000.0,
                    help="Mass of the gravity sink (gravity-sink mode)")
parser.add_argument("--num_particles", type=int, default=50,
                    help="Number of particles (direct-collapse mode)")
parser.add_argument("--collapse_radius", type=float, default=5.0,
                    help="Initial sphere radius (direct-collapse mode)")
parser.add_argument("--collapse_G", type=float, default=10.0,
                    help="Gravitational coupling (direct-collapse mode)")
args = parser.parse_args()

# Auto-detect velocity clamp: ON for swarm/orbit modes, OFF for collision/sink modes
if args.velocity_clamp == -1:
    USE_VELOCITY_CLAMP = args.mode in ["3-body-orbit"]
else:
    USE_VELOCITY_CLAMP = bool(args.velocity_clamp)

if args.mode == "direct-collapse":
    N = args.num_particles
elif args.mode in ["3-body-scatter", "3-body-orbit"]:
    N = 3
else:
    N = 2
DIM = 3                 # 3D spatial
DT = 0.001              # High-resolution time step
TOTAL_TICKS = 5000      # Long run to track scattering post-impact
SAVE_INTERVAL = 1      # Save every frame to ensure SINDy matrix is overdetermined
C = 100.0               # Speed of light in dimensionless units
LAMBDA_VAC = args.vacuum
PAULI_SCALAR = args.pauli
TORSION_G = args.torsion

# Use AMD ROCm GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing High-Fidelity Teleparallel Collider on: {device}")
print(f"  Mode: {args.mode} | MassA: {args.mass_a} | MassB: {args.mass_b}")
print(f"  Velocity Clamp: {'ON (0.99c cap)' if USE_VELOCITY_CLAMP else 'OFF (free momentum)'}")
print(f"  Pauli: {args.pauli} | Vacuum: {args.vacuum} | Torsion: {args.torsion}")
if args.mode == 'gravity-sink':
    print(f"  Sink Mass: {args.sink_mass} | Entangled: {bool(args.entangled)}")
if args.mode == 'direct-collapse':
    print(f"  N Particles: {N} | Radius: {args.collapse_radius} | G: {args.collapse_G}")

# ---------------------------------------------------------
# 10D Tensor Initialization
# State Vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
# ---------------------------------------------------------
state = torch.zeros((N, 10), device=device, dtype=torch.float32)

# Set base mass to specific particles
if args.mode == "direct-collapse":
    # All particles share the same mass
    state[:, 7] = args.mass_a
else:
    state[0, 7] = args.mass_a
    state[1, 7] = args.mass_b

if N == 3 and args.mode != "direct-collapse":
    state[2, 7] = args.mass_c
    
if args.mode == "3-body-scatter":
    # Position Setup: Highly Asymmetric 3-Body Scattering
    # Breaking symmetry completely so PySINDy detects chaotic shear forces
    state[0, 1] = -20.0
    state[0, 2] = 0.5
    state[1, 1] = 20.0
    state[1, 2] = -0.3
    state[2, 1] = 2.0
    state[2, 2] = 20.0
    
    # Momentum Setup: All fire towards origin with slight chaotic offsets
    mag0 = 701.8 * args.mass_a
    state[0, 4] = mag0 
    state[0, 5] = -mag0 * 0.025
    
    mag1 = 701.8 * args.mass_b
    state[1, 4] = -mag1 
    state[1, 5] = mag1 * 0.015
    
    mag2 = 701.8 * args.mass_c
    state[2, 4] = -mag2 * 0.1
    state[2, 5] = -mag2
    
    # Hue Setup
    state[0, 8] = 0.0
    state[1, 8] = 2.094 # 120 deg
    state[2, 8] = 4.188 # 240 deg

elif args.mode == "3-body-orbit":
    # The famous Figure-8 stable orbit initial conditions (scaled up by factor of 20 for visibility)
    # v is scaled up to roughly counteract G=1.0 and M=1.0 at this distance.
    S = 20.0
    state[0, 1] = 0.97000436 * S
    state[0, 2] = -0.24308753 * S
    
    state[1, 1] = -0.97000436 * S
    state[1, 2] = 0.24308753 * S
    
    state[2, 1] = 0.0
    state[2, 2] = 0.0
    
    # Momentum = mass * velocity * gamma (approx mass * velocity at low speeds)
    # For a perfect figure 8, mass is assumed identical.
    V_SCALE = 3.0  # Tuned for our G and distance
    vx1 = 0.4662036850 * V_SCALE
    vy1 = 0.4323657300 * V_SCALE
    
    state[0, 4] = args.mass_a * vx1
    state[0, 5] = args.mass_a * vy1
    
    state[1, 4] = args.mass_b * vx1
    state[1, 5] = args.mass_b * vy1
    
    state[2, 4] = args.mass_c * (-2.0 * vx1)
    state[2, 5] = args.mass_c * (-2.0 * vy1)
    
    state[0, 8] = 0.0
    state[1, 8] = 2.094
    state[2, 8] = 4.188

elif args.mode == "gravity-sink":
    # Position Setup: 3D Orbital Gravity Sink Test
    state[0, 1] = 10.0   # Particle A orbits the sink
    state[0, 2] = 5.0
    state[0, 3] = 2.0
    state[1, 1] = 50.0   # Particle B starts far away and will drift in
    state[1, 2] = 0.0
    
    # Momentum Setup: Give orbital momentum to Particle A
    state[0, 4] = 0.0
    state[0, 5] = 20.0   # Initial Y momentum (orbital velocity)
    state[0, 6] = 10.0   # Initial Z momentum (orbital velocity)
    # Particle B has 0 momentum, it will be pulled straight into the sink
    state[1, 4] = 0.0
    state[1, 5] = 0.0
    state[1, 6] = 0.0
    
    state[0, 8] = 0.0
    state[1, 8] = 3.1415

elif args.mode == "direct-collapse":
    # Direct Collapse: N particles in a random sphere, zero momentum.
    # No central anchor. Mutual gravity is the only attractor.
    # The math decides if they collapse, bounce, or scatter.
    R = args.collapse_radius
    np.random.seed(42)  # Reproducible initial conditions
    for i in range(N):
        # Uniform random positions inside a sphere of radius R
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)
        r = R * u ** (1.0 / 3.0)  # Cube root for uniform volume distribution
        theta = np.arccos(costheta)
        state[i, 1] = r * np.sin(theta) * np.cos(phi)
        state[i, 2] = r * np.sin(theta) * np.sin(phi)
        state[i, 3] = r * np.cos(theta)
        # Zero momentum — particles start at rest
        state[i, 4] = 0.0
        state[i, 5] = 0.0
        state[i, 6] = 0.0
        # Random hue phases distributed evenly
        state[i, 8] = np.random.uniform(0, 2 * np.pi)
    print(f"  Direct collapse initialized: {N} particles in sphere R={R}")

else:
    # 2-body-collision (Head-on collision at 0.99c)
    state[0, 1] = -20.0
    state[0, 2] = 0.01   # Slight Y offset
    state[1, 1] = 20.0
    state[1, 2] = -0.01  # Slight Y offset

    # Momentum Setup: Extreme relativistic approach (scaled for exactly 0.99c velocity)
    state[0, 4] = 701.8 * args.mass_a
    state[1, 4] = -701.8 * args.mass_b

    state[0, 8] = 0.0
    state[1, 8] = 3.1415

# Save initial exact coordinates for T-Symmetry divergence check
initial_positions = state[:, 1:4].clone().cpu()

# Track trajectory for SINDy extraction later
trajectory_history = []

def run_ticks(current_state, ticks, save_history=True):
    history = []
    total_radiation_shed = 0.0
    
    W_ij = bool(args.entangled)
    
    start_time = time.time()
    firewall_triggered = False
    for tick in range(ticks):
        pos = current_state[:, 1:4]
        mom = current_state[:, 4:7]
        m0 = current_state[:, 7].unsqueeze(1)
        
        p_squared = torch.sum(mom**2, dim=1, keepdim=True)
        gamma = torch.sqrt(1.0 + p_squared / (m0**2 * C**2))
        current_state[:, 9] = gamma.squeeze()
        
        vel = mom / (gamma * m0)
        
        # --- AMPS FIREWALL YIELD CHECK ---
        # Only applies when an ER=EPR bond is active (W_ij=True).
        # No wormhole = no bond to snap = no firewall energy deposit.
        dGamma = (gamma.max() - gamma.min()).item()
        if W_ij and dGamma > 4.0 and not firewall_triggered:
            firewall_triggered = True
            W_ij = False # SNAP THE WORMHOLE BOND
            print("-" * 60)
            print(f"*** WARNING: TENSION DIFFERENTIAL CRITICAL (dGamma = {dGamma:.3f}) ***")
            print("*** AMPS FIREWALL TRIGGERED! WORMHOLE BOND SNAPPED! ***")
            
            old_m0 = current_state[0, 7].item()
            # The mass spikes (m0 -> 3m0) representing deposited pair-production energy
            current_state[:, 7] *= 3.0 
            m0 = current_state[:, 7].unsqueeze(1)
            
            new_m0 = current_state[0, 7].item()
            print(f"*** ENERGY SPIKE DETECTED AT EVENT HORIZON: m_0 -> {new_m0:.3f} MeV ***")
            print("-" * 60)
            
        # --- RADIATIVE SHEDDING (B*+c -> B+c + gamma AFTERMATH) ---
        if firewall_triggered:
            # Vent the core topological strain as a photon
            # Decay m0 back down to its stable initial mass baseline
            shedding_rate = 5.0
            
            def shed(idx, target_mass):
                nonlocal total_radiation_shed
                if current_state[idx, 7] > target_mass:
                    drop = shedding_rate * (current_state[idx, 7] - target_mass) * DT
                    # Calculate max possible drop so we don't overshoot target_mass
                    drop = min(drop, current_state[idx, 7].item() - target_mass)
                    current_state[idx, 7] -= drop
                    total_radiation_shed += drop
                    
            shed(0, args.mass_a)
            shed(1, args.mass_b)
            if N == 3:
                shed(2, args.mass_c)
            
            m0 = current_state[:, 7].unsqueeze(1)
            
        # --- ER=EPR ENTANGLEMENT SYNC ---
        if W_ij and current_state.shape[0] >= 2:
            # Bidirectional phase synchronization (mutual average)
            # Neither particle is the "master" — both converge to the midpoint,
            # preserving action-reaction symmetry and frame-independence.
            avg_hue = (current_state[0, 8] + current_state[1, 8]) / 2.0
            current_state[0, 8] = avg_hue
            current_state[1, 8] = avg_hue

            
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2) + 1e-6
        
        v_diff = vel.unsqueeze(1) - vel.unsqueeze(0)
        
        if args.torsion_enabled:
            torsion_force = torch.sum(TORSION_G * diff * torch.cross(diff, v_diff, dim=2) / dist_sq.unsqueeze(2)**2, dim=1)
        else:
            torsion_force = torch.zeros_like(pos)
        
        if args.pauli_enabled:
            hue_diff = current_state[:, 8].unsqueeze(1) - current_state[:, 8].unsqueeze(0)
            pauli_phase_coupling = torch.cos(hue_diff)
            # Phase-coupled Pauli force: diff/dist_sq^2 = r_vec/r^4 = r_hat/r^3
            # The 1/r^3 scaling (vs gravity's 1/r^2) is consistent with the exchange
            # pressure spreading across 3 spatial dims + 1 compactified internal dim (hue),
            # a signature of the 10D->4D kinematic isomorphism (Paper 1, Sec. 3).
            pauli_force = torch.sum(PAULI_SCALAR * pauli_phase_coupling.unsqueeze(2) * diff / dist_sq.unsqueeze(2)**2, dim=1)
        else:
            pauli_force = torch.zeros_like(pos)
        
        if args.vacuum_enabled:
            damping_force = -LAMBDA_VAC * vel
        else:
            damping_force = torch.zeros_like(pos)
            
        if args.mode == "gravity-sink":
            # Supermassive Gravity Sink at Origin
            M = args.sink_mass
            G = 1.0
            sink_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
            diff_grav = pos - sink_pos
            dist_grav_sq = torch.sum(diff_grav**2, dim=1).unsqueeze(1) + 1e-6
            # Force vector towards sink: F = - G * M * m0 * r_vec / r^3
            grav_force = -(G * M * m0) * diff_grav / (dist_grav_sq ** 1.5)
        elif args.mode in ["direct-collapse", "3-body-orbit"]:
            # Mutual pairwise gravity between all particles.
            # No central anchor. F = -G * m_i * m_j * r_hat / r^2
            G_local = args.collapse_G if args.mode == "direct-collapse" else 100.0
            grav_force = torch.zeros_like(pos)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        diff_g = pos[i] - pos[j]
                        d_sq = torch.sum(diff_g**2) + 1e-6
                        # Force on i from j: F = -G * m_i * m_j * diff / d^3
                        force_ij = -G_local * m0[i] * m0[j] * diff_g / (d_sq ** 1.5)
                        grav_force[i] += force_ij.squeeze()
        else:
            grav_force = torch.zeros_like(pos)
        
        total_force = torsion_force + pauli_force + damping_force + grav_force
        
        # --- RELATIVISTIC INTEGRATION ---
        # Update momentum from forces
        current_state[:, 4:7] = mom + total_force * DT
        p_new = current_state[:, 4:7]
        p_new_sq = torch.sum(p_new**2, dim=1, keepdim=True)
        
        if USE_VELOCITY_CLAMP:
            # --- 5-STEP INTEGRATION WITH 0.99c CAP (Manuscript 1, Section 4) ---
            # Used for multi-particle swarm stability testing.
            # The 0.99c cap is a pragmatic numerical guard; the exact continuum
            # limit is v -> c as p -> infinity.
            v_temp = p_new / torch.sqrt(m0**2 + p_new_sq / (C**2))
            v_mag = torch.sqrt(torch.sum(v_temp**2, dim=1, keepdim=True) + 1e-12)
            max_v = 0.99 * C
            vel_new = torch.where(v_mag > max_v, v_temp * (max_v / v_mag), v_temp)
            v_sq = torch.sum(vel_new**2, dim=1, keepdim=True)
            gamma_new = 1.0 / torch.sqrt(1.0 - v_sq / (C**2))
            current_state[:, 9] = gamma_new.squeeze()
            # Re-synchronize momentum to capped velocity
            current_state[:, 4:7] = gamma_new * m0 * vel_new
        else:
            # --- FREE MOMENTUM INTEGRATION (Manuscript 2) ---
            # No artificial clamp. The relativistic relation v = p/(gamma*m0)
            # naturally guarantees |v| < c for any finite momentum.
            # Required for firewall testing where gamma must grow without bound.
            gamma_new = torch.sqrt(1.0 + p_new_sq / (m0**2 * C**2))
            current_state[:, 9] = gamma_new.squeeze()
            vel_new = p_new / (gamma_new * m0)
        
        # Position and Phase Updates (shared by both modes)
        current_state[:, 1:4] += vel_new * DT
        current_state[:, 8] = (current_state[:, 8] + (m0.squeeze() / gamma_new.squeeze()) * DT) % (2 * np.pi)
        current_state[:, 0] += DT

        if save_history and tick % SAVE_INTERVAL == 0:
            history.append(current_state.cpu().numpy().copy())
            if tick % 1000 == 0:
                print(f"Collision Tick {tick}/{ticks} - Gamma Max: {gamma.max().item():.3f}")
                
    elapsed = time.time() - start_time
    print(f"Loop Complete! Elapsed Time: {elapsed:.2f}s")
    print(f"TOTAL_RADIATION_SHED={total_radiation_shed:.6f}")
    
    print("\n--- SIMULATION RESOURCE USAGE ---")
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"CPU RAM USED     : {ram_mb:.2f} MB")
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"GPU VRAM USED    : {vram_mb:.2f} MB")
    
    return history

print("--- RUNNING FORWARD TIME ---")
trajectory_history = run_ticks(state, TOTAL_TICKS, save_history=True)

if args.t_symmetry:
    print("\n--- INITIATING TIME REVERSAL (T-SYMMETRY TEST) ---")
    print("Inverting momentum vectors...")
    state[:, 4:7] *= -1.0  # Flip momentum (p -> -p)
    print("Running matrix in reverse...")
    run_ticks(state, TOTAL_TICKS, save_history=False)
    
    final_positions = state[:, 1:4].cpu()
    # Calculate Euclidean divergence error from origin
    divergence = torch.norm(final_positions - initial_positions, dim=1).sum().item()
    print(f"\nT_SYMMETRY_DIVERGENCE={divergence:.5f}")

# Save 10D tensor map for PySINDy extraction
history_array = np.array(trajectory_history)
np.save("electron_trajectory.npy", history_array)
print("Trajectory saved to 'electron_trajectory.npy'")
print("\nNext step: Run the script and observe the scattering angle in the generated tensor data!")
