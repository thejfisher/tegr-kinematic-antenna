import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import time
import os
import sys
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
parser.add_argument("--photon_emission", type=int, default=0,
                    help="0=Passive monitor, 1=Active (snap tethers and shed mass)")
parser.add_argument("--amps_cooling_cap", type=float, default=1.0,
                    help="Multiplier for the shed mass to prevent NaN detonation (e.g. 0.01)")
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
parser.add_argument("--slit_width", type=float, default=2.0,
                    help="Width of each slit opening (double-slit mode)")
parser.add_argument("--slit_separation", type=float, default=6.0,
                    help="Center-to-center distance between slits (double-slit mode)")
parser.add_argument("--beam_momentum", type=float, default=5000.0,
                    help="Forward momentum of beam particles (double-slit mode)")
parser.add_argument("--impact_parameter", type=float, default=0.5,
                    help="Y-offset for scattering (head-on mode)")
parser.add_argument("--wall_mass", type=float, default=1000.0,
                    help="Mass of wall particles (double-slit mode). Lower = softer barrier.")
parser.add_argument("--screen_x", type=float, default=20.0,
                    help="X-position of the detector screen (double-slit mode).")
parser.add_argument("--total_ticks", type=int, default=5000,
                    help="Total simulation ticks per run.")
parser.add_argument("--dt", type=float, default=0.001,
                    help="Simulation time step.")
parser.add_argument("--run_label", type=str, default="Custom",
                    help="Human-readable label for this run (e.g. preset name)")
parser.add_argument("--batch_size", type=int, default=0,
                    help="GPU batch size for Tonomura trials. 0=auto (GPU:100, CPU:1).")
parser.add_argument("--num_slits", type=int, default=2,
                    help="Number of slits: 1=single-slit control, 2=double-slit (default).")
parser.add_argument("--wall_depth", type=int, default=1,
                    help="Wall depth in x-layers (1=original flat wall, >1=tunnel). Spacing matches wall_spacing.")
parser.add_argument("--wall_z_layers", type=int, default=1,
                    help="Wall layers in z-axis (1=2D, >1=3D tunnel compression). Spacing matches wall_spacing.")
parser.add_argument("--eraser_active", type=int, default=0,
                    help="0=Which-Path (Detector Wall), 1=Eraser (No Wall)")
parser.add_argument("--photon_freq", type=float, default=1.0,
                    help="Phase clock frequency for incoming photon (photoelectric mode)")
parser.add_argument("--work_function", type=float, default=100.0,
                    help="Restoring force constant holding electron in lattice (photoelectric mode)")
parser.add_argument("--paper1_exact", type=int, default=0,
                    help="Master toggle: 1=force Paper 1 exact math (spin_coupling=ON, pauli_power=2)")
parser.add_argument("--spin_coupling", type=int, default=0,
                    help="Enable spin-vorticity (omega_i . omega_j) term in Pauli force")
parser.add_argument("--pauli_power", type=int, default=3,
                    help="Pauli force power law: 2=Paper 1 (1/r^2), 3=KK signature (1/r^3)")
parser.add_argument("--wall_thermal_phase", type=int, default=0,
                    help="Randomize wall hues (1) instead of sync (0)")
parser.add_argument("--antenna_file", type=str, default="",
                    help="CSV file containing time series data for the anchor particle in antenna mode")
parser.add_argument("--output_file", type=str, default="electron_trajectory.npy",
                    help="Output filename for trajectory data (.npy format)")
parser.add_argument("--polarization_mode", type=str, default="isotropic",
                    help="Polarization mode for antenna: plus, cross, mixed, or isotropic")
parser.add_argument("--num_anchors", type=int, default=1,
                    help="Number of anchor particles in the array")
parser.add_argument("--qed_vacpol", type=int, default=0,
                    help="Enable Vacuum Polarization (Uehling potential screening)")
parser.add_argument("--qed_lamb", type=int, default=0,
                    help="Enable Lamb Shift (Self-energy Zitterbewegung jitter)")
parser.add_argument("--qed_compton", type=int, default=0,
                    help="Enable Compton Scattering (Radiation pressure wave)")
parser.add_argument("--galactic_spin", type=float, default=0.0,
                    help="Angular velocity of the galactic rotation (antenna mode)")
# --- Network & Hardware Dials ---
parser.add_argument("--device", type=str, default="auto",
                    help="Compute device: auto, cpu, cuda, cuda:0, etc.")
parser.add_argument("--zmq_port", type=int, default=0,
                    help="ZeroMQ PUSH port to bind and stream tensors (0=off)")
parser.add_argument("--zmq_target", type=str, default="",
                    help="IP:Port of ZeroMQ receiver (e.g., 100.66.100.83:7777)")
parser.add_argument("--zmq_flush_rate", type=int, default=100,
                    help="Number of ticks to buffer in RAM before flushing to network")
parser.add_argument("--cmb_noise", type=float, default=0.0,
                    help="Magnitude of the Cosmic Microwave Background expansion noise")
parser.add_argument("--pilot_wave", type=int, default=0,
                    help="Enable pilot-wave (dBB) guidance: particle velocity from torsion grid gradient")
parser.add_argument("--interpolation_order", type=str, default="linear",
                    help="Interpolation order for pilot-wave gradient sampling: linear or cubic")
parser.add_argument("--breit_wheeler", type=int, default=0,
                    help="Enable Breit-Wheeler pair production: FDTD grid spawns e-/e+ pairs at topological snap threshold")
parser.add_argument("--bw_max_pairs", type=int, default=100,
                    help="Max electron-positron pairs the vacuum buffer can hold (2 slots per pair)")
parser.add_argument("--bw_threshold", type=float, default=2.044,
                    help="Topological snap threshold in phi_curr amplitude (4*m_e = 2.044 MeV)")
parser.add_argument("--dbb_guidance", type=int, default=0,
                    help="de Broglie-Bohm guidance: wave gradient directly sets transverse velocity (0=off, 1=on)")
parser.add_argument("--dbb_strength", type=float, default=0.01,
                    help="dBB guidance coupling strength (0.001=whisper, 0.01=moderate, 1.0=full)")

args = parser.parse_args()

# Force Pilot Wave ON if the run label requests it (overrides GUI GUI defaults)
if "Pilot Wave" in args.run_label:
    args.pilot_wave = 1

# Early flag: needed by config print block before grid setup
BW_ENABLED = bool(getattr(args, 'breit_wheeler', 0))
DBB_GUIDANCE = bool(getattr(args, 'dbb_guidance', 0))

# Master toggle: Paper 1 Exact Math overrides individual settings
if args.paper1_exact:
    args.spin_coupling = 1
    args.pauli_power = 2

# Auto-detect velocity clamp: ON for swarm/orbit modes, OFF for collision/sink modes
if args.velocity_clamp == -1:
    USE_VELOCITY_CLAMP = args.mode in ["3-body-orbit"]
else:
    USE_VELOCITY_CLAMP = bool(args.velocity_clamp)

# --- Double-slit wall construction ---
wall_positions = []  # List of (x, y, z) tuples for all wall particles
wall_y_positions = []  # Kept for backward compat in slit-detection logic
N_beam = 0
N_wall = 0
SCREEN_X = args.screen_x
WALL_X = 0.0
if args.mode in ["double-slit", "quantum-eraser", "heat-sink-eraser"]:
    N_beam = args.num_particles
    sw = args.slit_width
    ss = args.slit_separation
    slit1_lo = ss / 2.0 - sw / 2.0   # Lower edge of slit 1
    slit1_hi = ss / 2.0 + sw / 2.0   # Upper edge of slit 1
    slit2_lo = -ss / 2.0 - sw / 2.0  # Lower edge of slit 2
    slit2_hi = -ss / 2.0 + sw / 2.0  # Upper edge of slit 2
    # Build wall as a SOLID PLATE in the Y-Z plane with rectangular slit openings.
    # In real experiments (Tonomura 1989, Jönsson 1961), the barrier is a copper foil
    # that extends continuously in Y and Z. The slits are narrow rectangular openings
    # cut through the foil. Particles cannot leak around the edges in any direction.
    #
    # Geometry:
    #   - Wall face: Y ∈ [-wall_extent, +wall_extent], Z ∈ [-wall_z_extent, +wall_z_extent]
    #   - Wall depth (X): wall_depth layers from WALL_X backward (tunnel)
    #   - Slit openings: rectangular holes defined by (Y: slit boundaries, Z: ±slit_z_half)
    #   - Wall particles placed everywhere on the plate EXCEPT inside the slit rectangles
    #
    # IMPORTANT: The hard wall reflection (checked every tick) enforces an INFINITE
    # solid plate — any particle crossing x > WALL_X gets reflected unless it's inside
    # a slit rectangle. The wall PARTICLES below only provide Pauli repulsion force
    # (the squeeze through the tunnel). So we only need particles near the slits where
    # the beam actually interacts. Far-field particles waste GPU memory (O(N²) distance
    # matrix) without contributing meaningful force.
    #
    wall_extent = 8.0   # Pauli particles cover Y ∈ [-8, 8] (beam range is ±7)
    wall_spacing = 1.0  # Coarser grid (Pauli 1/r³ has range; hard reflection does blocking)
    wall_depth = args.wall_depth  # Number of x-layers (1=flat, >1=tunnel)
    
    # Z-extent of Pauli particle plate: cover beam region only
    # Hard reflection enforces the full infinite plate regardless
    #
    # Slit Z-height: Based on Jönsson 1961 geometry.
    # Real slits are ~100× longer (Z) than wide (Y). In the original experiment:
    #   slit_width=500nm, slit_length=50,000nm → ratio = 100:1
    # With slit_width=4.0, slit_z_half = 200 → slit open from z=-200 to z=+200.
    # This is effectively infinite compared to beam spread (~±3), matching reality.
    slit_z_half = args.slit_width * 50  # Jönsson ratio: slit_length = 100 × slit_width
    wall_z_extent = min(slit_z_half + 2.0, 8.0)  # Pauli particles only near beam (cap at ±8)
    
    # Build ALL y-positions (solid wall + slit regions)
    all_y_positions = []
    y_pos = -wall_extent
    while y_pos <= wall_extent:
        all_y_positions.append(y_pos)
        y_pos += wall_spacing
    
    # Also keep the non-slit y-positions for backward compat
    for yp in all_y_positions:
        in_slit1 = slit1_lo <= yp <= slit1_hi
        in_slit2 = slit2_lo <= yp <= slit2_hi if args.num_slits >= 2 else False
        if not in_slit1 and not in_slit2:
            wall_y_positions.append(yp)
    
    # Build all z-positions for the wall plate
    all_z_positions = []
    z_pos = -wall_z_extent
    while z_pos <= wall_z_extent:
        all_z_positions.append(z_pos)
        z_pos += wall_spacing
    
    # Build the wall: place particles at every (x, y, z) EXCEPT inside slit openings
    # A slit opening is defined as: (Y within slit bounds) AND (|Z| < slit_z_half)
    for x_layer in range(wall_depth):
        wx = WALL_X - x_layer * wall_spacing
        for wy in all_y_positions:
            # Check if this Y is inside a slit
            in_slit1_y = slit1_lo <= wy <= slit1_hi
            in_slit2_y = (slit2_lo <= wy <= slit2_hi) if args.num_slits >= 2 else False
            in_slit_y = in_slit1_y or in_slit2_y
            
            for wz in all_z_positions:
                if in_slit_y and abs(wz) < slit_z_half:
                    continue  # This is INSIDE the slit opening — leave it empty
                wall_positions.append((wx, wy, wz))
    
    if args.mode == "double-slit" and args.eraser_active:
        slit1_center = (slit1_lo + slit1_hi) / 2.0
        detector_x = WALL_X - 1.0
        detector_y = slit1_center
        detector_z = 0.0
        # Insert at the beginning so it becomes index 1 (the first wall particle)
        wall_positions.insert(0, (detector_x, detector_y, detector_z))
        print(f"  [DETECTOR] Added Detector Defect at ({detector_x:.1f}, {detector_y:.1f}, {detector_z:.1f})")

    N_wall = len(wall_positions)
    N_y_positions = len(wall_y_positions)  # Non-slit y-positions for display
    N = N_beam + N_wall
    N_z_positions = len(all_z_positions)
    N_cap_z = sum(1 for z in all_z_positions if abs(z) >= slit_z_half)  # Z-positions that form caps
    slit_label = "Single-slit" if args.num_slits == 1 else "Double-slit"
    print(f"  {slit_label}: {N_beam} beam + {N_wall} wall = {N} total particles")
    print(f"  Solid plate: Y=[{-wall_extent:.0f},{wall_extent:.0f}] Z=[{-wall_z_extent:.0f},{wall_z_extent:.0f}]")
    print(f"  Wall geometry: {wall_depth} x-layers × {len(all_y_positions)} y × {N_z_positions} z")
    print(f"  Slit openings: Y-bounded, Z ∈ [{-slit_z_half:.1f}, {slit_z_half:.1f}] (height={2*slit_z_half:.1f})")
    print(f"  Cap layers above/below slits: {N_cap_z} z-positions seal the tunnel")
    if wall_depth > 1:
        print(f"  Tunnel depth: {wall_depth} layers = {(wall_depth-1)*wall_spacing:.1f} sim units")
        print(f"  Tunnel x-range: [{WALL_X - (wall_depth-1)*wall_spacing:.1f}, {WALL_X:.1f}]")
    print(f"  Slit 1: y=[{slit1_lo:.1f}, {slit1_hi:.1f}]")
    if args.num_slits >= 2:
        print(f"  Slit 2: y=[{slit2_lo:.1f}, {slit2_hi:.1f}]")
    else:
        print(f"  Slit 2: CLOSED (filled with wall)")
elif args.mode in ["direct-collapse", "holographic", "antenna", "einstein-stick", "ads-cft", "pilot-wave"]:
    N = args.num_particles
elif args.mode in ["3-body-scatter", "3-body-orbit"]:
    N = 3
else:
    N = 2
DIM = 3                 # 3D spatial
DT = args.dt              # Dynamic time step
TOTAL_TICKS = args.total_ticks
SAVE_INTERVAL = max(1, int(0.001 / DT))  # Dynamically scale output frequency so low dt runs don't overflow RAM
ZMQ_TICK_RATE = SAVE_INTERVAL  # Stream at same rate as local save (no 10x decimation)
C = 100.0               # Speed of light in dimensionless units
LAMBDA_VAC = args.vacuum
PAULI_SCALAR = args.pauli
TORSION_G = args.torsion

# Use selected device or auto-detect ROCm GPU
if args.device.lower() == "auto":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print(f"Initializing High-Fidelity Teleparallel Collider on: {device}")
print(f"  Mode: {args.mode} | MassA: {args.mass_a} | MassB: {args.mass_b}")
print(f"  Velocity Clamp: {'ON (0.99c cap)' if USE_VELOCITY_CLAMP else 'OFF (free momentum)'}")
print(f"  Pauli: {args.pauli} | Vacuum: {args.vacuum} | Torsion: {args.torsion}")

# --- CLEAR RUN CONFIGURATION BLOCK ---
print("\n" + "=" * 60)
print(f"RUN LABEL: {args.run_label}")
print(f"MODE:      {args.mode}")
print(f"MASS A:    {args.mass_a} MeV | MASS B: {args.mass_b} MeV | MASS C: {args.mass_c} MeV")
print(f"PAULI:     {args.pauli} (enabled={args.pauli_enabled})")
print(f"VACUUM:    {args.vacuum} (enabled={args.vacuum_enabled})")
print(f"TORSION:   {args.torsion} (enabled={args.torsion_enabled})")
print(f"V-CLAMP:   {'ON (0.99c)' if USE_VELOCITY_CLAMP else 'OFF (free)'}")
print(f"ENTANGLED: {bool(args.entangled)}")
print(f"PAPER1 EX: {bool(args.paper1_exact)}")
print(f"SPIN COUP: {bool(args.spin_coupling)}")
print(f"PAULI POW: 1/r^{args.pauli_power}")
if args.mode == 'double-slit':
    print(f"SLIT W:    {args.slit_width} | SLIT SEP: {args.slit_separation}")
    print(f"BEAM P:    {args.beam_momentum} | N BEAM: {args.num_particles}")
    print(f"WALL MASS: {args.wall_mass:.0f}")
elif args.mode == 'direct-collapse':
    print(f"N PART:    {args.num_particles} | RADIUS: {args.collapse_radius} | G: {args.collapse_G}")
elif args.mode == 'gravity-sink':
    print(f"SINK MASS: {args.sink_mass}")
elif args.mode == 'head-on':
    print(f"BEAM P:    {args.beam_momentum} | IMPACT: {args.impact_parameter}")
print("=" * 60 + "\n")
if args.mode == 'gravity-sink':
    print(f"  Sink Mass: {args.sink_mass} | Entangled: {bool(args.entangled)}")
if args.mode == 'direct-collapse':
    print(f"  N Particles: {N} | Radius: {args.collapse_radius} | G: {args.collapse_G}")
if args.mode == 'photoelectric':
    print(f"  Photon Freq: {args.photon_freq} | Work Function (k): {args.work_function}")
if args.mode == 'holographic':
    print(f"  N Particles: {N} | Tethers: {args.entangled} | Radius: {args.collapse_radius}")
if args.mode == 'antenna':
    print(f"  Antenna File: {args.antenna_file} | N Particles: {N} | Radius: {args.collapse_radius}")
if args.mode == 'einstein-stick':
    print(f"  N Particles: {N} | Entangled: {bool(args.entangled)}")
if args.mode == 'pilot-wave':
    print(f"  Pilot-Wave (dBB): N={N} | Interp: {args.interpolation_order}")
if BW_ENABLED:
    print(f"  Breit-Wheeler: threshold={args.bw_threshold:.3f} MeV | buffer={args.bw_max_pairs * 2} slots")
if DBB_GUIDANCE:
    print(f"  dBB GUIDANCE: strength={args.dbb_strength} | Wave gradient sets transverse velocity (Pauli wall ON)")

# ---------------------------------------------------------
# ZeroMQ Telemetry Setup (RAM-to-RAM streaming)
# ---------------------------------------------------------
zmq_socket = None
if args.zmq_port > 0 or args.zmq_target != "":
    try:
        import zmq
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUSH)
        if args.zmq_target != "":
            zmq_socket.connect(f"tcp://{args.zmq_target}")
            print(f"\n[NETWORK] ZeroMQ PUSH socket connected to {args.zmq_target}. Streaming LIVE.")
        else:
            zmq_socket.bind(f"tcp://*:{args.zmq_port}")
            print(f"\n[NETWORK] ZeroMQ PUSH socket securely bound to port {args.zmq_port}. Streaming LIVE.")
    except Exception as e:
        print(f"\n[NETWORK ERROR] Failed to setup ZeroMQ: {e}")
        zmq_socket = None

# ---------------------------------------------------------
# 10D Tensor Initialization
# State Vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
# ---------------------------------------------------------
state = torch.zeros((N, 10), device=device, dtype=torch.float32)
# Spin-vorticity vector (scalar omega_z per particle)
# Paper 1: beam particles get random +/-0.5, wall particles get +0.5
spin = torch.zeros(N, device=device, dtype=torch.float32)

# Set base mass to specific particles with +/- 1% micro-variance to shatter regression collinearity
if args.mode in ["direct-collapse", "holographic", "antenna", "ads-cft", "holographic-shell", "holographic-ring"]:
    # All particles share the same mass
    state[:, 7] = args.mass_a * (1.0 + 0.01 * (2 * torch.rand(N, device=device) - 1))
else:
    state[0, 7] = args.mass_a * (1.0 + 0.01 * (2 * torch.rand(1, device=device).item() - 1))
    if N > 1:
        state[1, 7] = args.mass_b * (1.0 + 0.01 * (2 * torch.rand(1, device=device).item() - 1))

if N == 3 and args.mode != "direct-collapse":
    state[2, 7] = args.mass_c * (1.0 + 0.01 * (2 * torch.rand(1, device=device).item() - 1))
    
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

elif args.mode == "double-slit":
    # Double-Slit Ensemble: Beam particles with identical forward momentum, randomized θ_hue
    # Particles are spread uniformly in y to illuminate both slits.
    # Wall particles are stationary (mass=1e6) and line the barrier with slit gaps.
    np.random.seed(42)
    beam_start_x = -10.0
    beam_spacing = 0.3  # Stagger in x so they don't stack on each other
    # Spread beam across y range that covers both slits + some wall
    beam_y_range = args.slit_separation / 2.0 + args.slit_width / 2.0 + 2.0  # e.g. ±5 for default params
    for i in range(N_beam):
        state[i, 1] = beam_start_x - i * beam_spacing  # Stagger along x
        state[i, 2] = np.random.uniform(-beam_y_range, beam_y_range)  # Spread across both slits
        state[i, 3] = 0.0
        state[i, 4] = args.beam_momentum  # Forward momentum
        state[i, 5] = 0.0
        state[i, 6] = 0.0
        state[i, 7] = args.mass_a
        state[i, 8] = np.random.uniform(0, 2 * np.pi)  # THE ONLY DIFFERENCE THAT MATTERS
        spin[i] = 0.5 if np.random.random() > 0.5 else -0.5  # Random spin ±0.5
    # Wall particles: mass determines barrier penetrability
    # Too heavy (1e6) = impenetrable force field. Too light = beam punches through wall.
    # Sweet spot (~1e3): beam squeezes through slit under localized Pauli pressure.
    WALL_HUE = 0.0  # Fixed reference phase
    WALL_SPIN = 0.5  # Wall atoms: all spin-up
    for j, (wx, wy, wz) in enumerate(wall_positions):
        idx = N_beam + j
        state[idx, 1] = wx
        state[idx, 2] = wy
        state[idx, 3] = wz
        state[idx, 7] = args.wall_mass / N_wall  # Distribute total wall mass evenly across all particles
        if args.wall_thermal_phase == 1:
            state[idx, 8] = np.random.uniform(0, 2 * np.pi)
        else:
            state[idx, 8] = WALL_HUE
        spin[idx] = WALL_SPIN
    # Estimate slit throughput
    sw = args.slit_width
    ss = args.slit_separation
    slit_area = 2.0 * sw  # Two slits
    beam_area = 2.0 * beam_y_range
    expected_through = slit_area / beam_area * 100
    print(f"  Beam: x=[{beam_start_x:.1f} to {beam_start_x - (N_beam-1)*beam_spacing:.1f}], y=[{-beam_y_range:.1f}, {beam_y_range:.1f}], px={args.beam_momentum}")
    print(f"  Wall: {N_wall} particles ({args.wall_depth} layers), mass={args.wall_mass:.0e}")
    print(f"  Expected throughput: ~{expected_through:.0f}% (slit area {slit_area:.1f} / beam width {beam_area:.1f})")

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

elif args.mode == "antenna":
    R = min(args.collapse_radius, 10.0)  # Cap radius so 1/r^3 Pauli coupling is detectable
    np.random.seed(42)
    N_anchors = args.num_anchors
    for i in range(N_anchors):
        r = R * (np.random.random() ** (1/3.0))
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        state[i, 1] = r * np.sin(theta) * np.cos(phi)
        state[i, 2] = r * np.sin(theta) * np.sin(phi)
        state[i, 3] = r * np.cos(theta)
        if args.galactic_spin > 0.0:
            vx = -args.galactic_spin * state[i, 2].item()
            vy = args.galactic_spin * state[i, 1].item()
            v_sq = vx**2 + vy**2
            gamma = 1.0 / np.sqrt(max(1.0 - v_sq/(100.0**2), 1e-6))
            state[i, 4] = args.mass_a * gamma * vx
            state[i, 5] = args.mass_a * gamma * vy
            state[i, 6] = 0.0
        else:
            state[i, 4:7] = 0.0 # Pulsars start at rest
        state[i, 7] = args.mass_a # Normal mass
        state[i, 8] = 0.0
        spin[i] = 0.5
        
    for i in range(N_anchors, N):
        # Central source
        state[i, 1:4] = 0.0 # Center
        state[i, 4:7] = 0.0 # Rest
        state[i, 7] = args.mass_b  # Same mass as pulsars — no artificial dampening
        state[i, 8] = 0.0
        spin[i] = 0.5
    print(f"  Antenna initialized: {N_anchors} Pulsars + {N-N_anchors} Source at R=0")

elif args.mode == "head-on":
    # Head-on collision with impact parameter for Veneziano amplitude test
    state[0, 1] = -10.0
    state[0, 2] = args.impact_parameter
    state[1, 1] = 10.0
    state[1, 2] = -args.impact_parameter

    state[0, 4] = args.beam_momentum
    state[1, 4] = -args.beam_momentum

    state[0, 8] = 0.0
    state[1, 8] = 0.0
    
    spin[0] = 0.5
    spin[1] = 0.5

elif args.mode == "photoelectric":
    # Particle 0 = Electron (in lattice well at origin)
    state[0, 1] = 0.0
    state[0, 2] = 0.0
    state[0, 4] = 0.0
    state[0, 8] = 0.0
    spin[0] = 0.5
    
    # Particle 1 = Photon (incoming from -X)
    state[1, 1] = -0.05  # Moved very close to compensate for extreme dt=0.000001
    state[1, 2] = 0.01  # Slight impact parameter
    state[1, 4] = args.beam_momentum # "Intensity"
    state[1, 8] = 0.0
    spin[1] = 0.5

elif args.mode == "einstein-stick":
    d0 = args.slit_separation
    # Particle 0 = Hammer
    state[0, 1] = -10.0
    state[0, 2:7] = 0.0
    state[0, 4] = args.beam_momentum
    state[0, 7] = args.mass_a
    state[0, 8] = 0.0
    spin[0] = 0.5
    
    # Particles 1 to N-1 = Stick
    for i in range(1, N):
        state[i, 1] = (i - 1) * d0
        state[i, 2:7] = 0.0
        state[i, 7] = args.mass_b
        state[i, 8] = 0.0
        spin[i] = 0.5

elif args.mode == "qed":
    # 2 particles: 0 = moving electron, 1 = heavy proton (static)
    N = 2
    state = torch.zeros((N, 10), device=device, dtype=torch.float32)
    
    # Moving electron at origin
    state[0, 1:4] = torch.tensor([0.0, 0.0, 0.0], device=device)
    state[0, 7] = 1.0 # m0
    state[0, 8] = 0.0 # hue (U(1) phase)
    state[0, 9] = 1.0 # gamma
    
    # Heavy static proton nearby
    state[1, 1:4] = torch.tensor([2.0, 2.0, 0.0], device=device)
    state[1, 7] = 10000.0 # m0
    state[1, 8] = np.pi # Opposite phase for attractive Coulomb force
    state[1, 9] = 1.0
    
    spin[0] = 0.5
    spin[1] = 0.5

elif args.mode == "holographic":
    # N particles in a spherical cloud
    radius = args.collapse_radius
    r_rand = radius * torch.rand(N, device=device)**(1/3)
    theta = torch.acos(2 * torch.rand(N, device=device) - 1)
    phi = 2 * np.pi * torch.rand(N, device=device)
    
    state[:, 1] = r_rand * torch.sin(theta) * torch.cos(phi)
    state[:, 2] = r_rand * torch.sin(theta) * torch.sin(phi)
    state[:, 3] = r_rand * torch.cos(theta)
    state[:, 8] = 2 * np.pi * torch.rand(N, device=device)
    spin[:] = 0.5

elif args.mode == "ads-cft":
    # ==========================================
    # AdS/CFT HOLOGRAPHIC CORRESPONDENCE TEST
    # ==========================================
    # Boundary (CFT): first N_boundary particles on the spherical shell surface
    # Bulk (AdS):     remaining particles uniformly distributed inside the sphere
    # W_mat:          ER=EPR bridges connecting boundary <-> bulk exclusively
    # ==========================================
    radius = args.collapse_radius
    N_boundary = args.num_anchors  # Reuse num_anchors as boundary particle count
    N_bulk = N - N_boundary
    print(f"  AdS/CFT Setup: {N_boundary} boundary (CFT) + {N_bulk} bulk (AdS) particles")
    print(f"  Boundary Radius: {radius}")
    
    # --- BOUNDARY PARTICLES (CFT) ---
    # Placed on the surface of a sphere at exactly R = collapse_radius
    theta_bnd = torch.acos(2 * torch.rand(N_boundary, device=device) - 1)
    phi_bnd = 2 * np.pi * torch.rand(N_boundary, device=device)
    state[:N_boundary, 1] = radius * torch.sin(theta_bnd) * torch.cos(phi_bnd)
    state[:N_boundary, 2] = radius * torch.sin(theta_bnd) * torch.sin(phi_bnd)
    state[:N_boundary, 3] = radius * torch.cos(theta_bnd)
    state[:N_boundary, 8] = 2 * np.pi * torch.rand(N_boundary, device=device)
    
    # --- BULK PARTICLES (AdS) ---
    # Uniformly distributed inside the sphere (cube-root sampling for uniform volume)
    r_bulk = radius * 0.9 * torch.rand(N_bulk, device=device)**(1/3)  # 0.9R to avoid shell overlap
    theta_blk = torch.acos(2 * torch.rand(N_bulk, device=device) - 1)
    phi_blk = 2 * np.pi * torch.rand(N_bulk, device=device)
    state[N_boundary:, 1] = r_bulk * torch.sin(theta_blk) * torch.cos(phi_blk)
    state[N_boundary:, 2] = r_bulk * torch.sin(theta_blk) * torch.sin(phi_blk)
    state[N_boundary:, 3] = r_bulk * torch.cos(theta_blk)
    state[N_boundary:, 8] = 2 * np.pi * torch.rand(N_bulk, device=device)
    
    # All particles start at rest
    state[:, 4:7] = 0.0
    spin[:] = 0.5

elif args.mode in ["holographic-shell", "holographic-ring"]:
    # N-1 particles form a hollow shape, 1 particle fired through
    radius = args.collapse_radius
    
    if args.mode == "holographic-shell":
        # Surface of a 3D sphere
        theta = torch.acos(2 * torch.rand(N-1, device=device) - 1)
        phi = 2 * np.pi * torch.rand(N-1, device=device)
        
        state[:N-1, 1] = radius * torch.sin(theta) * torch.cos(phi)
        state[:N-1, 2] = radius * torch.sin(theta) * torch.sin(phi)
        state[:N-1, 3] = radius * torch.cos(theta)
    else:
        # A 2D Ring (Circle) on the YZ plane
        theta = 2 * np.pi * torch.rand(N-1, device=device)
        state[:N-1, 1] = 0.0 # Flat on the X=0 plane
        state[:N-1, 2] = radius * torch.cos(theta)
        state[:N-1, 3] = radius * torch.sin(theta)
        
    state[:N-1, 8] = 2 * np.pi * torch.rand(N-1, device=device) # phase lock
    
    # 1 particle fired from outside straight through the origin
    state[-1, 1] = radius * 3.0 # start at 3R on X axis
    state[-1, 2] = 0.0
    state[-1, 3] = 0.0
    state[-1, 4] = -50.0 * args.mass_a # velocity towards the origin
    state[-1, 8] = 0.0 # Phase
    
    spin[:] = 0.5

elif args.mode == "pilot-wave":
    # Pilot-Wave (dBB): N particles as point defects in the torsion vacuum.
    # Each particle is guided by the gradient of the Eulerian torsion field.
    # Uses the existing FDTD conv3d solver as the pilot wave.
    R = args.collapse_radius
    np.random.seed(42)
    
    # Wall construction for double-slit geometry
    sw = args.slit_width
    ss = args.slit_separation
    slit1_lo = ss / 2.0 - sw / 2.0
    slit1_hi = ss / 2.0 + sw / 2.0
    slit2_lo = -ss / 2.0 - sw / 2.0
    slit2_hi = -ss / 2.0 + sw / 2.0
    
    for i in range(N):
        # Start all particles behind the wall with forward momentum
        state[i, 1] = -10.0  # x: behind wall
        beam_y_range = ss / 2.0 + sw / 2.0 + 2.0
        state[i, 2] = np.random.uniform(-beam_y_range, beam_y_range)  # y: spread across slits
        state[i, 3] = np.random.uniform(-2.0, 2.0)  # z: collimated
        state[i, 4] = args.beam_momentum  # px: forward
        state[i, 5] = 0.0
        state[i, 6] = 0.0
        state[i, 7] = args.mass_a
        state[i, 8] = np.random.uniform(0, 2 * np.pi)  # Random phase
        spin[i] = 0.5 if np.random.random() > 0.5 else -0.5
    print(f"  Pilot-wave initialized: {N} particles, beam x=-10, mom_x={args.beam_momentum}")

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
    state[1, 8] = 0.0
    
    spin[0] = 0.5
    spin[1] = 0.5

# Save initial exact coordinates for T-Symmetry divergence check
initial_positions = state[:, 1:4].clone().cpu()

# Setup antenna signal if applicable
antenna_signal = None
if args.mode == "antenna" and args.antenna_file:
    import os
    if os.path.exists(args.antenna_file):
        import pandas as pd
        data = pd.read_csv(args.antenna_file)
        data.columns = data.columns.str.strip()
        
        if 'Residuals' in data.columns:
            # --- RESIDUAL TIME-SERIES INGESTION ---
            # Sparse monthly data → cubic interpolation across all simulation ticks
            from scipy.interpolate import interp1d
            raw_residuals = data['Residuals'].values.astype(float)
            
            # Normalize to [-1, 1] range for phase injection
            res_max = np.max(np.abs(raw_residuals)) + 1e-12
            normalized = raw_residuals / res_max
            
            # Map N data points smoothly across total_ticks
            data_ticks = np.linspace(0, args.total_ticks - 1, len(normalized))
            interp_func = interp1d(data_ticks, normalized, kind='cubic', fill_value="extrapolate")
            full_signal = interp_func(np.arange(args.total_ticks))
            
            antenna_signal = torch.tensor(full_signal, dtype=torch.float32, device=device)
            print(f"Loaded {len(raw_residuals)} residual data points -> interpolated to {args.total_ticks} ticks")
            print(f"  Residual range: [{np.min(raw_residuals):.0f}, {np.max(raw_residuals):.0f}] -> normalized to [-1, 1]")
        else:
            if 'Tick' in data.columns or 'Time' in data.columns:
                data = data.drop(columns=['Tick', 'Time'], errors='ignore')
            antenna_signal = torch.tensor(data.values, dtype=torch.float32, device=device)
            args.num_anchors = antenna_signal.shape[1] if len(antenna_signal.shape) > 1 else 1
            print(f"Loaded {antenna_signal.shape[0]} ticks of antenna data for {args.num_anchors} anchors.")
    else:
        print(f"Warning: Antenna file {args.antenna_file} not found.")

# Track trajectory for SINDy extraction later
trajectory_history = []

# ---------------------------------------------------------
# Eulerian Pilot Wave Grid (3D FDTD) Setup
# ---------------------------------------------------------
GRID_RES = 250
GRID_MIN = -50.0
GRID_MAX = 50.0
GRID_SIZE = GRID_MAX - GRID_MIN
DX = GRID_SIZE / GRID_RES

# phi represents the local WeitzenbÃ¶ck torsion (wave amplitude)
phi_curr = torch.zeros((1, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
phi_prev = torch.zeros((1, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
phi_next = torch.zeros((1, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)

# 3D Laplacian Kernel (7-point stencil)
laplacian_kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=torch.float32)
laplacian_kernel[0, 0, 1, 1, 1] = -6.0
laplacian_kernel[0, 0, 1, 1, 0] = 1.0
laplacian_kernel[0, 0, 1, 1, 2] = 1.0
laplacian_kernel[0, 0, 1, 0, 1] = 1.0
laplacian_kernel[0, 0, 1, 2, 1] = 1.0
laplacian_kernel[0, 0, 0, 1, 1] = 1.0
laplacian_kernel[0, 0, 2, 1, 1] = 1.0

# Courant stability: c * dt / dx <= 1/sqrt(3) (~0.577)
# Clamp wave speed if dt is large (e.g. gravity-sink mode dt=0.02)
max_stable_speed = 0.5 * DX / DT
WAVE_SPEED = min(C, max_stable_speed)
C_SQUARED = (WAVE_SPEED * DT / DX) ** 2
TORSION_DECAY = 0.999

# ---------------------------------------------------------
# Breit-Wheeler Pair Production (Resonant Vacuum Condensate)
# ---------------------------------------------------------
# BW_ENABLED is defined early (line ~128) for config print block
BW_MAX_PAIRS = getattr(args, 'bw_max_pairs', 100)
BW_THRESHOLD = getattr(args, 'bw_threshold', 2.044)
BW_ELECTRON_MASS = 0.511  # MeV/c^2 — rest mass of electron
BW_PAIR_COST = 2.0 * BW_ELECTRON_MASS  # 1.022 MeV to create the pair
BW_BUFFER_SIZE = BW_MAX_PAIRS * 2  # 2 particles per pair (e- and e+)
bw_spawned_count = 0  # Track how many BW slots have been activated

if BW_ENABLED:
    # 3x3x3 Gaussian drain kernel (normalized to sum=1.0)
    # Smoothly removes energy from the FDTD grid at the snap point
    # to prevent Dirac delta shockwaves in the leapfrog solver
    _gauss_1d = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=torch.float32)
    bw_drain_kernel = (_gauss_1d.view(1,1,3,1,1) * _gauss_1d.view(1,1,1,3,1) * _gauss_1d.view(1,1,1,1,3))
    bw_drain_kernel = bw_drain_kernel / bw_drain_kernel.sum()  # Normalize to 1.0
    print(f"\n[BREIT-WHEELER] Pair production ENABLED")
    print(f"  Threshold: {BW_THRESHOLD:.3f} MeV | Pair cost: {BW_PAIR_COST:.3f} MeV")
    print(f"  Vacuum buffer: {BW_BUFFER_SIZE} slots ({BW_MAX_PAIRS} pairs max)")
    print(f"  Gaussian drain: 3x3x3 kernel (smooth energy extraction)")


# 3D Obstacle Mask (1.0 = vacuum, 0.0 = copper wall)
obstacle_mask = torch.ones((1, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
# Map analytical wall_positions to discrete grid indices
for wx, wy, wz in wall_positions:
    # Convert coordinate to index (clamp to grid bounds)
    ix = int((wx - GRID_MIN) / DX)
    iy = int((wy - GRID_MIN) / DX)
    iz = int((wz - GRID_MIN) / DX)
    if 0 <= ix < GRID_RES and 0 <= iy < GRID_RES and 0 <= iz < GRID_RES:
        # We inflate the wall particle slightly so it's a solid barrier (3x3x3 voxel cube)
        ix_min, ix_max = max(0, ix-1), min(GRID_RES, ix+2)
        iy_min, iy_max = max(0, iy-1), min(GRID_RES, iy+2)
        iz_min, iz_max = max(0, iz-1), min(GRID_RES, iz+2)
        obstacle_mask[0, 0, ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] = 0.0

@torch.no_grad()
def run_ticks(current_state, ticks, save_history=True):
    global phi_curr, phi_prev, phi_next, bw_spawned_count
    history = []
    chunk_buffer = [] # ZMQ Network Stream buffer
    total_radiation_shed = 0.0
    
    # For double-slit: track particles that reach the detector screen
    screen_detected = {}  # {particle_index: y_position_at_screen}
    
    # Initialize W_ij matrix for entanglement
    W_mat = torch.zeros((N, N), dtype=torch.bool, device=device)
    if args.mode in ["holographic", "holographic-shell", "holographic-ring"]:
        num_tethers = int(args.entangled) if int(args.entangled) > 0 else 300
        for _ in range(num_tethers):
            idx1 = np.random.randint(0, N)
            idx2 = np.random.randint(0, N)
            if idx1 != idx2:
                W_mat[idx1, idx2] = True
                W_mat[idx2, idx1] = True
    elif args.mode == "ads-cft":
        # ER=EPR bridges connecting BOUNDARY <-> BULK exclusively
        # This enforces the holographic correspondence: entanglement only crosses the boundary
        N_boundary = args.num_anchors
        N_bulk = N - N_boundary
        num_tethers = int(args.entangled) if int(args.entangled) > 0 else 300
        tethers_placed = 0
        max_attempts = num_tethers * 5
        attempts = 0
        while tethers_placed < num_tethers and attempts < max_attempts:
            bnd_idx = np.random.randint(0, N_boundary)         # Boundary particle
            blk_idx = np.random.randint(N_boundary, N)         # Bulk particle
            if not W_mat[bnd_idx, blk_idx]:
                W_mat[bnd_idx, blk_idx] = True
                W_mat[blk_idx, bnd_idx] = True
                tethers_placed += 1
            attempts += 1
        print(f"  AdS/CFT: Placed {tethers_placed} boundary<->bulk ER=EPR tethers")
    elif args.mode == "antenna":
        N_anchors = args.num_anchors
        for a in range(N_anchors):
            for i in range(N_anchors, N):
                W_mat[a, i] = True
                W_mat[i, a] = True
    elif args.mode == "qed":
        W_mat[0, 1] = True
        W_mat[1, 0] = True
    elif args.mode == "einstein-stick" and int(args.entangled) > 0:
        for i in range(1, N - 1):
            W_mat[i, i+1] = True
            W_mat[i+1, i] = True
    elif args.entangled and N >= 2:
        W_mat[0, 1] = True
        W_mat[1, 0] = True
    
    # Grid initialization is now in global scope

    start_time = time.time()
    firewall_triggered = False
    
    # Calculate H_0 (initial total energy)
    # H = sum(gamma * m_0)
    # This Hamiltonian tracks total relativistic energy to ensure that discrete mesh integration
    # does not result in non-physical kinematic drift over time.
    p_squared_0 = torch.sum(current_state[:, 4:7]**2, dim=1)
    gamma_0_vec = torch.sqrt(1.0 + p_squared_0 / (current_state[:, 7]**2 * C**2))
    H_0 = torch.sum(gamma_0_vec * current_state[:, 7]).item() + 1

    for tick in range(ticks):
        # 1. 3D Convolution (Laplacian) (cuDNN optimized)
        laplacian = torch.nn.functional.conv3d(phi_curr, laplacian_kernel, padding=1)
        
        # 2. Wave Equation Update (In-place to prevent allocation)
        torch.mul(phi_curr, 2.0, out=phi_next)
        phi_next.sub_(phi_prev)
        phi_next.add_(laplacian, alpha=C_SQUARED)
        
        # 3. Apply Torsional Wake Decay (Linger)
        phi_next.mul_(TORSION_DECAY)
        
        # 4. Apply Obstacle Mask (Copper walls block wave)
        phi_next.mul_(obstacle_mask)
        
        # 5. Shift state buffers (zero-allocation swap)
        temp = phi_prev
        phi_prev = phi_curr
        phi_curr = phi_next
        phi_next = temp
        
        pos = current_state[:, 1:4]
        mom = current_state[:, 4:7]
        m0 = current_state[:, 7].unsqueeze(1)
        hue = current_state[:, 8]
        
        # --- EULERIAN CONTINUOUS EMISSION (Vectorized) ---
        if getattr(args, 'photon_emission', 0):
            indices = ((pos - GRID_MIN) / DX).long()  # (N, 3)
            valid = (indices[:, 0] >= 0) & (indices[:, 0] < GRID_RES) & \
                    (indices[:, 1] >= 0) & (indices[:, 1] < GRID_RES) & \
                    (indices[:, 2] >= 0) & (indices[:, 2] < GRID_RES)
            if valid.any():
                v_idx = indices[valid]  # (M, 3)
                v_mom = mom[valid]      # (M, 3)
                v_m0 = m0[valid]        # (M, 1)
                v_hue = hue[valid]      # (M,)
                
                # Mathematically pure: gamma * m0 = sqrt(m0^2 + p^2/c^2). Avoids inf * 0 = NaN when m0 -> 0.
                energy_term = torch.sqrt(v_m0**2 + torch.sum(v_mom**2, dim=1, keepdim=True) / (C**2))
                emission = (10.0 * energy_term).squeeze(-1) * torch.cos(v_hue)
                flat_idx = v_idx[:, 0] * GRID_RES * GRID_RES + v_idx[:, 1] * GRID_RES + v_idx[:, 2]
                phi_flat = phi_curr[0, 0].view(-1)
                phi_flat.scatter_add_(0, flat_idx, emission)
        
        p_squared = torch.sum(mom**2, dim=1, keepdim=True)
        gamma = torch.sqrt(1.0 + p_squared / (m0**2 * C**2))
        current_state[:, 9] = gamma.squeeze()
        
        vel = mom / (gamma * m0)
        # Calculate forces
        force = torch.zeros_like(pos)
        
        if args.pilot_wave:
            # dBB: particles are guided ONLY by the wave field.
            # No inter-particle forces — skip the O(N^2) computation entirely.
            torsion_force = torch.zeros_like(pos)
            pauli_force = torch.zeros_like(pos)
            damping_force = -LAMBDA_VAC * vel if args.vacuum_enabled else torch.zeros_like(pos)
            grav_force = torch.zeros_like(pos)
        else:
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist_sq = torch.sum(diff**2, dim=2) + 0.1  # Plummer softening (lattice-scale UV cutoff, matches gravity channel L802)
            
            v_diff = vel.unsqueeze(1) - vel.unsqueeze(0)
            
            if args.torsion_enabled:
                torsion_force = torch.sum(TORSION_G * diff * torch.cross(diff, v_diff, dim=2) / dist_sq.unsqueeze(2)**2, dim=1)
            else:
                torsion_force = torch.zeros_like(pos)
            
            if args.pauli_enabled:
                hue_diff = current_state[:, 8].unsqueeze(1) - current_state[:, 8].unsqueeze(0)
                
                # Spatial Phase Component: p * r
                m0_tensor = current_state[:, 7].unsqueeze(1)     # (N, 1)
                gamma_tensor = current_state[:, 9].unsqueeze(1)  # (N, 1)
                p_tensor = m0_tensor * gamma_tensor * vel        # (N, 3)
                
                # Absolute spatial phase: p_i * pos_i
                spatial_phase = torch.sum(p_tensor * pos, dim=1).unsqueeze(1) # (N, 1)
                
                total_phase_diff = hue_diff + spatial_phase
                pauli_phase_coupling = torch.cos(total_phase_diff)

                
                # Spin-vorticity coupling: (omega_i . omega_j)
                # Paper 1, Sec. 3C: F_pauli = chi * cos(dtheta) * (w_i . w_j) / r^n * r_hat
                if args.spin_coupling:
                    spin_dot = spin.unsqueeze(1) * spin.unsqueeze(0)  # (N, N) outer product of scalars
                    pauli_phase_coupling = pauli_phase_coupling * spin_dot
                    
                    # Removed m0 @ m0.T override; we need cos(hue_diff) to drive the carrier wave
                
                # Configurable power law: dist_sq**((n+1)/2) where n is the force power
                # n=2 (Paper 1): diff / dist_sq^1.5 = r_hat / r^2
                # n=3 (KK/Code): diff / dist_sq^2.0 = r_hat / r^3
                power_exp = (args.pauli_power + 1) / 2.0
                
                # --- RANK-2 TRANSVERSE-TRACELESS TENSOR INJECTION ---
                force_vector = diff
                if args.polarization_mode != "isotropic":
                    # For gravitational waves, the force acts as a transverse tidal strain
                    # projecting perpendicular to the propagation axis (Z).
                    x_comp = diff[:, :, 0]
                    y_comp = diff[:, :, 1]
                    z_comp = diff[:, :, 2]
                    
                    if args.polarization_mode == 'plus':
                        tt_diff = torch.stack([x_comp, -y_comp, torch.zeros_like(z_comp)], dim=2)
                    elif args.polarization_mode == 'cross':
                        tt_diff = torch.stack([y_comp, x_comp, torch.zeros_like(z_comp)], dim=2)
                    elif args.polarization_mode == 'mixed':
                        tt_diff = torch.stack([x_comp + y_comp, x_comp - y_comp, torch.zeros_like(z_comp)], dim=2)
                    else:
                        tt_diff = diff
                        
                    force_vector = tt_diff
                    
                pauli_force_ij = PAULI_SCALAR * pauli_phase_coupling.unsqueeze(2) * force_vector / dist_sq.unsqueeze(2)**power_exp
                if args.mode == "einstein-stick":
                    # Mask out stick-stick interactions (i>0 and j>0)
                    stick_mask = torch.ones((N, N, 1), device=device)
                    stick_mask[1:, 1:, :] = 0.0
                    pauli_force_ij = pauli_force_ij * stick_mask
                pauli_force = torch.sum(pauli_force_ij, dim=1)
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
            elif args.mode in ["direct-collapse", "holographic", "holographic-shell", "holographic-ring", "ads-cft"]:
                # Full N-Body Newtonian Gravity calculation (Vectorized)
                G_local = args.collapse_G if args.mode == "direct-collapse" else 100.0
                diff_g = pos.unsqueeze(1) - pos.unsqueeze(0)
                d_sq = torch.sum(diff_g**2, dim=2) + 0.1 # Softening parameter
                force_ij = -G_local * m0.view(N, 1, 1) * m0.view(1, N, 1) * diff_g / (d_sq.unsqueeze(2) ** 1.5)
                # Mask out self-interaction (diagonal)
                mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(2)
                force_ij.masked_fill_(mask, 0.0)
                grav_force = torch.sum(force_ij, dim=1)
            elif args.mode == "photoelectric":
                # Apply restoring spring force (work function) to electron at origin
                grav_force = torch.zeros_like(pos)
                dx = pos[0, 0] - 0.0
                dy = pos[0, 1] - 0.0
                dz = pos[0, 2] - 0.0
                
                # Hooke's Law towards origin
                grav_force[0, 0] = -args.work_function * dx
                grav_force[0, 1] = -args.work_function * dy
                grav_force[0, 2] = -args.work_function * dz
                
                # Simple interaction to push electron
                if pos[0, 0] > 0.1: grav_force[0] = 0.0
            elif args.mode == "einstein-stick":
                grav_force = torch.zeros_like(pos)
                k = args.work_function
                d0 = args.slit_separation
                
                # Extract phase coupling from the Pauli module for ER=EPR stiffness
                hue_diff_local = current_state[:, 8].unsqueeze(1) - current_state[:, 8].unsqueeze(0)
                phase_coupling = torch.cos(hue_diff_local)
                
                # Spring force between adjacent stick particles
                for i in range(N):
                    if i < N - 1:
                        dx_right = pos[i+1] - pos[i]
                        dist_right = torch.norm(dx_right)
                        dir_right = dx_right / (dist_right + 1e-6)
                        
                        if args.entangled:
                            # Entangled rod: Phase alignment stiffens the structural bonds (Wormhole tension)
                            # We use the teleparallel phase difference to modulate elasticity.
                            effective_k = k * (1.0 + 10.0 * torch.clamp(phase_coupling[i, i+1], min=0.0))
                        else:
                            effective_k = k
                            
                        force_right = effective_k * (dist_right - d0) * dir_right
                        grav_force[i] += force_right
                        
                    if i > 0:
                        dx_left = pos[i-1] - pos[i]
                        dist_left = torch.norm(dx_left)
                        dir_left = dx_left / (dist_left + 1e-6)
                        
                        if args.entangled:
                            effective_k_left = k * (1.0 + 10.0 * torch.clamp(phase_coupling[i, i-1], min=0.0))
                        else:
                            effective_k_left = k
                            
                        force_left = effective_k_left * (dist_left - d0) * dir_left
                        grav_force[i] += force_left
            elif args.mode == "qed":
                # Quantum Electrodynamics continuous potentials for SINDy extraction
                grav_force = torch.zeros_like(pos)
                diff_g = pos.unsqueeze(1) - pos.unsqueeze(0)
                d_sq = torch.sum(diff_g ** 2, dim=2) + 1e-6
                d = torch.sqrt(d_sq)
                
                hue = current_state[:, 8]
                hue_diff = hue.unsqueeze(1) - hue.unsqueeze(0)
                
                # Base Coulomb Force (U(1) phase mediated: attractive if opposite phase)
                F_coulomb = 500.0 * torch.cos(hue_diff) / d_sq
                
                # 1. Vacuum Polarization (Uehling Potential Screening)
                if args.qed_vacpol:
                    F_coulomb *= (1.0 + 0.5 * torch.exp(-2.0 * d))
                
                force_ij = F_coulomb.unsqueeze(2) * (diff_g / d.unsqueeze(2))
                grav_force = torch.sum(force_ij, dim=1)
                
                # 2. Lamb Shift (Self-Energy Zitterbewegung Jitter)
                if args.qed_lamb:
                    jitter_amp = 0.1
                    # Jitter dependent on own phase and momentum
                    jitter = jitter_amp * torch.sin(50.0 * hue).unsqueeze(1) * current_state[:, 4:7]
                    grav_force += jitter
                    
                # 3. Compton Scattering (Radiation Pressure Wave)
                if args.qed_compton:
                    omega = 10.0
                    k_wave = 2.0
                    t_current = tick * DT
                    wave_phase = omega * t_current - k_wave * pos[:, 0]
                    rad_force = 150.0 * torch.sin(wave_phase)
                    # Apply in +x direction
                    rad_vec = torch.zeros_like(pos)
                    rad_vec[:, 0] = rad_force
                    grav_force += rad_vec
            else:
                grav_force = torch.zeros_like(pos)
            
        # Harmonic confining potential for galactic spin
        confining_force = torch.zeros_like(pos)
        if args.mode == "antenna" and args.galactic_spin > 0.0:
            omega = args.galactic_spin
            # pos[:, 0:2] is x and y
            confining_force[:, 0:2] = -m0 * (omega**2) * pos[:, 0:2]
            # Center source should not be confined by this since it's at r=0 anyway
        
        # --- EULERIAN PILOT WAVE GUIDANCE ---
        pilot_wave_force = torch.zeros_like(pos)
        if args.pilot_wave:
            # dBB mode: vectorised gradient interpolation at continuous particle positions
            from utils import trilinear_interpolate_gradient
            grad_at_particle = trilinear_interpolate_gradient(
                phi_curr, pos, GRID_MIN, GRID_MAX, GRID_RES, DX
            )
            # Guidance equation: dp/dt = (hbar/m) * grad(phi)
            # In our units hbar=1, so coupling = 1/m0 (or raw gradient for massless)
            m0_safe = m0.clamp(min=1e-6)
            pilot_wave_force = (1.0 / m0_safe) * grad_at_particle * 50.0
        
        total_force = torsion_force + pauli_force + damping_force + grav_force + confining_force + pilot_wave_force
        
        # --- RELATIVISTIC INTEGRATION ---
        # Update momentum from forces natively. 
        # Total force and momentum are allowed to rise organically. 
        # Singularities will be natively dissipated by the AMPS Cooler trigger.
        current_state[:, 4:7] = mom + total_force * DT
        p_new = current_state[:, 4:7]
        p_new_sq = torch.sum(p_new**2, dim=1, keepdim=True)
        
        # --- FREE MOMENTUM INTEGRATION (Manuscript 2) ---
        # No artificial clamp. The relativistic relation v = p/(gamma*m0)
        # naturally guarantees |v| < c for any finite momentum.
        # Required for un-clamped firewall testing where gamma must grow organically.
        gamma_new = torch.sqrt(1.0 + p_new_sq / (m0**2 * C**2))
        current_state[:, 9] = gamma_new.squeeze()
        vel_new = p_new / (gamma_new * m0)
        
        # Free memory cache periodically to prevent OOM
        if tick > 0 and tick % 1000 == 0:
            torch.cuda.empty_cache()
            
        # Position and Phase Updates (shared by both modes)
        current_state[:, 1:4] += vel_new * DT
        current_state[:, 8] = (current_state[:, 8] + (m0.squeeze() / gamma_new.squeeze()) * DT) % (2 * np.pi)
        
        # --- ER=EPR ENTANGLEMENT SYNC ---
        if args.entangled and args.mode in ["einstein-stick", "gravity-sink"]:
            # Kuramoto synchronization for entangled particles
            hue_diff_sync = current_state[:, 8].unsqueeze(1) - current_state[:, 8].unsqueeze(0)
            sync_force = torch.sum(W_mat * torch.sin(hue_diff_sync), dim=1)
            current_state[:, 8] = (current_state[:, 8] + 50.0 * sync_force * DT) % (2 * np.pi)
            
            # --- AMPS FIREWALL DECOHERENCE (TOPOLOGICAL SNAP) ---
            if args.mode == "gravity-sink":
                gamma_diff = torch.abs(gamma_new.squeeze().unsqueeze(1) - gamma_new.squeeze().unsqueeze(0))
                # Identify connected pairs exceeding the critical pair-production threshold
                snap_mask = W_mat & (gamma_diff > 4.0)
                if snap_mask.any():
                    snapped_indices = torch.nonzero(snap_mask, as_tuple=False)
                    for pair in snapped_indices:
                        i, j = pair[0].item(), pair[1].item()
                        if W_mat[i, j]:  # If still connected
                            W_mat[i, j] = False
                            W_mat[j, i] = False
                            print(f"*** AMPS FIREWALL TRIGGERED! WORMHOLE BOND SNAPPED between {i} and {j} ***")
                            print(f"*** TENSION DIFFERENTIAL CRITICAL (dGamma = {gamma_diff[i, j].item():.3f}) ***")
                            # Deposit 2*m0 into both particles
                            m0_val_i = current_state[i, 7].item()
                            m0_val_j = current_state[j, 7].item()
                            current_state[i, 7] += 2.0 * m0_val_i
                            current_state[j, 7] += 2.0 * m0_val_j
                            # Update local m0 so it reflects properly
                            m0[i, 0] = current_state[i, 7]
                            m0[j, 0] = current_state[j, 7]
                            print(f"*** ENERGY SPIKE AT EVENT HORIZON: Particle {i} m_0 -> {current_state[i, 7].item():.2f} MeV ***")
                            print(f"*** ENERGY SPIKE AT EVENT HORIZON: Particle {j} m_0 -> {current_state[j, 7].item():.2f} MeV ***")

        if args.mode == "ads-cft":
            # Kuramoto phase synchronization across all ER=EPR bridges
            # This drives phase coherence between boundary (CFT) and bulk (AdS)
            hue_diff_sync = current_state[:, 8].unsqueeze(1) - current_state[:, 8].unsqueeze(0)
            sync_force = torch.sum(W_mat * torch.sin(hue_diff_sync), dim=1)
            current_state[:, 8] = (current_state[:, 8] + 20.0 * sync_force * DT) % (2 * np.pi)
        
        if args.mode == "photoelectric":
            # Override photon phase clock with specified frequency
            # To avoid the standard mass/gamma slowdown, we force its ticking rate
            current_state[1, 8] = (current_state[1, 8] + args.photon_freq * DT) % (2 * np.pi)
        elif args.mode == "antenna":
            if antenna_signal is not None:
                if len(antenna_signal.shape) == 1:
                    # --- RESIDUAL MODE: 1D signal drives the SOURCE particle (index -1) ---
                    if tick < len(antenna_signal):
                        current_state[-1, 8] = antenna_signal[tick].item()
                    else:
                        current_state[-1, 8] = 0.0
                else:
                    # --- MULTI-ANCHOR MODE: matrix drives pulsar hues ---
                    N_anchors = args.num_anchors
                    if tick < len(antenna_signal):
                        current_state[:N_anchors, 8] = antenna_signal[tick, :N_anchors] % (2 * np.pi)
                    else:
                        current_state[:N_anchors, 8] = 0.0
            else:
                # Without an external dataset, artificially drive the central source particle (index -1)
                # to create a continuous topological gravitational wave (0.5 Hz)
                # We use a small amplitude oscillation so SINDy's polynomial library can Taylor expand cos(hue)
                t_val = current_state[-1, 0].item()
                current_state[-1, 8] = 0.5 * np.sin(1.57 * t_val)

        current_state[:, 0] += DT

        # --- DOUBLE-SLIT: Hard wall reflection + screen detection ---
        # The wall is a solid plate — particles can only pass through slit openings.
        # A slit opening requires BOTH correct Y (within slit bounds) AND correct Z (within slit height).
        if args.mode == "double-slit":
            sw = args.slit_width
            ss = args.slit_separation
            s1_lo = ss / 2.0 - sw / 2.0
            s1_hi = ss / 2.0 + sw / 2.0
            s2_lo = -ss / 2.0 - sw / 2.0
            s2_hi = -ss / 2.0 + sw / 2.0
            slit_z_h = args.slit_width  # Slit Z-height = slit width (square cross-section)
            for bi in range(N_beam):
                if bi in screen_detected:
                    continue  # Already hit the screen, skip
                bx = current_state[bi, 1].item()
                by = current_state[bi, 2].item()
                bz = current_state[bi, 3].item()
                # --- SCREEN DETECTION (must check BEFORE wall reflection) ---
                if bx > SCREEN_X:
                    screen_detected[bi] = by  # Record y at screen
                    # Freeze particle: stop it from interacting further
                    current_state[bi, 4:7] = 0.0  # Zero momentum
                    current_state[bi, 1] = SCREEN_X + 0.1  # Park at screen
                    continue
                # --- WALL REFLECTION (solid plate with rectangular slit openings) ---
                if bx > WALL_X:  # Crossed the wall plane
                    in_s1 = s1_lo <= by <= s1_hi
                    in_s2 = s2_lo <= by <= s2_hi
                    in_slit_z = abs(bz) < slit_z_h  # Within slit Z-height
                    # Particle passes ONLY if in a slit Y-range AND within slit Z-height
                    passes_through = (in_s1 or in_s2) and in_slit_z
                    if not passes_through:
                        # HIT THE SOLID WALL — reflect back
                        current_state[bi, 1] = WALL_X - 0.01
                        current_state[bi, 4] = -abs(current_state[bi, 4].item())

        if save_history and tick % SAVE_INTERVAL == 0:
            state_snap = current_state.cpu().numpy().copy()
            history.append(state_snap)
            
            # --- ZMQ Stream Flush (dynamically scaled to prevent GPU→CPU bottleneck) ---
            if zmq_socket is not None and tick % ZMQ_TICK_RATE == 0:
                chunk_buffer.append(state_snap)
                if len(chunk_buffer) >= args.zmq_flush_rate:
                    zmq_socket.send_pyobj({
                        "status": "STREAMING",
                        "chunk_id": tick,
                        "data": np.array(chunk_buffer) # Shape: (flush_rate, N, 10)
                    })
                    # -- NEW: Crash Checkpoint --
                    np.savez("checkpoint_latest.npz", chunk=np.array(chunk_buffer), tick=tick)
                    chunk_buffer = [] # Clear buffer to free local RAM
            
            if tick % 1000 == 0:
                print(f"Collision Tick {tick}/{ticks} - Gamma Max: {gamma.max().item():.3f}")
                sys.stdout.flush()
        
        # Early stopping for single-particle double-slit (Tonomura protocol)
        if args.mode == "double-slit" and N_beam == 1:
            if len(screen_detected) > 0:
                break  # Particle hit the detector screen
            bx0 = current_state[0, 1].item()
            bpx0 = current_state[0, 4].item()
            if bx0 < -30.0 and bpx0 < 0:
                break  # Reflected and moving away — won't return
                
    elapsed = time.time() - start_time
    if save_history:
        print(f"Loop Complete! Elapsed Time: {elapsed:.2f}s")
        print(f"TOTAL_RADIATION_SHED={total_radiation_shed:.6f}")
        
        print("\n--- SIMULATION RESOURCE USAGE ---")
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f"CPU RAM USED     : {ram_mb:.2f} MB")
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"GPU VRAM USED    : {vram_mb:.2f} MB")

    # --- ZMQ Post-Loop Flush ---
    if zmq_socket is not None and len(chunk_buffer) > 0:
        zmq_socket.send_pyobj({
            "status": "STREAMING",
            "chunk_id": -1,
            "data": np.array(chunk_buffer)
        })
        print(f"[ZMQ] Flushed final {len(chunk_buffer)} remaining chunks from run_ticks")
        sys.stdout.flush()
        chunk_buffer = []

    # --- EULERIAN GRID EXPORT ---
    phi_slice = phi_curr[0, 0, :, :, GRID_RES // 2].cpu().numpy()
    np.save("pilot_wave_slice.npy", phi_slice)

    return history, screen_detected, W_mat

if args.mode in ["double-slit", "quantum-eraser", "heat-sink-eraser"]:
    # =================================================================
    # TONOMURA PROTOCOL: Single-Particle Double-Slit
    # =================================================================
    
    N_trials = N_beam  # Total particles to fire (from --num_particles)
    if args.mode == "quantum-eraser":
        N_single = 2 + N_wall + 10 # 2 beams + N_wall for slit + 10 for detector wall
    elif args.mode == "heat-sink-eraser":
        N_single = 1 + N_wall + 1 # 1 beam + N_wall + 1 heat sink
    else:
        N_single = 1 + N_wall  # particles per trial: 1 beam + wall
    
    # --- BREIT-WHEELER VACUUM BUFFER ---
    # Pre-allocate dormant slots for emergent pair production.
    # These rows sit at the end of the state tensor and are activated
    # when the FDTD grid amplitude crosses the topological snap threshold.
    bw_start_idx = N_single  # First BW slot index (right after all existing particles)
    if BW_ENABLED:
        N_single += BW_BUFFER_SIZE
        print(f"  [BW] N_single expanded: {bw_start_idx} base + {BW_BUFFER_SIZE} vacuum buffer = {N_single} total")

    
    # Resolve batch size: 0=auto, 1=sequential, >1=batched
    if args.batch_size == 0:
        BATCH_SIZE = 100 if torch.cuda.is_available() else 1
    else:
        BATCH_SIZE = args.batch_size
    
    USE_BATCH = BATCH_SIZE > 1
    
    aggregate_screen_hits = []
    aggregate_reflected = 0
    aggregate_slit_used = []
    aggregate_final_states = []    # Full 10D state at screen hit or end
    aggregate_initial_states = []  # Initial beam y + hue for correlation
    aggregate_outcomes = []        # 1=hit, -1=reflected
    aggregate_spins = []           # Beam spin ±0.5 for each trial
    representative_history = None
    
    np.random.seed(42)  # Reproducible sequence
    
    beam_start_x = -10.0
    beam_y_range = args.slit_separation / 2.0 + args.slit_width / 2.0 + 2.0
    
    # 3D Beam Spread: collimated beam with minimal Z-spread (real electron guns)
    # Jönsson's beam was highly collimated — electrons travel nearly parallel.
    # Z-spread should be small relative to slit dimensions.
    slit_z_half = args.slit_width * 50  # Match wall construction
    beam_z_range = min(args.slit_width * 0.5, 2.0)  # Collimated: ±2.0 max Z-spread

    WALL_HUE = 0.0  # Fixed reference phase
    WALL_SPIN = 0.5  # Wall atoms: all spin-up
    m_per_particle = args.wall_mass
    
    # Slit boundaries (used by both paths)
    sw = args.slit_width
    ss = args.slit_separation
    s1_lo = ss / 2.0 - sw / 2.0
    s1_hi = ss / 2.0 + sw / 2.0
    s2_lo = -ss / 2.0 - sw / 2.0
    s2_hi = -ss / 2.0 + sw / 2.0
    s1_center = ss / 2.0
    s2_center = -ss / 2.0
    
    start_all = time.time()
    
    if USE_BATCH:
        # =============================================================
        # GPU BATCH PATH: Run B trials in parallel per batch
        # =============================================================
        n_batches = (N_trials + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"--- TONOMURA PROTOCOL (GPU BATCH): {N_trials} trials in {n_batches} batches of {BATCH_SIZE} ---")
        print(f"  Device: {device} | Each batch: {BATCH_SIZE} beams (m={args.mass_a}) + {N_wall} wall (m={m_per_particle:.0e})")
        
        trials_done = 0
        zmq_chunk_count = 0
        
        for batch_idx in range(n_batches):
            B = min(BATCH_SIZE, N_trials - trials_done)
            
            # --- BUILD BATCH STATE: (B, N_single, 10) ---
            batch_state = torch.zeros((B, N_single, 10), device=device, dtype=torch.float32)
            batch_spin = torch.zeros((B, N_single), device=device, dtype=torch.float32)
            
            # Beam particles (index 0 in each trial)
            trial_ys = np.random.uniform(-beam_y_range, beam_y_range, size=B)
            trial_zs = np.random.uniform(-beam_z_range, beam_z_range, size=B) if beam_z_range > 0 else np.zeros(B)
            trial_hues = np.zeros(B)  # Coherent phase for shared pilot wave
            batch_state[:, 0, 1] = beam_start_x                                    # x
            batch_state[:, 0, 2] = torch.tensor(trial_ys, device=device, dtype=torch.float32)  # y
            batch_state[:, 0, 3] = torch.tensor(trial_zs, device=device, dtype=torch.float32)  # z
            batch_state[:, 0, 4] = args.beam_momentum                              # px
            batch_state[:, 0, 7] = args.mass_a                                     # m0
            batch_state[:, 0, 8] = torch.tensor(trial_hues, device=device, dtype=torch.float32)  # hue
            if args.spin_coupling:
                batch_spin[:, 0] = torch.where(torch.rand(B, device=device) > 0.5,
                                                torch.tensor(0.5, device=device),
                                                torch.tensor(-0.5, device=device))
            
            wall_start_idx = 1
            if args.mode == "quantum-eraser":
                batch_state[:, 1, 1] = beam_start_x
                batch_state[:, 1, 2] = torch.tensor(trial_ys, device=device, dtype=torch.float32)
                batch_state[:, 1, 3] = torch.tensor(trial_zs, device=device, dtype=torch.float32)
                batch_state[:, 1, 4] = -args.beam_momentum
                batch_state[:, 1, 7] = args.mass_a
                batch_state[:, 1, 8] = torch.tensor(trial_hues, device=device, dtype=torch.float32)
                if args.spin_coupling:
                    batch_spin[:, 1] = -batch_spin[:, 0]
                wall_start_idx = 2
            
            # Wall particles
            for j, (wx, wy, wz) in enumerate(wall_positions):
                idx = wall_start_idx + j
                batch_state[:, idx, 1] = wx
                batch_state[:, idx, 2] = wy
                batch_state[:, idx, 3] = wz
                batch_state[:, idx, 7] = m_per_particle
                if args.wall_thermal_phase == 1:
                    batch_state[:, idx, 8] = torch.rand(B, device=device, dtype=torch.float32) * 2 * np.pi
                else:
                    batch_state[:, idx, 8] = WALL_HUE
                if args.spin_coupling:
                    batch_spin[:, idx] = WALL_SPIN
                
            if args.mode == "quantum-eraser" and not args.eraser_active:
                detector_start_idx = wall_start_idx + len(wall_positions)
                for j in range(10):
                    idx = detector_start_idx + j
                    batch_state[:, idx, 1] = -30.0
                    batch_state[:, idx, 2] = -5.0 + j * 1.0
                    batch_state[:, idx, 3] = 0.0
                    batch_state[:, idx, 7] = 50000.0
                    if args.wall_thermal_phase == 1:
                        batch_state[:, idx, 8] = torch.rand(B, device=device, dtype=torch.float32) * 2 * np.pi
                    else:
                        batch_state[:, idx, 8] = WALL_HUE
            
            if args.mode == "heat-sink-eraser":
                sink_idx = wall_start_idx + len(wall_positions)
                batch_state[:, sink_idx, 1] = 0.0
                batch_state[:, sink_idx, 2] = 50.0  # Far away
                batch_state[:, sink_idx, 3] = 0.0
                batch_state[:, sink_idx, 7] = 50000.0  # Huge mass
                batch_state[:, sink_idx, 8] = torch.rand(B, device=device, dtype=torch.float32) * 2 * np.pi

            save_first = (batch_idx == 0)
            history = [] if save_first else None
            chunk_buffer = [] # ZMQ Stream buffer for batch
            trial_status = torch.zeros(B, device=device, dtype=torch.int32)
            screen_y = torch.zeros(B, device=device, dtype=torch.float32)
            
            initial_y = batch_state[:, 0, 2].cpu().numpy().copy()
            initial_z = batch_state[:, 0, 3].cpu().numpy().copy()
            initial_hue = batch_state[:, 0, 8].cpu().numpy().copy()
            
            # --- BATCHED PHYSICS LOOP ---
            for tick in range(TOTAL_TICKS):
                active = (trial_status == 0)
                if not active.any():
                    break
                
                # --- EULERIAN GRID WAVE PROPAGATION ---
                laplacian = torch.nn.functional.conv3d(phi_curr, laplacian_kernel, padding=1)
                torch.mul(phi_curr, 2.0, out=phi_next)
                phi_next.sub_(phi_prev)
                phi_next.add_(laplacian, alpha=C_SQUARED)
                phi_next.mul_(TORSION_DECAY)
                phi_next.mul_(obstacle_mask)
                temp = phi_prev
                phi_prev = phi_curr
                phi_curr = phi_next
                phi_next = temp
                
                # --- BREIT-WHEELER TOPOLOGICAL SNAPPING ---
                # Scan the FDTD grid for cells exceeding the pair-production threshold.
                # When found, spawn an electron-positron pair into the vacuum buffer
                # and drain the equivalent energy from the grid via Gaussian smoothing.
                if BW_ENABLED and bw_spawned_count < BW_BUFFER_SIZE:
                    phi_abs = phi_curr[0, 0].abs()
                    snap_mask = phi_abs > BW_THRESHOLD
                    if snap_mask.any():
                        # Get indices of ALL cells above threshold
                        snap_indices = torch.nonzero(snap_mask, as_tuple=False)  # (M, 3)
                        # Process up to available buffer slots (2 per pair)
                        n_available = (BW_BUFFER_SIZE - bw_spawned_count) // 2
                        n_snaps = min(len(snap_indices), n_available)
                        
                        if n_snaps > 0:
                            for si in range(n_snaps):
                                ix, iy, iz = snap_indices[si]
                                
                                # 1. Convert grid index to world coordinates
                                wx = GRID_MIN + (ix.float() + 0.5) * DX
                                wy = GRID_MIN + (iy.float() + 0.5) * DX
                                wz = GRID_MIN + (iz.float() + 0.5) * DX
                                
                                # 2. Calculate incident wave momentum from FDTD gradient
                                #    (Poynting vector analogue: direction the wave energy travels)
                                gx = (phi_curr[0, 0, min(ix+1, GRID_RES-1), iy, iz] - 
                                      phi_curr[0, 0, max(ix-1, 0), iy, iz]) / (2.0 * DX)
                                gy = (phi_curr[0, 0, ix, min(iy+1, GRID_RES-1), iz] - 
                                      phi_curr[0, 0, ix, max(iy-1, 0), iz]) / (2.0 * DX)
                                gz = (phi_curr[0, 0, ix, iy, min(iz+1, GRID_RES-1)] - 
                                      phi_curr[0, 0, ix, iy, max(iz-1, 0)]) / (2.0 * DX)
                                grad_vec = torch.tensor([gx, gy, gz], device=device)
                                grad_mag = grad_vec.norm() + 1e-12
                                n_wave = grad_vec / grad_mag  # Unit direction of wave
                                
                                # 3. Calculate transverse orthogonal vector for spatial separation
                                #    Find a vector perpendicular to n_wave
                                if abs(n_wave[0]) < 0.9:
                                    arbitrary = torch.tensor([1.0, 0.0, 0.0], device=device)
                                else:
                                    arbitrary = torch.tensor([0.0, 1.0, 0.0], device=device)
                                n_trans = torch.cross(n_wave, arbitrary)
                                n_trans = n_trans / (n_trans.norm() + 1e-12)
                                
                                # 4. Momentum assignment:
                                #    - Forward momentum: split wave momentum equally
                                #    - Transverse kick: excess energy (BW_THRESHOLD - BW_PAIR_COST)
                                #      converted to kinetic escape velocity along ±n_trans
                                excess_energy = BW_THRESHOLD - BW_PAIR_COST  # ~1.022 MeV
                                # p = sqrt(E^2 - m^2) for each particle (half the excess each)
                                e_each = excess_energy / 2.0
                                p_kick = (e_each**2 + 2 * e_each * BW_ELECTRON_MASS)**0.5 if e_each > 0 else 0.0
                                
                                # Forward momentum from wave (split equally)
                                p_forward = grad_mag * 0.5
                                
                                # 5. Spawn electron (slot bw_start_idx + bw_spawned_count)
                                e_idx = bw_start_idx + bw_spawned_count
                                p_idx = bw_start_idx + bw_spawned_count + 1
                                
                                # Electron: offset by +DX along n_trans
                                batch_state[:, e_idx, 1] = wx + n_trans[0] * DX  # x
                                batch_state[:, e_idx, 2] = wy + n_trans[1] * DX  # y
                                batch_state[:, e_idx, 3] = wz + n_trans[2] * DX  # z
                                batch_state[:, e_idx, 4] = n_wave[0] * p_forward + n_trans[0] * p_kick  # px
                                batch_state[:, e_idx, 5] = n_wave[1] * p_forward + n_trans[1] * p_kick  # py
                                batch_state[:, e_idx, 6] = n_wave[2] * p_forward + n_trans[2] * p_kick  # pz
                                batch_state[:, e_idx, 7] = BW_ELECTRON_MASS  # m0 = 0.511
                                batch_state[:, e_idx, 8] = 0.0  # hue = 0 (electron phase)
                                
                                # Positron: offset by -DX along n_trans, opposite transverse kick
                                batch_state[:, p_idx, 1] = wx - n_trans[0] * DX
                                batch_state[:, p_idx, 2] = wy - n_trans[1] * DX
                                batch_state[:, p_idx, 3] = wz - n_trans[2] * DX
                                batch_state[:, p_idx, 4] = n_wave[0] * p_forward - n_trans[0] * p_kick
                                batch_state[:, p_idx, 5] = n_wave[1] * p_forward - n_trans[1] * p_kick
                                batch_state[:, p_idx, 6] = n_wave[2] * p_forward - n_trans[2] * p_kick
                                batch_state[:, p_idx, 7] = BW_ELECTRON_MASS  # m0 = 0.511
                                batch_state[:, p_idx, 8] = np.pi  # hue = π (positron: out-of-phase)
                                
                                # Spins: electron spin-up, positron spin-down (conserve angular momentum)
                                batch_spin[:, e_idx] = 0.5
                                batch_spin[:, p_idx] = -0.5
                                
                                bw_spawned_count += 2
                                
                                # 6. Gaussian drain: remove BW_THRESHOLD energy from the grid
                                #    Apply 3x3x3 Gaussian kernel centered on the snap voxel
                                ix_s, iy_s, iz_s = max(0, ix-1), max(0, iy-1), max(0, iz-1)
                                ix_e, iy_e, iz_e = min(GRID_RES, ix+2), min(GRID_RES, iy+2), min(GRID_RES, iz+2)
                                # Calculate how much to drain (scale kernel to remove BW_THRESHOLD)
                                local_amp = phi_curr[0, 0, ix, iy, iz].item()
                                drain_scale = min(abs(local_amp), BW_THRESHOLD)
                                sign = 1.0 if local_amp > 0 else -1.0
                                # Extract the kernel slice that fits within grid bounds
                                kx_s, ky_s, kz_s = ix_s - (ix-1), iy_s - (iy-1), iz_s - (iz-1)
                                kx_e, ky_e, kz_e = kx_s + (ix_e - ix_s), ky_s + (iy_e - iy_s), kz_s + (iz_e - iz_s)
                                drain_patch = bw_drain_kernel[0, 0, kx_s:kx_e, ky_s:ky_e, kz_s:kz_e]
                                phi_curr[0, 0, ix_s:ix_e, iy_s:iy_e, iz_s:iz_e] -= sign * drain_scale * drain_patch
                                
                            if n_snaps > 0:
                                print(f"  [BW SNAP] Tick {tick}: {n_snaps} pair(s) spawned at grid hotspots | Total BW particles: {bw_spawned_count}")
                
                # Lagrangian force calc uses only base particles (beam + wall),
                # NOT the BW vacuum buffer (those update ballistically below).
                # bw_start_idx == N_single when BW is disabled, so this is always correct.
                n_base = bw_start_idx  # number of real particles (excludes BW buffer)
                pos = batch_state[:, :n_base, 1:4]
                mom = batch_state[:, :n_base, 4:7]
                m0 = batch_state[:, :n_base, 7:8]
                p_sq = torch.sum(mom**2, dim=2, keepdim=True)
                gamma = torch.sqrt(1.0 + p_sq / (m0.clamp(min=1e-6)**2 * C**2))
                vel = mom / (gamma * m0.clamp(min=1e-6))
                
                N_moving = 2 if (args.mode == "quantum-eraser" or (args.mode == "double-slit" and args.eraser_active)) else 1
                if args.mode == "qed":
                    N_moving = 1 # Proton is fixed
                pos_moving = pos[:, :N_moving, :]
                vel_moving = vel[:, :N_moving, :]
                mom_moving = mom[:, :N_moving, :]
                m0_moving = m0[:, :N_moving, :]
                
                # --- EULERIAN CONTINUOUS EMISSION (Vectorized) ---
                if getattr(args, 'photon_emission', 0):
                    indices = ((pos_moving - GRID_MIN) / DX).long()  # (B, N_moving, 3)
                    valid = (indices[:, :, 0] >= 0) & (indices[:, :, 0] < GRID_RES) & \
                            (indices[:, :, 1] >= 0) & (indices[:, :, 1] < GRID_RES) & \
                            (indices[:, :, 2] >= 0) & (indices[:, :, 2] < GRID_RES)
                    if valid.any():
                        v_idx = indices[valid]  # (M, 3)
                        v_mom = mom_moving[valid]      # (M, 3)
                        v_m0 = m0_moving[valid]        # (M, 1)
                        v_hue = batch_state[:, :N_moving, 8][valid]      # (M,)
                        
                        # Mathematically pure: gamma * m0 = sqrt(m0^2 + p^2/c^2). Avoids inf * 0 = NaN when m0 -> 0.
                        energy_term = torch.sqrt(v_m0**2 + torch.sum(v_mom**2, dim=1, keepdim=True) / (C**2))
                        emission = (10.0 * energy_term).squeeze(-1) * torch.cos(v_hue) / B  # (M,)
                        flat_idx = v_idx[:, 0] * GRID_RES * GRID_RES + v_idx[:, 1] * GRID_RES + v_idx[:, 2]
                        phi_flat = phi_curr[0, 0].view(-1)
                        phi_flat.scatter_add_(0, flat_idx, emission)

                diff = pos_moving.unsqueeze(2) - pos.unsqueeze(1)
                dist_sq = torch.sum(diff**2, dim=3) + 1e-6
                
                if args.torsion_enabled:
                    v_diff = vel_moving.unsqueeze(2) - vel.unsqueeze(1)
                    cross = torch.cross(diff, v_diff, dim=3)
                    torsion_force = torch.sum(TORSION_G * diff * cross / dist_sq.unsqueeze(3)**2, dim=2)
                else:
                    torsion_force = torch.zeros_like(pos_moving)
                
                if args.pauli_enabled:
                    hue = batch_state[:, :n_base, 8]
                    hue_moving = hue[:, :N_moving]
                    hue_diff = hue_moving.unsqueeze(2) - hue.unsqueeze(1)
                    
                    # Spatial Phase Component: p * r
                    p_moving = m0[:, :N_moving] * gamma[:, :N_moving] * vel_moving  # (B, N_moving, 3)
                    
                    # Absolute spatial phase: p * x
                    pos_moving = batch_state[:, :N_moving, 1:4]
                    spatial_phase = torch.sum(p_moving * pos_moving, dim=2) # (B, N_moving)
                    
                    total_phase_diff = hue_diff + spatial_phase.unsqueeze(2)
                    phase_coupling = torch.cos(total_phase_diff)
                    if args.spin_coupling:
                        spin_moving = batch_spin[:, :N_moving]
                        spin_dot = spin_moving.unsqueeze(2) * batch_spin[:, :n_base].unsqueeze(1)
                        phase_coupling = phase_coupling * spin_dot
                    # Base power exponent for beam-beam interactions
                    base_power_exp = (args.pauli_power + 1) / 2.0
                    
                    # Create a tensor for the power exponent that varies by target particle j
                    power_exp_tensor = torch.full((1, 1, n_base), base_power_exp, device=device, dtype=torch.float32)
                    
                    # Wall particles get a macroscopic 1/r^2 profile (power_exp = 1.5)
                    if n_base > N_moving:
                        power_exp_tensor[0, 0, N_moving:] = 1.5
                        
                    pauli_force = torch.sum(PAULI_SCALAR * phase_coupling.unsqueeze(3) * diff / dist_sq.unsqueeze(3)**power_exp_tensor.unsqueeze(3), dim=2)
                else:
                    pauli_force = torch.zeros_like(pos_moving)
                
                if args.vacuum_enabled:
                    damping_force = -LAMBDA_VAC * vel_moving
                else:
                    damping_force = torch.zeros_like(pos_moving)
                
                # --- CMB EXPANSION NOISE ---
                if args.cmb_noise > 0.0:
                    # Inject a randomized, radially outward jitter (scaled by position to emulate spatial expansion)
                    cmb_vector = pos_moving * (torch.rand_like(pos_moving) * args.cmb_noise)
                    cmb_force = cmb_vector
                else:
                    cmb_force = torch.zeros_like(pos_moving)
                
                # --- EULERIAN PILOT WAVE GUIDANCE ---
                pilot_wave_force = torch.zeros_like(pos_moving)
                if args.pilot_wave:
                    from utils import trilinear_interpolate_gradient
                    # pos_moving is (B, N_moving, 3). Flatten to (B*N_moving, 3) for interpolation
                    pos_flat = pos_moving.reshape(-1, 3)
                    grad_at_particle_flat = trilinear_interpolate_gradient(
                        phi_curr, pos_flat, GRID_MIN, GRID_MAX, GRID_RES, DX
                    )
                    grad_at_particle = grad_at_particle_flat.view(B, N_moving, 3)
                    m0_safe = m0_moving.clamp(min=1e-6)
                    # Use args.vacuum for force coupling
                    pilot_wave_force = (1.0 / m0_safe) * grad_at_particle * (C**2) * args.vacuum
                    pilot_wave_force = torch.clamp(pilot_wave_force, min=-25000.0, max=25000.0)
                
                if DBB_GUIDANCE and getattr(args, 'photon_emission', 0):
                    # ===== de BROGLIE-BOHM GUIDANCE =====
                    # The wave gradient directly SETS the electron's transverse velocity.
                    # The electron surfs the wave — the wave IS the equation of motion.
                    # Forward (x) momentum preserved from beam. Pauli wall protection active.
                    from utils import trilinear_interpolate_gradient
                    pos_flat_dbb = pos_moving.reshape(-1, 3)
                    grad_dbb_flat = trilinear_interpolate_gradient(
                        phi_curr, pos_flat_dbb, GRID_MIN, GRID_MAX, GRID_RES, DX
                    )
                    grad_dbb = grad_dbb_flat.view(B, N_moving, 3)
                    m0_safe_dbb = m0_moving.clamp(min=1e-6)
                    
                    # dBB velocity: v = strength × (C²/m₀) × ∇φ
                    dbb_str = args.dbb_strength
                    v_dbb = dbb_str * (C**2 / m0_safe_dbb) * grad_dbb
                    
                    # Preserve forward (x) momentum from beam, override transverse (y,z)
                    new_mom = mom_moving.clone()
                    new_mom[:, :, 1:2] = m0_safe_dbb * v_dbb[:, :, 1:2]  # y from wave
                    new_mom[:, :, 2:3] = m0_safe_dbb * v_dbb[:, :, 2:3]  # z from wave
                    
                    # Macroscopic Lensing active: Pilot Wave gradient handles geometry.
                    # Discrete Pauli point-charge force is disabled to prevent premature scattering.
                    # (Hard-wall bounds handle exact slit boundary reflections below).
                    
                    batch_state[:, :N_moving, 4:7] = new_mom
                else:
                    # ===== ORIGINAL FORCE-BASED APPROACH =====
                    total_force = torsion_force + pauli_force + damping_force + cmb_force + pilot_wave_force
                    new_mom = mom_moving + total_force * DT
                    batch_state[:, :N_moving, 4:7] = new_mom
                
                p_new_sq = torch.sum(new_mom**2, dim=2, keepdim=True)
                gamma_new = torch.sqrt(1.0 + p_new_sq / (m0_moving**2 * C**2))
                vel_new = new_mom / (gamma_new * m0_moving)
                batch_state[:, :N_moving, 1:4] += vel_new * DT
                
                if args.mode == "double-slit" and args.eraser_active:
                    # Anchor the detector so it doesn't fly away
                    batch_state[:, 1, 4:7] = 0.0
                    batch_state[:, 1, 1] = WALL_X - 1.0
                    batch_state[:, 1, 2] = args.slit_separation / 2.0
                    batch_state[:, 1, 3] = 0.0

                batch_state[:, :N_moving, 8] = (batch_state[:, :N_moving, 8] + (m0_moving.squeeze(2) / gamma_new.squeeze(2)) * DT) % (2 * np.pi)
                batch_state[:, :N_moving, 9] = gamma_new.squeeze(2)
                
                # --- QUANTUM ERASER ER=EPR SYNC ---
                if args.mode == "quantum-eraser":
                    hue_sig = batch_state[:, 0, 8]
                    hue_idl = batch_state[:, 1, 8]
                    avg_x = (torch.cos(hue_sig) + torch.cos(hue_idl)) / 2.0
                    avg_y = (torch.sin(hue_sig) + torch.sin(hue_idl)) / 2.0
                    avg_hue = torch.atan2(avg_y, avg_x) % (2 * np.pi)
                    batch_state[:, 0, 8] = avg_hue
                    batch_state[:, 1, 8] = avg_hue
                
                if args.mode == "heat-sink-eraser":
                    sink_idx = wall_start_idx + len(wall_positions)
                    # Heat sink phase scrambles chaotically
                    batch_state[:, sink_idx, 8] = torch.rand(B, device=device) * 2 * np.pi
                    
                    # Tether wall (detector) atoms to heat sink
                    hue_hs = batch_state[:, sink_idx, 8].unsqueeze(1)
                    hue_wall = batch_state[:, wall_start_idx:sink_idx, 8]
                    avg_x = (torch.cos(hue_hs) + torch.cos(hue_wall)) / 2.0
                    avg_y = (torch.sin(hue_hs) + torch.sin(hue_wall)) / 2.0
                    avg_hue = torch.atan2(avg_y, avg_x) % (2 * np.pi)
                    batch_state[:, wall_start_idx:sink_idx, 8] = avg_hue
                    if not args.eraser_active:
                        idler_x = batch_state[:, 1, 1]
                        hit_wall = (idler_x <= -30.0) & active
                        if hit_wall.any():
                            batch_state[hit_wall, 1, 4:7] = 0.0
                            batch_state[hit_wall, 1, 1] = -30.01
                            batch_state[hit_wall, 1, 8] = torch.rand(hit_wall.sum(), device=device) * 2 * np.pi
                
                batch_state[:, :, 0] += DT
                
                # --- BREIT-WHEELER PARTICLE BALLISTIC UPDATE ---
                # BW-spawned pairs coast on their initial momentum (free propagation).
                # They are NOT included in the main N_moving force loop to avoid
                # disrupting the existing Lagrangian matrix math.
                if BW_ENABLED and bw_spawned_count > 0:
                    bw_slice = slice(bw_start_idx, bw_start_idx + bw_spawned_count)
                    bw_mom = batch_state[:, bw_slice, 4:7]
                    bw_m0 = batch_state[:, bw_slice, 7:8]
                    bw_active = bw_m0.squeeze(2) > 0  # Only update slots that have been spawned
                    if bw_active.any():
                        bw_p_sq = torch.sum(bw_mom**2, dim=2, keepdim=True)
                        bw_gamma = torch.sqrt(1.0 + bw_p_sq / (bw_m0.clamp(min=1e-6)**2 * C**2))
                        bw_vel = bw_mom / (bw_gamma * bw_m0.clamp(min=1e-6))
                        batch_state[:, bw_slice, 1:4] += bw_vel * DT
                        # Phase evolution: de Broglie phase = m0/gamma * dt
                        batch_state[:, bw_slice, 8] = (batch_state[:, bw_slice, 8] + (bw_m0.squeeze(2) / bw_gamma.squeeze(2)) * DT) % (2 * np.pi)
                        batch_state[:, bw_slice, 9] = bw_gamma.squeeze(2)
                
                # --- SCREEN DETECTION + WALL REFLECTION ---
                bx = batch_state[:, 0, 1]
                by = batch_state[:, 0, 2]
                screen_mask = (bx > SCREEN_X) & active
                if screen_mask.any():
                    trial_status[screen_mask] = 1
                    screen_y[screen_mask] = by[screen_mask]
                    batch_state[screen_mask, 0, 4:7] = 0.0
                    batch_state[screen_mask, 0, 1] = SCREEN_X + 0.1
                
                wall_cross = (bx > WALL_X) & active & (trial_status == 0)
                if wall_cross.any():
                    bz = batch_state[:, 0, 3]
                    slit_z_h = args.slit_width  # Slit Z-height = slit width (square cross-section)
                    in_slit_y = ((by >= s1_lo) & (by <= s1_hi)) | ((by >= s2_lo) & (by <= s2_hi))
                    in_slit_z = torch.abs(bz) < slit_z_h  # Within slit Z-height
                    passes_through = in_slit_y & in_slit_z  # Must satisfy BOTH Y and Z
                    reflect_mask = wall_cross & ~passes_through
                    if reflect_mask.any():
                        batch_state[reflect_mask, 0, 1] = WALL_X - 0.01
                        batch_state[reflect_mask, 0, 4] = -torch.abs(batch_state[reflect_mask, 0, 4])
                
                if save_first and tick % SAVE_INTERVAL == 0:
                    history.append(batch_state[:20].cpu().numpy().copy())
                
                # --- ZMQ Stream Flush (Batched Protocol) ---
                if zmq_socket is not None and tick % SAVE_INTERVAL == 0:
                    if tick == 0 and batch_idx == 0:
                        print(f"[ZMQ DEBUG] Tick 0: Entering ZMQ flush path. zmq_flush_rate={args.zmq_flush_rate}")
                        sys.stdout.flush()
                    # Send trial 0's trajectory - shape (N, 10) matches SINDy server expectation
                    if batch_idx == 0:
                        chunk_buffer.append(batch_state[0].cpu().numpy().copy())
                        if len(chunk_buffer) >= args.zmq_flush_rate:
                            chunk_data = np.array(chunk_buffer)
                            zmq_socket.send_pyobj({
                                "status": "STREAMING",
                                "chunk_id": tick,
                                "batch_idx": batch_idx,
                                "data": chunk_data # Shape: (flush_rate, N, 10)
                            })
                            zmq_chunk_count += 1
                            chunk_buffer = []
            
            # --- COLLECT BATCH RESULTS ---
            trial_status[trial_status == 0] = -1
            hit_mask = (trial_status == 1)
            if hit_mask.any():
                aggregate_screen_hits.extend(screen_y[hit_mask].cpu().tolist())
                for y_hit in screen_y[hit_mask].cpu().tolist():
                    aggregate_slit_used.append(1 if abs(y_hit - s1_center) < abs(y_hit - s2_center) else 2)
            aggregate_reflected += (trial_status == -1).sum().item()
            
            for i in range(B):
                aggregate_final_states.append(batch_state[i, 0, :].cpu().numpy())
                aggregate_initial_states.append([initial_y[i], initial_z[i], initial_hue[i]])
                aggregate_outcomes.append(int(trial_status[i].item()))
                aggregate_spins.append(batch_spin[i, 0].item() if args.spin_coupling else 0.0)
            
            if save_first and history:
                hist_arr = np.array(history)
                best_idx = np.where(trial_status[:20].cpu().numpy() == 1)[0]
                best_idx = best_idx[0] if len(best_idx) > 0 else 0
                representative_history = list(hist_arr[:, best_idx, :, :])
            
            trials_done += B
            print(f"  Batch {batch_idx+1}/{n_batches} | Trials: {trials_done}/{N_trials} | Hits: {len(aggregate_screen_hits)} | ZMQ chunks: {zmq_chunk_count}")
            sys.stdout.flush()
        
        # Flush any remaining chunks after all batches complete
        print(f"[ZMQ DEBUG] Post-batch: zmq_socket={'SET' if zmq_socket is not None else 'NONE'}, chunk_buffer_len={len(chunk_buffer)}")
        sys.stdout.flush()
        if zmq_socket is not None and len(chunk_buffer) > 0:
            zmq_socket.send_pyobj({
                "status": "STREAMING",
                "chunk_id": -1,
                "batch_idx": -1,
                "data": np.array(chunk_buffer)
            })
            print(f"[ZMQ] Flushed final {len(chunk_buffer)} remaining chunks")
            sys.stdout.flush()
            chunk_buffer = []
        elif zmq_socket is not None and len(chunk_buffer) == 0:
            print("[ZMQ DEBUG] All chunks already flushed during tick loop (buffer evenly divisible by flush_rate)")
            sys.stdout.flush()
        
    else:
        # =============================================================
        # CPU SEQUENTIAL PATH
        # =============================================================
        N_beam = 1
        for trial in range(N_trials):
            trial_state = torch.zeros((N_single, 10), device=device, dtype=torch.float32)
            trial_state[0, 1] = beam_start_x
            trial_state[0, 2] = np.random.uniform(-beam_y_range, beam_y_range)
            trial_state[0, 3] = np.random.uniform(-beam_z_range, beam_z_range) if beam_z_range > 0 else 0.0
            trial_state[0, 4] = args.beam_momentum
            trial_state[0, 7] = args.mass_a
            trial_state[0, 8] = np.random.uniform(0, 2 * np.pi)
            
            trial_spin = torch.zeros(N_single, device=device, dtype=torch.float32)
            if args.spin_coupling: trial_spin[0] = 0.5 if np.random.random() > 0.5 else -0.5
            
            for j, (wx, wy, wz) in enumerate(wall_positions):
                idx = 1 + j
                trial_state[idx, 1] = wx
                trial_state[idx, 2] = wy
                trial_state[idx, 3] = wz
                trial_state[idx, 7] = m_per_particle
                if args.wall_thermal_phase == 1:
                    trial_state[idx, 8] = np.random.uniform(0, 2 * np.pi)
                else:
                    trial_state[idx, 8] = WALL_HUE
                if args.spin_coupling:
                    trial_spin[idx] = WALL_SPIN

            # Determine whether to save history for visualization
            save_this = (trial == 0)

            # Run standard physics integration loop
            history, screen, _ = run_ticks(trial_state, trial_spin, N_single, TOTAL_TICKS, save_history=save_this)
            
            if save_this:
                representative_history = history
            
            trial_y = trial_state[0, 2].item()
            
            # Classify outcome
            if screen:
                y_hit = list(screen.values())[0]
                aggregate_screen_hits.append(y_hit)
                if abs(trial_y - s1_center) < abs(trial_y - s2_center):
                    aggregate_slit_used.append(1)
                else:
                    aggregate_slit_used.append(2)
            else:
                aggregate_reflected += 1
            
            # Progress update
            if (trial + 1) % 20 == 0 or trial == 0:
                elapsed = time.time() - start_all
                rate = (trial + 1) / elapsed if elapsed > 0 else 0
                eta = (N_trials - trial - 1) / rate if rate > 0 else 0
                print(f"  Trial {trial+1}/{N_trials} | Hits: {len(aggregate_screen_hits)} | Reflected: {aggregate_reflected} | ETA: {eta:.0f}s")
        
        N_beam = N_beam_orig  # Restore
    
    total_elapsed = time.time() - start_all
    print(f"\nTonomura Protocol Complete! {N_trials} trials in {total_elapsed:.1f}s ({total_elapsed/N_trials:.2f}s/trial)")
    if USE_BATCH:
        print(f"  Mode: GPU BATCH (batch_size={BATCH_SIZE}, device={device})")
    else:
        print(f"  Mode: SEQUENTIAL (device={device})")
    
    # Restore N_beam for reporting
    N_beam = N_trials
    
    # Save representative trajectory for SINDy
    if representative_history:
        trajectory_history = representative_history
    else:
        trajectory_history = []
    
    # Build aggregate histogram
    import json
    screen_hits = aggregate_screen_hits
    reflected = aggregate_reflected
    in_flight = N_trials - len(screen_hits) - reflected
    
    if screen_hits:
        y_min = min(screen_hits)
        y_max = max(screen_hits)
        n_bins = 50
        margin = 0.5
        y_min -= margin
        y_max += margin
        bin_width = (y_max - y_min) / n_bins
        bins = [0] * n_bins
        bin_edges = [y_min + i * bin_width for i in range(n_bins + 1)]
        for yp in screen_hits:
            b = min(int((yp - y_min) / bin_width), n_bins - 1)
            bins[b] += 1
    else:
        bins = []
        bin_edges = []
    
    result = {
        "beam_total": N_trials,
        "screen_hits": len(screen_hits),
        "reflected": reflected,
        "in_flight": in_flight,
        "y_positions": screen_hits,
        "slit_used": aggregate_slit_used,
        "histogram_bins": bins,
        "histogram_edges": bin_edges,
        "slit_width": args.slit_width,
        "slit_separation": args.slit_separation,
        "protocol": "tonomura_batch_gpu" if USE_BATCH else "tonomura_sequential",
        "batch_size": BATCH_SIZE,
        "device": str(device),
    }
    with open("double_slit_results.json", "w") as f:
        json.dump(result, f)
    
    print(f"\n--- DOUBLE-SLIT RESULTS (Tonomura Protocol) ---")
    print(f"Total trials:   {N_trials}")
    print(f"Reached screen: {len(screen_hits)} ({100*len(screen_hits)/N_trials:.0f}%)")
    print(f"Reflected:      {reflected} ({100*reflected/N_trials:.0f}%)")
    if screen_hits:
        from_s1 = aggregate_slit_used.count(1)
        from_s2 = aggregate_slit_used.count(2)
        print(f"Through slit 1: {from_s1} | Through slit 2: {from_s2}")
        print(f"Y range:        [{min(screen_hits):.2f}, {max(screen_hits):.2f}]")
    print(f"DOUBLE_SLIT_JSON=double_slit_results.json")
    
    # Save aggregate states for macro-scale SINDy (Test 3)
    if aggregate_final_states:
        agg_final = np.array(aggregate_final_states)    # (N_trials, 10)
        agg_initial = np.array(aggregate_initial_states) # (N_trials, 2)
        agg_outcome = np.array(aggregate_outcomes)       # (N_trials,)
        agg_spins = np.array(aggregate_spins)           # (N_trials,)
        np.savez("aggregate_states.npz",
                 final_states=agg_final,
                 initial_states=agg_initial,
                 outcomes=agg_outcome,
                 spins=agg_spins)
        print(f"Aggregate states saved: {agg_final.shape[0]} trials -> aggregate_states.npz")
    
    # --- BREIT-WHEELER SUMMARY ---
    if BW_ENABLED:
        n_pairs = bw_spawned_count // 2
        print(f"\n{'=' * 60}")
        print(f"BREIT-WHEELER PAIR PRODUCTION SUMMARY")
        print(f"  Threshold:     {BW_THRESHOLD:.3f} MeV")
        print(f"  Pairs spawned: {n_pairs} ({bw_spawned_count} particles)")
        print(f"  Buffer used:   {bw_spawned_count}/{BW_BUFFER_SIZE} slots")
        if n_pairs > 0:
            print(f"  STATUS: TOPOLOGICAL SNAP DETECTED — Matter emerged from vacuum geometry!")
        else:
            print(f"  STATUS: No snaps detected — wave amplitude never reached {BW_THRESHOLD:.3f} MeV")
        print(f"{'=' * 60}")


else:
    # Standard execution for 2-body, 3-body, direct-collapse, holographic
    print("--- RUNNING FORWARD TIME ---")
    try:
        trajectory_history, screen_detected, W_mat = run_ticks(state, TOTAL_TICKS, save_history=True)
        
        if args.t_symmetry:
            print("\n--- INITIATING TIME REVERSAL (T-SYMMETRY TEST) ---")
            print("Inverting momentum vectors...")
            state[:, 4:7] *= -1.0  # Flip momentum (p -> -p)
            print("Running matrix in reverse...")
            run_ticks(state, TOTAL_TICKS, save_history=False)[0]
            
            # T-symmetry divergence measurement
            final_positions = state[:, 1:4].cpu()
            divergence = torch.norm(final_positions - initial_positions, dim=1).sum().item()
            print(f"\nT_SYMMETRY_DIVERGENCE={divergence:.5f}")
            
    except BaseException as e:
        print(f"\n[FATAL ERROR] Physics engine crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        np.savez("crash_dump.npz", last_state=state.cpu().numpy())
        print("[DEBUG] Saved crash_dump.npz to disk.")
        sys.exit(1)
        
    # --- RYU-TAKAYANAGI ENTROPY MEASUREMENT ---
    if args.mode in ["holographic", "holographic-shell", "holographic-ring", "ads-cft"]:
        print("\n--- RYU-TAKAYANAGI ENTROPY MEASUREMENT ---")
        # Ensure we use final positions
        final_pos = state[:, 1:4].cpu()
        com = torch.mean(final_pos, dim=0)
        shifted_pos = final_pos - com
        r_dist = torch.sqrt(torch.sum(shifted_pos**2, dim=1))
        
        max_r = torch.max(r_dist).item()
        if max_r > 0:
            r_steps = np.linspace(0.1, max_r, 20)
            entropy_S = []
            
            # For boolean matrix
            W_cpu = W_mat.cpu()
            
            for R in r_steps:
                inside = r_dist < R
                outside = ~inside
                # Count crossed edges
                # W_cpu[inside] returns rows that are inside.
                # W_cpu[inside][:, outside] returns a sub-matrix of crossed edges.
                crossings = torch.sum(W_cpu[inside][:, outside]).item()
                entropy_S.append(crossings)
                
            print("Radius (R) | Area (R^2) | Volume (R^3) | Entropy S(R)")
            for R, S in zip(r_steps, entropy_S):
                print(f"{R:8.3f} | {R**2:10.3f} | {R**3:12.3f} | {S}")
            
            # Simple linear fit
            R_np = np.array(r_steps)
            S_np = np.array(entropy_S)
            if np.sum(S_np) > 0 and len(S_np) > 1:
                area_corr = np.corrcoef(R_np**2, S_np)[0, 1]
                vol_corr = np.corrcoef(R_np**3, S_np)[0, 1]
                print(f"\nCorrelation with Area (R^2):   {area_corr:.4f} (|{abs(area_corr):.4f}|)")
                print(f"Correlation with Volume (R^3): {vol_corr:.4f} (|{abs(vol_corr):.4f}|)")
                if abs(area_corr) > abs(vol_corr):
                    print(">>> HOLOGRAPHIC PRINCIPLE CONFIRMED (AREA LAW DOMINATES) <<<")
                else:
                    print(">>> VOLUME LAW DOMINATES <<<")
            else:
                print("No entropy detected across boundary.")
        else:
            print("Universe collapsed to a single point.")

    print("\nNext step: Run the script and observe the scattering angle in the generated tensor data!")

# Save 10D tensor map for PySINDy extraction
history_array = np.array(trajectory_history)
if zmq_socket is None:
    np.save(args.output_file, history_array)
    print(f"Trajectory saved to '{args.output_file}'")
else:
    # Flush any remaining chunk_buffer before DONE (only exists in Tonomura batch path)
    try:
        if len(chunk_buffer) > 0:
            zmq_socket.send_pyobj({
                "status": "STREAMING",
                "chunk_id": -1,
                "data": np.array(chunk_buffer)
            })
            print(f"[ZMQ] Flushed final {len(chunk_buffer)} remaining chunks")
            chunk_buffer = []
    except NameError:
        pass  # chunk_buffer doesn't exist in direct-collapse/run_ticks path
    print("Simulation complete. Sending DONE signal over ZMQ (3x for reliability)...")
    zmq_socket.setsockopt(zmq.LINGER, 5000)  # Wait up to 5s for messages to flush on close
    done_msg = {"status": "DONE", "run_label": args.run_label, "dt": DT}
    for i in range(3):
        zmq_socket.send_pyobj(done_msg)
        time.sleep(0.1)
    expected_frames = TOTAL_TICKS / SAVE_INTERVAL
    gb_skipped = (expected_frames * N * 10 * 4) / (1024**3)
    print(f"ZMQ Stream finished. Skipped saving {gb_skipped:.1f}GB to SSD.")

