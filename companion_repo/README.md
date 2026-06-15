# Teleparallel Collider Simulation: Companion Code Repository
# Resonant Wave Defects in a Teleparallel Vacuum
# J. Byron Fisher (2025)

## Overview

This repository contains the complete source code for the Teleparallel Collider simulation described in:

1. **Manuscript 1**: "Resonant Vortex-Dislocation Defects in a Teleparallel Vacuum: A Kinematic Isomorphism Between 10D String Models and 4D TEGR"
2. **Manuscript 2**: "Resonant Wave Defects at the Event Horizon: ER=EPR and the AMPS Firewall in a Teleparallel Vacuum"

**DISCLAIMER**: This toy model is not proposing a physical medium of the nothingness of space (i.e. it does not support aether theory). Particles are treated as resonant wave defects in a teleparallel vacuum strictly as a mathematical and topological framework for modeling gravitational and physical interactions.

---

## System Architecture

The simulation consists of three components:

1. **Teleparallel Collider** (`teleparallel_collider.py`): The core PyTorch simulation engine. Generates particle trajectories as `.npy` files.
2. **PySINDy Extractor** (`sindy_extract.py`): Extracts governing differential equations from trajectories using Sparse Identification of Nonlinear Dynamics. Optionally pipes results to a local LLM via Ollama.
3. **Web Dashboard** (`collider-web/`): A React/Vite frontend that provides a visual interface for configuring and running simulations.
4. **FastAPI Backend** (`api.py`): Bridges the web UI to the Python simulation scripts.

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for the web dashboard)
- **PyTorch** (CPU or CUDA/ROCm for GPU acceleration)
- **PySINDy** (`pip install pysindy`)
- **Ollama** (optional, for AI synthesis of equations) — [https://ollama.com](https://ollama.com)

### Python Dependencies
```bash
pip install torch numpy pysindy fastapi uvicorn requests psutil
```

### Web Dashboard Dependencies
```bash
cd collider-web
npm install
```

---

## Running the Simulation

### Option A: Command-Line Only

Run the collider directly:
```bash
# Manuscript 1 — 50-particle accretion swarm (with 0.99c velocity clamp)
python teleparallel_collider.py --mode 3-body-orbit --velocity_clamp 1

# Manuscript 2 — ER=EPR Firewall test (free momentum growth)
python teleparallel_collider.py --mode gravity-sink --mass_a 0.511 --mass_b 0.511 --entangled 1 --velocity_clamp 0 --sink_mass 50000

# 2-body head-on collision at 0.99c
python teleparallel_collider.py --mode 2-body-collision --mass_a 16.125 --mass_b 16.125

# Extract governing equations with PySINDy
python sindy_extract.py --file electron_trajectory.npy --context "electron Møller scattering"
```

### Option B: Full Web Dashboard

Open **three separate terminal tabs**:

**Terminal 1 — AI Bridge** (optional, only if using a remote Ollama server):
```bash
ssh -N -L 11434:localhost:11434 <YOUR_USERNAME>@<YOUR_OLLAMA_SERVER_IP>
```

**Terminal 2 — Physics API Backend**:
```bash
cd <path-to-this-repo>
python api.py
```

**Terminal 3 — React Dashboard**:
```bash
cd <path-to-this-repo>/collider-web
npm run dev
```

Then open your browser to `http://localhost:5173`.

---

## Key Simulation Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Mode | `--mode` | `2-body-collision` | Simulation type: `2-body-collision`, `gravity-sink`, `3-body-scatter`, `3-body-orbit` |
| Particle Mass A | `--mass_a` | 1.0 | Rest mass of Particle A (in electron mass units, 0.511 MeV) |
| Particle Mass B | `--mass_b` | 1.0 | Rest mass of Particle B |
| Pauli Coupling | `--pauli` | 500.0 | Exchange pressure coupling constant (χ) |
| Kinetic Damping | `--vacuum` | 0.001 | Kinetic damping coefficient (Λ) |
| Torsion | `--torsion` | 1.0 | Torsion tensor coupling strength |
| Entangled | `--entangled` | 0 | Enable ER=EPR wormhole bond (W_ij) |
| Velocity Clamp | `--velocity_clamp` | -1 | -1=auto, 0=off (Paper 2), 1=on (Paper 1) |
| Sink Mass | `--sink_mass` | 50000.0 | Gravity sink mass (gravity-sink mode only) |
| T-Symmetry Test | `--t_symmetry` | false | Run time-reversal divergence test |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `teleparallel_collider.py` | Core simulation engine (PyTorch, Symplectic Euler) |
| `sindy_extract.py` | PySINDy equation extraction + Ollama LLM synthesis |
| `api.py` | FastAPI backend bridging web UI to simulation |
| `collider-web/src/App.jsx` | React dashboard UI |
| `collider-web/src/App.css` | Dashboard styling |
| `collider-web/src/index.css` | Global CSS design system |

---

## License

This code is provided as companion material to the referenced manuscripts for academic review and reproducibility purposes.
