# Pilot-Wave (de Broglie–Bohm) Preset

## Overview

The **Pilot-Wave** preset implements a de Broglie–Bohm (dBB) guidance
mechanism on top of the existing TEGR torsion-matrix engine.  Instead of
treating the Eulerian `torsion_grid` as a secondary effect, this mode
promotes it to the **primary dynamical object**: particles become passive
*defects* whose velocities are determined entirely by the spatial gradient
of the propagating wave field.

## Physics Mapping

| Standard dBB | TEGR Matrix Engine |
|---|---|
| Wavefunction ψ(x,t) evolving via Schrödinger eq. | 3-D `torsion_grid` (φ) evolving via FDTD conv3d wave solver |
| Guidance equation: v = (ℏ/m) ∇S / \|ψ\| | v += (1/m₀) · ∇φ · Δt  (no division by amplitude) |
| Born rule \|ψ\|² gives probability density | Emergent from ensemble of N independent trials |
| Quantum potential Q = −(ℏ²/2m) ∇²\|ψ\| / \|ψ\| | Implicit in the FDTD propagation + Pauli interaction |

### Key Simplification

Standard dBB requires dividing by the wavefunction amplitude |ψ|, which
creates **numerical singularities** at nodes.  Our implementation avoids
this entirely: the gradient of the real-valued torsion field is used
directly as a guiding force, with no amplitude denominator.

## Usage

### Command Line

```bash
python teleparallel_collider.py \
    --mode pilot-wave \
    --pilot_wave 1 \
    --num_particles 10000 \
    --mass_a 0.511 \
    --slit_width 0.5 \
    --slit_separation 9.6 \
    --beam_momentum 5000.0 \
    --wall_depth 1 \
    --dt 0.001 \
    --total_ticks 5000
```

### GUI Preset

Select **"30. Pilot Wave (de Broglie-Bohm)"** from the preset dropdown.

### Tri-Node Pipeline

```bash
python teleparallel_collider.py \
    --mode pilot-wave \
    --pilot_wave 1 \
    --num_particles 10000 \
    --mass_a 0.511 \
    --slit_width 0.5 \
    --slit_separation 9.6 \
    --beam_momentum 5000.0 \
    --wall_depth 1 \
    --dt 0.001 \
    --total_ticks 5000 \
    --zmq_target 100.66.100.83:7777
```

## Recommended Hyper-Parameters

| Parameter | Default | Description |
|---|---|---|
| `--pilot_wave` | 0 | Enable dBB guidance (1=on, 0=off) |
| `--interpolation_order` | linear | Gradient interpolation: `linear` or `cubic` |
| `--num_particles` | 10000 | Number of independent electron trials |
| `--mass_a` | 0.511 | Electron rest mass (MeV) |
| `--slit_width` | 0.5 | Width of each slit opening (sim units) |
| `--slit_separation` | 9.6 | Center-to-center slit distance (sim units) |
| `--beam_momentum` | 5000.0 | Forward momentum of each electron |
| `--wall_depth` | 1 | Thickness of the barrier wall (layers) |
| `--dt` | 0.001 | Time step |
| `--total_ticks` | 5000 | Simulation duration |

## How It Works

1. **Wave Initialisation**: The FDTD `torsion_grid` (φ) starts as a
   zero field.  Each particle continuously *emits* into the grid at its
   current position (proportional to its relativistic mass γ·m₀).

2. **Wave Propagation**: The standard 7-point Laplacian conv3d solver
   advances φ at each tick, with torsional wake decay and obstacle
   masking (copper walls).

3. **Gradient Interpolation**: Using `trilinear_interpolate_gradient()`
   from `utils.py`, the engine samples ∇φ at each particle's exact
   continuous position via `torch.nn.functional.grid_sample`.

4. **Guidance Update**: The interpolated gradient is converted to a
   force via `F = (1/m₀) · ∇φ · coupling`, which updates the particle's
   momentum.  The relativistic velocity `v = p/(γ·m₀)` naturally
   enforces |v| < c.

5. **Boundary Handling**: Particles outside the grid extent are clamped
   at the grid boundary (no wrapping or reflection).

## Differences from Standard Presets

- **No Pauli interaction between particles**: Each trial is independent.
  The wave field is the sole guiding mechanism.
- **Vectorised gradient**: Uses GPU-accelerated `grid_sample` instead of
  per-particle integer index lookups.
- **No amplitude division**: Avoids the standard dBB singularity at
  wavefunction nodes.

## Expected Output

With the recommended parameters, the screen-hit distribution should show
interference-like fringes in the Y-axis histogram, driven by the wave
field propagating through both slits simultaneously and guiding
particles toward constructive interference regions.
