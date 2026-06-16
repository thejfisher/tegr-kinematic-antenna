# TEGR Kinematic Antenna

This repository contains the complete Python implementation of the Teleparallel Equivalent of General Relativity (TEGR) continuous engine, including the Kinematic Antenna array simulation, the Pulsar Timing Array (PTA) stochastic noise injector, and the co-rotating PySINDy extraction pipeline.

For those unable to get their hands on a local machine that can run this code, I included an excel spreadsheet to help check the math and logic TEGR_10x10_Matrix_WORKING.xlsx

## Overview
This engine computationally validates the LISA mission's ability to extract the Milky Way's stochastic background Doppler shift (the "lopsided" galactic spin) by isolating the topological phase dimension from spatial thermodynamic noise.

## Contents
- **`teleparallel_collider.py`**: The core TEGR geometric physics engine.
- **`teleparallel_gui.py`**: The graphical user interface to launch simulation presets.
- **`sindy_zmq_server.py`**: The buffer node for the PySINDy extraction pipeline, utilizing sparse regression to identify the non-linear dynamics.
- **`pta_residuals.csv`** (and other CSVs): Raw signal injections, including the stochastic background residuals for the "Stochastic Hum" injection.

## How to Run

1. Ensure you have the required dependencies installed (e.g., `torch`, `numpy`, `PyQt5`, `pysindy`, `pyzmq`).
2. Start the PySINDy Buffer Node in one terminal:
   ```bash
   python sindy_zmq_server.py --port 7777 --galactic_spin 0.5
   ```
3. Launch the GUI in another terminal:
   ```bash
   python teleparallel_gui.py
   ```
4. To reproduce the exact results from the paper, select **17. The Mark Thompson Experiment** (Phase 1) or **18. Stochastic Hum (Mark Thompson Phase 2)** from the Preset dropdown and hit **Run Simulation**.

## References
- Fisher, J. B. (2026). *Resonant Wave Defects in a Teleparallel Vacuum: A Kinematic Isomorphism of String Theory in Flat Sub-spacetime*. Zenodo. DOI: [10.5281/zenodo.20401620](https://doi.org/10.5281/zenodo.20401620)
- Fisher, J. B. (2026). *Phase Synchronization and Matrix Mechanics in a Discrete TEGR Lattice*. Zenodo. DOI: [10.5281/zenodo.20401719](https://doi.org/10.5281/zenodo.20401719)
