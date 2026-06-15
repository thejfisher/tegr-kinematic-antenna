# TEGR Particle Collider: Laboratory Notes

## Calibration Target 1: The Electron (Møller Scattering Baseline)

**Objective:** Map the deterministic Weitzenböck lattice and Resonant Vacuum Condensate (TEGR) framework to replicate known real-world electron-electron collision scattering probabilities (Møller scattering).

### 1. Dimensionless Translation Mapping
To bridge $mc^2 = hf$ into the 10D tensor `[t, x, y, z, px, py, pz, m0, hue, gamma]`:

* **Mass ($m_0$):** Set to a dimensionless baseline of `1.0`. All other particles will be scaled relative to this (e.g., Proton $m_0 \approx 1836.0$).
* **Hue Phase Clock ($\theta_{hue}$):** Mapped to the Compton frequency of the electron. At rest, an electron's frequency is extremely high, meaning the `hue` value will oscillate rapidly, creating a dense "waveform" probability shell around the particle.
* **Spin-Vorticity ($\omega$):** Fermionic spin (1/2) is mapped to the internal torsional twist.

### 2. Experimental Setup (High-Fidelity Single Run)
* **Initial State:** Two electrons placed on the X-axis at `x = -10.0` and `x = 10.0`.
* **Impact Parameter (y-offset):** `0.01` to prevent a perfectly mathematically singular head-on collision, forcing the torsional shear to dictate the scattering angle.
* **Momentum:** High relativistic bounds ($p_x = 50.0$ and $-50.0$), forcing $\gamma$ to spike as they approach impact.

### 3. Tuning Dials (To be calibrated against real-world data)
* **$F_{pauli}$ (Exclusion Spring):** Scalar multiplier dictating how hard identical fermions repel when their wave-phases overlap.
* **$\Lambda$ (Vacuum Viscosity/Drag):** Controls how much kinetic energy bleeds into the vacuum during extreme acceleration.
* **$T_{\mu\nu}$ (Torsional Shear):** The geometric twisting of the Weitzenböck lattice that replaces standard $1/r^2$ electromagnetism/gravity.

### 4. Observations
*(To be filled after initial runs)*
