# Classical Edge Scattering in a Teleparallel Double-Slit: Falsification Tests and the Limits of Kinematic Emergence

**J. Byron Fisher**  
_Affiliation (Independent Researcher)_  
Corresponding author: j.byron.fisher@gmail.com

**Abstract**  
We implement the Tonomura single-particle double-slit protocol within the GPU-accelerated Teleparallel Equivalent of General Relativity (TEGR) collider established in Papers 1–3 [1,2,3]. Firing 10,000 sequential beam particles through a double-slit wall composed of massive Pauli-repulsive lattice defects, we observe a striking bimodal banding pattern: particles cluster at the inner and outer edges of each slit aperture, separated by a dead zone at slit center with ~5% occupancy. This pattern superficially resembles quantum interference fringes. However, a battery of three falsification tests—single-slit control, slit separation variation, and aggregate state analysis—conclusively demonstrates that the observed banding is **classical edge scattering** from the $1/r^3$ Pauli pressure gradient at slit boundaries, not wave interference. The single-slit control produces an identical per-slit pattern. Doubling slit separation shifts the bands to new edge locations with no fringe compression. Aggregate correlation analysis reveals a near-perfect ($r = 0.999$) geometric correspondence between initial beam position and final landing position, with negligible phase dependence ($r = 0.068$). We report these negative results in full, establishing the precise boundary between what the TEGR lattice model can and cannot reproduce, and identifying the specific physical mechanisms absent from the current framework that would be required for genuine wave interference.

## 1. Introduction

### 1.1 Motivation

The double-slit experiment occupies a singular position in physics: it is the simplest demonstration that matter exhibits wave-like behavior at the quantum scale [4]. When single particles are fired sequentially through a double-slit barrier, an interference pattern gradually accumulates on the detection screen—despite each particle traversing the apparatus alone [5]. This phenomenon is often cited as the definitive evidence that quantum mechanics cannot be reduced to classical particle dynamics.

In Papers 1–3 [1,2,3], we developed a computational framework in which particles are modeled as resonant wave defects propagating through a flat Weitzenböck lattice, subject to phase-coupled Pauli exchange pressure ($1/r^3$), torsion-mediated coupling, and relativistic integration. The framework reproduced several phenomena suggestive of quantum-like behavior: entanglement synchronization, firewall events, mass-dependent dynamical complexity, and emergent Kaluza-Klein dimensional signatures.

The natural next question is whether this framework can reproduce the canonical signature of quantum mechanics: double-slit interference. This paper reports our attempt, our initial (incorrect) interpretation, the falsification tests that corrected it, and the precise physical conclusion.

### 1.2 The Tonomura Protocol

Following Tonomura et al. [5], we implement a single-particle-at-a-time protocol. Each trial fires one beam particle (mass $m_A = 100$ MeV, forward momentum $p_x = 1000$) toward a wall of 51 massive ($m_{\text{wall}} = 1000$ MeV) Pauli-repulsive lattice defects positioned at $x = 0$. Two slit openings of width $w = 2.0$ are centered at $y = \pm 3.0$ (separation $d = 6.0$). A detection screen at $x = 50$ records the $y$-position of any particle that passes through. Particles that fail to penetrate a slit are reflected by Pauli pressure and classified as reflected.

Each beam particle is assigned a random initial $y$-position drawn uniformly from $[-6.0, 6.0]$ and a random hue phase angle $\theta_{\text{hue}} \in [0, 2\pi]$. No two particles interact; each trial is independent.

### 1.3 GPU Batch Acceleration

To achieve the statistical volume required for definitive pattern analysis, we implemented a GPU-parallel batch engine using PyTorch CUDA on an NVIDIA RTX 5070 (12 GB VRAM). The engine stacks $B = 100$ independent trials into a single $(B, 52, 10)$ state tensor and vectorizes all pairwise force calculations—Pauli exchange, torsion, vacuum damping, and relativistic integration—across the batch dimension. The physics is identical to the sequential CPU implementation; only the loop structure differs.

This reduced the per-trial computation time from 2.3 s (CPU sequential) to 0.11 s (GPU batch), a **21× speedup**, enabling 10,000 trials in 18.6 minutes.

## 2. Results: The 10,000-Trial Dataset

### 2.1 Aggregate Statistics

| Parameter | Value |
|---|---|
| Total trials | 10,000 |
| Screen hits | 2,160 (21.6%) |
| Reflected | 7,840 (78.4%) |
| Through slit 1 ($y > 0$) | 1,070 (49.5% of hits) |
| Through slit 2 ($y < 0$) | 1,090 (50.5% of hits) |
| Slit balance ratio | 0.982 |
| $y$-range of hits | $[-3.97, 3.96]$ |

The 21.6% throughput is significantly below the geometric expectation of 33.3% (slit area / beam width = 4.0 / 12.0), indicating that Pauli pressure extends into the slit openings and reflects an additional ~12% of geometrically eligible particles.

### 2.2 Throughput Stability

The throughput remained constant across five independent runs spanning two orders of magnitude in sample size:

| Run | $N$ | Device | Hits | Throughput |
|---|---|---|---|---|
| 200 | 200 | CPU | 42 | 21.0% |
| 500 | 500 | CPU | 108 | 21.6% |
| 1,000 | 1,000 | CPU | 209 | 20.9% |
| 5,000 | 5,000 | CPU | 1,062 | 21.2% |
| 10,000 | 10,000 | GPU | 2,160 | 21.6% |

Within-run stability (per-1000-trial segments of the 10K dataset) fluctuates between 20.3% and 23.6%, consistent with binomial sampling noise around a true rate of $\mu = 21.5\% \pm 1.0\%$.

### 2.3 The Bimodal Edge-Banding Pattern

The distribution of screen hits is not uniform across the slit aperture. For slit 1 ($y \in [2.0, 4.0]$), center at $y = 3.0$:

| Band | $y$-range | Hits | Fraction |
|---|---|---|---|
| Inner edge | $[2.0, 2.6)$ | 525 | 49.1% |
| Dead zone | $[2.6, 3.4]$ | 46 | 4.3% |
| Outer edge | $(3.4, 4.0]$ | 499 | 46.6% |

Slit 2 ($y \in [-4.0, -2.0]$) exhibits a mirror-symmetric distribution:

| Band | $y$-range | Hits | Fraction |
|---|---|---|---|
| Inner edge | $(-2.6, -2.0]$ | 492 | 45.1% |
| Dead zone | $[-3.4, -2.6]$ | 58 | 5.3% |
| Outer edge | $[-4.0, -3.4)$ | 540 | 49.5% |

Each slit produces two concentration bands at its edges, with a depleted central region containing only ~5% of arrivals. The overall pattern consists of **four discrete bands**—two per slit—with no structure in the inter-slit region.

## 3. A Cautionary Note on AI-Assisted Interpretation

### 3.1 The SINDy Misattribution

The PySINDy sparse regression pipeline [6], applied to the 10,000-trial dataset, extracted the following governing equations:

$$
\dot{v}_x = -0.001\,m_0^2, \quad \dot{v}_y = 0.000, \quad \dot{\theta}_{\text{hue}} = 0.010\,m_0^2
$$

An AI analysis model interpreted the absence of spatial variables ($x$, $y$, $r$) from these equations as evidence of "macro-scale emergence"—claiming that the spatial steering terms visible in smaller runs had statistically washed out across 10,000 trials, leaving only the mass-invariant core dynamics.

### 3.2 The Data Pipeline Error

This interpretation was incorrect. Examination of the data architecture revealed that PySINDy was fitted on a $(10000, 52, 10)$ trajectory tensor representing **10,000 integration ticks of a single trial**, not the aggregate of 10,000 trials. The first axis is the temporal index within one particle's flight, not the trial index across the ensemble. SINDy never saw the macroscopic aggregate data.

The simplified equations reflect the dynamics of one beam particle traversing 50 units of free space after passing through the slit—a mostly straight, forward trajectory with gradual vacuum damping. The absence of spatial complexity is a consequence of the particle's simple post-slit kinematics, not statistical cancellation across an ensemble.

> **Note:** This misattribution is reported in full because it illustrates a systematic risk in AI-assisted scientific analysis: language models can construct plausible-sounding physical narratives from data they have not correctly parsed. Independent verification of the data pipeline is essential before accepting AI-generated interpretations of simulation output.

## 4. Falsification Tests

The bimodal edge-banding pattern superficially resembles a two-fringe interference pattern. To determine whether the effect is wave interference or classical boundary scattering, we designed three falsification tests following standard experimental methodology.

### 4.1 Test 1: Single-Slit Control

**Hypothesis:** If the banding is caused by wave interference between paths through slits 1 and 2, then closing one slit should eliminate the interference pattern and produce a qualitatively different distribution (e.g., a single central peak, as in single-slit Fraunhofer diffraction).

**Method:** Slit 2 was sealed with wall particles while slit 1 geometry remained identical. 1,000 trials were run with all other parameters unchanged.

**Result:**

| Band | Double-slit (slit 1) | Single-slit (slit 1) |
|---|---|---|
| Inner edge | 49.1% | 49.5% |
| Dead zone | 4.3% | 2.8% |
| Outer edge | 46.6% | 47.7% |

The distributions are statistically indistinguishable. Slit 2's presence or absence has no measurable effect on slit 1's banding pattern. The per-slit distribution is entirely self-contained.

**Additional finding:** 51 particles (32% of all hits) in the single-slit configuration landed on the negative-$y$ side of the screen, despite slit 2 being sealed. These are beam particles that entered slit 1 at its inner edge ($y \approx 2.0$) and were deflected at steep angles by the asymmetric Pauli gradient, analogous to high-angle Rutherford scattering at a potential boundary.

**Throughput comparison:** Single-slit efficiency (actual/geometric) was 96%, compared to 65% in double-slit. The second slit's Pauli field creates cross-talk pressure that reduces per-slit throughput by approximately 31%.

**Conclusion:** The banding is **localized edge scattering**, not inter-slit interference.

### 4.2 Test 2: Slit Separation Variation

**Hypothesis:** In wave interference, the fringe spacing scales as $\Delta y = \lambda L / d$, where $d$ is the slit separation. Doubling $d$ should halve the fringe spacing. In edge scattering, the band pattern should simply translate to the new slit locations with identical internal structure.

**Method:** Slit separation was doubled from $d = 6.0$ to $d = 12.0$, placing slits at $y = [5.0, 7.0]$ and $y = [-7.0, -5.0]$. 1,000 trials were run.

**Result:**

| Metric | $d = 6.0$ | $d = 12.0$ |
|---|---|---|
| Slit 1 center | $y = 3.0$ | $y = 6.0$ |
| Inner band fraction | 49.1% | 50.7% |
| Dead zone fraction | 4.3% | 1.4% |
| Outer band fraction | 46.6% | 47.9% |
| Efficiency (actual/geo) | 65% | 67% |

The band fractions are preserved to within statistical noise. The bands shifted from $y \approx [2.1, 2.5]$ and $[3.5, 3.9]$ to $y \approx [5.1, 5.5]$ and $[6.5, 6.9]$—tracking the slit edges exactly. No fringe compression, no new oscillatory structure, and no inter-slit coupling was observed.

**Conclusion:** Band positions are determined by slit edge geometry, not wavelength-dependent interference.

### 4.3 Test 3: Aggregate State Analysis

**Hypothesis:** If the hue phase angle $\theta_{\text{hue}}$ acts as a wave-like internal degree of freedom, it should correlate with the final landing position, potentially steering particles toward specific bands based on their phase.

**Method:** The full 10-element state vector was captured for every beam particle at both trial initialization and trial completion across 1,000 GPU-batched trials. Pearson correlation coefficients were computed between initial state variables and final screen $y$-position for all 222 screen hits.

**Result:**

| Correlation | Pearson $r$ | Interpretation |
|---|---|---|
| Initial $y$ → Final $y$ | **0.9989** | Near-perfect geometric correspondence |
| Initial $\theta_{\text{hue}}$ → Final $y$ | **0.0683** | No meaningful phase steering |

The near-unity $y$-to-$y$ correlation establishes that particles land almost exactly where they entered the slit. The negligible hue correlation establishes that the internal phase angle does not steer particles laterally.

**Hit probability by initial $y$-position:**

| Initial $y$ band | Hit rate |
|---|---|
| $y_0 \in [-6, -4)$ | 0% (wall) |
| $y_0 \in [-4, -2)$ | 68% (slit 2) |
| $y_0 \in [-2, +2)$ | 0% (inter-slit wall) |
| $y_0 \in [+2, +4)$ | 67% (slit 1) |
| $y_0 \in [+4, +6)$ | 0% (wall) |

Only particles whose initial trajectory falls within a slit's $y$-range reach the screen. Zero particles from outside the slit bands ever penetrate the wall.

**Final momentum analysis:** Screen-arriving particles have $p_y = (1 \pm 12) \times 10^{-6}$ and $\gamma = 1.000000$. By the time particles reach the screen at $x = 50$, vacuum damping has dissipated all lateral momentum acquired during slit transit.

**Conclusion:** The landing pattern is determined by **geometry alone**. The engine operates as a classical ballistic aperture filter.

## 5. Physical Mechanism: The Pauli Pressure Nozzle

The three falsification tests establish a consistent physical picture. The bimodal edge-banding is not interference; it is the mechanical consequence of the $1/r^3$ Pauli pressure gradient at slit boundaries acting as a "pressure nozzle."

### 5.1 The Slit Transit Mechanism

A beam particle approaching slit 1 ($y \in [2.0, 4.0]$) encounters two Pauli pressure walls:

- **Inner edge** ($y = 2.0$): Wall particles below the slit at $y = 1.5, 1.0, 0.5, \ldots$ create an upward pressure gradient
- **Outer edge** ($y = 4.0$): Wall particles above the slit at $y = 4.5, 5.0, 5.5, \ldots$ create a downward pressure gradient

A particle aimed at the slit center ($y \approx 3.0$) experiences **symmetric opposing pressure** from both edges simultaneously. If this net pressure exceeds the particle's forward momentum, it is reflected. The ~5% dead zone occupancy indicates that slit-center trajectories experience the highest net repulsion.

Particles aimed at the edges ($y \approx 2.2$ or $y \approx 3.8$) experience **asymmetric pressure**: strong repulsion from the near edge, weak repulsion from the far edge. The net lateral force pushes them slightly toward the nearest edge but does not reflect them, because the one-sided pressure is insufficient to reverse the forward momentum. These edge-skimming trajectories constitute the two concentration bands.

### 5.2 The Dead Zone as a Pressure Equilibrium

The dead zone at slit center is not a destructive interference node. It is the region where the Pauli pressure from both slit walls achieves approximate balance, creating a maximum in the net repulsive force along the beam axis ($-x$ direction). Particles traversing this region are the most likely to be reflected.

### 5.3 Throughput Cross-Talk

In the double-slit configuration, the inter-slit wall section ($y \in [-2, +2]$) projects Pauli repulsion into both slit apertures, reducing per-slit throughput to 65% of geometric expectation. In the single-slit configuration, this cross-talk is partially mitigated (96% efficiency), confirming that the overlapping pressure fields from adjacent wall sections influence the effective aperture.

## 6. What Is Missing: The Gap to Quantum Interference

The TEGR lattice model, as currently implemented, cannot reproduce quantum double-slit interference. This section identifies precisely what is absent.

### 6.1 No Path Integral Summation

In quantum mechanics, a particle's probability amplitude at a detection point is the sum of amplitudes over all possible paths connecting source to detector [7]. In the double-slit geometry, this means contributions from both slits combine coherently—even for a single particle. The TEGR collider computes a **single deterministic trajectory** for each particle. There is no mechanism for a particle traversing slit 1 to "know about" slit 2.

### 6.2 No de Broglie Wavelength

Quantum interference fringe spacing depends on the de Broglie wavelength $\lambda = h/p$. The TEGR model assigns no wavelength to its particles. The hue phase angle $\theta_{\text{hue}}$ evolves as $\dot{\theta} = m_0 / \gamma$, but this does not couple to spatial periodicity. Test 3 confirmed that $\theta_{\text{hue}}$ has negligible correlation ($r = 0.068$) with landing position.

### 6.3 No Superposition

The model tracks definite particle states—position, momentum, phase—at every integration step. There is no state vector in a Hilbert space, no wavefunction collapse, and no Born rule sampling. Each trial produces a single classical outcome.

### 6.4 What the Model Does Reproduce

Despite the absence of wave mechanics, the TEGR framework successfully reproduces:

- **Throughput filtering**: Pauli pressure creates effective aperture sizes smaller than the geometric slit width.
- **Edge-dependent scattering**: Particles interacting with slit boundaries acquire characteristic deflection patterns based on their proximity to wall particles.
- **Deterministic phase dynamics**: The hue phase evolves smoothly and deterministically, even though it does not influence spatial trajectories in the double-slit geometry.
- **Scalable GPU parallelism**: The matrix structure of the force calculations enables efficient batch processing without altering the physics.

These are properties of a **classical N-body system with short-range repulsive interactions**, not quantum wave mechanics.

## 7. Conclusion

We constructed a double-slit experiment within the TEGR collider and observed bimodal edge-banding that initially appeared consistent with quantum interference. Three falsification tests—single-slit control, slit separation variation, and aggregate state analysis—conclusively demonstrated that the effect is classical edge scattering from the $1/r^3$ Pauli pressure gradient at slit boundaries.

The model operates as a deterministic ballistic aperture filter. Landing position is determined by initial beam geometry ($r = 0.999$), not internal phase ($r = 0.068$). The band pattern tracks slit edge locations, does not depend on the presence of the second slit, and does not exhibit wavelength-dependent fringe spacing.

These negative results are reported without qualification. The TEGR lattice model, in its current form, does not reproduce quantum double-slit interference. The specific physical mechanisms absent from the framework—path integral summation, de Broglie wavelength coupling, and quantum superposition—are identified as necessary ingredients for any future extension aimed at recovering wave-mechanical behavior.

The edge-scattering phenomenon itself, however, remains a well-characterized and reproducible feature of the Pauli exchange force in confined geometries, and may find independent utility as a model of classical pressure-mediated aperture dynamics.

---

## Acknowledgments

The author acknowledges the assistance of AI language models in data analysis and drafting. A critical finding of this work—the correction of an AI-generated misinterpretation of SINDy results (Section 3)—underscores the necessity of independent verification when using AI tools in scientific research. Thanks are extended to Joseph O'Brien for his continued intellectual support.

---

## References

[1] J. B. Fisher, "Resonant Wave Defects in a Teleparallel Vacuum: A Kinematic Isomorphism of String Theory in Flat Spacetime," *Zenodo* (2025).

[2] J. B. Fisher, "Resonant Wave Defects at the Event Horizon: ER=EPR and the AMPS Firewall in a Teleparallel Vacuum," *Zenodo* (2025).

[3] J. B. Fisher, "Emergent Dimensional Signatures in a Teleparallel Collider: Kaluza-Klein Scaling and Mass-Dependent Dynamics Under Extreme Strain," *Zenodo* (2025).

[4] R. P. Feynman, R. B. Leighton, and M. Sands, *The Feynman Lectures on Physics*, Vol. III (Addison-Wesley, Reading, MA, 1965).

[5] A. Tonomura, J. Endo, T. Matsuda, T. Kawasaki, and H. Ezawa, "Demonstration of single-electron buildup of an interference pattern," *Am. J. Phys.* **57**(2), 117–120 (1989).

[6] B. M. de Silva, K. Champion, M. Quade, J.-C. Loiseau, J. N. Kutz, and S. L. Brunton, "PySINDy: A Python package for the Sparse Identification of Nonlinear Dynamical Systems from data," *J. Open Source Softw.* **5**(49), 2104 (2020).

[7] R. P. Feynman and A. R. Hibbs, *Quantum Mechanics and Path Integrals* (McGraw-Hill, New York, 1965).
