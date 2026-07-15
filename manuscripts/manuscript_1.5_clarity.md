# Phantom Crossing and Pilot Waves: A Kinematic Bridge Between Cosmological Fluid Dynamics and Micro-Scale Defect Routing in Teleparallel Gravity

**J. Byron Fisher**  
*Affiliation (Independent Researcher)*  
Corresponding author: j.byron.fisher@gmail.com

## Abstract
We present an exploratory comparative analysis bridging the macro-scale cosmological fluid dynamics of Böhmer et al. (2025) with the micro-scale kinematic defect models proposed in Fisher (2026). Recent literature demonstrates that non-minimal coupling of fluid variables to the teleparallel bulk term ($G$) and boundary term ($B$) generically produces a late-time accelerating universe and phantom energy ($w < -1$) via the variational principle. Here, we propose that the non-local pilot-wave guidance equations used to route discrete topological defects through double-slit interference share mathematical parallels with these phantom energy boundary couplings. While Böhmer derives this effect from a rigorous first-principles Lagrangian, our phenomenological toy model approximates the same geometry to achieve computational stability in FDTD particle simulations. This "clarity" paper attempts to map our ad-hoc parameters to the $B$-coupled fluid equations, establishing a potential conceptual framework that links dark energy expansion with quantum potential interference.

---

## 1. Introduction

In recent years, the Teleparallel Equivalent of General Relativity (TEGR) has provided a fertile ground for resolving long-standing problems in both cosmology and quantum kinematics. In TEGR, gravity is mediated not by the curvature of spacetime, but by its torsion. The geometric decomposition of the Ricci scalar $R$ into the torsion scalar (or bulk term) $G$ and a total derivative boundary term $B$, such that $R = -G + B$, is central to this framework. 

Two parallel, yet historically disconnected, avenues of research have leveraged this decomposition:
1. **Cosmological Phantom Energy (Böhmer et al. 2024, 2025; Bahamonde & Wright 2016):** By non-minimally coupling the boundary term $B$ or bulk term $G$ to scalar fields and the thermodynamic variables of a relativistic fluid, researchers have produced robust dynamical systems that exhibit early-time inflation and late-time accelerated expansion. These models naturally cross the "phantom barrier" ($w_{eff} < -1$) without requiring an artificial cosmological constant.
2. **Micro-Scale Resonant Wave Defects (Fisher 2026):** In a separate effort to map 10-dimensional String Theory kinematics onto 4-dimensional flat spacetime, fundamental particles have been modeled as "resonant wave defects." These localized defects possess an internal phase clock ($\theta_{hue}$) and are routed through a background vacuum by a non-local pilot wave, reproducing emergent quantum behaviors such as double-slit interference.

This paper serves as an initial exploration to bridge these two frameworks. We intend to investigate whether the "phantom energy" driving cosmological expansion in Böhmer's fluid-$G$ coupling is mathematically analogous to the non-local "quantum pressure" (or Bohmian potential) driving particle interference in our micro-scale simulation. We clarify where our computational models rely on phenomenological approximations, and how they might eventually align with the rigorous variational principles of the teleparallel literature.

---

## 2. Theoretical Comparison: Macro vs. Micro

### A. The Macroscopic Fluid Coupling (Böhmer's Model)
In the framework of Böhmer et al. (2025), a relativistic fluid is modeled using Brown's variational approach. The action is modified by introducing an algebraic coupling between the fluid's particle number density $n$ and the bulk term $G$ (or boundary term $B$). 

The resulting field equations yield an effective equation of state $w_{eff}$. Böhmer demonstrates that this coupling introduces a "phantom" negative pressure that generically drives the fluid to a late-time accelerating state. Specifically, the boundary-term coupling $f(n, \phi, G)$ induces gradients that alter the standard geodesic flow of the fluid, pulling it outward in an accelerated expansion.

### B. The Microscopic Defect Routing (Fisher's Model)
In our micro-scale simulation, particles are treated as discrete fluid droplets—topological defects in the tetrad field. Each defect possesses an internal phase $\theta_{hue}$. To simulate quantum interference (e.g., the double-slit experiment), we introduce a background scalar grid $\Phi$ (the pilot wave).

The routing of the particle is governed by an effective force proportional to the gradient of this field:
$$ \mathbf{F}_{\text{pilot}} \propto \nabla \Phi $$
In our numerical implementation, this is controlled by phenomenological scaling factors (such as `dbb_strength` and `rae_grad_scale`). This gradient force routes the particle away from classical Newtonian trajectories, guiding it into the interference fringes characteristic of quantum mechanics.

---

## 3. Mathematical Parallels: Phantom Energy and Quantum Potential

To explore a potential bridge between these frameworks, we can draw a phenomenological mapping between the macroscopic Lagrangian and our micro-scale kinematics.

### A. The Macroscopic Variational Source
In Böhmer et al. (2025), the cosmological dark energy is derived from a non-minimal interaction Lagrangian that couples the teleparallel bulk term $G$ (which functions similarly to the boundary term framework) to the fluid's particle number density $n$ and a scalar field $\phi$:
$$ \mathcal{L}_{int} = -\sqrt{-g} f(n, \phi) G $$
Taking the derivative of this non-minimal coupling with respect to local spatial coordinates generates a geometric pressure. In the cosmological fluid equations, this pressure acts as a repulsive force that drives the universe's accelerated expansion.

### B. The Microscopic Kinematic Consequence
At the micro-scale, our teleparallel collider relies on a discrete computational analogue. Using PySINDy, we previously extracted the phenomenological routing equation—the Relativistic Adler Equation—that governs the phase evolution and subsequent spatial routing of a wave defect:
$$ \dot{\theta} = \alpha \left( \frac{m_0}{\gamma} \right) - \kappa \sin(\theta) + \nabla \Phi_{Pauli} $$
Here, the non-local pilot wave gradient $\nabla \Phi_{Pauli}$ serves as the computational steering force that guides particles into interference fringes.

### C. Building the Conceptual Bridge
We can propose a formal mathematical mapping of the state variables:
1. **Fluid Density ($n$)** maps to our discrete FDTD voxel particle count.
2. **Scalar Field ($\phi$)** maps to the defect's internal phase clock ($\theta$).
3. **Geometric Pressure ($\nabla \mathcal{L}_{int}$)** maps to our pilot wave gradient ($\nabla \Phi_{Pauli}$).

In cosmology, "phantom energy" ($w_{eff} < -1$) denotes a state where the energy density of the vacuum *increases* as the universe expands, actively pumping energy into the system. 

This bears a striking resemblance to what happens during microscopic quantum interference. When a particle "sharply turns" to land in an interference fringe (e.g., passing through the double-slit), it diverges from classical Newtonian inertia. To make this sudden turn, it must borrow lateral kinetic energy from the topological background. The vacuum gradient ($\nabla \Phi_{Pauli}$) acts analogously to a phantom energy source, temporarily injecting momentum into the defect to steer it. Thus, the phenomenological quantum potential in our simulation might be viewed as a discrete, micro-scale manifestation of Böhmer's phantom fluid.

---

## 4. Honest Divergence: Variational Principles vs. Phenomenological Couplings

While the mathematical parallels are intriguing, it is vital to explicitly state the methodological divergence between these two bodies of work.

Böhmer et al. derive their fluid couplings using a rigorous, first-principles application of the variational principle (the Brown action). Their field equations are exact and algebraically derived from the Lagrangian.

Conversely, our resonant wave defect framework was constructed purely as an **effective kinematic toy model** designed for high-resolution computational simulation. To maintain stability during discrete-time Symplectic Euler integration (especially to prevent NaN explosions during mass accretion), we rely on phenomenological scaling constants. 

We do not claim to have derived our pilot-wave mechanics directly from the Brown fluid action. Instead, we recognized that to stabilize a discrete fluid-defect simulation in a flat torsional background, we had to invent *ad-hoc* non-local gradient forces that mathematically approximate the exact boundary-term couplings discovered by Böhmer. 

## 5. Addressing Potential Theoretical Roadblocks: Translating Continuous Rigor to Discrete Computation

While the conceptual bridge between our micro-scale toy model and Böhmer's macro-scale fluids is promising, translating continuous variational mathematics into a discrete FDTD particle simulation introduces several fundamental differences. To ensure our computational explorations respect the rigorous standards of teleparallel gravity, we must acknowledge four theoretical roadblocks.

### A. The "Proper Frame" and Covariant Teleparallelism
In the literature, "pure tetrad" teleparallel gravity—which sets the inertial spin connection to zero—is known to be frame-dependent, violating local Lorentz covariance. To maintain full invariance, a non-trivial spin connection is required. Our model, however, relies on a discrete Cartesian FDTD grid. By mapping particles to fixed voxels, our simulation functionally operates in a highly specific, rigid frame of reference. Rather than simulating a globally invariant tensor field, our engine computes dynamics within a strict "proper frame" where the spin connection is set to vanish. The Relativistic Adler Equation remains an effective routing mechanism within this chosen coordinate system, serving as a phenomenological computational proxy rather than a globally covariant tensor equation.

### B. Localized Energy Exchange vs. Gravitational Pseudotensors
Böhmer and Hehl (2018) formally demonstrate that gravitational energy-momentum in teleparallel theories cannot be localized into a true tensor; it requires a coordinate-dependent pseudotensor, such as Freud’s superpotential. Yet, our simulation routes localized kinetic energy from the vacuum gradient ($\nabla \Phi$) to a single particle during "phantom crossing." How is this localized exchange justified in a toy model? Because our FDTD grid locks the simulation into a specific Cartesian coordinate frame, global pseudotensors can be approximated locally within that specific frame. The localized pilot-wave energy transfer in our model acts as a discrete computational evaluation of this pseudotensor in our chosen proper frame.

### C. Discrete Integration Artifacts vs. Continuous Dynamical Systems
The application of dynamical systems to cosmology relies on advanced continuous mathematics to identify non-hyperbolic fixed points and scaling solutions. Our simulation, relying on discrete Symplectic Euler integration and PySINDy regression, must eventually prove that its "phantom attractors" are not merely discrete numerical artifacts. Symplectic integrators are specifically utilized for their ability to conserve phase space volume in Hamiltonian systems. The fact that PySINDy successfully extracts stable, continuous dynamical equations from our discrete temporal data provides initial evidence that our observed interference fringes may represent true dynamical attractors rather than numerical integration errors.

### D. Dimensional Topologies and the Internal Phase Clock
Böhmer's group has explored teleparallel structures in 11 dimensions, where compactified topologies naturally drive vacuum dynamics. Our simulation maps these complex kinematics down to a 4-dimensional flat spacetime grid. To account for the degrees of freedom normally assigned to higher dimensional topologies (e.g., 6 extra non-spatial dimensions + 4 spatial = 10, plus an 11-dimensional tracking matrix), our model utilizes the 4D phase-clock ($\theta_{hue}$) and the internal states of the computation matrix. The internal phase rotation attempts to absorb the higher-dimensional topological degrees of freedom, allowing us to approximate 11D dynamics on a 4D computational grid without explicitly simulating hidden geometric topologies.

## 6. Conclusion

The kinematic routing of topological defects in a teleparallel vacuum simulation offers a unique tool for exploring quantum mechanics. It provides a potential micro-scale visualization of the same geometric boundary-term couplings that drive cosmological dark energy. By recognizing that the non-local pilot wave may act as a discrete computational analogue to a boundary-coupled teleparallel fluid, we hope to establish a conceptual bridge between microscopic quantum potential and macroscopic phantom energy, inviting further rigorous analytical development.

## 7. References

1. **Bahamonde, S., & Wright, M. (2016).** *Teleparallel quintessence with a nonminimal coupling to a boundary term.* Physical Review D, 92(8), 084034. arXiv:1508.06580v4 [gr-qc].
2. **Böhmer, C. G., & Hehl, F. W. (2018).** *Freud’s superpotential in general relativity and in Einstein-Cartan theory.* Physical Review D, 97(4), 044028.
3. **Böhmer, C. G., Fiorini, F., González, P. A., & Vásquez, Y. (2019).** *D = 11 Cosmologies with Teleparallel Structure.*
4. **Böhmer, C. G., & d'Alfonso del Sordo, A. (2024).** *Cosmological fluids with boundary term couplings.* arXiv:2404.05301v2 [gr-qc].
5. **Ashi, H. A., Böhmer, C. G., d’Alfonso del Sordo, A., & Jensko, E. (2025).** *Cosmological dynamical systems of non-minimally coupled fluids and scalar fields.* arXiv:2509.14440v2 [gr-qc].
6. **Fisher, J. B. (2026).** *Teleparallel Kinematics: Resonant Wave Defects in a Torsional Vacuum.* (Manuscript 1).

---

## AI Disclosure
The author acknowledges the use of artificial intelligence tools (specifically Large Language Models and AI coding assistants) during the development of this comparative analysis, the PySINDy data regression, and the drafting/editing of this manuscript. All theoretical concepts, experimental designs, and final conclusions are the sole responsibility of the author.
