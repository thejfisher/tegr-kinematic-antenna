# Manuscript 2 — Exact Edits
## "Resonant Wave Defects at the Event Horizon"

These are the 5 changes you need to make. Each one shows the **exact text to find** and the **exact replacement**.

---

## Edit 1 — Section 2 (Spin-Vorticity Sync)
**Reason:** ω is demonstrated in Manuscript 1 [1] but not in the current ER=EPR simulation code. Add a citation so readers know where to find the demonstration.

**FIND this text:**
> Rather than traversing the spatial grid r_ij, the internal properties of the defects undergo instantaneous non-local synchronization:
>
> θ_hue,j → θ_hue,i
> ω_j → −ω_i

**REPLACE with:**
> Rather than traversing the spatial grid r_ij, the internal properties of the defects undergo instantaneous non-local synchronization:
>
> θ_hue,j → θ_hue,i
> ω_j → −ω_i
>
> The spin-vorticity coupling ω and its role in the defect force budget are developed in [1]; in this work we focus on the phase synchronization channel θ_hue, which is sufficient to demonstrate the firewall mechanism.

---

## Edit 2 — Section 4.2 (Topological Thermal Recoil)
**Reason:** "Comparable" to annihilation is imprecise. 4m₀ = 2.044 MeV is exactly **twice** the electron-positron annihilation energy (1.022 MeV).

**FIND this text:**
> comparable in relative magnitude to matter-antimatter annihilation at this scale

**REPLACE with:**
> twice the energy yield of electron-positron annihilation at this scale

---

## Edit 3 — Section 4.1 (Pair-Production Justification)
**Reason:** The QCD references [9,10] describe ONE pair at a single break point. Your argument for TWO pairs (4m₀) is physically motivated but needs one sentence to distinguish it from local QCD string breaking.

**FIND this text:**
> In quantum chromodynamics, severing a color flux tube requires the vacuum to spontaneously generate new particle-antiparticle pairs to "cap off" both exposed terminations [9,10], preventing the severed ends from bleeding infinite energy into the vacuum. This is the same mechanism underlying the Schwinger pair-production limit [11]. To cap both severed terminations of the wormhole across the event horizon, the vacuum must supply the energy for two complete pairs: 4m₀.

**REPLACE with:**
> In quantum chromodynamics, severing a color flux tube requires the vacuum to spontaneously generate a new particle-antiparticle pair at the break point to cap both exposed terminations [9,10], preventing the severed ends from bleeding infinite energy into the vacuum. This is the same mechanism underlying the Schwinger pair-production limit [11]. Unlike a local QCD flux tube, which breaks at a single spatial point, the ER bridge spans two causally disconnected regions of spacetime. Each severed termination must independently satisfy the vacuum topology, requiring a separate particle-antiparticle pair at each end—yielding a total cost of 4m₀.

---

## Edit 4 — Section 4.3 (Last Paragraph)
**Reason:** The firewall trigger (Δγ > 4.0) and energy deposit (m₀ → 3m₀) are hardcoded rules in the simulation, derived from the math in Section 4.1. The simulation *demonstrates* them, it doesn't independently *derive* them.

**FIND this text:**
> The simulation naturally, and deterministically, derives the AMPS Firewall not as an ad-hoc radiation burst, but as a mandatory topological phase transition whose energy cost is fixed by the topology of the Weitzenböck connection.

**REPLACE with:**
> Given the decoherence threshold derived in Section 4.1 and the energy-conservation constraint of Section 4.2, the simulation demonstrates that the AMPS Firewall emerges not as an ad-hoc radiation burst, but as a mandatory topological phase transition whose energy cost is fixed by the topology of the Weitzenböck connection.

---

## Edit 5 — Section 4.3 (After Simulation Log)
**Reason:** The paper describes a specific simulation configuration (Particle A at x=−5.0, sink at x=−10.0, M=250.0). If someone wants to reproduce these exact results, they need to know how. Add a reproducibility note.

**FIND this text (the paragraph starting after the simulation log):**
> The simulation reveals a profound causal mechanism.

**INSERT before that sentence:**
> The simulation parameters for the above run (particle positions, sink mass, coupling constants) are available in the companion code repository referenced in [1].

---

## Edit 6 — Section 3 (After the Time Dilation paragraph)
**Reason:** Manuscript 1 [1] describes a 0.99c velocity clamp for numerical stability. In this paper, we removed it so gamma can grow without bound (required for the firewall). The reader needs to know this.

**INSERT at the end of Section 3** (after the γ equation and time dilation paragraph):

> **Numerical note:** In [1], the integration employed a pragmatic velocity cap at 0.99c to ensure numerical stability in the 50-particle accretion swarm. For the present study, this cap is removed to allow the relativistic tension factor γ to grow without bound, as required for the firewall trigger mechanism described in Section 4. The causal speed limit v < c is enforced naturally by the relativistic momentum-velocity relation v = p / (γm₀), which guarantees |v| < c for any finite momentum.

---

## Summary Checklist

| # | Section | Change | Status |
|---|---------|--------|--------|
| 1 | §2 | Add citation to [1] for ω demonstration | ☐ |
| 2 | §4.2 | "comparable" → "twice the energy yield" | ☐ |
| 3 | §4.1 | Add sentence distinguishing ER bridge from QCD tube | ☐ |
| 4 | §4.3 | "derives" → "demonstrates" (with Section refs) | ☐ |
| 5 | §4.3 | Add reproducibility note after simulation log | ☐ |
| 6 | §3 | Add numerical note about removing velocity clamp from [1] | ☐ |

> [!TIP]
> None of these edits change the physics or the thesis. They tighten the language to match exactly what the simulation does vs. what the math derives.
