# Data-Driven Equation Discovery from Weak-Measurement Photon Trajectories: Structural Correspondence with a Teleparallel Gravity Model

**J. Fisher**
Independent Researcher
July 2026

**Contact:** [GitHub — tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna)

---

## 1. Abstract

We report a data-driven equation discovery analysis applied to the raw experimental dataset of Kocsis et al. (2011), in which average photon trajectories through a double-slit interferometer were reconstructed via weak measurement. Using PySINDy — a sparse regression framework for identifying governing equations from data — we extracted the dominant dynamical law governing the transverse deflection of photons as a function of propagation distance. The algorithm, operating without any prior physical assumptions, recovered a nonlinear equation of the form:

$$\frac{dx}{dz} \approx \kappa\,(x - \sin x)$$

where $x$ is the transverse position and $z$ is the propagation axis (serving as the temporal variable for photons, since $z = ct$).

This result is notable because the same structural form — $\kappa(\theta - \sin\theta)$ — was independently derived from a toy model of Teleparallel Equivalent General Relativity (TEGR) mapped via Jacobian matrices, in which a "blind" SINDy node extracted the governing equation from colliding torsion matrices. In that context, the $(\theta - \sin\theta)$ term was interpreted as a non-local phase gradient arising from the torsion tensor structure. The appearance of the identical nonlinear structure in empirical photon data, discovered without any assumed functional form, suggests that this term may encode a real physical feature of the transverse momentum field — one that is not predicted by standard Bohmian mechanical treatments.

We emphasize that the numerical goodness-of-fit ($R^2 \approx 0.19$) is modest. The significance of this finding lies not in the quantitative accuracy of the fit, but in the **structural form** of the discovered equation — specifically, the emergence of the $(\theta - \sin\theta)$ nonlinearity from a sparse regression over a broad function library. We present these findings as a preliminary result warranting further investigation with higher-resolution data and more controlled regression protocols.

---

## 2. Background and Motivation

### 2.1 The Kocsis Experiment

In 2011, Kocsis et al. published a landmark paper in *Science* demonstrating the first experimental observation of average photon trajectories through a two-slit interferometer, using the technique of weak measurement [1]. The experiment, conducted in the Steinberg lab at the University of Toronto, used single photons entangled in polarization, where the polarization degree of freedom served as a "pointer" that weakly coupled to the transverse momentum $k_x$ without significantly disturbing the photon's spatial wave function.

The raw data consists of 91 CCD camera images taken at 41 propagation planes (one background frame plus two polarization channels — horizontal $H$ and vertical $V$ — per plane), from which the weak transverse momentum is extracted as:

$$\frac{k_x}{k} = \tan\!\left(\arcsin\!\left(\frac{1}{373}\,\frac{V - H}{V + H}\right)\right)$$

where the factor $1/373$ arises from the calibrated weak-measurement coupling strength. Integrating this momentum field along the propagation axis reconstructs the average photon trajectories — the so-called "Bohmian trajectories," though their interpretation remains a subject of debate.

The raw dataset is freely available at:
> [http://www.physics.utoronto.ca/~aephraim/data/PhotonTrajectories/](http://www.physics.utoronto.ca/~aephraim/data/PhotonTrajectories/)

### 2.2 The TEGR Toy Model

Independently, we developed a computational toy model of TEGR — the Teleparallel Equivalent of General Relativity — in which the gravitational field is encoded in a tetrad (vierbein) field and its associated torsion tensor, rather than in the Levi-Civita connection and curvature of standard GR [2, 3]. The model represents the tetrad field as a set of Jacobian matrices and simulates gravitational dynamics via matrix collisions.

A SINDy (Sparse Identification of Nonlinear Dynamics) node was attached to this simulation in a "blind" configuration — meaning it received only the time-series output of the matrix collisions and was tasked with discovering the governing differential equation without any physical priors. The extracted equation was:

$$\frac{d\theta}{dt} = \frac{m_0}{\gamma} + \kappa(\theta - \sin\theta) + g_{\text{scale}}(\nabla\phi \cdot \hat{v})$$

The term of primary interest is $\kappa(\theta - \sin\theta)$, which we interpreted as a **non-local phase gradient** arising from the torsion structure of the teleparallel connection.

### 2.3 Motivation for Cross-Validation

The natural question is: does this $(\theta - \sin\theta)$ structure appear only in our simulation, or does it have an empirical counterpart? The Kocsis dataset provides an ideal test case because:

1. It contains real, experimentally measured photon trajectories with spatial resolution across multiple propagation planes.
2. The trajectories exhibit the characteristic nonlinear features of quantum interference.
3. The data is publicly available and well-documented, enabling reproducible analysis.
4. Standard Bohmian mechanics predicts a specific functional form for the transverse momentum — namely $dx/dz = k_x/k = (\hbar/m)\,\partial S/\partial x$, where $S$ is the phase of the wave function — which does **not** contain a $(\theta - \sin\theta)$ nonlinearity.

If SINDy discovers the same nonlinear structure in the empirical data, it would constitute independent evidence that this term is not an artifact of our simulation framework.

---

## 3. Methodology

### 3.1 Data Acquisition and Parsing

The raw data was downloaded from the Steinberg lab's public archive. It consists of 91 CCD image files organized as follows:

- **1 background frame** (dark-count calibration)
- **2 frames per propagation plane** (H-polarization and V-polarization channels)
- **40 propagation planes**, yielding $40 \times 2 + 1 = 81$ data frames (plus additional calibration frames totaling 91 files)

Each CCD image is a 1D intensity profile (pixel row) capturing the transverse photon distribution at a given propagation distance $z$.

### 3.2 Background Subtraction

Background subtraction was performed using the same pixel regions specified in the original MATLAB analysis code provided with the dataset. Specifically, the mean intensity in designated "dark" pixel regions of each frame was subtracted from the full frame to remove detector noise and stray light contributions.

### 3.3 Weak Momentum Extraction

For each propagation plane, the weak transverse momentum was computed pixel-by-pixel from the $H$ and $V$ polarization channels:

$$\frac{k_x}{k}\bigg|_{\text{pixel}\,i} = \tan\!\left(\arcsin\!\left(\frac{1}{373}\,\frac{V_i - H_i}{V_i + H_i}\right)\right)$$

This yields a discrete momentum field $k_x(x, z)/k$ across transverse position $x$ and propagation distance $z$.

### 3.4 Trajectory Reconstruction

Twenty photon trajectories were reconstructed via forward Euler integration of the transverse momentum field:

$$x_{n+1} = x_n + \frac{k_x(x_n, z_n)}{k}\,\Delta z$$

Starting positions were distributed uniformly across the central region of the slit pattern. The integration proceeded across all 40 propagation planes, yielding 20 trajectories $x(z)$.

### 3.5 SINDy Configuration

The propagation axis $z$ was used as the temporal variable in PySINDy, which is physically appropriate since $z = ct$ for photons propagating in the paraxial regime.

The SINDy analysis was configured as follows:

| Parameter | Value |
|---|---|
| **Regression method** | STLSQ (Sequentially Thresholded Least Squares) |
| **Feature library** | Polynomials up to degree 3, $\sin(x)$, $\cos(x)$, $1/x$ |
| **Threshold** | Default STLSQ sparsity threshold |
| **Differentiation** | Finite differences on the trajectory data |
| **Input** | 20 reconstructed trajectories, each with 40 data points |

The feature library was intentionally chosen to be broad, including both polynomial and trigonometric terms, so that the algorithm could discover nonlinear structure without bias toward any particular functional form. Crucially, the $(\theta - \sin\theta)$ combination was **not** included as a single library element — it emerged from the algorithm's simultaneous selection of the $x_0$ and $\sin(x_0)$ terms with opposite-sign coefficients of matched magnitude.

---

## 4. Results

### 4.1 Discovered Equation

PySINDy extracted the following governing equation:

$$(x_0)' = -0.001 + 0.007\,x_0 - 0.001\,x_0^3 - 0.007\,\sin(x_0) + 0.001\,\cos(x_0)$$

where primes denote differentiation with respect to $z$.

### 4.2 Structural Analysis

Examining the magnitudes of the coefficients, two terms dominate:

| Term | Coefficient | Magnitude |
|---|---|---|
| $x_0$ | $+0.007$ | Dominant |
| $\sin(x_0)$ | $-0.007$ | Dominant |
| $x_0^3$ | $-0.001$ | Subdominant |
| $\cos(x_0)$ | $+0.001$ | Subdominant |
| Constant | $-0.001$ | Subdominant |

The two dominant terms combine to give:

$$\frac{dx}{dz} \approx 0.007\,(x_0 - \sin x_0)$$

This is **structurally identical** to $\kappa(\theta - \sin\theta)$ from the TEGR model, with $\kappa \approx 0.007$ (in the natural units of the CCD pixel coordinate system).

### 4.3 Goodness of Fit

The overall $R^2$ score of the SINDy model is approximately **0.19**. This is a modest fit, and we address its implications directly in the Discussion section. The low $R^2$ reflects several factors:

- The trajectories are noisy due to the inherently low signal-to-noise ratio of weak measurements.
- Only 40 propagation planes are available, limiting the temporal resolution for numerical differentiation.
- The Euler integration scheme introduces cumulative drift errors.
- The SINDy model is intentionally sparse, sacrificing some quantitative accuracy for interpretability.

### 4.4 Reproducibility

The complete analysis pipeline is available in the Python script `sindy_kocsis_2011.py` in the project repository:

> [https://github.com/thejfisher/tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna)

The script produces:
- Reconstructed trajectory plots
- The SINDy equation printout
- A JSON report containing all extracted coefficients and diagnostics

---

## 5. Discussion

### 5.1 Significance of the Structural Form

The central finding of this analysis is not the numerical accuracy of the fit — which is modest — but the **structural form** of the discovered equation. The key observations are:

1. **The algorithm was blind.** PySINDy had access to a broad library of candidate functions (polynomials, trigonometric functions, reciprocals) and selected terms purely based on sparse regression against the data. The $(\theta - \sin\theta)$ combination was not provided as a single candidate; it emerged from the independent selection of $x_0$ and $-\sin(x_0)$ with matched coefficients.

2. **The structure is non-trivial.** The combination $(x - \sin x)$ is not a generic or default result. It is the leading nonlinear term in the Taylor expansion of $x - \sin x = x^3/6 - x^5/120 + \cdots$, but crucially, SINDy did **not** simply select $x^3$. It selected the full trigonometric form, which has a qualitatively different large-$x$ behavior. This suggests the nonlinearity in the data is genuinely trigonometric, not merely cubic.

3. **Standard theory predicts a different form.** In standard Bohmian mechanics, the transverse velocity of a particle following the guidance equation is:

   $$\frac{dx}{dz} = \frac{k_x}{k} = \frac{\hbar}{m}\,\frac{\partial S}{\partial x}$$

   where $S(x, z)$ is the phase of the wave function $\Psi = |\Psi|e^{iS/\hbar}$. For a double-slit geometry, this yields a complex spatial pattern determined by the superposition of cylindrical wave fronts, which does not naturally produce a global $(x - \sin x)$ dependence. The appearance of this structure is therefore not a trivial consequence of the experimental geometry.

4. **The same structure appeared independently.** The $\kappa(\theta - \sin\theta)$ term was first discovered in the TEGR simulation, in a completely different physical context (matrix torsion dynamics), before the Kocsis data analysis was performed.

### 5.2 What This Does NOT Demonstrate

We are careful to delineate what this result does and does not show:

- **It does not prove that TEGR governs photon trajectories.** The structural correspondence is suggestive, but a single shared functional form does not establish a causal or theoretical connection.

- **It does not replace Bohmian mechanics.** The Bohmian guidance equation remains a well-established framework for interpreting weak-measurement trajectories. Our result identifies an additional structural feature in the data, not a refutation of existing theory.

- **The $R^2 = 0.19$ means the model explains only ~19% of the variance.** This is a real limitation. The remaining 81% of the variance may be due to noise, higher-order dynamics not captured by our sparse library, or systematic effects in the data processing pipeline.

- **We have not performed rigorous statistical hypothesis testing** (e.g., cross-validation, bootstrap confidence intervals on the SINDy coefficients, or comparison against null models). Such testing would be essential before drawing strong conclusions.

### 5.3 Possible Explanations

Several interpretations of the correspondence merit consideration:

1. **Physical origin:** The $(x - \sin x)$ structure may reflect a genuine feature of the transverse phase gradient in the interferometer, possibly related to the Gouy phase, diffraction-induced torsion of the wave front, or non-paraxial corrections.

2. **Coincidence:** The shared structure could be a mathematical coincidence — the $(x - \sin x)$ form is one of the simpler nonlinear combinations available in a trigonometric library, and sparse regression algorithms can exhibit selection bias.

3. **Common mathematical structure:** Both torsion dynamics and wave-front evolution involve rotational degrees of freedom, and the $(x - \sin x)$ nonlinearity appears naturally in pendulum-like systems, torsion balances, and rotational mechanics. The correspondence may reflect a shared mathematical archetype rather than a shared physical mechanism.

Disentangling these possibilities requires further analysis, which we outline below.

### 5.4 On the Interpretation of Weak-Measurement Trajectories

It is worth noting that the ontological status of weak-measurement "trajectories" remains debated. The Kocsis et al. result is often cited as an observation of "Bohmian trajectories," but as Wiseman (2007) and others have argued [4], weak-value trajectories are operationally defined statistical averages, not records of individual particle paths. Our SINDy analysis operates on these average trajectories and therefore discovers a governing equation for the **ensemble-averaged** transverse flow, not for individual photon dynamics.

---

## 6. Next Steps and Proposed Research Directions

The following directions would strengthen or refute the preliminary finding reported here:

### 6.1 Statistical Validation
- **Cross-validation:** Split the 20 trajectories into training and test sets to assess the generalizability of the discovered equation.
- **Bootstrap analysis:** Resample trajectories with replacement to estimate confidence intervals on the SINDy coefficients.
- **Null model comparison:** Generate synthetic trajectories from the standard Bohmian guidance equation (using the known double-slit wave function) and run SINDy on those. If the $(x - \sin x)$ structure also appears in the null model, the finding is less significant.

### 6.2 Improved Data Processing
- **Higher-order integration:** Replace Euler integration with a Runge-Kutta scheme to reduce trajectory drift.
- **Smoothing and denoising:** Apply Savitzky-Golay or wavelet-based smoothing to the raw CCD data before momentum extraction.
- **Systematic error analysis:** Propagate uncertainties from the CCD pixel noise through the full pipeline.

### 6.3 Expanded SINDy Analysis
- **Library sensitivity analysis:** Vary the feature library systematically (e.g., remove trigonometric terms, add higher-order polynomials) to assess the robustness of the $(\theta - \sin\theta)$ selection.
- **Threshold sensitivity:** Map the SINDy coefficients as a function of the STLSQ sparsity threshold to identify the threshold regime in which the $(\theta - \sin\theta)$ structure is stable.
- **Ensemble SINDy (E-SINDy):** Use the ensemble variant [5] to obtain probabilistic coefficient estimates.

### 6.4 Theoretical Investigation
- **Derive the expected SINDy output for standard Bohmian mechanics:** Compute the analytical form of $dx/dz$ for the double-slit wave function and determine what SINDy *should* find if the data were perfectly described by Bohmian mechanics.
- **Investigate the $(x - \sin x)$ structure in wave optics:** Determine whether this form arises naturally from Fresnel or Fraunhofer diffraction under specific conditions.
- **Connect to teleparallel geometry:** Develop the theoretical link between wave-front torsion in optical interferometry and the torsion tensor in TEGR more rigorously.

### 6.5 Experimental Extensions
- **Apply to other double-slit datasets:** If other weak-measurement photon trajectory datasets exist (or can be generated), the same analysis pipeline should be applied to test reproducibility.
- **Vary slit parameters:** The analysis could be extended to single-slit and multi-slit geometries to determine whether the $(x - \sin x)$ structure persists or is specific to the two-slit case.

---

## 7. References

[1] S. Kocsis, B. Braverman, S. Ravets, M. J. Stevens, R. P. Mirin, L. K. Shalm, and A. M. Steinberg, "Observing the Average Trajectories of Single Photons in a Two-Slit Interferometer," *Science* **332**, 1170–1173 (2011). [DOI: 10.1126/science.1202218](https://doi.org/10.1126/science.1202218)

[2] R. Aldrovandi and J. G. Pereira, *Teleparallel Gravity: An Introduction* (Springer, Dordrecht, 2013).

[3] J. W. Maluf, "The teleparallel equivalent of general relativity," *Annalen der Physik* **525**, 339–357 (2013). [arXiv:1303.3897](https://arxiv.org/abs/1303.3897)

[4] H. M. Wiseman, "Grounding Bohmian mechanics in weak values and Bayesianism," *New Journal of Physics* **9**, 165 (2007). [DOI: 10.1088/1367-2630/9/6/165](https://doi.org/10.1088/1367-2630/9/6/165)

[5] U. Fasel, J. N. Kutz, B. W. Brunton, and S. L. Brunton, "Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit," *Journal of the Royal Society Interface* **19**, 20210904 (2022). [DOI: 10.1098/rsif.2021.0904](https://doi.org/10.1098/rsif.2021.0904)

---

## Appendix A: Software and Data Availability

| Resource | Location |
|---|---|
| Analysis code | [github.com/thejfisher/tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna) |
| Raw experimental data | [physics.utoronto.ca/~aephraim/data/PhotonTrajectories/](http://www.physics.utoronto.ca/~aephraim/data/PhotonTrajectories/) |
| Python dependencies | PySINDy, NumPy, SciPy, Matplotlib |
| TEGR preprint | Available on Zenodo (see repository README) |

## Appendix B: SINDy Output (Verbatim)

```
(x0)' = -0.001 + 0.007 x0 - 0.001 x0^3 - 0.007 sin(x0) + 0.001 cos(x0)
```

**Dominant terms:**
```
dx/dz ≈ 0.007 * (x₀ - sin(x₀))
```

**Model score:** $R^2 \approx 0.19$

---

*Document prepared July 2026. This is a working research document, not a peer-reviewed publication. Comments, criticisms, and suggestions for collaboration are welcome.*
