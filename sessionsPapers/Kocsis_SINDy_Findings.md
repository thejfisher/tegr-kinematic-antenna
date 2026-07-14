# Data-Driven Equation Discovery from Weak-Measurement Photon Trajectories: A Collinearity-Limited Test of Structural Correspondence with a Teleparallel Gravity Model

**J. Fisher**
Independent Researcher
July 2026 (Revised after sanity checks)

**Contact:** [GitHub — tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna)

---

## 1. Abstract

We report a data-driven equation discovery analysis applied to the raw experimental dataset of Kocsis et al. (2011), in which average photon trajectories through a double-slit interferometer were reconstructed via weak measurement. Using PySINDy — a sparse regression framework for identifying governing equations from data [6] — we extracted governing dynamical equations for the transverse deflection of photons as a function of propagation distance.

When applied to a mixed polynomial–trigonometric library in the original (mm-scale) units, SINDy produced an equation whose dominant terms combine to give $dx/dz \approx \kappa(x - \sin x)$. This same structural form — $\kappa(\theta - \sin\theta)$ — was independently derived from a toy model of Teleparallel Equivalent General Relativity (TEGR). The apparent correspondence motivated further investigation.

**However, subsequent sanity checks reveal that this structural identification is fundamentally ambiguous.** In the transverse amplitude range of the Kocsis data ($|x/L| \leq 0.32$ in dimensionless units, where $L = 4.68$ mm is the slit separation), the functions $(x - \sin x)$ and $x^3/6$ are correlated at $r > 0.99999$. They are numerically indistinguishable. A polynomial-only library recovers a cubic term with $R^2 = 0.105$ in dimensionless units; adding trigonometric terms improves this only marginally to $R^2 = 0.151$. The SINDy algorithm cannot discriminate between these two functional forms at the amplitudes present in the data.

The nonlinear signal in the data is **real** — both the cubic and the trigonometric fits detect it — but its functional form is **underdetermined**. Resolving whether the underlying nonlinearity is genuinely trigonometric (as the TEGR model predicts) or merely cubic (as a generic Taylor expansion would produce) requires data at larger transverse excursions where $(x - \sin x)$ and $x^3/6$ diverge. We present these findings as a precisely defined open question suitable for further experimental and theoretical investigation.

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

If SINDy discovers the same nonlinear structure in the empirical data, it would constitute independent evidence that this term is not an artifact of our simulation framework. **However, as the sanity checks in Section 4.5 demonstrate, establishing this correspondence requires care — the data must span a sufficient amplitude range to distinguish the trigonometric form from its leading Taylor approximation.**

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

### 4.1 Initial Discovered Equation (mm-scale, full library)

PySINDy, applied to the mm-scale trajectory data with the full polynomial + trigonometric + reciprocal library, extracted the following governing equation:

$$(x_0)' = -0.001 + 0.007\,x_0 - 0.001\,x_0^3 - 0.007\,\sin(x_0) + 0.001\,\cos(x_0)$$

where primes denote differentiation with respect to $z$.

### 4.2 Structural Analysis of Initial Fit

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

**Model score:** $R^2 \approx 0.19$ (mm-scale data, full library).

### 4.3 Limitations of the Initial Analysis

The initial result, taken at face value, appeared to show that SINDy preferentially selected the trigonometric form $\kappa(x - \sin x)$ over a purely polynomial approximation. However, this interpretation was premature. There are two critical issues:

1. **Unit-dependence of trigonometric functions.** The transverse positions in mm-scale coordinates (range approximately $\pm 1.5$ mm) mean that $\sin(x)$ in the SINDy library is being evaluated at arguments far from the regime where sine's periodic structure becomes relevant. At $|x| \lesssim 1.5$, the function $\sin(x) \approx x - x^3/6$, and the combination $(x - \sin x) \approx x^3/6$.

2. **Threshold sensitivity in mm-scale units.** The coefficient magnitudes ($\sim 0.007$) are small enough that the default STLSQ thresholding behaves differently depending on library composition. This was not systematically explored in the initial analysis.

These concerns motivated the sanity checks presented in Section 4.5.

### 4.4 Goodness of Fit

The overall $R^2$ score of the initial SINDy model is approximately **0.19** in mm-scale units. In dimensionless units (see Section 4.5), the comparable fits yield $R^2$ between **0.105** (polynomial only) and **0.151** (full library). The low $R^2$ reflects several factors:

- The trajectories are noisy due to the inherently low signal-to-noise ratio of weak measurements.
- Only 40 propagation planes are available, limiting the temporal resolution for numerical differentiation.
- The Euler integration scheme introduces cumulative drift errors.
- The SINDy model is intentionally sparse, sacrificing some quantitative accuracy for interpretability.

### 4.5 Sanity Checks and the Collinearity Problem

After the initial analysis, we performed a series of diagnostic tests to assess the robustness and interpretability of the SINDy result. These checks reveal a fundamental ambiguity that was not apparent in the initial analysis.

#### 4.5.1 Test 1: Null Test — Polynomial-Only Library (mm-scale)

To test whether the trigonometric terms are necessary, we ran SINDy with a polynomial-only library (no trigonometric or reciprocal terms) on the mm-scale data.

| Library | Discovered Equation | $R^2$ |
|---|---|---|
| Polynomial degree 3 | $(x_0)' = 0.000$ | $-0.000227$ |
| Polynomial degree 5 | $(x_0)' = 0.000$ | $-0.000227$ |

**Result:** All polynomial coefficients were eliminated by the STLSQ threshold. The algorithm found nothing.

**Interpretation:** This does **not** mean polynomials cannot fit the data. It means the default sparsity threshold is too aggressive for the tiny coefficient magnitudes that arise in mm-scale units. The polynomial coefficients (which would be $\sim 10^{-4}$ or smaller) fall below the threshold and are zeroed out, while the trigonometric coefficients ($\sim 0.007$) survive. This is a **unit-dependent artifact**, not evidence for the intrinsic superiority of the trigonometric form.

#### 4.5.2 Test 2: Dimensionless Units

To remove the unit-dependence, we normalized the transverse position by the slit separation $L = 4.68$ mm, defining $\tilde{x} = x / L$. In these units, the transverse excursion range is $|\tilde{x}| \leq 0.32$.

| Library | Discovered Equation | $R^2$ |
|---|---|---|
| Polynomial only | $(\tilde{x}_0)' = 0.004\,\tilde{x}_0^3$ | $0.1054$ |
| Full (poly + trig + $1/x$) | $(\tilde{x}_0)' = 8.708\,\tilde{x}_0 - 1.430\,\tilde{x}_0^3 - 8.709\,\sin(\tilde{x}_0)$ | $0.1512$ |

**Key finding:** The cubic term alone captures most of the nonlinear signal ($R^2 = 0.105$). Adding trigonometric terms improves the fit only marginally ($R^2 = 0.151$), an improvement of $\Delta R^2 = 0.046$.

In the full-library dimensionless fit, the dominant terms again combine to $\approx 8.7\,(\tilde{x}_0 - \sin\tilde{x}_0)$, but this is expected: in the range $|\tilde{x}| \leq 0.32$, the function $(x - \sin x)$ is almost perfectly proportional to $x^3/6$, so the two representations are interchangeable.

#### 4.5.3 Test 3: Threshold Sweep (mm-scale, full library)

We swept the STLSQ sparsity threshold across several orders of magnitude to test whether $\sin(x)$ is preferentially retained or preferentially eliminated:

| Threshold | Surviving Terms | $R^2$ |
|---|---|---|
| $5 \times 10^{-4}$ | $x_0$, $\sin(x_0)$, and others | $0.1518$ |
| $1 \times 10^{-3}$ | None (all eliminated) | — |

**Result:** At threshold $5 \times 10^{-4}$, the $\sin(x_0)$ term persists. At threshold $1 \times 10^{-3}$, **all** terms are eliminated — not just sine. The trigonometric term is not uniquely fragile; it disappears alongside the polynomial terms. This is consistent with the entire signal being near the noise floor of the threshold, not with sine being a spurious addition to an otherwise robust polynomial.

#### 4.5.4 Collinearity Diagnostic

The root cause of the ambiguity is quantified by a direct correlation analysis. In the dimensionless amplitude range $|\tilde{x}| \leq 0.32$:

$$\text{corr}\!\left(\tilde{x} - \sin\tilde{x},\;\frac{\tilde{x}^3}{6}\right) = 0.99999957$$

The two functional forms are numerically indistinguishable at the amplitudes present in the Kocsis data. **No regression method — sparse or otherwise — can reliably discriminate between them at these scales.** They diverge only for $|x| \gtrsim 1$ (in radians), corresponding to transverse excursions comparable to the slit separation or larger.

### 4.6 Reproducibility

The complete analysis pipeline, including all sanity checks, is available in the Python script `sindy_kocsis_2011.py` in the project repository:

> [https://github.com/thejfisher/tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna)

The script produces:
- Reconstructed trajectory plots
- The SINDy equation printout
- Sanity check results (polynomial-only, dimensionless, threshold sweep)
- A JSON report containing all extracted coefficients and diagnostics

---

## 5. Discussion

### 5.1 The Signal is Real, but Its Form is Underdetermined

The central finding of this analysis is that the Kocsis photon trajectory data contains a real nonlinear signal — both polynomial-only and trigonometric-enriched libraries detect it, with $R^2$ values in the range $0.10$–$0.15$ in dimensionless units. The signal is weak but reproducible.

However, the **functional form** of this nonlinearity cannot be determined from the available data. The SINDy algorithm's preference for the trigonometric representation $\kappa(x - \sin x)$ over the polynomial $\kappa x^3/6$ is an artifact of:

1. **The collinearity trap.** At transverse amplitudes $|\tilde{x}| \leq 0.32$, the Taylor series $x - \sin x = x^3/6 - x^5/120 + \cdots$ converges so rapidly that the two forms are indistinguishable to any regression method. The correlation coefficient exceeds $0.99999$.

2. **Unit-dependent threshold effects.** In mm-scale coordinates, the polynomial coefficients are numerically tiny and are eliminated by the default STLSQ threshold, while the trigonometric coefficients (which absorb the nonlinearity into a different parameterization) happen to be large enough to survive. This gives a misleading impression of the algorithm "choosing" the trigonometric form.

3. **The dimensionless sine problem.** When $\sin(x)$ is evaluated at arguments $|x| \ll 1$, it is effectively a polynomial. The SINDy library's trigonometric basis functions do not provide independent information in this regime — they are redundant with the polynomial basis.

### 5.2 Reinterpretation of the Structural Correspondence

The initial claim — that SINDy "discovered" the $(\theta - \sin\theta)$ structure in the Kocsis data, matching the TEGR model — must be substantially weakened:

- **What we can say:** There is a nonlinear component in the transverse deflection data that is consistent with a leading cubic dependence on position. Both $x^3$ and $(x - \sin x)$ fit this signal approximately equally well.

- **What we cannot say:** That the data prefer the trigonometric form over the polynomial form, or that the structural correspondence with the TEGR model is anything more than a consequence of both models sharing the same leading-order Taylor expansion.

- **What would be needed:** Data at transverse amplitudes $|x/L| \gtrsim 1$, where $(x - \sin x)$ and $x^3/6$ diverge qualitatively. The trigonometric form saturates and oscillates; the cubic grows without bound. An experiment with wider slit separation, higher-order diffraction maxima, or a different interferometer geometry could in principle reach this regime.

### 5.3 What This Does NOT Demonstrate

We are careful to delineate what this result does and does not show:

- **It does not demonstrate that SINDy preferentially selects the trigonometric form.** The apparent preference is an artifact of unit-dependent thresholding. In dimensionless units with a polynomial-only library, a simple cubic captures most of the signal.

- **It does not prove that TEGR governs photon trajectories.** The structural correspondence between SINDy's output and the TEGR model is, at present, indistinguishable from a shared cubic leading-order behavior.

- **It does not replace Bohmian mechanics.** The Bohmian guidance equation remains a well-established framework for interpreting weak-measurement trajectories. Our result identifies a weak nonlinear component in the data, not a refutation of existing theory.

- **The $R^2 \approx 0.10$–$0.15$ means the model explains only 10–15% of the variance.** The remaining variance may be due to noise, higher-order dynamics not captured by our sparse library, or systematic effects in the data processing pipeline.

- **We have not performed rigorous statistical hypothesis testing** beyond the sanity checks reported here. Cross-validation, bootstrap confidence intervals, and comparison against null models would be essential before drawing strong conclusions.

### 5.4 The Collinearity Problem as a Research Opportunity

Rather than viewing the collinearity as a negative result, we frame it as a **precisely defined experimental question**: at what transverse amplitude does the data begin to discriminate between $\kappa(x - \sin x)$ and $\kappa x^3/6$?

The answer is known analytically. The next-to-leading-order correction to $(x - \sin x) \approx x^3/6$ is $-x^5/120$. The fractional deviation between the two forms exceeds 1% when $|x| \gtrsim 0.77$ (radians), i.e., when the transverse excursion is roughly $0.77 L$, or about 3.6 mm for the Kocsis geometry. The maximum excursion in the Kocsis data is approximately $0.32 L \approx 1.5$ mm — a factor of $\sim 2.4$ too small.

This means a future experiment would need either:
- A wider slit separation (reducing $L$, so the same physical excursion maps to larger $\tilde{x}$), or
- A longer propagation distance (allowing trajectories to diverge further before measurement), or
- A different interferometric geometry that produces larger transverse excursions relative to the natural angular scale

### 5.5 Possible Explanations for the Nonlinear Signal

The **existence** of the cubic/trigonometric nonlinear signal (as opposed to its precise functional form) merits physical interpretation:

1. **Diffraction-induced nonlinearity:** The transverse momentum field of a double-slit interference pattern is inherently nonlinear. The Bohmian guidance equation $dx/dz \propto \partial S / \partial x$ can produce cubic and higher-order dependencies on $x$ near the intensity nodes. The observed nonlinearity may simply reflect the expected structure of the quantum phase gradient.

2. **Non-paraxial corrections:** The paraxial approximation assumes small transverse momenta. Corrections to this approximation introduce nonlinear terms in $k_x/k$ that could contribute a cubic signal.

3. **Gouy phase and wave-front curvature:** The Gouy phase shift and Gaussian beam divergence introduce $z$-dependent corrections to the transverse phase gradient that could manifest as a position-dependent nonlinearity.

4. **Torsion-related structure (TEGR hypothesis):** If the transverse phase gradient genuinely encodes a $(\theta - \sin\theta)$ nonlinearity — as opposed to a cubic — this would be a non-trivial signature potentially connected to geometric torsion of the wave front. This hypothesis is not falsified by the present data, but it is also not confirmed.

Disentangling these possibilities requires both theoretical calculation (what does Bohmian mechanics predict for the SINDy output?) and experimental data at larger amplitudes.

### 5.6 On the Interpretation of Weak-Measurement Trajectories

It is worth noting that the ontological status of weak-measurement "trajectories" remains debated. The Kocsis et al. result is often cited as an observation of "Bohmian trajectories," but as Wiseman (2007) and others have argued [4], weak-value trajectories are operationally defined statistical averages, not records of individual particle paths. Our SINDy analysis operates on these average trajectories and therefore discovers a governing equation for the **ensemble-averaged** transverse flow, not for individual photon dynamics.

---

## 6. Next Steps and Proposed Research Directions

### 6.1 Completed Sanity Checks

The following diagnostics have been performed and are reported in Section 4.5:

- ✅ **Null test (polynomial-only library, mm-scale):** Confirmed that threshold artifacts suppress polynomial fits in mm-scale units.
- ✅ **Dimensionless reformulation:** Showed that the cubic alone captures $R^2 = 0.105$, with marginal improvement from trig terms.
- ✅ **Threshold sweep:** Confirmed that $\sin(x)$ is not uniquely fragile — all terms vanish together.
- ✅ **Collinearity diagnostic:** Quantified the $r = 0.99999957$ correlation between $(x - \sin x)$ and $x^3/6$ in the data range.

### 6.2 Statistical Validation (Outstanding)

- **Cross-validation:** Split the 20 trajectories into training and test sets to assess the generalizability of the discovered equation.
- **Bootstrap analysis:** Resample trajectories with replacement to estimate confidence intervals on the SINDy coefficients.
- **Null model comparison:** Generate synthetic trajectories from the standard Bohmian guidance equation (using the known double-slit wave function) and run SINDy on those. If a cubic or $(x - \sin x)$ structure also appears in the null model, the finding is less significant — but it would also clarify whether the observed nonlinearity is simply the expected Bohmian phase structure.

### 6.3 Improved Data Processing

- **Higher-order integration:** Replace Euler integration with a Runge-Kutta scheme to reduce trajectory drift.
- **Smoothing and denoising:** Apply Savitzky-Golay or wavelet-based smoothing to the raw CCD data before momentum extraction.
- **Systematic error analysis:** Propagate uncertainties from the CCD pixel noise through the full pipeline.

### 6.4 Expanded SINDy Analysis

- **Ensemble SINDy (E-SINDy):** Use the ensemble variant [5] to obtain probabilistic coefficient estimates and assess model uncertainty.
- **Information-theoretic model selection:** Apply AIC/BIC or cross-validated likelihood to formally compare the cubic model against the trigonometric model, accounting for the difference in model complexity.
- **Extended library tests:** Include $x^5$, $x^7$ terms to test whether higher-order polynomial corrections improve the fit comparably to trigonometric terms.

### 6.5 Theoretical Investigation

- **Derive the expected SINDy output for standard Bohmian mechanics:** Compute the analytical form of $dx/dz$ for the double-slit wave function and determine what SINDy *should* find if the data were perfectly described by Bohmian mechanics. This is the single most important outstanding task — it would determine whether the observed nonlinearity is anomalous or expected.
- **Investigate the $(x - \sin x)$ structure in wave optics:** Determine whether this form arises naturally from Fresnel or Fraunhofer diffraction under specific conditions.
- **Connect to teleparallel geometry:** Develop the theoretical link between wave-front torsion in optical interferometry and the torsion tensor in TEGR more rigorously — but only if the experimental ambiguity can first be resolved.

### 6.6 The Critical Experiment

The most direct path to resolving the collinearity ambiguity is an experiment (or dataset) in which photon trajectories reach transverse excursions $|x/L| \gtrsim 1$. Specific options:

- **Wider slit separation datasets:** If other weak-measurement experiments exist with different slit geometries, the same SINDy pipeline should be applied.
- **Multi-slit or grating geometries:** Higher diffraction orders produce larger transverse excursions that could break the degeneracy.
- **Numerical experiment:** Generate synthetic trajectories from a wave function that is exactly known, at both small and large amplitudes, and test whether SINDy can distinguish the cubic from the trigonometric form as a function of amplitude range.

---

## 7. Conclusion

The Kocsis weak-measurement photon trajectory data contains a detectable nonlinear signal in the transverse deflection dynamics ($R^2 \approx 0.10$–$0.15$ in dimensionless units). This signal is consistent with either a cubic dependence $dx/dz \propto x^3$ or a trigonometric form $dx/dz \propto (x - \sin x)$. The two representations are numerically indistinguishable at the transverse amplitudes present in the data, with a correlation exceeding $0.99999$.

The question of whether the underlying nonlinearity is genuinely trigonometric — as predicted by the TEGR toy model — or merely cubic remains **open**. This is not a failure of the analysis; it is a precise statement of the limits of what the current data can resolve. The resolution requires either:

1. Experimental data at larger transverse amplitudes ($|x/L| \gtrsim 1$), or
2. A theoretical prediction from standard Bohmian mechanics that specifies the expected SINDy output and thereby determines whether the observed signal is anomalous.

We present this as a well-defined research question: **does the transverse phase gradient of a double-slit interference pattern encode a periodic nonlinearity, or is the observed signal fully accounted for by the leading polynomial term?** The answer has implications for both the interpretation of weak-measurement trajectories and for the potential connection between wave-optical torsion and teleparallel geometry.

---

## 8. References

[1] S. Kocsis, B. Braverman, S. Ravets, M. J. Stevens, R. P. Mirin, L. K. Shalm, and A. M. Steinberg, "Observing the Average Trajectories of Single Photons in a Two-Slit Interferometer," *Science* **332**, 1170–1173 (2011). [DOI: 10.1126/science.1202218](https://doi.org/10.1126/science.1202218)

[2] R. Aldrovandi and J. G. Pereira, *Teleparallel Gravity: An Introduction* (Springer, Dordrecht, 2013).

[3] J. W. Maluf, "The teleparallel equivalent of general relativity," *Annalen der Physik* **525**, 339–357 (2013). [arXiv:1303.3897](https://arxiv.org/abs/1303.3897)

[4] H. M. Wiseman, "Grounding Bohmian mechanics in weak values and Bayesianism," *New Journal of Physics* **9**, 165 (2007). [DOI: 10.1088/1367-2630/9/6/165](https://doi.org/10.1088/1367-2630/9/6/165)

[5] U. Fasel, J. N. Kutz, B. W. Brunton, and S. L. Brunton, "Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit," *Journal of the Royal Society Interface* **19**, 20210904 (2022). [DOI: 10.1098/rsif.2021.0904](https://doi.org/10.1098/rsif.2021.0904)

[6] S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering governing equations from data by sparse identification of nonlinear dynamical systems," *Proceedings of the National Academy of Sciences* **113**, 3932–3937 (2016). [DOI: 10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

---

## Appendix A: Software and Data Availability

| Resource | Location |
|---|---|
| Analysis code | [github.com/thejfisher/tegr-kinematic-antenna](https://github.com/thejfisher/tegr-kinematic-antenna) |
| Raw experimental data | [physics.utoronto.ca/~aephraim/data/PhotonTrajectories/](http://www.physics.utoronto.ca/~aephraim/data/PhotonTrajectories/) |
| Python dependencies | PySINDy, NumPy, SciPy, Matplotlib |
| TEGR preprint | Available on Zenodo (see repository README) |

## Appendix B: SINDy Output (Verbatim)

### B.1 Initial Run (mm-scale, full library)

```
(x0)' = -0.001 + 0.007 x0 - 0.001 x0^3 - 0.007 sin(x0) + 0.001 cos(x0)
```

**Dominant terms:**
```
dx/dz ≈ 0.007 * (x₀ - sin(x₀))
```

**Model score:** $R^2 \approx 0.19$

### B.2 Dimensionless (x̃ = x/L, polynomial only)

```
(x̃₀)' = 0.004 x̃₀³
```

**Model score:** $R^2 = 0.1054$

### B.3 Dimensionless (x̃ = x/L, full library)

```
(x̃₀)' = 8.708 x̃₀ - 1.430 x̃₀³ - 8.709 sin(x̃₀)
```

**Model score:** $R^2 = 0.1512$

### B.4 Collinearity

$$\text{corr}\!\left(\tilde{x} - \sin\tilde{x},\;\frac{\tilde{x}^3}{6}\right) = 0.99999957 \quad \text{for } |\tilde{x}| \leq 0.32$$

---

*Document prepared July 2026, revised after sanity checks. This is a working research document, not a peer-reviewed publication. The ambiguity reported here defines a precise experimental question that we believe warrants further investigation. Comments, criticisms, and suggestions for collaboration are welcome.*
