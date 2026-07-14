#!/usr/bin/env python3
"""
sindy_kocsis_2011.py
====================

Sparse Identification of Nonlinear Dynamics (SINDy) applied to the
empirical average photon trajectories measured by Kocsis, Braverman,
Ravber, Stevens, Mirin, Shalm, and Steinberg (Science 332, 1170, 2011).

Background
----------
In 2011, Sacha Kocsis and colleagues at the University of Toronto performed
a landmark experiment: they used *weak measurements* to reconstruct the
average trajectories of single photons passing through a double-slit
interferometer.  The key idea is that by weakly coupling the photon's
transverse momentum to its polarization state, one can read out the
local transverse wave-vector kx/k at each spatial position without
significantly disturbing the interference pattern.

The raw data consist of CCD camera images taken at 91 discrete imaging
planes along the propagation axis z.  Each image file stores 2048
comma-separated integers:

    - Indices 0–1023   → Horizontal (H) polarization photon counts
    - Indices 1024–2047 → Vertical   (V) polarization photon counts

This script:
    1. Loads those raw CCD files and reconstructs photon trajectories
       by integrating the weak-measurement transverse momentum field
       through the 91 imaging planes.
    2. Normalizes the trajectories to dimensionless coordinates by
       dividing x and z by L_slit (the double-slit separation).
    3. Feeds the resulting trajectory data into PySINDy, treating the
       normalised propagation distance ζ = z/L_slit as the independent
       variable, to discover a sparse governing equation dξ/dζ = f(ξ),
       where ξ = x/L_slit.
    4. Runs TWO SINDy analyses side by side:
         (a) Polynomial-only (degree 3) — a null/baseline test.
         (b) Full library (poly + trig + 1/x) — the experimental test.
       Both results are printed for comparison.
    5. Saves a plot of the trajectories and a JSON report of both
       discovered equations, feature names, and model scores.

Dimensionless normalization
---------------------------
The slit separation L_slit ≈ 4.68 mm is the natural length scale of this
experiment.  It is computed from the Kocsis MATLAB analysis as:

    L_slit = rightmean − leftmean ≈ 19.12 − 14.44 = 4.68 mm

Dividing all x and z values by L_slit makes the SINDy state variable ξ
dimensionless.  This is essential because:
  - sin(x) is only meaningful when x is dimensionless (radians).
  - Without normalization, x is in millimetres, and sin(x_mm) is
    physically meaningless — it conflates units with the function argument.
  - The TEGR prediction θ − sin(θ) requires θ to be a dimensionless
    angle or phase variable.

Collinearity caveat
-------------------
IMPORTANT: In the dimensionless regime where |ξ| < ~0.3 (i.e. roughly
|x| < 1.5 mm, which is the entire range of the Kocsis seed positions),
the Taylor expansion gives:

    sin(ξ) ≈ ξ − ξ³/6 + O(ξ⁵)

Therefore:

    ξ − sin(ξ) ≈ ξ³/6

This means that a cubic polynomial term x³ and the combination
(x − sin(x)) are nearly perfectly correlated in this amplitude range:

    corr(x³, x − sin(x)) > 0.999999

Consequence: SINDy CANNOT reliably distinguish between:
    dξ/dζ = a·ξ³        (pure cubic nonlinearity)
    dξ/dζ = b·(ξ − sin(ξ))   (TEGR-predicted structure)
because both produce essentially identical predictions on this data.

The dual analysis below (polynomial-only vs full-library) is designed to
make this ambiguity explicit.  If both models achieve similar R² scores,
the nonlinear signal is real but its functional form is underdetermined
by the data alone.

References
----------
[1] S. Kocsis et al., "Observing the Average Trajectories of Single
    Photons in a Two-Slit Interferometer," Science 332, 1170 (2011).
[2] S. L. Brunton, J. L. Proctor, J. N. Kutz, "Discovering governing
    equations from data by sparse identification of nonlinear dynamical
    systems," PNAS 113, 3932 (2016).  (SINDy method)
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import os       # Filesystem path manipulation and existence checks
import json     # Serialise the SINDy report dictionary to disk

# ---------------------------------------------------------------------------
# Third-party scientific imports
# ---------------------------------------------------------------------------
import numpy as np                    # Numerical arrays and linear algebra
import matplotlib.pyplot as plt       # Plotting reconstructed trajectories
import pysindy as ps                  # Sparse Identification of Nonlinear Dynamics
from scipy.interpolate import interp1d  # Cubic interpolation of the kx field


# ═══════════════════════════════════════════════════════════════════════════
#  COLLINEARITY NOTE
# ═══════════════════════════════════════════════════════════════════════════
#
# For small dimensionless x (|x| < ~0.3, the regime of this experiment):
#
#     x - sin(x)  ≈  x³/6         (Taylor expansion)
#
# This means the library columns "x³" and "x - sin(x)" have a Pearson
# correlation exceeding 0.999999 over the data range.  SINDy's sparse
# regression cannot meaningfully separate them.
#
# Both the polynomial-only fit  dx/dz = a·x³  and the trig-inclusive fit
# dx/dz = b·x + c·sin(x) (≈ b·(x - sin(x)) when b ≈ -c) will produce
# nearly identical R² scores and nearly identical predictions.
#
# The nonlinear signal is REAL (both models find it), but its functional
# form — cubic vs. sinusoidal — is AMBIGUOUS at these amplitudes.
# Resolving this would require data at larger |x/L_slit| where the cubic
# and sinusoidal forms diverge, or independent theoretical constraints.
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# PHYSICAL CONSTANT: SLIT SEPARATION
# ---------------------------------------------------------------------------
# L_slit is the double-slit separation derived from the Kocsis MATLAB
# analysis (Bohmdataread.m).  The slit centres in pixel-pitch coordinates
# are:
#     leftmean  ≈ 14.44 mm   (left slit centre)
#     rightmean ≈ 19.12 mm   (right slit centre)
#
# L_slit = rightmean − leftmean ≈ 4.68 mm
#
# This is the natural length scale of the double-slit geometry and is used
# to make all coordinates dimensionless: ξ = x / L_slit,  ζ = z / L_slit.
L_SLIT_MM = 4.68  # mm — double-slit separation


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTION: load_and_reconstruct_trajectories
# ═══════════════════════════════════════════════════════════════════════════

def load_and_reconstruct_trajectories(data_dir):
    """
    Load raw CCD image data from the Kocsis et al. (2011) weak-measurement
    experiment and reconstruct average photon trajectories by numerically
    integrating the measured transverse momentum field across imaging planes.

    Physics overview
    ----------------
    At each of the 91 imaging planes (positions along the propagation axis z),
    the CCD records horizontal (H) and vertical (V) polarization counts for
    1024 transverse pixels.  The *weak value* of the transverse momentum is
    extracted from the polarization contrast via:

        kx/k = tan( arcsin( MeasCoeff × (V − H) / (V + H) ) )

    where MeasCoeff = 1/373 is an experimentally calibrated constant
    (determined by Sacha Kocsis and recorded in the original MATLAB source
    code, Bohmdataread.m, included in the published data archive).

    A set of "seed" transverse positions is chosen at the first imaging
    plane, and each seed is propagated forward through successive planes
    using a simple Euler step:

        x_{n+1} = x_n  +  Δz × (kx/k)|_{x_n}

    This yields a family of reconstructed average photon trajectories that
    reproduce the characteristic curving paths through the double-slit
    interference pattern.

    Parameters
    ----------
    data_dir : str
        Absolute path to the directory containing the 91 raw CCD data
        files named Image0.txt, Image1.txt, …, Image90.txt.

    Returns
    -------
    z_range : np.ndarray, shape (end_plane − init_plane,)
        The subset of propagation-axis positions (in mm) over which the
        trajectories are reconstructed.  This serves as the independent
        "time" variable for SINDy.

    trajectory_points : np.ndarray, shape (len(z_range), num_seeds)
        Each column is a single reconstructed trajectory: the transverse
        position x (in mm) as a function of propagation distance z.
    """
    print("Loading empirical data...")

    # ------------------------------------------------------------------
    # PHYSICAL CONSTANTS AND CALIBRATION PARAMETERS
    # ------------------------------------------------------------------

    # Pixel pitch of the CCD camera sensor, in millimetres.
    # Each pixel spans 26 µm = 0.026 mm in physical space.  This converts
    # the integer pixel index into a real-space transverse coordinate x.
    pixelpitch = 0.026  # mm

    # Weak-measurement calibration coefficient.
    # This dimensionless factor was determined experimentally by Sacha
    # Kocsis during calibration of the polarization-based weak measurement.
    # It encodes the coupling strength between the photon's transverse
    # momentum and the induced polarization rotation.  The value 1/373
    # appears in the original MATLAB analysis code (Bohmdataread.m) that
    # was distributed with the published dataset.
    meas_coeff = 1.0 / 373.0

    # ------------------------------------------------------------------
    # NUMBER OF CCD IMAGE FILES
    # ------------------------------------------------------------------
    # The experiment recorded images at 91 discrete positions of a
    # motorised translation stage along the propagation axis z.
    # Each file is named Image0.txt through Image90.txt.
    num_files = 91

    # ------------------------------------------------------------------
    # ACCUMULATORS FOR RAW POLARIZATION DATA
    # ------------------------------------------------------------------
    # H_arrays will collect the 1024-element Horizontal-polarization
    # count vectors from every successfully loaded image file.
    # V_arrays will collect the corresponding Vertical-polarization vectors.
    H_arrays = []
    V_arrays = []

    # ------------------------------------------------------------------
    # FILE LOADING LOOP
    # ------------------------------------------------------------------
    # Iterate over all 91 expected image files.  Each file contains 2048
    # comma-separated integers: the first 1024 are the H-polarization
    # counts per pixel, and the second 1024 are the V-polarization counts.
    for i in range(num_files):
        filename = os.path.join(data_dir, f"Image{i}.txt")

        # Silently skip any missing files (the dataset may be incomplete
        # on some machines, or file numbering may have gaps).
        if not os.path.exists(filename):
            continue

        with open(filename, 'r') as f:
            content = f.read().strip()

            # Skip empty files.
            if not content:
                continue

            # Parse the comma-separated ASCII integers into a float array.
            # The original files use integer counts, but we cast to float
            # for subsequent arithmetic (background subtraction, division).
            raw_data = np.array([float(x) for x in content.split(',') if x.strip()])

            # Each valid file must contain at least 2048 values (1024 H + 1024 V).
            # If not, the file is corrupt or truncated — skip it.
            if len(raw_data) < 2048:
                continue

            # Slice out the two polarization channels.
            # rawh: Horizontal polarization counts, pixels 0–1023.
            # rawv: Vertical   polarization counts, pixels 1024–2047.
            rawh = raw_data[0:1024]
            rawv = raw_data[1024:2048]

            H_arrays.append(rawh)
            V_arrays.append(rawv)

    # Stack into 2-D arrays:  shape = (num_loaded_planes, 1024).
    # Each row is one imaging plane; each column is one CCD pixel.
    H_arrays = np.array(H_arrays)
    V_arrays = np.array(V_arrays)

    print(f"Loaded {len(H_arrays)} image frames.")

    # ------------------------------------------------------------------
    # DEFINE BACKGROUND AND REGION-OF-INTEREST (ROI) PIXEL RANGES
    # ------------------------------------------------------------------
    # These index ranges come directly from the original MATLAB analysis
    # code (Bohmdataread.m) in the Kocsis data archive.
    #
    # Background pixels (bg_indices):
    #   Pixels 400–474 and 750–899 (0-indexed: 399–474, 749–899).
    #   These lie outside the double-slit interference pattern and are
    #   used to estimate the uniform background (stray light + dark
    #   current) that must be subtracted from the signal.
    #
    # Region of interest (roi_indices):
    #   Pixels 475–724 (0-indexed: 474–724).
    #   This window brackets the central portion of the CCD where the
    #   double-slit interference fringes appear.
    bg_indices = np.concatenate((np.arange(399, 475), np.arange(749, 900)))
    roi_indices = np.arange(474, 725)

    # Convert pixel indices to physical transverse positions in mm.
    # Pixel 1 → 0.026 mm, pixel 2 → 0.052 mm, …, pixel 1024 → 26.624 mm.
    # (1-indexed to match the MATLAB convention in the original code.)
    xvals = pixelpitch * np.arange(1, 1025)

    # Extract the transverse coordinate array for the ROI pixels only.
    x_roi = xvals[roi_indices]

    # Centre the ROI coordinates about zero by subtracting the mean.
    # This places the optical axis at x = 0, analogous to what the
    # original MATLAB routine DBAnalyze does.  The centering is purely
    # cosmetic / conventional — it does not affect the physics, but it
    # makes the SINDy-discovered equation coefficients more interpretable.
    x_roi = x_roi - np.mean(x_roi)

    # ------------------------------------------------------------------
    # PROPAGATION AXIS (z) SETUP
    # ------------------------------------------------------------------
    # The 91 translation-stage positions are assumed to be uniformly
    # spaced between 2500 mm and 8500 mm along the propagation axis z
    # (i.e. the distance from the double-slit to the imaging plane).
    # This range spans 6 metres of free-space propagation and covers the
    # full evolution from the near-field (Fresnel) to the far-field
    # (Fraunhofer) diffraction regime.
    z_vals = np.linspace(2500, 8500, len(H_arrays))

    # We reconstruct trajectories only within a subset of the imaging
    # planes.  init_plane = 10 skips the first ten planes where the
    # interference pattern is still forming and the fringe contrast is
    # low, leading to noisy momentum estimates.  end_plane = 50 (or the
    # total number of planes if fewer than 50 were loaded) avoids the
    # far-field regime where the fringes have broadened so much that
    # the weak-measurement signal degrades.
    init_plane = 10
    end_plane = min(50, len(H_arrays))

    # The z-values corresponding to the selected subset of planes.
    # This array will serve as the independent variable ("time") for SINDy.
    z_range = z_vals[init_plane:end_plane]

    # ------------------------------------------------------------------
    # SEED THE INITIAL TRANSVERSE POSITIONS
    # ------------------------------------------------------------------
    # We choose 20 equally spaced seed positions spanning x ∈ [−1.5, +1.5] mm
    # at the first reconstruction plane.  Each seed will be advected
    # forward through the measured transverse-momentum field, tracing
    # out one average photon trajectory.  The spacing and range are chosen
    # to sample the central region of the interference pattern.
    start_x = np.linspace(-1.5, 1.5, 20)

    # current_x holds the instantaneous transverse positions of all 20
    # trajectories at the current imaging plane.  It is updated in-place
    # during the Euler integration loop below.
    current_x = start_x.copy()

    # Accumulator for the trajectory history.  After the loop, this will
    # be a list of arrays, each of shape (20,), one per imaging plane.
    trajectory_points = [current_x.copy()]

    # ------------------------------------------------------------------
    # EULER INTEGRATION OF THE WEAK-MEASUREMENT MOMENTUM FIELD
    # ------------------------------------------------------------------
    # At each pair of successive imaging planes (i → i+1), we:
    #   1. Extract the H and V polarization counts in the ROI.
    #   2. Subtract the background estimated from the bg pixels.
    #   3. Compute normalised probability distributions prob_h, prob_v.
    #   4. Apply the weak-measurement formula to obtain kx/k(x).
    #   5. Interpolate kx/k onto the current trajectory positions.
    #   6. Euler-step: x_{new} = x_{old} + Δz × (kx/k).
    #
    # This is a first-order (Euler) ODE integrator for dx/dz = kx/k.
    # The propagation distance z plays the role of "time."
    for i in range(init_plane, end_plane - 1):
        # ----------------------------------------------------------
        # (1) Raw polarization counts in the ROI for this plane
        # ----------------------------------------------------------
        h = H_arrays[i][roi_indices]   # H-pol counts, shape (251,)
        v = V_arrays[i][roi_indices]   # V-pol counts, shape (251,)

        # ----------------------------------------------------------
        # (2) Background subtraction
        # ----------------------------------------------------------
        # Estimate the uniform background level as the mean count in
        # the background pixel regions (outside the interference pattern).
        bg_h = np.mean(H_arrays[i][bg_indices])
        bg_v = np.mean(V_arrays[i][bg_indices])

        # Subtract the background, clamping to a tiny positive floor
        # (1e-11) to prevent division by zero or negative counts in
        # the subsequent normalisation step.
        h_sub = np.maximum(h - bg_h, 1e-11)
        v_sub = np.maximum(v - bg_v, 1e-11)

        # ----------------------------------------------------------
        # (3) Normalised probability distributions
        # ----------------------------------------------------------
        # Convert background-subtracted counts into probability
        # distributions (sum to 1) for each polarization channel.
        prob_h = h_sub / np.sum(h_sub)
        prob_v = v_sub / np.sum(v_sub)

        # ----------------------------------------------------------
        # (4) Weak-measurement formula for transverse momentum
        # ----------------------------------------------------------
        # The core physics equation from Kocsis et al. (2011).
        #
        # Step 4a: Compute the "raw imaginary weak value" of momentum:
        #
        #   rmimag = MeasCoeff × (prob_V − prob_H) / (prob_H + prob_V)
        #
        # The factor (V−H)/(V+H) is the normalised Stokes parameter
        # that encodes the polarization rotation induced by the weak
        # coupling between transverse momentum and polarization.
        # MeasCoeff = 1/373 converts this rotation into a physical
        # momentum ratio.
        rmimag = meas_coeff * (prob_v - prob_h) / (prob_h + prob_v)

        # Clip to [−0.5, +0.5] to keep arcsin well-behaved.
        # Values outside this range would correspond to unphysical
        # momentum ratios and arise only from noise in low-count pixels.
        rmimag = np.clip(rmimag, -0.5, 0.5)

        # Step 4b: Convert from the weak value to the transverse
        # wave-vector ratio kx/k:
        #
        #   kx/k = tan( arcsin( rmimag ) )
        #
        # The arcsin accounts for the geometric relationship between
        # the weak value and the actual deflection angle; the tan
        # converts from the sine of the angle to the tangent, which
        # is the ratio of transverse to longitudinal momentum
        # (kx / kz ≈ kx / k for small angles).
        kx_k_weak = np.tan(np.arcsin(rmimag))

        # ----------------------------------------------------------
        # (5) Interpolate kx/k onto current trajectory positions
        # ----------------------------------------------------------
        # The kx/k field is known on the discrete ROI pixel grid x_roi.
        # We need its value at each trajectory's current transverse
        # position current_x, which generally falls between pixel
        # centres.  A cubic spline interpolation provides a smooth
        # estimate; extrapolation handles trajectories that have
        # drifted slightly beyond the ROI boundaries.
        f_kx = interp1d(x_roi, kx_k_weak, kind='cubic', fill_value="extrapolate")
        kx_at_x = f_kx(current_x)

        # ----------------------------------------------------------
        # (6) Euler step along the propagation axis
        # ----------------------------------------------------------
        # Advance each trajectory's transverse position by one plane:
        #   x_{i+1} = x_i + Δz × (kx/k)_i
        #
        # Δz is the spacing between consecutive imaging planes (not
        # uniform in general, because z_vals is constructed from
        # linspace over the loaded planes, but it is uniform here).
        dz = z_vals[i + 1] - z_vals[i]
        current_x = current_x + dz * kx_at_x

        # Store the updated positions for this plane.
        trajectory_points.append(current_x.copy())

    # Convert the list of 1-D arrays into a 2-D array:
    #   shape = (number_of_planes, 20)
    # Each column is a complete trajectory; each row is a snapshot at
    # one propagation distance z.
    trajectory_points = np.array(trajectory_points)

    return z_range, trajectory_points


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTION: normalize_trajectories
# ═══════════════════════════════════════════════════════════════════════════

def normalize_trajectories(z_range, trajectories, L_slit=L_SLIT_MM):
    """
    Convert dimensional trajectories (x in mm, z in mm) to dimensionless
    coordinates by dividing by the slit separation L_slit.

    Why this is necessary
    ---------------------
    The SINDy library includes trigonometric functions sin(ξ) and cos(ξ).
    For these to be physically meaningful, ξ must be dimensionless —
    you cannot compute sin(3.2 mm) in any meaningful sense.  The slit
    separation L_slit ≈ 4.68 mm is the natural length scale of the
    double-slit geometry and converts x into a dimensionless phase-like
    variable ξ = x / L_slit.

    Similarly, normalizing z → ζ = z / L_slit ensures that the discovered
    equation dξ/dζ = f(ξ) has dimensionless coefficients, making it
    directly comparable to theoretical predictions that are expressed in
    terms of dimensionless angles.

    Note on the collinearity consequence:
    With |ξ| = |x| / L_slit ≈ 1.5 / 4.68 ≈ 0.32 at the extremes,
    sin(ξ) ≈ ξ − ξ³/6 to better than 0.2% accuracy.  This means
    ξ − sin(ξ) ≈ ξ³/6, making cubic and (ξ − sin(ξ)) terms nearly
    indistinguishable.  See the collinearity note at the top of this file.

    Parameters
    ----------
    z_range : np.ndarray, shape (N,)
        Propagation distances in mm.
    trajectories : np.ndarray, shape (N, M)
        Transverse positions in mm for M trajectories at N planes.
    L_slit : float
        Slit separation in mm (default: 4.68 mm).

    Returns
    -------
    z_norm : np.ndarray, shape (N,)
        Dimensionless propagation distances ζ = z / L_slit.
    traj_norm : np.ndarray, shape (N, M)
        Dimensionless transverse positions ξ = x / L_slit.
    """
    z_norm = z_range / L_slit
    traj_norm = trajectories / L_slit
    return z_norm, traj_norm


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTION: run_sindy_poly_only
# ═══════════════════════════════════════════════════════════════════════════

def run_sindy_poly_only(z_norm, traj_norm):
    """
    Null / baseline SINDy analysis using ONLY polynomial terms up to
    degree 3:  { 1, ξ, ξ², ξ³ }.

    This serves as the null hypothesis: if a purely polynomial model
    fits the data just as well as a model with trig functions, then
    the data alone cannot justify the presence of sin/cos terms.

    Due to the collinearity issue (ξ − sin(ξ) ≈ ξ³/6 for small ξ),
    we expect this model to achieve an R² score very close to the
    full-library model.

    Parameters
    ----------
    z_norm : np.ndarray, shape (N,)
        Dimensionless propagation distances ζ = z / L_slit.
    traj_norm : np.ndarray, shape (N, M)
        Dimensionless transverse positions ξ = x / L_slit.

    Returns
    -------
    model : ps.SINDy
        Fitted polynomial-only SINDy model.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS A: Polynomial-only library (degree 3) — NULL/BASELINE")
    print("=" * 70)

    poly_library = ps.PolynomialLibrary(degree=3)

    # Same STLSQ settings as the full-library analysis for fair comparison.
    optimizer = ps.STLSQ(threshold=1e-4, alpha=0.01)

    model = ps.SINDy(
        feature_library=poly_library,
        optimizer=optimizer
    )

    x_train = [traj_norm[:, i].reshape(-1, 1) for i in range(traj_norm.shape[1])]
    model.fit(x_train, t=z_norm)

    print("\nDiscovered equation (polynomial only):")
    model.print()

    score = float(model.score(x_train, t=z_norm))
    print(f"R^2 score (polynomial only): {score:.10f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTION: run_sindy_full_library
# ═══════════════════════════════════════════════════════════════════════════

def run_sindy_full_library(z_norm, traj_norm):
    """
    Experimental SINDy analysis using the full candidate library:
    polynomial (degree 3) + sin(ξ) + cos(ξ) + 1/ξ.

    Library design rationale
    ------------------------
    The candidate function library is the union of:

    (a) A polynomial library up to degree 3  (1, ξ, ξ², ξ³).
        This captures any low-order Taylor-series behaviour.

    (b) A custom library containing sin(ξ), cos(ξ), and 1/ξ.
        These are included because the TEGR (Teleparallel Equivalent of
        General Relativity) model predicts a governing equation with a
        θ − sin(θ) structure in the non-local torsion gradient term.
        Including sin and cos lets SINDy discover this structure if it
        is present in the data.  The 1/ξ term is a generic singularity
        that often appears in radial/optical equations.

    IMPORTANT CAVEAT: At the amplitudes present in this dataset
    (|ξ| ≤ ~0.32), the terms ξ³ and (ξ − sin(ξ)) are nearly
    perfectly collinear.  If this model achieves an R² score similar
    to the polynomial-only model, the nonlinear signal is real but
    its functional form is AMBIGUOUS — the data cannot distinguish
    cubic from sinusoidal nonlinearity.

    Parameters
    ----------
    z_norm : np.ndarray, shape (N,)
        Dimensionless propagation distances ζ = z / L_slit.
    traj_norm : np.ndarray, shape (N, M)
        Dimensionless transverse positions ξ = x / L_slit.

    Returns
    -------
    model : ps.SINDy
        Fitted full-library SINDy model.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS B: Full library (poly + trig + 1/x) — EXPERIMENTAL")
    print("=" * 70)

    # (A) Polynomial library
    poly_library = ps.PolynomialLibrary(degree=3)

    # (B) Custom nonlinear library
    # NOTE: After normalization, ξ is dimensionless, so sin(ξ) and cos(ξ)
    # are now well-defined mathematical operations (not sin-of-millimetres).
    library_functions = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: 1.0 / (x + 1e-5)
    ]

    function_names = [
        lambda x: f"sin({x})",
        lambda x: f"cos({x})",
        lambda x: f"1/({x})"
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names
    )

    # (C) Combined library: { 1, ξ, ξ², ξ³, sin(ξ), cos(ξ), 1/ξ }
    feature_library = poly_library + custom_library

    # (D) Same STLSQ settings as the polynomial-only analysis.
    optimizer = ps.STLSQ(threshold=1e-4, alpha=0.01)

    # (E) Build and fit
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )

    x_train = [traj_norm[:, i].reshape(-1, 1) for i in range(traj_norm.shape[1])]
    model.fit(x_train, t=z_norm)

    print("\nDiscovered equation (full library):")
    model.print()

    score = float(model.score(x_train, t=z_norm))
    print(f"R^2 score (full library): {score:.10f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTION: main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Top-level driver that orchestrates the full analysis pipeline:

    1. Load raw CCD data from the Kocsis et al. (2011) data archive.
    2. Reconstruct average photon trajectories via weak-measurement
       momentum integration.
    3. Plot and save the trajectory figure.
    4. Normalize trajectories to dimensionless coordinates (ξ, ζ).
    5. Run TWO SINDy analyses side by side:
       (a) Polynomial-only (degree 3) — null/baseline.
       (b) Full library (poly + trig + 1/x) — experimental.
    6. Print a side-by-side comparison of both results.
    7. Save a JSON report with both sets of results and the R² delta.
    """

    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------
    # data_dir: location of the 91 raw CCD image text files.
    # The subfolder name "45-90 equiv both 0.05 15 sec" encodes the
    # experimental parameters: 45° and 90° half-wave-plate settings,
    # equivalent for both slits, 0.05 step size, 15-second exposure.
    data_dir = r"Z:\teleparallel_sim_photons\Kocsis_Data\OnlineArchive\Data\45-90 equiv both 0.05 15 sec\pics"

    # out_dir: directory where the trajectory plot and SINDy report will
    # be saved.  Created if it does not already exist.
    out_dir = r"Z:\teleparallel_sim_photons\sessionsPapers"

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 & 2: Load data and reconstruct trajectories
    # ------------------------------------------------------------------
    z_range, trajectories = load_and_reconstruct_trajectories(data_dir)

    # ------------------------------------------------------------------
    # STEP 3: Plot the reconstructed average photon trajectories
    # ------------------------------------------------------------------
    # Each of the 20 trajectories is plotted as a blue semi-transparent
    # line.  The x-axis is propagation distance z (mm), playing the role
    # of "time."  The y-axis is the transverse position x (mm).
    # The resulting figure should visually resemble Figure 3 of the
    # Kocsis et al. (2011) Science paper.
    plt.figure(figsize=(10, 6))
    for i in range(trajectories.shape[1]):
        plt.plot(z_range, trajectories[:, i], 'b-', alpha=0.5)
    plt.xlabel("Propagation distance z (mm)")
    plt.ylabel("Transverse position x (mm)")
    plt.title("Empirical Average Photon Trajectories (Kocsis 2011)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "Kocsis_Empirical_Trajectories.png"))
    plt.close()

    # ------------------------------------------------------------------
    # STEP 4: Normalize to dimensionless coordinates
    # ------------------------------------------------------------------
    # Divide x and z by L_slit = 4.68 mm (the double-slit separation).
    #
    # WHY: You cannot take sin(x) when x is in millimetres — the argument
    # to a trigonometric function must be dimensionless (radians).  The
    # slit separation is the natural length scale of the experiment,
    # making ξ = x/L_slit a dimensionless phase-like variable.
    #
    # After normalization, the seed positions span
    # ξ ∈ [−1.5/4.68, +1.5/4.68] ≈ [−0.321, +0.321].
    # At these amplitudes, sin(ξ) ≈ ξ − ξ³/6 to ~0.2% accuracy,
    # making ξ³ and (ξ − sin(ξ)) nearly indistinguishable (see
    # collinearity note at top of file).
    z_norm, traj_norm = normalize_trajectories(z_range, trajectories)

    print(f"\nNormalization: L_slit = {L_SLIT_MM} mm")
    print(f"Dimensionless x range: [{traj_norm.min():.4f}, {traj_norm.max():.4f}]")
    print(f"Dimensionless z range: [{z_norm.min():.1f}, {z_norm.max():.1f}]")

    # ------------------------------------------------------------------
    # STEP 5: Run BOTH SINDy analyses
    # ------------------------------------------------------------------

    # (A) Polynomial-only — the null/baseline test
    model_poly = run_sindy_poly_only(z_norm, traj_norm)

    # (B) Full library — the experimental test
    model_full = run_sindy_full_library(z_norm, traj_norm)

    # ------------------------------------------------------------------
    # STEP 6: Side-by-side comparison
    # ------------------------------------------------------------------
    x_train = [traj_norm[:, i].reshape(-1, 1) for i in range(traj_norm.shape[1])]

    score_poly = float(model_poly.score(x_train, t=z_norm))
    score_full = float(model_full.score(x_train, t=z_norm))
    r2_delta = score_full - score_poly

    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"\n  Polynomial-only R^2:  {score_poly:.10f}")
    print(f"  Full-library R^2:    {score_full:.10f}")
    print(f"  Delta R^2 (full - poly):  {r2_delta:+.10f}")

    if abs(r2_delta) < 0.01:
        print("\n  INTERPRETATION: Both models achieve similar R^2 scores.")
        print("  The nonlinear signal is REAL (both capture it), but its")
        print("  functional form is AMBIGUOUS -- the data cannot distinguish")
        print("  cubic nonlinearity from sinusoidal (TEGR) structure at")
        print("  these small dimensionless amplitudes (|xi| <= 0.32).")
        print("  This is the expected consequence of the collinearity")
        print("  between xi^3 and (xi - sin(xi)) for small |xi|.")
    else:
        print(f"\n  INTERPRETATION: Delta R^2 = {r2_delta:+.6f} suggests the")
        print("  full library provides a meaningfully different fit.")
        print("  However, given the small-amplitude regime, independent")
        print("  validation (e.g., data at larger |xi|) is still recommended.")

    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 7: Assemble and save the JSON report
    # ------------------------------------------------------------------
    # The report now contains BOTH analyses for honest comparison.
    report = {
        "normalization": {
            "L_slit_mm": L_SLIT_MM,
            "description": (
                "All x and z values divided by L_slit = 4.68 mm "
                "(double-slit separation: rightmean - leftmean from "
                "Kocsis MATLAB analysis). This makes the SINDy state "
                "variable dimensionless, which is required for "
                "trigonometric library terms to be physically meaningful."
            ),
            "dimensionless_x_range": [float(traj_norm.min()), float(traj_norm.max())],
            "dimensionless_z_range": [float(z_norm.min()), float(z_norm.max())]
        },
        "poly_only": {
            "description": (
                "Null/baseline: polynomial-only library (degree 3). "
                "Candidate terms: {1, xi, xi^2, xi^3}."
            ),
            "equations": [eq for eq in model_poly.equations()],
            "features": model_poly.get_feature_names(),
            "R2_score": score_poly
        },
        "full_library": {
            "description": (
                "Experimental: full library (poly degree 3 + sin + cos + 1/x). "
                "Candidate terms: {1, xi, xi^2, xi^3, sin(xi), cos(xi), 1/xi}."
            ),
            "equations": [eq for eq in model_full.equations()],
            "features": model_full.get_feature_names(),
            "R2_score": score_full
        },
        "comparison": {
            "R2_delta_full_minus_poly": r2_delta,
            "collinearity_note": (
                "For |xi| < ~0.32 (the regime of this data), "
                "xi - sin(xi) ~ xi^3/6 with correlation > 0.999999. "
                "SINDy cannot reliably distinguish cubic from sinusoidal "
                "nonlinearity at these amplitudes. The nonlinear signal "
                "is real but its functional form is ambiguous."
            )
        }
    }

    report_path = os.path.join(out_dir, "sindy_kocsis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nResults saved to {out_dir}")
    print(f"  Report: {report_path}")
    print(f"  Plot:   {os.path.join(out_dir, 'Kocsis_Empirical_Trajectories.png')}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
# Standard Python idiom: execute main() only when this script is run
# directly (not when imported as a module by another script).
if __name__ == "__main__":
    main()
