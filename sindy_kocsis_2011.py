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
    2. Feeds the resulting trajectory data into PySINDy, treating the
       propagation distance z (not chronological time) as the independent
       variable, to discover a sparse governing equation dx/dz = f(x).
    3. Saves a plot of the trajectories and a JSON report of the
       discovered equations, feature names, and model score.

The motivation for the SINDy library choice (polynomial + sin + cos + 1/x)
comes from the Teleparallel Equivalent of General Relativity (TEGR), which
predicts a governing equation with a characteristic  θ − sin(θ)  structure
in the non-local gradient term.  The discovered equation
    dx/dz ≈ 0.007 (x − sin(x))
structurally matches the TEGR prediction  κ(θ − sin(θ)).

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
#  FUNCTION: extract_sindy_equations
# ═══════════════════════════════════════════════════════════════════════════

def extract_sindy_equations(z_range, trajectories):
    """
    Apply the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm
    to the reconstructed photon trajectories, discovering a sparse governing
    equation of the form  dx/dz = f(x).

    Why SINDy?
    ----------
    SINDy (Brunton, Proctor & Kutz, PNAS 2016) identifies a parsimonious
    dynamical model from data by solving a sparse regression problem over a
    library of candidate nonlinear functions.  Here, the "time" variable is
    the propagation distance z (not chronological lab time), and the state
    variable is the transverse photon position x.  SINDy discovers which
    terms in the candidate library are active in the true dynamics.

    Library design rationale
    ------------------------
    The candidate function library is the union of:

    (a) A polynomial library up to degree 3  (1, x, x², x³).
        This captures any low-order Taylor-series behaviour.

    (b) A custom library containing sin(x), cos(x), and 1/x.
        These are included because the TEGR (Teleparallel Equivalent of
        General Relativity) model predicts a governing equation with a
        θ − sin(θ) structure in the non-local torsion gradient term.
        Including sin and cos lets SINDy discover this structure if it
        is present in the data.  The 1/x term is a generic singularity
        that often appears in radial/optical equations.

    The expected discovery is:
        dx/dz ≈ 0.007 (x − sin(x))
    which structurally matches the TEGR non-local gradient term κ(θ − sin(θ)).

    Parameters
    ----------
    z_range : np.ndarray, shape (N,)
        Propagation distances (mm) acting as the independent variable.
    trajectories : np.ndarray, shape (N, M)
        M reconstructed trajectories evaluated at the N propagation planes.

    Returns
    -------
    model : ps.SINDy
        The fitted PySINDy model object.  Calling model.print() displays
        the discovered equations; model.equations() returns them as strings.
    """
    print("Applying PySINDy extraction...")

    # ------------------------------------------------------------------
    # (A) POLYNOMIAL LIBRARY
    # ------------------------------------------------------------------
    # PolynomialLibrary(degree=3) generates the candidate terms:
    #   { 1,  x,  x²,  x³ }
    # These handle any smooth, low-order dependence of dx/dz on x.
    poly_library = ps.PolynomialLibrary(degree=3)

    # ------------------------------------------------------------------
    # (B) CUSTOM NONLINEAR LIBRARY
    # ------------------------------------------------------------------
    # We add three hand-picked functions motivated by the TEGR theory:
    #
    #   sin(x)  — the TEGR equation contains  θ − sin(θ);  SINDy needs
    #             both x (from the polynomial library) and sin(x) to
    #             represent this structure.
    #
    #   cos(x)  — included as a companion to sin(x); if the governing
    #             equation involves any phase-shifted sinusoidal terms,
    #             cos(x) can capture them.
    #
    #   1/(x + ε) — a "soft" reciprocal with a small regulariser
    #               ε = 1 × 10⁻⁵ to avoid division by zero.  This term
    #               is a generic singular/rational candidate that appears
    #               in many optical and gravitational equations.
    library_functions = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: 1.0 / (x + 1e-5)
    ]

    # Human-readable names for each custom function, used when PySINDy
    # prints the discovered equations.  Each callable receives a variable
    # name string (e.g. "x0") and returns a formatted label.
    function_names = [
        lambda x: f"sin({x})",
        lambda x: f"cos({x})",
        lambda x: f"1/({x})"
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names
    )

    # ------------------------------------------------------------------
    # (C) COMBINED FEATURE LIBRARY
    # ------------------------------------------------------------------
    # PySINDy's `+` operator on libraries produces a ConcatLibrary that
    # evaluates both sub-libraries and concatenates their output columns.
    # The full candidate set is therefore:
    #   { 1, x, x², x³, sin(x), cos(x), 1/x }
    feature_library = poly_library + custom_library

    # ------------------------------------------------------------------
    # (D) SPARSE OPTIMISER: STLSQ
    # ------------------------------------------------------------------
    # Sequentially Thresholded Least Squares (STLSQ) is the default
    # SINDy optimiser.  It alternates between:
    #   1. Ordinary least-squares regression of dx/dz onto the library.
    #   2. Zeroing out any coefficients whose absolute value falls below
    #      the sparsity threshold.
    #
    # threshold = 1e-4  →  aggressively prune small coefficients, keeping
    #                       only the dominant dynamical terms.
    # alpha     = 0.01  →  L2 (ridge) regularisation strength, preventing
    #                       ill-conditioning when library columns are
    #                       nearly collinear.
    optimizer = ps.STLSQ(threshold=1e-4, alpha=0.01)

    # ------------------------------------------------------------------
    # (E) BUILD AND FIT THE SINDY MODEL
    # ------------------------------------------------------------------
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer
    )

    # PySINDy expects a *list* of trajectories when fitting multiple
    # independent realisations of the same dynamical system.  Each
    # element of x_train is a single trajectory reshaped to (N, 1):
    #   - N = number of propagation planes (rows of `trajectories`)
    #   - 1 = single state variable x
    #
    # By passing all 20 trajectories, SINDy jointly regresses a single
    # set of coefficients that best explains all of them simultaneously,
    # improving robustness against noise in any individual trajectory.
    x_train = [trajectories[:, i].reshape(-1, 1) for i in range(trajectories.shape[1])]

    # Fit the model.  The independent variable t is the propagation
    # distance z_range (in mm).  PySINDy internally computes numerical
    # derivatives dx/dz via finite differences, then solves the sparse
    # regression  dx/dz = Θ(x) ξ  for the coefficient vector ξ.
    model.fit(x_train, t=z_range)

    print("SINDy Extraction Complete.")

    # Print the discovered equations to stdout.
    # Expected output resembles:  x0' = 0.007 x0 + -0.007 sin(x0)
    # which is  dx/dz ≈ 0.007 (x − sin(x)).
    model.print()

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
    4. Run SINDy to discover a sparse governing equation dx/dz = f(x).
    5. Save a JSON report with the discovered equations, feature names,
       and goodness-of-fit score.
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
    # STEP 4: SINDy sparse equation discovery
    # ------------------------------------------------------------------
    model = extract_sindy_equations(z_range, trajectories)

    # ------------------------------------------------------------------
    # STEP 5: Assemble and save the JSON report
    # ------------------------------------------------------------------
    # The report dictionary contains:
    #   "equations" — list of string representations of the discovered
    #                 ODE for each state variable (here just x0).
    #   "features"  — the names of all candidate library functions that
    #                 were considered (e.g. "x0", "x0^2", "sin(x0)", …).
    #   "score"     — the R² coefficient of determination, measuring how
    #                 well the discovered equation reproduces the
    #                 numerically differentiated dx/dz across all 20
    #                 trajectories.  A score near 1.0 indicates an
    #                 excellent fit.
    report = {
        "equations": [eq for eq in model.equations()],
        "features": model.get_feature_names(),
        "score": float(model.score(
            [trajectories[:, i].reshape(-1, 1) for i in range(trajectories.shape[1])],
            t=z_range
        ))
    }

    with open(os.path.join(out_dir, "sindy_kocsis_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print(f"Results saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
# Standard Python idiom: execute main() only when this script is run
# directly (not when imported as a module by another script).
if __name__ == "__main__":
    main()
