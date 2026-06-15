import pandas as pd
import numpy as np
import pysindy as ps
import warnings
warnings.filterwarnings("ignore")

print("Starting Meta-SINDy Analysis on Emergent Coefficients...")

df = pd.read_csv('Z:\\teleparallel_sim\\analysis_full.csv')

# Group by the 4 parameters in case of duplicate runs, and average the results
df_grouped = df.groupby(['pauli', 'vac', 'tor', 'cmb']).mean().reset_index()

X = df_grouped[['pauli', 'vac', 'tor', 'cmb']].values
y_cr = df_grouped[['c_r']].values
y_inv = df_grouped[['c_inv_r3']].values

# We create a polynomial library up to degree 4.
# This gives the optimizer features like P^4, P^3 * V, etc.
poly_lib = ps.PolynomialLibrary(degree=3)

# We use STLSQ (Sequential Thresholded Least Squares)
# normalize_columns=True is helpful when features have vastly different scales (like P^4 vs C)
optimizer = ps.STLSQ(threshold=10000.0, alpha=0.01, normalize_columns=True)

model_cr = ps.SINDy(feature_library=poly_lib, optimizer=optimizer)

try:
    model_cr.fit(X, t=1.0, x_dot=y_cr)
    print("\n--- MASTER EQUATION FOR c_r (Metric Expansion/Confinement Scalar) ---")
    model_cr.print()
except Exception as e:
    print("Error during fit (c_r):", e)

# Fit for c_inv_r3
optimizer_inv = ps.STLSQ(threshold=10000.0, alpha=0.01, normalize_columns=True)
model_inv = ps.SINDy(feature_library=poly_lib, optimizer=optimizer_inv)

try:
    model_inv.fit(X, t=1.0, x_dot=y_inv)
    print("\n--- MASTER EQUATION FOR c_inv_r3 (Singularity Core Repulsion) ---")
    model_inv.print()
except Exception as e:
    print("Error during fit (c_inv_r3):", e)

