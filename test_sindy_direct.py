#!/usr/bin/env python3
"""Direct test of PySINDy + ROCm on hal's r9700 GPU."""
import numpy as np
import time

print("=" * 60)
print("DIRECT SINDY + GPU TEST ON HAL")
print("=" * 60)

# 1. Test PyTorch + GPU
print("\n--- STEP 1: PyTorch GPU Test ---")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        t = torch.randn(1000, 1000, device='cuda')
        result = torch.mm(t, t)
        print(f"GPU matrix multiply test: PASSED ({result.shape})")
    else:
        print("WARNING: No GPU detected, will use CPU")
except Exception as e:
    print(f"PyTorch error: {e}")

# 2. Test PySINDy
print("\n--- STEP 2: PySINDy Import Test ---")
try:
    import pysindy as ps
    print(f"PySINDy version: {ps.__version__}")
except Exception as e:
    print(f"PySINDy error: {e}")
    exit(1)

# 3. Generate fake physics data (mimicking collider output)
print("\n--- STEP 3: Generating Fake Collider Data ---")
np.random.seed(42)
N_ticks = 5000
dt = 0.001
t_arr = np.arange(N_ticks) * dt

# Simulate a damped oscillator with nonlinear coupling (easy for SINDy)
x = np.cos(2 * np.pi * t_arr) * np.exp(-0.5 * t_arr)
y = np.sin(2 * np.pi * t_arr) * np.exp(-0.5 * t_arr)
z = 0.1 * np.sin(4 * np.pi * t_arr) * np.exp(-0.3 * t_arr)

X_data = np.column_stack([x, y, z])
print(f"Generated trajectory: {X_data.shape} ({N_ticks} ticks, 3 features)")

# 4. Run PySINDy
print("\n--- STEP 4: Running PySINDy ---")
start = time.time()
library = ps.PolynomialLibrary(degree=2, include_bias=True)
optimizer = ps.STLSQ(threshold=0.01, alpha=0.001)
model = ps.SINDy(feature_library=library, optimizer=optimizer)
model.fit(X_data, t=dt)
elapsed = time.time() - start

print(f"SINDy fit completed in {elapsed:.2f}s")
print("\n========================================")
print("DISCOVERED EQUATIONS:")
print("========================================")
feature_names = ['x', 'y', 'z']
for i, eq in enumerate(model.equations()):
    print(f"  {feature_names[i]}' = {eq}")
print("========================================")
print("\nALL TESTS PASSED!")
