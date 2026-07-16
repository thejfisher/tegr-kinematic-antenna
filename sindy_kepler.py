import numpy as np
import pysindy as ps
import warnings
warnings.filterwarnings("ignore")

history = np.load("electron_trajectory.npy")
tracker_idx = 0 # Track the moving particle (index 0)

print(f"Tracking Particle {tracker_idx}")

x = history[:, tracker_idx, 1]
y = history[:, tracker_idx, 2]
z = history[:, tracker_idx, 3]
r = np.sqrt(x**2 + y**2 + z**2 + 1e-12)
hue = np.unwrap(history[:, tracker_idx, 8])
gamma = history[:, tracker_idx, 9]
m0 = history[:, tracker_idx, 7]
px = history[:, tracker_idx, 4]
py = history[:, tracker_idx, 5]
pz = history[:, tracker_idx, 6]

vx = px / (gamma * m0)
vy = py / (gamma * m0)
vz = pz / (gamma * m0)
r_dot = (x * vx + y * vy + z * vz) / (r + 1e-6)

inv_r3 = np.clip(1.0 / (r**3 + 1e-12), -1e6, 1e6)
m0_over_gamma = m0 / (gamma + 1e-12)
sin_hue = np.sin(hue)
cos_hue = np.cos(hue)

X_features = np.column_stack((x, y, z, r, hue, gamma, m0, inv_r3, sin_hue, cos_hue, m0_over_gamma))
feature_names = ['x', 'y', 'z', 'r', 'hue', 'gamma', 'm0', '1/r^3', 'sin(hue)', 'cos(hue)', 'm0/gamma']

other_idx = 1
other_pos = history[:, other_idx, 1:4]
tracker_pos_2b = history[:, tracker_idx, 1:4]
other_hue = history[:, other_idx, 8]
sep_vec = tracker_pos_2b - other_pos
d12 = np.sqrt(sep_vec[:, 0]**2 + sep_vec[:, 1]**2 + sep_vec[:, 2]**2 + 1e-12)
inv_d12_2 = np.clip(1.0 / (d12**2 + 1e-12), -1e6, 1e6)
inv_d12_3 = np.clip(1.0 / (d12**3 + 1e-12), -1e6, 1e6)
d_hue = np.unwrap(history[:, tracker_idx, 8]) - np.unwrap(other_hue)
cos_dhue = np.cos(d_hue)

X_features = np.column_stack((X_features, d12, inv_d12_2, inv_d12_3, cos_dhue))
feature_names += ['d12', '1/d12^2', '1/d12^3', 'cos(d_hue)']

dt = 0.005 # Match the dt of the physics run
fd = ps.FiniteDifference()

internal_vars = np.column_stack((hue, gamma, m0))
internal_dots = fd._differentiate(internal_vars, t=dt)

X_ddots = fd._differentiate(np.column_stack((vx, vy, vz, r_dot)), t=dt)
X_targets = np.column_stack((X_ddots, internal_dots))

# Zero variance filter to prevent SVD convergence errors when normalizing
variances = np.var(X_features, axis=0)
valid_feat_idx = np.where(variances > 1e-12)[0]
X_features_filtered = X_features[:, valid_feat_idx]
feature_names_filtered = [feature_names[i] for i in valid_feat_idx]

print(f"Removed low-variance features: {[feature_names[i] for i in range(len(feature_names)) if i not in valid_feat_idx]}")

generalized_library = ps.PolynomialLibrary(degree=1, include_bias=True)
optimizer = ps.STLSQ(threshold=0.005, alpha=0.001, normalize_columns=True)
model = ps.SINDy(feature_library=generalized_library, optimizer=optimizer)

valid_mask = np.isfinite(X_features_filtered).all(axis=1) & np.isfinite(X_targets).all(axis=1)
X_features_filtered = X_features_filtered[valid_mask]
X_targets = X_targets[valid_mask]

model.fit(X_features_filtered, t=dt, x_dot=X_targets, feature_names=feature_names_filtered)
model.print()

from sklearn.metrics import r2_score
predictions = model.predict(X_features_filtered)
r2_per_var = r2_score(X_targets, predictions, multioutput='raw_values')
overall_r2 = model.score(X_features_filtered, t=dt, x_dot=X_targets)

lhs = ["vx'", "vy'", "vz'", "r''", "hue'", "gamma'", "m0'"]
print("\n--- FORMATTED EQUATIONS ---")
for i, eq in enumerate(model.equations()):
    print(f"{lhs[i]} = {eq}   (R^2 = {r2_per_var[i]:.4f})")
print(f"Overall R^2 Score: {overall_r2:.6f}")
