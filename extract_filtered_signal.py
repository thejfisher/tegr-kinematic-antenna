"""
Extract the filtered signal from the antenna simulation - v2.
The DERIVATIVE of the pulsar's hue (hue') is the filtered signal, not hue itself.
The pulsar rotates at a base rate, and the injected building data modulates that rate.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. Load data ---
trajectory = np.load("electron_trajectory.npy")
df = pd.read_csv("dustin_residuals.csv")
df.columns = df.columns.str.strip()
raw_residuals = df['Residuals'].values.astype(float)
timestamps = pd.to_datetime(df['Timestamp'])
n_months = len(raw_residuals)
n_ticks = trajectory.shape[0]
DT = 0.001

# --- 2. Extract hue' (derivative) for pulsar 0 ---
pulsar_hue = trajectory[:, 0, 8]
pulsar_hue_unwrapped = np.unwrap(pulsar_hue)

# Compute the instantaneous rotation rate (hue')
# Use central differences for smoother derivative
hue_dot = np.gradient(pulsar_hue_unwrapped, DT)

# --- 3. Downsample hue' to monthly values ---
# Average hue' over each month's tick range (more robust than point sampling)
monthly_hue_dot = np.zeros(n_months)
ticks_per_month = n_ticks / n_months

for i in range(n_months):
    start = int(i * ticks_per_month)
    end = int((i + 1) * ticks_per_month)
    end = min(end, n_ticks)
    monthly_hue_dot[i] = np.mean(hue_dot[start:end])

# --- 4. Remove the baseline rotation rate ---
# The pulsar has a natural rotation rate even without signal injection.
# The DEVIATION from the mean rate IS the filtered signal.
baseline_rate = np.mean(monthly_hue_dot)
filtered_signal = monthly_hue_dot - baseline_rate

# --- 5. Scale the filtered signal to match residual units ---
# Use a correlation-preserving linear rescale
signal_std = np.std(filtered_signal) + 1e-12
residual_std = np.std(raw_residuals)
scaled_signal = filtered_signal * (residual_std / signal_std)

# Shift to match residual mean
scaled_signal = scaled_signal - np.mean(scaled_signal) + np.mean(raw_residuals)

# --- 6. Compute the improvement ---
noise_remaining = raw_residuals - scaled_signal
corr = np.corrcoef(raw_residuals, scaled_signal)[0, 1]
variance_explained = 1 - np.var(noise_remaining) / np.var(raw_residuals)

# --- 7. Print results ---
print(f"{'='*60}")
print(f"ANTENNA FILTER RESULTS (v2 - Derivative Method)")
print(f"{'='*60}")
print(f"Baseline pulsar rotation rate: {baseline_rate:.4f} rad/s")
print(f"Signal deviation std:          {np.std(filtered_signal):.4f} rad/s")
print(f"")
print(f"Raw residual std:      {np.std(raw_residuals):>10.1f} kWh")
print(f"Filtered signal std:   {np.std(scaled_signal):>10.1f} kWh")
print(f"Remaining noise std:   {np.std(noise_remaining):>10.1f} kWh")
print(f"Correlation (r):       {corr:>10.4f}")
print(f"Variance explained:    {variance_explained:>10.1%}")
print(f"{'='*60}")

# --- 8. Save the deliverable CSV ---
output = pd.DataFrame({
    'Timestamp': timestamps,
    'Raw_Residual_kWh': raw_residuals,
    'Filtered_Signal_kWh': np.round(scaled_signal, 1),
    'Noise_Removed_kWh': np.round(noise_remaining, 1),
    'Pulsar_Hue_Rate': np.round(monthly_hue_dot, 6),
    'Corrected_Energy': df['Corrected Energy'].values,
    'Gross_Expected_Energy': df['Gross Expected Energy'].values
})
output.to_csv("dustin_filtered_signal.csv", index=False)
print(f"\nSaved: dustin_filtered_signal.csv")

# --- 9. Plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Physics-Informed Signal Filter: Building Energy Residuals\n"
             "(Teleparallel Antenna + Kuramoto Phase Coupling)", 
             fontsize=13, fontweight='bold')

# Panel 1: Raw residuals
colors = ['#cc4444' if r > 0 else '#4444cc' for r in raw_residuals]
axes[0].bar(timestamps, raw_residuals, width=25, color=colors, alpha=0.7)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('kWh')
axes[0].set_title('Raw XGBoost Residuals (Prediction Error)')
axes[0].text(0.02, 0.95, f'Std: {np.std(raw_residuals):.0f} kWh', 
             transform=axes[0].transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Overlay
axes[1].bar(timestamps, raw_residuals, width=25, color='#cccccc', alpha=0.5, label='Raw Residuals')
axes[1].plot(timestamps, scaled_signal, color='#2288ff', linewidth=2.5, 
             label=f'Filtered Signal (r={corr:.3f})', zorder=5)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel('kWh')
axes[1].set_title('Kuramoto-Filtered Seasonal Harmonic')
axes[1].legend(loc='upper left')

# Panel 3: Remaining noise
colors2 = ['#44aa44' if abs(n) < np.std(noise_remaining) else '#ff8800' for n in noise_remaining]
axes[2].bar(timestamps, noise_remaining, width=25, color=colors2, alpha=0.7)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_ylabel('kWh')
axes[2].set_title('Remaining Noise (Operational Anomalies for XGBoost)')
axes[2].text(0.02, 0.95, f'Std: {np.std(noise_remaining):.0f} kWh', 
             transform=axes[2].transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("dustin_filter_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dustin_filter_comparison.png")
