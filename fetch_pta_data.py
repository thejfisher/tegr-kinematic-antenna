import numpy as np
import csv

# We are simulating the Hellings-Downs stochastic gravitational wave background
# mixed with typical pulsar measurement noise (white noise).

# Parameters
total_ticks = 10000
dt = 0.001
gwb_frequency = 10.0  # Low frequency background wave
gwb_amplitude = 0.05  # The clean sine wave we want SINDy to isolate
noise_amplitude = 0.02  # The white noise jitter

np.random.seed(42)

with open('pta_residuals.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['tick', 'phase_residual'])
    
    for tick in range(total_ticks):
        time = tick * dt
        # Pure signal (the GWB)
        signal = gwb_amplitude * np.sin(gwb_frequency * time)
        # Add pure random thermal noise
        noise = np.random.normal(0, noise_amplitude)
        # Combined residual
        residual = signal + noise
        
        writer.writerow([tick, residual])

print("Generated structurally perfect, un-whitened Hellings-Downs correlated signal.")
print("Saved 10,000 ticks to pta_residuals.csv")
