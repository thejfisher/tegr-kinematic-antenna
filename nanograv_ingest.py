import numpy as np
import pandas as pd
import sys

def hellings_downs_correlation(theta):
    # theta is angular separation in radians
    if theta == 0:
        return 1.0
    
    x = (1 - np.cos(theta)) / 2.0
    if x == 0:
        return 1.0
    
    hd = 1.5 * x * np.log(x) - 0.25 * x + 0.5
    return hd

def generate_pulsar_positions(num_pulsars):
    # Generate random points on a sphere (RA, Dec)
    ra = np.random.uniform(0, 2*np.pi, num_pulsars)
    dec = np.arcsin(np.random.uniform(-1, 1, num_pulsars))
    
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack((x, y, z)).T

def generate_hd_data(num_pulsars, num_ticks, output_file):
    print(f"Generating Hellings-Downs array for {num_pulsars} pulsars over {num_ticks} ticks...")
    np.random.seed(42)
    
    positions = generate_pulsar_positions(num_pulsars)
    
    cov = np.zeros((num_pulsars, num_pulsars))
    for i in range(num_pulsars):
        for j in range(num_pulsars):
            if i == j:
                cov[i, j] = 1.0 + 0.01
            else:
                cos_theta = np.dot(positions[i], positions[j])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                cov[i, j] = hellings_downs_correlation(theta)
                
    # Normalize covariance to avoid huge numbers
    A_gwb = 1.0
    cov *= A_gwb**2
    
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("Covariance matrix is not positive definite. Adding small jitter to diagonal.")
        cov += np.eye(num_pulsars) * 0.1
        L = np.linalg.cholesky(cov)
        
    # Generate correlated time series
    Z = np.random.normal(0, 1, (num_pulsars, num_ticks))
    X = L @ Z
    X = X.T
    
    df = pd.DataFrame(X, columns=[f"Pulsar_{i}" for i in range(num_pulsars)])
    df.insert(0, "Tick", np.arange(num_ticks))
    df.to_csv(output_file, index=False)
    
    print(f"Successfully generated Hellings-Downs stochastic time series: {output_file}")
    err = np.abs(np.cov(X.T) - cov).mean()
    print(f"Empirical vs Theoretical Covariance Mean Error: {err:.4f}")

if __name__ == "__main__":
    num_pulsars = 15 # 15 anchors for the array
    num_ticks = 5000
    output_file = "nanograv_anchor_feed.csv"
    
    if len(sys.argv) > 1:
        num_pulsars = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_ticks = int(sys.argv[2])
    
    generate_hd_data(num_pulsars, num_ticks, output_file)
