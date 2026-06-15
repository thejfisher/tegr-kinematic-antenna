import torch
import csv

class TEGRMacroEngine:
    def __init__(self, N=10, dt=0.01):
        self.N = N
        self.dt = dt
        self.state = torch.zeros((N, 10))
        # Default mass for particles in dimensionless units
        self.state[:, 7] = 0.511
        self.state[:, 9] = 1.0

    def apply_kinematics(self, GM, Lambda):
        self.state[:, 0] += self.dt
        F_total = torch.zeros((self.N, 3))
        bh_pos = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(self.N):
            r_vec = self.state[i, 1:4] - bh_pos
            r_mag = torch.norm(r_vec)
            if r_mag > 0.05:
                # Inward Gravity
                F_grav = - (GM * self.state[i, 7] / (r_mag**2)) * (r_vec / r_mag)
                # Outward Anti-Pressure
                F_lambda = (1.0 / 3.0) * Lambda * self.state[i, 7] * r_vec
                F_total[i] = F_grav + F_lambda

        # Simple Euler step for momentum and velocity (omitting relativistic cap for speed in this demo)
        self.state[:, 4:7] += F_total * self.dt
        v_new = self.state[:, 4:7] / self.state[:, 7].unsqueeze(1)
        self.state[:, 1:4] += v_new * self.dt

def run_scaled_workflow():
    print("============================================================")
    print("1. RUNNING DIMENSIONLESS SIMULATION (COMPUTE KERNEL)")
    print("============================================================")
    
    # We choose dimensionless values optimized for the GPU (0.001 to 100.0)
    GM = 10.0
    Lambda = 0.001
    
    # Set up 3 test galaxies at different radii
    r_crit = (3.0 * GM / Lambda)**(1/3) # ~31.07
    test_radii = [15.0, 31.07, 60.0]
    
    engine = TEGRMacroEngine(N=len(test_radii), dt=0.5)
    engine.state[:, 1] = torch.tensor(test_radii)
    
    # Save the dimensionless data
    dim_csv_path = "z:/TEGR_sim/dimensionless_output.csv"
    with open(dim_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tick', 'particle_id', 'x', 'y', 'z', 'mass'])
        
        for tick in range(11):
            if tick > 0:
                engine.apply_kinematics(GM, Lambda)
                
            for i in range(len(test_radii)):
                writer.writerow([
                    tick, i, 
                    engine.state[i, 1].item(), 
                    engine.state[i, 2].item(), 
                    engine.state[i, 3].item(), 
                    engine.state[i, 7].item()
                ])
                
    print(f"Simulation complete. Saved pure tensor data to {dim_csv_path}")
    
    print("\n============================================================")
    print("2. POST-PROCESSING: SCALING TO SI UNITS (GALACTIC SCALE)")
    print("============================================================")
    
    # Define our Geometrized Base Units (c=1, G=1)
    # Let 1 dimensionless unit of Length = 1 Mega-parsec (Mpc)
    L_0 = 3.086e22  # meters (1 Mpc)
    
    # c = 2.998e8 m/s
    c_SI = 2.9979e8
    G_SI = 6.6743e-11
    
    # Time unit T_0 = L_0 / c
    T_0 = L_0 / c_SI  # ~1.029e14 seconds (approx 3.26 million years)
    
    # Mass unit M_0 = L_0 * c^2 / G
    M_0 = L_0 * (c_SI**2) / G_SI  # ~4.15e49 kg (approx 20 billion Milky Way galaxies)
    
    print("Base Scaling Factors Derived:")
    print(f" 1 Unit of Length = {L_0:.2e} meters (1 Mpc)")
    print(f" 1 Unit of Time   = {T_0:.2e} seconds (~3.26 Million Years)")
    print(f" 1 Unit of Mass   = {M_0:.2e} kg")
    print("-" * 60)
    
    # Read the dimensionless CSV and write an SI-converted CSV
    si_csv_path = "z:/TEGR_sim/si_scaled_output.csv"
    with open(dim_csv_path, 'r') as infile, open(si_csv_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write SI headers
        writer.writerow(['time_seconds', 'particle_id', 'x_meters', 'y_meters', 'z_meters', 'mass_kg'])
        
        for row in reader:
            tick_val = float(row['tick'])
            # We used a dt of 0.5 dimensionless time units per tick
            time_dim = tick_val * 0.5 
            
            # Apply scaling multiplications
            time_si = time_dim * T_0
            x_si = float(row['x']) * L_0
            y_si = float(row['y']) * L_0
            z_si = float(row['z']) * L_0
            mass_si = float(row['mass']) * M_0
            
            writer.writerow([time_si, row['particle_id'], x_si, y_si, z_si, mass_si])
            
            if tick_val == 10.0:
                print(f"Particle {row['particle_id']} at Tick 10:")
                print(f"  X-Position: {x_si:.4e} meters")
                print(f"  Mass:       {mass_si:.4e} kg")
                
    print(f"\nScaling complete. Ready for real-world visualization: {si_csv_path}")

if __name__ == "__main__":
    run_scaled_workflow()
