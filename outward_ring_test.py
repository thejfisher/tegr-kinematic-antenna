import torch
import math
import numpy as np
import sys

class TEGR_Cosmology_Engine:
    def __init__(self, N=10, dt=0.01, speed_cap=0.99):
        self.N = N
        self.dt = dt
        self.speed_cap = speed_cap
        self.state = torch.zeros((N, 10))
        # state: [t, x, y, z, p_x, p_y, p_z, m_0, theta_hue, gamma]

    def setup_outward_ring(self, radius, momentum_mag, m0=1.0):
        """
        Set up N particles in a 2D ring (xy-plane) facing radially outwards.
        """
        self.state = torch.zeros((self.N, 10))
        for i in range(self.N):
            angle = 2.0 * math.pi * i / self.N
            # Position
            self.state[i, 1] = radius * math.cos(angle)
            self.state[i, 2] = radius * math.sin(angle)
            self.state[i, 3] = 0.0
            
            # Outward Momentum
            self.state[i, 4] = momentum_mag * math.cos(angle)
            self.state[i, 5] = momentum_mag * math.sin(angle)
            self.state[i, 6] = 0.0
            
            # Mass and gamma
            self.state[i, 7] = m0
            # Calculate initial gamma
            p_mag = momentum_mag
            v_mag = p_mag / math.sqrt(m0**2 + p_mag**2)
            self.state[i, 9] = 1.0 / math.sqrt(1.0 - v_mag**2)
            
    def apply_kinematics(self, GM, pauli_k):
        self.state[:, 0] += self.dt
        F_total = torch.zeros((self.N, 3))
        
        # Calculate pairwise forces (N^2)
        for i in range(self.N):
            f_i = torch.zeros(3)
            for j in range(self.N):
                if i == j: continue
                r_vec = self.state[i, 1:4] - self.state[j, 1:4]
                r_mag = torch.norm(r_vec)
                
                if r_mag > 0.01:
                    # 1. Inward Gravitational Force (1/r^2)
                    F_grav = - (GM * self.state[i, 7] * self.state[j, 7] / (r_mag**2)) * (r_vec / r_mag)
                    
                    # 2. Outward Pauli Exclusion Pressure (1/r^3 to mimic KK signature)
                    F_pauli = (pauli_k * self.state[i, 7] * self.state[j, 7] / (r_mag**3)) * (r_vec / r_mag)
                    
                    f_i += F_grav + F_pauli
            F_total[i] = f_i

        # Momentum-Velocity Sync (Symplectic integration step)
        p_temp = self.state[:, 4:7] + F_total * self.dt
        v_temp = p_temp / torch.sqrt(
            self.state[:, 7].unsqueeze(1)**2 +
            torch.norm(p_temp, dim=1, keepdim=True)**2
        )
        
        v_norms = torch.norm(v_temp, dim=1, keepdim=True)
        v_capped = torch.where(
            v_norms >= self.speed_cap,
            (v_temp / v_norms) * self.speed_cap,
            v_temp
        )
        
        self.state[:, 9] = 1.0 / torch.sqrt(1.0 - torch.norm(v_capped, dim=1)**2)
        self.state[:, 4:7] = self.state[:, 9].unsqueeze(1) * self.state[:, 7].unsqueeze(1) * v_capped
        self.state[:, 1:4] += v_capped * self.dt


def run_outward_ring_test():
    print("============================================================")
    print("TEST: OUTWARD-FACING RING (EXPLOSION VS COLLAPSE)")
    print("============================================================")
    
    # We will test two scenarios: one with weak gravity (explosion wins)
    # and one with strong gravity (black hole / collapse wins)
    
    scenarios = [
        {"name": "Weak Gravity (Supernova Explosion)", "GM": 0.5, "p_out": 20.0},
        {"name": "Strong Gravity (Black Hole Collapse)", "GM": 5000.0, "p_out": 20.0}
    ]
    
    radius = 5.0
    N = 20
    dt = 0.01
    pauli_k = 100.0
    
    for s in scenarios:
        print(f"\nScenario: {s['name']}")
        print(f"Initial Radius: {radius} | GM: {s['GM']} | Outward Momentum: {s['p_out']}")
        
        engine = TEGR_Cosmology_Engine(N=N, dt=dt)
        engine.setup_outward_ring(radius, s['p_out'])
        
        initial_mean_radius = torch.mean(torch.norm(engine.state[:, 1:4], dim=1)).item()
        
        for tick in range(1, 1001): # 10 seconds simulation
            engine.apply_kinematics(s['GM'], pauli_k)
            
            if tick % 500 == 0:
                current_radius = torch.mean(torch.norm(engine.state[:, 1:4], dim=1)).item()
                print(f"  Tick {tick:4d} | Mean Radius: {current_radius:.3f}")
                
        final_radius = torch.mean(torch.norm(engine.state[:, 1:4], dim=1)).item()
        if final_radius > radius * 2:
            print("  Result: EXPLOSION (Outward pressure / momentum won)")
        elif final_radius < radius:
            print("  Result: COLLAPSE (Gravity overpowered the explosion)")
        else:
            print("  Result: STABLE ORBIT / RING (Forces balanced)")
            
if __name__ == "__main__":
    run_outward_ring_test()
