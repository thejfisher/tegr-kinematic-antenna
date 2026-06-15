import torch
import math

class TEGR_Cosmology_Engine:
    def __init__(self, N=10, dt=0.01, speed_cap=0.99):
        self.N = N
        self.dt = dt
        self.speed_cap = speed_cap
        self.state = torch.zeros((N, 10))
        self.vorticity = torch.zeros((N, 3))

    def setup_particles(self, radii):
        """
        Set up N particles at given radii with zero initial velocity 
        to isolate the radial forces (Gravity vs Anti-Pressure).
        """
        self.state = torch.zeros((self.N, 10))
        self.state[:, 1] = torch.tensor(radii)  # x positions = radii
        self.state[:, 7] = 0.511                # m0
        self.state[:, 9] = 1.0                  # gamma

    def apply_kinematics(self, GM, Lambda):
        self.state[:, 0] += self.dt
        F_total = torch.zeros((self.N, 3))
        bh_pos = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(self.N):
            r_vec = self.state[i, 1:4] - bh_pos
            r_mag = torch.norm(r_vec)
            
            if r_mag > 0.05:
                # 1. Inward Gravitational Force (Pressure Sink)
                # F_grav = - GM * m0 / r^2
                F_grav = - (GM * self.state[i, 7] / (r_mag**2)) * (r_vec / r_mag)
                
                # 2. Outward Vacuum Anti-Pressure (Cosmological Constant)
                # F_lambda = (1/3) * Lambda * m0 * r
                # This naturally emerges from integrating P_Lambda over the volumetric space
                F_lambda = (1.0 / 3.0) * Lambda * self.state[i, 7] * r_vec
                
                F_total[i] = F_grav + F_lambda

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


def test_anti_pressure_shatter_point():
    print("============================================================")
    print("TEST: VACUUM ANTI-PRESSURE & THE STRUCTURAL SHATTER POINT")
    print("============================================================")
    
    # Reduced dimensionless units for compute stability
    GM = 10.0
    Lambda = 0.001 
    
    # The volumetric tipping point (Shatter Point) occurs where F_grav = F_lambda:
    # GM / r^2 = (1/3) * Lambda * r  =>  r_crit = (3 * GM / Lambda)^(1/3)
    r_crit = (3.0 * GM / Lambda)**(1/3)
    
    print(f"Gravitational Mass (GM): {GM}")
    print(f"Vacuum Anti-Pressure (Lambda): {Lambda}")
    print(f"Theoretical Shatter Point (r_crit): {r_crit:.2f}")
    print("-" * 60)
    
    # We will place particles slightly inside, exactly at, and outside the critical radius
    # to watch the structural limit emerge dynamically.
    test_radii = [
        r_crit * 0.5,   # Deep inside gravity well
        r_crit * 0.9,   # Near edge but inside
        r_crit,         # Exactly on the shatter point
        r_crit * 1.1,   # Just outside
        r_crit * 1.5    # Deep in expansion territory
    ]
    
    engine = TEGR_Cosmology_Engine(N=len(test_radii), dt=0.05)
    engine.setup_particles(test_radii)
    
    for tick in range(1, 1001):
        engine.apply_kinematics(GM, Lambda)
        
    print(f"{'Initial Radius':<15} | {'Final Radius (Tick 1000)':<25} | {'State'}")
    print("-" * 60)
    
    for i in range(len(test_radii)):
        r_init = test_radii[i]
        r_final = torch.norm(engine.state[i, 1:4]).item()
        
        state_str = ""
        if r_final < r_init * 0.9:
            state_str = "COLLAPSING (Gravity Dominated)"
        elif r_final > r_init * 1.1:
            state_str = "SHATTERED (Anti-Pressure Dominated)"
        else:
            state_str = "EQUILIBRIUM (Tension Balanced)"
            
        print(f"{r_init:<15.2f} | {r_final:<25.2f} | {state_str}")

if __name__ == "__main__":
    test_anti_pressure_shatter_point()
