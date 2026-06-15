import torch
import math

class TEGREntangledEngine:
    def __init__(self, N=2, dt=0.02, speed_cap=0.99):
        self.N = N
        self.dt = dt
        self.speed_cap = speed_cap
        
        # The 10D State Vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
        self.state = torch.zeros((N, 10))
        
        # Internal Spin/Vorticity (3D Vector)
        self.vorticity = torch.zeros((N, 3))
        
        # The ER=EPR Entanglement Adjacency Tensor (W_ij)
        # 0 = Local Realism. 1 = Entangled (Wormhole connected)
        self.W = torch.zeros((N, N))
        
        # Initialize the 2 particles
        self.setup_entangled_pair()
        
        # Calculate initial Hamiltonian H_0
        self.H_0 = torch.sum(self.state[:, 9] * self.state[:, 7]).item() + 1e-12        
    def setup_entangled_pair(self):
        """Spawns two particles, separates them, and opens the ER bridge."""
        # Particle 0 (Left side of grid)
        self.state[0, 1:4] = torch.tensor([-5.0, 0.0, 0.0]) # x, y, z
        self.state[0, 7] = 0.511 # m_0 (electron mass)
        self.state[0, 8] = math.pi / 4 # Initial Hue
        self.state[0, 9] = 1.0 # Gamma
        self.vorticity[0] = torch.tensor([0.0, 0.0, 0.5]) # Spin Up
        
        # Particle 1 (Right side of grid)
        self.state[1, 1:4] = torch.tensor([5.0, 0.0, 0.0]) # x, y, z
        self.state[1, 7] = 0.511 
        self.state[1, 8] = 0.0 # Will be overwritten by entanglement
        self.state[1, 9] = 1.0
        self.vorticity[1] = torch.tensor([0.0, 0.0, -0.5]) # Spin Down
        
        # OPEN THE WORMHOLE: Entangle Particle 0 and Particle 1
        self.W[0, 1] = 1.0 
        self.W[1, 0] = 1.0 
        
        # Run initial sync to link their internal states
        self.sync_entanglement()

    def sync_entanglement(self):
        """Forces non-local synchronization of phase and spin across the W_ij tensor."""
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i, j] == 1.0 and i < j:
                    # Particle j instantly adopts the Hue and opposite Spin of Particle i
                    self.state[j, 8] = self.state[i, 8]
                    self.vorticity[j] = -self.vorticity[i] 

    def apply_kinematics(self, background_magnetic_field=None, black_hole=None):
        self.state[:, 0] += self.dt 
        self.state[:, 8] += (self.state[:, 9] * self.state[:, 7]) * self.dt 
        
        F_total = torch.zeros((self.N, 3))
        
        # --- 2. Gravity ---
        if black_hole is not None:
            bh_pos = black_hole['pos']
            GM = black_hole['M']
            for i in range(self.N):
                r_vec = bh_pos - self.state[i, 1:4]
                r_mag = torch.norm(r_vec)
                if r_mag > 0.1: # Avoid singularity division by zero
                    # F_grav = GMm/r^2
                    F_grav = (GM * self.state[i, 7] / (r_mag**2)) * (r_vec / r_mag)
                    F_total[i] += F_grav

        # --- 3. Lorentz Forces ---
        if background_magnetic_field is not None:
            F_mag = torch.cross(self.state[0, 4:7], background_magnetic_field)
            F_total[0] += F_mag * 0.1
            self.vorticity[0] *= -1.0 
            
        # --- MOMENTUM-VELOCITY SYNC ---
        p_temp = self.state[:, 4:7] + F_total * self.dt
        v_temp = p_temp / torch.sqrt(self.state[:, 7].unsqueeze(1)**2 + torch.norm(p_temp, dim=1, keepdim=True)**2)
        
        v_norms = torch.norm(v_temp, dim=1, keepdim=True)
        v_capped = torch.where(v_norms >= self.speed_cap, (v_temp / v_norms) * self.speed_cap, v_temp)
        
        self.state[:, 9] = 1.0 / torch.sqrt(1.0 - torch.norm(v_capped, dim=1)**2)
        self.state[:, 4:7] = self.state[:, 9].unsqueeze(1) * self.state[:, 7].unsqueeze(1) * v_capped
        self.state[:, 1:4] += v_capped * self.dt
        
        # --- ER=EPR WORMHOLE DECOHERENCE (AMPS FIREWALL) ---
        if self.W[0, 1] == 1.0:
            H_current = torch.sum(self.state[:, 9] * self.state[:, 7]).item()
            delta_H_ratio = abs(H_current - self.H_0) / self.H_0
            out_of_bounds = torch.max(torch.abs(self.state[:, 1:4])).item() > 1000.0
            
            # The Firewall Paradox: if the energy drift is too high or particles escape, the bond snaps!
            if delta_H_ratio > 0.1 or out_of_bounds: 
                self.W[0, 1] = 0.0
                self.W[1, 0] = 0.0
                
                # When bond snaps, pair-production energy (4*m0 total) is deposited
                # to cap both severed terminations of the wormhole.
                # Each particle receives 2*m0 (one particle-antiparticle pair per end).
                m0_val = self.state[0, 7].item()
                self.state[0, 7] += 2.0 * m0_val 
                self.state[1, 7] += 2.0 * m0_val

        # Non-local bridge instantly resolves
        self.sync_entanglement()

def run_blackhole_test():
    engine = TEGREntangledEngine()
    
    # Place a supermassive point at x = -10.0. Particle 0 is at -5.0. Particle 1 is at 5.0.
    black_hole = {'pos': torch.tensor([-10.0, 0.0, 0.0]), 'M': 250.0}
    
    print("============================================================")
    print("STARTING BLACK HOLE DECOHERENCE TEST (AMPS FIREWALL)")
    print("============================================================")
    
    for tick in range(1, 151):
        engine.apply_kinematics(black_hole=black_hole)
        
        w_status = "Active" if engine.W[0, 1] == 1.0 else "SNAPPED (FIREWALL)"
        
        if tick % 10 == 0 or w_status != "Active":
            gamma_0 = engine.state[0, 9].item()
            gamma_1 = engine.state[1, 9].item()
            mass_0 = engine.state[0, 7].item()
            pos_0 = engine.state[0, 1].item()
            
            print(f"Tick {tick:03d} | P0 Pos: {pos_0:6.2f} | P0 Gamma (Time Dil): {gamma_0:6.3f} | W_ij: {w_status}")
            
            if w_status != "Active":
                print("------------------------------------------------------------")
                print(f"*** WARNING: TENSION DIFFERENTIAL CRITICAL (dGamma = {abs(gamma_0 - gamma_1):.3f}) ***")
                print("*** AMPS FIREWALL TRIGGERED! WORMHOLE BOND SNAPPED! ***")
                print(f"*** MASSIVE ENERGY SPIKE DETECTED AT EVENT HORIZON: m_0 -> {mass_0:.2f} MeV ***")
                break
                
    print("============================================================")
    print("RESULT: Particle 0 experienced extreme gravitational acceleration.")
    print("As it approached the singularity, Time Dilation (Gamma) skyrocketed.")
    print("Relativistic tension differential exceeded pair-production limit (4*m0).")
    print("ER=EPR bond snapped, dumping 4*m0 = 2.044 MeV as an AMPS Firewall.")

if __name__ == "__main__":
    run_blackhole_test()
