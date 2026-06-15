import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class TEGRCollisionGPU:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing 10D TEGR Tensor Kernel on: {self.device}")
        
        # Simulation Parameters
        self.G = 1.0
        self.c = 1.0 # Speed of light limit
        self.Lambda = 0.0001
        self.dt = 0.05
        self.softening = 0.5
        
        self.N_MW = 10000
        self.M_MW_core = 2000.0
        
        self.N_GE = 2500
        self.M_GE_core = 500.0
        
        self.N_total = self.N_MW + self.N_GE + 2
        
        # 10D TEGR State Vector: [t, x, y, z, px, py, pz, m0, hue, gamma]
        self.state = torch.zeros((self.N_total, 10), device=self.device)
        
        self._initialize_galaxies()

    def _initialize_galaxies(self):
        # 1. Milky Way Core
        self.state[0, 1:4] = torch.tensor([0.0, 0.0, 0.0])
        self.state[0, 7] = self.M_MW_core # m0
        
        # 2. Gaia-Enceladus Core
        self.state[1, 1:4] = torch.tensor([120.0, 60.0, 20.0])
        self.state[1, 4:7] = torch.tensor([-3.0, -1.0, -0.5]) * self.M_GE_core # Momentum = v * m0
        self.state[1, 7] = self.M_GE_core
        
        # 3. Milky Way Stars
        mw_start = 2
        mw_end = mw_start + self.N_MW
        self._init_disk(mw_start, mw_end, center=self.state[0, 1:4], core_mass=self.M_MW_core, 
                        r_min=3.0, r_max=40.0, core_p=self.state[0, 4:7], tilt_x=0.0, tilt_y=0.0)
                        
        # 4. Gaia-Enceladus Stars
        ge_start = mw_end
        ge_end = self.N_total
        self._init_disk(ge_start, ge_end, center=self.state[1, 1:4], core_mass=self.M_GE_core, 
                        r_min=2.0, r_max=15.0, core_p=self.state[1, 4:7] / self.M_GE_core * 0.01, tilt_x=0.5, tilt_y=0.8)
                        
        # Initialize gamma for all particles
        self._update_gamma()

    def _init_disk(self, start_idx, end_idx, center, core_mass, r_min, r_max, core_p, tilt_x, tilt_y):
        N_stars = end_idx - start_idx
        star_mass = 0.01
        
        u = torch.rand(N_stars, device=self.device)
        radii = r_min + (r_max - r_min) * torch.sqrt(u)
        angles = torch.rand(N_stars, device=self.device) * 2 * math.pi
        
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        z = torch.zeros(N_stars, device=self.device) + (torch.randn(N_stars, device=self.device) * 0.5)
        
        v_mag = torch.sqrt(self.G * core_mass / radii)
        vx = -v_mag * torch.sin(angles)
        vy = v_mag * torch.cos(angles)
        vz = torch.zeros(N_stars, device=self.device)
        
        pos_disk = torch.stack((x, y, z), dim=1)
        vel_disk = torch.stack((vx, vy, vz), dim=1)
        
        rot_x = torch.tensor([[1, 0, 0], [0, math.cos(tilt_x), -math.sin(tilt_x)], [0, math.sin(tilt_x), math.cos(tilt_x)]], device=self.device)
        rot_y = torch.tensor([[math.cos(tilt_y), 0, math.sin(tilt_y)], [0, 1, 0], [-math.sin(tilt_y), 0, math.cos(tilt_y)]], device=self.device)
        
        pos_disk = pos_disk @ rot_x @ rot_y
        vel_disk = vel_disk @ rot_x @ rot_y
        
        self.state[start_idx:end_idx, 1:4] = pos_disk + center
        
        # Assign m0
        self.state[start_idx:end_idx, 7] = star_mass
        
        # Assign momentum: p = m0 * v (approximate classical start, gamma will correct it)
        core_v = core_p / (core_mass + 1e-9)
        self.state[start_idx:end_idx, 4:7] = (vel_disk + core_v) * star_mass

    def _update_gamma(self):
        # Relativistic Kinematics: gamma = sqrt(1 + (p / (m0 * c))^2)
        p = self.state[:, 4:7]
        p_mag_sq = (p**2).sum(dim=1)
        m0 = self.state[:, 7]
        self.state[:, 9] = torch.sqrt(1.0 + p_mag_sq / ((m0 * self.c)**2))

    def step(self):
        # 1. Update internal time tensor
        self.state[:, 0] += self.dt
        
        pos = self.state[:, 1:4]
        m0 = self.state[:, 7]
        
        # 2. Torsion Gradient Field (O(N^2) Tensor Broadcast)
        delta = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist_sq = (delta**2).sum(dim=2) + self.softening**2
        dist = torch.sqrt(dist_sq)
        
        # TEGR classical limit force coupling via rest mass
        force_mag = self.G * m0.unsqueeze(0) * m0.unsqueeze(1) / (dist**3)
        force_grav = (force_mag.unsqueeze(2) * delta).sum(dim=1)
        
        # Vacuum Anti-Pressure Field
        force_lambda = (1.0/3.0) * self.Lambda * m0.unsqueeze(1) * pos
        
        total_force = force_grav + force_lambda
        
        # 3. Update Momentum (p_new = p + F * dt)
        self.state[:, 4:7] += total_force * self.dt
        
        # 4. Enforce Relativistic Limits
        self._update_gamma()
        
        # 5. Derive velocity and update positions (v = p / (gamma * m0))
        gamma = self.state[:, 9].unsqueeze(1)
        m0_col = m0.unsqueeze(1)
        v = self.state[:, 4:7] / (gamma * m0_col)
        
        self.state[:, 1:4] += v * self.dt

def render_frame(engine, frame_num, output_dir="tegr_frames"):
    os.makedirs(output_dir, exist_ok=True)
    
    pos_cpu = engine.state[:, 1:4].cpu().numpy()
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    ax.scatter(pos_cpu[0, 0], pos_cpu[0, 1], pos_cpu[0, 2], c='white', s=50, marker='*')
    ax.scatter(pos_cpu[1, 0], pos_cpu[1, 1], pos_cpu[1, 2], c='yellow', s=30, marker='*')
    
    mw_end = engine.N_MW + 2
    ax.scatter(pos_cpu[2:mw_end, 0], pos_cpu[2:mw_end, 1], pos_cpu[2:mw_end, 2], c='cyan', s=0.5, alpha=0.3)
    ax.scatter(pos_cpu[mw_end:, 0], pos_cpu[mw_end:, 1], pos_cpu[mw_end:, 2], c='orange', s=0.5, alpha=0.8)
               
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-60, 60])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.2)
    ax.tick_params(colors='white', labelsize=8)
    
    plt.title(f"10D TEGR Collision (Relativistic): Tick {frame_num}", color='white', fontsize=16)
    
    filepath = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)

if __name__ == "__main__":
    print("Starting Authentic 10D TEGR Galactic Collision...")
    engine = TEGRCollisionGPU()
    
    TOTAL_TICKS = 800
    SAVE_INTERVAL = 4
    
    start_time = time.time()
    frame_count = 0
    
    # Pre-allocate CPU RAM array to hold the full 800 ticks for all 12,500 particles
    history = torch.zeros((TOTAL_TICKS, engine.N_total, 10), device='cpu')
    
    for tick in range(1, TOTAL_TICKS + 1):
        engine.step()
        
        # High-speed memory dump from GPU to CPU
        history[tick - 1] = engine.state.cpu()
        
        if tick % SAVE_INTERVAL == 0:
            frame_count += 1
            render_frame(engine, frame_count)
            elapsed = time.time() - start_time
            print(f"Rendered Frame {frame_count:03d} - Elapsed: {elapsed:.1f}s")
            
    print("\nSaving raw 10D tensor trajectory for PySINDy extraction...")
    np.save("tegr_trajectory.npy", history.numpy())
    print(f"Simulation complete! Images saved to 'tegr_frames/' directory.")
    print("\nTo stitch the relativistic frames into a video:")
    print("ffmpeg -y -framerate 30 -pattern_type glob -i 'tegr_frames/*.png' -c:v libx264 -pix_fmt yuv420p tegr_collision.mp4")
