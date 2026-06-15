import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class GalaxyCollisionGPU:
    def __init__(self):
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing $O(N^2)$ Compute Kernel on: {self.device}")
        
        # Simulation Parameters
        self.G = 1.0
        self.Lambda = 0.0001 # Very small vacuum tension so gravity dominates the collision
        self.dt = 0.05
        self.softening = 0.5 # Prevents infinite gravity when particles get too close
        
        # Milky Way (MW)
        self.N_MW = 10000
        self.M_MW_core = 2000.0
        
        # Gaia-Enceladus (GE) - Mass ratio approx 1:4
        self.N_GE = 2500
        self.M_GE_core = 500.0
        
        self.N_total = self.N_MW + self.N_GE + 2 # +2 for the cores
        
        # Tensors
        self.pos = torch.zeros((self.N_total, 3), device=self.device)
        self.vel = torch.zeros((self.N_total, 3), device=self.device)
        self.mass = torch.ones(self.N_total, device=self.device) * 0.01 # Star mass is tiny
        
        self._initialize_galaxies()

    def _initialize_galaxies(self):
        # 1. Milky Way Core (Index 0)
        self.pos[0] = torch.tensor([0.0, 0.0, 0.0])
        self.vel[0] = torch.tensor([0.0, 0.0, 0.0])
        self.mass[0] = self.M_MW_core
        
        # 2. Gaia-Enceladus Core (Index 1)
        # Positioned far out, moving inward and slightly "up" for a 3D strike
        self.pos[1] = torch.tensor([120.0, 60.0, 20.0])
        # Orbital velocity aimed for a grazing collision
        self.vel[1] = torch.tensor([-3.0, -1.0, -0.5]) 
        self.mass[1] = self.M_GE_core
        
        # 3. Milky Way Stars (Indices 2 to N_MW+1)
        mw_start = 2
        mw_end = mw_start + self.N_MW
        self._init_disk(mw_start, mw_end, center=self.pos[0], core_mass=self.M_MW_core, 
                        r_min=3.0, r_max=40.0, core_vel=self.vel[0], tilt_x=0.0, tilt_y=0.0)
                        
        # 4. Gaia-Enceladus Stars (Indices N_MW+2 to end)
        ge_start = mw_end
        ge_end = self.N_total
        # Give the dwarf galaxy a tilted disk relative to the Milky Way
        self._init_disk(ge_start, ge_end, center=self.pos[1], core_mass=self.M_GE_core, 
                        r_min=2.0, r_max=15.0, core_vel=self.vel[1], tilt_x=0.5, tilt_y=0.8)

    def _init_disk(self, start_idx, end_idx, center, core_mass, r_min, r_max, core_vel, tilt_x, tilt_y):
        N_stars = end_idx - start_idx
        
        # Random distribution of radii and angles
        u = torch.rand(N_stars, device=self.device)
        radii = r_min + (r_max - r_min) * torch.sqrt(u) # Flat disk density
        angles = torch.rand(N_stars, device=self.device) * 2 * math.pi
        
        # 2D Disk Positions
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        z = torch.zeros(N_stars, device=self.device) + (torch.randn(N_stars, device=self.device) * 0.5) # Slight thickness
        
        # 2D Disk Velocities (v = sqrt(GM/r))
        v_mag = torch.sqrt(self.G * core_mass / radii)
        vx = -v_mag * torch.sin(angles)
        vy = v_mag * torch.cos(angles)
        vz = torch.zeros(N_stars, device=self.device)
        
        # Apply 3D Tilt
        pos_disk = torch.stack((x, y, z), dim=1)
        vel_disk = torch.stack((vx, vy, vz), dim=1)
        
        # Simple rotation matrices for tilt
        rot_x = torch.tensor([[1, 0, 0], [0, math.cos(tilt_x), -math.sin(tilt_x)], [0, math.sin(tilt_x), math.cos(tilt_x)]], device=self.device)
        rot_y = torch.tensor([[math.cos(tilt_y), 0, math.sin(tilt_y)], [0, 1, 0], [-math.sin(tilt_y), 0, math.cos(tilt_y)]], device=self.device)
        
        pos_disk = pos_disk @ rot_x @ rot_y
        vel_disk = vel_disk @ rot_x @ rot_y
        
        # Offset to core center and add core velocity
        self.pos[start_idx:end_idx] = pos_disk + center
        self.vel[start_idx:end_idx] = vel_disk + core_vel

    def compute_n_body_gravity(self):
        # O(N^2) Full Tensor Broadcast
        # pos[j] - pos[i]
        # Memory check: 12500 x 12500 x 3 = 468 Million floats (1.8 GB). Well within 44GB VRAM!
        delta = self.pos.unsqueeze(0) - self.pos.unsqueeze(1) # shape: [N, N, 3]
        
        # r^2 + softening^2
        dist_sq = (delta**2).sum(dim=2) + self.softening**2 # shape: [N, N]
        dist = torch.sqrt(dist_sq) # shape: [N, N]
        
        # Acceleration magnitude: G * m_j / r^3
        accel_mag = self.G * self.mass.unsqueeze(0) / (dist**3) # shape: [N, N]
        
        # Vector acceleration
        accel_grav = (accel_mag.unsqueeze(2) * delta).sum(dim=1) # shape: [N, 3]
        
        # Add Vacuum Anti-Pressure expansion
        accel_lambda = (1.0/3.0) * self.Lambda * self.pos
        
        return accel_grav + accel_lambda

    def step(self):
        accel = self.compute_n_body_gravity()
        self.vel += accel * self.dt
        self.pos += self.vel * self.dt

def render_frame(engine, frame_num, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pull data to CPU
    pos_cpu = engine.pos.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Milky Way Core
    ax.scatter(pos_cpu[0, 0], pos_cpu[0, 1], pos_cpu[0, 2], c='white', s=50, marker='*')
    # Gaia-Enceladus Core
    ax.scatter(pos_cpu[1, 0], pos_cpu[1, 1], pos_cpu[1, 2], c='yellow', s=30, marker='*')
    
    # Milky Way Stars
    mw_end = engine.N_MW + 2
    ax.scatter(pos_cpu[2:mw_end, 0], pos_cpu[2:mw_end, 1], pos_cpu[2:mw_end, 2], 
               c='cyan', s=0.5, alpha=0.3)
               
    # Gaia-Enceladus Stars
    ax.scatter(pos_cpu[mw_end:, 0], pos_cpu[mw_end:, 1], pos_cpu[mw_end:, 2], 
               c='orange', s=0.5, alpha=0.8)
               
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-60, 60])
    
    # Formatting
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.2)
    ax.tick_params(colors='white', labelsize=8)
    
    plt.title(f"Galactic Collision: Tick {frame_num}", color='white', fontsize=16)
    
    filepath = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)

if __name__ == "__main__":
    print("Starting $O(N^2)$ Galactic Collision Simulation...")
    engine = GalaxyCollisionGPU()
    
    TOTAL_TICKS = 800
    SAVE_INTERVAL = 4
    
    start_time = time.time()
    
    frame_count = 0
    for tick in range(1, TOTAL_TICKS + 1):
        engine.step()
        
        if tick % SAVE_INTERVAL == 0:
            render_frame(engine, tick)
            frame_count += 1
            elapsed = time.time() - start_time
            print(f"Rendered Frame {frame_count:03d} (Tick {tick}/{TOTAL_TICKS}) - Elapsed: {elapsed:.1f}s")
            
    print(f"\nSimulation complete! {frame_count} frames saved to 'frames/' directory.")
    print("\nTo stitch these frames into a video, run this command in your Ubuntu terminal:")
    print("ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p collision.mp4")
