import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class TEGRMacroGPU:
    def __init__(self, N=10000, dt=0.5):
        self.N = N
        self.dt = dt
        
        # Use CUDA if available, fallback to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Compute Kernel on Device: {self.device}")
        
        self.state = torch.zeros((N, 10), device=self.device)
        self.state[:, 7] = 0.511 # mass
        self.state[:, 9] = 1.0   # gamma

    def setup_spherical_volume(self, r_max):
        """Randomly distribute particles in a 3D sphere up to r_max."""
        # Random radius distribution weighted for uniform volume density
        # r = r_max * (random_uniform)^(1/3)
        u = torch.rand(self.N, device=self.device)
        radii = r_max * torch.pow(u, 1.0/3.0)
        
        # Random directions
        costheta = torch.empty(self.N, device=self.device).uniform_(-1.0, 1.0)
        phi = torch.empty(self.N, device=self.device).uniform_(0.0, 2 * math.pi)
        
        sintheta = torch.sqrt(1.0 - costheta**2)
        
        # Assign spherical coords to x,y,z
        self.state[:, 1] = radii * sintheta * torch.cos(phi)
        self.state[:, 2] = radii * sintheta * torch.sin(phi)
        self.state[:, 3] = radii * costheta
        
        return self.state[:, 1:4].clone()

    def apply_kinematics(self, GM, Lambda):
        self.state[:, 0] += self.dt
        
        bh_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        positions = self.state[:, 1:4]
        
        # Compute vector to center for all particles
        r_vec = positions - bh_pos
        r_mag = torch.norm(r_vec, dim=1, keepdim=True)
        
        # Avoid division by zero
        safe_r_mag = torch.clamp(r_mag, min=0.05)
        
        # 1. Gravity (Inverse Square)
        masses = self.state[:, 7].unsqueeze(1)
        F_grav = - (GM * masses / (safe_r_mag**2)) * (r_vec / safe_r_mag)
        
        # 2. Vacuum Anti-Pressure (Directly Proportional to distance)
        F_lambda = (1.0 / 3.0) * Lambda * masses * r_vec
        
        # Total force tensor
        F_total = F_grav + F_lambda
        
        # Update Momentum (Simple Euler, omitting relativistic speed cap for structural clarity)
        self.state[:, 4:7] += F_total * self.dt
        
        # Derive velocity and update positions
        v_new = self.state[:, 4:7] / masses
        self.state[:, 1:4] += v_new * self.dt

def run_gpu_simulation():
    # Simulation Constants
    N = 10000
    GM = 10.0
    Lambda = 0.001
    dt = 0.5
    total_ticks = 100
    
    # Calculate Theoretical Shatter Point
    r_crit = (3.0 * GM / Lambda)**(1/3) # ~31.07
    r_max = r_crit * 2.0 # Spread universe to 2x the critical boundary
    
    print("============================================================")
    print("MACRO-SCALE GPU COSMOLOGY SIMULATION")
    print("============================================================")
    print(f"Total Particles:  {N}")
    print(f"Gravitational Mass (GM): {GM}")
    print(f"Vacuum Anti-Pressure (Lambda): {Lambda}")
    print(f"Shatter Point (r_crit): {r_crit:.2f} dimensionless units")
    print("------------------------------------------------------------")
    
    engine = TEGRMacroGPU(N=N, dt=dt)
    
    # Save starting positions
    initial_positions = engine.setup_spherical_volume(r_max)
    initial_radii = torch.norm(initial_positions, dim=1)
    
    # Main integration loop
    start_time = time.time()
    for tick in range(1, total_ticks + 1):
        engine.apply_kinematics(GM, Lambda)
        
        if tick % 20 == 0:
            print(f"Tick {tick:03d} / {total_ticks} processed...")
            
    print(f"Integration complete in {time.time() - start_time:.2f} seconds.")
    
    # Analysis for plotting
    final_positions = engine.state[:, 1:4]
    final_radii = torch.norm(final_positions, dim=1)
    
    # Logic: if final_radius < initial_radius, it fell inward (gravity)
    # If final_radius > initial_radius, it flew outward (anti-pressure expansion)
    collapsed_mask = final_radii < initial_radii
    expanded_mask = final_radii >= initial_radii
    
    print("------------------------------------------------------------")
    print(f"Galaxies Collapsed Inward: {collapsed_mask.sum().item()}")
    print(f"Galaxies Expanded Outward: {expanded_mask.sum().item()}")
    
    # Generate 3D Plot
    print("\nGenerating 3D Visualization...")
    # Move data to CPU for plotting
    init_pos_cpu = initial_positions.cpu().numpy()
    col_mask_cpu = collapsed_mask.cpu().numpy()
    exp_mask_cpu = expanded_mask.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Plot galaxies that collapse (Gravity won) -> Blue
    ax.scatter(init_pos_cpu[col_mask_cpu, 0], 
               init_pos_cpu[col_mask_cpu, 1], 
               init_pos_cpu[col_mask_cpu, 2], 
               c='cyan', s=1.5, alpha=0.5, label='Collapse (Gravity)')
               
    # Plot galaxies that expanded (Anti-Pressure won) -> Red
    ax.scatter(init_pos_cpu[exp_mask_cpu, 0], 
               init_pos_cpu[exp_mask_cpu, 1], 
               init_pos_cpu[exp_mask_cpu, 2], 
               c='crimson', s=1.5, alpha=0.4, label='Expansion (Vacuum Tension)')
               
    # Plot a wireframe sphere representing the exact mathematical Shatter Point
    u = torch.linspace(0, 2 * math.pi, 30)
    v = torch.linspace(0, math.pi, 15)
    x_sphere = r_crit * torch.outer(torch.cos(u), torch.sin(v)).numpy()
    y_sphere = r_crit * torch.outer(torch.sin(u), torch.sin(v)).numpy()
    z_sphere = r_crit * torch.outer(torch.ones_like(u), torch.cos(v)).numpy()
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='white', alpha=0.15)
    
    # Formatting the plot
    ax.set_title('Cosmological Expansion Boundary (Shatter Point)', color='white', fontsize=14)
    
    # Set dark theme axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.3)
    ax.tick_params(colors='white')
    
    ax.set_xlim([-r_max, r_max])
    ax.set_ylim([-r_max, r_max])
    ax.set_zlim([-r_max, r_max])
    
    legend = ax.legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
        
    output_filename = "shatter_point_vis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved highly detailed 3D visualization to: {output_filename}")

if __name__ == "__main__":
    run_gpu_simulation()
