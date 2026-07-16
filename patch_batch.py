import re

with open('teleparallel_collider.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add GRID initialization
grid_init = '''        n_batches = (N_trials + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"--- TONOMURA PROTOCOL (GPU BATCH): {N_trials} trials in {n_batches} batches of {BATCH_SIZE} ---")
        print(f"  Device: {device} | Each batch: {BATCH_SIZE} beams (m={args.mass_a}) + {N_wall} wall (m={m_per_particle:.0e})")
        
        # --- EULERIAN PILOT WAVE GRID (BATCH MODE) ---
        GRID_RES = 100
        GRID_MIN = -50.0
        GRID_MAX = 50.0
        GRID_SIZE = GRID_MAX - GRID_MIN
        DX = GRID_SIZE / GRID_RES
        
        laplacian_kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=torch.float32)
        laplacian_kernel[0, 0, 1, 1, 1] = -6.0
        laplacian_kernel[0, 0, 1, 1, 0] = 1.0
        laplacian_kernel[0, 0, 1, 1, 2] = 1.0
        laplacian_kernel[0, 0, 1, 0, 1] = 1.0
        laplacian_kernel[0, 0, 1, 2, 1] = 1.0
        laplacian_kernel[0, 0, 0, 1, 1] = 1.0
        laplacian_kernel[0, 0, 2, 1, 1] = 1.0
        
        max_stable_speed = 0.5 * DX / DT
        WAVE_SPEED = min(C, max_stable_speed)
        C_SQUARED = (WAVE_SPEED * DT / DX) ** 2
        TORSION_DECAY = 0.95'''
content = re.sub(r'        n_batches = \(N_trials \+ BATCH_SIZE \- 1\) // BATCH_SIZE\n        print\(f"\-\-\- TONOMURA PROTOCOL \(GPU BATCH\): \{N_trials\} trials in \{n_batches\} batches of \{BATCH_SIZE\} \-\-\-"\)\n        print\(f"  Device: \{device\} \| Each batch: \{BATCH_SIZE\} beams \(m=\{args\.mass_a\}\) \+ \{N_wall\} wall \(m=\{m_per_particle:\.0e\}\)"\)', grid_init, content)

# 2. Add phi init
phi_init = '''            initial_hue = batch_state[:, 0, 8].cpu().numpy().copy()
            
            if args.pilot_wave:
                phi_curr = torch.zeros((B, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
                phi_prev = torch.zeros((B, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
                phi_next = torch.zeros((B, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
                obstacle_mask = torch.ones((1, 1, GRID_RES, GRID_RES, GRID_RES), device=device, dtype=torch.float32)
                for wx, wy, wz in wall_positions:
                    ix = int((wx - GRID_MIN) / DX)
                    iy = int((wy - GRID_MIN) / DX)
                    iz = int((wz - GRID_MIN) / DX)
                    if 0 <= ix < GRID_RES and 0 <= iy < GRID_RES and 0 <= iz < GRID_RES:
                        obstacle_mask[0, 0, ix, iy, iz] = 0.0'''
content = content.replace('            initial_hue = batch_state[:, 0, 8].cpu().numpy().copy()', phi_init)

# 3. Add wave integration and force
wave_int = '''                m0_moving = m0[:, :N_moving, :]
                
                pilot_wave_force = torch.zeros_like(pos_moving)
                if args.pilot_wave:
                    import torch.nn.functional as F
                    # 1. Eulerian Wave Propagation
                    phi_next = 2.0 * phi_curr - phi_prev + C_SQUARED * (
                        F.conv3d(phi_curr, laplacian_kernel, padding=1)
                    ) - 0.05 * (phi_curr - phi_prev)
                    phi_next = phi_next * TORSION_DECAY * obstacle_mask
                    
                    # 2. Continuous Eulerian Emission
                    indices = ((pos_moving - GRID_MIN) / DX).long()
                    valid = (indices[:, :, 0] >= 0) & (indices[:, :, 0] < GRID_RES) & \\
                            (indices[:, :, 1] >= 0) & (indices[:, :, 1] < GRID_RES) & \\
                            (indices[:, :, 2] >= 0) & (indices[:, :, 2] < GRID_RES)
                            
                    if valid.any():
                        gamma_v = torch.sqrt(1.0 + torch.sum(mom_moving**2, dim=2) / (m0_moving.squeeze(2)**2 * C**2))
                        hue_m = batch_state[:, :N_moving, 8]
                        emission = (10.0 * gamma_v * m0_moving.squeeze(2)) * torch.cos(hue_m)
                        
                        b_idx, n_idx = torch.where(valid)
                        v_idx = indices[b_idx, n_idx]
                        v_emission = emission[b_idx, n_idx]
                        flat_idx = b_idx * (GRID_RES**3) + v_idx[:, 0] * (GRID_RES**2) + v_idx[:, 1] * GRID_RES + v_idx[:, 2]
                        phi_flat = phi_next.view(-1)
                        phi_flat.scatter_add_(0, flat_idx, v_emission)
                        
                    phi_prev = phi_curr
                    phi_curr = phi_next
                    
                    # 3. Pilot Wave Force via trilinear gradient sampling
                    grad_x = torch.zeros_like(phi_curr)
                    grad_x[:, :, 1:-1, :, :] = (phi_curr[:, :, 2:, :, :] - phi_curr[:, :, :-2, :, :]) / (2.0 * DX)
                    grad_y = torch.zeros_like(phi_curr)
                    grad_y[:, :, :, 1:-1, :] = (phi_curr[:, :, :, 2:, :] - phi_curr[:, :, :, :-2, :]) / (2.0 * DX)
                    grad_z = torch.zeros_like(phi_curr)
                    grad_z[:, :, :, :, 1:-1] = (phi_curr[:, :, :, :, 2:] - phi_curr[:, :, :, :, :-2]) / (2.0 * DX)
                    
                    grad_field = torch.cat([grad_x, grad_y, grad_z], dim=1)
                    
                    norm_coords = 2.0 * (pos_moving - GRID_MIN) / GRID_SIZE - 1.0
                    norm_coords = norm_coords.flip(-1) # (z, y, x) order
                    
                    grid = norm_coords.unsqueeze(1).unsqueeze(1)
                    sampled = F.grid_sample(grad_field, grid, mode='bilinear', padding_mode='border', align_corners=True)
                    grad_vec = sampled.squeeze(2).squeeze(2).transpose(1, 2)
                    
                    pilot_wave_force = -args.mass_a * (C**2) * grad_vec'''
content = content.replace('                m0_moving = m0[:, :N_moving, :]', wave_int)

# 4. Add force to total
content = content.replace('                total_force = torsion_force + pauli_force + damping_force + cmb_force',
                          '                total_force = torsion_force + pauli_force + damping_force + cmb_force + pilot_wave_force')

with open('teleparallel_collider.py', 'w', encoding='utf-8') as f:
    f.write(content)
