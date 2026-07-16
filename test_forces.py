import torch

device = "cpu"
WALL_X = 0.0
slit_bounds = [(-5.0, -1.0), (1.0, 5.0)]
slit_z_<INSERT_BUFFER_USERNAME_HERE>f = 200.0
wall_positions = []
for wy in range(-15, 16):
    in_slit_y = any(lo <= wy <= hi for lo, hi in slit_bounds)
    for wz in range(-8, 9):
        if in_slit_y and abs(wz) < slit_z_<INSERT_BUFFER_USERNAME_HERE>f:
            continue
        wall_positions.append((WALL_X, float(wy), float(wz)))

pos_wall = torch.tensor(wall_positions, device=device)
# Particle approaching slit 2 center (y=3, z=0)
pos_moving = torch.tensor([[-2.0, 3.0, 0.0], [-1.0, 3.0, 0.0], [-0.5, 3.0, 0.0], [-0.1, 3.0, 0.0]], device=device)

PAULI_SCALAR = 1.0
args_pauli_reach = 2.0

diff = pos_moving.unsqueeze(1) - pos_wall.unsqueeze(0)  # (4, N_wall, 3)
dist_sq = torch.sum(diff**2, dim=2)

r_dist = torch.sqrt(dist_sq)
decay_factor = torch.exp(-r_dist / args_pauli_reach)
pauli_force = torch.sum(PAULI_SCALAR * decay_factor.unsqueeze(2) * diff / dist_sq.unsqueeze(2)**1.5, dim=1)

print("Pos: ", pos_moving[:, 0])
print("Force X: ", pauli_force[:, 0])
