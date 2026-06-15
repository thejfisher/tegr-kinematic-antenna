import torch
import numpy as np

B = 1
N = 2
pos = torch.zeros(B, N, 3)
pos[0, 0] = torch.tensor([-0.001, 4.9, 0.0]) # electron
pos[0, 1] = torch.tensor([0.0, 5.0, 0.0])    # wall

hue = torch.zeros(B, N)
hue[0, 0] = 0.0 # hue_i
hue[0, 1] = 0.0 # hue_j

diff = pos.unsqueeze(2) - pos.unsqueeze(1)
dist_sq = torch.sum(diff**2, dim=3) + 1e-6

hue_diff = hue.unsqueeze(2) - hue.unsqueeze(1)
phase_coupling = torch.cos(hue_diff)

pauli_force = torch.sum(500.0 * phase_coupling.unsqueeze(3) * diff / dist_sq.unsqueeze(3)**2, dim=2)
print("Pauli Force on electron:", pauli_force[0, 0])
