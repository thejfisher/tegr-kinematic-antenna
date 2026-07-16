import torch
import torch.nn.functional as F

# 1. Create a dummy grid of shape (D, H, W) = (X, Y, Z) = (2, 2, 2)
# X=0 -> val=0, X=1 -> val=10
# Y=0 -> val=0, Y=1 -> val=100
# Z=0 -> val=0, Z=1 -> val=1000
phi = torch.zeros(2, 2, 2)
phi[1, :, :] += 10
phi[:, 1, :] += 100
phi[:, :, 1] += 1000

input_tensor = phi.view(1, 1, 2, 2, 2).float()

# We want to sample at X=1, Y=0, Z=0.
# The value should be 10.
# Let's map X=1 to +1, Y=0 to -1, Z=0 to -1
# If we pass [X, Y, Z] = [+1, -1, -1]
grid_xyz = torch.tensor([[[[ [1., -1., -1.] ]]]])
val_xyz = F.grid_sample(input_tensor, grid_xyz, align_corners=True)

# If we pass [Z, Y, X] = [-1, -1, +1]
grid_zyx = torch.tensor([[[[ [-1., -1., 1.] ]]]])
val_zyx = F.grid_sample(input_tensor, grid_zyx, align_corners=True)

print("Target value for X=1, Y=0, Z=0 is 10")
print("Passing [X,Y,Z] gives:", val_xyz.item())
print("Passing [Z,Y,X] gives:", val_zyx.item())
