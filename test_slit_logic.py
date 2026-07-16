import torch

device = "cpu"
slit_bounds = [(-5.0, -1.0), (1.0, 5.0)]
slit_lo_tensor = torch.tensor([lo for lo, hi in slit_bounds], device=device)
slit_hi_tensor = torch.tensor([hi for lo, hi in slit_bounds], device=device)

by = torch.tensor([-6.0, -3.0, 0.0, 3.0, 6.0], device=device)

by_expanded = by.unsqueeze(1)
in_any_slit_y = ((by_expanded >= slit_lo_tensor.unsqueeze(0)) & (by_expanded <= slit_hi_tensor.unsqueeze(0))).any(dim=1)

for i in range(len(by)):
    print(f"y={by[i].item()} -> passes? {in_any_slit_y[i].item()}")
