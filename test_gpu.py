import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    t = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print("GPU tensor test:", t * 2)
else:
    print("NO GPU DETECTED - falling back to CPU")
