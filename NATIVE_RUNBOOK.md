# Native Teleparallel Simulator Runbook

This runbook outlines how to operate the new, high-performance **Native Desktop GUI**, which replaces the legacy web-browser stack.

## Architecture Change
We have completely eliminated the React web frontend (`collider-web`) and the FastAPI server (`api.py`). The application is now a single, monolithic native Python desktop application.

Because it is native, the PyTorch engine (`teleparallel_collider.py`) calculates the massive matrices on your **NVIDIA CUDA GPU**, and the native charting engine (`pyqtgraph`) reads those arrays directly from your ultra-fast NVMe storage/RAM and renders them instantly using OpenGL hardware acceleration.

## How to Launch

You do **NOT** need to start any web servers or `npm run dev` environments anymore.

Simply open PowerShell in your `z:\teleparallel_sim` directory and run:

```powershell
python teleparallel_gui.py
```

This single command launches the entire control center.

## GUI Features

1. **Presets:** Selecting a preset from the top left dropdown will automatically update all the visual parameter boxes below it to match the exact mathematical states required for those experiments.
2. **AI SINDy Interpretation:** If left unchecked (default), the mathematical extraction (PySINDy) will run, but the Ollama AI synthesis will be skipped. This saves ~3 minutes per run, allowing you to rapidly iterate on 10,000-particle batches.
3. **Hardware-Accelerated Visualization:** The top-right panels use `pyqtgraph` to render the exact `vy'` scattering coordinates natively.

## GPU Verification
Yes! The physics engine is still the exact same `teleparallel_collider.py` script. When you click run in the GUI, it executes the PyTorch code directly on your local NVIDIA GPU (the exact same AI machine hardware we have been using). You can confirm this by reading the top of the GUI log output, which will state: `Initializing High-Fidelity Teleparallel Collider on: cuda`
