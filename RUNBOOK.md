# Teleparallel Collider Simulation: Runbook

This document outlines how the simulation environment is architected and the exact steps required to boot it up from scratch.

## System Architecture

Because the simulation requires both local file access to the Windows `Z:` drive and heavy AI inference via Ollama (`deepseek-r1:14b`) running on your Ubuntu machine (`thejfisher-AI`), the system runs in a hybrid environment:

1. **Ollama AI Server**: Runs on the Ubuntu machine (`127.0.0.1`).
2. **FastAPI Backend (`api.py`)**: Runs locally on Windows to execute PySINDy extraction and mathematical physics safely on the `Z:` drive.
3. **React Frontend (`collider-web`)**: Runs locally on Windows, providing the UI dashboard.
4. **SSH Tunnel**: A secure port-forwarding bridge that connects the Windows Backend directly to the Ubuntu Ollama server.

---

## Boot Sequence (How to start the environment)

If you ever restart your computer or close your terminals, open **three separate Windows terminal tabs** and run the following commands in order:

### 1. Establish the AI Bridge (SSH Tunnel)
This forwards the local port `11434` on your Windows machine to the Ollama server on the Ubuntu machine. Keep this terminal open in the background.
```bash
# In Terminal 1 (Windows PowerShell/Command Prompt)
ssh -N -L 11434:localhost:11434 thejfisher@127.0.0.1
```

### 2. Start the Physics API Backend
This runs the Python server that manages the collider, PySINDy extraction, and talks to Ollama via the tunnel.
```bash
# In Terminal 2 (Windows PowerShell/Command Prompt)
cd Z:\teleparallel_sim
python api.py
```

### 3. Start the React Dashboard
This runs the Vite server for the web interface.
```bash
# In Terminal 3 (Windows PowerShell/Command Prompt)
cd Z:\teleparallel_sim\collider-web
npm run dev
```

Once all three are running, simply open your browser to `http://localhost:5173` to access the dashboard!

---

**DISCLAIMER**: This toy model is not proposing a physical medium of the nothingness of space (i.e. it does not support aether theory). Particles are treated as resonant wave defects in a teleparallel vacuum strictly as a mathematical and topological framework for modeling gravitational and physical interactions.
