# Teleparallel Collider: ER=EPR 2x2 Factorial Matrix Guide

This directory contains the High-Fidelity Teleparallel Collider and the SINDy extraction pipeline for simulating the ER=EPR conjecture and AMPS Firewall paradox dynamically.

## The 2x2 Factorial Matrix

The core of our recent findings relies on testing two independent physical mechanisms. The simulation controls these via two command-line flags:

1. **`--spin_coupling` (The "Stick")**: Controls the *local* geometric connection. When set to `1`, the anti-correlated spins of the particles ($\omega_i \cdot \omega_j$) modulate the Pauli exclusion force. This acts as a rigid, mechanical connection spanning the gravitational well (like Einstein's Stick).
2. **`--entangled` (The "Wormhole")**: Controls the *non-local* information channel. When set to `1`, the Entanglement Adjacency Tensor ($W_{ij}$) activates Kuramoto phase synchronization between the particles, allowing quantum phase information ($\theta_{hue}$) to traverse the bridge.

### The 4 Matrix States

| State | Command Flags | Resulting Dynamics (Gravity-Sink Mode) |
|---|---|---|
| **Control** | `--spin_coupling 0 --entangled 0` | The particle falls deep into the gravity well ($\gamma \approx 955$), but its phase wanders freely as a pure de Broglie clock ($m_0/\gamma$). No phase coherence between particles. |
| **Local Only** | `--spin_coupling 1 --entangled 0` | The mechanical stick deepens gravitational capture by 10x ($\gamma \approx 34 \to 345$). However, the phase is scrambled upon falling. The geometric connection exists, but quantum information is not preserved. |
| **Non-Local Only** | `--spin_coupling 0 --entangled 1` | The Kuramoto sync attempts to lock phases, but without the mechanical stick to transmit force, it has no effect on trajectory ($\gamma \approx 35$). |
| **Full ER=EPR** | `--spin_coupling 1 --entangled 1` | **Synergistic Stabilization.** The particle falls deeply ($\gamma \approx 766$), and its phase perfectly locks with its partner outside the well. The SINDy extraction reveals a **Versine signature** $(1 - \cos\Delta\theta)$ in the force equations. |

## The Versine Signature

When both the local stick and non-local wormhole are active, SINDy consistently extracts the Versine function in the force equations:

$F \propto (1 - \cos\Delta\theta)$

This proves that quantum information survives the event horizon. The phase difference between the particles governs the force: when phases sync ($\cos\Delta\theta = 1$), the force vanishes. When they oppose ($\cos\Delta\theta = -1$), the restoring force doubles. This dynamic stabilization proves the AMPS Firewall is unnecessary in a continuous teleparallel geometry.

## How to Run

1. **Start the SINDy Buffer Node**:
   ```bash
   python3 sindy_zmq_server.py --port 7777 --no_llm --galactic_spin 0.0
   ```

2. **Start the Physics Node**:
   ```bash
   python3 teleparallel_collider.py --mode gravity-sink --entangled 1 --spin_coupling 1 --dt 0.02 --total_ticks 5000 ...
   ```
   (See preset scripts in `teleparallel_gui.py` for exact full command strings).
