#!/usr/bin/env python3
"""End-to-end ZMQ pipeline test: Simulates what the collider does.
Run on thejfisher, sends to hal, hal processes and prints."""
import zmq
import numpy as np
import time
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "100.66.100.83:7777"
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{target}")
print(f"[SENDER] Connected to {target}")
time.sleep(1)  # Let connection establish

# Simulate what the collider sends: (ticks, N_particles, 10) chunks
N = 50
for chunk_id in range(5):
    fake_data = np.random.randn(100, N, 10).astype(np.float32)
    socket.send_pyobj({
        "status": "STREAMING",
        "chunk_id": chunk_id * 100,
        "data": fake_data
    })
    print(f"[SENDER] Sent STREAMING chunk {chunk_id}: shape={fake_data.shape}")
    sys.stdout.flush()

# Send DONE
socket.send_pyobj({"status": "DONE"})
print("[SENDER] Sent DONE signal")
print("[SENDER] TEST COMPLETE")
sys.stdout.flush()
socket.close()
context.term()
