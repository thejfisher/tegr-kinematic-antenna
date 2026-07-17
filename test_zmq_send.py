#!/usr/bin/env python3
"""Test ZeroMQ PUSH sender - mimics the collider sending data"""
import zmq
import numpy as np
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:7777"
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{target}")
print(f"ZMQ PUSH connected to {target}")

# Send a test chunk
fake_data = np.random.randn(100, 10, 10).astype(np.float32)
socket.send_pyobj({"status": "STREAMING", "data": fake_data})
print(f"Sent STREAMING chunk: {fake_data.shape}")

socket.send_pyobj({"status": "DONE"})
print("Sent DONE signal")
print("ZMQ SEND TEST: PASSED!")

socket.close()
context.term()
