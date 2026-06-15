#!/usr/bin/env python3
"""Test ZeroMQ PULL server on hal - mimics sindy_zmq_server.py"""
import zmq
import numpy as np
import sys

port = 7777
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind(f"tcp://*:{port}")
print(f"ZMQ PULL server bound on port {port}. Waiting for data...")
sys.stdout.flush()

# Wait up to 30 seconds for a message
socket.setsockopt(zmq.RCVTIMEO, 30000)
try:
    message = socket.recv_pyobj()
    print(f"RECEIVED: {type(message)}")
    if isinstance(message, dict):
        print(f"  Status: {message.get('status')}")
        if 'data' in message:
            print(f"  Data shape: {message['data'].shape}")
    print("ZMQ RECEIVE TEST: PASSED!")
except zmq.Again:
    print("TIMEOUT: No data received in 30 seconds")
    print("ZMQ RECEIVE TEST: FAILED (no sender)")
finally:
    socket.close()
    context.term()
