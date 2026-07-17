import argparse
import subprocess
import time
import os
import itertools
from multiprocessing import Pool, current_process

# Configure your Swarm setup
# 3 GPUs on thejfisher: cuda:0 (3060), cuda:1 (4060 Ti), cuda:2 (5060 Ti)
GPU_IDS = [0, 1, 2] 
def run_physics_job(val_pauli, val_vac, val_torsion, val_cmb, test_mode):
    """
    Spawns a subprocess for the physics engine with isolated VRAM.
    """

    
    # Dynamically assign GPU and ZMQ port based on the worker process ID
    # This guarantees 100% GPU utilization without cross-talk on ZMQ ports
    worker_id = (current_process()._identity[0] - 1) % len(GPU_IDS)
    gpu_id = GPU_IDS[worker_id]
    zmq_port = 7777 + worker_id
    
    run_label = f"P_{val_pauli:.1f}_V_{val_vac}_T_{val_torsion}_C_{val_cmb}"
    
    ticks = 100 if test_mode else 5000
    particles = 50 if test_mode else 1000
    
    cmd = [
        "python3", "teleparallel_collider_BZ.py",
        "--mode", "direct-collapse",
        "--num_particles", str(particles),
        "--total_ticks", str(ticks),
        "--pauli_power", str(int(val_pauli)),
        "--vacuum", str(val_vac),
        "--torsion", str(val_torsion),
        "--cmb_noise", str(val_cmb),
        "--device", "cuda:0",
        "--zmq_target", f"127.0.0.1:{zmq_port}",
        "--run_label", run_label
    ]
    
    print(f"[{run_label}] Starting on GPU {gpu_id} -> ZMQ Port {zmq_port} ...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    start_t = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    duration = time.time() - start_t
    
    if result.returncode != 0:
        print(f"[{run_label}] FAILED with code {result.returncode}.")
        if result.stderr:
            print(f"[{run_label}] ERROR TRACEBACK:\n{result.stderr}")
        return False
    else:
        print(f"[{run_label}] SUCCESS. Completed in {duration:.1f}s.")
        return True

def job_wrapper(combo):
    return run_physics_job(combo[0], combo[1], combo[2], combo[3], combo[4])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run a quick test cycle.")
    args = parser.parse_args()
    
    # 1. Define the Sweep Grid
    if args.test_mode:
        pauli_vals = [2.0, 3.0]
        vac_vals = [0.0]
        tor_vals = [1.0]
        cmb_vals = [0.01]
    else:
        # Huge Stress Test Grid
        pauli_vals = [2.0, 3.0, 4.0, 5.0, 6.0]
        vac_vals = [0.0, 0.01, 0.1]
        tor_vals = [0.5, 1.0, 2.0]
        cmb_vals = [0.0, 0.01, 0.05]
        
    all_combinations = list(itertools.product(pauli_vals, vac_vals, tor_vals, cmb_vals))
    total_jobs = len(all_combinations)
    
    print(f"Starting Multi-GPU Auto-Stress Test.")
    print(f"Total Parameter Combinations: {total_jobs}")
    print(f"Distributing across GPUs: {GPU_IDS}")
    
    # Add test_mode to each combination so we only have one argument for pool.map
    job_configs = [combo + (args.test_mode,) for combo in all_combinations]
    
    # 3. Execute dynamically. The Pool distributes the 135 jobs to the next available worker.
    # The worker's _identity securely locks it to one GPU and one ZMQ Port!
    start_time = time.time()
    
    with Pool(processes=len(GPU_IDS)) as pool:
        results = pool.map(job_wrapper, job_configs)
        
    total_duration = time.time() - start_time
    success_count = sum(1 for r in results if r)
    
    print(f"\n--- GRID SEARCH COMPLETE ---")
    print(f"Total Jobs: {total_jobs}")
    print(f"Successful: {success_count} / {total_jobs}")
    print(f"Total Time Elapsed: {total_duration / 3600:.2f} hours")

if __name__ == "__main__":
    main()
