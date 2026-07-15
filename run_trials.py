import argparse
import subprocess
import concurrent.futures
import time
import numpy as np
import os
import sys

def run_single_trial(args):
    trial_id, cmd_args = args
    output_file = f"electron_trajectory_trial_{trial_id}.npy"
    
    cmd = ["python3", "teleparallel_collider.py", "--output_file", output_file] + cmd_args
    
    print(f"Starting Trial {trial_id}...")
    start_time = time.time()
    # Run the process
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Trial {trial_id} failed!")
        print(result.stderr)
        return None
        
    print(f"Trial {trial_id} completed in {elapsed:.1f}s")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Parallel Trial Driver for Teleparallel Collider")
    parser.add_argument("--trials", type=int, default=4, help="Number of independent trials to run in parallel")
    parser.add_argument("--output_file", type=str, default="electron_trajectory.npy", help="Final merged output file")
    
    # Parse our args and leave the rest for the collider
    args, collider_args = parser.parse_known_args()
    
    print(f"--- STARTING {args.trials} PARALLEL TRIALS ---")
    print(f"Collider args: {' '.join(collider_args)}")
    
    start_all = time.time()
    
    # Prepare arguments for each trial
    trial_args = [(i, collider_args) for i in range(args.trials)]
    
    output_files = []
    
    # Run in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.trials, 16)) as executor:
        results = executor.map(run_single_trial, trial_args)
        for res in results:
            if res is not None:
                output_files.append(res)
                
    total_elapsed = time.time() - start_all
    print(f"All {len(output_files)} trials completed in {total_elapsed:.1f}s")
    
    if len(output_files) == 0:
        print("Error: All trials failed.")
        sys.exit(1)
        
    # Merge outputs into a batched tensor
    print(f"Merging outputs into {args.output_file}...")
    trajectories = []
    for f in output_files:
        try:
            data = np.load(f)
            # data is usually (Ticks, N, 10). Let's check shape
            trajectories.append(data)
            os.remove(f) # Clean up temp file
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    # Stack along a new batch dimension at axis 0: (B, Ticks, N, 10)
    batched_trajectory = np.stack(trajectories, axis=0)
    
    np.save(args.output_file, batched_trajectory)
    print(f"Successfully saved merged trajectory of shape {batched_trajectory.shape} to {args.output_file}")

if __name__ == "__main__":
    main()
