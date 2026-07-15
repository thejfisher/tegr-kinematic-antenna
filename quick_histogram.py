import json
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.exists('double_slit_results.json'):
    print("Could not find double_slit_results.json")
    exit(1)

with open('double_slit_results.json', 'r') as f:
    results = json.load(f)

y_positions = results.get('y_positions', [])
if not y_positions:
    print('No particles hit the screen!')
else:
    plt.figure(figsize=(10, 6))
    
    # Check if we have spin data
    data = np.load('aggregate_states.npz')
    outcomes = data['outcomes']
    hit_mask = outcomes == 1
    
    if 'spins' in data and np.any(data['spins'] != 0):
        spins = data['spins'][hit_mask]
        y_positions_arr = np.array(y_positions)
        
        spin_up = y_positions_arr[spins == 1]
        spin_down = y_positions_arr[spins == -1]
        
        plt.hist([spin_up, spin_down], bins=50, stacked=True, color=['red', 'cyan'], label=['Spin Up', 'Spin Down'])
    else:
        plt.hist(y_positions, bins=50, color='blue', alpha=0.7)
        
    plt.title('Double Slit Interference - Y-Axis Histogram')
    plt.xlabel('Y Position on Screen')
    plt.ylabel('Particle Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
