import numpy as np
data = np.load('aggregate_states.npz')
final_x = data['final_states'][:, 1]
initial_x = data['initial_states'][:, 0]  # wait, initial_states has shape (B, 3) [y, z, hue] 
print('Max X:', np.max(final_x))
print('Count of particles with final_x < -25:', np.sum(final_x < -25))
print('Count of particles with final_x > -1:', np.sum(final_x > -1))
