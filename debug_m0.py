import numpy as np

history = np.load('electron_trajectory.npy')
m0_history = history[:, 0, 7]
print("m0 values:", m0_history[:45])
