import numpy as np
s = np.load('/home/<INSERT_USERNAME_HERE>/AI_Vault/teleparallel_sim_photons/tegr_output.npy')
print(f'Shape: {s.shape}')
A = s[-1, 0, :]
B = s[-1, 1, :]
print(f'ParticleA: tau={A[0]:.6f} pos=({A[1]:.2f},{A[2]:.2f},{A[3]:.2f}) p=({A[4]:.3f},{A[5]:.3f},{A[6]:.3f}) gamma={A[9]:.4f} m0={A[7]:.4f}')
print(f'ParticleB: tau={B[0]:.6f} pos=({B[1]:.2f},{B[2]:.2f},{B[3]:.2f}) p=({B[4]:.3f},{B[5]:.3f},{B[6]:.3f}) gamma={B[9]:.4f} m0={B[7]:.4f}')
print(f'CoordTime: {5000 * 0.02:.1f}')
print(f'tau_ratio: {B[0]/A[0]:.6f}')
gA = np.sqrt(1 + (A[4]**2+A[5]**2+A[6]**2)/(A[7]**2 * 100**2))
gB = np.sqrt(1 + (B[4]**2+B[5]**2+B[6]**2)/(B[7]**2 * 100**2))
print(f'gamma_A_computed: {gA:.6f}')
print(f'gamma_B_computed: {gB:.6f}')
# Check initial state too
A0 = s[0, 0, :]
B0 = s[0, 1, :]
print(f'ParticleA_t0: tau={A0[0]:.6f} pos=({A0[1]:.2f},{A0[2]:.2f},{A0[3]:.2f}) p=({A0[4]:.3f},{A0[5]:.3f},{A0[6]:.3f})')
print(f'ParticleB_t0: tau={B0[0]:.6f} pos=({B0[1]:.2f},{B0[2]:.2f},{B0[3]:.2f}) p=({B0[4]:.3f},{B0[5]:.3f},{B0[6]:.3f})')
