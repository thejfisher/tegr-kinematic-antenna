$ErrorActionPreference = "Stop"

Write-Host "Running Coefficient Stability Test (10 Trials for High-Gamma)"
for ($i=1; $i -le 10; $i++) {
    Write-Host "Running Trial $i..."
    python teleparallel_collider.py --mode double-slit --num_particles 10000 --mass_a 1.0 --mass_b 1.0 --pauli 500.0 --vacuum 0.001 --torsion 1.0 --slit_width 4.0 --wall_z_layers 1 --wall_depth 1 --beam_momentum 25000.0 --dt 0.001 --total_ticks 5000 --pauli_power 2 --spin_coupling --run_label "High_Gamma_Trial_$i"
}

Write-Host "Aggregating SINDy Coefficients..."
python aggregate_coeffs.py
