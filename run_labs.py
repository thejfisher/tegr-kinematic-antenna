import sys
import time
import subprocess
import csv
from PyQt6.QtWidgets import QApplication
from teleparallel_gui import TeleparallelGUI

def main():
    print("Initializing Teleparallel Control Center (Headless Mode)...")
    app = QApplication(sys.argv)
    gui = TeleparallelGUI()
    
    # We want to loop from 1 to 15 (skipping 0, which is "Custom")
    total_presets = gui.preset_combo.count()
    
    csv_filename = "labs.csv"
    
    print(f"Opening {csv_filename} for writing...")
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write CSV Header
        header = ["Preset_ID", "Preset_Name", "Mode", "Num_Particles", "Mass_A", "Mass_B", "Torsion", "Vacuum", "Output_File"]
        writer.writerow(header)
        
        for i in range(1, total_presets):
            preset_name = gui.preset_combo.itemText(i)
            print(f"\n========================================================")
            print(f"LAB {i}: {preset_name}")
            print(f"========================================================")
            
            gui.preset_combo.setCurrentIndex(i)
            params = gui.collect_params()
            
            for trial in range(1, 11):
                print(f"  --> Trial {trial}/10")
                # Format output filename
                safe_name = preset_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
                # Remove leading number prefix (e.g. "1._1_layer...")
                if safe_name[0].isdigit():
                    safe_name = safe_name.split("_", 1)[1].strip("_")
                output_npy = f"trajectory_{safe_name}_trial_{trial}.npy"
                
                # Use preset_name for label but we can append trial number
                run_label = f"{preset_name}_trial_{trial}"
                
                # Build the subprocess command for teleparallel_collider.py
                cmd = [
                    sys.executable, "teleparallel_collider.py",
                    "--mode", params["mode"],
                    "--num_particles", str(params["num_particles"]),
                    "--mass_a", str(params["mass_a"]),
                    "--mass_b", str(params["mass_b"]),
                    "--pauli", str(params["pauli"]),
                    "--vacuum", str(params["vacuum"]),
                    "--torsion", str(params["torsion"]),
                    "--slit_width", str(params["slit_width"]),
                    "--wall_z_layers", str(params["wall_z_layers"]),
                    "--wall_depth", str(params["wall_depth"]),
                    "--entangled", str(params["entangled"]),
                    "--sink_mass", str(params["sink_mass"]),
                    "--collapse_radius", str(params["collapse_radius"]),
                    "--collapse_G", str(params["collapse_G"]),
                    "--beam_momentum", str(params["beam_momentum"]),
                    "--photon_freq", str(params["photon_freq"]),
                    "--work_function", str(params["work_function"]),
                    "--impact_parameter", str(params["impact_parameter"]),
                    "--amps_cooling_cap", str(params["amps_cooling_cap"]),
                    "--dt", str(params["dt"]),
                    "--total_ticks", str(params["total_ticks"]),
                    "--num_anchors", str(params["num_anchors"]),
                    "--run_label", run_label
                ]
                
                if params["antenna_file"]:
                    cmd.extend(["--antenna_file", params["antenna_file"]])
                if params.get("paper1_exact", 0):
                    cmd.append("--paper1_exact")
                if params.get("spin_coupling", 0):
                    cmd.append("--spin_coupling")
                if params.get("wall_thermal_phase", 0):
                    cmd.append("--wall_thermal_phase")
                if "pauli_power" in params:
                    cmd.extend(["--pauli_power", str(params["pauli_power"])])
                if params.get("qed_vacpol", 0):
                    cmd.append("--qed_vacpol")
                if params.get("qed_lamb", 0):
                    cmd.append("--qed_lamb")
                if params.get("qed_compton", 0):
                    cmd.append("--qed_compton")
                if params.get("polarization_mode") and params["polarization_mode"] != "isotropic":
                    cmd.extend(["--polarization_mode", params["polarization_mode"]])
    
                print(f"Executing: {' '.join(cmd[:4])} ...")
                
                try:
                    # Run the collider
                    subprocess.run(cmd, check=True)
                    
                    # The collider always outputs to 'electron_trajectory.npy' by default.
                    # We need to rename it to our preset-specific file.
                    import os
                    if os.path.exists("electron_trajectory.npy"):
                        if os.path.exists(output_npy):
                            os.remove(output_npy)
                        os.rename("electron_trajectory.npy", output_npy)
                        print(f"Saved trajectory to {output_npy}")
                    else:
                        output_npy = "FAILED (No File)"
                        
                except subprocess.CalledProcessError as e:
                    print(f"ERROR executing lab {i} trial {trial}: {e}")
                    output_npy = "FAILED (Crash)"
                
                # Log results to CSV
                row = [
                    i, 
                    preset_name, 
                    params["mode"], 
                    params["num_particles"], 
                    params["mass_a"], 
                    params["mass_b"], 
                    params["torsion"], 
                    params["vacuum"],
                    output_npy
                ]
                writer.writerow(row)
                
                # small breather between runs
                time.sleep(1)

    print(f"\nAll labs completed. Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
