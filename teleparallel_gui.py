import sys
import time
import subprocess
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox, QTextEdit, QSplitter, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ==========================================
# Worker Thread to run Python Subprocesses
# ==========================================
class PhysicsWorker(QThread):
    log_signal = pyqtSignal(str)
    math_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            pipeline_enabled = self.params.get("pipeline_enabled", False)
            script_name = "teleparallel_collider.py"
            args = []
            
            for k, v in self.params.items():
                if k in ["ai_translation", "ai_model", "ollama_url", "pipeline_enabled", 
                         "physics_ip", "physics_user", "physics_dir", 
                         "buffer_ip", "buffer_user", "buffer_dir", "buffer_port"]:
                    continue
                if isinstance(v, bool):
                    if v:
                        args.append(f"--{k}")
                        if k in ["pauli_enabled", "vacuum_enabled", "torsion_enabled", "entangled"]:
                            args.append("1")
                else:
                    # Skip empty-string params (e.g. antenna_file left blank)
                    if str(v).strip() == "":
                        continue
                    args.extend([f"--{k}", str(v)])

            if pipeline_enabled:
                physics_ip = self.params["physics_ip"]
                physics_user = self.params["physics_user"]
                physics_dir = self.params["physics_dir"]
                buffer_ip = self.params["buffer_ip"]
                buffer_port = self.params["buffer_port"]
                buffer_user = self.params["buffer_user"]
                buffer_dir = self.params["buffer_dir"]
                ollama_url = self.params.get("ollama_url", "")
                ai_model = self.params.get("ai_model", "")
                ai_translation = self.params.get("ai_translation", True)
                
                self.log_signal.emit("--- TRI-NODE PING-PONG PIPELINE INITIATED ---")
                
                # 0. Kill any stale SINDy server holding the port
                cleanup_cmd = f"fuser -k {buffer_port}/tcp 2>/dev/null; pkill -f sindy_zmq_server 2>/dev/null; sleep 1; echo 'Port {buffer_port} cleared'"
                ssh_cleanup = ["ssh", f"{buffer_user}@{buffer_ip}", cleanup_cmd]
                self.log_signal.emit(f"Clearing stale processes on {buffer_ip}:{buffer_port}...")
                cleanup_proc = subprocess.run(ssh_cleanup, capture_output=True, text=True, timeout=10)
                self.log_signal.emit(f"[CLEANUP] {cleanup_proc.stdout.strip()}")
                
                # 1. Start Buffer Node SINDy Server (hal)
                self.log_signal.emit(f">>> STARTING BUFFER NODE SINDY SERVER ({buffer_user}@{buffer_ip}) <<<")
                
                gal_spin = self.params.get("galactic_spin", 0.0)
                server_cmd = f"cd {buffer_dir} && python3 -u sindy_zmq_server.py --port {buffer_port} --no_llm --galactic_spin {gal_spin}"
                ssh_server_cmd = ["ssh", f"{buffer_user}@{buffer_ip}", server_cmd]
                self.log_signal.emit(f"Server launch: {' '.join(ssh_server_cmd)}")
                server_process = subprocess.Popen(ssh_server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                
                # Wait for server to initialize (ROCm PyTorch import takes ~10s on first load)
                self.log_signal.emit("Waiting 15s for SINDy server to initialize (ROCm first-load)...")
                import select, os
                server_ready = False
                for sec in range(15):
                    time.sleep(1)
                    # Check if server has printed anything (means it's past import stage)
                    self.log_signal.emit(f"  ... {sec+1}s")
                
                self.log_signal.emit("SINDy server init window complete. Launching physics...")
                
                # 2. Start Physics Node (thejfisher) via SSH
                self.log_signal.emit(f">>> STARTING PHYSICS ENGINE ON ({physics_user}@{physics_ip}) <<<")
                args.extend(["--zmq_target", f"{buffer_ip}:{buffer_port}"])
                
                # Quote arguments for shell execution
                args_str = " ".join([f"'{arg}'" if " " in arg else arg for arg in args])
                physics_cmd = f"cd {physics_dir} && python3 -u teleparallel_collider.py {args_str}"
                
                ssh_physics_cmd = ["ssh", f"{physics_user}@{physics_ip}", physics_cmd]
                self.log_signal.emit(f"Physics launch: {' '.join(ssh_physics_cmd)}")
                physics_process = subprocess.Popen(ssh_physics_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                
                for line in iter(physics_process.stdout.readline, ''):
                    self.log_signal.emit("[PHYSICS] " + line.strip())
                    # Windows SSH doesn't close stdout on remote exit - break on known final lines
                    if "ZMQ Stream finished" in line or "Trajectory saved" in line:
                        break
                physics_process.stdout.close()
                physics_process.terminate()
                try:
                    physics_process.wait(timeout=5)
                except Exception:
                    physics_process.kill()
                
                self.log_signal.emit("[DEBUG] Physics complete. Proceeding to buffer output...")
                
                self.log_signal.emit(">>> WAITING FOR BUFFER NODE SINDY EXTRACTION TO FINISH <<<")
                
                # Read server output with a timeout - server should exit after DONE processing
                import time as _time
                math_output = []
                deadline = _time.time() + 120  # 2 minute max wait for SINDy
                while _time.time() < deadline:
                    line = server_process.stdout.readline()
                    if line == '':
                        break  # EOF - server exited
                    self.log_signal.emit("[BUFFER] " + line.strip())
                    math_output.append(line)
                else:
                    self.log_signal.emit("[DEBUG] Server timeout (120s). Killing server process.")
                    server_process.kill()
                
                server_process.stdout.close()
                try:
                    server_process.wait(timeout=5)
                except Exception:
                    server_process.kill()
                
                math_text = "".join(math_output)
                clean_math = ""
                if "Cleaned Math Equations:" in math_text:
                    try:
                        clean_math = math_text.split("Cleaned Math Equations:")[1].split("========================================")[0].strip()
                        self.math_signal.emit(clean_math)
                    except IndexError:
                        pass
                
                # 3. If AI enabled, GUI sends HTTP POST to Ollama API
                if ai_translation and ollama_url and clean_math:
                    self.log_signal.emit("\n>>> PIPING EQUATIONS TO AI NODE VIA GUI <<<")
                    self.log_signal.emit(f"GUI requesting Ollama API at {ollama_url} ({ai_model})...")
                    
                    prompt = f"Act as a world-class theoretical physicist analyzing mathematical data. I have built a simulation of a particle collision system in a Weitzenböck lattice (a Teleparallel equivalent to General Relativity).\n\nHERE ARE THE PYSINDY DIFFERENTIAL EQUATIONS FOR THE SYSTEM:\n{clean_math}\n\nINSTRUCTIONS FOR TRANSLATION:\n1. Decode this mathematical output strictly using the TEGR / ER=EPR framework provided.\n2. Analyze how the internal variables behave.\n3. Do NOT use generic terms or spoon-fed answers. Synthesize a profound, original theoretical physics conclusion."
                    
                    payload = {
                        "model": ai_model,
                        "prompt": prompt,
                        "stream": False
                    }
                    try:
                        api_url = ollama_url.rstrip('/') + "/api/generate"
                        response = requests.post(api_url, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            self.log_signal.emit("\n========================================")
                            self.log_signal.emit("OLLAMA TRANSLATION:")
                            self.log_signal.emit("========================================\n")
                            self.log_signal.emit(result.get("response", "No response text found."))
                        else:
                            self.log_signal.emit(f"Ollama API Error: {response.status_code} - {response.text}")
                    except Exception as e:
                        self.log_signal.emit(f"\nERROR: Could not connect to Ollama from GUI: {e}")
                
            else:
                # Classic local execution
                self.log_signal.emit("--- STARTING LOCAL PHYSICS ENGINE ---")
                cmd = [sys.executable, script_name] + args
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                
                for line in iter(process.stdout.readline, ''):
                    self.log_signal.emit(line.strip())
                process.stdout.close()
                process.wait()
                
                if process.returncode != 0:
                    self.log_signal.emit("\n*** ERROR: PHYSICS ENGINE CRASHED! ***")
                    self.finished_signal.emit()
                    return

                self.log_signal.emit("\n[SUCCESS] LOCAL PHYSICS ENGINE COMPLETE")
                self.log_signal.emit("--- RUNNING LOCAL PYSINDY EXTRACTION ---")
                
                mode = self.params.get("mode", "")
                sindy_script = "sindy_qed.py" if mode == "qed" else "sindy_extract.py"
                context = "electron QED scattering" if mode == "qed" else "electron Moller scattering"
                sindy_cmd = [sys.executable, sindy_script, "--file", "electron_trajectory.npy", "--context", context]
                if not self.params.get("ai_translation", True):
                    sindy_cmd.append("--no_llm")
                if "ai_model" in self.params:
                    sindy_cmd.extend(["--model", self.params["ai_model"]])
                if "ollama_url" in self.params:
                    sindy_cmd.extend(["--url", self.params["ollama_url"]])
                    
                sindy_process = subprocess.Popen(sindy_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                math_output = []
                for line in iter(sindy_process.stdout.readline, ''):
                    self.log_signal.emit(line.strip())
                    math_output.append(line)
                sindy_process.stdout.close()
                sindy_process.wait()
                
                math_text = "".join(math_output)
                if "Cleaned Math Equations:" in math_text:
                    clean_math = math_text.split("Cleaned Math Equations:")[1].split("Piping")[0].strip()
                    self.math_signal.emit(clean_math)
            
            self.finished_signal.emit()
            
        except Exception as e:
            self.log_signal.emit(f"ERROR: {str(e)}")
            self.finished_signal.emit()

# ==========================================
# 3D OpenGL Viewer Window
# ==========================================
class Teleparallel3DViewer(QMainWindow):
    def __init__(self, parent=None, mode="double-slit"):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle("Teleparallel 3D Geometry Viewer")
        self.resize(1200, 900)
        self.gl_widget = gl.GLViewWidget()
        self.setCentralWidget(self.gl_widget)
        
        # Grid
        grid = gl.GLGridItem()
        grid.scale(2, 2, 2)
        self.gl_widget.addItem(grid)
        
        # Load Data
        if self.mode in ["double-slit", "quantum-eraser"]:
            self.load_aggregate_data()
        else:
            self.load_trajectory_data()

    def load_trajectory_data(self):
        try:
            data = np.load("electron_trajectory.npy")
            ticks, N, _ = data.shape
            if N < 2: return
            
            p0 = data[:, 0, 1:4]
            if not np.isfinite(p0).all():
                p0 = np.nan_to_num(p0)
            c0 = np.zeros((ticks, 4))
            c0[:, 0] = 1.0; c0[:, 3] = 0.8
            l0 = gl.GLLinePlotItem(pos=p0, color=c0, mode='line_strip', width=2)
            self.gl_widget.addItem(l0)
            
            p1 = data[:, 1, 1:4]
            if not np.isfinite(p1).all():
                p1 = np.nan_to_num(p1)
            c1 = np.zeros((ticks, 4))
            c1[:, 1] = 0.9; c1[:, 2] = 1.0; c1[:, 3] = 0.8
            l1 = gl.GLLinePlotItem(pos=p1, color=c1, mode='line_strip', width=2)
            self.gl_widget.addItem(l1)
            
            self.gl_widget.setCameraPosition(distance=30, elevation=20, azimuth=45)
        except Exception as e:
            print(f"Error loading 3D trajectory data: {e}")
            
    def load_aggregate_data(self):
        try:
            data = np.load("aggregate_states.npz")
            initial_states = data['initial_states']
            final_states = data['final_states']
            outcomes = data['outcomes']
            
            hit_mask = outcomes == 1
            hit_initial = initial_states[hit_mask]
            hit_final = final_states[hit_mask]
            
            has_spin = 'spins' in data and np.any(data['spins'] != 0)
            hit_spins = data['spins'][hit_mask] if has_spin else np.zeros(len(hit_final))
            
            if len(hit_final) == 0:
                return
                
            pos_array = np.zeros((len(hit_final), 3))
            color_array = np.zeros((len(hit_final), 4))
            
            line_pos = np.zeros((len(hit_final) * 2, 3))
            line_color = np.zeros((len(hit_final) * 2, 4))
            
            for i in range(len(hit_final)):
                pos_array[i] = [hit_final[i, 1], hit_final[i, 2], hit_final[i, 3]]
                
                if has_spin:
                    if hit_spins[i] > 0:
                        color_array[i] = [1.0, 0.27, 0.27, 0.8] # Red
                    else:
                        color_array[i] = [0.0, 0.9, 1.0, 0.8] # Cyan
                else:
                    color_array[i] = [0.73, 0.52, 0.98, 0.8] # Purple

                start_pos = [-10.0, hit_initial[i, 0], hit_final[i, 3]]
                line_pos[2*i] = start_pos
                line_pos[2*i+1] = pos_array[i]
                
                c = color_array[i].copy()
                c[3] = 0.15 # Highly transparent beam lines
                line_color[2*i] = c
                line_color[2*i+1] = c
                
            scatter = gl.GLScatterPlotItem(pos=pos_array, color=color_array, size=5)
            self.gl_widget.addItem(scatter)
            
            lines = gl.GLLinePlotItem(pos=line_pos, color=line_color, mode='lines', antialias=True)
            self.gl_widget.addItem(lines)
            
            # Set Camera to see X (depth), Y (width), Z (height)
            self.gl_widget.setCameraPosition(distance=25, elevation=15, azimuth=45)
            
        except Exception as e:
            print(f"Error loading 3D data: {e}")

# ==========================================
# Main Native GUI
# ==========================================
class TeleparallelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teleparallel Collider Control Center - Native Python Edition")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #e0e0e0; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { color: #00e5ff; font-weight: bold; border: 1px solid #333; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QLabel { color: #aaa; }
            QLineEdit, QComboBox { background-color: #1e1e1e; color: #fff; border: 1px solid #444; padding: 5px; border-radius: 3px; }
            QPushButton { background-color: #bb86fc; color: #000; font-weight: bold; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #9965f4; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QTextEdit { background-color: #0d0d0d; color: #00ff00; font-family: Consolas, monospace; border: 1px solid #333; }
        """)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel (Controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_group.setCheckable(True)
        preset_group.setChecked(True)
        self.preset_container = QWidget()
        preset_layout = QVBoxLayout(self.preset_container)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["0. Custom", "1. 1-Layer Classical Scatter", "2. 5-Layer Phase Router", "3. 3D Phase Router (5x3)", "4. AMPS Firewall (Gravity Sink)", "5. Direct Collapse (N-Body)", "6. Delayed Choice Quantum Eraser", "7. Delayed Choice (Heat Sink Eraser)", "8. Veneziano Amplitude (Soft Scattering)", "9. Photoelectric Effect", "10. Holographic Entanglement", "11. Holographic Shell (Dark Matter Void)", "12. Holographic Ring (Accretion Void)", "13. Quantum Electrodynamics (QED)", "14. Gravitational Wave (NANOGrav)", "15. GHZ Entanglement (Active Firewall)", "16. High-Gamma Test Preset", "17. The Mark Thompson Experiment", "18. Stochastic Hum (Mark Thompson Phase 2)"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        preset_group_layout = QVBoxLayout()
        preset_group_layout.addWidget(self.preset_container)
        preset_group.setLayout(preset_group_layout)
        preset_group.toggled.connect(self.preset_container.setVisible)
        left_layout.addWidget(preset_group)

        # Physics Parameters
        param_group = QGroupBox("Physics Parameters (Click to Toggle)")
        param_group.setCheckable(True)
        param_group.setChecked(False)
        
        self.param_container = QWidget()
        form_layout = QFormLayout(self.param_container)
        
        self.inputs = {}
        def add_input(label, default_val):
            le = QLineEdit(str(default_val))
            self.inputs[label] = le
            form_layout.addRow(QLabel(label), le)

        add_input("mode", "double-slit")
        add_input("num_particles", "10000")
        add_input("mass_a", "1.0")
        add_input("mass_b", "1.0")
        add_input("pauli", "500.0")
        add_input("vacuum", "0.001")
        add_input("torsion", "1.0")
        add_input("slit_width", "4.0")
        add_input("wall_z_layers", "3")
        add_input("wall_depth", "5")
        add_input("entangled", "0")
        add_input("sink_mass", "50000.0")
        add_input("collapse_radius", "20.0")
        add_input("collapse_G", "1.0")
        add_input("beam_momentum", "5000.0")
        add_input("photon_freq", "100.0")
        add_input("work_function", "10.0")
        add_input("impact_parameter", "0.5")
        add_input("amps_cooling_cap", "1.0")
        add_input("num_anchors", "1")
        add_input("antenna_file", "")
        add_input("galactic_spin", "0.0")
        add_input("dt", "0.001")
        add_input("total_ticks", "5000")
        
        self.eraser_active_cb = QCheckBox("Eraser Active (No Measurement Wall)")
        self.eraser_active_cb.setChecked(False)
        form_layout.addRow(self.eraser_active_cb)
        
        self.firewall_active_cb = QCheckBox("AMPS Firewall Active (Shed Momentum)")
        self.firewall_active_cb.setChecked(False)
        form_layout.addRow(self.firewall_active_cb)
        
        polarization_layout = QHBoxLayout()
        polarization_layout.addWidget(QLabel("Gravitational Polarization:"))
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["isotropic", "plus", "cross", "mixed"])
        polarization_layout.addWidget(self.polarization_combo)
        form_layout.addRow(polarization_layout)

        # Force Law Configuration (Paper 1 vs Code)
        force_group = QGroupBox("Force Law Configuration")
        force_group.setCheckable(True)
        force_group.setChecked(False)
        force_group.setStyleSheet("QGroupBox { color: #ff9800; }")
        self.force_container = QWidget()
        force_layout = QVBoxLayout(self.force_container)
        
        self.paper1_exact_cb = QCheckBox("Paper 1 Exact Math (overrides below)")
        self.paper1_exact_cb.setChecked(False)
        self.paper1_exact_cb.stateChanged.connect(self.on_paper1_toggle)
        force_layout.addWidget(self.paper1_exact_cb)
        
        self.spin_coupling_cb = QCheckBox("Spin-Vorticity Coupling (\u03c9\u1d62 \u00b7 \u03c9\u2c7c)")
        self.spin_coupling_cb.setChecked(False)
        force_layout.addWidget(self.spin_coupling_cb)
        
        self.thermal_bath_cb = QCheckBox("Wall Thermal Bath (Random Phase)")
        self.thermal_bath_cb.setChecked(False)
        force_layout.addWidget(self.thermal_bath_cb)
        
        self.qed_vacpol_cb = QCheckBox("Vacuum Polarization (Uehling)")
        self.qed_vacpol_cb.setChecked(False)
        force_layout.addWidget(self.qed_vacpol_cb)
        
        self.qed_lamb_cb = QCheckBox("Lamb Shift (Self-Energy Jitter)")
        self.qed_lamb_cb.setChecked(False)
        force_layout.addWidget(self.qed_lamb_cb)
        
        self.qed_compton_cb = QCheckBox("Compton Scattering (Radiation Wave)")
        self.qed_compton_cb.setChecked(False)
        force_layout.addWidget(self.qed_compton_cb)
        
        pauli_power_layout = QHBoxLayout()
        pauli_power_layout.addWidget(QLabel("Pauli Power Law:"))
        self.pauli_power_combo = QComboBox()
        self.pauli_power_combo.addItems(["1/r\u00b3 (Code/KK)", "1/r\u00b2 (Paper 1)"])
        pauli_power_layout.addWidget(self.pauli_power_combo)
        force_layout.addLayout(pauli_power_layout)
        
        force_group_layout = QVBoxLayout()
        force_group_layout.addWidget(self.force_container)
        force_group.setLayout(force_group_layout)
        force_group.toggled.connect(self.force_container.setVisible)
        self.force_container.setVisible(False)
        left_layout.addWidget(force_group)

        param_group_layout = QVBoxLayout()
        self.unlock_params_cb = QCheckBox("Unlock Parameters (Manual Override)")
        self.unlock_params_cb.setChecked(False)
        self.unlock_params_cb.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.unlock_params_cb.stateChanged.connect(lambda state: self.param_container.setEnabled(bool(state)))
        param_group_layout.addWidget(self.unlock_params_cb)
        param_group_layout.addWidget(self.param_container)
        param_group.setLayout(param_group_layout)
        self.param_container.setEnabled(False)
        self.param_container.setVisible(False)
        self.unlock_params_cb.setVisible(False)
        def on_param_toggle(checked):
            self.param_container.setVisible(checked)
            self.unlock_params_cb.setVisible(checked)
        param_group.toggled.connect(on_param_toggle)
        left_layout.addWidget(param_group)

        # Execution Engine
        exec_group = QGroupBox("Execution Engine (Tri-Node Pipeline)")
        exec_group.setCheckable(True)
        exec_group.setChecked(True)
        self.exec_container = QWidget()
        exec_layout = QFormLayout(self.exec_container)
        
        self.pipeline_checkbox = QCheckBox("Enable ZMQ Ping-Pong Distributed Pipeline")
        self.pipeline_checkbox.setChecked(True)
        
        self.physics_ip_input = QLineEdit("100.122.147.67")
        self.physics_user_input = QLineEdit("thejfisher")
        self.physics_dir_input = QLineEdit("~/AI_Vault/teleparallel_sim")
        
        self.buffer_ip_input = QLineEdit("100.66.100.83")
        self.buffer_user_input = QLineEdit("hal")
        self.buffer_port_input = QLineEdit("7777")
        self.buffer_dir_input = QLineEdit("~/hxseq-vsgx4/teleparallel_sim")
        
        exec_layout.addRow(self.pipeline_checkbox)
        exec_layout.addRow("Physics Node IP (thejfisher):", self.physics_ip_input)
        exec_layout.addRow("Physics Node Username:", self.physics_user_input)
        exec_layout.addRow("Physics Node Directory:", self.physics_dir_input)
        exec_layout.addRow("Buffer Node IP (hal):", self.buffer_ip_input)
        exec_layout.addRow("Buffer Node Username:", self.buffer_user_input)
        exec_layout.addRow("Buffer Node Port:", self.buffer_port_input)
        exec_layout.addRow("Buffer Node Directory:", self.buffer_dir_input)
        
        exec_group_layout = QVBoxLayout()
        exec_group_layout.addWidget(self.exec_container)
        exec_group.setLayout(exec_group_layout)
        exec_group.toggled.connect(self.exec_container.setVisible)
        self.exec_container.setVisible(True)
        
        left_layout.addWidget(exec_group)

        # Run Button
        self.run_btn = QPushButton("RUN SIMULATION")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("background-color: #03dac6; color: #000; font-size: 16px;")
        left_layout.addWidget(self.run_btn)
        
        # 3D Viewer Button
        self.view_3d_btn = QPushButton("OPEN 3D VIEWER")
        self.view_3d_btn.clicked.connect(self.open_3d_viewer)
        self.view_3d_btn.setStyleSheet("background-color: #bb86fc; color: #000; font-size: 16px; margin-top: 10px;")
        left_layout.addWidget(self.view_3d_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right Panel (Visualization & Logs)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top right: Graphics
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('#121212')
        self.plot_spatial = self.graph_widget.addPlot(title="Spatial Y-Distribution (Histogram)")
        self.plot_spatial.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial.setLabel('left', 'Hits')
        self.plot_spatial.setLabel('bottom', 'Screen Y Position')

        self.plot_phase = self.graph_widget.addPlot(title="Phase Router (Hue vs Final Y)")
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.plot_phase.setLabel('left', 'Final Y Position')
        self.plot_phase.setLabel('bottom', 'Initial Hidden Phase (Hue)')
        
        right_splitter.addWidget(self.graph_widget)

        # Bottom right: Logs and Math
        log_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_splitter.addWidget(self.log_console)
        
        self.math_console = QTextEdit()
        self.math_console.setReadOnly(True)
        self.math_console.setStyleSheet("color: #00e5ff; font-weight: bold;")
        log_splitter.addWidget(self.math_console)
        
        right_splitter.addWidget(log_splitter)
        right_splitter.setSizes([500, 300])

        main_layout.addWidget(right_splitter)

    def apply_preset(self, index):
        text = self.preset_combo.currentText()
        if ". " in text:
            text = text.split(". ", 1)[1]
        
        # Baseline physics configuration for the mathematical model
        self.paper1_exact_cb.setChecked(False)
        self.spin_coupling_cb.setChecked(True)
        self.pauli_power_combo.setCurrentIndex(1) # Default to 1/r^2
        self.qed_vacpol_cb.setChecked(False)
        self.qed_lamb_cb.setChecked(False)
        self.qed_compton_cb.setChecked(False)

        if text == "1-Layer Classical Scatter":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
        elif text == "5-Layer Phase Router":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("5")
        elif text == "3D Phase Router (5x3)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.inputs["mode"].setText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("3")
            self.inputs["wall_depth"].setText("5")
            self.inputs["entangled"].setText("0")
        elif text == "AMPS Firewall (Gravity Sink)":
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("50000.0")
        elif text == "Direct Collapse (N-Body)":
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("direct-collapse")
            self.inputs["num_particles"].setText("50")
            self.inputs["mass_a"].setText("100.0")
            self.inputs["mass_b"].setText("100.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.01")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
        elif text == "Delayed Choice Quantum Eraser":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("quantum-eraser")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["entangled"].setText("1")
            self.eraser_active_cb.setChecked(False)
        elif text == "Delayed Choice (Heat Sink Eraser)":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("heat-sink-eraser")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["entangled"].setText("0")
        elif text == "Veneziano Amplitude (Soft Scattering)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3 KK
            self.inputs["mode"].setText("head-on")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500000.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["beam_momentum"].setText("50000.0")
            self.inputs["impact_parameter"].setText("0.1")
            self.inputs["dt"].setText("0.0001")
            self.inputs["total_ticks"].setText("20000")
            self.inputs["entangled"].setText("0")
        elif text == "Photoelectric Effect":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3 KK
            self.inputs["mode"].setText("photoelectric")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("5000.0")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["torsion"].setText("1.0")
            self.inputs["beam_momentum"].setText("5000.0")
            self.inputs["photon_freq"].setText("500.0")
            self.inputs["work_function"].setText("1000.0")
            self.inputs["entangled"].setText("0")
            self.inputs["dt"].setText("0.000001")
            self.inputs["total_ticks"].setText("10000")
            
        elif text == "Holographic Entanglement":
            self.inputs["mode"].setText("holographic")
            self.inputs["num_particles"].setText("50")
            self.inputs["collapse_radius"].setText("50.0")
            self.inputs["entangled"].setText("300")
            self.inputs["collapse_G"].setText("10.0")
            
        elif text == "Holographic Shell (Dark Matter Void)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3
            self.inputs["mode"].setText("holographic-shell")
            self.inputs["num_particles"].setText("1000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["torsion"].setText("1.0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("100.0") # Need gravity to test collapse
            self.inputs["entangled"].setText("1000") # fully entangled shell
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("15000")
            
        elif text == "Holographic Ring (Accretion Void)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3
            self.inputs["mode"].setText("holographic-ring")
            self.inputs["num_particles"].setText("1000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["torsion"].setText("1.0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("100.0") # Need gravity to test collapse
            self.inputs["entangled"].setText("1000") # fully entangled ring
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("15000")
            
        elif text == "Quantum Electrodynamics (QED)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3
            self.inputs["mode"].setText("qed")
            self.inputs["num_particles"].setText("1000")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["dt"].setText("0.001")
            self.qed_vacpol_cb.setChecked(True)
            self.qed_lamb_cb.setChecked(True)
            self.qed_compton_cb.setChecked(True)
            self.inputs["total_ticks"].setText("15000")
            self.inputs["dt"].setText("0.001")
            self.inputs["photon_freq"].setText("1.0")
            self.inputs["work_function"].setText("1000.0")
            self.inputs["impact_parameter"].setText("0.01")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["torsion"].setText("1.0")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["beam_momentum"].setText("0.0")

        elif text == "Gravitational Wave (NANOGrav)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3
            self.inputs["mode"].setText("antenna")
            self.inputs["num_particles"].setText("16") # 1 source, 15 anchors
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["torsion"].setText("1.0")
            self.inputs["collapse_radius"].setText("50.0")
            self.inputs["entangled"].setText("15") # fully entangled array
            self.inputs["num_anchors"].setText("15")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("15000")
            self.polarization_combo.setCurrentText("plus")
        elif text == "GHZ Entanglement (Active Firewall)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("direct-collapse")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.01")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("50000.0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("50000")
            self.eraser_active_cb.setChecked(True)
            self.firewall_active_cb.setChecked(True)
            self.pauli_power_combo.setCurrentIndex(1) # 1/r^2
        elif text == "High-Gamma Test Preset":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["beam_momentum"].setText("25000.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("5000")
            self.eraser_active_cb.setChecked(False)
            self.firewall_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(1)
        elif text == "The Mark Thompson Experiment":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("antenna")
            self.inputs["num_particles"].setText("101") # 1 source, 100 anchors
            self.inputs["num_anchors"].setText("100")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["collapse_radius"].setText("5.0")
            self.inputs["galactic_spin"].setText("0.5")  # Milky way spin analogue
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("10000")
            self.eraser_active_cb.setChecked(False)
            self.firewall_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(2) # 1/r^3 Pauli for TEGR Antenna
        elif text == "Stochastic Hum (Mark Thompson Phase 2)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.inputs["mode"].setText("antenna")
            self.inputs["num_particles"].setText("101") # 1 source, 100 anchors
            self.inputs["num_anchors"].setText("100")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["collapse_radius"].setText("5.0")
            self.inputs["galactic_spin"].setText("0.5")  # Milky way spin analogue
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("10000")
            self.eraser_active_cb.setChecked(False)
            self.firewall_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(2) # 1/r^3 Pauli
            self.inputs["antenna_file"].setText("pta_residuals.csv") # Stochastic Hum Injection

    def run_simulation(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("SIMULATING...")
        self.log_console.clear()
        self.math_console.clear()
        self.plot_spatial.clear()
        self.plot_phase.clear()
        if hasattr(self, 'plot_phase_zoom'):
            self.plot_spatial_zoom.clear()
            self.plot_phase_zoom.clear()

        params = {
            "run_label": self.preset_combo.currentText(),
            "mode": self.inputs["mode"].text(),
            "num_particles": self.inputs["num_particles"].text(),
            "mass_a": self.inputs["mass_a"].text(),
            "mass_b": self.inputs["mass_b"].text(),
            "pauli": self.inputs["pauli"].text(),
            "vacuum": self.inputs["vacuum"].text(),
            "torsion": self.inputs["torsion"].text(),
            "eraser_active": 1 if self.eraser_active_cb.isChecked() else 0,
            "firewall_active": 1 if self.firewall_active_cb.isChecked() else 0,
            "slit_width": self.inputs["slit_width"].text(),
            "wall_z_layers": self.inputs["wall_z_layers"].text(),
            "wall_depth": self.inputs["wall_depth"].text(),
            "entangled": self.inputs["entangled"].text(),
            "sink_mass": self.inputs["sink_mass"].text(),
            "collapse_radius": self.inputs["collapse_radius"].text(),
            "collapse_G": self.inputs["collapse_G"].text(),
            "beam_momentum": self.inputs["beam_momentum"].text(),
            "photon_freq": self.inputs["photon_freq"].text(),
            "work_function": self.inputs["work_function"].text(),
            "impact_parameter": self.inputs["impact_parameter"].text(),
            "amps_cooling_cap": self.inputs["amps_cooling_cap"].text(),
            "dt": self.inputs["dt"].text(),
            "total_ticks": self.inputs["total_ticks"].text(),
            "num_anchors": self.inputs["num_anchors"].text(),
            "antenna_file": self.inputs["antenna_file"].text(),
            "galactic_spin": self.inputs["galactic_spin"].text(),
            "paper1_exact": 1 if self.paper1_exact_cb.isChecked() else 0,
            "spin_coupling": 1 if self.spin_coupling_cb.isChecked() else 0,
            "wall_thermal_phase": 1 if self.thermal_bath_cb.isChecked() else 0,
            "pauli_power": 2 if self.pauli_power_combo.currentIndex() == 1 else 3,
            "qed_vacpol": 1 if self.qed_vacpol_cb.isChecked() else 0,
            "qed_lamb": 1 if self.qed_lamb_cb.isChecked() else 0,
            "qed_compton": 1 if self.qed_compton_cb.isChecked() else 0,
            "polarization_mode": self.polarization_combo.currentText(),
            "ai_translation": False,
            "ai_model": "",
            "ollama_url": "",
            "pipeline_enabled": self.pipeline_checkbox.isChecked(),
            "physics_ip": self.physics_ip_input.text().strip(),
            "physics_user": self.physics_user_input.text().strip(),
            "physics_dir": self.physics_dir_input.text().strip(),
            "buffer_ip": self.buffer_ip_input.text().strip(),
            "buffer_user": self.buffer_user_input.text().strip(),
            "buffer_port": self.buffer_port_input.text().strip(),
            "buffer_dir": self.buffer_dir_input.text().strip()
        }

        self.worker = PhysicsWorker(params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.math_signal.connect(self.set_math)
        self.worker.finished_signal.connect(self.on_run_finished)
        self.worker.start()

    def append_log(self, text):
        self.log_console.append(text)
        # Auto-scroll to bottom
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_math(self, text):
        self.math_console.setPlainText(text)

    def open_3d_viewer(self):
        current_mode = self.inputs["mode"].text()
        self.viewer_3d = Teleparallel3DViewer(self, mode=current_mode)
        self.viewer_3d.show()

    def on_paper1_toggle(self, state):
        """Master toggle: when Paper 1 Exact is checked, force sub-settings to match Paper 1."""
        is_paper1 = bool(state)
        if is_paper1:
            self.spin_coupling_cb.setChecked(True)
            self.pauli_power_combo.setCurrentIndex(1)  # 1/r^2
        self.spin_coupling_cb.setEnabled(not is_paper1)
        self.pauli_power_combo.setEnabled(not is_paper1)
        self.thermal_bath_cb.setEnabled(not is_paper1)

    def on_run_finished(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN SIMULATION")
        self.update_plots()

    def update_plots(self):
        try:
            mode = self.inputs["mode"].text().strip()
            if mode != "double-slit":
                self.plot_spatial.clear()
                self.plot_phase.clear()
                self.append_log(f"Plots cleared for {mode} mode.")
                return

            # Load the data generated by the backend directly into numpy and pyqtgraph
            data = np.load("aggregate_states.npz")
            final_states = data['final_states']      
            initial_states = data['initial_states']  
            outcomes = data['outcomes']
            
            # Load spin data if available
            has_spin = 'spins' in data and np.any(data['spins'] != 0)
            all_spins = data['spins'] if 'spins' in data else np.zeros(len(outcomes))

            hit_mask = outcomes == 1
            hit_initial_states = initial_states[hit_mask]
            hit_final_states = final_states[hit_mask]
            hit_spins = all_spins[hit_mask]

            if len(hit_final_states) == 0:
                self.append_log("No particles hit the screen.")
                return

            final_ys = hit_final_states[:, 2]
            final_zs = hit_final_states[:, 3]
            
            if hit_initial_states.shape[1] == 3:
                initial_ys = hit_initial_states[:, 0]
                initial_zs = hit_initial_states[:, 1]
                initial_hues = hit_initial_states[:, 2]
            else:
                initial_ys = hit_initial_states[:, 0]
                initial_zs = np.zeros_like(initial_ys)
                initial_hues = hit_initial_states[:, 1]
            
            is_3d = (final_zs.max() - final_zs.min()) > 0.01

            # Color scheme: spin-based or Y-based
            if has_spin:
                spin_up = hit_spins > 0
                spin_dn = ~spin_up
                color_up = pg.mkColor('#ff4444')  # Red = spin-up
                color_dn = pg.mkColor('#00e5ff')  # Cyan = spin-down
                brushes = [color_up if s > 0 else color_dn for s in hit_spins]
                n_up = spin_up.sum()
                n_dn = spin_dn.sum()
                self.append_log(f"Spin coloring: {n_up} spin-up (red) | {n_dn} spin-down (cyan)")
            else:
                norm_y = (initial_ys - initial_ys.min()) / (initial_ys.max() - initial_ys.min() + 1e-9)
                cmap = pg.colormap.get('viridis')
                brushes = cmap.map(norm_y, mode='qcolor')

            # 1. Spatial Distribution — Histogram (1D) or Scatter (2D)
            if is_3d:
                self.plot_spatial.setTitle("3D Spatial Distribution (Y vs Z Screen Hits)")
                self.plot_spatial.setLabel('left', 'Screen Z Position')
                self.plot_spatial.setLabel('bottom', 'Screen Y Position')
                
                # Plot as scatter points (maintaining spin coloring)
                scatter_spatial = pg.ScatterPlotItem(x=final_ys, y=final_zs, size=2.0, brush=brushes, pen=None)
                self.plot_spatial.addItem(scatter_spatial)
            else:
                self.plot_spatial.setTitle("Spatial Y-Distribution (Histogram)")
                self.plot_spatial.setLabel('left', 'Hits')
                self.plot_spatial.setLabel('bottom', 'Screen Y Position')
                
                if has_spin:
                    ys_up = final_ys[spin_up]
                    ys_dn = final_ys[spin_dn]
                    all_min = final_ys.min()
                    all_max = final_ys.max()
                    bins = np.linspace(all_min, all_max, 201)
                    y_up, _ = np.histogram(ys_up, bins=bins)
                    y_dn, _ = np.histogram(ys_dn, bins=bins)
                    bars_up = pg.BarGraphItem(x=bins[:-1], height=y_up, width=(bins[1]-bins[0]), brush=pg.mkColor(255, 68, 68, 150))
                    bars_dn = pg.BarGraphItem(x=bins[:-1], height=y_dn, width=(bins[1]-bins[0]), brush=pg.mkColor(0, 229, 255, 150))
                    self.plot_spatial.addItem(bars_dn)
                    self.plot_spatial.addItem(bars_up)
                else:
                    y, x = np.histogram(final_ys, bins=200)
                    bars = pg.BarGraphItem(x=x[:-1], height=y, width=0.1, brush='#bb86fc')
                    self.plot_spatial.addItem(bars)

            # 2. Scatter Plot (Hue vs Final Y) - FULL RANGE
            scatter = pg.ScatterPlotItem(x=initial_hues, y=final_ys, size=1.5, brush=brushes, pen=None)
            self.plot_phase.addItem(scatter)
            if has_spin:
                self.plot_phase.setTitle("Phase Router (Red=Spin↑  Cyan=Spin↓)")
            
            # 3. ZOOMED Phase Router - focus on Y=±10 to reveal fine structure
            if not hasattr(self, 'plot_phase_zoom'):
                self.graph_widget.nextRow()
                self.plot_spatial_zoom = self.graph_widget.addPlot(title="Histogram (Zoomed Y ±10)")
                self.plot_spatial_zoom.showGrid(x=True, y=True, alpha=0.3)
                self.plot_spatial_zoom.setLabel('left', 'Hits')
                self.plot_spatial_zoom.setLabel('bottom', 'Screen Y Position')
                
                self.plot_phase_zoom = self.graph_widget.addPlot(title="Phase Router (Zoomed Y ±10)")
                self.plot_phase_zoom.showGrid(x=True, y=True, alpha=0.3)
                self.plot_phase_zoom.setLabel('left', 'Final Y Position')
                self.plot_phase_zoom.setLabel('bottom', 'Initial Hidden Phase (Hue)')
            else:
                self.plot_spatial_zoom.clear()
                self.plot_phase_zoom.clear()
            
            # Zoomed histogram or scatter
            zoom_mask = (final_ys > -10) & (final_ys < 10)
            zoom_ys = final_ys[zoom_mask]
            zoom_zs = final_zs[zoom_mask]
            if len(zoom_ys) > 0:
                if is_3d:
                    self.plot_spatial_zoom.setTitle("3D Zoomed (Y vs Z)")
                    self.plot_spatial_zoom.setLabel('left', 'Screen Z Position')
                    self.plot_spatial_zoom.setLabel('bottom', 'Screen Y Position')
                    zoom_brushes = [brushes[i] for i in range(len(brushes)) if zoom_mask[i]]
                    scatter_zoom = pg.ScatterPlotItem(x=zoom_ys, y=zoom_zs, size=3.0, brush=zoom_brushes, pen=None)
                    self.plot_spatial_zoom.addItem(scatter_zoom)
                else:
                    self.plot_spatial_zoom.setTitle("Histogram (Zoomed Y ±10)")
                    self.plot_spatial_zoom.setLabel('left', 'Hits')
                    self.plot_spatial_zoom.setLabel('bottom', 'Screen Y Position')
                    if has_spin:
                        zoom_spins = hit_spins[zoom_mask]
                        zoom_up = zoom_spins > 0
                        zoom_dn = ~zoom_up
                        bins_z = np.linspace(-10, 10, 201)
                        yz_up, _ = np.histogram(final_ys[zoom_mask][zoom_up], bins=bins_z)
                        yz_dn, _ = np.histogram(final_ys[zoom_mask][zoom_dn], bins=bins_z)
                        bars_z_up = pg.BarGraphItem(x=bins_z[:-1], height=yz_up, width=0.1, brush=pg.mkColor(255, 68, 68, 150))
                        bars_z_dn = pg.BarGraphItem(x=bins_z[:-1], height=yz_dn, width=0.1, brush=pg.mkColor(0, 229, 255, 150))
                        self.plot_spatial_zoom.addItem(bars_z_dn)
                        self.plot_spatial_zoom.addItem(bars_z_up)
                    else:
                        y_z, x_z = np.histogram(zoom_ys, bins=100)
                        bars_z = pg.BarGraphItem(x=x_z[:-1], height=y_z, width=0.1, brush='#bb86fc')
                        self.plot_spatial_zoom.addItem(bars_z)
            
            # Zoomed scatter - spin colored
            scatter_zoom = pg.ScatterPlotItem(x=initial_hues, y=final_ys, size=1.5, brush=brushes, pen=None)
            self.plot_phase_zoom.addItem(scatter_zoom)
            self.plot_phase_zoom.setYRange(-10, 10)
            if has_spin:
                self.plot_phase_zoom.setTitle("Phase Router Zoomed (Red=Spin↑  Cyan=Spin↓)")
            
            self.append_log(f"Successfully loaded and plotted {len(final_ys)} screen hits out of {len(outcomes)} total trials.")
            self.append_log("Zero transport latency: Web browser JSON bottleneck successfully bypassed.")

        except Exception as e:
            self.append_log(f"Error loading plot data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TeleparallelGUI()
    window.show()
    sys.exit(app.exec())
