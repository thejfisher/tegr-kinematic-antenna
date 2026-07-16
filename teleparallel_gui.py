import sys
import time
import subprocess
import shlex
import json
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox, QTextEdit, QSplitter, QGroupBox, QFormLayout, QScrollArea)
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
                cleanup_proc = subprocess.run(ssh_cleanup, capture_output=True, text=True, timeout=30)
                self.log_signal.emit(f"[CLEANUP] {cleanup_proc.stdout.strip()}")
                
                # 1. Start Buffer Node SINDy Server (hal)
                self.log_signal.emit(f">>> STARTING BUFFER NODE SINDY SERVER ({buffer_user}@{buffer_ip}) <<<")
                
                gal_spin = self.params.get("galactic_spin", 0.0)
                server_cmd = f"cd {buffer_dir} && python3 -u sindy_zmq_server.py --port {buffer_port} --no_llm --galactic_spin {gal_spin}"
                ssh_server_cmd = ["ssh", f"{buffer_user}@{buffer_ip}", server_cmd]
                self.log_signal.emit(f"Server launch: {' '.join(ssh_server_cmd)}")
                server_process = subprocess.Popen(ssh_server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                
                # Immediately start reading from SINDy so its pipe doesn't block while physics runs
                import time as _time
                import queue, threading
                buffer_q = queue.Queue()
                def _buffer_reader(proc, q):
                    try:
                        for line in iter(proc.stdout.readline, ''):
                            q.put(line)
                    except Exception:
                        pass
                    q.put(None)
                
                buffer_thread = threading.Thread(target=_buffer_reader, args=(server_process, buffer_q), daemon=True)
                buffer_thread.start()
                
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
                args_str = " ".join([shlex.quote(arg) for arg in args])
                physics_cmd = f"cd {physics_dir} && python3 -u teleparallel_collider.py {args_str}"
                
                ssh_physics_cmd = ["ssh", f"{physics_user}@{physics_ip}", physics_cmd]
                self.log_signal.emit(f"Physics launch: {' '.join(ssh_physics_cmd)}")
                physics_process = subprocess.Popen(ssh_physics_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                
                # Use a thread-based reader to avoid Windows SSH readline hang.
                # Windows SSH does NOT reliably close stdout when the remote process exits,
                # so iter(readline, '') blocks forever. Instead, we use a queue + timeout.
                import queue, threading
                physics_q = queue.Queue()
                def _physics_reader(proc, q):
                    try:
                        for line in iter(proc.stdout.readline, ''):
                            q.put(line)
                    except Exception:
                        pass
                    q.put(None)  # sentinel: EOF or pipe broken
                
                reader_thread = threading.Thread(target=_physics_reader, args=(physics_process, physics_q), daemon=True)
                reader_thread.start()
                
                physics_done = False
                INACTIVITY_TIMEOUT = 60  # seconds without output before checking if process died
                while not physics_done:
                    try:
                        line = physics_q.get(timeout=INACTIVITY_TIMEOUT)
                    except queue.Empty:
                        # No output for INACTIVITY_TIMEOUT seconds — check if SSH process is still alive
                        if physics_process.poll() is not None:
                            self.log_signal.emit("[DEBUG] SSH physics process exited (returncode={})".format(physics_process.returncode))
                            physics_done = True
                            break
                        self.log_signal.emit("[DEBUG] No physics output for {}s — process still alive, waiting...".format(INACTIVITY_TIMEOUT))
                        continue
                    
                    if line is None:
                        # EOF reached
                        physics_done = True
                        break
                    
                    self.log_signal.emit("[PHYSICS] " + line.strip())
                    # Break on known final-line markers
                    if any(marker in line for marker in [
                        "ZMQ Stream finished",
                        "Trajectory saved",
                        "Next step:",
                        "[FATAL ERROR]",
                        "Sending DONE signal",
                    ]):
                        physics_done = True
                        break
                
                physics_process.stdout.close()
                physics_process.terminate()
                try:
                    physics_process.wait(timeout=5)
                except Exception:
                    physics_process.kill()
                
                self.log_signal.emit("[DEBUG] Physics complete. Proceeding to buffer output...")
                
                self.log_signal.emit(">>> WAITING FOR BUFFER NODE SINDY EXTRACTION TO FINISH <<<")
                math_output = []
                BUFFER_TIMEOUT = 120  # 2 minute max wait for SINDy
                deadline = _time.time() + BUFFER_TIMEOUT
                buffer_done = False
                while not buffer_done and _time.time() < deadline:
                    try:
                        line = buffer_q.get(timeout=5)
                    except queue.Empty:
                        # No output for 5s — check if SSH process is still alive
                        if server_process.poll() is not None:
                            self.log_signal.emit("[DEBUG] SSH buffer process exited (returncode={})".format(server_process.returncode))
                            buffer_done = True
                            break
                        continue
                    
                    if line is None:
                        # EOF reached — server exited cleanly
                        buffer_done = True
                        break
                    
                    self.log_signal.emit("[BUFFER] " + line.strip())
                    math_output.append(line)
                
                if not buffer_done:
                    self.log_signal.emit("[DEBUG] Server timeout (120s). Killing server process.")
                    server_process.kill()
                
                server_process.stdout.close()
                try:
                    server_process.wait(timeout=5)
                except Exception:
                    server_process.kill()
                
                # Fetch the aggregate states from the physics node
                self.log_signal.emit(f"[PIPELINE] Fetching aggregate_states.npz from {physics_user}@{physics_ip}...")
                scp_cmd = ["scp", f"{physics_user}@{physics_ip}:{physics_dir}/aggregate_states.npz", "./aggregate_states.npz"]
                self.log_signal.emit(f"Running: {' '.join(scp_cmd)}")
                try:
                    subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
                    self.log_signal.emit("[PIPELINE] Successfully downloaded aggregate_states.npz")
                except subprocess.CalledProcessError as e:
                    self.log_signal.emit(f"[PIPELINE ERROR] Failed to download aggregate_states.npz: {e.stderr}")

                math_text = "".join(math_output)
                clean_math = ""
                if "Cleaned Math Equations:" in math_text:
                    try:
                        parts = math_text.split("Cleaned Math Equations:")[1].split("========================================")
                        clean_math = parts[0].strip() if len(parts) > 0 else math_text.split("Cleaned Math Equations:")[1].strip()
                        if clean_math:
                            self.math_signal.emit(clean_math)
                        else:
                            self.log_signal.emit("[GUI SINDy Parser] WARNING: clean_math was empty after parsing.")
                    except Exception as e:
                        self.log_signal.emit(f"[GUI SINDy Parser] ERROR: {e}")
                else:
                    self.log_signal.emit("[GUI SINDy Parser] ERROR: 'Cleaned Math Equations:' not found in ZMQ Server output. Server may have crashed.")
                
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
                    try:
                        parts = math_text.split("Cleaned Math Equations:")[1].split("Piping")
                        clean_math = parts[0].strip() if len(parts) > 0 else math_text.split("Cleaned Math Equations:")[1].strip()
                        if clean_math:
                            self.math_signal.emit(clean_math)
                        else:
                            self.log_signal.emit("[GUI SINDy Parser] WARNING: clean_math was empty after parsing local output.")
                    except Exception as e:
                        self.log_signal.emit(f"[GUI SINDy Parser] ERROR during local parsing: {e}")
                else:
                    self.log_signal.emit("[GUI SINDy Parser] ERROR: 'Cleaned Math Equations:' not found in local SINDy output! SINDy may have crashed or timed out.")
            
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
class NoScrollComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()

class TeleparallelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teleparallel Collider — Quantum Field Theory Simulator")
        self.resize(1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0e17; color: #e0e0e0; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
            QGroupBox { 
                color: #00e5ff; font-weight: bold; 
                border: 1px solid #1a2a44; 
                border-radius: 8px;
                margin-top: 14px; 
                padding-top: 14px;
                background-color: #0d1520;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; left: 12px; 
                padding: 2px 8px; 
                background-color: #0a0e17;
                border-radius: 4px;
            }
            QLabel { color: #8899aa; font-size: 12px; }
            QLineEdit, QComboBox { 
                background-color: #111a2e; color: #e0e8f0; 
                border: 1px solid #1a2a44; 
                padding: 6px 10px; border-radius: 6px; 
                selection-background-color: #bb86fc;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #00e5ff;
            }
            QComboBox::drop-down {
                border: none; width: 24px;
            }
            QComboBox::down-arrow {
                image: none; 
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #00e5ff;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #111a2e; color: #e0e8f0;
                border: 1px solid #1a2a44;
                selection-background-color: #1a2a44;
            }
            QPushButton { 
                background-color: #bb86fc; color: #000; 
                font-weight: bold; padding: 10px 16px; 
                border-radius: 6px; border: none;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #d4a5ff; }
            QPushButton:pressed { background-color: #9965f4; }
            QPushButton:disabled { background-color: #1a2a44; color: #445566; }
            QTextEdit { 
                background-color: #060a10; color: #00ff88; 
                font-family: 'Cascadia Code', 'Fira Code', Consolas, monospace; 
                font-size: 12px;
                border: 1px solid #1a2a44; border-radius: 6px; 
                padding: 8px;
            }
            QCheckBox { color: #8899aa; spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #334455; background: #111a2e; }
            QCheckBox::indicator:checked { background: #00e5ff; border-color: #00e5ff; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #0a0e17; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #1a2a44; border-radius: 4px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #2a3a54; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSplitter::handle { background: #1a2a44; }
        """)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # =============================================
        # LEFT PANEL (Controls) — 380px wide
        # =============================================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(380)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        left_scroll.setWidget(left_panel)

        # --- B1. Header ---
        header = QLabel("⟨ TELEPARALLEL COLLIDER ⟩")
        header.setStyleSheet("color: #00e5ff; font-size: 20px; font-weight: bold; font-family: 'Segoe UI', sans-serif; padding: 12px 0; letter-spacing: 3px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(header)

        subtitle = QLabel("Quantum Field Theory Simulator")
        subtitle.setStyleSheet("color: #445566; font-size: 11px; letter-spacing: 2px; margin-bottom: 8px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(subtitle)

        # --- B2. Mode Selector ---
        mode_group = QGroupBox("Experiment Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems([
            "simple-collider",
            "double-slit", "single-edge", "quantum-eraser", "heat-sink-eraser",
            "einstein-stick", "direct-collapse", "gravity-sink",
            "holographic", "holographic-shell", "holographic-ring",
            "antenna", "ads-cft", "head-on", "photoelectric", "qed",
            "3-body-scatter", "3-body-orbit", "pilot-wave", "string-sink"
        ])
        self.mode_combo.setStyleSheet("QComboBox { font-size: 15px; padding: 8px; color: #00e5ff; font-weight: bold; }")
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        left_layout.addWidget(mode_group)

        # --- B3. Preset Selector ---
        preset_group = QGroupBox("Presets")
        preset_group.setCheckable(True)
        preset_group.setChecked(True)
        self.preset_container = QWidget()
        preset_layout_inner = QVBoxLayout(self.preset_container)
        self.preset_combo = NoScrollComboBox()
        self.preset_combo.addItems(["0. Custom", "1. 1-Layer Classical Scatter", "2. 5-Layer Phase Router", "3. 3D Phase Router (5x3)", "4. AMPS Firewall (Gravity Sink)", "5. Direct Collapse (N-Body)", "6. Delayed Choice Quantum Eraser", "7. Delayed Choice (Heat Sink Eraser)", "8. Veneziano Amplitude (Soft Scattering)", "9. Photoelectric Effect", "10. Holographic Entanglement", "11. Holographic Shell (Dark Matter Void)", "12. Holographic Ring (Accretion Void)", "13. Quantum Electrodynamics (QED)", "14. Gravitational Wave (NANOGrav)", "15. GHZ Entanglement (Active Firewall)", "16. High-Gamma Test Preset", "17. The Mark Thompson Experiment", "18. Stochastic Hum (Mark Thompson Phase 2)", "19. Mass Sweep: Electron (0.511 MeV)", "20. Mass Sweep: Pion (134.98 MeV)", "21. Mass Sweep: Kaon (493.68 MeV)", "22. Mass Sweep: Proton (938.27 MeV)", "23. Mass Sweep: 4xProton (3753.05 MeV)", "24. Direct Collapse (N=10)", "25. Single-Slit Control", "26. Doubled Separation (d=12.0)", "27. Einstein's Stick (Classical)", "28. Einstein's Stick (Entangled)", "29. AdS-CFT Correspondence", "30. Pilot Wave (de Broglie-Bohm)", "31. Pilot Wave Double-Slit (Histogram)", "32. Couder Pilot Wave (Memory Bath)", "33. Couder Pilot Wave (Interference Capture)", "34. SR Time Dilation (Proper Time)", "35. Twin Paradox (Tau Comparison)", "36. Kepler Orbit (Newtonian)", "37. Gravity Slide (Terminator)", "38. Couder Pilot Wave (CFL Unlocked c=100)", "39. Bohmian Reverse (PDH 1979 Verification)", "40. Experimental Reverse Validation (Tonomura)", "41. Analytical Wave Initialization (Plane Wave)", "42. MOSFET Corner Diffraction (Leakage)", "43. MOSFET Plane Wave (Corner + dBB)", "44. Simple Collider (Sandbox)", "45. Quantum Tunneling (Pauli Barrier)", "46. Kepler Orbit (Vanishing Sun)", "47. RAE v2.1 Validation Matrix (Batch)", "48. Arndt Sweep 1: Sub-electron (m=0.1)", "49. Arndt Sweep 2: Light (m=0.3)", "50. Arndt Sweep 3: Electron (m=0.511)", "51. Arndt Sweep 4: Transition (m=0.75)", "52. Arndt Sweep 5: 2x Electron (m=1.0)", "53. Arndt Sweep 6: 4x Electron (m=2.0)", "54. Arndt Sweep 7: 10x Electron (m=5.0)", "55. RAE Mass Explorer (Free Slider)", "56. RAE Compensated Sweep (7-Run Batch)"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout_inner.addWidget(self.preset_combo)
        preset_group_layout = QVBoxLayout()
        preset_group_layout.addWidget(self.preset_container)
        preset_group.setLayout(preset_group_layout)
        preset_group.toggled.connect(self.preset_container.setVisible)
        left_layout.addWidget(preset_group)

        # =============================================
        # B4. Parameter Groups (in a scroll-friendly area)
        # =============================================
        self.inputs = {}
        def add_input(label, default_val, layout):
            le = QLineEdit(str(default_val))
            self.inputs[label] = le
            layout.addRow(QLabel(label), le)

        # --- Group 1: Beam & Particles ---
        self.beam_group = QGroupBox("Beam & Particles")
        self.beam_group.setCheckable(True)
        self.beam_group.setChecked(True)
        self.beam_container = QWidget()
        beam_layout = QFormLayout(self.beam_container)
        beam_vbox = QVBoxLayout(self.beam_group)
        beam_vbox.addWidget(self.beam_container)
        self.beam_group.toggled.connect(self.beam_container.setVisible)

        add_input("num_particles", "10000", beam_layout)
        add_input("mass_a", "1.0", beam_layout)
        add_input("mass_b", "1.0", beam_layout)
        add_input("beam_momentum", "5000.0", beam_layout)
        add_input("impact_parameter", "0.5", beam_layout)

        left_layout.addWidget(self.beam_group)

        # --- Group 2: Slit Geometry ---
        self.slit_group = QGroupBox("Slit Geometry")
        self.slit_group.setCheckable(True)
        self.slit_group.setChecked(True)
        self.slit_container = QWidget()
        slit_layout = QFormLayout(self.slit_container)
        slit_vbox = QVBoxLayout(self.slit_group)
        slit_vbox.addWidget(self.slit_container)
        self.slit_group.toggled.connect(self.slit_container.setVisible)

        add_input("slit_width", "4.0", slit_layout)
        add_input("slit_separation", "6.0", slit_layout)
        add_input("num_slits", "2", slit_layout)
        add_input("wall_z_layers", "3", slit_layout)
        add_input("wall_depth", "5", slit_layout)
        add_input("screen_x", "20.0", slit_layout)
        add_input("beam_start_x", "-25.0", slit_layout)

        self.soft_wall_cb = QCheckBox("Soft Wall (Pauli-only — enables tunneling)")
        self.soft_wall_cb.setChecked(False)
        self.soft_wall_cb.setStyleSheet("QCheckBox { color: #ff9800; }")
        slit_layout.addRow(self.soft_wall_cb)

        left_layout.addWidget(self.slit_group)

        # --- Group 3: Force Configuration ---
        self.forces_group = QGroupBox("Force Configuration")
        self.forces_group.setCheckable(True)
        self.forces_group.setChecked(True)
        self.forces_container = QWidget()
        forces_layout = QVBoxLayout(self.forces_container)
        forces_form = QFormLayout()
        forces_layout.addLayout(forces_form)
        forces_vbox = QVBoxLayout(self.forces_group)
        forces_vbox.addWidget(self.forces_container)
        self.forces_group.toggled.connect(self.forces_container.setVisible)

        add_input("pauli", "500.0", forces_form)
        add_input("vacuum", "0.001", forces_form)
        add_input("torsion", "1.0", forces_form)

        # Pauli Power Law
        pauli_power_layout = QHBoxLayout()
        pauli_power_layout.addWidget(QLabel("Pauli Power Law:"))
        self.pauli_power_combo = NoScrollComboBox()
        self.pauli_power_combo.addItems(["1/r\u00b3 (Code/KK)", "1/r\u00b2 (Paper 1)"])
        pauli_power_layout.addWidget(self.pauli_power_combo)
        forces_layout.addLayout(pauli_power_layout)

        self.paper1_exact_cb = QCheckBox("Paper 1 Exact Math (overrides below)")
        self.paper1_exact_cb.setChecked(False)
        self.paper1_exact_cb.stateChanged.connect(self.on_paper1_toggle)
        forces_layout.addWidget(self.paper1_exact_cb)

        self.spin_coupling_cb = QCheckBox("Spin-Vorticity Coupling (\u03c9\u1d62 \u00b7 \u03c9\u2c7c)")
        self.spin_coupling_cb.setChecked(False)
        forces_layout.addWidget(self.spin_coupling_cb)

        self.thermal_bath_cb = QCheckBox("Wall Thermal Bath (Random Phase)")
        self.thermal_bath_cb.setChecked(False)
        forces_layout.addWidget(self.thermal_bath_cb)

        self.qed_vacpol_cb = QCheckBox("Vacuum Polarization (Uehling)")
        self.qed_vacpol_cb.setChecked(False)
        forces_layout.addWidget(self.qed_vacpol_cb)

        self.qed_lamb_cb = QCheckBox("Lamb Shift (Self-Energy Jitter)")
        self.qed_lamb_cb.setChecked(False)
        forces_layout.addWidget(self.qed_lamb_cb)

        self.qed_compton_cb = QCheckBox("Compton Scattering (Radiation Wave)")
        self.qed_compton_cb.setChecked(False)
        forces_layout.addWidget(self.qed_compton_cb)

        left_layout.addWidget(self.forces_group)

        # --- Group 4: Wave & Pilot Wave ---
        self.wave_group = QGroupBox("Wave & Pilot Wave")
        self.wave_group.setCheckable(True)
        self.wave_group.setChecked(True)
        self.wave_container = QWidget()
        wave_layout = QVBoxLayout(self.wave_container)
        wave_form = QFormLayout()
        wave_layout.addLayout(wave_form)
        wave_vbox = QVBoxLayout(self.wave_group)
        wave_vbox.addWidget(self.wave_container)
        self.wave_group.toggled.connect(self.wave_container.setVisible)

        add_input("wave_speed", "100.0", wave_form)
        add_input("wave_dissipation", "0.999", wave_form)
        add_input("jitter_amp", "0.0", wave_form)
        add_input("photon_freq", "100.0", wave_form)

        self.photon_emission_cb = QCheckBox("Continuous Photon Emission (Pilot Wave)")
        self.photon_emission_cb.setChecked(False)
        wave_layout.addWidget(self.photon_emission_cb)

        self.pilot_wave_cb = QCheckBox("Pilot Wave Force (Gradient Coupling)")
        self.pilot_wave_cb.setChecked(False)
        wave_layout.addWidget(self.pilot_wave_cb)

        self.dbb_guidance_cb = QCheckBox("dBB Wave Guidance (Velocity Override)")
        self.dbb_guidance_cb.setChecked(False)
        wave_layout.addWidget(self.dbb_guidance_cb)

        dbb_strength_layout = QHBoxLayout()
        dbb_strength_layout.addWidget(QLabel("dBB Strength:"))
        self.dbb_strength_input = QLineEdit("0.01")
        self.dbb_strength_input.setFixedWidth(80)
        dbb_strength_layout.addWidget(self.dbb_strength_input)
        wave_layout.addLayout(dbb_strength_layout)

        self.reverse_integration_cb = QCheckBox("Reverse Integration (PDH 1979 Verification)")
        self.reverse_integration_cb.setChecked(False)
        wave_layout.addWidget(self.reverse_integration_cb)

        reverse_particles_layout = QHBoxLayout()
        reverse_particles_layout.addWidget(QLabel("Reverse Particles:"))
        self.reverse_num_particles_input = QLineEdit("500")
        self.reverse_num_particles_input.setFixedWidth(80)
        reverse_particles_layout.addWidget(self.reverse_num_particles_input)
        wave_layout.addLayout(reverse_particles_layout)

        band_source_layout = QHBoxLayout()
        band_source_layout.addWidget(QLabel("Band Source:"))
        self.band_source_input = QLineEdit("kde")
        self.band_source_input.setFixedWidth(120)
        self.band_source_input.setToolTip("kde = from screen hits, analytical = diffraction eq, file:path.npy = external data")
        band_source_layout.addWidget(self.band_source_input)
        wave_layout.addLayout(band_source_layout)

        self.crossing_diagnostics_cb = QCheckBox("Crossing Diagnostics (angle, velocity, slit position)")
        self.crossing_diagnostics_cb.setChecked(False)
        wave_layout.addWidget(self.crossing_diagnostics_cb)

        self.plane_wave_init_cb = QCheckBox("Initialize Custom Wave (Pilot)")
        self.plane_wave_init_cb.setChecked(False)
        wave_layout.addWidget(self.plane_wave_init_cb)

        wave_shape_layout = QHBoxLayout()
        wave_shape_layout.addWidget(QLabel("Wave Shape:"))
        self.wave_shape_combo = NoScrollComboBox()
        self.wave_shape_combo.addItems(["plane", "gaussian", "spherical"])
        self.wave_shape_combo.setToolTip("Geometry of the initialized wave.")
        wave_shape_layout.addWidget(self.wave_shape_combo)
        wave_layout.addLayout(wave_shape_layout)

        wave_freq_layout = QHBoxLayout()
        wave_freq_layout.addWidget(QLabel("Wave Freq (0=Auto):"))
        self.wave_freq_input = QLineEdit("0.0")
        self.wave_freq_input.setToolTip("Wave frequency (k). If 0.0, locks to beam_momentum.")
        self.wave_freq_input.setFixedWidth(80)
        wave_freq_layout.addWidget(self.wave_freq_input)
        wave_layout.addLayout(wave_freq_layout)

        wave_amp_layout = QHBoxLayout()
        wave_amp_layout.addWidget(QLabel("Wave Amplitude:"))
        self.plane_wave_amp_input = QLineEdit("2.0")
        self.plane_wave_amp_input.setToolTip("Plane wave amplitude. Sweet spot ~1-3. Too high = standing wave trap.")
        self.plane_wave_amp_input.setFixedWidth(80)
        wave_amp_layout.addWidget(self.plane_wave_amp_input)
        wave_layout.addLayout(wave_amp_layout)

        self.breit_wheeler_cb = QCheckBox("Breit-Wheeler Pair Production (RVC)")
        self.breit_wheeler_cb.setChecked(False)
        wave_layout.addWidget(self.breit_wheeler_cb)

        bw_threshold_layout = QHBoxLayout()
        bw_threshold_layout.addWidget(QLabel("BW Threshold (MeV):"))
        self.bw_threshold_input = QLineEdit("2.044")
        self.bw_threshold_input.setFixedWidth(80)
        bw_threshold_layout.addWidget(self.bw_threshold_input)
        wave_layout.addLayout(bw_threshold_layout)

        left_layout.addWidget(self.wave_group)

        # --- Group 5: Which-Path Detector (NEW) ---
        self.detector_group = QGroupBox("Which-Path Detector")
        self.detector_group.setCheckable(True)
        self.detector_group.setChecked(True)
        self.detector_group.setStyleSheet("QGroupBox { color: #ff9800; border-color: #3d2800; }")
        self.detector_container = QWidget()
        detector_layout = QVBoxLayout(self.detector_container)
        detector_vbox = QVBoxLayout(self.detector_group)
        detector_vbox.addWidget(self.detector_container)
        self.detector_group.toggled.connect(self.detector_container.setVisible)

        self.which_path_cb = QCheckBox("Enable Which-Path Detection")
        self.which_path_cb.setChecked(False)
        detector_layout.addWidget(self.which_path_cb)

        det_mass_layout = QHBoxLayout()
        det_mass_layout.addWidget(QLabel("Detector Mass:"))
        self.detector_mass_input = QLineEdit("0.001")
        self.detector_mass_input.setFixedWidth(80)
        det_mass_layout.addWidget(self.detector_mass_input)
        detector_layout.addLayout(det_mass_layout)

        det_phase_layout = QHBoxLayout()
        det_phase_layout.addWidget(QLabel("Detector Phase:"))
        self.detector_phase_combo = NoScrollComboBox()
        self.detector_phase_combo.addItems(["thermal", "coherent"])
        det_phase_layout.addWidget(self.detector_phase_combo)
        detector_layout.addLayout(det_phase_layout)

        left_layout.addWidget(self.detector_group)

        # --- Group 6: Gravity & Collapse ---
        self.gravity_group = QGroupBox("Gravity & Collapse")
        self.gravity_group.setCheckable(True)
        self.gravity_group.setChecked(True)
        self.gravity_container = QWidget()
        gravity_layout = QFormLayout(self.gravity_container)
        gravity_vbox = QVBoxLayout(self.gravity_group)
        gravity_vbox.addWidget(self.gravity_container)
        self.gravity_group.toggled.connect(self.gravity_container.setVisible)

        add_input("sink_mass", "50000.0", gravity_layout)
        add_input("collapse_radius", "20.0", gravity_layout)
        add_input("collapse_G", "1.0", gravity_layout)
        add_input("amps_cooling_cap", "1.0", gravity_layout)

        self.vanish_sink_cb = QCheckBox("Vanish Sink at 50% Time (Einstein vs Newton)")
        self.vanish_sink_cb.setChecked(False)
        self.vanish_sink_cb.setStyleSheet("QCheckBox { color: #f44336; }")
        gravity_layout.addRow(self.vanish_sink_cb)

        self.fdtd_gravity_cb = QCheckBox("FDTD Gravity (Relativistic Delay)")
        self.fdtd_gravity_cb.setChecked(False)
        self.fdtd_gravity_cb.setStyleSheet("QCheckBox { color: #2196F3; }")
        gravity_layout.addRow(self.fdtd_gravity_cb)

        left_layout.addWidget(self.gravity_group)

        # --- Group 6b: Relativistic Adler Equation (RAE) ---
        self.rae_group = QGroupBox("Relativistic Adler Equation (RAE)")
        self.rae_group.setCheckable(True)
        self.rae_group.setChecked(True)
        self.rae_group.setStyleSheet("QGroupBox { color: #FF9800; border-color: #3d2800; }")
        self.rae_container = QWidget()
        rae_layout = QFormLayout(self.rae_container)
        rae_vbox = QVBoxLayout(self.rae_group)
        rae_vbox.addWidget(self.rae_container)
        self.rae_group.toggled.connect(self.rae_container.setVisible)

        self.rae_mode_cb = QCheckBox("Enable RAE Phase Override")
        self.rae_mode_cb.setChecked(False)
        self.rae_mode_cb.setStyleSheet("QCheckBox { color: #FF9800; font-weight: bold; }")
        rae_layout.addRow(self.rae_mode_cb)
        self.inputs["rae_kappa_scale"] = QLineEdit("1.0")
        self.inputs["rae_grad_scale"] = QLineEdit("1.0")
        rae_layout.addRow("RAE kappa Scale:", self.inputs["rae_kappa_scale"])
        rae_layout.addRow("RAE grad Scale:", self.inputs["rae_grad_scale"])

        left_layout.addWidget(self.rae_group)

        # --- Group 7: Entanglement & Coupling ---
        self.entangle_group = QGroupBox("Entanglement & Coupling")
        self.entangle_group.setCheckable(True)
        self.entangle_group.setChecked(True)
        self.entangle_container = QWidget()
        entangle_layout = QVBoxLayout(self.entangle_container)
        entangle_form = QFormLayout()
        entangle_layout.addLayout(entangle_form)
        entangle_vbox = QVBoxLayout(self.entangle_group)
        entangle_vbox.addWidget(self.entangle_container)
        self.entangle_group.toggled.connect(self.entangle_container.setVisible)

        add_input("entangled", "0", entangle_form)

        self.eraser_active_cb = QCheckBox("Eraser Active (No Measurement Wall)")
        self.eraser_active_cb.setChecked(False)
        entangle_layout.addWidget(self.eraser_active_cb)

        left_layout.addWidget(self.entangle_group)

        # --- Group 8: Antenna & Cosmology ---
        self.antenna_group = QGroupBox("Antenna & Cosmology")
        self.antenna_group.setCheckable(True)
        self.antenna_group.setChecked(True)
        self.antenna_container = QWidget()
        antenna_layout = QVBoxLayout(self.antenna_container)
        antenna_form = QFormLayout()
        antenna_layout.addLayout(antenna_form)
        antenna_vbox = QVBoxLayout(self.antenna_group)
        antenna_vbox.addWidget(self.antenna_container)
        self.antenna_group.toggled.connect(self.antenna_container.setVisible)

        add_input("num_anchors", "1", antenna_form)
        add_input("antenna_file", "", antenna_form)
        add_input("galactic_spin", "0.0", antenna_form)

        polarization_layout = QHBoxLayout()
        polarization_layout.addWidget(QLabel("Gravitational Polarization:"))
        self.polarization_combo = NoScrollComboBox()
        self.polarization_combo.addItems(["isotropic", "plus", "cross", "mixed"])
        polarization_layout.addWidget(self.polarization_combo)
        antenna_layout.addLayout(polarization_layout)

        left_layout.addWidget(self.antenna_group)

        # --- Group 9: Simulation Control (always visible) ---
        self.sim_group = QGroupBox("Simulation Control")
        self.sim_group.setCheckable(True)
        self.sim_group.setChecked(True)
        self.sim_container = QWidget()
        sim_layout = QFormLayout(self.sim_container)
        sim_vbox = QVBoxLayout(self.sim_group)
        sim_vbox.addWidget(self.sim_container)
        self.sim_group.toggled.connect(self.sim_container.setVisible)

        add_input("dt", "0.001", sim_layout)
        add_input("total_ticks", "5000", sim_layout)
        add_input("batch_size", "0", sim_layout)
        add_input("plot_dot_size", "2.0", sim_layout)

        # Hidden inputs that presets reference but don't have their own visible group
        add_input("work_function", "10.0", sim_layout)

        left_layout.addWidget(self.sim_group)

        # =============================================
        # Execution Engine (Tri-Node Pipeline)
        # =============================================
        exec_group = QGroupBox("Execution Engine (Tri-Node Pipeline)")
        exec_group.setCheckable(True)
        exec_group.setChecked(True)
        self.exec_container = QWidget()
        exec_layout = QFormLayout(self.exec_container)
        
        self.pipeline_checkbox = QCheckBox("Enable ZMQ Ping-Pong Distributed Pipeline")
        self.pipeline_checkbox.setChecked(True)
        
        self.physics_ip_input = QLineEdit("100.122.147.67")
        self.physics_user_input = QLineEdit("thejfisher")
        self.physics_dir_input = QLineEdit("~/AI_Vault/teleparallel_sim_photons")
        
        self.buffer_ip_input = QLineEdit("100.66.100.83")
        self.buffer_user_input = QLineEdit("hal")
        self.buffer_port_input = QLineEdit("7777")
        self.buffer_dir_input = QLineEdit("~/hxseq-vsgx4/teleparallel_sim_photons")
        
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

        # =============================================
        # Button Row (Run & Load)
        # =============================================
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("RUN SIMULATION")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("QPushButton { background-color: #00e5ff; color: #000; font-size: 16px; font-weight: bold; padding: 12px; border-radius: 8px; } QPushButton:hover { background-color: #33eeff; } QPushButton:disabled { background-color: #1a2a44; color: #445566; }")
        btn_layout.addWidget(self.run_btn)
        
        self.load_btn = QPushButton("LOAD LAST RUN")
        self.load_btn.clicked.connect(self.load_last_run)
        self.load_btn.setStyleSheet("QPushButton { background-color: #cf6679; color: #000; font-size: 16px; font-weight: bold; padding: 12px; border-radius: 8px; } QPushButton:hover { background-color: #e88899; }")
        btn_layout.addWidget(self.load_btn)
        
        left_layout.addLayout(btn_layout)
        
        # --- Batch Queue Status ---
        self.batch_queue = []  # List of dicts: [{"label": ..., "setup_fn": callable}, ...]
        self.batch_running = False
        self._rae_slider_active = False  # Free slider auto-compute flag
        self.batch_label = QLabel("")
        self.batch_label.setStyleSheet("QLabel { color: #FF9800; font-weight: bold; font-size: 13px; }")
        self.batch_label.setWordWrap(True)
        self.batch_label.setVisible(False)
        left_layout.addWidget(self.batch_label)
        
        # 3D Viewer Button
        self.view_3d_btn = QPushButton("OPEN 3D VIEWER")
        self.view_3d_btn.clicked.connect(self.open_3d_viewer)
        self.view_3d_btn.setStyleSheet("QPushButton { background-color: #bb86fc; color: #000; font-size: 16px; font-weight: bold; padding: 12px; border-radius: 8px; margin-top: 8px; } QPushButton:hover { background-color: #d4a5ff; }")
        left_layout.addWidget(self.view_3d_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_scroll)

        # =============================================
        # RIGHT PANEL (Visualization & Logs)
        # =============================================
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top right: Graphics
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('#0a0e17')
        self.plot_spatial = self.graph_widget.addPlot(title="Spatial Y-Distribution (Histogram)")
        self.plot_spatial.showGrid(x=True, y=True, alpha=0.3)
        self.plot_spatial.setLabel('left', 'Hits')
        self.plot_spatial.setLabel('bottom', 'Screen Y Position')

        self.plot_phase = self.graph_widget.addPlot(title="Phase Router (Hue vs Final Y)")
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.plot_phase.setLabel('left', 'Final Y Position')
        self.plot_phase.setLabel('bottom', 'Final Hidden Phase (Hue)')
        
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

        # Set initial mode visibility
        self.on_mode_changed(self.mode_combo.currentText())

    # ==========================================
    # Mode Visibility Map
    # ==========================================
    def on_mode_changed(self, mode_text):
        """Show/hide parameter groups based on selected mode."""
        mode_groups = {
            "double-slit": ["beam", "slit", "forces", "wave", "detector", "entangle", "sim"],
            "single-edge": ["beam", "slit", "forces", "wave", "sim"],
            "quantum-eraser": ["beam", "slit", "forces", "wave", "entangle", "sim"],
            "heat-sink-eraser": ["beam", "slit", "forces", "wave", "entangle", "sim"],
            "einstein-stick": ["beam", "forces", "gravity", "entangle", "sim"],
            "direct-collapse": ["beam", "forces", "gravity", "sim"],
            "gravity-sink": ["beam", "forces", "gravity", "entangle", "sim"],
            "holographic": ["beam", "forces", "gravity", "entangle", "sim"],
            "holographic-shell": ["beam", "forces", "gravity", "entangle", "sim"],
            "holographic-ring": ["beam", "forces", "gravity", "entangle", "sim"],
            "antenna": ["beam", "forces", "antenna", "sim"],
            "ads-cft": ["beam", "forces", "entangle", "sim"],
            "head-on": ["beam", "forces", "sim"],
            "photoelectric": ["beam", "forces", "wave", "sim"],
            "qed": ["beam", "forces", "wave", "sim"],
            "3-body-scatter": ["beam", "forces", "sim"],
            "3-body-orbit": ["beam", "forces", "sim"],
            "pilot-wave": ["beam", "slit", "forces", "wave", "sim"],
            "string-sink": ["beam", "forces", "gravity", "sim"],
            "simple-collider": ["beam", "forces", "sim"],
        }
        
        all_groups = ["beam", "slit", "forces", "wave", "detector", "gravity", "entangle", "antenna"]
        visible = mode_groups.get(mode_text, all_groups)
        
        group_map = {
            "beam": self.beam_group,
            "slit": self.slit_group,
            "forces": self.forces_group,
            "wave": self.wave_group,
            "detector": self.detector_group,
            "gravity": self.gravity_group,
            "entangle": self.entangle_group,
            "antenna": self.antenna_group,
        }
        for name, group in group_map.items():
            group.setVisible(name in visible)

    # ==========================================
    # --- RAE Mass Explorer: Auto-compute derived parameters ---
    def auto_compute_rae_from_mass(self):
        """When RAE Mass Explorer is active, auto-fill momentum/kappa/grad/amp
        from the current mass_a value using the theoretical RAE linear scaling."""
        if not self._rae_slider_active:
            return
        try:
            m = float(self.inputs["mass_a"].text())
            if m <= 0:
                return
        except (ValueError, AttributeError):
            return
            
        # Unglue beam momentum from mass. Read it instead so we can calculate diagnostics.
        try:
            p = float(self.inputs["beam_momentum"].text())
        except (ValueError, AttributeError):
            p = 10.0
            
        # Theoretical RAE Scaling Law (Anchor to electron mass)
        m_e = 0.511
        ratio = m / m_e
        
        # Auto-couple wave speed: The wave slows down precisely with the particle's velocity (v=p/m)
        base_wave_speed = 65.0
        new_wave_speed = base_wave_speed * (m_e / m)

        self._is_updating = True
        try:
            self.inputs["rae_kappa_scale"].setText(f"{ratio:.4f}")
            self.inputs["rae_grad_scale"].setText(f"{ratio:.4f}")
            
            # The dBB force scales with C^2 (which scales as 1/m^2). 
            # To conserve the exact transverse momentum deflection (so the fringes don't shrink),
            # we must scale the wave amplitude by m^2 to perfectly offset it.
            self.plane_wave_amp_input.setText(f"{20.0 * (ratio**2):.2f}")
            
            self.inputs["wave_speed"].setText(f"{new_wave_speed:.4f}")
        finally:
            self._is_updating = False

        # Timestep: resolve the faster clock (need ~10 steps per cycle)
        dt_max = min(0.001, 2 * 3.14159 / (m * 10))
        self.inputs["dt"].setText(f"{dt_max:.6f}")
        # Diagnostic readout
        lam = 2 * 3.14159 / p
        lt = 36.0 / lam  # d^2=6^2=36
        xi = 20.0 / lt
        self.batch_label.setText(
            f"RAE Explorer: m={m:.4f} MeV | cw={new_wave_speed:.2f} | λ={lam:.4f} | ξ={xi:.4f} | κ_s={ratio:.2f}")
        self.batch_label.setVisible(True)

    # Preset Logic (all 43 presets)
    # ==========================================
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
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["dt"].setText("0.02")
        elif text == "5-Layer Phase Router":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("5")
            self.inputs["dt"].setText("0.02")
        elif text == "3D Phase Router (5x3)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("3")
            self.inputs["wall_depth"].setText("5")
            self.inputs["entangled"].setText("0")
        elif text == "AMPS Firewall (Gravity Sink)":
            # --- Core physics ---
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("50000.0")
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.inputs["dt"].setText("0.02")
            # --- Coupling & toggles (previously inherited) ---
            self.spin_coupling_cb.setChecked(True)
            self.eraser_active_cb.setChecked(False)
            self.qed_vacpol_cb.setChecked(False)
            self.qed_lamb_cb.setChecked(False)
            self.qed_compton_cb.setChecked(False)
            # --- Gravity-sink parameters ---
            self.inputs["collapse_radius"].setText("10.0")  # Particle A at r=10
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")    # Tangential py=20
            self.inputs["impact_parameter"].setText("5.0")   # Particle B at r=50
            self.inputs["amps_cooling_cap"].setText("1.0")
            # --- Slit geometry (not used in gravity-sink but set explicitly for hygiene) ---
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["num_slits"].setText("2")
            self.inputs["wall_z_layers"].setText("3")
            self.inputs["wall_depth"].setText("5")
            # --- Timing & misc ---
            self.inputs["total_ticks"].setText("5000")
            self.inputs["num_anchors"].setText("1")
            self.inputs["galactic_spin"].setText("0.0")
        elif text == "Direct Collapse (N-Body)":
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("direct-collapse")
            self.inputs["num_particles"].setText("50")
            self.inputs["mass_a"].setText("100.0")
            self.inputs["mass_b"].setText("100.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.01")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.inputs["dt"].setText("0.02")
        elif text == "Delayed Choice Quantum Eraser":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("quantum-eraser")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["entangled"].setText("1")
            self.inputs["dt"].setText("0.02")
            self.eraser_active_cb.setChecked(False)
        elif text == "Delayed Choice (Heat Sink Eraser)":
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("heat-sink-eraser")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["entangled"].setText("0")
            self.inputs["dt"].setText("0.02")
        elif text == "Veneziano Amplitude (Soft Scattering)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3 KK
            self.mode_combo.setCurrentText("head-on")
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
            self.mode_combo.setCurrentText("photoelectric")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("5000.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["beam_momentum"].setText("5000.0")
            self.inputs["photon_freq"].setText("500.0")
            self.inputs["work_function"].setText("1000.0")
            self.inputs["entangled"].setText("0")
            self.inputs["dt"].setText("0.000001")
            self.inputs["total_ticks"].setText("100000")
            self.inputs["zmq_flush_rate"] = QLineEdit("10") # Prevent ZMQ connection timeout
            
        elif text == "Holographic Entanglement":
            self.mode_combo.setCurrentText("holographic")
            self.inputs["num_particles"].setText("50")
            self.inputs["collapse_radius"].setText("50.0")
            self.inputs["entangled"].setText("300")
            self.inputs["collapse_G"].setText("10.0")
            
        elif text == "Holographic Shell (Dark Matter Void)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0) # 1/r^3
            self.mode_combo.setCurrentText("holographic-shell")
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
            self.mode_combo.setCurrentText("holographic-ring")
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
            self.mode_combo.setCurrentText("qed")
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
            self.mode_combo.setCurrentText("antenna")
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
            self.mode_combo.setCurrentText("direct-collapse")
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
            self.pauli_power_combo.setCurrentIndex(1) # 1/r^2
        elif text == "High-Gamma Test Preset":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["beam_momentum"].setText("50.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("5000")
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(1)
        elif text == "The Mark Thompson Experiment":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("antenna")
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
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
        elif text == "Stochastic Hum (Mark Thompson Phase 2)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("antenna")
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
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.inputs["antenna_file"].setText("pta_residuals.csv") # Stochastic Hum Injection
        elif text == "Mass Sweep: Electron (0.511 MeV)":
            # Paper 3, Section 6: Mass-Dependent Dynamical Complexity
            # Gravity-sink with M=25,000, entangled pair, 1/r^3 KK
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("25000.0")
            self.inputs["collapse_radius"].setText("10.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["dt"].setText("0.02")
        elif text == "Mass Sweep: Pion (134.98 MeV)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("134.98")
            self.inputs["mass_b"].setText("134.98")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("25000.0")
            self.inputs["collapse_radius"].setText("10.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["dt"].setText("0.02")
        elif text == "Mass Sweep: Kaon (493.68 MeV)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("493.68")
            self.inputs["mass_b"].setText("493.68")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("25000.0")
            self.inputs["collapse_radius"].setText("10.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["dt"].setText("0.02")
        elif text == "Mass Sweep: Proton (938.27 MeV)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("938.27")
            self.inputs["mass_b"].setText("938.27")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("25000.0")
            self.inputs["collapse_radius"].setText("10.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["dt"].setText("0.02")
        elif text == "Mass Sweep: 4xProton (3753.05 MeV)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("3753.05")
            self.inputs["mass_b"].setText("3753.05")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("25000.0")
            self.inputs["collapse_radius"].setText("10.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("20.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["dt"].setText("0.02")
        elif text == "Direct Collapse (N=10)":
            # Paper 3, Section 7.3: N-Dependent Complexity comparison
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("direct-collapse")
            self.inputs["num_particles"].setText("10")
            self.inputs["mass_a"].setText("100.0")
            self.inputs["mass_b"].setText("100.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.01")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 Kaluza-Klein
            self.inputs["dt"].setText("0.02")
        elif text == "Single-Slit Control":
            # Paper 3, Section 8: Aperture Geometry control experiment
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["dt"].setText("0.02")
            self.inputs["num_slits"].setText("1")  # Single slit
        elif text == "Doubled Separation (d=12.0)":
            # Paper 3, Section 8: Slit separation variation
            self.paper1_exact_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("12.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["dt"].setText("0.02")
        elif text == "Einstein's Stick (Classical)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("einstein-stick")
            self.inputs["num_particles"].setText("50")  # 1 hammer, 49 stick
            self.inputs["mass_a"].setText("10.0")      # Hammer mass
            self.inputs["mass_b"].setText("1.0")       # Stick mass
            self.inputs["beam_momentum"].setText("5000.0") # Hammer strike momentum
            self.inputs["work_function"].setText("50000.0") # Hooke's k for the stick (very stiff)
            self.inputs["slit_separation"].setText("0.1")  # Rest distance d0
            self.inputs["pauli"].setText("5000.0")     # High repulsion to prevent pass-through
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("0")
            self.inputs["dt"].setText("0.0001")        # High res dt to resolve stiff stick
            self.inputs["total_ticks"].setText("10000")
            self.pauli_power_combo.setCurrentIndex(1)  # 1/r^2 Inverse Square
        elif text == "Einstein's Stick (Entangled)":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("einstein-stick")
            self.inputs["num_particles"].setText("50")  # 1 hammer, 49 stick
            self.inputs["mass_a"].setText("10.0")      # Hammer mass
            self.inputs["mass_b"].setText("1.0")       # Stick mass
            self.inputs["beam_momentum"].setText("5000.0") # Hammer strike momentum
            self.inputs["work_function"].setText("50000.0") # Hooke's k for the stick (very stiff)
            self.inputs["slit_separation"].setText("0.1")  # Rest distance d0
            self.inputs["pauli"].setText("5000.0")     # High repulsion to prevent pass-through
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")      # Enable sync tensor (holographic stick)
            self.inputs["dt"].setText("0.0001")
            self.inputs["total_ticks"].setText("10000")
            self.pauli_power_combo.setCurrentIndex(1)  # 1/r^2 Inverse Square
        elif text == "RAE v2.1 Validation Matrix (Batch)":
            # Queue 8 consecutive Einstein's Stick runs covering the full RAE matrix:
            # 4 quadrants × 2 Pauli powers = 8 runs total.
            # Each run is an (entangled, coupling, pauli_power, rae_mode) configuration.
            self.batch_queue = []
            matrix = [
                # (label_suffix, entangled, spin_coupling, pauli_power_idx, rae_mode)
                ("Baseline Q1 (1/r²): Ent=0 Coup=0", "0", False, 1, 0),
                ("Baseline Q2 (1/r²): Ent=0 Coup=1", "0", True,  1, 0),
                ("Baseline Q3 (1/r²): Ent=1 Coup=0", "1", False, 1, 0),
                ("Baseline Q4 (1/r²): Ent=1 Coup=1", "1", True,  1, 0),
                ("RAE Q1 (1/r²): Ent=0 Coup=0",      "0", False, 1, 1),
                ("RAE Q2 (1/r²): Ent=0 Coup=1",      "0", True,  1, 1),
                ("RAE Q3 (1/r²): Ent=1 Coup=0",      "1", False, 1, 1),
                ("RAE Q4 (1/r²): Ent=1 Coup=1",      "1", True,  1, 1),
                ("Baseline Q1 (1/r³): Ent=0 Coup=0", "0", False, 0, 0),
                ("Baseline Q2 (1/r³): Ent=0 Coup=1", "0", True,  0, 0),
                ("Baseline Q3 (1/r³): Ent=1 Coup=0", "1", False, 0, 0),
                ("Baseline Q4 (1/r³): Ent=1 Coup=1", "1", True,  0, 0),
                ("RAE Q1 (1/r³): Ent=0 Coup=0",      "0", False, 0, 1),
                ("RAE Q2 (1/r³): Ent=0 Coup=1",      "0", True,  0, 1),
                ("RAE Q3 (1/r³): Ent=1 Coup=0",      "1", False, 0, 1),
                ("RAE Q4 (1/r³): Ent=1 Coup=1",      "1", True,  0, 1),
            ]
            for label_suffix, ent, coup, pp_idx, rae in matrix:
                def make_setup(e=ent, c=coup, p=pp_idx, r=rae, lbl=label_suffix):
                    def setup():
                        self.paper1_exact_cb.setChecked(False)
                        self.spin_coupling_cb.setChecked(c)
                        self.thermal_bath_cb.setChecked(False)
                        self.mode_combo.setCurrentText("einstein-stick")
                        self.inputs["num_particles"].setText("50")
                        self.inputs["mass_a"].setText("10.0")
                        self.inputs["mass_b"].setText("1.0")
                        self.inputs["beam_momentum"].setText("5000.0")
                        self.inputs["work_function"].setText("50000.0")
                        self.inputs["slit_separation"].setText("0.1")
                        self.inputs["pauli"].setText("5000.0")
                        self.inputs["vacuum"].setText("0.001")
                        self.inputs["torsion"].setText("1.0")
                        self.inputs["entangled"].setText(e)
                        self.inputs["dt"].setText("0.0001")
                        self.inputs["total_ticks"].setText("10000")
                        self.pauli_power_combo.setCurrentIndex(p)
                        self.rae_mode_cb.setChecked(bool(r))
                    return setup
                self.batch_queue.append({"label": f"RAE Matrix: {label_suffix}", "setup_fn": make_setup()})
            
            # Apply the first run's settings immediately for preview
            if self.batch_queue:
                self._batch_total = len(self.batch_queue)
                self.batch_queue[0]["setup_fn"]()
                self.batch_label.setText(f"BATCH QUEUED: {len(self.batch_queue)} runs | Next: {self.batch_queue[0]['label']}")
                self.batch_label.setVisible(True)
        elif text == "AdS-CFT Correspondence":
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("ads-cft")
            self.inputs["num_particles"].setText("100")  # 30 boundary + 70 bulk
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["num_anchors"].setText("30")     # 30 particles on the boundary (CFT)
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("100.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.01")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("300")      # 300 boundary<->bulk ER=EPR tethers
            self.inputs["dt"].setText("0.01")
            self.inputs["total_ticks"].setText("5000")
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 KK force
        elif text == "Pilot Wave Double-Slit (Histogram)":
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("1.0")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("3")
            self.inputs["wall_depth"].setText("5")
            self.inputs["num_slits"].setText("2")
            self.inputs["beam_momentum"].setText("50.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(1)
            self.photon_emission_cb.setChecked(True)
        elif text == "Pilot Wave (de Broglie-Bohm)":
            # dBB guidance: particle velocity from torsion grid gradient
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.mode_combo.setCurrentText("pilot-wave")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("500.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("0.5")
            self.inputs["slit_separation"].setText("9.6")
            self.inputs["wall_depth"].setText("1")
            self.inputs["beam_momentum"].setText("5000.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["entangled"].setText("0")
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 KK force
            self.eraser_active_cb.setChecked(False)
        elif text == "Couder Pilot Wave (Memory Bath)":
            # Re-activating FDTD Pilot Wave to steer particles across slits
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")       # Fine-Structure Constant Alpha
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("200.0")
            self.inputs["beam_momentum"].setText("2.28") 
            self.inputs["dt"].setText("0.005")
            self.inputs["total_ticks"].setText("150000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("0.999")
            self.inputs["jitter_amp"].setText("2.5")    # Zitterbewegung "lob pass" amplitude
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)       # Spin coupling ON
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)  # 1/r^3 force (Code/KK)
            self.photon_emission_cb.setChecked(True)   # Turn ON FDTD Grid!
            self.breit_wheeler_cb.setChecked(False)    # No Breit-Wheeler pair production
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)        # Turn ON Pilot Wave Force!
            self.dbb_guidance_cb.setChecked(True)     # Turn ON strict DBB override
            self.dbb_strength_input.setText("0.00005")
        elif text == "Couder Pilot Wave (Interference Capture)":
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("8.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("250.0")
            self.inputs["beam_momentum"].setText("2.28") 
            self.inputs["dt"].setText("0.005")
            self.inputs["total_ticks"].setText("60000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("0.999")
            self.inputs["jitter_amp"].setText("2.5")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(True)
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.002")
        elif text == "Couder Pilot Wave (CFL Unlocked c=100)":
            # PRESET 38: Identical to Preset 33 except:
            #   screen_x:  250 -> 50  (far-field: 4.2x more flight time for dBB steering)
            #   dt:        0.005 -> 0.001  (CFL limit = 0.5*0.25/0.001 = 125, accepts c=100)
            #   ticks:     60000 -> 20000  (particle arrives in ~11200 ticks at v=4.46)
            #   Result: Longer post-slit flight lets gentle dBB accumulate transverse deflection
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")
            self.inputs["slit_separation"].setText("8.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("15.0")
            self.inputs["beam_momentum"].setText("2.28")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("20000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("0.999")
            self.inputs["jitter_amp"].setText("2.5")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(True)
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
        elif text == "Bohmian Reverse (PDH 1979 Verification)":
            # PRESET 39: Time-Reversed Bohmian Trajectory Verification
            # Based on Philippidis, Dewdney & Hiley (1979)
            # Runs the standard double-slit forward at Half-Talbot distance,
            # then reverses the FDTD wave and integrates test particles backward
            # through the slits to verify the Bohmian non-crossing theorem.
            # CRITICAL: wave_dissipation=1.0 (time-reversible), jitter_amp=0.0 (no noise)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("6.53")
            self.inputs["beam_momentum"].setText("2.28")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("20000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("1.0")       # MUST be 1.0 for time-reversibility
            self.inputs["jitter_amp"].setText("0.0")             # MUST be 0.0 (no stochastic noise)
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(True)
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(True)          # ENABLE Phase B
            self.reverse_num_particles_input.setText("500")
        elif text == "Experimental Reverse Validation (Tonomura)":
            # PRESET 40: Experimental Reverse Validation
            # Same geometry as Preset 39 but with:
            #   - band_source = analytical (uses diffraction equation, not KDE)
            #   - crossing_diagnostics = ON (records angle, velocity, slit position)
            #   - Can toggle to single-slit (num_slits=1) for diffraction envelope test
            # Future: Tonomura (1989) photo digitization → band_source = file:tonomura_bands.npy
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("6.53")
            self.inputs["beam_momentum"].setText("2.28")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("20000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("1.0")
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(True)
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(True)
            self.reverse_num_particles_input.setText("500")
            self.band_source_input.setText("analytical")          # Diffraction equation bands
            self.crossing_diagnostics_cb.setChecked(True)          # Full crossing diagnostics ON
        elif text == "Analytical Wave Initialization (Plane Wave)":
            # PRESET 41: Plane wave initialization
            # Same geometry and pilot wave settings as Preset 38 but:
            #   - plane_wave_init = ON (starts grid with coherent plane wave)
            #   - wave_dissipation = 1.0 (conservative wave propagation)
            #   - jitter_amp = 0.0 (no noise for clean plane wave)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("6.53")
            self.inputs["beam_momentum"].setText("2.28")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("20000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("0.9999")  # 0.01%/tick damp kills reflections, preserves diffraction
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)         # Irrelevant (Pauli zeroed by pilot_wave)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)       # OFF: pure dBB — particles RIDE the wave, don't emit
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)               # ENABLE plane wave init!
            self.plane_wave_amp_input.setText("2.0")                # Sweet spot for visible deflection

        elif text == "MOSFET Corner Diffraction (Leakage)":
            # PRESET 42: MOSFET Corner Leakage Test
            # Tests whether electrons bend around a single edge barrier (Y=0).
            # Solid wall at Y < 0, open space at Y > 0.
            # Beam concentrated near edge (Y=-5 to +5) with moderate momentum.
            # Wall is 3 layers thick with strong Pauli for robust blocking.
            self.mode_combo.setCurrentText("single-edge")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("10.0")            # Strong exclusion for solid wall
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")  # Ignored by single-edge mode
            self.inputs["slit_separation"].setText("6.0") # Ignored by single-edge mode
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("3")          # 3-layer thick wall
            self.inputs["num_slits"].setText("1")
            self.inputs["screen_x"].setText("6.53")
            self.inputs["beam_momentum"].setText("10.0")    # Moderate — 26 ticks in wall zone
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("3000")      # Enough for p=10 to reach screen (~861 ticks)
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("100.0")
            self.inputs["wave_dissipation"].setText("0.9999")  # Damp reflections
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)       # Pure dBB
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)
            self.plane_wave_amp_input.setText("0.5")        # Sweet spot found experimentally

        elif text == "MOSFET Plane Wave (Corner + dBB)":
            # PRESET 43: Preset 41 double-slit + Preset 42 wall tuning
            # Double-slit plane wave with dBB pilot wave guidance
            # but with hardened wall physics (pauli=10, wall_depth=3, p=10)
            # to prevent electrons leaking through the barrier.
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("10.0")            # Hardened wall (was 1.0 in Preset 41)
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("3")          # 3-layer thick wall (was 1 in Preset 41)
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("20.0")
            self.inputs["beam_start_x"].setText("-10.0")
            self.inputs["beam_momentum"].setText("10.0")    # Proper wall interaction (was 2.28)
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("3000")      # p=10 reaches screen in ~861 ticks
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("65.0")
            self.inputs["wave_dissipation"].setText("0.9999")
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)       # Pure dBB
            self.breit_wheeler_cb.setChecked(False)
            self.bw_threshold_input.setText("2.044")
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)        # Plane wave ON
            self.plane_wave_amp_input.setText("20.0")       # Resonant amplitude
        
        elif text == "SR Time Dilation (Proper Time)":
            # TEST 2A: Pure SR time dilation — no gravity
            # With C=100: for gamma=5, p = m0*C*sqrt(gamma^2-1) = 0.511*100*sqrt(24) = 250
            # Particle A: p=250 (gamma~5, v=0.98c)
            # Particle B: at rest (gamma=1)
            # After 5000 ticks at dt=0.02 (100 time units):
            #   tau_A = sum(dt/gamma_A) ~ 100/5 = 20
            #   tau_B = sum(dt/gamma_B) ~ 100/1 = 100
            # Ratio: tau_B/tau_A ~ 5.0 = gamma_A
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("0.0")  # OFF — pure SR test
            self.inputs["vacuum"].setText("0.0")  # OFF — no damping
            self.inputs["torsion"].setText("0.0")  # OFF — no torsion
            self.inputs["entangled"].setText("0")
            self.inputs["sink_mass"].setText("0.0")  # No gravity — free particles
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("0.0")
            self.inputs["beam_momentum"].setText("250.0")  # gamma ~ 5 at C=100
            self.inputs["impact_parameter"].setText("100.0")  # Particle B at r=2000
            self.inputs["amps_cooling_cap"].setText("1.0")
            self.inputs["dt"].setText("0.02")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["galactic_spin"].setText("0.0")
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
        elif text == "Twin Paradox (Tau Comparison)":
            # TEST 2D: Gravitational twin paradox
            # Particle A: free-falls into M=50000 well from r=20 (gains huge gamma)
            # Particle B: at r=500, barely affected by gravity
            # With C=100, M=50000: v_freefall ~ sqrt(2GM/r0) = sqrt(5000) = 70.7
            # v/c = 0.707, gamma ~ 1.41 at closest approach
            # After run: tau_A < tau_B (accelerated twin ages less)
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("0.0")  # OFF — pure gravity
            self.inputs["vacuum"].setText("0.0")  # OFF — no damping
            self.inputs["torsion"].setText("0.0")  # OFF — pure GR test
            self.inputs["entangled"].setText("0")
            self.inputs["sink_mass"].setText("50000.0")  # Strong gravity well
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("0.0")  # Start at rest — pure freefall
            self.inputs["impact_parameter"].setText("25.0")  # Particle B at r=500
            self.inputs["amps_cooling_cap"].setText("1.0")
            self.inputs["dt"].setText("0.005")  # Fine timestep for freefall accuracy
            self.inputs["total_ticks"].setText("10000")  # 50 time units
            self.inputs["galactic_spin"].setText("0.0")
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
        elif text == "Kepler Orbit (Newtonian)":
            # TEST 1D: Stable orbit around central sink
            # With C=100: v_circ = sqrt(GM/r) = sqrt(5/20) = 0.5, v/c = 0.005 (deeply Newtonian)
            # p = m0 * v = 0.511 * 0.5 = 0.256 (gamma ≈ 1.0000)
            # T = 2π*r/v = 2π*20/0.5 = 251 time units
            # 120k ticks × dt=0.005 = 600 time units ≈ 2.4 orbits
            # ALL non-gravitational forces OFF for pure Kepler test
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("0.0")  # No Pauli — pure gravity
            self.inputs["vacuum"].setText("0.0")  # No damping
            self.inputs["torsion"].setText("0.0")  # No torsion — pure Newtonian
            self.inputs["entangled"].setText("0")
            self.inputs["sink_mass"].setText("5.0")  # GM/r = 5/20 = 0.25
            self.inputs["collapse_radius"].setText("20.0")  # Particle A starts here
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("0.272")  # Relativistic circular orbit p
            self.inputs["impact_parameter"].setText("5.0")  # Particle B at r=100 (spectator)
            self.inputs["amps_cooling_cap"].setText("1.0")
            self.inputs["dt"].setText("0.005")
            self.inputs["total_ticks"].setText("120000")  # ~2.25 orbits
            self.inputs["galactic_spin"].setText("0.0")
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
        elif text == "Gravity Slide (Terminator)":
            # TEST 5A: Sweep G to watch SINDy coefficients change
            # 2-body head-on with gravity sink active
            # Run this multiple times with different sink_mass values
            # to map how wave equation coefficients respond to gravity
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("1.0")
            self.inputs["vacuum"].setText("0.001")
            self.inputs["torsion"].setText("1.0")
            self.inputs["entangled"].setText("1")
            self.inputs["sink_mass"].setText("100.0")  # Start moderate, sweep up
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("50.0")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["amps_cooling_cap"].setText("1.0")
            self.inputs["dt"].setText("0.02")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["galactic_spin"].setText("0.0")
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)

        elif text == "Simple Collider (Sandbox)":
            # PRESET 44: Clean 2-body collider for beginners
            # Particle A (mass_a) starts left, heading right.
            # Particle B (mass_b) starts right, heading left.
            # Adjust mass_a, mass_b, beam_momentum, impact_parameter to experiment.
            # No slits, no gravity, no entanglement — just two particles colliding.
            self.mode_combo.setCurrentText("simple-collider")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("1.0")           # Particle A rest mass
            self.inputs["mass_b"].setText("1.0")           # Particle B rest mass
            self.inputs["pauli"].setText("50.0")           # Repulsive force strength
            self.inputs["vacuum"].setText("0.01")          # Damping
            self.inputs["torsion"].setText("1.0")          # Spin-vorticity coupling
            self.inputs["beam_momentum"].setText("5.0")    # Both particles get this momentum
            self.inputs["impact_parameter"].setText("0.5") # Y-offset: 0=head-on, higher=glancing
            self.inputs["entangled"].setText("0")
            self.inputs["dt"].setText("0.01")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["galactic_spin"].setText("0.0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(True)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)      # 1/r^3 Kaluza-Klein
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)

        elif text == "Quantum Tunneling (Pauli Barrier)":
            # PRESET 45: Quantum Tunneling through a solid Pauli barrier
            # Based on Preset 42 (MOSFET Corner Diffraction) but:
            #   - num_slits=0: SOLID wall with no openings
            #   - soft_wall=1: Hard reflection DISABLED — Pauli repulsion is the only barrier
            #   - Electrons must tunnel through the Pauli particle lattice
            # QM predicts: T ~ exp(-2*kappa*d), kappa = sqrt(2m(V-E)/hbar^2)
            # Vary wall_depth (1,2,3,5) to measure transmission vs thickness.
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")           # Electron mass
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("5.0")              # Barrier height — moderate so tunneling is possible
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("4.0")          # Controls z-extent of wall
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["num_slits"].setText("0")             # ZERO slits = SOLID WALL
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")            # Start thin — vary 1,2,3,5 for sweep
            self.inputs["screen_x"].setText("20.0")
            self.inputs["beam_start_x"].setText("-10.0")
            self.inputs["beam_momentum"].setText("10.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("5000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("65.0")
            self.inputs["wave_dissipation"].setText("0.9999")
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)
            self.breit_wheeler_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)
            self.plane_wave_amp_input.setText("20.0")
            self.soft_wall_cb.setChecked(True)                # KEY: Pauli-only barrier

        elif text == "Kepler Orbit (Vanishing Sun)":
            # PRESET 46: Exact duplicate of 36 (Kepler) but deletes the sun at 50% time
            self.paper1_exact_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentText("gravity-sink")
            self.inputs["num_particles"].setText("2")
            self.inputs["mass_a"].setText("0.511")
            self.inputs["mass_b"].setText("0.511")
            self.inputs["pauli"].setText("0.0")
            self.inputs["vacuum"].setText("0.0")
            self.inputs["torsion"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.inputs["sink_mass"].setText("5.0")
            self.inputs["collapse_radius"].setText("20.0")
            self.inputs["collapse_G"].setText("1.0")
            self.inputs["beam_momentum"].setText("0.272")
            self.inputs["impact_parameter"].setText("5.0")
            self.inputs["amps_cooling_cap"].setText("1.0")
            self.inputs["dt"].setText("0.005")
            self.inputs["total_ticks"].setText("120000")  # ~2.25 orbits total
            self.inputs["galactic_spin"].setText("0.0")
            self.photon_emission_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(False)
            self.dbb_guidance_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.vanish_sink_cb.setChecked(True)  # <-- The core feature for this preset
            self.fdtd_gravity_cb.setChecked(True) # <-- Uses vacuum memory grid instead of instantaneous math

        # ── ARNDT MASS SWEEP (Presets 48–54) ──────────────────────────
        # Velocity-matched sweep anchored to the REAL electron (0.511 MeV).
        # Anchor: Preset 43 → mass=0.511, p=10.0, v=p/m=19.57
        # All 7 runs hold v=19.57 constant → p = mass × 19.57
        # This means λ_dB = 2π/p shrinks with mass (correct physics).
        # mc²=hf is preserved: internal clock ω=m₀/γ scales with mass.
        # Goal: Find the mass where RAE phase-locking breaks → fringes vanish.
        elif text.startswith("Arndt Sweep"):
            # --- Shared Arndt base (identical to Preset 43 geometry) ---
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("10.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("3")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("20.0")
            self.inputs["beam_start_x"].setText("-10.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("3000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("65.0")
            self.inputs["wave_dissipation"].setText("0.9999")
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)
            self.breit_wheeler_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)
            self.plane_wave_amp_input.setText("20.0")
            self.rae_mode_cb.setChecked(True)
            self.inputs["rae_kappa_scale"].setText("1.0")
            self.inputs["rae_grad_scale"].setText("1.0")

            # --- Per-run mass and momentum ---
            # v = 19.57 for ALL runs (anchored to electron: 10.0/0.511)
            # p = mass × 19.57,  λ = 2π/p,  L_T = d²/λ,  ξ = 20/L_T
            arndt_params = {
                # PRESET 48: Sub-electron — deep quantum regime
                # p=1.96, λ=3.21, L_T=11.2, ξ=1.78 → Strong fringes
                "Arndt Sweep 1: Sub-electron (m=0.1)":     ("0.1",   "1.96"),
                # PRESET 49: Light particle — quantum regime
                # p=5.87, λ=1.07, L_T=33.6, ξ=0.60 → Clear fringes
                "Arndt Sweep 2: Light (m=0.3)":            ("0.3",   "5.87"),
                # PRESET 50: Electron — anchor point (matches Preset 43)
                # p=10.0, λ=0.628, L_T=57.3, ξ=0.35 → Known fringes ✓
                "Arndt Sweep 3: Electron (m=0.511)":       ("0.511", "10.0"),
                # PRESET 51: Transition zone — near your observed limit
                # p=14.68, λ=0.428, L_T=84.1, ξ=0.24 → Fringes weakening?
                "Arndt Sweep 4: Transition (m=0.75)":      ("0.75",  "14.68"),
                # PRESET 52: 2× electron — above your observed 0.6 cutoff
                # p=19.57, λ=0.321, L_T=112.1, ξ=0.18 → Classical?
                "Arndt Sweep 5: 2x Electron (m=1.0)":      ("1.0",   "19.57"),
                # PRESET 53: 4× electron — deep classical expected
                # p=39.14, λ=0.161, L_T=224.2, ξ=0.089 → Classical blob
                "Arndt Sweep 6: 4x Electron (m=2.0)":      ("2.0",   "39.14"),
                # PRESET 54: 10× electron — fully classical
                # p=97.85, λ=0.064, L_T=560.5, ξ=0.036 → Fully classical
                "Arndt Sweep 7: 10x Electron (m=5.0)":     ("5.0",   "97.85"),
            }
            # Strip the number prefix to match: "48. Arndt Sweep..." → "Arndt Sweep..."
            key = text.split(". ", 1)[1] if ". " in text else text
            if key in arndt_params:
                mass_a, momentum = arndt_params[key]
                self.inputs["mass_a"].setText(mass_a)
                self.inputs["beam_momentum"].setText(momentum)

        # ── PRESET 55: RAE MASS EXPLORER (FREE SLIDER) ─────────────
        # Type ANY mass in mass_a and all derived parameters auto-update.
        # Uses the RAE differential rules from the analysis document.
        elif text == "RAE Mass Explorer (Free Slider)":
            # Set base geometry (same as Preset 43)
            self.mode_combo.setCurrentText("double-slit")
            self.inputs["num_particles"].setText("10000")
            self.inputs["mass_a"].setText("0.511")  # Start at electron
            self.inputs["mass_b"].setText("1.0")
            self.inputs["pauli"].setText("10.0")
            self.inputs["vacuum"].setText("0.007")
            self.inputs["torsion"].setText("1.0")
            self.inputs["slit_width"].setText("5.0")
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("3")
            self.inputs["num_slits"].setText("2")
            self.inputs["screen_x"].setText("20.0")
            self.inputs["beam_start_x"].setText("-10.0")
            self.inputs["dt"].setText("0.001")
            self.inputs["total_ticks"].setText("3000")
            self.inputs["batch_size"].setText("10000")
            self.inputs["wave_speed"].setText("65.0")
            self.inputs["wave_dissipation"].setText("0.9999")
            self.inputs["jitter_amp"].setText("0.0")
            self.inputs["entangled"].setText("0")
            self.paper1_exact_cb.setChecked(False)
            self.spin_coupling_cb.setChecked(False)
            self.thermal_bath_cb.setChecked(False)
            self.eraser_active_cb.setChecked(False)
            self.pauli_power_combo.setCurrentIndex(0)
            self.photon_emission_cb.setChecked(False)
            self.breit_wheeler_cb.setChecked(False)
            self.pilot_wave_cb.setChecked(True)
            self.dbb_guidance_cb.setChecked(True)
            self.dbb_strength_input.setText("0.01")
            self.reverse_integration_cb.setChecked(False)
            self.plane_wave_init_cb.setChecked(True)
            self.plane_wave_amp_input.setText("20.0")
            self.rae_mode_cb.setChecked(True)
            self.inputs["rae_kappa_scale"].setText("1.0")
            self.inputs["rae_grad_scale"].setText("1.0")
            self.inputs["beam_momentum"].setText("10.0")
            # Activate the auto-compute connection
            self._rae_slider_active = True
            try:
                self.inputs["mass_a"].textChanged.disconnect(self.auto_compute_rae_from_mass)
            except (TypeError, RuntimeError):
                pass
            self.inputs["mass_a"].textChanged.connect(self.auto_compute_rae_from_mass)
            self.batch_label.setText("RAE Explorer: Type any mass in 'Mass A' → all params auto-update")
            self.batch_label.setVisible(True)

        # ── PRESET 56: RAE COMPENSATED SWEEP (7-RUN BATCH) ────────────
        # Velocity-matched AND coupling-compensated sweep.
        # Tests the RAE prediction: if κ and ∇Φ scale with m₀,
        # the geometric routing mechanism works at ANY mass.
        elif text == "RAE Compensated Sweep (7-Run Batch)":
            self._rae_slider_active = False
            # Define the 7-run matrix: (label, mass, momentum, kappa_s, grad_s, amp)
            # All at v=19.57, with couplings scaled by m/m_e
            m_e = 0.511
            v_e = 19.569  # 10.0 / 0.511
            sweep = [
                ("Sub-electron (m=0.1)",   0.1),
                ("Light (m=0.3)",          0.3),
                ("Electron (m=0.511)",     0.511),
                ("Transition (m=0.75)",    0.75),
                ("2x Electron (m=1.0)",    1.0),
                ("Proton (m=938.27)",      938.27),
                ("Neutron (m=939.57)",     939.57),
            ]
            self.batch_queue = []
            for label, mass in sweep:
                def make_setup(m=mass, lbl=label):
                    def setup():
                        ratio = m / m_e
                        p = m * v_e
                        dt_val = min(0.001, 2 * 3.14159 / (m * 10))
                        # Ticks: at least enough for transit + 50% margin
                        ticks = max(3000, int(1.5 * 30.0 / (v_e * dt_val)))
                        self.mode_combo.setCurrentText("double-slit")
                        self.inputs["num_particles"].setText("10000")
                        self.inputs["mass_a"].setText(f"{m}")
                        self.inputs["mass_b"].setText("1.0")
                        self.inputs["pauli"].setText("10.0")
                        self.inputs["vacuum"].setText("0.007")
                        self.inputs["torsion"].setText("1.0")
                        self.inputs["slit_width"].setText("5.0")
                        self.inputs["slit_separation"].setText("6.0")
                        self.inputs["wall_z_layers"].setText("1")
                        self.inputs["wall_depth"].setText("3")
                        self.inputs["num_slits"].setText("2")
                        self.inputs["screen_x"].setText("20.0")
                        self.inputs["beam_start_x"].setText("-10.0")
                        self.inputs["beam_momentum"].setText(f"{p:.4f}")
                        self.inputs["dt"].setText(f"{dt_val:.6f}")
                        self.inputs["total_ticks"].setText(f"{ticks}")
                        self.inputs["batch_size"].setText("10000")
                        self.inputs["wave_speed"].setText("65.0")
                        self.inputs["wave_dissipation"].setText("0.9999")
                        self.inputs["jitter_amp"].setText("0.0")
                        self.inputs["entangled"].setText("0")
                        self.paper1_exact_cb.setChecked(False)
                        self.spin_coupling_cb.setChecked(False)
                        self.thermal_bath_cb.setChecked(False)
                        self.eraser_active_cb.setChecked(False)
                        self.pauli_power_combo.setCurrentIndex(0)
                        self.photon_emission_cb.setChecked(False)
                        self.breit_wheeler_cb.setChecked(False)
                        self.pilot_wave_cb.setChecked(True)
                        self.dbb_guidance_cb.setChecked(True)
                        self.dbb_strength_input.setText("0.01")
                        self.reverse_integration_cb.setChecked(False)
                        self.plane_wave_init_cb.setChecked(True)
                        self.plane_wave_amp_input.setText(f"{20.0 * ratio:.2f}")
                        self.rae_mode_cb.setChecked(True)
                        self.inputs["rae_kappa_scale"].setText(f"{ratio:.4f}")
                        self.inputs["rae_grad_scale"].setText(f"{ratio:.4f}")
                    return setup
                self.batch_queue.append({"label": f"RAE Compensated: {label}", "setup_fn": make_setup()})
            # Apply first run for preview
            if self.batch_queue:
                self._batch_total = len(self.batch_queue)
                self.batch_queue[0]["setup_fn"]()
                self.batch_label.setText(f"BATCH QUEUED: {len(self.batch_queue)} runs | Next: {self.batch_queue[0]['label']}")
                self.batch_label.setVisible(True)

        # Disconnect slider auto-compute when leaving that preset
        if not text.endswith("(Free Slider)"):
            self._rae_slider_active = False

        # Update mode visibility after preset changes
        self.on_mode_changed(self.mode_combo.currentText())

    def run_simulation(self):
        self.run_btn.setEnabled(False)
        if self.batch_queue and not self.batch_running:
            # Starting a batch: apply first setup, mark batch mode
            self.batch_running = True
            current = self.batch_queue[0]
            current["setup_fn"]()
            self.run_btn.setText(f"BATCH [{1}/{len(self.batch_queue) + 0}]...")
            self.append_log(f"\n{'='*60}")
            self.append_log(f"BATCH RUN {1}/{len(self.batch_queue)}: {current['label']}")
            self.append_log(f"{'='*60}\n")
        elif self.batch_running and self.batch_queue:
            # Continuing batch: apply next setup
            current = self.batch_queue[0]
            current["setup_fn"]()
            remaining_total = len(self.batch_queue)
            self.run_btn.setText(f"BATCH [{remaining_total} left]...")
            self.append_log(f"\n{'='*60}")
            self.append_log(f"BATCH CONTINUING: {current['label']} ({remaining_total} remaining)")
            self.append_log(f"{'='*60}\n")
        else:
            self.run_btn.setText("SIMULATING...")
        if not self.batch_running:
            self.log_console.clear()
        self.math_console.clear()
        self.plot_spatial.clear()
        self.plot_phase.clear()
        if hasattr(self, 'plot_phase_zoom'):
            self.plot_spatial_zoom.clear()
            self.plot_phase_zoom.clear()

        # Use batch label for unique run labels, or preset text for single runs
        if self.batch_running and self.batch_queue:
            run_label = self.batch_queue[0]["label"]
        else:
            run_label = self.preset_combo.currentText()
        
        params = {
            "run_label": run_label,
            "mode": self.mode_combo.currentText(),
            "num_particles": self.inputs["num_particles"].text(),
            "mass_a": self.inputs["mass_a"].text(),
            "mass_b": self.inputs["mass_b"].text(),
            "pauli": self.inputs["pauli"].text(),
            "vacuum": self.inputs["vacuum"].text(),
            "torsion": self.inputs["torsion"].text(),
            "eraser_active": 1 if self.eraser_active_cb.isChecked() else 0,
            "photon_emission": 1 if getattr(self, 'photon_emission_cb', None) and self.photon_emission_cb.isChecked() else 0,
            "slit_width": self.inputs["slit_width"].text(),
            "slit_separation": self.inputs["slit_separation"].text(),
            "num_slits": self.inputs["num_slits"].text(),
            "screen_x": self.inputs["screen_x"].text(),
            "beam_start_x": self.inputs["beam_start_x"].text(),
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
            "batch_size": self.inputs["batch_size"].text() if "batch_size" in self.inputs else "0",
            "wave_speed": self.inputs["wave_speed"].text() if "wave_speed" in self.inputs else "100.0",
            "wave_dissipation": self.inputs["wave_dissipation"].text() if "wave_dissipation" in self.inputs else "0.999",
            "jitter_amp": self.inputs["jitter_amp"].text() if "jitter_amp" in self.inputs else "0.0",
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
            "buffer_dir": self.buffer_dir_input.text().strip(),
            "pilot_wave": 1 if getattr(self, 'pilot_wave_cb', None) and self.pilot_wave_cb.isChecked() else 0,
            "interpolation_order": "linear",
            "breit_wheeler": 1 if getattr(self, 'breit_wheeler_cb', None) and self.breit_wheeler_cb.isChecked() else 0,
            "bw_threshold": self.bw_threshold_input.text() if hasattr(self, 'bw_threshold_input') else "2.044",
            "dbb_guidance": 1 if getattr(self, 'dbb_guidance_cb', None) and self.dbb_guidance_cb.isChecked() else 0,
            "dbb_strength": self.dbb_strength_input.text() if hasattr(self, 'dbb_strength_input') else "0.01",
            "reverse_integration": 1 if getattr(self, 'reverse_integration_cb', None) and self.reverse_integration_cb.isChecked() else 0,
            "reverse_num_particles": self.reverse_num_particles_input.text() if hasattr(self, 'reverse_num_particles_input') else "500",
            "band_source": self.band_source_input.text() if hasattr(self, 'band_source_input') else "kde",
            "crossing_diagnostics": 1 if getattr(self, 'crossing_diagnostics_cb', None) and self.crossing_diagnostics_cb.isChecked() else 0,
            "plane_wave_init": 1 if getattr(self, 'plane_wave_init_cb', None) and self.plane_wave_init_cb.isChecked() else 0,
            "wave_shape": self.wave_shape_combo.currentText() if hasattr(self, 'wave_shape_combo') else "plane",
            "wave_freq": self.wave_freq_input.text() if hasattr(self, 'wave_freq_input') else "0.0",
            "plane_wave_amp": self.plane_wave_amp_input.text() if hasattr(self, 'plane_wave_amp_input') else "2.0",
            "which_path": 1 if getattr(self, 'which_path_cb', None) and self.which_path_cb.isChecked() else 0,
            "detector_mass": self.detector_mass_input.text() if hasattr(self, 'detector_mass_input') else "0.001",
            "detector_phase": self.detector_phase_combo.currentText() if hasattr(self, 'detector_phase_combo') else "thermal",
            "soft_wall": 1 if getattr(self, 'soft_wall_cb', None) and self.soft_wall_cb.isChecked() else 0,
            "vanish_sink": 1 if getattr(self, 'vanish_sink_cb', None) and self.vanish_sink_cb.isChecked() else 0,
            "fdtd_gravity": 1 if getattr(self, 'fdtd_gravity_cb', None) and self.fdtd_gravity_cb.isChecked() else 0,
            "rae_mode": 1 if getattr(self, 'rae_mode_cb', None) and self.rae_mode_cb.isChecked() else 0,
            "rae_kappa_scale": float(self.inputs.get("rae_kappa_scale", QLineEdit("1.0")).text()),
            "rae_grad_scale": float(self.inputs.get("rae_grad_scale", QLineEdit("1.0")).text()),
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
        spaced_text = text.replace('\n', '\n\n')
        self.math_console.setPlainText(spaced_text)

    def open_3d_viewer(self):
        current_mode = self.mode_combo.currentText()
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
        self.update_plots()
        
        # --- Batch Queue Auto-Advance ---
        if self.batch_running and self.batch_queue:
            # Pop the completed run
            completed = self.batch_queue.pop(0)
            self.append_log(f"\n✅ BATCH COMPLETE: {completed['label']}")
            
            if self.batch_queue:
                # More runs remain — auto-launch next
                next_run = self.batch_queue[0]
                total = getattr(self, '_batch_total', len(self.batch_queue) + 1)
                done = total - len(self.batch_queue)
                self.batch_label.setText(f"BATCH: {done}/{total} done | Next: {next_run['label']}")
                self.append_log(f"\n🔄 AUTO-LAUNCHING next run in 5 seconds...")
                self.append_log(f"   Next: {next_run['label']}")
                self.append_log(f"   Remaining: {len(self.batch_queue)} runs\n")
                
                # Brief delay then auto-launch
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(5000, self.run_simulation)
                return  # Don't re-enable button yet
            else:
                # All done!
                total = getattr(self, '_batch_total', 0)
                self.batch_running = False
                self.batch_label.setText(f"✅ BATCH COMPLETE: All {total} runs finished!")
                self.append_log(f"\n{'='*60}")
                self.append_log(f"🎉 FULL RAE VALIDATION MATRIX COMPLETE — 16/16 RUNS DONE")
                self.append_log(f"{'='*60}\n")
        
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN SIMULATION")

    def load_last_run(self):
        import os
        if os.path.exists("aggregate_states.npz"):
            self.append_log("Loading legacy data from aggregate_states.npz...")
            self.update_plots()
        else:
            self.append_log("Error: 'aggregate_states.npz' not found in current directory.")

    def update_plots(self):
        try:
            dot_size = float(self.inputs["plot_dot_size"].text())
        except Exception:
            dot_size = 2.0
            
        try:
            mode = self.mode_combo.currentText().strip()
            if mode not in ("double-slit", "single-edge"):
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
                initial_hues = hit_final_states[:, 8]  # CHANGED TO FINAL HUE
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
                scatter_spatial = pg.ScatterPlotItem(x=final_ys, y=final_zs, size=dot_size, brush=brushes, pen=None)
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
                    if all_min == all_max:
                        all_min -= 1.0
                        all_max += 1.0
                    bins = np.linspace(all_min, all_max, 201)
                    y_up, _ = np.histogram(ys_up, bins=bins)
                    y_dn, _ = np.histogram(ys_dn, bins=bins)
                    bars_up = pg.BarGraphItem(x=bins[:-1], height=y_up, width=(bins[1]-bins[0]), brush=pg.mkColor(255, 68, 68, 150))
                    bars_dn = pg.BarGraphItem(x=bins[:-1], height=y_dn, width=(bins[1]-bins[0]), brush=pg.mkColor(0, 229, 255, 150))
                    self.plot_spatial.addItem(bars_dn)
                    self.plot_spatial.addItem(bars_up)
                else:
                    y, x = np.histogram(final_ys, bins=200)
                    bin_width = x[1] - x[0] if len(x) > 1 and x[1] > x[0] else 0.1
                    bars = pg.BarGraphItem(x=x[:-1], height=y, width=bin_width, brush='#bb86fc')
                    self.plot_spatial.addItem(bars)

            # --- SINGLE-EDGE: Fresnel theory overlay + edge marker ---
            if mode == "single-edge":
                try:
                    from scipy.special import fresnel as fresnel_integrals
                    beam_p = float(self.inputs["beam_momentum"].text())
                    screen_d = float(self.inputs["screen_x"].text())
                    lam_dB = 2.0 * np.pi / beam_p if beam_p > 0 else 1.0
                    fresnel_scale = np.sqrt(2.0 / (lam_dB * screen_d)) if lam_dB * screen_d > 0 else 1.0
                    y_theory = np.linspace(final_ys.min() - 1, final_ys.max() + 1, 500)
                    v_theory = y_theory * fresnel_scale
                    S_th, C_th = fresnel_integrals(v_theory)
                    I_theory = 0.5 * ((C_th + 0.5)**2 + (S_th + 0.5)**2)
                    # Scale theory to match histogram peak
                    peak_hist = max(y.max(), 1) if not is_3d else 1
                    I_scaled = I_theory * peak_hist
                    theory_curve = pg.PlotCurveItem(x=y_theory, y=I_scaled, pen=pg.mkPen('#ff9800', width=2, style=Qt.PenStyle.DashLine))  # Orange dashed
                    self.plot_spatial.addItem(theory_curve)
                    self.append_log(f"Fresnel theory overlay: λ_dB={lam_dB:.4f}, screen={screen_d}, scale={fresnel_scale:.3f}")
                except Exception as fe:
                    self.append_log(f"Fresnel overlay skipped: {fe}")
                # Vertical edge marker at Y=0
                edge_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ff5252', width=2, style=Qt.PenStyle.DotLine))  # Red dotted
                self.plot_spatial.addItem(edge_line)
                self.plot_spatial.setTitle("Corner Diffraction — Histogram vs Fresnel Theory")
            
            # 2. Scatter Plot (Hue vs Final Y) - FULL RANGE
            scatter = pg.ScatterPlotItem(x=initial_hues, y=final_ys, size=dot_size * 0.75, brush=brushes, pen=None)
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
                self.plot_phase_zoom.setLabel('bottom', 'Final Hidden Phase (Hue)')
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
                    scatter_zoom = pg.ScatterPlotItem(x=zoom_ys, y=zoom_zs, size=dot_size * 1.5, brush=zoom_brushes, pen=None)
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
                        bin_width_z = x_z[1] - x_z[0] if len(x_z) > 1 and x_z[1] > x_z[0] else 0.1
                        bars_z = pg.BarGraphItem(x=x_z[:-1], height=y_z, width=bin_width_z, brush='#bb86fc')
                        self.plot_spatial_zoom.addItem(bars_z)
                # Add edge marker and Fresnel overlay to zoomed panel for single-edge
                if mode == "single-edge":
                    edge_line_z = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#ff5252', width=2, style=Qt.PenStyle.DotLine))
                    self.plot_spatial_zoom.addItem(edge_line_z)
                    self.plot_spatial_zoom.setTitle("Corner Diffraction — Zoomed (Y ±10)")
                    try:
                        from scipy.special import fresnel as fresnel_integrals
                        beam_p = float(self.inputs["beam_momentum"].text())
                        screen_d = float(self.inputs["screen_x"].text())
                        lam_dB = 2.0 * np.pi / beam_p if beam_p > 0 else 1.0
                        fresnel_scale = np.sqrt(2.0 / (lam_dB * screen_d)) if lam_dB * screen_d > 0 else 1.0
                        y_th_z = np.linspace(-10, 10, 500)
                        v_th_z = y_th_z * fresnel_scale
                        S_z, C_z = fresnel_integrals(v_th_z)
                        I_th_z = 0.5 * ((C_z + 0.5)**2 + (S_z + 0.5)**2)
                        peak_z = max(y_z.max(), 1) if not is_3d else 1
                        theory_z = pg.PlotCurveItem(x=y_th_z, y=I_th_z * peak_z, pen=pg.mkPen('#ff9800', width=2, style=Qt.PenStyle.DashLine))
                        self.plot_spatial_zoom.addItem(theory_z)
                    except Exception:
                        pass
            # Zoomed scatter - spin colored
            scatter_zoom = pg.ScatterPlotItem(x=initial_hues, y=final_ys, size=dot_size * 0.75, brush=brushes, pen=None)
            self.plot_phase_zoom.addItem(scatter_zoom)
            self.plot_phase_zoom.setYRange(-10, 10)
            if has_spin:
                self.plot_phase_zoom.setTitle("Phase Router Zoomed (Red=Spin↑  Cyan=Spin↓)")
            
            self.append_log(f"Successfully loaded and plotted {len(final_ys)} screen hits out of {len(outcomes)} total trials.")
            self.append_log("Zero transport latency: Web browser JSON bottleneck successfully bypassed.")

        except Exception as e:
            self.append_log(f"Error loading plot data: {str(e)}")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    window = TeleparallelGUI()
    window.show()
    sys.exit(app.exec())
