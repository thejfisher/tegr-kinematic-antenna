import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QPushButton, QSplitter,
                             QFormLayout, QGroupBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer
import pysindy as ps

class MacroNodeEngine:
    def __init__(self):
        self.load_meta_sindy()
        
        # 1:1 Scaling Constants
        # Distance: 1.0 = 1 Astronomical Unit (AU)
        # Mass: 1.0 = 1 Solar Mass (M_sun)
        # Velocity scaled so that a stable orbit at 1 AU is approx v=1.0 for standard gravity
        
        # Array shape: [N, 2]
        self.positions = []
        self.velocities = []
        self.masses = []
        self.labels = []
        self.colors = []
        
        self.dt = 0.001
        self.G_macro = 0.002238 # Tuned scaling factor to map SINDy forces to AU scale for V=1.0 at R=1.0
        
        self.c_r = 0.0
        self.c_inv_r3 = 0.0
        self.physics_model = "SINDy" # or "Newtonian"

        
        self.initialize_galaxy()

    def load_meta_sindy(self):
        print("Building Topological Manifold from Matrix A data...")
        df = pd.read_csv('analysis_full.csv')
        df_grouped = df.groupby(['pauli', 'vac', 'tor', 'cmb']).mean().reset_index()
        X = df_grouped[['pauli', 'vac', 'tor', 'cmb']].values
        y_cr = df_grouped['c_r'].values
        y_inv = df_grouped['c_inv_r3'].values

        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        self.interp_cr_lin = LinearNDInterpolator(X, y_cr)
        self.interp_cr_near = NearestNDInterpolator(X, y_cr)
        
        self.interp_inv_lin = LinearNDInterpolator(X, y_inv)
        self.interp_inv_near = NearestNDInterpolator(X, y_inv)
        print("Topological Manifold Initialized.")

    def update_coefficients(self, p, v, t_val, c):
        params = np.array([[p, v, t_val, c]])
        
        cr_val = self.interp_cr_lin(params)[0]
        if np.isnan(cr_val):
            cr_val = self.interp_cr_near(params)[0]
            
        inv_val = self.interp_inv_lin(params)[0]
        if np.isnan(inv_val):
            inv_val = self.interp_inv_near(params)[0]
            
        self.c_r = float(cr_val)
        self.c_inv_r3 = float(inv_val)

    def initialize_galaxy(self):
        self.labels = []
        self.positions = []
        self.velocities = []
        self.masses = []
        self.colors = []
        
        # Add Central Supermassive Black Hole
        self.add_node("Sagittarius A*", [0.0, 0.0], [0.0, 0.0], 10.0, (255, 255, 255))
        
        def get_circular_velocity(r):
            if self.physics_model == "Newtonian":
                # F_g = G * M / r^2. Scale to ensure outer stars don't fly away
                return np.sqrt(10.0 / r)
            else:
                # SINDy
                softening = 0.05
                r_soft = np.sqrt(r**2 + softening**2)
                f = - (self.c_r * r_soft + self.c_inv_r3 / (r_soft**3))
                f_acc = f * self.G_macro
                if f_acc < 0:
                    return 0.0 # Repulsive, cannot have circular orbit
                return np.sqrt(f_acc * r)
        
        # Spawn 150 stars randomly in a galactic disk to save CPU
        np.random.seed(42) # Fixed seed for consistent visualization
        for i in range(150):
            r = np.random.uniform(1.5, 48.0) # Avoid the exact center to prevent extreme Pauli forces
            theta = np.random.uniform(0, 2 * np.pi)
            v_mag = get_circular_velocity(r)
            m = np.random.uniform(1e-6, 5e-6)
            
            # Slight color variations for stars
            c_val = np.random.randint(180, 255)
            color = (c_val, np.random.randint(180, c_val+1), 255) if np.random.rand() > 0.5 else (255, c_val, c_val)
            
            pos = [r * np.cos(theta), r * np.sin(theta)]
            vel = [-v_mag * np.sin(theta), v_mag * np.cos(theta)]
            self.add_node(f"Star_{i}", pos, vel, m, color)
        
        self.positions = np.array(self.positions, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.masses = np.array(self.masses, dtype=np.float32)

    def add_node(self, label, pos, vel, mass, color):
        self.labels.append(label)
        self.positions.append(pos)
        self.velocities.append(vel)
        self.masses.append(mass)
        self.colors.append(color)

    def compute_accelerations(self, pos):
        N = len(self.masses)
        acc = np.zeros_like(pos)
        
        if N == 0:
            return acc
            
        for i in range(N):
            a_i = np.zeros(2)
            for j in range(N):
                if i == j:
                    continue
                r_vec = pos[j] - pos[i]
                r_sq = np.dot(r_vec, r_vec)
                
                softening = 0.05
                r = np.sqrt(r_sq + softening**2)
                
                if self.physics_model == "Newtonian":
                    a_mag = 10.0 * self.masses[j] / (r**2)
                    force_mag = a_mag / self.G_macro
                else:
                    force_mag = - (self.c_r * r + self.c_inv_r3 / (r**3))
                
                r_true = np.sqrt(r_sq) + 1e-12
                r_hat = r_vec / r_true
                
                a_i += self.masses[j] * force_mag * r_hat * self.G_macro
            acc[i] = a_i
        return acc

    def step_verlet(self):
        a_t = self.compute_accelerations(self.positions)
        new_pos = self.positions + self.velocities * self.dt + 0.5 * a_t * (self.dt**2)
        a_next = self.compute_accelerations(new_pos)
        new_vel = self.velocities + 0.5 * (a_t + a_next) * self.dt
        
        self.positions = new_pos
        self.velocities = new_vel

class Galaxy_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meta-SINDy Galaxy Simulation")
        self.resize(1400, 900)
        
        self.engine = MacroNodeEngine()
        
        self.is_running = False
        
        self.setup_ui()
        self.update_physics_params()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
        
        self.toggle_play()
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        from PyQt6.QtWidgets import QComboBox, QLabel
        
        # Left Panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        # Physics Toggle
        phys_group = QGroupBox("Physics Engine")
        phys_layout = QVBoxLayout(phys_group)
        self.combo_physics = QComboBox()
        self.combo_physics.addItems(["Meta-SINDy (Matrix A)", "Newtonian (Real World)"])
        self.combo_physics.currentIndexChanged.connect(self.change_physics)
        phys_layout.addWidget(self.combo_physics)
        left_layout.addWidget(phys_group)
        
        # Galaxy Info
        info_group = QGroupBox("Galaxy Information")
        info_layout = QVBoxLayout(info_group)
        info_label = QLabel("Simulating 150 Stars\nCentral Supermassive Black Hole\nOrbiting in a Galactic Disk")
        info_layout.addWidget(info_label)
        
        self.btn_restart = QPushButton("REGENERATE GALAXY")
        self.btn_restart.setStyleSheet("background-color: #ffb86c; color: #000; font-size: 14px; font-weight: bold;")
        self.btn_restart.clicked.connect(self.restart_sim)
        info_layout.addWidget(self.btn_restart)
        
        left_layout.addWidget(info_group)
        
        control_group = QGroupBox("Meta-SINDy Topology")
        c_layout = QFormLayout(control_group)
        
        self.sl_p = self.create_slider(0, 1000, 500)
        self.sl_v = self.create_slider(0, 100, 0)
        self.sl_t = self.create_slider(0, 500, 100)
        self.sl_c = self.create_slider(0, 500, 0)
        
        self.lbl_p = QLabel("Pauli Stress (5.0)")
        self.lbl_v = QLabel("Vacuum (0.0)")
        self.lbl_t = QLabel("Torsion (1.0)")
        self.lbl_c = QLabel("CMB (0.0)")
        
        c_layout.addRow(self.lbl_p, self.sl_p)
        c_layout.addRow(self.lbl_v, self.sl_v)
        c_layout.addRow(self.lbl_t, self.sl_t)
        c_layout.addRow(self.lbl_c, self.sl_c)
        
        self.sl_p.valueChanged.connect(self.update_physics_params)
        self.sl_v.valueChanged.connect(self.update_physics_params)
        self.sl_t.valueChanged.connect(self.update_physics_params)
        self.sl_c.valueChanged.connect(self.update_physics_params)
        
        left_layout.addWidget(control_group)
        
        # Scaling Controls
        scale_group = QGroupBox("Engine Scales")
        s_layout = QFormLayout(scale_group)
        
        self.sl_size = self.create_slider(1, 100, 15)
        self.lbl_size = QLabel("Visual Dot Size (15)")
        self.sl_size.valueChanged.connect(self.update_visuals)
        s_layout.addRow(self.lbl_size, self.sl_size)
        
        self.spin_g = QDoubleSpinBox()
        self.spin_g.setRange(1e-10, 1.0)
        self.spin_g.setDecimals(8)
        self.spin_g.setSingleStep(1e-6)
        self.spin_g.setValue(self.engine.G_macro)
        self.spin_g.valueChanged.connect(self.update_g)
        s_layout.addRow("G_Macro Scale:", self.spin_g)
        
        self.sl_speed = self.create_slider(1, 100, 10)
        self.lbl_speed = QLabel("Simulation Speed (10)")
        self.sl_speed.valueChanged.connect(self.update_speed)
        s_layout.addRow(self.lbl_speed, self.sl_speed)
        self.steps_per_frame = 10
        
        left_layout.addWidget(scale_group)
        
        # Playback Controls
        self.btn_play = QPushButton("PLAY")
        self.btn_play.setStyleSheet("background-color: #03dac6; color: #000; font-size: 16px; font-weight: bold;")
        self.btn_play.clicked.connect(self.toggle_play)
        left_layout.addWidget(self.btn_play)
        
        # Real-time Coefficients
        self.coef_label = QLabel("c_r: 0\nc_inv_r3: 0")
        self.coef_label.setStyleSheet("color: #bb86fc; font-family: monospace; font-size: 14px;")
        left_layout.addWidget(self.coef_label)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Right Panel: Plot
        self.graph = pg.GraphicsLayoutWidget()
        self.plot = self.graph.addPlot(title="Matrix B: Galactic Center")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setXRange(-40, 40)
        self.plot.setYRange(-40, 40)
        self.plot.setAspectLocked(True)
        
        self.scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), pxMode=True)
        self.plot.addItem(self.scatter)
        
        main_layout.addWidget(self.graph)

    def create_slider(self, min_val, max_val, default):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def update_physics_params(self):
        p = self.sl_p.value() / 100.0
        v = self.sl_v.value() / 100.0
        t_val = self.sl_t.value() / 100.0
        c = self.sl_c.value() / 100.0
        
        self.lbl_p.setText(f"Pauli Stress ({p:.2f})")
        self.lbl_v.setText(f"Vacuum ({v:.2f})")
        self.lbl_t.setText(f"Torsion ({t_val:.2f})")
        self.lbl_c.setText(f"CMB ({c:.2f})")
        
        self.engine.update_coefficients(p, v, t_val, c)
        
        self.coef_label.setText(f"c_r: {self.engine.c_r:,.2f}\nc_inv_r3: {self.engine.c_inv_r3:,.2f}")

    def update_visuals(self):
        size = self.sl_size.value()
        self.lbl_size.setText(f"Visual Dot Size ({size})")
        self.draw_frame()

    def update_g(self):
        self.engine.G_macro = self.spin_g.value()

    def update_speed(self):
        self.steps_per_frame = self.sl_speed.value()
        self.lbl_speed.setText(f"Simulation Speed ({self.steps_per_frame})")

    def change_physics(self):
        if self.combo_physics.currentIndex() == 0:
            self.engine.physics_model = "SINDy"
        else:
            self.engine.physics_model = "Newtonian"
        self.restart_sim()

    def toggle_play(self):
        if self.is_running:
            self.is_running = False
            self.btn_play.setText("PLAY")
            self.btn_play.setStyleSheet("background-color: #03dac6; color: #000; font-size: 16px; font-weight: bold;")
            self.timer.stop()
        else:
            self.is_running = True
            self.btn_play.setText("PAUSE")
            self.btn_play.setStyleSheet("background-color: #cf6679; color: #000; font-size: 16px; font-weight: bold;")
            self.timer.start(16) # ~60 FPS

    def game_loop(self):
        # N physics ticks per render frame for stability
        for _ in range(self.steps_per_frame):
            self.engine.step_verlet()
        self.draw_frame()

    def restart_sim(self):
        self.engine.initialize_galaxy()
        self.draw_frame()

    def draw_frame(self):
        spots = []
        size = self.sl_size.value()
        for i in range(len(self.engine.masses)):
            spots.append({
                'pos': (self.engine.positions[i, 0], self.engine.positions[i, 1]),
                'data': 1,
                'brush': pg.mkBrush(self.engine.colors[i]),
                'size': size
            })
        self.scatter.setData(spots)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Galaxy_GUI()
    win.show()
    sys.exit(app.exec())
