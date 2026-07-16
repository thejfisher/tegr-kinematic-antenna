import re

file_path = "Z:/teleparallel_sim_photons/teleparallel_gui.py"
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Pattern to find the RAE block in gravity layout
pattern = re.compile(
    r"[ \t]*# --- RAE Phase Override ---.*?"
    r"left_layout\.addWidget\(self\.gravity_group\)",
    re.DOTALL
)

replacement = """        left_layout.addWidget(self.gravity_group)

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

        left_layout.addWidget(self.rae_group)"""

new_content, count = pattern.subn(replacement, content)
if count > 0:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Success! Applied GUI changes.")
else:
    print("Failed to find the pattern.")
