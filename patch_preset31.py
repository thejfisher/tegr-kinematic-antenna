import re

with open('teleparallel_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

preset_31 = '''        elif text == "31. Pilot Wave Double-Slit (Histogram)":
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
            self.inputs["slit_separation"].setText("6.0")
            self.inputs["num_slits"].setText("2")
            self.inputs["wall_z_layers"].setText("1")
            self.inputs["wall_depth"].setText("1")
            self.inputs["beam_momentum"].setText("5000.0")
            self.inputs["dt"].setText("0.01")
            self.inputs["total_ticks"].setText("1200")
            self.inputs["entangled"].setText("0")
            self.pauli_power_combo.setCurrentIndex(0)
            self.eraser_active_cb.setChecked(False)
            self.photon_emission_cb.setChecked(True)
            self.firewall_active_cb.setChecked(False)
'''

# Find the end of preset 30
target = "            self.firewall_active_cb.setChecked(False)\n"
idx = content.rfind(target)
if idx != -1:
    content = content[:idx + len(target)] + preset_31 + content[idx + len(target):]
    with open('teleparallel_gui.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Preset 31 added.")
else:
    print("Could not find insertion point.")
