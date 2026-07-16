import re

with open('teleparallel_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('self.inputs["dt"].setText("0.01")', 'self.inputs["dt"].setText("0.006")')
content = content.replace('self.inputs["total_ticks"].setText("1200")', 'self.inputs["total_ticks"].setText("2000")')

with open('teleparallel_gui.py', 'w', encoding='utf-8') as f:
    f.write(content)
