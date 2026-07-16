import re

with open('teleparallel_collider.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('GRID_RES = 50', 'GRID_RES = 25')

with open('teleparallel_collider.py', 'w', encoding='utf-8') as f:
    f.write(content)
