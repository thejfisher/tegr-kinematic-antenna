import pandas as pd
import re
import math

df = pd.read_csv('Z:\\teleparallel_sim\\sindy_results_full.csv')

targets = {
    'g-factor': 2.0023,
    'Spin-1/2': 0.5,
    'Mass Ratio': 1836.15,
    'Von Klitzing': 25812.8,
    'Von Klitzing (x10)': 258128.0,
    'Fine Structure': 0.007297
}

# Regex to find all numbers (including scientific notation and decimals)
num_pattern = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\b\d+\b')

findings = []

for index, row in df.iterrows():
    run_label = row['run_label']
    for col in df.columns:
        if col == 'run_label': continue
        
        eq = str(row[col])
        if eq == '0.0' or eq == 'nan': continue
        
        # Extract all numbers from the equation string
        nums = [float(n) for n in num_pattern.findall(eq)]
        
        for n in nums:
            val = abs(n)
            if val == 0: continue
            
            for t_name, t_val in targets.items():
                # 2.0% tolerance to catch them
                if abs(val - t_val) / t_val < 0.02:
                    findings.append({
                        'Run': run_label,
                        'Target': t_name,
                        'Expected': t_val,
                        'Found': val,
                        'Variable': col,
                        'Equation': eq
                    })

if len(findings) == 0:
    print('No constants found within 2.0% tolerance.')
else:
    for f in findings:
        print(f"[*] {f['Target']} found in {f['Variable']} (Run: {f['Run']})")
        print(f"    Expected: {f['Expected']} | Found: {f['Found']}")
        print(f"    Equation: {f['Equation']}")
        print('-'*60)
