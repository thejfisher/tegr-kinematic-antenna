import pandas as pd
import re

df = pd.read_csv('Z:\\teleparallel_sim\\sindy_results_full.csv')
print(f'Total extracted equations: {len(df)}')

data = []
for index, row in df.iterrows():
    label = row['run_label']
    if not label.startswith('P_'): continue
    
    parts = label.split('_')
    if len(parts) >= 8:
        pauli = float(parts[1])
        vac = float(parts[3])
        tor = float(parts[5])
        cmb = float(parts[7])
        
        # Look at r''
        r_eq = row["r''"]
        
        # Extract the coefficient of 1/r^3 and 'r' (distance repulsion)
        # Using regex
        match_inv_r3 = re.search(r'([-+]?\s*\d+\.\d+)\s*1/r\^3', r_eq)
        match_r = re.search(r'([-+]?\s*\d+\.\d+)\s*r(?!\w|\')', r_eq)
        
        c_inv_r3 = float(match_inv_r3.group(1).replace(' ', '')) if match_inv_r3 else 0.0
        c_r = float(match_r.group(1).replace(' ', '')) if match_r else 0.0
        
        data.append({
            'pauli': pauli, 'vac': vac, 'tor': tor, 'cmb': cmb,
            'c_inv_r3': c_inv_r3, 'c_r': c_r
        })

res = pd.DataFrame(data)
res.sort_values(by=['pauli', 'vac', 'tor', 'cmb'], inplace=True)
res.to_csv('Z:\\teleparallel_sim\\analysis_full.csv', index=False)
print('Saved to analysis_full.csv')
