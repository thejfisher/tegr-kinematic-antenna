import json
import glob
import re

files = glob.glob('sindy_report_RAE Compensated_*.json')
results = {}

for f in files:
    with open(f, 'r') as fp: data = json.load(fp)
    m = float(re.search(r'm=([\d\.]+)', data['run_label']).group(1))
    eqs = data['equations']
    vx_eq = eqs.get("vx'", "")
    match_r3 = re.search(r'([\-\.\d]+)\s+1/r\^3', vx_eq)
    match_sin = re.search(r'([\-\.\d]+)\s+sin\(hue\)', vx_eq)
    results[m] = {
        'r3': abs(float(match_r3.group(1))) if match_r3 else 1.0,
        'sin': abs(float(match_sin.group(1))) if match_sin else 1.0
    }

m_e = 0.511
ref_r3 = results[m_e]['r3']
ref_sin = results[m_e]['sin']

masses = sorted(results.keys())
kappa_scales = [results[m]['r3'] / ref_r3 for m in masses]
grad_scales = [results[m]['sin'] / ref_sin for m in masses]

print('masses =', masses)
print('kappa_scales =', [round(x, 4) for x in kappa_scales])
print('grad_scales =', [round(x, 4) for x in grad_scales])
