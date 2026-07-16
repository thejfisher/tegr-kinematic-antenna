import json
import glob

files = sorted(glob.glob('sindy_report_RAE Matrix_*.json'))
print(f'Found {len(files)} reports.')
print(f"| Run Config | Overall R^2 | hue' Equation |")
print('|---|---|---|')

for f in files:
    name = f.replace('sindy_report_RAE Matrix_ ', '').replace('.json', '')
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            overall = data.get('r2_score', 'N/A')
            eqs = data.get('equations', {})
            hue_str = eqs.get("hue'", 'N/A')
            
            # format values
            overall_fmt = f"{overall:.4f}" if isinstance(overall, float) else str(overall)
            
            print(f'| {name} | {overall_fmt} | {hue_str} |')
    except Exception as e:
        print(f'| {name} | ERROR | ERROR | {str(e)} |')
