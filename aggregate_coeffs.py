import pandas as pd
import numpy as np
import sys
import re

def parse_equation(eq_str):
    """Parse a SINDy equation string to extract coefficients and terms.
    Returns a dict mapping term to coefficient.
    """
    coeffs = {}
    if not isinstance(eq_str, str) or not eq_str.strip():
        return coeffs
    
    # Split by + or - keeping the sign
    parts = re.findall(r'[+-]?[^+-]+', eq_str.replace(' ', ''))
    
    for part in parts:
        if not part: continue
        # Extract coefficient
        match = re.match(r'([+-]?\d+\.?\d*)\s*(.*)', part)
        if match:
            val, term = match.groups()
            if not term:
                term = '1' # Bias term
            coeffs[term] = float(val)
    return coeffs

def main():
    print("Aggregating SINDy Coefficients...")
    try:
        df = pd.read_csv("sindy_results.csv")
    except FileNotFoundError:
        print("sindy_results.csv not found.")
        sys.exit(1)
        
    print(f"Loaded {len(df)} records.")
    
    # Group by run_label (preset)
    groups = df.groupby(df['run_label'].str.extract(r'(.*)_trial_\d+')[0].fillna(df['run_label']))
    
    for name, group in groups:
        print(f"\n==============================")
        print(f"Stability for Preset: {name}")
        print(f"==============================")
        print(f"Trials: {len(group)}")
        print(f"Mean R^2: {group['r2_score'].mean():.4f} ± {group['r2_score'].std():.4f}")
        
        # Analyze stability of vx' equation as an example
        eq_col = "vx'"
        if eq_col in group.columns:
            all_coeffs = []
            for eq in group[eq_col]:
                all_coeffs.append(parse_equation(eq))
            
            # Get all unique terms
            terms = set()
            for c in all_coeffs:
                terms.update(c.keys())
                
            print(f"\nCoefficients for {eq_col}:")
            for term in sorted(list(terms)):
                vals = [c.get(term, 0.0) for c in all_coeffs]
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                if abs(mean_v) > 1e-4 or std_v > 1e-4:
                    print(f"  {term:12s}: {mean_v:8.4f} ± {std_v:8.4f}")

if __name__ == "__main__":
    main()
