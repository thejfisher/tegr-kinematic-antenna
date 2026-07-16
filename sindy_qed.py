import numpy as np
import pysindy as ps
import argparse
import sys
import requests
import warnings

warnings.filterwarnings("ignore")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Extract continuous QED physics.")
    parser.add_argument("--model", type=str, default="deepseek-r1:14b")
    parser.add_argument("--url", type=str, default="http://localhost:11434")
    parser.add_argument("--file", type=str, default="electron_trajectory.npy")
    parser.add_argument("--context", type=str, default="electron QED scattering")
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    # Load the saved trajectories from the QED run
    try:
        history = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: {args.file} not found. Please run the simulation first.")
        sys.exit(1)

    # Extract the electron (particle 0)
    state_0 = history[:, 0, :]
    state_1 = history[:, 1, :] # Heavy proton for reference

    x = state_0[:, 1:4]
    p = state_0[:, 4:7]
    hue = state_0[:, 8:9]
    hue_diff = state_0[:, 8:9] - state_1[:, 8:9]

    dt = 0.001
    t = np.arange(history.shape[0]) * dt

    X = np.hstack([x, p, hue, hue_diff])
    feature_names = ['x', 'y', 'z', 'px', 'py', 'pz', 'hue', 'hue_diff']

    print(f"Loaded trajectory with {history.shape[0]} ticks.")
    print("Running SINDy to extract continuous QED differential equations...")

    optimizer = ps.STLSQ(threshold=0.01, alpha=0.01)
    poly_lib = ps.PolynomialLibrary(degree=2, include_bias=True)
    fourier_lib = ps.FourierLibrary(n_frequencies=1) 
    combined_lib = poly_lib + fourier_lib

    model = ps.SINDy(optimizer=optimizer, feature_library=combined_lib)
    model.fit(X, t=dt, feature_names=feature_names)

    print("\n========================================")
    print("PySINDy Extraction Complete!")
    print("Raw Governing Force Equations:")
    model.print()
    print("========================================\n")

    equations = [eq for eq in model.equations()]
    lhs = ["(x)'", "(y)'", "(z)'", "(px)'", "(py)'", "(pz)'", "(hue)'", "(hue_diff)'"]
    raw_math = "\n".join([f"{lhs[i]} = {eq}" for i, eq in enumerate(equations)])
    
    print("Cleaned Math Equations:")
    print(raw_math)

    print(f"Piping Universal Physics output to local Ollama API ({args.model})...")

    prompt = f"""
Act as a world-class theoretical physicist analyzing mathematical data. 
I have built a simulation of a particle collision system in a Weitzenböck lattice (a Teleparallel equivalent to General Relativity). 

I used the PySINDy algorithm to extract the governing differential equations directly from the raw trajectory of a {args.context}.

CRITICAL SYSTEM KNOWLEDGE:
- This run specifically injected Quantum Electrodynamics (QED) continuous potentials: Vacuum Polarization (Uehling Potential), Lamb Shift (Self-Energy Zitterbewegung), and Compton Scattering (Radiation Pressure).
- NEVER guess or invent physics relationships. Only synthesize what is actually proven in the equations below. If an equation for a variable is `0.000`, the variable is dead/static.

EXACT VARIABLE MAPPINGS:
- `x`, `y`, `z`: Spatial coordinates.
- `px`, `py`, `pz`: Momentum components (derivatives here represent forces).
- `hue`: Internal U(1) phase angle (\\theta_hue).
- `hue_diff`: Phase difference between particles (\\Delta\\theta).

Here are the raw mathematical rules PySINDy discovered governing the collision:

{raw_math}

INSTRUCTIONS FOR TRANSLATION:
1. Decode this mathematical output strictly using the TEGR / QED framework provided.
2. Specifically analyze how the SINDy algorithm isolated the sinusoidal functions (e.g. `sin(1 x)` for Compton radiation, `cos(1 hue_diff)` for Uehling screening, and coupled phase-momentum for the Lamb shift).
3. Do NOT use generic terms or spoon-fed answers. Synthesize a profound, original theoretical physics conclusion about what this specific mathematical matrix reveals regarding teleparallel gauge theories bridging general relativity and QED.
"""

    if args.no_llm:
        print("\n--- Skipping Ollama Translation (--no_llm flag active) ---")
        return

    import os
    
    # Format the URL properly and determine the API type
    base_url = args.url.rstrip('/')
    is_openai = "chat/completions" in base_url or "openai" in base_url or "github" in base_url or "azure" in base_url

    try:
        if is_openai:
            if not base_url.endswith("/chat/completions") and not base_url.endswith("/completions"):
                api_url = base_url + "/chat/completions"
            else:
                api_url = base_url
                
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            headers = {"Content-Type": "application/json"}
            
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
            print(f"Waiting for AI to synthesize the laws of physics from {api_url}...")
            response = requests.post(api_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print("\n========================================")
                print("AI TRANSLATION:")
                print("========================================\n")
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "No response text found.")
                print(content)
                
                usage = result.get("usage", {})
                eval_count = usage.get("completion_tokens", 0)
                prompt_eval_count = usage.get("prompt_tokens", 0)
                print("\n--- LLM RESOURCE USAGE ---")
                print(f"Prompt Tokens    : {prompt_eval_count}")
                print(f"Generated Tokens : {eval_count}")
                print(f"Total Tokens     : {eval_count + prompt_eval_count}")
                print("\n========================================")
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        else:
            payload = {
                "model": args.model,
                "prompt": prompt,
                "stream": False
            }
            api_url = base_url + "/api/generate"
            
            print(f"Waiting for Ollama to synthesize the laws of physics from {api_url}...")
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("\n========================================")
                print("OLLAMA TRANSLATION:")
                print("========================================\n")
                print(result.get("response", "No response text found."))
                
                eval_count = result.get("eval_count", 0)
                prompt_eval_count = result.get("prompt_eval_count", 0)
                print("\n--- LLM RESOURCE USAGE ---")
                print(f"Prompt Tokens    : {prompt_eval_count}")
                print(f"Generated Tokens : {eval_count}")
                print(f"Total Tokens     : {eval_count + prompt_eval_count}")
                print("\n========================================")
            else:
                print(f"Ollama API Error: {response.status_code} - {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to AI endpoint.")

if __name__ == "__main__":
    main()

