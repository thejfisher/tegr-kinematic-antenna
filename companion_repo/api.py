import subprocess
import sys
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationParams(BaseModel):
    mode: str = "2-body"
    mass_a: float
    mass_b: float
    mass_c: float = 1.0
    pauli: float
    vacuum: float
    torsion: float
    t_symmetry: bool = False
    pauli_enabled: bool = True
    vacuum_enabled: bool = True
    torsion_enabled: bool = True
    entangled: bool = False
    gravity_well_enabled: bool = False
    velocity_clamp: int = -1  # -1=auto, 0=off (Paper 2), 1=on (Paper 1)
    sink_mass: float = 50000.0

@app.post("/api/extract")
async def extract_math(params: SimulationParams):
    try:
        print(f"Running Collider with: MassA={params.mass_a}, MassB={params.mass_b}, Pauli={params.pauli}, Vacuum={params.vacuum}, Torsion={params.torsion}")
        
        # 1. Run the particle collider to generate the trajectory file
        collider_cmd = [
            sys.executable, "teleparallel_collider.py",
            "--mode", params.mode,
            "--mass_a", str(params.mass_a),
            "--mass_b", str(params.mass_b),
            "--mass_c", str(params.mass_c),
            "--pauli", str(params.pauli),
            "--vacuum", str(params.vacuum),
            "--torsion", str(params.torsion),
            "--pauli_enabled", "1" if params.pauli_enabled else "0",
            "--vacuum_enabled", "1" if params.vacuum_enabled else "0",
            "--torsion_enabled", "1" if params.torsion_enabled else "0",
            "--entangled", "1" if params.entangled else "0",
            "--velocity_clamp", str(params.velocity_clamp),
            "--sink_mass", str(params.sink_mass)
        ]
        if params.t_symmetry:
            collider_cmd.append("--t_symmetry")
            
        collider_result = subprocess.run(collider_cmd, capture_output=True, text=True, encoding='utf-8', check=True)
        collider_output = collider_result.stdout
        
        print("Running PySINDy and Ollama...")
        # 2. Run the SINDy extraction and Llama translation
        sindy_cmd = [
            sys.executable, "sindy_extract.py",
            "--file", "electron_trajectory.npy",
            "--context", "electron Møller scattering",
            "--model", "deepseek-r1:14b"
        ]
        result = subprocess.run(sindy_cmd, capture_output=True, text=True, encoding='utf-8', check=True)
        
        output = result.stdout
        
        # 3. Parse the output to separate Raw Math from Ollama Translation
        raw_math = ""
        translation = ""
        divergence_error = None
        
        if "T_SYMMETRY_DIVERGENCE=" in collider_output:
            match = re.search(r"T_SYMMETRY_DIVERGENCE=([0-9.]+)", collider_output)
            if match:
                divergence_error = match.group(1)
                
        total_radiation_shed = None
        if "TOTAL_RADIATION_SHED=" in collider_output:
            match = re.search(r"TOTAL_RADIATION_SHED=([0-9.]+)", collider_output)
            if match:
                total_radiation_shed = float(match.group(1))
                
        firewall_triggered = "*** AMPS FIREWALL TRIGGERED" in collider_output
        
        if "Cleaned Math Equations:" in output:
            parts = output.split("Cleaned Math Equations:")
            math_section = parts[1].split("Piping Universal Physics output")[0].strip()
            raw_math = math_section
        elif "Raw Governing Force Equations (x0' = x_accel, etc):" in output:
            parts = output.split("Raw Governing Force Equations (x0' = x_accel, etc):")
            math_section = parts[1].split("========================================")[0].strip()
            raw_math = math_section
            
        if "OLLAMA TRANSLATION:" in output:
            translation_section = output.split("OLLAMA TRANSLATION:")[1].split("========================================")[1].strip()
            translation = translation_section
            
        return {
            "success": True,
            "raw_math": raw_math,
            "translation": translation,
            "full_log": collider_output + "\n" + output,
            "divergence_error": divergence_error,
            "total_radiation_shed": total_radiation_shed,
            "firewall_triggered": firewall_triggered
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.cmd}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
