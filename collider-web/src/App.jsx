import React, { useState } from 'react';

const ELEMENTS = [
    { symbol: 'e⁻', name: 'Electron', mass: 0.511 },
    { symbol: 'B*c', name: 'B*c (Tuned to 64.5 MeV CERN Target)', mass: 16.125 },
];

function App() {
    const [params, setParams] = useState({
        pauliScalar: 500.0,
        vacuumLambda: 0.001,
        torsionG: 1.0,
    });
    
    // Default to the B*c Tuned particle to hit the 64.5 MeV CERN Target
    const [particleA, setParticleA] = useState(ELEMENTS[1]);
    const [particleB, setParticleB] = useState(ELEMENTS[1]);
    const [particleC, setParticleC] = useState(ELEMENTS[1]);
    const [showModal, setShowModal] = useState(null); // 'A', 'B', or 'C'
    const [customMass, setCustomMass] = useState('');
    const [mode, setMode] = useState("2-body-collision");
    const [tSymmetry, setTSymmetry] = useState(false);
    const [selectedPaper, setSelectedPaper] = useState(4);

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [lastTime, setLastTime] = useState(null);
    const [totalTime, setTotalTime] = useState(0);
    const [runCount, setRunCount] = useState(0);

    const [pauliEnabled, setPauliEnabled] = useState(true);
    const [vacuumEnabled, setVacuumEnabled] = useState(true);
    const [torsionEnabled, setTorsionEnabled] = useState(true);
    const [entangledEnabled, setEntangledEnabled] = useState(false);
    const [velocityClamp, setVelocityClamp] = useState(-1); // -1=auto, 0=off, 1=on
    const [sinkMass, setSinkMass] = useState(50000.0);
    const [numParticles, setNumParticles] = useState(50);
    const [collapseRadius, setCollapseRadius] = useState(20.0);
    const [collapseG, setCollapseG] = useState(1.0);
    const [slitWidth, setSlitWidth] = useState(2.0);
    const [slitSeparation, setSlitSeparation] = useState(6.0);
    const [beamMomentum, setBeamMomentum] = useState(5000.0);
    const [wallMass, setWallMass] = useState(1000.0);
    const [screenX, setScreenX] = useState(20.0);
    const [totalTicks, setTotalTicks] = useState(5000);
    const [activePreset, setActivePreset] = useState(null);
    const [showTonomuraModal, setShowTonomuraModal] = useState(false);
    const [customTrials, setCustomTrials] = useState('');
    const [numSlits, setNumSlits] = useState(2);
    const [wallDepth, setWallDepth] = useState(1);
    const [wallZLayers, setWallZLayers] = useState(1);
    const [aiTranslation, setAiTranslation] = useState(true);

    // =========================================================
    // EXPERIMENT PRESETS — One-click replication of published results
    // =========================================================
    const PRESETS = [
        // --- Paper 2 ---
        { id: 'p2-electron', paper: 2, label: 'Electron Collision', desc: '0.99c head-on, e⁻ vs e⁻',
          mode: '2-body-collision', massA: 0.511, massB: 0.511, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p2-bstar', paper: 2, label: 'B*c → 64.5 MeV', desc: 'Tuned to CERN target',
          mode: '2-body-collision', massA: 16.125, massB: 16.125, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p2-sink', paper: 2, label: 'Gravity Sink Orbit', desc: 'Entangled pair + M=50k sink',
          mode: 'gravity-sink', massA: 0.511, massB: 0.511, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: true, vClamp: 0, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        // --- Paper 3: Mass Sweep ---
        { id: 'p3-sweep-e', paper: 3, label: 'Sweep: Electron', desc: 'm₀ = 0.511 MeV',
          mode: 'gravity-sink', massA: 0.511, massB: 0.511, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: true, vClamp: 0, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p3-sweep-pi', paper: 3, label: 'Sweep: Pion', desc: 'm₀ = 134.98 MeV',
          mode: 'gravity-sink', massA: 134.98, massB: 134.98, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: true, vClamp: 0, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p3-sweep-k', paper: 3, label: 'Sweep: Kaon', desc: 'm₀ = 493.68 MeV',
          mode: 'gravity-sink', massA: 493.68, massB: 493.68, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: true, vClamp: 0, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p3-sweep-p', paper: 3, label: 'Sweep: Proton', desc: 'm₀ = 938.27 MeV',
          mode: 'gravity-sink', massA: 938.27, massB: 938.27, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: true, vClamp: 0, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false },
        // --- Paper 3: Direct Collapse ---
        { id: 'p3-collapse-10', paper: 3, label: 'Collapse N=10', desc: '100 MeV, R=20, G=1 (~15s)',
          mode: 'direct-collapse', massA: 100.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.01, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 10, cRadius: 20, cG: 1.0, tSym: false },
        { id: 'p3-collapse-50', paper: 3, label: 'Collapse N=50', desc: '100 MeV, R=20, G=1 (~6min)',
          mode: 'direct-collapse', massA: 100.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.01, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 2.0, slitS: 6.0, beamP: 5000, wallM: 1000, scrX: 20, tTicks: 5000 },
        // --- Paper 4: Double Slit ---
        // Wall mass 1e3: soft enough for beam to squeeze through, hard enough to stop direct hits.
        // Beam p=1000: slow enough for Pauli edge interaction to sort by θ_hue.
        { id: 'p4-slit-50', paper: 4, label: '1-Layer Classical Scatter', desc: 'p=1000, wall=1e3, 1-layer depth',
          mode: 'double-slit', massA: 100.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.0001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 2.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 2, wallD: 1, wallZ: 1 },
        { id: 'p4-slit-200', paper: 4, label: '1-Layer Classical Scatter (N=200)', desc: 'High-res histogram, p=1000, wall=1e3',
          mode: 'double-slit', massA: 100.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.0001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 200, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 2.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 2, wallD: 1, wallZ: 1 },
        { id: 'p4-slit-1slit', paper: 4, label: 'Single Slit Control', desc: 'One slit open, slit 2 filled with wall',
          mode: 'double-slit', massA: 100.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.0001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 50, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 2.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 1, wallD: 1, wallZ: 1 },
        { id: 'p4-tunnel-5', paper: 4, label: '5-Layer Phase Router', desc: 'Deep tunnel proves diffraction fringes',
          mode: 'double-slit', massA: 1.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 5000, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 4.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 2, wallD: 5, wallZ: 1 },
        { id: 'p4-tunnel-3d', paper: 4, label: '3D Phase Router (5x3)', desc: '3D copper matrix (5 depth, 3 z-layers)',
          mode: 'double-slit', massA: 1.0, massB: 1.0, massC: 1.0,
          pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
          entangled: false, vClamp: -1, sinkMass: 50000, nPart: 5000, cRadius: 20, cG: 1.0, tSym: false,
          slitW: 4.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 2, wallD: 5, wallZ: 3 },
    ];

    // Base config for Tonomura runs (trial count set by modal)
    const TONOMURA_BASE = {
        mode: 'double-slit', massA: 1.0, massB: 1.0, massC: 1.0,
        pauli: 500, vacuum: 0.001, torsion: 1.0, pauliOn: true, vacuumOn: true, torsionOn: true,
        entangled: false, vClamp: -1, sinkMass: 50000, cRadius: 20, cG: 1.0, tSym: false,
        slitW: 4.0, slitS: 6.0, beamP: 1000, wallM: 1000, scrX: 20, tTicks: 5000, nSlits: 2, wallD: 5, wallZ: 1
    };

    const applyTonomura = (nTrials) => {
        const preset = { ...TONOMURA_BASE, id: `p4-tonomura-${nTrials}`, paper: 4, label: `Tonomura ${nTrials >= 1000 ? (nTrials/1000)+'K' : nTrials}`, desc: `${nTrials} trials, screen@50`, nPart: nTrials };
        applyPreset(preset);
        setShowTonomuraModal(false);
        setCustomTrials('');
    };

    const applyPreset = (preset) => {
        setActivePreset(preset.id);
        setMode(preset.mode);
        setParticleA({ symbol: `${preset.massA}`, name: `Preset (${preset.massA} MeV)`, mass: preset.massA });
        setParticleB({ symbol: `${preset.massB}`, name: `Preset (${preset.massB} MeV)`, mass: preset.massB });
        setParticleC({ symbol: `${preset.massC}`, name: `Preset (${preset.massC} MeV)`, mass: preset.massC });
        setParams({ pauliScalar: preset.pauli, vacuumLambda: preset.vacuum, torsionG: preset.torsion });
        setPauliEnabled(preset.pauliOn);
        setVacuumEnabled(preset.vacuumOn);
        setTorsionEnabled(preset.torsionOn);
        setEntangledEnabled(preset.entangled);
        setVelocityClamp(preset.vClamp);
        setSinkMass(preset.sinkMass);
        setNumParticles(preset.nPart);
        setCollapseRadius(preset.cRadius);
        setCollapseG(preset.cG);
        setTSymmetry(preset.tSym);
        setSlitWidth(preset.slitW || 2.0);
        setSlitSeparation(preset.slitS || 6.0);
        setBeamMomentum(preset.beamP || 5000);
        setWallMass(preset.wallM || 1000);
        setScreenX(preset.scrX || 20);
        setTotalTicks(preset.tTicks || 5000);
        setNumSlits(preset.nSlits || 2);
        setWallDepth(preset.wallD || 1);
        setWallZLayers(preset.wallZ || 1);
    };

    const handleParamChange = (name, value) => {
        setActivePreset(null); // Clear preset highlight when manually changing
        setParams(prev => ({ ...prev, [name]: parseFloat(value) }));
    };

    const handleElementSelect = (element) => {
        if (showModal === 'A') setParticleA(element);
        if (showModal === 'B') setParticleB(element);
        if (showModal === 'C') setParticleC(element);
        setShowModal(null);
    };

    const handleExtract = async () => {
        const startTime = performance.now();
        setLoading(true);
        setError(null);
        setResult(null);
        
        try {
            const response = await fetch('http://localhost:8000/api/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: mode,
                    mass_a: particleA.mass,
                    mass_b: particleB.mass,
                    mass_c: particleC.mass,
                    pauli: params.pauliScalar,
                    vacuum: params.vacuumLambda,
                    torsion: params.torsionG,
                    t_symmetry: tSymmetry,
                    pauli_enabled: pauliEnabled,
                    vacuum_enabled: vacuumEnabled,
                    torsion_enabled: torsionEnabled,
                    entangled: entangledEnabled,
                    velocity_clamp: velocityClamp,
                    sink_mass: sinkMass,
                    num_particles: numParticles,
                    collapse_radius: collapseRadius,
                    collapse_G: collapseG,
                    slit_width: slitWidth,
                    slit_separation: slitSeparation,
                    beam_momentum: beamMomentum,
                    wall_mass: wallMass,
                    screen_x: screenX,
                    total_ticks: totalTicks,
                    batch_size: 0,
                    num_slits: numSlits,
                    wall_depth: wallDepth,
                    wall_z_layers: wallZLayers,
                    ai_translation: aiTranslation,
                    run_label: activePreset ? (PRESETS.find(p => p.id === activePreset)?.label || activePreset) : 'Custom'
                })
            });
            
            if (!response.ok) {
                const errData = await response.json();
                const errorMessage = typeof errData.detail === 'string' 
                    ? errData.detail 
                    : JSON.stringify(errData.detail, null, 2);
                throw new Error(errorMessage);
            }
            
            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            setLastTime(duration);
            setTotalTime(prev => prev + parseFloat(duration));
            setRunCount(prev => prev + 1);
            setLoading(false);
        }
    };

    // Calculate Live Diagnostics
    const totalMass = mode === 'direct-collapse' ? particleA.mass * numParticles : mode === 'double-slit' ? particleA.mass * numParticles : particleA.mass + particleB.mass + (mode.includes("3-body") ? particleC.mass : 0);
    const c = 100.0;
    const restEnergy = totalMass * c * c;
    
    // Estimate Gamma (tension) based on damping and mass
    // Less damping = higher speeds = higher tension. More mass = harder to accelerate = lower tension.
    // This is a pseudo-calculation for the dashboard visual
    const estimatedGamma = 1.0 + (1.0 / (params.vacuumLambda * totalMass + 0.0001)) * 0.001;
    const localGravity = (params.torsionG * totalMass) / 10000.0;

    return (
        <div className="dashboard-container" style={{ position: 'relative' }}>
            
            {/* PERIODIC TABLE MODAL */}
            {showModal && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, 
                    background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', 
                    alignItems: 'center', justifyContent: 'center'
                }}>
                    <div className="glass-panel" style={{ width: '500px', padding: '30px' }}>
                        <h2 style={{ color: '#00f0ff', marginBottom: '20px', textAlign: 'center' }}>
                            SELECT PARTICLE {showModal}
                        </h2>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '15px' }}>
                            {ELEMENTS.map(el => (
                                <div 
                                    key={el.symbol}
                                    onClick={() => handleElementSelect(el)}
                                    style={{
                                        background: 'rgba(0, 240, 255, 0.1)',
                                        border: '1px solid #00f0ff',
                                        borderRadius: '8px',
                                        padding: '15px',
                                        textAlign: 'center',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease'
                                    }}
                                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(0, 240, 255, 0.3)'}
                                    onMouseOut={(e) => e.currentTarget.style.background = 'rgba(0, 240, 255, 0.1)'}
                                >
                                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>{el.symbol}</div>
                                    <div style={{ fontSize: '0.9rem', color: '#aaa', marginTop: '5px' }}>{el.name}</div>
                                    <div style={{ fontSize: '0.7rem', color: '#ff00ff', marginTop: '5px' }}>m₀: {el.mass} MeV</div>
                                </div>
                            ))}
                        </div>
                        <div style={{ marginTop: '20px', padding: '15px', border: '1px dashed #ffcc00', borderRadius: '8px', background: 'rgba(255,204,0,0.05)' }}>
                            <div style={{ fontSize: '0.85rem', color: '#ffcc00', marginBottom: '10px', fontWeight: 'bold' }}>CUSTOM PARTICLE (m₀ in MeV)</div>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <input 
                                    type="number" 
                                    step="0.001" 
                                    min="0.001"
                                    value={customMass}
                                    onChange={(e) => setCustomMass(e.target.value)}
                                    placeholder="Enter mass in MeV..."
                                    style={{ flex: 1, padding: '10px', background: '#1a1a2e', border: '1px solid #ffcc00', borderRadius: '4px', color: '#fff', fontSize: '1rem' }}
                                />
                                <button
                                    onClick={() => {
                                        const m = parseFloat(customMass);
                                        if (m > 0) {
                                            handleElementSelect({ symbol: `${m}`, name: `Custom (${m} MeV)`, mass: m });
                                        }
                                    }}
                                    disabled={!customMass || parseFloat(customMass) <= 0}
                                    style={{ padding: '10px 20px', background: parseFloat(customMass) > 0 ? '#ffcc00' : '#333', color: '#000', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                                >
                                    SET
                                </button>
                            </div>
                        </div>
                        <button 
                            onClick={() => setShowModal(null)} 
                            style={{ width: '100%', marginTop: '20px', padding: '10px', background: 'transparent', border: '1px solid #aaa', color: '#aaa', cursor: 'pointer', borderRadius: '4px' }}
                        >
                            CANCEL
                        </button>
                    </div>
                </div>
            )}

            {/* TONOMURA TRIAL COUNT MODAL */}
            {showTonomuraModal && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, 
                    background: 'rgba(0,0,0,0.85)', zIndex: 1000, display: 'flex', 
                    alignItems: 'center', justifyContent: 'center'
                }}>
                    <div className="glass-panel" style={{ width: '420px', padding: '30px', border: '1px solid rgba(255,68,68,0.3)' }}>
                        <h2 style={{ color: '#ff4444', marginBottom: '8px', textAlign: 'center', fontSize: '1.2rem', letterSpacing: '2px' }}>
                            ⚡ TONOMURA PROTOCOL
                        </h2>
                        <div style={{ color: '#888', fontSize: '0.75rem', textAlign: 'center', marginBottom: '20px' }}>
                            Sequential single-particle trials · GPU auto-batch · Screen @ x=50
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                            {[
                                { n: 500, label: '500', time: '~1 min (GPU)', desc: 'Quick validation' },
                                { n: 1000, label: '1K', time: '~2 min (GPU)', desc: 'Publication stats' },
                                { n: 5000, label: '5K', time: '~8 min (GPU)', desc: 'Full Tonomura' },
                            ].map(opt => (
                                <div key={opt.n}
                                    onClick={() => applyTonomura(opt.n)}
                                    style={{
                                        background: 'rgba(255,68,68,0.08)',
                                        border: '1px solid rgba(255,68,68,0.3)',
                                        borderRadius: '8px',
                                        padding: '18px 12px',
                                        textAlign: 'center',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease'
                                    }}
                                    onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,68,68,0.25)'}
                                    onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,68,68,0.08)'}
                                >
                                    <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#fff' }}>{opt.label}</div>
                                    <div style={{ fontSize: '0.75rem', color: '#ff4444', marginTop: '4px' }}>{opt.time}</div>
                                    <div style={{ fontSize: '0.65rem', color: '#888', marginTop: '4px' }}>{opt.desc}</div>
                                </div>
                            ))}
                        </div>
                        <div style={{ marginTop: '16px', padding: '12px', border: '1px dashed #ff4444', borderRadius: '8px', background: 'rgba(255,68,68,0.03)' }}>
                            <div style={{ fontSize: '0.8rem', color: '#ff4444', marginBottom: '8px', fontWeight: 'bold' }}>CUSTOM TRIAL COUNT</div>
                            <div style={{ display: 'flex', gap: '10px' }}>
                                <input 
                                    type="number" 
                                    step="100" 
                                    min="10"
                                    value={customTrials}
                                    onChange={(e) => setCustomTrials(e.target.value)}
                                    placeholder="Enter trial count..."
                                    style={{ flex: 1, padding: '10px', background: '#1a1a2e', border: '1px solid #ff4444', borderRadius: '4px', color: '#fff', fontSize: '1rem' }}
                                />
                                <button
                                    onClick={() => {
                                        const n = parseInt(customTrials);
                                        if (n > 0) applyTonomura(n);
                                    }}
                                    disabled={!customTrials || parseInt(customTrials) <= 0}
                                    style={{ padding: '10px 20px', background: parseInt(customTrials) > 0 ? '#ff4444' : '#333', color: parseInt(customTrials) > 0 ? '#000' : '#666', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                                >
                                    SET
                                </button>
                            </div>
                            {customTrials && parseInt(customTrials) > 0 && (
                                <div style={{ fontSize: '0.7rem', color: '#888', marginTop: '6px' }}>
                                    Est. time: ~{((parseInt(customTrials) * 2.3) / 60).toFixed(0)} min · ~{Math.round(parseInt(customTrials) * 0.21)} expected hits
                                </div>
                            )}
                        </div>
                        <button 
                            onClick={() => { setShowTonomuraModal(false); setCustomTrials(''); }}
                            style={{ width: '100%', marginTop: '16px', padding: '10px', background: 'transparent', border: '1px solid #555', color: '#888', cursor: 'pointer', borderRadius: '4px', fontSize: '0.85rem' }}
                        >
                            CANCEL
                        </button>
                    </div>
                </div>
            )}

            {/* LEFT CONTROL PANEL */}
            <div className="sidebar" style={{ width: '380px' }}>
                <div className="header">
                    <h1>AI PHYSICS ENGINE</h1>
                    <p>Teleparallel Collision Discovery</p>
                </div>

                {/* EXPERIMENT PRESETS */}
                <div className="glass-panel" style={{ marginBottom: '20px' }}>
                    <h3 style={{ color: '#ffcc00', fontSize: '0.9rem', letterSpacing: '1px', margin: '0 0 12px 0' }}>
                        📋 EXPERIMENT PRESETS
                    </h3>
                    <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '10px' }}>
                        One-click replication of published results
                    </div>

                    <div style={{ marginBottom: '15px' }}>
                        <select
                            value={selectedPaper}
                            onChange={(e) => setSelectedPaper(parseInt(e.target.value))}
                            style={{ width: '100%', background: '#111', color: '#fff', border: '1px solid #444', padding: '6px', borderRadius: '4px', outline: 'none', cursor: 'pointer' }}
                        >
                            <option value={2}>Paper 2 — Firewall & ER=EPR</option>
                            <option value={3}>Paper 3 — Mass Sweep & Collapse</option>
                            <option value={4}>Paper 4 — Double Slit & Tonomura</option>
                        </select>
                    </div>

                    {/* Paper 2 */}
                    {selectedPaper === 2 && (
                        <div style={{ marginBottom: '8px' }}>
                            <div style={{ fontSize: '0.65rem', color: '#00f0ff', fontWeight: 'bold', marginBottom: '4px', letterSpacing: '1px' }}>FIREWALL &amp; ER=EPR</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                {PRESETS.filter(p => p.paper === 2).map(p => (
                                    <button key={p.id} onClick={() => applyPreset(p)} title={p.desc}
                                        style={{
                                            background: activePreset === p.id ? '#00f0ff' : 'rgba(0,240,255,0.08)',
                                            color: activePreset === p.id ? '#000' : '#aaa',
                                            border: activePreset === p.id ? '1px solid #00f0ff' : '1px solid rgba(0,240,255,0.2)',
                                            padding: '4px 8px', borderRadius: '4px', fontSize: '0.65rem',
                                            cursor: 'pointer', fontWeight: activePreset === p.id ? 'bold' : 'normal',
                                            transition: 'all 0.2s ease'
                                        }}
                                    >{p.label}</button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Paper 3: Mass Sweep & Direct Collapse */}
                    {selectedPaper === 3 && (
                        <>
                            <div style={{ marginBottom: '8px' }}>
                                <div style={{ fontSize: '0.65rem', color: '#ff00ff', fontWeight: 'bold', marginBottom: '4px', letterSpacing: '1px' }}>MASS SWEEP</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    {PRESETS.filter(p => p.paper === 3 && p.id.includes('sweep')).map(p => (
                                        <button key={p.id} onClick={() => applyPreset(p)} title={p.desc}
                                            style={{
                                                background: activePreset === p.id ? '#ff00ff' : 'rgba(255,0,255,0.08)',
                                                color: activePreset === p.id ? '#000' : '#aaa',
                                                border: activePreset === p.id ? '1px solid #ff00ff' : '1px solid rgba(255,0,255,0.2)',
                                                padding: '4px 8px', borderRadius: '4px', fontSize: '0.65rem',
                                                cursor: 'pointer', fontWeight: activePreset === p.id ? 'bold' : 'normal',
                                                transition: 'all 0.2s ease'
                                            }}
                                        >{p.label}</button>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.65rem', color: '#ff6600', fontWeight: 'bold', marginBottom: '4px', letterSpacing: '1px' }}>DIRECT COLLAPSE</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    {PRESETS.filter(p => p.paper === 3 && p.id.includes('collapse')).map(p => (
                                        <button key={p.id} onClick={() => applyPreset(p)} title={p.desc}
                                            style={{
                                                background: activePreset === p.id ? '#ff6600' : 'rgba(255,102,0,0.08)',
                                                color: activePreset === p.id ? '#000' : '#aaa',
                                                border: activePreset === p.id ? '1px solid #ff6600' : '1px solid rgba(255,102,0,0.2)',
                                                padding: '4px 8px', borderRadius: '4px', fontSize: '0.65rem',
                                                cursor: 'pointer', fontWeight: activePreset === p.id ? 'bold' : 'normal',
                                                transition: 'all 0.2s ease'
                                            }}
                                        >{p.label}</button>
                                    ))}
                                </div>
                            </div>
                        </>
                    )}

                    {/* Paper 4: Double Slit & Tonomura Protocol */}
                    {selectedPaper === 4 && (
                        <>
                            <div style={{ marginBottom: '8px' }}>
                                <div style={{ fontSize: '0.65rem', color: '#00e5ff', fontWeight: 'bold', marginBottom: '4px', letterSpacing: '1px' }}>DOUBLE SLIT & TUNNEL</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    {PRESETS.filter(p => p.paper === 4).map(p => (
                                        <button key={p.id} onClick={() => applyPreset(p)} title={p.desc}
                                            style={{
                                                background: activePreset === p.id ? '#00e5ff' : 'rgba(0,229,255,0.08)',
                                                color: activePreset === p.id ? '#000' : '#aaa',
                                                border: activePreset === p.id ? '1px solid #00e5ff' : '1px solid rgba(0,229,255,0.2)',
                                                padding: '4px 8px', borderRadius: '4px', fontSize: '0.65rem',
                                                cursor: 'pointer', fontWeight: activePreset === p.id ? 'bold' : 'normal',
                                                transition: 'all 0.2s ease'
                                            }}
                                        >{p.label}</button>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.65rem', color: '#ff4444', fontWeight: 'bold', marginBottom: '4px', letterSpacing: '1px' }}>TONOMURA PROTOCOL</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                    <button onClick={() => setShowTonomuraModal(true)}
                                        style={{
                                            background: activePreset?.startsWith?.('p4-tonomura') || (typeof activePreset === 'string' && activePreset.startsWith('p4-tonomura')) ? '#ff4444' : 'rgba(255,68,68,0.08)',
                                            color: (typeof activePreset === 'string' && activePreset.startsWith('p4-tonomura')) ? '#000' : '#aaa',
                                            border: (typeof activePreset === 'string' && activePreset.startsWith('p4-tonomura')) ? '1px solid #ff4444' : '1px solid rgba(255,68,68,0.2)',
                                            padding: '4px 12px', borderRadius: '4px', fontSize: '0.65rem',
                                            cursor: 'pointer', fontWeight: 'bold',
                                            transition: 'all 0.2s ease'
                                        }}
                                    >⚡ TONOMURA RUN</button>
                                </div>
                            </div>
                        </>
                    )}
                    {activePreset && (
                        <div style={{ marginTop: '10px', padding: '8px', background: 'rgba(255,204,0,0.05)', border: '1px solid rgba(255,204,0,0.2)', borderRadius: '4px' }}>
                            <div style={{ fontSize: '0.75rem', color: '#ffcc00' }}>
                                ✓ {PRESETS.find(p => p.id === activePreset)?.label}
                            </div>
                            <div style={{ fontSize: '0.65rem', color: '#888', marginTop: '2px' }}>
                                {PRESETS.find(p => p.id === activePreset)?.desc} — All parameters loaded. Hit EXTRACT to run.
                            </div>
                        </div>
                    )}
                </div>
                
                {/* AUTO-LOADER */}
                <div className="glass-panel" style={{ marginBottom: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                        <h3 style={{ color: '#00f0ff', fontSize: '0.9rem', letterSpacing: '1px', margin: 0 }}>INITIAL CONDITIONS</h3>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', background: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: '4px' }}>
                            <button onClick={() => setMode('2-body-collision')} style={{ background: mode === '2-body-collision' ? '#00f0ff' : 'transparent', color: mode === '2-body-collision' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>2-BODY COLLISION</button>
                            <button onClick={() => setMode('3-body-scatter')} style={{ background: mode === '3-body-scatter' ? '#ff00ff' : 'transparent', color: mode === '3-body-scatter' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>3-BODY SCATTER</button>
                            <button onClick={() => setMode('3-body-orbit')} style={{ background: mode === '3-body-orbit' ? '#ffcc00' : 'transparent', color: mode === '3-body-orbit' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>3-BODY ORBIT</button>
                            <button onClick={() => setMode('gravity-sink')} style={{ background: mode === 'gravity-sink' ? '#00ff00' : 'transparent', color: mode === 'gravity-sink' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>GRAVITY SINK</button>
                            <button onClick={() => setMode('direct-collapse')} style={{ background: mode === 'direct-collapse' ? '#ff6600' : 'transparent', color: mode === 'direct-collapse' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>DIRECT COLLAPSE</button>
                            <button onClick={() => setMode('double-slit')} style={{ background: mode === 'double-slit' ? '#00e5ff' : 'transparent', color: mode === 'double-slit' ? '#000' : '#888', border: 'none', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold' }}>DOUBLE SLIT</button>
                            
                            <button onClick={() => setTSymmetry(!tSymmetry)} style={{ background: tSymmetry ? '#ffcc00' : 'transparent', color: tSymmetry ? '#000' : '#888', border: '1px solid #ffcc00', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold', marginLeft: '5px' }}>{tSymmetry ? 'T-SYMMETRY: ON' : 'T-SYMMETRY: OFF'}</button>
                            <button onClick={() => setAiTranslation(!aiTranslation)} style={{ background: aiTranslation ? '#bb86fc' : 'transparent', color: aiTranslation ? '#000' : '#888', border: '1px solid #bb86fc', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold', marginLeft: '5px' }}>{aiTranslation ? 'AI SINDY: ON' : 'AI SINDY: OFF'}</button>
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <div 
                            onClick={() => setShowModal('A')}
                            style={{ flex: 1, border: '1px dashed #00f0ff', padding: '15px', textAlign: 'center', borderRadius: '8px', cursor: 'pointer', background: 'rgba(0,240,255,0.05)' }}
                        >
                            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '5px' }}>PARTICLE A</div>
                            <div style={{ fontSize: '1.8rem', color: '#fff', fontWeight: 'bold' }}>{particleA.symbol}</div>
                        </div>
                        {mode !== 'direct-collapse' && mode !== 'double-slit' && (<>
                        <div style={{ display: 'flex', alignItems: 'center', color: '#ff00ff', fontWeight: 'bold' }}>VS</div>
                        <div 
                            onClick={() => setShowModal('B')}
                            style={{ flex: 1, border: '1px dashed #ff00ff', padding: '15px', textAlign: 'center', borderRadius: '8px', cursor: 'pointer', background: 'rgba(255,0,255,0.05)' }}
                        >
                            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '5px' }}>PARTICLE B</div>
                            <div style={{ fontSize: '1.8rem', color: '#fff', fontWeight: 'bold' }}>{particleB.symbol}</div>
                        </div>
                        </>)}
                        {(mode === '3-body-scatter' || mode === '3-body-orbit') && (
                            <>
                                <div style={{ display: 'flex', alignItems: 'center', color: '#ffcc00', fontWeight: 'bold' }}>VS</div>
                                <div 
                                    onClick={() => setShowModal('C')}
                                    style={{ flex: 1, border: '1px dashed #ffcc00', padding: '15px', textAlign: 'center', borderRadius: '8px', cursor: 'pointer', background: 'rgba(255,204,0,0.05)' }}
                                >
                                    <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '5px' }}>PARTICLE C</div>
                                    <div style={{ fontSize: '1.8rem', color: '#fff', fontWeight: 'bold' }}>{particleC.symbol}</div>
                                </div>
                            </>
                        )}
                        {mode === 'direct-collapse' && (
                            <div style={{ flex: 1, border: '1px dashed #ff6600', padding: '15px', textAlign: 'center', borderRadius: '8px', background: 'rgba(255,102,0,0.05)' }}>
                                <div style={{ fontSize: '0.7rem', color: '#aaa', marginBottom: '3px' }}>ALL {numParticles} PARTICLES</div>
                                <div style={{ fontSize: '1.4rem', color: '#ff6600', fontWeight: 'bold' }}>{particleA.symbol} × {numParticles}</div>
                                <div style={{ fontSize: '0.7rem', color: '#888', marginTop: '3px' }}>Mutual Gravity · No Anchor</div>
                            </div>
                        )}
                        {mode === 'double-slit' && (
                            <div style={{ flex: 1, border: '1px dashed #00e5ff', padding: '15px', textAlign: 'center', borderRadius: '8px', background: 'rgba(0,229,255,0.05)' }}>
                                <div style={{ fontSize: '0.7rem', color: '#aaa', marginBottom: '3px' }}>BEAM: {numParticles} PARTICLES</div>
                                <div style={{ fontSize: '1.4rem', color: '#00e5ff', fontWeight: 'bold' }}>{particleA.symbol} × {numParticles}</div>
                                <div style={{ fontSize: '0.7rem', color: '#888', marginTop: '3px' }}>Same momentum · Random θ<sub>hue</sub></div>
                            </div>
                        )}
                    </div>
                </div>

                <div className="control-group glass-panel">
                    <h3 style={{ color: '#00f0ff', fontSize: '0.9rem', marginBottom: '15px', letterSpacing: '1px' }}>UNIVERSAL RULES</h3>
                    <div className="slider-container" title="Represents the inverse-square repulsion barrier between fermions." style={{ opacity: pauliEnabled ? 1 : 0.4 }}>
                        <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                            <input 
                                type="checkbox" 
                                checked={pauliEnabled} 
                                onChange={(e) => setPauliEnabled(e.target.checked)} 
                                style={{ marginRight: '10px', cursor: 'pointer' }}
                            />
                            <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Pauli Spring (F_p)</span>
                            <span className="slider-value">{params.pauliScalar.toFixed(1)}</span>
                        </div>
                        <input type="range" min="0" max="1000" step="10" 
                               value={params.pauliScalar} onChange={(e) => handleParamChange('pauliScalar', e.target.value)} disabled={!pauliEnabled} />
                    </div>

                    <div className="slider-container" title="Hydrodynamic drag in the lattice enforcing a speed limit." style={{ opacity: vacuumEnabled ? 1 : 0.4 }}>
                        <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                            <input 
                                type="checkbox" 
                                checked={vacuumEnabled} 
                                onChange={(e) => setVacuumEnabled(e.target.checked)} 
                                style={{ marginRight: '10px', cursor: 'pointer' }}
                            />
                            <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Kinetic Damping (&Lambda;)</span>
                            <span className="slider-value">{params.vacuumLambda.toFixed(4)}</span>
                        </div>
                        <input type="range" min="0" max="0.05" step="0.001" 
                               value={params.vacuumLambda} onChange={(e) => handleParamChange('vacuumLambda', e.target.value)} disabled={!vacuumEnabled} />
                    </div>
                    
                    <div className="slider-container" title="Background twisting shear creating cross-axis coupling (Lorentz-like force)." style={{ opacity: torsionEnabled ? 1 : 0.4 }}>
                        <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                            <input 
                                type="checkbox" 
                                checked={torsionEnabled} 
                                onChange={(e) => setTorsionEnabled(e.target.checked)} 
                                style={{ marginRight: '10px', cursor: 'pointer' }}
                            />
                            <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Torsion Tensor (T_μν)</span>
                            <span className="slider-value">{params.torsionG.toFixed(2)}</span>
                        </div>
                        <input type="range" min="-10" max="10" step="0.5" 
                               value={params.torsionG} onChange={(e) => handleParamChange('torsionG', e.target.value)} disabled={!torsionEnabled} />
                    </div>
                    
                    <div className="slider-container" title="ER=EPR Wormhole Non-Local Entanglement" style={{ opacity: entangledEnabled ? 1 : 0.4 }}>
                        <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                            <input 
                                type="checkbox" 
                                checked={entangledEnabled} 
                                onChange={(e) => setEntangledEnabled(e.target.checked)} 
                                style={{ marginRight: '10px', cursor: 'pointer' }}
                            />
                            <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>ER=EPR Wormhole (W_ij)</span>
                            <span className="slider-value" style={{ color: entangledEnabled ? '#00f0ff' : '#888' }}>
                                {entangledEnabled ? 'ENTANGLED' : 'LOCAL'}
                            </span>
                        </div>
                    </div>

                    <div className="slider-container" title="Velocity Clamp: ON = 0.99c cap (Paper 1 swarm), OFF = free momentum (Paper 2 firewall), Auto = mode-dependent">
                        <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                            <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Velocity Clamp</span>
                            <select 
                                value={velocityClamp} 
                                onChange={(e) => setVelocityClamp(parseInt(e.target.value))}
                                style={{ background: '#1a1a2e', color: velocityClamp === 1 ? '#ff6600' : velocityClamp === 0 ? '#00ff00' : '#00f0ff', border: '1px solid #333', borderRadius: '4px', padding: '4px 8px', fontSize: '0.8rem', cursor: 'pointer' }}
                            >
                                <option value={-1}>AUTO</option>
                                <option value={0}>OFF (Paper 2)</option>
                                <option value={1}>ON (Paper 1)</option>
                            </select>
                        </div>
                    </div>

                    {mode === 'gravity-sink' && (
                        <div className="slider-container" title="Mass of the central gravity sink.">
                            <div className="slider-header">
                                <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Sink Mass (M)</span>
                                <span className="slider-value">{sinkMass.toFixed(0)}</span>
                            </div>
                            <input type="range" min="100" max="100000" step="100" 
                                   value={sinkMass} onChange={(e) => setSinkMass(parseFloat(e.target.value))} />
                        </div>
                    )}

                    {mode === 'direct-collapse' && (
                        <>
                            <div style={{ borderTop: '1px solid #ff6600', margin: '10px 0', paddingTop: '10px' }}>
                                <div style={{ fontSize: '0.8rem', color: '#ff6600', fontWeight: 'bold', marginBottom: '10px', letterSpacing: '1px' }}>COLLAPSE PARAMETERS</div>
                            </div>
                            <div className="slider-container" title="Number of particles in the cloud.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>N Particles</span>
                                    <span className="slider-value">{numParticles}</span>
                                </div>
                                <input type="range" min="5" max="100" step="5" 
                                       value={numParticles} onChange={(e) => setNumParticles(parseInt(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Initial sphere radius for particle placement.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Sphere Radius (R)</span>
                                    <span className="slider-value">{collapseRadius.toFixed(1)}</span>
                                </div>
                                <input type="range" min="5" max="50" step="1" 
                                       value={collapseRadius} onChange={(e) => setCollapseRadius(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Gravitational coupling constant between particles.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Gravity (G)</span>
                                    <span className="slider-value">{collapseG.toFixed(1)}</span>
                                </div>
                                <input type="range" min="0.1" max="50" step="0.1" 
                                       value={collapseG} onChange={(e) => setCollapseG(parseFloat(e.target.value))} />
                            </div>
                        </>
                    )}

                    {mode === 'double-slit' && (
                        <>
                            <div style={{ borderTop: '1px solid #00e5ff', margin: '10px 0', paddingTop: '10px' }}>
                                <div style={{ fontSize: '0.8rem', color: '#00e5ff', fontWeight: 'bold', marginBottom: '10px', letterSpacing: '1px' }}>DOUBLE-SLIT PARAMETERS</div>
                            </div>
                            <div className="slider-container" title="Number of beam particles (each with random hue).">
                                <div className="slider-header" style={{ display: 'flex', alignItems: 'center' }}>
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>N Beam Particles</span>
                                    <input 
                                        type="number" 
                                        min="1" max="10000" 
                                        value={numParticles} 
                                        onChange={(e) => setNumParticles(parseInt(e.target.value) || 1)} 
                                        style={{ width: '60px', background: 'transparent', color: '#00e5ff', border: '1px solid #00e5ff', borderRadius: '4px', textAlign: 'right', padding: '2px 5px', fontSize: '0.85rem' }} 
                                    />
                                </div>
                                <input type="range" min="10" max="10000" step="10" 
                                       value={numParticles} onChange={(e) => setNumParticles(parseInt(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Width of each slit opening.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Slit Width</span>
                                    <span className="slider-value">{slitWidth.toFixed(1)}</span>
                                </div>
                                <input type="range" min="0.5" max="5.0" step="0.1" 
                                       value={slitWidth} onChange={(e) => setSlitWidth(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Center-to-center distance between the two slits.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Slit Separation</span>
                                    <span className="slider-value">{slitSeparation.toFixed(1)}</span>
                                </div>
                                <input type="range" min="2" max="15" step="0.5" 
                                       value={slitSeparation} onChange={(e) => setSlitSeparation(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Forward momentum of each beam particle.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Beam Momentum (p<sub>x</sub>)</span>
                                    <span className="slider-value">{beamMomentum.toFixed(0)}</span>
                                </div>
                                <input type="range" min="100" max="20000" step="50" 
                                       value={beamMomentum} onChange={(e) => setBeamMomentum(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Mass of wall particles. Lower = softer barrier (beam squeezes through). Higher = harder barrier.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Wall Mass (m<sub>wall</sub>)</span>
                                    <span className="slider-value">{wallMass.toExponential(0)}</span>
                                </div>
                                <input type="range" min="100" max="1000000" step="100" 
                                       value={wallMass} onChange={(e) => setWallMass(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="X-position of the detector screen. Farther = wider band separation.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Screen Distance (x)</span>
                                    <span className="slider-value">{screenX.toFixed(0)}</span>
                                </div>
                                <input type="range" min="10" max="200" step="5" 
                                       value={screenX} onChange={(e) => setScreenX(parseFloat(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Total simulation ticks per trial. Increase for farther screens.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Total Ticks</span>
                                    <span className="slider-value">{totalTicks}</span>
                                </div>
                                <input type="range" min="2000" max="20000" step="1000" 
                                       value={totalTicks} onChange={(e) => setTotalTicks(parseInt(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Wall depth in x-layers. >1 creates a tunnel the electron must navigate.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Wall Depth (x-layers)</span>
                                    <span className="slider-value">{wallDepth}</span>
                                </div>
                                <input type="range" min="1" max="50" step="1" 
                                       value={wallDepth} onChange={(e) => setWallDepth(parseInt(e.target.value))} />
                            </div>
                            <div className="slider-container" title="Wall layers in z-axis. >1 creates 3D tunnel compression.">
                                <div className="slider-header">
                                    <span className="slider-label" style={{borderBottom: '1px dotted #888', flex: 1}}>Z-Layers (3D depth)</span>
                                    <span className="slider-value">{wallZLayers}</span>
                                </div>
                                <input type="range" min="1" max="10" step="1" 
                                       value={wallZLayers} onChange={(e) => setWallZLayers(parseInt(e.target.value))} />
                            </div>
                        </>
                    )}
                </div>

                <button 
                    className="btn-primary" 
                    onClick={handleExtract} 
                    disabled={loading}
                    style={{ opacity: loading ? 0.5 : 1, marginBottom: '20px' }}
                >
                    {loading ? "RUNNING SIMULATION..." : "EXTRACT GOVERNING MATH"}
                </button>

                <div className="glass-panel" style={{ padding: '15px', marginTop: 'auto' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                        <span style={{ color: '#aaa', fontSize: '0.85rem' }}>Last Run Time:</span>
                        <span style={{ color: '#00f0ff', fontWeight: 'bold' }}>{lastTime ? `${lastTime}s` : '--'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: '#aaa', fontSize: '0.85rem' }}>Avg Run Time:</span>
                        <span style={{ color: '#ff00ff', fontWeight: 'bold' }}>{runCount > 0 ? `${(totalTime / runCount).toFixed(2)}s` : '--'}</span>
                    </div>
                </div>

                <div style={{ marginTop: '15px', padding: '10px', fontSize: '0.65rem', color: '#666', borderTop: '1px solid #333', textAlign: 'center' }}>
                    <strong>DISCLAIMER:</strong> This toy model is not proposing a physical medium of the nothingness of space (i.e. no aether). Particles are treated as resonant wave defects strictly for modeling topological teleparallel gravity equations.
                </div>
            </div>

            {/* RIGHT DASHBOARD */}
            <div className="canvas-container" style={{ padding: '24px', overflowY: 'auto' }}>
                
                {/* LIVE DIAGNOSTICS */}
                <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
                    <div className="glass-panel" style={{ flex: 1, padding: '20px', borderTop: '3px solid #ff00ff' }} title="Calculated from E₀ = m₀c². The innate potential energy of the collision system.">
                        <div style={{ color: '#aaa', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', borderBottom: '1px dotted #888', display: 'inline-block' }}>Total Rest Energy</div>
                        <div style={{ fontSize: '2rem', color: '#fff', fontWeight: 'bold' }}>{restEnergy.toExponential(2)}</div>
                    </div>
                    <div className="glass-panel" style={{ flex: 1, padding: '20px', borderTop: '3px solid #00f0ff' }} title="Calculated tension based on resistance to kinetic damping.">
                        <div style={{ color: '#aaa', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', borderBottom: '1px dotted #888', display: 'inline-block' }}>Relativistic Tension (γ)</div>
                        <div style={{ fontSize: '2rem', color: '#fff', fontWeight: 'bold' }}>{estimatedGamma.toFixed(4)}</div>
                    </div>
                    <div className="glass-panel" style={{ flex: 1, padding: '20px', borderTop: '3px solid #ffcc00' }} title="Warping of local space due to mass and torsion tensor.">
                        <div style={{ color: '#aaa', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', borderBottom: '1px dotted #888', display: 'inline-block' }}>Local Gravity Gradient</div>
                        <div style={{ fontSize: '2rem', color: '#fff', fontWeight: 'bold' }}>{localGravity > 0 ? '+' : ''}{localGravity.toFixed(2)}</div>
                    </div>
                </div>
                
                {result && (
                    <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
                        {typeof result.total_radiation_shed === 'number' && (
                            <div className="glass-panel" style={{ flex: 1, padding: '20px', borderTop: '3px solid #00ff00', background: 'rgba(0, 255, 0, 0.05)' }} title="Energy shed by core mass decay post-firewall.">
                                <div style={{ color: '#aaa', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', borderBottom: '1px dotted #888', display: 'inline-block' }}>Total Radiation Shed (MeV)</div>
                                <div style={{ fontSize: '2rem', color: '#00ff00', fontWeight: 'bold' }}>{result.total_radiation_shed.toFixed(4)}</div>
                            </div>
                        )}
                        <div className="glass-panel" style={{ flex: 1, padding: '20px', borderTop: `3px solid ${result.firewall_triggered ? '#ff4444' : '#00f0ff'}` }}>
                            <div style={{ color: '#aaa', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', borderBottom: '1px dotted #888', display: 'inline-block' }}>Wormhole Status</div>
                            <div style={{ fontSize: '1.5rem', color: result.firewall_triggered ? '#ff4444' : '#00f0ff', fontWeight: 'bold' }}>
                                {result.firewall_triggered ? 'SNAPPED (FIREWALL)' : 'BONDED'}
                            </div>
                        </div>
                    </div>
                )}

                <div className="glass-panel" style={{ minHeight: 'calc(100% - 150px)', display: 'flex', flexDirection: 'column' }}>
                    <h2 style={{ color: '#00f0ff', marginBottom: '20px', textTransform: 'uppercase', letterSpacing: '2px', fontSize: '1.2rem' }}>
                        AI Physics Synthesis
                    </h2>
                    
                    {result && result.divergence_error && (
                        <div style={{ background: 'rgba(255,204,0,0.1)', borderLeft: '4px solid #ffcc00', padding: '15px', marginBottom: '20px' }}>
                            <h4 style={{ color: '#ffcc00', margin: '0 0 10px 0', fontSize: '0.9rem', letterSpacing: '1px' }}>T-SYMMETRY TIME REVERSAL TEST</h4>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: '#ccc' }}>Mathematical Divergence Error from Origin:</span>
                                <span style={{ color: '#fff', fontWeight: 'bold', fontFamily: 'monospace' }}>{result.divergence_error}</span>
                            </div>
                            <div style={{ fontSize: '0.8rem', color: '#888', marginTop: '10px' }}>
                                (0.000 = Perfectly reversible integrable system. Large numbers = Chaotic floating-point divergence or unrecoverable entropy).
                            </div>
                        </div>
                    )}
                    
                    {loading && (
                        <div style={{ margin: 'auto', textAlign: 'center', color: '#ff00ff' }}>
                            <div className="spinner" style={{ 
                                width: '40px', height: '40px', border: '3px solid rgba(255,0,255,0.3)', 
                                borderTopColor: '#ff00ff', borderRadius: '50%', margin: '0 auto',
                                animation: 'spin 1s linear infinite'
                            }}></div>
                            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                            <p style={{ marginTop: '20px', letterSpacing: '1px', lineHeight: '1.8' }}>
                                SMASHING {particleA.symbol}, {particleB.symbol}{mode.includes('3-body') ? ` AND ${particleC.symbol}` : ''}<br/>
                                &darr;<br/>
                                PySINDy mapping trajectory<br/>
                                &darr;<br/>
                                DeepSeek R1 (14B) synthesizing math...
                            </p>
                        </div>
                    )}
                    
                    {error && (
                        <div style={{ color: '#ff4444', background: 'rgba(255, 0, 0, 0.1)', padding: '20px', borderRadius: '8px', border: '1px solid #ff4444' }}>
                            <h3 style={{ marginBottom: '10px', fontSize: '1.1rem' }}>CONNECTION FAILED</h3>
                            <p>{error}</p>
                            <p style={{ marginTop: '15px', fontSize: '0.9em', opacity: 0.8 }}>
                                Have you started the FastAPI server?<br/>
                                Ensure you have run:<br/>
                                <code style={{ display: 'block', padding: '10px', background: 'rgba(0,0,0,0.5)', marginTop: '5px' }}>
                                    python api.py
                                </code>
                            </p>
                        </div>
                    )}
                    
                    {result && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
                            <div style={{ background: 'rgba(0, 240, 255, 0.05)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid #00f0ff' }}>
                                <h3 style={{ color: '#00f0ff', marginBottom: '15px', fontSize: '0.9em', letterSpacing: '1px' }}>RAW PYSINDY DIFFERENTIAL EQUATIONS</h3>
                                <pre style={{ fontFamily: 'monospace', color: '#e0e0e0', whiteSpace: 'pre-wrap', fontSize: '14px', lineHeight: '1.4' }}>
                                    {result.raw_math || "No raw math extracted."}
                                </pre>
                            </div>
                            
                            <div style={{ background: 'rgba(255, 0, 255, 0.05)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid #ff00ff' }}>
                                <h3 style={{ color: '#ff00ff', marginBottom: '15px', fontSize: '0.9em', letterSpacing: '1px' }}>DEEPSEEK R1 (14B) SYNTHESIS</h3>
                                <div style={{ lineHeight: '1.6', color: '#e0e0e0', whiteSpace: 'pre-wrap', fontSize: '15px' }}>
                                    {result.translation || "No translation generated."}
                                </div>
                            </div>
                            
                            {result.double_slit && (
                            <div style={{ background: 'rgba(0,229,255,0.05)', padding: '20px', borderRadius: '8px', borderLeft: '4px solid #00e5ff' }}>
                                <h3 style={{ color: '#00e5ff', marginBottom: '15px', fontSize: '0.9em', letterSpacing: '1px' }}>DOUBLE-SLIT DETECTOR SCREEN</h3>
                                <div style={{ display: 'flex', gap: '20px', marginBottom: '15px', fontSize: '0.85rem' }}>
                                    <span style={{ color: '#aaa' }}>Beam: <strong style={{ color: '#fff' }}>{result.double_slit.beam_total}</strong></span>
                                    <span style={{ color: '#aaa' }}>Screen hits: <strong style={{ color: '#00ff00' }}>{result.double_slit.screen_hits}</strong></span>
                                    <span style={{ color: '#aaa' }}>Reflected: <strong style={{ color: '#ff4444' }}>{result.double_slit.reflected}</strong></span>
                                    <span style={{ color: '#aaa' }}>In flight: <strong style={{ color: '#ffcc00' }}>{result.double_slit.in_flight}</strong></span>
                                </div>
                                {result.double_slit.histogram_bins && result.double_slit.histogram_bins.length > 0 && (
                                    <div style={{ position: 'relative', height: '200px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px', padding: '10px', display: 'flex', alignItems: 'flex-end', gap: '1px' }}>
                                        {result.double_slit.histogram_bins.map((count, i) => {
                                            const maxCount = Math.max(...result.double_slit.histogram_bins);
                                            const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                                            return (
                                                <div key={i} style={{
                                                    flex: 1,
                                                    height: `${height}%`,
                                                    background: `linear-gradient(to top, #00e5ff, #00ff88)`,
                                                    borderRadius: '2px 2px 0 0',
                                                    opacity: count > 0 ? 0.9 : 0.15,
                                                    transition: 'height 0.3s ease'
                                                }} title={`y=[${result.double_slit.histogram_edges[i]?.toFixed(1)}, ${result.double_slit.histogram_edges[i+1]?.toFixed(1)}]: ${count} hits`} />
                                            );
                                        })}
                                    </div>
                                )}
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#666', marginTop: '4px' }}>
                                    <span>y = {result.double_slit.histogram_edges?.[0]?.toFixed(1)}</span>
                                    <span>Detector Screen Y-Position</span>
                                    <span>y = {result.double_slit.histogram_edges?.[result.double_slit.histogram_edges.length-1]?.toFixed(1)}</span>
                                </div>
                            </div>
                            )}

                            <details style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '15px', borderRadius: '8px', borderLeft: '4px solid #888' }}>
                                <summary style={{ cursor: 'pointer', color: '#aaa', fontWeight: 'bold' }}>View Full Console Log</summary>
                                <pre style={{ marginTop: '15px', fontFamily: 'monospace', color: '#888', whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                                    {result.full_log}
                                </pre>
                            </details>
                        </div>
                    )}
                    
                    {!loading && !result && !error && (
                        <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.5 }}>
                            <div style={{ fontSize: '3rem', marginBottom: '10px' }}>⚡</div>
                            <p>Adjust parameters and select particles.</p>
                            <p>Click "Extract Governing Math" to begin.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default App;
