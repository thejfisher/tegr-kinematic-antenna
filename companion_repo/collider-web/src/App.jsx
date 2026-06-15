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

    const handleParamChange = (name, value) => {
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
                    sink_mass: sinkMass
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
    const totalMass = particleA.mass + particleB.mass + (mode.includes("3-body") ? particleC.mass : 0);
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

            {/* LEFT CONTROL PANEL */}
            <div className="sidebar" style={{ width: '380px' }}>
                <div className="header">
                    <h1>AI PHYSICS ENGINE</h1>
                    <p>Teleparallel Collision Discovery</p>
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
                            
                            <button onClick={() => setTSymmetry(!tSymmetry)} style={{ background: tSymmetry ? '#ffcc00' : 'transparent', color: tSymmetry ? '#000' : '#888', border: '1px solid #ffcc00', padding: '4px 8px', borderRadius: '4px', fontSize: '0.7rem', cursor: 'pointer', fontWeight: 'bold', marginLeft: '5px' }}>{tSymmetry ? 'T-SYMMETRY: ON' : 'T-SYMMETRY: OFF'}</button>
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
                        <div style={{ display: 'flex', alignItems: 'center', color: '#ff00ff', fontWeight: 'bold' }}>VS</div>
                        <div 
                            onClick={() => setShowModal('B')}
                            style={{ flex: 1, border: '1px dashed #ff00ff', padding: '15px', textAlign: 'center', borderRadius: '8px', cursor: 'pointer', background: 'rgba(255,0,255,0.05)' }}
                        >
                            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '5px' }}>PARTICLE B</div>
                            <div style={{ fontSize: '1.8rem', color: '#fff', fontWeight: 'bold' }}>{particleB.symbol}</div>
                        </div>
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
