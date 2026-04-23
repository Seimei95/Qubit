"""
01_verify_env.py
Run this first. It tests the full pipeline on 3 circuits before you commit to overnight generation.
If this passes cleanly with no errors, you're good to run 02_generate_data.py.
"""

import sys
print(f"Python: {sys.version}")

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime

print(f"qiskit:              {qiskit.__version__}")
print(f"qiskit-aer:          {qiskit_aer.__version__}")
print(f"qiskit-ibm-runtime:  {qiskit_ibm_runtime.__version__}")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeManilaV2, FakeNairobiV2
import xgboost, sklearn, scipy
print(f"xgboost:     {xgboost.__version__}")
print(f"scikit-learn:{sklearn.__version__}")
print(f"scipy:       {scipy.__version__}")
print()

# ── Helper: Hellinger fidelity ────────────────────────────────────────────────
def hellinger_fidelity(probs_ideal: np.ndarray, counts_noisy: dict, n_qubits: int) -> float:
    """
    probs_ideal : numpy array of length 2^n_qubits (statevector probabilities, ideal ordering)
    counts_noisy: dict {bitstring -> count} from AerSimulator (big-endian, reversed qubit order)
    Returns Hellinger fidelity in [0, 1].
    """
    total_shots = sum(counts_noisy.values())
    probs_noisy = np.zeros(2 ** n_qubits)
    for bitstring, count in counts_noisy.items():
        # Qiskit bitstrings are big-endian (qubit 0 is rightmost)
        idx = int(bitstring.replace(" ", ""), 2)
        probs_noisy[idx] = count / total_shots
    fidelity = float(np.sum(np.sqrt(probs_ideal * probs_noisy)) ** 2)
    return float(np.clip(fidelity, 0.0, 1.0))


# ── Helper: extract circuit features (post-transpilation) ─────────────────────
def extract_circuit_features(circuit: QuantumCircuit) -> dict:
    op_counts = circuit.count_ops()
    # Remove non-gate ops
    for key in ("measure", "barrier", "reset"):
        op_counts.pop(key, None)
    total_gates = sum(op_counts.values())
    cx_count  = op_counts.get("cx",  0)
    ecr_count = op_counts.get("ecr", 0)
    rz_count  = op_counts.get("rz",  0)
    sx_count  = op_counts.get("sx",  0)
    x_count   = op_counts.get("x",   0)
    two_qubit = cx_count + ecr_count
    return {
        "total_gates":        total_gates,
        "cx_count":           cx_count,
        "ecr_count":          ecr_count,
        "rz_count":           rz_count,
        "sx_count":           sx_count,
        "x_count":            x_count,
        "depth":              circuit.depth(),
        "num_qubits":         circuit.num_qubits,
        "two_qubit_fraction": two_qubit / total_gates if total_gates > 0 else 0.0,
        "critical_path":      circuit.depth(),  # proxy; exact CPL requires DAG analysis
    }


# ── Helper: extract noise features from FakeV2 backend ───────────────────────
def extract_noise_features(backend) -> dict:
    # T1, T2 from qubit_properties
    t1_vals, t2_vals = [], []
    for q in range(backend.num_qubits):
        try:
            qp = backend.qubit_properties(q)
            if qp is not None:
                if getattr(qp, "t1", None) is not None: t1_vals.append(qp.t1)
                if getattr(qp, "t2", None) is not None: t2_vals.append(qp.t2)
        except Exception:
            pass

    # Single-qubit gate errors
    sq_errors = []
    for gate in ("sx", "x"):
        if gate in backend.target:
            for qargs, inst_prop in backend.target[gate].items():
                if inst_prop is not None and inst_prop.error is not None:
                    sq_errors.append(inst_prop.error)

    # Two-qubit gate errors
    tq_errors = []
    for gate in ("cx", "ecr"):
        if gate in backend.target:
            for qargs, inst_prop in backend.target[gate].items():
                if inst_prop is not None and inst_prop.error is not None:
                    tq_errors.append(inst_prop.error)

    # Readout errors
    ro_errors = []
    if "measure" in backend.target:
        for qargs, inst_prop in backend.target["measure"].items():
            if inst_prop is not None and inst_prop.error is not None:
                ro_errors.append(inst_prop.error)

    return {
        "mean_t1":       float(np.mean(t1_vals))   if t1_vals   else 0.0,
        "mean_t2":       float(np.mean(t2_vals))   if t2_vals   else 0.0,
        "mean_sq_error": float(np.mean(sq_errors)) if sq_errors else 0.0,
        "mean_tq_error": float(np.mean(tq_errors)) if tq_errors else 0.0,
        "mean_ro_error": float(np.mean(ro_errors)) if ro_errors else 0.0,
    }


# ── Helper: full pipeline for one (circuit, backend) pair ────────────────────
def run_pipeline(circuit: QuantumCircuit, backend, shots: int = 16384) -> dict:
    """
    Returns a flat dict with circuit features, noise features, and fidelity label.
    Raises on error — caller should catch.
    """
    n_qubits = circuit.num_qubits

    # 1. Ideal probabilities via StatevectorSimulator on the logical circuit
    #    Must remove measurements first
    circ_no_meas = circuit.remove_final_measurements(inplace=False) if hasattr(circuit, 'remove_final_measurements') else circuit.copy()
    sv_sim = StatevectorSimulator()
    sv_job = sv_sim.run(circ_no_meas)
    sv_result = sv_job.result()
    statevec = sv_result.get_statevector(circ_no_meas)
    probs_ideal = np.abs(np.array(statevec.data)) ** 2

    # 2. Add measurements to the logical circuit
    circ_with_meas = circuit.copy()
    circ_with_meas.measure_all()

    # 3. Transpile the measured circuit to backend native gate set
    transpiled = transpile(circ_with_meas, backend=backend, optimization_level=1, seed_transpiler=42)

    # 4. Circuit features (post-transpilation)
    circ_feats = extract_circuit_features(transpiled)

    # 5. Noise features
    noise_feats = extract_noise_features(backend)

    # 6. Noisy probabilities via AerSimulator + noise model
    noise_model = NoiseModel.from_backend(backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    if 'GPU' in noisy_sim.available_devices():
        noisy_sim.set_options(device='GPU')
        
    noisy_job = noisy_sim.run(transpiled, shots=shots, seed_simulator=42)
    counts_noisy = noisy_job.result().get_counts()

    # 7. Hellinger fidelity (using logical n_qubits)
    fidelity = hellinger_fidelity(probs_ideal, counts_noisy, n_qubits)

    return {**circ_feats, **noise_feats, "fidelity": fidelity, "backend": backend.name}


# ── Smoke test ───────────────────────────────────────────────────────────────
print("Running smoke test on 3 circuits (this takes ~30 sec)...")
backends = [FakeManilaV2(), FakeSherbrooke()]
test_circuits = [
    random_circuit(3, 3, seed=0),
    random_circuit(4, 4, seed=1),
    random_circuit(5, 3, seed=2),
]

for i, (circ, backend) in enumerate(zip(test_circuits, [backends[0], backends[0], backends[1]])):
    try:
        result = run_pipeline(circ, backend, shots=1024)
        print(f"  Circuit {i+1}: fidelity={result['fidelity']:.4f}  "
              f"gates={result['total_gates']}  depth={result['depth']}  "
              f"backend={result['backend']}")
    except Exception as e:
        print(f"  Circuit {i+1}: FAILED — {e}")
        raise

print()
print("✓ Smoke test passed. Environment is ready.")
print("  Next: run   python 02_generate_data.py")