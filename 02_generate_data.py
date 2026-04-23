"""
02_generate_data.py
────────────────────────────────────────────────────────────────────────────────
Generates 1,000–1,500 labeled (circuit, backend) samples with Hellinger fidelity.

OVERNIGHT-SAFE FEATURES:
  - Checkpointing: every sample is appended to data/checkpoint.jsonl immediately.
  - Resumable: on restart, already-computed samples are loaded and skipped.
  - Per-sample error handling: exceptions are caught, logged, and skipped.
  - Circuit deduplication: samples tracked by (circuit_id, backend_name).
  - Progress bar with ETA via tqdm.
  - Final CSV exported to data/fidelity_dataset.csv.

Run:
    python 02_generate_data.py

If interrupted: just re-run the same command. It will pick up from where it stopped.
"""

import os
import json
import time
import traceback
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeManilaV2, FakeNairobiV2

# ── Config ────────────────────────────────────────────────────────────────────
SHOTS        = 16_384
OPT_LEVEL    = 1
SEED_TRANSPILE = 42
SEED_SIM       = 42
DATA_DIR     = Path("data")
CHECKPOINT   = DATA_DIR / "checkpoint.jsonl"
FINAL_CSV    = DATA_DIR / "fidelity_dataset.csv"
LOG_FILE     = DATA_DIR / "generation.log"

# Target samples per (backend, qubit_bucket)
# bucket 0 = ≤5q, bucket 1 = 6-10q, bucket 2 = 11-15q
TARGET_PER_CELL = 50   # 3 backends × 3 buckets × 50 = 450 random circuits
                        # + QASMbench/MQTbench circuits on top → ~1,200 total

# ── Logging ───────────────────────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Backend registry ──────────────────────────────────────────────────────────
# Instantiate once — FakeBackends load calibration data on __init__
BACKENDS = {
    "FakeSherbrooke": FakeSherbrooke(),   # 127q — handles all sizes
    "FakeManilaV2":   FakeManilaV2(),     # 5q  — only ≤5q circuits
    "FakeNairobiV2":  FakeNairobiV2(),    # 7q  — only ≤7q circuits
}

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint() -> tuple[list[dict], set]:
    """Load already-computed samples. Returns (rows, done_ids)."""
    rows, done_ids = [], set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    rows.append(row)
                    done_ids.add(row["sample_id"])
                except json.JSONDecodeError:
                    pass  # corrupt line — skip
    log.info(f"Checkpoint loaded: {len(rows)} samples already done.")
    return rows, done_ids


def save_sample(row: dict):
    """Append one sample to checkpoint file immediately."""
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_circuit_features(circuit: QuantumCircuit) -> dict:
    op_counts = dict(circuit.count_ops())
    for key in ("measure", "barrier", "reset", "delay"):
        op_counts.pop(key, None)
    total_gates = sum(op_counts.values())
    cx_count    = op_counts.get("cx",  0)
    ecr_count   = op_counts.get("ecr", 0)
    rz_count    = op_counts.get("rz",  0)
    sx_count    = op_counts.get("sx",  0)
    x_count     = op_counts.get("x",   0)
    two_qubit   = cx_count + ecr_count
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
        "critical_path":      circuit.depth(),
    }


def extract_noise_features(backend) -> dict:
    t1_vals, t2_vals, sq_errors, tq_errors, ro_errors = [], [], [], [], []
    for q in range(backend.num_qubits):
        try:
            qp = backend.qubit_properties(q)
            if qp is not None:
                if getattr(qp, "t1", None) is not None: t1_vals.append(qp.t1)
                if getattr(qp, "t2", None) is not None: t2_vals.append(qp.t2)
        except Exception:
            pass
    for gate in ("sx", "x"):
        if gate in backend.target:
            for _, ip in backend.target[gate].items():
                if ip is not None and ip.error is not None:
                    sq_errors.append(ip.error)
    for gate in ("cx", "ecr"):
        if gate in backend.target:
            for _, ip in backend.target[gate].items():
                if ip is not None and ip.error is not None:
                    tq_errors.append(ip.error)
    if "measure" in backend.target:
        for _, ip in backend.target["measure"].items():
            if ip is not None and ip.error is not None:
                ro_errors.append(ip.error)
    return {
        "mean_t1":       float(np.mean(t1_vals))   if t1_vals   else 0.0,
        "mean_t2":       float(np.mean(t2_vals))   if t2_vals   else 0.0,
        "mean_sq_error": float(np.mean(sq_errors)) if sq_errors else 0.0,
        "mean_tq_error": float(np.mean(tq_errors)) if tq_errors else 0.0,
        "mean_ro_error": float(np.mean(ro_errors)) if ro_errors else 0.0,
    }


# ── Hellinger fidelity ────────────────────────────────────────────────────────
def hellinger_fidelity(probs_ideal: np.ndarray, counts_noisy: dict, n_qubits: int) -> float:
    total_shots = sum(counts_noisy.values())
    probs_noisy = np.zeros(2 ** n_qubits)
    for bitstring, count in counts_noisy.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs_noisy[idx] = count / total_shots
    fidelity = float(np.sum(np.sqrt(probs_ideal * probs_noisy)) ** 2)
    return float(np.clip(fidelity, 0.0, 1.0))


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    circuit: QuantumCircuit,
    backend,
    backend_name: str,
    circuit_id: str,
    family: str,
    shots: int = SHOTS,
) -> dict:
    n_qubits = circuit.num_qubits

    # 1. Transpile
    transpiled = transpile(
        circuit, backend=backend,
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
    )

    # 2. Circuit features (post-transpilation)
    circ_feats = extract_circuit_features(transpiled)

    # 3. Noise features
    noise_feats = extract_noise_features(backend)

    # 4. Ideal statevector
    circ_no_meas = transpiled.remove_final_measurements(inplace=False)
    sv_sim = StatevectorSimulator()
    sv_result = sv_sim.run(circ_no_meas).result()
    statevec = sv_result.get_statevector(circ_no_meas)
    probs_ideal = np.abs(np.array(statevec.data)) ** 2

    # 5. Noisy sim
    noise_model = NoiseModel.from_backend(backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    if 'GPU' in noisy_sim.available_devices():
        noisy_sim.set_options(device='GPU')
        
    circ_with_meas = transpiled.copy()
    circ_with_meas.measure_all()
    noisy_result = noisy_sim.run(
        circ_with_meas, shots=shots, seed_simulator=SEED_SIM
    ).result()
    counts_noisy = noisy_result.get_counts()

    # 6. Fidelity
    fidelity = hellinger_fidelity(probs_ideal, counts_noisy, n_qubits)

    return {
        "sample_id":   f"{circuit_id}__{backend_name}",
        "circuit_id":  circuit_id,
        "family":      family,
        "backend":     backend_name,
        **circ_feats,
        **noise_feats,
        "fidelity":    fidelity,
    }


# ── Circuit sources ───────────────────────────────────────────────────────────
def get_mqtbench_circuits(max_qubits: int = 15) -> list[tuple[QuantumCircuit, str, str]]:
    """Returns list of (circuit, circuit_id, family)."""
    results = []
    try:
        from mqt.bench import get_benchmark
        bench_configs = [
            # (algorithm, level, qubit_counts_to_try)
            ("ghz",           "alg",   [3, 5, 7, 10, 12, 15]),
            ("grover",        "alg",   [3, 5, 7]),
            ("qft",           "alg",   [3, 5, 8, 10, 12, 15]),
            ("qpeexact",      "alg",   [3, 5, 8, 10, 12]),
            ("qaoa",          "alg",   [3, 5, 7, 10]),
            ("portfolioqaoa", "alg",   [3, 5]),
            ("hhl",           "alg",   [3]),
            ("vqe",           "alg",   [3, 4, 5, 6]),
            ("dj",            "alg",   [3, 5, 8, 10, 12, 15]),
            ("wstate",        "alg",   [3, 5, 8, 10, 12, 15]),
        ]
        for name, level, qubit_list in bench_configs:
            for nq in qubit_list:
                if nq > max_qubits:
                    continue
                try:
                    # Using kwargs for compatibility with mqt.bench >= 2.0
                    qc = get_benchmark(benchmark_name=name, level=level, circuit_size=nq)
                    # Strip measurements — pipeline adds them
                    qc = qc.remove_final_measurements(inplace=False)
                    cid = f"mqt_{name}_{nq}q"
                    results.append((qc, cid, name))
                except Exception as e:
                    log.debug(f"MQTbench skip {name}/{nq}q: {e}")
    except ImportError:
        log.warning("mqt.bench not available — skipping MQTbench circuits.")
    log.info(f"MQTbench circuits loaded: {len(results)}")
    return results


def get_random_circuits(seed_offset: int = 0) -> list[tuple[QuantumCircuit, str, str]]:
    """Generate random circuits across 3 qubit buckets."""
    results = []
    configs = [
        # (n_qubits, depth, count, bucket_label)
        (2,  3,  8, "rand_2-5q"),
        (3,  4,  8, "rand_2-5q"),
        (4,  5,  8, "rand_2-5q"),
        (5,  5,  8, "rand_2-5q"),
        (6,  6,  8, "rand_6-10q"),
        (7,  7,  8, "rand_6-10q"),
        (8,  7,  8, "rand_6-10q"),
        (10, 8,  8, "rand_6-10q"),
        (11, 8,  8, "rand_11-15q"),
        (12, 9,  8, "rand_11-15q"),
        (13, 9,  6, "rand_11-15q"),
        (15, 10, 6, "rand_11-15q"),
    ]
    for (nq, depth, count, family) in configs:
        for i in range(count):
            seed = seed_offset + nq * 1000 + depth * 100 + i
            qc = random_circuit(nq, depth, seed=seed, measure=False)
            cid = f"random_{nq}q_d{depth}_s{seed}"
            results.append((qc, cid, family))
    log.info(f"Random circuits generated: {len(results)}")
    return results


# ── Backend routing — skip circuits too large for small backends ──────────────
def get_backends_for_circuit(n_qubits: int) -> list[tuple[str, object]]:
    valid = [("FakeSherbrooke", BACKENDS["FakeSherbrooke"])]
    if n_qubits <= 5:
        valid.append(("FakeManilaV2", BACKENDS["FakeManilaV2"]))
    if n_qubits <= 7:
        valid.append(("FakeNairobiV2", BACKENDS["FakeNairobiV2"]))
    return valid


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("Starting data generation")
    log.info("=" * 70)

    # Load checkpoint
    completed_rows, done_ids = load_checkpoint()

    # Build full circuit list
    all_circuits = []
    all_circuits.extend(get_mqtbench_circuits(max_qubits=15))
    all_circuits.extend(get_random_circuits(seed_offset=9999))

    # Build full job list: (circuit, circuit_id, family, backend_name, backend)
    all_jobs = []
    for (qc, cid, family) in all_circuits:
        for (bname, backend) in get_backends_for_circuit(qc.num_qubits):
            sample_id = f"{cid}__{bname}"
            if sample_id not in done_ids:
                all_jobs.append((qc, cid, family, bname, backend))

    log.info(f"Total jobs to run: {len(all_jobs)}  (already done: {len(done_ids)})")
    if not all_jobs:
        log.info("All samples already computed. Exporting CSV.")
    else:
        # Estimate time
        sec_per_sample = 12  # rough estimate: ~12s per sample on a laptop
        eta_min = len(all_jobs) * sec_per_sample / 60
        log.info(f"Estimated time: ~{eta_min:.0f} minutes ({eta_min/60:.1f} hours)")

    # ── Main generation loop ─────────────────────────────────────────────────
    n_errors = 0
    with tqdm(total=len(all_jobs), desc="Generating", unit="sample") as pbar:
        for (qc, cid, family, bname, backend) in all_jobs:
            t0 = time.time()
            try:
                row = run_pipeline(
                    circuit=qc,
                    backend=backend,
                    backend_name=bname,
                    circuit_id=cid,
                    family=family,
                    shots=SHOTS,
                )
                save_sample(row)
                completed_rows.append(row)
                elapsed = time.time() - t0
                pbar.set_postfix(
                    fidelity=f"{row['fidelity']:.3f}",
                    backend=bname[:8],
                    sec=f"{elapsed:.1f}",
                )
            except Exception:
                n_errors += 1
                log.error(
                    f"FAILED: {cid} / {bname}\n{traceback.format_exc()}"
                )
            pbar.update(1)

    log.info(f"Generation complete. Successes: {len(completed_rows)}  Errors: {n_errors}")

    # ── Export final CSV ──────────────────────────────────────────────────────
    if completed_rows:
        df = pd.DataFrame(completed_rows)
        df.to_csv(FINAL_CSV, index=False)
        log.info(f"Dataset saved → {FINAL_CSV}  ({len(df)} rows, {df.shape[1]} columns)")
        log.info(f"Fidelity stats:\n{df['fidelity'].describe()}")
        log.info(f"Samples per backend:\n{df['backend'].value_counts()}")
        log.info(f"Samples per family:\n{df['family'].value_counts()}")
    else:
        log.warning("No samples collected — check errors above.")

    print(f"\nDone. Dataset: {FINAL_CSV}")
    print(f"Next: python 03_train_and_evaluate.py")


if __name__ == "__main__":
    main()