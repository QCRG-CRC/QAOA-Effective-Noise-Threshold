"""QAOA for MaxCut on IBM Quantum hardware (and on Aer, noiseless or with a device noise model).

The defaults of run_QAOA() are the settings of the ibm_marrakesh runs reported in the Methods
(training and evaluation on hardware, 125 COBYLA iterations, 4096 estimator shots, 16384 sampler
shots, optimisation level 3, no error mitigation); run_hardware.py drives it.

The approximation ratio of a run is the cut value of the most frequent bitstring, ties broken
towards the larger cut (the first entry of result["top30"]), divided by the optimal cut.
"""
import numpy as np
import networkx as nx
import time
from scipy.optimize import minimize
from collections import Counter

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, EstimatorV2 as Estimator_runtime, Sampler, SamplerV2 as Sampler_runtime, Batch
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
import matplotlib.pyplot as plt


from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile

# Utility Functions

def maxcut_hamiltonian(G):
    n = G.number_of_nodes()
    edges = G.edges()
    paulis = []
    coeffs = []

    for i, j in edges:
        z = ["I"] * n
        z[i] = "Z"
        z[j] = "Z"

        paulis.append("".join(z))
        coeffs.append(0.5)

    return SparsePauliOp(paulis, coeffs)


# =========================
# Helper: shot scheduler
# =========================
def get_shots(iteration, maxiter):
    return 1024


# =========================
# Helper: classical MaxCut
# =========================
def compute_cut_value(bitstring, G):
    cut = 0
    for u, v in G.edges():
        if bitstring[u] != bitstring[v]:
            cut += 1
    return cut


# =========================
# Backend / primitive factory
# =========================
def get_estimator(mode, backend_real, backend_sim):

    if mode == "noiseless":
        return EstimatorV2()

    elif mode == "noisy_sim":
        return Estimator_runtime(mode=backend_sim)

    elif mode == "real":
        return Estimator_runtime(mode=backend_real)

    else:
        raise ValueError("Unknown mode")

def get_transpiled(circuit, mode, backend_real, backend_sim):

    if mode == "noiseless" or mode == "estimator":
        return transpile(circuit)

    elif mode == "noisy_sim":
        return transpile(circuit, backend=backend_sim)

    elif mode == "real":
        pm = generate_preset_pass_manager(backend=backend_real, optimization_level=3)
        isa_circuit = pm.run(circuit)
        return isa_circuit

    else:
        raise ValueError("Unknown mode")

def get_sampler(mode, backend_real, backend_sim):

    if mode == "noiseless":
        return SamplerV2()

    elif mode == "noisy_sim":
        return Sampler_runtime(mode=backend_sim)

    elif mode == "real":
        return Sampler_runtime(mode=backend_real)

    else:
        raise ValueError("Unknown mode")


# =========================
# Rolling Batch Manager
# =========================
class RollingBatchEstimator:
    """
    Wraps IBM Quantum's Batch to transparently rotate to a new batch
    whenever the current one approaches the time limit.

    Parameters
    ----------
    backend        : IBM backend object
    max_batch_secs : rotate to a new batch after this many seconds
                     (default 270 s = 4 min 30 s, giving 30 s headroom
                      before the hard 5-minute wall)
    max_time       : max_time string passed to each Batch (default "30m")
    resilience     : resilience_level for the estimator
    precision      : default_precision for the estimator
    """

    def __init__(
        self,
        backend,
        max_batch_secs: float = 270.0,
        max_time: str = "30m",
        resilience: int = 0,
        precision: float = 0.05,
    ):
        self.backend = backend
        self.max_batch_secs = max_batch_secs
        self.max_time = max_time
        self.resilience = resilience
        self.precision = precision

        self._batch = None
        self._estimator = None
        self._batch_open_time = None
        self._batch_count = 0

        self._open_new_batch()

    # ------------------------------------------------------------------
    def _open_new_batch(self):
        """Close any existing batch and open a fresh one."""
        self._close_current_batch()

        self._batch_count += 1
        print(f"[RollingBatch] Opening batch #{self._batch_count}")
        self._batch = Batch(backend=self.backend, max_time=self.max_time)
        self._batch.__enter__()

        self._estimator = Estimator_runtime(mode=self._batch)
        self._estimator.options.resilience_level = self.resilience
        self._estimator.options.default_precision = self.precision

        self._batch_open_time = time.time()

    # ------------------------------------------------------------------
    def _close_current_batch(self):
        """Gracefully exit the current batch context if one is open."""
        if self._batch is not None:
            try:
                self._batch.__exit__(None, None, None)
                print(f"[RollingBatch] Closed batch #{self._batch_count}")
            except Exception as e:
                print(f"[RollingBatch] Warning: error closing batch: {e}")
            self._batch = None
            self._estimator = None

    # ------------------------------------------------------------------
    def _rotate_if_needed(self):
        """Open a new batch if the current one is approaching its time limit."""
        elapsed = time.time() - self._batch_open_time
        if elapsed >= self.max_batch_secs:
            print(
                f"[RollingBatch] Batch #{self._batch_count} has been open "
                f"{elapsed:.1f}s — rotating to a new batch."
            )
            self._open_new_batch()

    # ------------------------------------------------------------------
    def run(self, circuits_hamiltonians, shots: int = 2048):
        """
        Run a single estimator job, rotating the batch first if needed.

        Parameters
        ----------
        circuits_hamiltonians : list of (circuit, hamiltonian, params?) tuples
        shots                 : number of shots for this call

        Returns
        -------
        result[0] from the estimator job
        """
        self._rotate_if_needed()

        self._estimator.options.default_shots = shots
        job = self._estimator.run(circuits_hamiltonians)
        return job.result()[0]

    # ------------------------------------------------------------------
    def close(self):
        """Must be called when optimization is finished."""
        self._close_current_batch()
        print("[RollingBatch] All batches closed.")


# =========================
# MAIN QAOA FUNCTION
# =========================
def run_QAOA(
    G,
    p,
    training_mode="real",
    evaluation_mode="real",
    backend_real_name="ibm_marrakesh",
    maxiter=125,
    bypass_training=False,
    opt_param=None,
    tag="",
    sampler_shot=16384,
    estimator_shot=4096,
    warm_start=False,
    init_param=None,
    # Rolling-batch tunables
    max_batch_secs: float = 270.0,   # rotate after 4 min 30 s
    batch_max_time: str = "30m",
):
    # =========================
    # Load backend (for noisy + real)
    # =========================
    service = QiskitRuntimeService()
    backend_real = service.backend(backend_real_name)
    noise_model = NoiseModel.from_backend(backend_real)
    backend_sim = AerSimulator(noise_model=noise_model)

    # =========================
    # Build Hamiltonian
    # =========================
    hamiltonian = maxcut_hamiltonian(G)

    # =========================
    # Ansatz
    # =========================
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=p)

    # =========================
    # Logging
    # =========================
    energy_history = []
    param_history = []
    time_history = []

    iteration_counter = {"i": 0}

    if not bypass_training:

        # ----------------------------------------------------------------
        # REAL BACKEND — rolling batch
        # ----------------------------------------------------------------
        if training_mode == "real":
            pm = generate_preset_pass_manager(backend=backend_real, optimization_level=3)
            isa_circuit = pm.run(ansatz)
            isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)

            init_params = init_param if warm_start else np.random.rand(2 * p)

            rolling = RollingBatchEstimator(
                backend=backend_real,
                max_batch_secs=max_batch_secs,
                max_time=batch_max_time,
                resilience=0,
                precision=0.05,
            )

            try:
                def objective_val(params):
                    start = time.time()
                    i = iteration_counter["i"]

                    result = rolling.run(
                        [(isa_circuit, isa_hamiltonian, params)],
                        shots=estimator_shot,
                    )
                    energy = result.data.evs

                    run_time = time.time() - start
                    energy_history.append(energy)
                    param_history.append(params.copy())
                    time_history.append(run_time)

                    print(
                        f"Iter {i} | Energy: {energy:.4f} | "
                        f"Shots: {estimator_shot} | Time: {run_time:.2f}s | "
                        f"Params: {params}"
                    )
                    iteration_counter["i"] += 1
                    return energy

                result = minimize(
                    objective_val,
                    init_params,
                    method="COBYLA",
                    options={"maxiter": maxiter},
                )

            finally:
                # Always close all open batches, even if optimization crashes
                rolling.close()

        # ----------------------------------------------------------------
        # NOISELESS / NOISY SIM
        # ----------------------------------------------------------------
        else:
            estimator = get_estimator(training_mode, backend_real, backend_sim)

            def objective_val(params):
                start = time.time()
                i = iteration_counter["i"]
                shots = estimator_shot

                circuit = ansatz.assign_parameters(params).decompose().decompose()
                transpiled = get_transpiled(circuit, training_mode, backend_real, backend_sim)

                if hasattr(estimator, "options"):
                    estimator.options.default_shots = shots
                else:
                    print("shot estimator setup failed")

                job = estimator.run([(transpiled, hamiltonian)])
                energy = job.result()[0].data.evs

                run_time = time.time() - start
                energy_history.append(energy)
                param_history.append(params.copy())
                time_history.append(run_time)

                print(
                    f"Iter {i} | Energy: {energy:.4f} | "
                    f"Shots: {shots} | Time: {run_time:.2f}s | "
                    f"Params: {params}"
                )
                iteration_counter["i"] += 1
                return energy

            init_params = init_param if warm_start else np.random.rand(2 * p)

            result = minimize(
                objective_val,
                init_params,
                method="COBYLA",
                options={"maxiter": maxiter},
            )

        # ----------------------------------------------------------------
        # Post-optimization bookkeeping (shared)
        # ----------------------------------------------------------------
        best_iteration = np.argmin(energy_history)
        print(f"Best iteration = {best_iteration}")
        optimal_params = param_history[best_iteration]

        plt.plot(energy_history)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy vs Iteration")

        print("\nOptimal params:", optimal_params)

    # =========================
    # FINAL EVALUATION (Sampler)
    # =========================
    start = time.time()
    sampler = get_sampler(evaluation_mode, backend_real, backend_sim)
    if bypass_training:
        optimal_params = opt_param

    optimal_circuit = ansatz.assign_parameters(optimal_params).decompose().decompose()
    optimal_circuit.measure_all()

    transpiled = get_transpiled(optimal_circuit, evaluation_mode, backend_real, backend_sim)

    shots_final = sampler_shot

    if hasattr(sampler, "options"):
        sampler.options.default_shots = shots_final
    else:
        print("shot sampler setup failed")

    job = sampler.run([(transpiled,)])
    result_sampler = job.result()[0]

    counts = result_sampler.data.meas.get_counts()
    sampling_time = time.time() - start

    # =========================
    # Classical post-processing
    # =========================
    energy_final = 0
    tmp = []
    for bitstring, count in counts.items():
        prob = count / shots_final
        cut = compute_cut_value(bitstring, G)
        energy_final += prob * cut
        tmp.append((bitstring, cut, count))

    print("\nFinal estimated cut value:", energy_final)

    sorted_tmp = sorted(tmp, key=lambda x: (-x[2], -x[1]))
    top30 = sorted_tmp[:30]

    print("\nTop 30 bitstrings: (no, bitstring, cut value, count)")
    for k in range(30):
        print(k + 1, *sorted_tmp[k])

    return {
        "optimal_params": optimal_params,
        "energy_history": energy_history,
        "param_history": param_history,
        "time_history": time_history,
        "sampling_time": sampling_time,
        "counts": counts,
        "top30": top30,
        "final_energy": energy_final,
    }
