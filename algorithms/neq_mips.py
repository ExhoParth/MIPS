import time
import numpy as np
from typing import Tuple
from quantization.norm_pq import NormPQ
from quantization.residual_pq import ResidualPQ
from quantization.base_pq import PQ
from quantization.utils import execute


from quantization.constants import (
    PARTITION,
    NUM_CODEBOOKS,
    NUM_CODEWORDS,
)

def generate_data(
    num_atoms: int = 10**3,
    len_signal: int = 10**4,
    num_signals: int = 1,
    num_best_atoms: int =5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

	
    rng = np.random.default_rng(seed)
    atoms_array = np.empty((num_atoms, len_signal))
    means = rng.normal(size=num_atoms)
    atoms_array = np.empty((num_atoms, len_signal))
    for idx in range(num_atoms):
      atoms_array[idx] = rng.normal(loc=means[idx], size=len_signal)
    signal = rng.normal(loc=rng.normal(), size=(num_signals, len_signal))
    if num_signals == 1:
        signal = signal.reshape(1,-1)

    return atoms_array, signal


seed = 3
num_best_atoms = 10

atoms, signals = generate_data(num_atoms=1000, len_signal=200, num_best_atoms=num_best_atoms)

naive_candidates_array = (
                np.matmul(atoms, signals.transpose())
                .argsort(axis=0)[:num_best_atoms]
                .transpose()
            )

pqs = [
    PQ(M=PARTITION, Ks=NUM_CODEWORDS) for _ in range(NUM_CODEBOOKS - 1)
]
quantizer = ResidualPQ(pqs=pqs)
quantizer = NormPQ(n_percentile=NUM_CODEWORDS, quantize=quantizer)
start_time = time.time()

candidates_array, budget_array = execute(
    seed=seed,
    top_k=num_best_atoms,
    pq=quantizer,
    X=atoms,
    Q=signals.astype("float32"),
    G=naive_candidates_array,
    num_codebooks=NUM_CODEBOOKS,
    num_codewords=NUM_CODEWORDS,
    train_size=atoms.shape[0],
)
# includes the time finding recall (other baselines don't)
runtime = time.time() - start_time

print(f"neq mips:{candidates_array}")
print(f"{runtime} seconds")
print(f"naive mips:{naive_candidates_array}")