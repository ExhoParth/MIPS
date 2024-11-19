import numpy as np
from typing import Tuple, List
import math
import numpy.linalg as la
import matplotlib.pyplot as plt


from pca_utils import *
# from data.custom_data import generate_custom_data
# from utils.constants import (
#     SCALING_DEPTH,
#     SCALING_TOPK,
# )


class PCA_MIPS:
    """
    Basic Class for PCA-MIPS described here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
    """

    def __init__(
        self,
        # atoms: np.ndarray,
        # signal: np.ndarray,
        delta: int = 2
    ):
        """
        :delta: the number of top principal components to use
        """
        self.delta = delta  # if delta is 0, this is the same as naive MIPS
        self.tree = Tree()
        self.p_atoms = None
        self.p_signal = None

    def preprocess_data(
        self,
        atoms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess atoms to turn into NN search problem, and building a pca-tree data structure.
        Here, note that the principal components are in "decreasing" order of variance.

        :atoms: The dataset to be preprocessed
        :returns: The principle component matrix W and the mean array mu
        """
        # concatenate data and matmul with principle components matrix
        concatenated_atoms = self.tree.first_reduction(atoms)
        W, mu, p_atoms = self.tree.second_reduction(concatenated_atoms)
        self.p_atoms = p_atoms.T

        # feed converted data into tree class and construct pca-tree
        self.tree.data = self.p_atoms
        self.tree.root = self.tree.construct_pca_tree(self.p_atoms.shape[1] - self.delta)
        return W, mu

    def preprocess_query(
        self,
        signal: np.ndarray,
        pca_matrix: np.ndarray,
        mu: np.ndarray
    ) -> None:
        """
        Preprocess signal to turn into NN search problem.

        :signal: the query to be preprocessed
        :pca_matrix: the pca matrix
        :mu: an array of atoms' column means
        :returns: None
        """
        _p_query = self.tree.first_reduction(signal, is_query=True)
        self.p_signal = np.matmul(pca_matrix.T, _p_query - mu)

    def run(
        self,
        top_k
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assumes that atoms and signal has already been preprocessed.
        Budget consists of two main operations:
           1. traversing pca-tree
           2. comparing distances

        :top_k: the number of best atoms we choose
        :returns: candidates and budget
        """
        budget = 0

        # traverse tree to find the query leaf
        curr_node = self.tree.root
        while curr_node.m is not None:
            if self.p_signal[curr_node.j] <= curr_node.m:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
            budget += 1

        # get top k candidates
        candidates = curr_node.idcs
        if len(candidates) < top_k:
            # print(f"have {len(candidates)} collisions but need {top_k}!")
            top_k = len(candidates)

        if len(candidates) == 0:
            return [], budget

        dist_vec = np.linalg.norm(self.p_atoms[candidates] - self.p_signal, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += len(candidates) * len(self.p_signal) + len(dist_vec)
        return candidates[idx[:top_k]], budget


def run_pca_mips(
        atoms: np.ndarray,
        signals: np.ndarray,
        num_best_atoms: int ,
        delta: int, 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PCA mips to find num_best_atoms candidates for the atoms and signals inputted.

    :param atoms: Atoms array
    :param signals: Signals array
    :param num_best_atoms: Number of the best atoms we choose
    :param delta: Depth of PCA-tree
    :return: An array of final answers for MIPS, and array of budgets (number of computations).
    """
    pca_object = PCA_MIPS(delta=delta)
    W, mu = pca_object.preprocess_data(atoms)
    candidates_array = []
    budgets_array = []

    
    for signal in signals:
        pca_object.preprocess_query(
            signal=signal,
            pca_matrix=W,
            mu=mu
        )
        start_time = time.time()
        candidates, budgets = pca_object.run(num_best_atoms)
        print(time.time() - start_time)
        candidates_array.append(candidates)
        budgets_array.append(budgets)

    
    return np.array(candidates_array, dtype=object), np.array(budgets_array)




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





if __name__ == "__main__":
    
    
    delta = 2  # corresponds to the depth of PCA-tree
    top_k = 5

    import time

    atoms, signal = generate_data(num_atoms=1000, len_signal=200, num_best_atoms=top_k)
    
    pca_object = PCA_MIPS(delta=delta)
    W, mu = pca_object.preprocess_data(atoms)

    pca_object.preprocess_query(
        signal=signal,
        pca_matrix=W,
        mu=mu
    )
    start_time = time.time()
    candidates, budget = pca_object.run(top_k)
    timetaken = time.time() - start_time
    
    print(f" =>> Top {top_k} candidates: ", sorted(candidates))
    print(" =>> budget: ", budget)


    

    


    
    

