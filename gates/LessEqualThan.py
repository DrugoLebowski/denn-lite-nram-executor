# Vendor
import numpy as np
from numpy import zeros_like

# Project
from gates.Gate import Gate


class LessEqualThan(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        def create_matrix(M, idx):
            new_matrix = zeros_like(M)
            new_matrix[0, idx:] = M[0, idx:]  # [[x, x, x], [0, x, x], [0, 0, x]]
            return new_matrix

        Z = zeros_like(A)
        B_prime = [create_matrix(B, idx) for idx in range(B.shape[1])]
        B = np.stack(B_prime, axis=1)
        A = A[..., None]
        Z[0, 1] = (A[0] * B[0]).sum()
        Z[0, 0] = (1 - Z[0, 1])

        return M, Z