# Vendor
import numpy as np
from numpy import tensordot, roll, transpose, stack

# Project
from gates.Gate import Gate


class Add(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        Z = np.zeros_like(A)
        value = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[1]):
                Z[0, value] += A[0, i] * B[0, (j - i) % M.shape[1]]
            value += 1
        return M, Z
