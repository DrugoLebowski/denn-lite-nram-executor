# Vendor
import numpy as np

# Project
from gates.Gate import Gate


class Sub(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        Z = np.zeros_like(A)
        value = 0
        for j in range(M.shape[1]):
            for i in range(M.shape[1]):
                Z[0, value] += A[0, i] * B[0, (i - j) % M.shape[1]]
            value += 1
        return M, Z
