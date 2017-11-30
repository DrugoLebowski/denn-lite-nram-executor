import numpy as np

from numpy import zeros_like
from gates.Gate import Gate


class LessEqualThan(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        Z = zeros_like(A)
        for i in range(M.shape[1]):
            for j in range(i, M.shape[1]):
                Z[0, 1] += A[0, i] * B[0, j]
        Z[0, 0] = (1 - Z[0, 1])

        return M, Z