# Vendor
import numpy as np
from numpy import zeros_like

# Project
from gates.Gate import Gate


class EqualThan(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        Z = zeros_like(A)
        Z[0, 1] = np.tensordot(A, B, axes=2)
        Z[0, 0] = (1 - Z[0, 1])
        return M, Z