# Vendor
import numpy as np
from numpy import tensordot, transpose, zeros_like, ones_like

# Project
from gates.Gate import Gate


class Write(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        erase   = tensordot(transpose(ones_like(A) - A, axes=[1, 0]), ones_like(A), axes=1)
        contrib = tensordot(transpose(A, axes=[1, 0]), B, axes=1)
        new_mem = (erase * M) + contrib
        return new_mem, np.array([x if i != 0 else x + 1 for i, x in enumerate(zeros_like(A))], dtype=np.float64)
