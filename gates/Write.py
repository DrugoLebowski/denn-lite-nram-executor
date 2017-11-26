import numpy as np

from numpy import tensordot, transpose, zeros_like, ones_like
from gates.Gate import Gate

class Write(Gate):

    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        erase   = tensordot(transpose(ones_like(A) - A, axes=[1, 0]), ones_like(A), axes=1)
        contrib = tensordot(transpose(A, axes=[1, 0]), B, axes=1)
        new_mem = (erase * M) + contrib
        return new_mem, zeros_like(A)
