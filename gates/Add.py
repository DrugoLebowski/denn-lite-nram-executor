# Vendor
import numpy as np
from numpy import tensordot, roll, transpose, stack

# Project
from gates.Gate import Gate


class Add(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        rows = [roll(B[:, ::-1], shift=shift + 1, axis=1)
                for shift in range(M.shape[1])]
        B_prime = transpose(stack(rows, axis=1), axes=[0, 2, 1])
        return M, tensordot(A, B_prime, axes=2)[None, ...]
