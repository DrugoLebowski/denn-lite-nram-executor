import numpy as np

from numpy import tensordot, roll, transpose, stack
from gates.Gate import Gate

class Sub(Gate):

    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        rows = [roll(B, shift=shift, axis=1)
                for shift in range(M.shape[1])]
        B_prime = transpose(stack(rows, axis=0), axes=[1, 0])
        return M, tensordot(A, B_prime, axes=1)
