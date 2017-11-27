import numpy as np

from numpy import tensordot
from gates.Gate import Gate

class Read(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        return M, tensordot(A, M, axes=1)
