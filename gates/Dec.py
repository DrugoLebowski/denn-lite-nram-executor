import numpy as np

from numpy import roll
from gates.Gate import Gate

class Dec(Gate):

    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        return M, roll(A, shift=-1)
