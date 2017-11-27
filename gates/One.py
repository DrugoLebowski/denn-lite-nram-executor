import numpy as np

from numpy import tensordot
from gates.Gate import Gate
from util import to_one_hot

class One(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        return M, to_one_hot(1, M.shape[1])