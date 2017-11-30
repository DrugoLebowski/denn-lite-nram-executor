# Vendor
import numpy as np

# Project
from util import to_one_hot
from gates.Gate import Gate


class Zero(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        return M, to_one_hot(0, M.shape[1])