# Vendor
import numpy as np

# Project
from gates.Gate import Gate, GateArity
from gates.LessThan import LessThan

class Min(Gate):

    def __call__(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        _, Z = LessThan(GateArity.BINARY.value)(M, A, B)
        return M, A if Z[0, 0] < Z[0, 1] else B
