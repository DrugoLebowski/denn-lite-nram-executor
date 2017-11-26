import numpy as np

from numpy import zeros_like, stack
from gates.Gate import Gate


class InequalityTest(Gate):

    def __init__(self, arity: int, type: int) -> None:
        super().__init__(arity)

        self.type = type

    def _create_matrix(self, idx: int, M: np.array) -> np.array:
        matrix = zeros_like(M)
        if self.type is 0: # Less Than
            # [0, x, x, ..., x], [0, 0, x, ..., x], [0, 0, 0, ..., x], ..., [0, 0, ..., 0]
            matrix[(idx + 1):] = M[(idx + 1):]
        elif self.type is 1: # Less Equal Than
            matrix[idx:] =  M[idx:] # [x, x, x, ..., x], [0, x, x, ..., x], ..., [0, 0, ..., 0, x]
        elif self.type is 2: # Equal
            matrix[idx] = M[idx] # [x, 0, ..., 0, 0], [0, ..., 0, x, 0, ..., 0], [0, 0, ..., 0, x]
        else:
            raise Exception("Type must be between one of [0, 1, 2]")
        return matrix

    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        rows = [self._create_matrix(idx, B) for idx in range(M.shape[1])]
        matrix = stack(rows, axis=0)

        A_pr = A[..., None]  # M [x, x, x] ->  M x 1 [[x], [x], [x]]
        prob = (A_pr * matrix).sum()

        Z = zeros_like(A)
        Z[0, 0] = (1 - prob)
        Z[0, 1] = prob

        return M, Z