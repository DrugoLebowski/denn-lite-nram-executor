import numpy as np

from gates.Gate import Gate, GateArity
from gates.InequalityTest import InequalityTest

class MinMaxTest(Gate):

    def __init__(self, arity: int, type: int) -> None:
        super().__init__(arity)

        self.type = type


    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        _, Z = InequalityTest(GateArity.BINARY.value, 0).module(M, A, B)

        if self.type is 0:
            return M, A if Z[0, 0] < Z[0, 1] else B
        elif self.type is 1:
            return M, A if Z[0, 0] > Z[0, 1] else B
        else:
            raise Exception("MinMaxFun: the operation must be one element of the set [0, 1]")
