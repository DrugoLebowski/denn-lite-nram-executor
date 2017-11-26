import numpy as np

from enum import Enum

class GateArity(Enum):

    # Gate arity
    CONST  = 0
    UNARY  = 1
    BINARY = 2

class Gate(object):
    """ Base class for Gate """

    def __init__(self, arity) -> None:
        super(Gate, self).__init__()
        
        self.arity = arity

    def module(self, M: np.array, A: np.array = None, B: np.array = None) -> (np.array, np.array):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__
