from Singleton import Singleton
from gates.Gate import GateArity
from gates.Read import Read
from gates.Zero import Zero
from gates.One import One
from gates.Two import Two
from gates.Inc import Inc
from gates.Add import Add
from gates.Sub import Sub
from gates.Dec import Dec
from gates.InequalityTest import InequalityTest
from gates.MinMaxTest import MinMaxTest
from gates.Write import Write

class GateFactory(metaclass=Singleton):

    @staticmethod
    def create(cls):
        return {
            "read":  Read(GateArity.UNARY.value),
            "zero":  Zero(GateArity.CONST.value),
            "one":   One(GateArity.CONST.value),
            "two":   Two(GateArity.CONST.value),
            "inc":   Inc(GateArity.UNARY.value),
            "add":   Add(GateArity.BINARY.value),
            "sub":   Sub(GateArity.BINARY.value),
            "dec":   Dec(GateArity.UNARY.value),
            "lt":    InequalityTest(GateArity.BINARY.value, 0),
            "let":   InequalityTest(GateArity.BINARY.value, 1),
            "eq":    InequalityTest(GateArity.BINARY.value, 2),
            "min":   MinMaxTest(GateArity.BINARY.value, 0),
            "max":   MinMaxTest(GateArity.BINARY.value, 1),
            "write": Write(GateArity.BINARY.value),
        }[cls]