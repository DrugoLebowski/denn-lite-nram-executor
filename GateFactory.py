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
from gates.LessThan import LessThan
from gates.LessEqualThan import LessEqualThan
from gates.EqualThan import EqualThan
from gates.Min import Min
from gates.Max import Max
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
            "lt":    LessThan(GateArity.BINARY.value),
            "let":   LessEqualThan(GateArity.BINARY.value),
            "eq":    EqualThan(GateArity.BINARY.value),
            "min":   Min(GateArity.BINARY.value),
            "max":   Max(GateArity.BINARY.value),
            "write": Write(GateArity.BINARY.value),
        }[cls]