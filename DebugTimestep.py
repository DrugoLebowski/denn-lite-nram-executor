# Vendor
import numpy as np

# Project
from NRamContext import NRamContext


class DebugTimestep(object):

    def __init__(self, context: NRamContext, timestep: int) -> None:
        self.context = context
        self.timestep = timestep
        self._gates = dict()
        self._regs = dict()
        self._mem = np.array([], dtype=np.float32)


    @property
    def gates(self) -> dict():
        return self._gates


    @gates.setter
    def gates(self, gates: dict) -> None:
        self._gates = gates


    @property
    def regs(self) -> dict():
        return self._regs


    @regs.setter
    def regs(self, regs: dict) -> None:
        self._regs = regs


    @property
    def mem(self) -> np.array:
        return self._mem


    @mem.setter
    def mem(self, mem: dict) -> None:
        self._mem = mem


    def __str__(self) -> str:

        def register_or_gates(type: int, idx: int, value: int = -1) -> str:
            if idx in range(self.context.num_regs): # Is a register
                return "R%d: %d" % (idx, value) if type is 0 else "R%d" % (idx)
            else: # Otherwise is a gate
                return "%s: %d" % (self.context.gates[idx - len(self.context.gates)].__str__(), value) if type is 0 \
                    else "%s" % (self.context.gates[idx - len(self.context.gates)].__str__())

        output = "Timestep %d\n" % (self.timestep)
        for g in self.context.gates:
            values = [self.gates[g.__str__()][str(a)] for a in range(g.arity)]
            output += "\t• %s(%s) => %d\n"\
                % (g, ", ".join([register_or_gates(0, *value) for value in values]), self.gates[g.__str__()]["res"])

        for r in range(self.context.num_regs):
            output += "\t• R%d' (%s) => %d\n" % (r, register_or_gates(1, self.regs[str(r)][0]), self.regs[str(r)][1])

        output += "\t• Mem => %s" % self.mem
        return output
