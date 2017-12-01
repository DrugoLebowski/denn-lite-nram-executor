# Vendor
import matplotlib.pyplot as plt
import numpy as np
import pygraphviz as pgv

# Project
from App import App
from NRamContext import NRamContext
from factories.TaskFactory import TaskFactory
from util import exists_or_create

class DebugTimestep(object):
    def __init__(self, context: NRamContext, timestep: int, sample: int) -> None:
        self.context = context
        self.timestep = timestep
        self.sample = sample
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

    def print_circuit(self) -> bool:
        """ Print the circuit for the samples """
        context = self.context

        def retrieve_gates_or_register(idx: int) -> str:
            if idx in range(context.num_regs):
                return "R%s" % idx
            else:
                return context.gates[idx - context.num_regs].__str__()

        task_path = "%s%s" % (App.get("images_path"),
                                context.task.__str__())
        exists_or_create(task_path)
        path = "%s/%s" % (task_path, self.sample)
        exists_or_create(path)

        G = pgv.AGraph(directed=True, name="%s - Timestep %s" % (self.context.task.__str__(), self.timestep))
        G.graph_attr["rankdir"] = "RL"
        for r in range(context.num_regs):
            G.add_node("R%s" % str(r))

        for g in context.gates:
            G.add_node(g.__str__(), shape="rectangle")
            G.get_node(g.__str__()).attr["style"] = "bold"
            for a in range(g.arity):
                coeff = retrieve_gates_or_register(self.gates[g.__str__()][str(a)][0])
                G.add_edge(coeff, g.__str__())

        for r in range(context.num_regs):
            G.add_node("R'%s" % str(r))
            G.add_edge(retrieve_gates_or_register(self.regs[str(r)][0]), "R'%s" % str(r))

        # Removes the unattached register not modified and gates not attached to other objects (gates/register)
        for node in G.nodes_iter():
            if len(list(G.neighbors(node))) == 0:
                G.remove_node(node.name)

        G.layout(prog="twopi")
        G.draw("%s/%s.%s.png" % (path, context.print_circuits_filename[0], self.timestep), format="png")

        return True

    def __str__(self) -> str:

        def register_or_gates(type: int, idx: int, value: int = -1) -> str:
            if idx in range(self.context.num_regs):  # Is a register
                return "R%d: %d" % (idx, value) if type is 0 else "R%d" % (idx)
            else:  # Otherwise is a gate
                return "%s: %d" % (self.context.gates[idx - self.context.num_regs].__str__(), value) if type is 0 \
                    else "%s" % (self.context.gates[idx - self.context.num_regs].__str__())

        output = "Timestep %d\n" % (self.timestep)
        for g in self.context.gates:
            values = [self.gates[g.__str__()][str(a)] for a in range(g.arity)]
            output += "\t• %s(%s) => %d\n" \
                      % (
                      g, ", ".join([register_or_gates(0, *value) for value in values]), self.gates[g.__str__()]["res"])

        for r in range(self.context.num_regs):
            output += "\t• R%d' (%s) => %d\n" % (r, register_or_gates(1, self.regs[str(r)][0]), self.regs[str(r)][1])

        output += "\t• Mem => %s" % self.mem
        return output
