# Standard
from subprocess import Popen

# Vendor
import numpy as np
import pygraphviz as pgv

# Project
from Node import Node
from NRamContext import NRamContext


class DebugTimestep(object):
    def __init__(self, context: NRamContext, timestep: int, sample: int) -> None:
        self.context = context
        self.timestep = timestep
        self.sample = sample
        self.__gates = dict()
        self.__regs = dict()
        self.__regs_previous_mod = dict()
        self.__mem = np.array([], dtype=np.float32)
        self.__mem_previous_mod = np.array([], dtype=np.float32)
        self.__fi = 0

    @property
    def gates(self) -> dict():
        return self.__gates

    @gates.setter
    def gates(self, gates: dict) -> None:
        self.__gates = gates

    @property
    def regs(self) -> dict:
        return self.__regs

    @regs.setter
    def regs(self, regs: dict) -> None:
        self.__regs = regs

    @property
    def mem(self) -> np.ndarray:
        return self.__mem

    @mem.setter
    def mem(self, mem: dict) -> None:
        self.__mem = mem

    @property
    def mem_previous_mod(self) -> np.ndarray:
        return self.__mem_previous_mod

    @mem_previous_mod.setter
    def mem_previous_mod(self, mem: dict) -> None:
        self.__mem_previous_mod = mem

    @property
    def regs_previous_mod(self) -> dict:
        return self.__regs_previous_mod

    @regs_previous_mod.setter
    def regs_previous_mod(self, regs: dict) -> None:
        self.__regs_previous_mod = regs

    @property
    def fi(self) -> float:
        return self.__fi

    @fi.setter
    def fi(self, fi):
        self.__fi = fi

    def __retrieve_gates_or_register(self, idx: int) -> str:
        """Retrieve the right name for a coefficient (i.e. if it is a Register or a Gate)"""
        if idx in range(self.context.num_regs):
            return "R%s" % idx
        else:
            return self.context.gates[idx - self.context.num_regs].__str__()

    def print_pruned_circuit(self, path: str) -> bool:
        """ Print the circuit for the samples """

        context = self.context
        nodes = {}

        for r in range(context.num_regs):
            nodes["R%s" % str(r)] = Node(Node.Register, "R%s" % str(r))

        for g in context.gates:
            nodes[g.__str__()] = Node(Node.Register, g.__str__(), g.arity)
            for a in range(g.arity):
                coeff = self.__retrieve_gates_or_register(self.gates[g.__str__()][str(a)][0])
                nodes[coeff].add_node(nodes[g.__str__()])

        for r in range(context.num_regs):
            node_name = "R'%s" % str(r)
            nodes["R'%s" % str(r)] = Node(Node.Register, node_name)
            nodes[self.__retrieve_gates_or_register(self.regs[str(r)][0])].add_node(nodes[node_name])

        right_nodes = []
        for key, node in nodes.items():
            if node.check_validity() and key not in right_nodes:
                right_nodes.append(node.name)

        return self.print_circuit(path, right_nodes)

    def print_circuit(self, path: str, nodes_to_prune: list = list()) -> bool:
        context = self.context

        G = pgv.AGraph(directed=True, strict=False, name="%s - Timestep %s" % (self.context.tasks[0].__str__(), self.timestep))
        G.graph_attr["rankdir"] = "LR"
        for r in range(context.num_regs):
            node_name = "R%s" % str(r)
            if len(nodes_to_prune) == 0 or node_name in nodes_to_prune:
                G.add_node(node_name, shape="circle")

        for g in context.gates:
            if len(nodes_to_prune) == 0 or g.__str__() in nodes_to_prune:
                G.add_node(g.__str__(), shape="rectangle")
                G.get_node(g.__str__()).attr["style"] = "bold"
                for a in range(g.arity):
                    coeff = self.__retrieve_gates_or_register(self.gates[g.__str__()][str(a)][0])
                    G.add_edge(coeff, g.__str__())
                    if g.__str__() is "Write":
                        G.get_edge(coeff, g.__str__()).attr["label"] = "ptr" if a is 0 else "val"
                    elif g.__str__() is "Read":
                        G.get_edge(coeff, g.__str__()).attr["label"] = "ptr"
                    elif g.__str__() in ["Add", "Sub", "LessThan", "LessEqualThan", "EqualThan", "Min", "Max"]:
                        G.get_edge(coeff, g.__str__()).attr["label"] = "x" if a is 0 else "y"

        for r in range(context.num_regs):
            node_name = "R'%s" % str(r)
            if len(nodes_to_prune) == 0 or node_name in nodes_to_prune:
                G.add_node("R'%s" % str(r))
                G.add_node(node_name, shape="circle")
                G.add_edge(self.__retrieve_gates_or_register(self.regs[str(r)][0]), node_name)

        # Removes the unattached register not modified and gates not attached to other objects (gates/register)
        for node in G.nodes_iter():
            if len(list(G.neighbors(node))) == 0:
                G.remove_node(node.name)

        G.layout(prog="dot")
        circuit_path = "%s/circuit.%s" % (path, self.timestep)
        G.draw("%s.png" % circuit_path, format="png")
        G.write("%s.dot" % circuit_path)

        Popen("dot2tex -ftikz %s.dot > %s.tex" % (circuit_path, circuit_path), shell=True).wait()

        return True

    def print_memory_to_file(self, path: str, max_timestep: int):
        t = self.timestep
        with open("%s/memories.txt" % path, "a+") as f:
            timestep_regs = [reg[1] for idx, reg in self.regs.items()]
            timestep_regs_previous_mod = [reg[1] for idx, reg in self.regs_previous_mod.items()]
            if t + 1 < max_timestep:
                f.write("%d & %s & %s & p:%s & p:%s v:%s \\\\ \n"
                        % (t + 1,
                           " & ".join(["%d" % v for v in self.mem_previous_mod]),
                           " & ".join(["%d" % r for r in timestep_regs]),
                           self.gates["Read"]['0'][1],
                           self.gates["Write"]['0'][1], self.gates["Write"]['1'][1]))
            else:
                f.write("%d & %s & %s & p:%s & p:%s v:%s\\\\ \hline \n"
                        % ( t + 1,
                            " & ".join(["%d" % v for v in self.mem_previous_mod]),
                            " & ".join(["%d" % r for r in timestep_regs]),
                            self.gates["Read"]['0'][1],
                            self.gates["Write"]['0'][1], self.gates["Write"]['1'][1]))
                f.write("\\rowcolor{Gray}Final & %s & %s & $\\times$ & $\\times$ \\\\"
                        % (" & ".join(["%d" % v for v in self.mem]),
                           " & ".join(["%d" % r for r in timestep_regs])))

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

        output += "\t• Fi => %s\n" % self.fi
        output += "\t• Mem => %s" % self.mem
        return output
