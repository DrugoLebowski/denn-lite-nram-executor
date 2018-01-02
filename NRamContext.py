# Vendor
import numpy as np

# Project
from factories.TaskFactory import TaskFactory


class NRamContext(object):
    def __init__(self, batch_size: int, max_int: int, timesteps: int,
                 task_type: str, gates: list, network: list, debug_is_active: bool,
                 print_circuits: str, print_memories: str,
                 path_config_file: str, ) -> None:
        self.gates = gates
        self.num_regs = len(network[0][0])
        self.num_hidden_layers = len(network[0:len(network) - 1])
        self.network = self.mlp_params(network, self.gates)

        self.batch_size = batch_size
        self.max_int = max_int
        self.timesteps = timesteps
        self.task = TaskFactory.create(task_type, self.batch_size, self.max_int, self.num_regs, self.timesteps)

        # Every entry of the debug list is associated to a sample
        self.debug = list()
        self.debug_is_active = debug_is_active

        # If None then the circuits will be not draw
        self.print_circuits = print_circuits

        # Like above, but with memories
        self.print_memories = print_memories

        self.path_config_file = path_config_file

    def mlp_params(self, network: list, gates_list: list) -> list:
        # Hidden layers (Not output)
        layers = []
        for idx, l in enumerate(network[:-1]):
            layers.append(np.array(l[0], dtype=np.float32))  # Weights
            layers.append(np.array(l[1], dtype=np.float32))  # Bias

        # Output layers (for every gate coefficient)
        ptr = 0
        output_layer = network[-1]
        output_layer_weights = np.array(output_layer[0], dtype=np.float32)
        output_layer_bias = np.array(output_layer[1], dtype=np.float32)
        num_registers = self.num_regs
        for idx, g in enumerate(gates_list):
            for _ in range(g.arity):
                layers.append(output_layer_weights[:, ptr:ptr + num_registers])  # Weights
                layers.append(output_layer_bias[:, ptr:ptr + num_registers])  # Bias
                ptr += num_registers
            num_registers += 1

        # Output layers (for every register coefficient)
        for r in range(self.num_regs):
            layers.append(output_layer_weights[:, ptr:ptr + num_registers])  # Weights
            layers.append(output_layer_bias[:, ptr:ptr + num_registers])  # Bias
            ptr += num_registers

        # Output layer for the willingness of finish f_t
        layers.append(output_layer_weights[:, -1])
        layers.append(output_layer_bias[:, -1])

        return layers
