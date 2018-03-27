# Vendor
import numpy as np

# Project
from factories.TaskFactory import TaskFactory


class NRamContext(object):
    def __init__(self,
                 batch_size:        int,
                 l_max_int:         list,
                 l_sequence_size:   list,
                 l_timesteps:       list,
                 task_type:         str,
                 gates:             list,
                 network:           list,
                 print_circuits:    str,
                 print_memories:    bool,
                 path_config_file:  str,
                 info_is_active:    bool,
                 process_pool:      int,
                 stop_at_the_will:  bool) -> None:
        self.gates = gates
        self.num_regs = len(network[0][0])
        self.num_hidden_layers = len(network[0:len(network) - 1])
        self.network = self.mlp_params(network, self.gates)

        self.batch_size = batch_size
        self.l_max_int = l_max_int
        self.l_sequence_size = l_sequence_size
        self.l_timesteps = l_timesteps
        self.tasks = list()
        for max_int, sequence_size, timesteps in zip(self.l_max_int, self.l_sequence_size, self.l_timesteps):
            self.tasks.append(
                TaskFactory.create(task_type, self.batch_size, max_int, self.num_regs, timesteps, sequence_size))

        # Every entry of the debug list is associated to a sample
        self.info_is_active = info_is_active

        # If None then the circuits will be not draw
        self.print_circuits = print_circuits

        self.print_memories = print_memories

        self.path_config_file = path_config_file

        self.process_pool = process_pool

        self.stop_at_the_will = stop_at_the_will

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
        layers.append(output_layer_weights[:, ptr:ptr + 1])
        layers.append(output_layer_bias[:, ptr:ptr + 1])

        return layers
