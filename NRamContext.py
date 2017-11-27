import numpy as np

from util import to_one_hot

class NRamContext(object):

    def __init__(self, batch_size: int, max_int: int, timesteps: int,
                 task_type: str, gates: list, network: list) -> None:
        self.gates = gates
        self.num_regs = len(network[0][0])
        self.num_hidden_layers = len(network[0:len(network) - 1])
        self.network = list(self.mlp_params(network, self.gates))

        self.batch_size = batch_size
        self.max_int = max_int
        self.timesteps = timesteps
        self.task_type = task_type

    def mlp_params(self, network: list, gates_list: list) -> None:
        # Layers (Not output)
        for idx, l in enumerate(network):
            if idx < len(network) - 1:
                weights = np.array(l[0], dtype=np.float32)
                bias = np.array(l[1], dtype=np.float32)
                yield np.array(weights, dtype=np.float32)  # Weights
                yield np.array(bias, dtype=np.float32)   # Bias

        # Output layers (for every gate coefficient)
        ptr = 0
        output_layer = network[-1]
        output_layer_weights = np.array(output_layer[0], dtype=np.float32)
        output_layer_bias = np.array(output_layer[1], dtype=np.float32)
        num_registers = self.num_regs
        for idx, g in enumerate(gates_list):
            for _ in range(g.arity):
                yield np.array(output_layer_weights[:, ptr:ptr + num_registers], dtype=np.float32)  # Weights
                yield np.array(output_layer_bias[:, ptr:ptr + num_registers], dtype=np.float32)  # Bias
                ptr += num_registers
            num_registers += 1

        # Output layers (for every register coefficient)
        for r in range(output_layer_weights.shape[0]):
            yield np.array(output_layer_weights[:, ptr:ptr + num_registers], dtype=np.float32)  # Weights
            yield np.array(output_layer_bias[:, ptr:ptr + num_registers], dtype=np.float32)  # Bias
            ptr += num_registers

        # Output layer for the willingness of finish f_t
        yield np.array(output_layer_weights[:, -1], dtype=np.float32)
        yield np.array(output_layer_bias[:, -1], dtype=np.float32)

    def generate_mems(self):
        in_mem, out_mem = getattr(self, self.task_type)()
        return self.fuzzyfy_mem(in_mem), out_mem, self.init_regs(np.zeros((self.batch_size, self.num_regs, self.max_int)))

    def init_regs(self, regs):
        regs[:, :, 0] = 1.0
        return regs

    def fuzzyfy_mem(self, M: np.array) -> np.array:
        fuzzyfied_mems = []
        for s in M:
            sample_fuzzyfied_mem = []
            for n in s:
                sample_fuzzyfied_mem.append(to_one_hot(n, M.shape[1]))
            fuzzyfied_mems.append(np.stack(sample_fuzzyfied_mem, axis=0))

        return np.stack(fuzzyfied_mems, axis=0)

    def task_access(self):
        """Task 1: Access the position in memory listed in the first position of the latter"""

        init_mem = np.random.randint(0, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = 4
        init_mem[:, self.max_int - 1] = 0

        out_mem = init_mem.copy()
        out_mem[:, 0] = out_mem[:, 4]

        return init_mem, out_mem


    def task_copy(self):
        """Task 2: Copy"""

        starting_point = np.floor(self.max_int / 2).__int__() \
            if self.max_int % 2 == 0 \
            else np.ceil(self.max_int / 2).__int__()

        init_mem = np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = starting_point
        init_mem[:, starting_point:self.max_int] = \
            np.zeros((self.batch_size, self.max_int - starting_point))

        out_mem = init_mem.copy()
        out_mem[:, starting_point:self.max_int - 1] = out_mem[:, (1 if self.max_int % 2 == 0 else 2):starting_point]

        return init_mem, out_mem
