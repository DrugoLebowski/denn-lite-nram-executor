# Vendor
import numpy as np

# Project
from tasks.Task import Task


class TaskAccess(Task):
    """ [Access]
    Given a value k and an array A, return A[k]. Input is given as k, A[0], .., A[n - 1], NULL
    and the network should replace the first memory cell with A[k].
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        if self.sequence_size + 2 != self.max_int:
            raise Exception("Incompatible size of Max Int and input sequence.")
        init_mem = np.random.randint(0, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = np.random.randint(1, self.max_int - 1, size=(self.batch_size), dtype=np.int32)
        init_mem[:, self.max_int - 1] = 0

        out_mem = init_mem.copy()
        for sample, idx in enumerate(init_mem[:, 0]):
            out_mem[sample, 0] = init_mem[sample, idx]

        error_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        error_mask[:, 0] = 1

        return init_mem, out_mem, error_mask
