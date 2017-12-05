# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskSwap(Task):
    """ Access the position in memory listed in the first position of the latter """

    def create(self) -> (np.array, np.array):
        init_mem = np.random.randint(0, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, self.max_int - 1] = 0

        out_mem = init_mem.copy()
        out_mem[:, int(self.max_int / 3)], out_mem[:, int(self.max_int / 2) + 1] = \
            np.copy(out_mem[:, int(self.max_int / 2) + 1]), np.copy(out_mem[:, int(self.max_int / 3)])

        return init_mem, out_mem