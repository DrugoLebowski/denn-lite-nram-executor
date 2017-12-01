# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskAccess(Task):
    """ Access the position in memory listed in the first position of the latter """

    def create(self) -> (np.array, np.array):
        init_mem = np.random.randint(0, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = 4
        init_mem[:, self.max_int - 1] = 0

        out_mem = init_mem.copy()
        out_mem[:, 0] = out_mem[:, 4]

        return init_mem, out_mem