# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskCopy(Task):

    def create(self) -> (np.array, np.array):
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