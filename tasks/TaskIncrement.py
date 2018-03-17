# Vendor
import numpy as np

# Project
from tasks.Task import Task


class TaskIncrement(Task):
    """ [Increment]
    Given an array A, increment all its elements by 1. Input is given as
    A[0], ..., A[n − 1], NULL and the expected output is A[0] + 1, ..., A[n − 1] + 1.
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Task 5: Increment"""
        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, :self.max_int - 1] = \
            np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int - 1), dtype=np.int32)

        out_mem = init_mem.copy()
        out_mem[:, :self.timesteps] = np.mod(np.add(init_mem[:, :self.timesteps], [1]), self.max_int)

        cost_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        cost_mask[:, 0:self.timesteps] = 1

        return init_mem, out_mem, cost_mask
