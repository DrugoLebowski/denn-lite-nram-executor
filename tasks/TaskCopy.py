# Vendor
import numpy as np

# Project
from tasks.Task import Task


class TaskCopy(Task):
    """ [Copy]
    Given an array and a pointer to the destination, copy all elements from the array to
    the given location. Input is given as p, A[0], ..., A[n−1] where p points to one element after
    A[n−1]. The expected output is A[0], ..., A[n−1] at positions p, ..., p+n−1 respectively.
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        remaining_size = int(self.max_int - 2)
        vector_size = int(remaining_size / 2)
        starting_point = int(np.floor(self.max_int / 2))

        init_mem = np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = starting_point
        init_mem[:, starting_point:] = np.zeros((self.batch_size, self.max_int - starting_point))

        out_mem = init_mem.copy()
        out_mem[:, starting_point:starting_point + vector_size] = np.copy(out_mem[:, 1:1 + vector_size])

        error_mask = np.ones((self.batch_size, self.max_int), dtype=np.int8)
        error_mask[:, 0:starting_point] = np.zeros((self.batch_size, starting_point))
        error_mask[:, -1] = np.zeros((self.batch_size))

        return init_mem, out_mem, error_mask
