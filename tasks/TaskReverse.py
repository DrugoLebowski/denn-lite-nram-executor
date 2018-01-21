# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskReverse(Task):
    """ [Reverse]
    Given an array and a pointer to the destination, copy all elements from the array
	in reversed order. Input is given as p, A[0], ..., A[n − 1] where p points one element after
    A[n−1]. The expected output is A[n−1], ..., A[0] at positions p, ..., p+n−1 respectively.
    """

    def create(self) -> (np.array, np.array):
        remaining_size = int(self.max_int - 2)
        vector_size = int(remaining_size / 2)
        starting_point = int(np.floor(self.max_int / 2))

        init_mem = np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = starting_point
        init_mem[:, starting_point:self.max_int - starting_point] = \
            np.zeros((self.batch_size, self.max_int - starting_point))

        out_mem = init_mem.copy()
        out_mem[:, starting_point:starting_point + vector_size] = np.flip(out_mem[:, 1:1 + vector_size], axis=1)

        cost_mask = np.ones((self.batch_size, self.max_int))
        cost_mask[:, 0:starting_point] = np.zeros((self.batch_size, starting_point))

        return init_mem, out_mem