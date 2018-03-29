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

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        remaining_size = int(self.max_int - 2)
        vector_size = min(int(remaining_size / 2), self.sequence_size)
        starting_point = int(np.floor(self.max_int / 2))

        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = starting_point
        init_mem[:, 1:1 + vector_size] = \
            np.random.randint(1, self.max_int, size=(self.batch_size, vector_size), dtype=np.int32)

        out_mem = init_mem.copy()
        out_mem[:, -1 - vector_size: - 1] = np.flip(out_mem[:, 1:1 + vector_size], axis=1)

        error_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        error_mask[:, -1 - vector_size: - 1] = np.ones((self.batch_size, vector_size))

        return init_mem, out_mem, error_mask
