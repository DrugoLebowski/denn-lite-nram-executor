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
        starting_point = np.floor(self.max_int / 2).__int__() \
            if self.max_int % 2 == 0 \
            else np.ceil(self.max_int / 2).__int__()

        init_mem = np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = starting_point
        init_mem[:, starting_point:self.max_int] = \
            np.zeros((self.batch_size, self.max_int - starting_point))

        out_mem = init_mem.copy()
        out_mem[:, starting_point:self.max_int - 1] = \
            np.flip(out_mem[:, (1 if self.max_int % 2 == 0 else 2):starting_point], axis=1)

        return init_mem, out_mem