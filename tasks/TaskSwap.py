# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskSwap(Task):
    """ [Swap]
    Given two pointers p, q and an array A, swap elements A[p] and A[q]. Input is
    given as p, q, A[0], .., A[p], ..., A[q], ..., A[n − 1], 0. The expected modified array A is:
    A[0], ..., A[q], ..., A[p], ..., A[n − 1].
    """

    def create(self) -> (np.array, np.array):
        idx_1 = np.random.randint(2, self.max_int - 3, size=(self.batch_size), dtype=np.int32)
        idx_2 = np.array([], dtype=np.int32)
        for i, n in enumerate(idx_1):
            idx_2 = np.append(idx_2, np.random.randint(idx_1[i] + 1, self.max_int - 2))
        init_mem = np.random.randint(1, self.max_int, size=(self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, self.max_int - 1] = 0
        init_mem[:, 0] = idx_1
        init_mem[:, 1] = idx_2

        out_mem = init_mem.copy()
        for i in range(self.batch_size):
            out_mem[i, idx_1[i]], out_mem[i, idx_2[i]] = np.copy(out_mem[i, idx_2[i]]), np.copy(out_mem[i, idx_1[i]])

        return init_mem, out_mem