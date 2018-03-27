# Vendor
import numpy as np

# Project
from tasks.Task import Task
from util import encode


class TaskPermutation(Task):
    """ [Permutation]
        Given two arrays of n elements: P (contains a permutation of numbers 0, . . . , n − 1) and
        A (contains random elements), permutate A according to P. Input is given as
        a, P[0], ..., P[n − 1], A[0], ..., A[n − 1], where a is a pointer to the array A. The
        expected output is A[P[0]], ..., A[P[n − 1]], which should override the array P.
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        pointer = int(np.ceil(self.max_int / 2) if self.max_int % 2 != 0 else self.max_int / 2)

        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        init_mem[:, 0] = pointer
        for idx in range(self.batch_size):
            init_mem[idx, 1:(1 + self.sequence_size)] = np.random.permutation(self.sequence_size)
        init_mem[:, pointer:(pointer + self.sequence_size)] = \
            np.random.randint(1, self.max_int, size=(self.batch_size, self.sequence_size), dtype=np.int32)

        out_mem = init_mem.copy()
        permutations = encode(out_mem[:, 1:(1 + self.sequence_size)])
        for idx in range(self.batch_size):
            out_mem[idx, 1:(1 + self.sequence_size)] = np.tensordot(out_mem[idx, pointer:(pointer + self.sequence_size)],
                                                   permutations[idx], axes=(0, 1))

        cost_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        cost_mask[:, 1:(1 + self.sequence_size)] = 1

        return init_mem, out_mem, cost_mask
