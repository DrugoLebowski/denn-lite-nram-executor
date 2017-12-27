# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskMerge(Task):
    """ [Merge]
    Given pointers to 2 sorted arrays A and B, and the pointer to the output o,
    merge the two arrays into one sorted array. The input is given as: a, b, o, A[0], .., A[n −1],
    G, B[0], ..., B[m − 1], G, where G is a special guardian value, a and b point to the first
    elements of arrays A and B respectively, and o points to the address after the second G.
    The n + m element should be written in correct order starting from position o.
    """

    def create(self) -> (np.array, np.array):
        remaining_size = self.max_int - 6
        offset = 3
        odd_space = not remaining_size % 2 == 0
        odd_subvector_space = not remaining_size % 4 == 0
        list_size_a = int(remaining_size / 4)
        list_size_b = int(list_size_a + 1 if odd_subvector_space else list_size_a)
        list_size_a_plus_b = list_size_a + list_size_b

        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        list_elements_a = np.sort(np.random.randint(1, self.max_int, size=(self.batch_size, list_size_a), dtype=np.int32))
        list_elements_b = np.sort(np.random.randint(1, self.max_int, size=(self.batch_size, list_size_b), dtype=np.int32))
        list_elements_a_union_b = np.sort(np.concatenate((list_elements_a, list_elements_b), axis=1))

        init_mem[:, 0] = offset
        init_mem[:, 1] = offset + list_size_a + 1
        init_mem[:, 2] = offset + (2 * list_size_a) + (3 if odd_subvector_space else 2)
        init_mem[:, offset:(offset + list_size_a)] = list_elements_a
        init_mem[:, (offset + list_size_a)] = -1
        init_mem[:, (offset + list_size_a + 1):(offset + list_size_a_plus_b + 1)] = list_elements_b
        init_mem[:, (offset + list_size_a_plus_b + 1)] = -1
        if odd_space:
            init_mem[:, -2:] = -1
        else:
            init_mem[:, -1] = -1

        out_mem = init_mem.copy()
        out_mem[:, (offset + list_size_a_plus_b + 2):(offset + (2 * list_size_a_plus_b) + 2)] = list_elements_a_union_b

        return init_mem, out_mem