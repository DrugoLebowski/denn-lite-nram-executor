# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskSum(Task):
    """ [Sum]
    Given pointers to 2 arrays A and B, and the pointer to the output o,
    sum the two arrays into one array. The input is given as:
    a, b, o, A[0], .., A[n − 1], G, B[0], ..., B[m − 1], G,
    where G is a special guardian value, a and b point to
    the first elements of arrays A and B respectively, and o
    points to the address after the second G.
    The A + B array should be written starting from position o.
    """

    def create(self) -> (np.array, np.array):
        offset = 3
        remaining_size = self.max_int - 6
        arrays_memory_size = int(remaining_size / 3)
        if not remaining_size % 3 == 0:
            raise Exception("%s: Memory space is not sufficient!" % TaskSum.__class__.__name__)

        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        list_elements_a = np.random.randint(1, self.max_int, size=(self.batch_size, arrays_memory_size), dtype=np.int32)
        list_elements_b = np.random.randint(1, self.max_int, size=(self.batch_size, arrays_memory_size), dtype=np.int32)
        list_elements_a_plus_b = np.mod(list_elements_a + list_elements_b, self.max_int)

        init_mem[:, 0] = offset
        init_mem[:, 1] = offset + arrays_memory_size + 1
        init_mem[:, 2] = offset + (2 * arrays_memory_size) + 2
        init_mem[:, offset:(offset + arrays_memory_size)] = list_elements_a
        init_mem[:, (offset + arrays_memory_size + 1):(offset + (2 * arrays_memory_size) + 1)] = list_elements_b

        out_mem = init_mem.copy()
        out_mem[:, (offset + (2 * arrays_memory_size) + 2):(offset + (3 * arrays_memory_size) + 2)] = \
            list_elements_a_plus_b

        return init_mem, out_mem