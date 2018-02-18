# Vendor
import numpy as np

# Project
from tasks.Task import Task


class TaskProduct(Task):
    """ [Product]
    Given pointers to 2 arrays A and B, and the pointer to the output o,
    sum the two arrays into one array. The input is given as:
    a, b, o, A[0], .., A[n − 1], G, B[0], ..., B[m − 1], G,
    where G is a special guardian value, a and b point to the first
    elements of arrays A and B respectively, and o is a slot for the output.
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        offset = 3
        remaining_size = int(self.max_int - 5)
        arrays_memory_size = int(remaining_size / 2)

        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        list_elements_a = np.random.randint(1, self.max_int, size=(self.batch_size, arrays_memory_size), dtype=np.int32)
        list_elements_b = np.random.randint(1, self.max_int, size=(self.batch_size, arrays_memory_size), dtype=np.int32)
        prod_a_b = np.mod(np.sum(np.multiply(list_elements_a, list_elements_b), axis=1), self.max_int)

        init_mem[:, 0] = offset
        init_mem[:, 1] = offset + arrays_memory_size + 1
        init_mem[:, 2] = offset + (2 * arrays_memory_size) + 2
        init_mem[:, offset:(offset + arrays_memory_size)] = list_elements_a
        init_mem[:, (offset + arrays_memory_size + 1):(offset + (2 * arrays_memory_size + 1))] = list_elements_b

        out_mem = init_mem.copy()
        out_mem[:, offset - 1] = prod_a_b

        cost_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        cost_mask[:, offset - 1] = 1

        return init_mem, out_mem, cost_mask
