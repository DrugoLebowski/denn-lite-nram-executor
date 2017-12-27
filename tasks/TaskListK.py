# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskListK(Task):
    """ [ListK]
    Given a pointer to the head of a linked list and a number k, find the value of the
    k-th element on the list. List nodes are represented as two adjacent memory cells: a pointer
    to the next node and a value. Elements are in random locations in the memory, so that
    the network needs to follow the pointers to find the correct element. Input is given as:
    head, k, out, ... where head is a pointer to the first node on the list, k indicates how many
    hops are needed and out is a cell where the output should be put.
    """

    def create(self) -> (np.ndarray, np.ndarray):
        list_size = int((self.max_int - 4) / 2)
        hops = np.random.randint(0, list_size, size=(self.batch_size))
        list_elements = np.random.randint(0, self.max_int, size=(self.batch_size, list_size))
        lists_elements_permutations = np.stack([np.random.permutation(list_size) for _ in range(self.batch_size)], axis=0)
        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)

        # Create for each example the list
        for example in range(self.batch_size):
            for j, permidx in enumerate(lists_elements_permutations[example]):
                next_element_pointer = np.where(lists_elements_permutations[example] == permidx + 1)[0]
                if permidx == 0: # If the node is the first than set the pointer in the first memory position
                    init_mem[example, 0] = 3 + 2 * j

                init_mem[example, 3 + (2 * j)] = \
                    -1.0 if len(next_element_pointer) == 0 else 3 + (2 * next_element_pointer[0]) # Set the pointer to the next list node
                init_mem[example, 3 + (2 * j) + 1] = list_elements[example, j] # Set the value of the list node
        init_mem[:, 2] = 2
        init_mem[:, 1] = hops
        init_mem[:, -1] = -1

        out_mem = init_mem.copy()
        for example in range(self.batch_size):
            output_value = -1.0
            pointer = out_mem[example, 0]
            for hop in range(out_mem[example, 1] + 1):
                output_value = out_mem[example, pointer + 1]
                pointer = out_mem[example, pointer]
            out_mem[example, out_mem[example, 2]] = output_value
        return init_mem, out_mem