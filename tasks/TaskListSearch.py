# Vendor
import numpy as np

# Project
from tasks.Task import Task

class TaskListSearch(Task):
    """ [ListSearch]
    Given a pointer to the head of a linked list and a value `v` to find return a pointer
    to the first node on the list with the value `v`. The list is placed in memory in the same way
    as in the task ListK. We fill empty memory with “trash” values to prevent the network from
    “cheating” and just iterating over the whole memory.
    """

    def create(self) -> (np.ndarray, np.ndarray, np.ndarray):
        list_size = int((self.max_int - 2) / 2)
        list_elements = np.random.randint(0, self.max_int, size=(self.batch_size, list_size))
        lists_elements_permutations = np.stack([np.random.permutation(list_size) for _ in range(self.batch_size)], axis=0)
        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)

        # Create for each example the list
        for example in range(self.batch_size):
            for j, permidx in enumerate(lists_elements_permutations[example]):
                next_element_pointer = np.where(lists_elements_permutations[example] == permidx + 1)[0]
                if permidx == 0: # If the node is the first than set the pointer in the first memory position
                    init_mem[example, 0] = 2 + 2 * j

                init_mem[example, 2 + (2 * j)] = \
                    -1.0 if len(next_element_pointer) == 0 else 2 + (2 * next_element_pointer[0]) # Set the pointer to the next list node
                init_mem[example, 2 + (2 * j) + 1] = list_elements[example, j] # Set the value of the list node
        init_mem[:, 1] = list_elements[:, 0] # Set the elements to search in the list
        if self.max_int % 2 != 0:
            init_mem[:, -1] = -1

        out_mem = init_mem.copy()
        for example in range(self.batch_size):
            found = False
            pointer = out_mem[example, 0]
            while not found and pointer != -1:
                if out_mem[example, pointer + 1] == out_mem[example, 1]:
                    out_mem[example, 0] = pointer
                    found = True
                else:
                    pointer = out_mem[example, pointer]

        cost_mask = np.zeros((self.batch_size, self.max_int), dtype=np.int8)
        cost_mask[:, 0] = 1

        return init_mem, out_mem, cost_mask
