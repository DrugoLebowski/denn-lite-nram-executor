# Vendor
import numpy as np

# Project
from tasks.Task import Task


class TaskWalkBST(Task):
    """ [WalkBST]
    Given a pointer to the root of a Binary Search Tree, and a path to be traversed,
    return the element at the end of the path. The BST nodes are represented as triples (v, l,
    r), where v is the value, and l, r are pointers to the left/right child. The triples are placed
    randomly in the memory. Input is given as root, out, d1, d2, ..., dk, NULL, ..., where root
    points to the root node and out is a slot for the output. The sequence d1...dk, di ∈ {0, 1}
    represents the path to be traversed: di = 0 means that the network should go to the left
    child, di = 1 represents going to the right child.
    """

    def create(self) -> (np.array, np.array):
        def get_element_index(permutation: np.ndarray, idx: int) -> int:
            """ Get the index where the permutation have the value idx. """
            return np.where(permutation == idx)[0][0]

        def insert_in_bst(bst: np.ndarray, pointer: int, element: int, element_pointer_in_memory: int) -> np.ndarray:
            """ Insert an element in the BST. """
            if bst[pointer] > element:
                if bst[pointer + 1] == 0:
                    bst[pointer + 1] = element_pointer_in_memory
                    return bst
                else:
                    return insert_in_bst(bst, int(bst[pointer + 1]), element, element_pointer_in_memory)
            else:
                if bst[pointer + 2] == 0:
                    bst[pointer + 2] = element_pointer_in_memory
                    return bst
                else:
                    return insert_in_bst(bst, int(bst[pointer + 2]), element, element_pointer_in_memory)

        def walk_bst(bst: np.ndarray, walk: np.ndarray, previous_pointer: int, pointer: int):
            """ Walk, set a walk sequence, the bst and return the last element encountered. """
            if pointer == 0 and previous_pointer != -1:
                return bst[previous_pointer]
            elif len(walk) == 0:
                return bst[pointer]
            else:
                if walk[0] == 0:
                    return walk_bst(bst, walk[1:], pointer, int(bst[pointer + 1]))
                else:
                    return walk_bst(bst, walk[1:], pointer, int(bst[pointer + 2]))


        remaining_size = self.max_int - 3
        num_elements = int(remaining_size / 4)
        offset = 2
        init_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)
        out_mem = np.zeros((self.batch_size, self.max_int), dtype=np.int32)

        # Create and initialize elements of bsts
        list_elements = np.random.randint(1, self.max_int, size=(self.batch_size, num_elements), dtype=np.int32)

        # Create and initialize the walk in bst
        walks_bst = np.random.randint(0, 2, size=(self.batch_size, num_elements), dtype=np.int32)

        # Create the elements order in memory through permutation matrices
        orders_in_memory = np.stack([np.random.permutation(num_elements)
                                     for _ in range(self.batch_size)], axis=0)

        for e in range(self.batch_size):
            # Create a temporary vector that contains a BST for an example
            example_bst = np.zeros(num_elements * 3)
            root_pointer = -1
            for i in range(num_elements):
                pointer = get_element_index(orders_in_memory[e], i)
                example_bst[pointer * 3] = list_elements[e, i]
                if i is 0:
                    root_pointer = pointer * 3

            # Initialize BST
            for i in range(num_elements):
                example_bst = insert_in_bst(example_bst, root_pointer, list_elements[e, i], \
                                            get_element_index(orders_in_memory[e], i) * 3)

            # Fill the memories
            init_mem[e, 0] = offset + num_elements + root_pointer
            init_mem[e, offset:(offset + num_elements)] = walks_bst[e]

            # Walk the bst before the normalization
            value_found = walk_bst(example_bst, walks_bst[e], -1, root_pointer)

            # Normalize bst pointers for the memory and then save it
            for idx in range(example_bst.size):
                if idx % 3 != 0 and example_bst[idx] != 0:
                    example_bst[idx] += offset + num_elements

            init_mem[e, (offset + num_elements):(offset + (4 * num_elements))] = example_bst

            out_mem[e] = init_mem[e]
            out_mem[e, 1] = value_found

        return init_mem, out_mem