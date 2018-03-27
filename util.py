# Standard
import os
import shutil

# Vendor
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_one_hot(val, shape: int = None) -> np.ndarray:

    def to_one_hot_array(val: np.ndarray) -> np.ndarray:
        b = np.zeros((val.shape[0], val.shape[0]), dtype=np.float32)
        b[np.arange(val.shape[0]), val] = 1
        return b

    def to_one_hot_number(val: int, shape: int) -> np.ndarray:
        b = np.zeros((shape), dtype=np.float32)
        b[val] = 1
        return b

    return to_one_hot_array(val) if shape is None else to_one_hot_number(val, shape)


def encode(M: np.array) -> np.ndarray:
    """ Make the fuzzy version of a list of integer memories """
    return np.stack([to_one_hot(s) for s in M], axis=0)


def create_dir(path: str, flush: bool = False) -> None:
    if not os.path.exists(path):
        os.mkdir(path, 0o755)

    if flush:
        for fd in os.listdir(path):
            file_path = os.path.join(path, fd)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def print_memories(M: np.ndarray, desired_mem: np.ndarray, cost_mask: np.ndarray, path: str, test_idx: int) -> bool:
    """ Print the memories of the samples """
    int_M = M.argmax(axis=2)
    c = 0 # See paper Pag. 7, Sub section 4.2 Tasks
    m = np.sum(cost_mask) # See paper Pag. 7, Sub section 4.2 Tasks
    one_hot_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.int)

    for sample in range(desired_mem.shape[0]):
        for col in range(desired_mem.shape[1]):
            if desired_mem[sample, col] == int_M[sample, col]:
                one_hot_mem[sample, col] = 1
                if cost_mask[0, col] == 1:
                    c += 1
    perc_correct = c / m
    with open("%s/tests-results.csv" % os.path.abspath(os.path.join(path, os.pardir)), "a+") as f:
        f.write("%d,%f\n" % (np.sum(cost_mask[0]), 1 - perc_correct))

    fig = plt.figure()
    fig.suptitle('Correct: %f, Error: %f' % (perc_correct, (1 - perc_correct)), fontsize=14)

    plt.imshow(one_hot_mem, cmap="gray", vmin=0.0, vmax=1.0)
    plt.savefig("%s/%d.memories.grey.png" % (path, test_idx))

    differences_mem = np.zeros(
        (desired_mem.shape[0], desired_mem.shape[1]), dtype=np.float32)
    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            differences_mem[s, c] = M[s, c, desired_mem[s, c]]

    plt.imshow(differences_mem, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.savefig("%s/%d.memories.blues.png" % (path, test_idx))

    return True