# Standard
import os

# Vendor
import numpy as np
import matplotlib.pyplot as plt

# Project
from App import App


def to_one_hot(val: int, size: int) -> np.array:
    b = np.zeros((size), dtype=np.float32)
    b[val] = 1.0
    return b


def fuzzyfy_mem(M: np.array) -> np.array:
    """ Make the fuzzy version of a list of integer memories """
    fuzzyfied_mems = []
    for s in M:
        sample_fuzzyfied_mem = []
        for n in s:
            sample_fuzzyfied_mem.append(to_one_hot(n, M.shape[1]))
        fuzzyfied_mems.append(np.stack(sample_fuzzyfied_mem, axis=0))

    return np.stack(fuzzyfied_mems, axis=0)

def exists_or_create(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path, 0o755)

def print_memories(context, M: np.array, desired_mem: np.array) -> bool:
    """ Print the memories of the samples """
    int_M = M.argmax(axis=2)
    one_hot_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.int)

    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            if desired_mem[s, c] == int_M[s, c]:
                one_hot_mem[s, c] = 1

    plt.imshow(one_hot_mem, cmap="gray")
    plt.show()

    differences_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.float32)
    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            differences_mem[s, c] = desired_mem[s, c] - (M[s, c].argmax() * M[s, c].max())

    plt.imshow(differences_mem, cmap="Blues")
    plt.show()

    return True