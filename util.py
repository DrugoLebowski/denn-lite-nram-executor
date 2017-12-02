# Standard
import os
import shutil

# Vendor
import matplotlib.pyplot as plt
import numpy as np

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


def exists_or_create(path: str, flush: bool = False) -> None:
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


def create_test_dir(context) -> str:
    base_task_path = "%s%s" % (App.get("images_path"),
                     context.task.__str__())
    exists_or_create(base_task_path)

    config_filename_without_extension = os.path.splitext(os.path.basename(context.path_config_file))[0]
    path_test = "%s/%s" % (base_task_path, config_filename_without_extension)
    exists_or_create(path_test, True)

    return path_test


def create_sample_dir(test_path: str, sample: int) -> str:
    path = "%s/%s" % (test_path, sample)
    exists_or_create(path)
    return path


def print_memories(context, M: np.array, desired_mem: np.array, path: str) -> bool:
    """ Print the memories of the samples """
    int_M = M.argmax(axis=2)
    one_hot_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.int)

    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            if desired_mem[s, c] == int_M[s, c]:
                one_hot_mem[s, c] = 1

    plt.imshow(one_hot_mem, cmap="gray")
    plt.savefig("%s/memories.grey.png" % path)

    differences_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.float32)
    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            differences_mem[s, c] = desired_mem[s, c].argmax() * M[s, c, desired_mem[s, c]].max() / desired_mem[s, c]

    plt.imshow(differences_mem, cmap="Blues")
    plt.savefig("%s/memories.blues.png" % path)

    return True