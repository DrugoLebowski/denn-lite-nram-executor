# Standard
import os
import shutil

# Vendor
import matplotlib.pyplot as plt
import numpy as np

# Project
from App import App


def to_one_hot(val, shape: int = None) -> np.ndarray:

    def to_one_hot_array(val: np.ndarray) -> np.ndarray:
        b = np.zeros((val.shape[0], val.shape[0]), dtype=np.float64)
        b[np.arange(val.shape[0]), val] = 1
        return b

    def to_one_hot_number(val: int, shape: int) -> np.ndarray:
        b = np.zeros((shape), dtype=np.float64)
        b[val] = 1
        return b

    return to_one_hot_array(val) if shape is None else to_one_hot_number(val, shape)


def encode(M: np.array) -> np.ndarray:
    """ Make the fuzzy version of a list of integer memories """
    return np.stack([to_one_hot(s) for s in M], axis=0)


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


def print_memories(context, M: np.ndarray, desired_mem: np.ndarray, cost_mask: np.ndarray, path: str) -> bool:
    """ Print the memories of the samples """
    int_M = M.argmax(axis=2)
    num_samples = M.shape[0]
    c = 0 # See paper Pag. 7, Sub section 4.2 Tasks
    m = np.sum(cost_mask) # See paper Pag. 7, Sub section 4.2 Tasks
    one_hot_mem = np.zeros((desired_mem.shape[0], desired_mem.shape[1]), dtype=np.int)

    for sample in range(desired_mem.shape[0]):
        for col in range(desired_mem.shape[1]):
            if desired_mem[sample, col] == int_M[sample, col]:
                one_hot_mem[sample, col] = 1
                if cost_mask[0, col] == 1:
                    c += 1
    error = c / m

    fig = plt.figure()
    fig.suptitle('Error: %f' % error, fontsize=14)

    ax = fig.add_subplot(111)
    ax.imshow(one_hot_mem, cmap="gray")
    fig.savefig("%s/memories.grey.png" % path)

    differences_mem = np.zeros(
        (desired_mem.shape[0], desired_mem.shape[1]), dtype=np.float64)
    for s in range(desired_mem.shape[0]):
        for c in range(desired_mem.shape[1]):
            differences_mem[s, c] = desired_mem[s, c]

    plt.imshow(differences_mem, cmap="Blues")
    plt.savefig("%s/memories.blues.png" % path)

    return True