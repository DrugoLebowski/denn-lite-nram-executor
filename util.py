import numpy as np

def to_one_hot(val: int, size: int) -> np.array:
    b = np.zeros((size), dtype=np.float32)
    b[val] = 1.0
    return b