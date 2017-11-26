import numpy as np

def to_one_hot(a: np.array, size: int) -> np.array:
    b = np.zeros((len(a), size + 1), dtype=np.float32)
    b[np.arange(len(a)), a] = 1.0
    return b