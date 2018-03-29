# Vendor
import numpy as np


def relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0)

def softmax(a: np.ndarray) -> np.ndarray:
    return np.exp(a - np.max(a)) / np.exp(a - np.max(a)).sum(1)[..., None]

def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-np.array(a, dtype=np.float64)))