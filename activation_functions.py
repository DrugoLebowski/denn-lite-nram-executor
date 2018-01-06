# Vendor
import numpy as np


def relu(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.abs(a))

def softmax(a: np.ndarray) -> np.ndarray:
    return np.exp(a) / np.sum(np.exp(a), axis=1)[..., None]

def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))