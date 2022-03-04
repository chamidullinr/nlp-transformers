import numpy as np


def softmax(x, axis=-1, temperature=1):
    x = x / temperature  # temperature scaling for classifier calibration
    e = np.exp(x - x.max())  # x.max() makes function exp more stable
    return e / e.sum(axis, keepdims=True)


def normalize(x, p=2, axis=1):
    denom = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    denom = np.clip(denom, a_min=1e-9, a_max=np.inf)
    return x / denom
