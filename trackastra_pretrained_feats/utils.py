import numpy as np


def percentile_norm(b):
    for i, im in enumerate(b):
        p1, p99 = np.percentile(im, (1, 99.8))
        b[i] = (im - p1) / (p99 - p1)
        b[i] = np.clip(b[i], 0, 1)
    return b
