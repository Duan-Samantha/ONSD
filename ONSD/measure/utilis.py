import numpy as np

def normalize(data, a, b):
    """
    :return:
    """
    y_max = np.max(data)
    y_min = np.min(data)
    k = (b - a) / (y_max - y_min)

    norY = a + k * (data - y_min)

    return norY

def find_roots(f, g, mean):
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    for i in range(len(idx)):
        if idx[i] <= mean and idx[i+1] >= mean:
            return idx[i], idx[i+1]
    # return