import numpy as np


def idf80_saturation(vector_text):
    v = vector_text.toarray()[0]
    v = v[v != 0]
    if len(v) == 0:
        return 1
    return np.percentile(v, 0.8) / np.median(v)