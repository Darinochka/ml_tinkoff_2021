import numpy as np
from typing import Optional, Union, Tuple
from functools import reduce


def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    precision_at_k = lambda n: len(np.intersect1d(predicted[:n], actual)) / n
    relevant = lambda x: int(predicted[x] in actual)