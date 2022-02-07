import numpy as np
from typing import Optional, Union, Tuple
from functools import reduce
import math

def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k (ap@K)
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    precision_at_k = lambda n: len(np.intersect1d(predicted[:n], actual)) / n
    relevant = lambda x: int(predicted[x] in actual)

    result_apk = sum(map(lambda x: precision_at_k(x+1) * relevant(x), range(k)))
    return result_apk / k

def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k (map@K)
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    return sum(lambda x, y: apk(x, y, k), zip(actual, predicted)) / len(actual)

def dcgk(scores: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the discounted cumulative gain at K (DCG@K)
    Args:
        actual: a list of elements that are to be predicted
        predicted: a list of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The the discounted cumulative gain at K over the input lists
    """

    indices = np.arange(1, k+1)
    return np.sum(scores[:k] / np.log2(indices + 1))

def rrk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    for i, obj in enumerate(predicted[:k]):
        if obj in actual:
            return 1 / (i + 1)

def mrr(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the Mean reciprocal rank (MRR) at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The Mean reciprocal rank (MRR) at k over the input lists
    """
    result = 0
    for act, pred in zip(actual, predicted):
        result += rrk(act, pred, k)
    return result / len(actual)


    


