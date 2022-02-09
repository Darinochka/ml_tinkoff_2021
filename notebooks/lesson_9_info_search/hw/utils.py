import numpy as np
from typing import Optional, Union, Tuple
from functools import lru_cache, reduce
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
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    len_u = len(actual)
    return reduce(lambda a, x: a + x, map(lambda x: apk(actual[x], predicted[x], k), range(len_u))) / len_u

def dcgk(scores: np.array, k: int = 10) -> float:
    """
    Compute the discounted cumulative gain at K (DCG@K)
    Args:
        scores: a list of scores of the elements that are to be predicted
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

def prel(scores: np.array, i: int):
    if scores[i] > 0:
        return scores[i]
    return 0

def plook(scores: np.array, i: int) -> float:
    if i == 0:
        return 1
    else:
        return plook(scores, i-1) * (1 - prel(scores, i-1))
    
def pfound(scores: np.array, k: int = 10) -> float:
    """
    Compute the PFound at K (PFound@K)
    Args:
        scores: a list of scores of the elements that are to be predicted
        k: the maximum number of predicted elements
    Returns:
        The the discounted cumulative gain at K over the input lists
    """
    return sum(map(lambda i: plook(scores, i) * prel(scores, i), range(k)))



