import numpy as np
from typing import Optional, Union, Tuple
from functools import reduce

def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """

    return np.sqrt(np.sum((x - y) ** 2))


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1 / (1 + euclidean_distance(x, y))


def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    f = lambda z: np.sum((z - z.mean()) ** 2)

    return np.sum((x - x.mean()) * (y - y.mean())) / np.sqrt(f(x) * f(y))


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

    # result_apk = 0
    # for i in range(1, k+1):
    #     result_apk += precision_at_k(i) * relevant(i-1)
    
    result_apk = reduce(lambda a, x: a + x, map(lambda x: precision_at_k(x+1) * relevant(x), range(k)))
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
