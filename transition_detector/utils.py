from typing import List, Tuple

import numpy as np


def remove_short_ones(seq: np.ndarray, minimum_zeros_length: int) -> np.ndarray:
    """
    If group of ones is interrupted by a group of zeros with length < `minimal`, this group is filled with ones.

    :param seq: input sequence
    :param minimum_zeros_length: minimal possible length of zeros group
    :return: processed sequence
    """
    if minimum_zeros_length % 2 == 0:
        seq = seq[1:]

    kernel = np.ones(minimum_zeros_length)
    seq = (np.convolve(seq, kernel, mode='same') == 0).astype(int)
    seq = (np.convolve(seq, kernel, mode='same') == 0).astype(int)

    if minimum_zeros_length % 2 == 0:
        seq = np.insert(seq, len(seq), seq[-1])
    return seq


def split_bins(conditional_array: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns list of start and end indices of groups of ones in `conditional array`

    Example:
    conditional_array: [0, 0, 1, 1, 1, 0, 1, 0, 0]
    split_bins(conditional_array): [(2, 5), (6, 7)]

    :param conditional_array: array with values 0 or 1
    :return: list of start and end indices of groups of ones in `conditional array`
    """
    ids = np.argwhere(conditional_array).flatten()
    if len(ids) == 0:
        return []

    d = np.diff(ids)
    gap_ids = np.argwhere(d > 1).flatten()
    starts = [ids[0]]
    ends = []
    for idx in gap_ids:
        ends.append(ids[idx] + 1)
        starts.append(ids[idx + 1])
    ends.append(ids[-1] + 1)
    clips = list(zip(starts, ends))
    return clips


def value_in_interval(value: int, intervals: List[Tuple[int, int]]) -> bool:
    """
    Check if `value` is in one of `intervals`.

    :param value: just value to check if it is in interval
    :param intervals: list of intervals (tuples), 1st element in tuple is start and 2nd is the end of interval
    :return: True if value is in one of intervals else False
    """
    for start, end in intervals:
        if start <= value < end:
            return True
    return False
