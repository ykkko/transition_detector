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


def get_changed_crops(prev_frame_crops_vals, curr_frame_crops_vals, thr_difference_crop_cut):
    """
    Compares corresponding crops between frames to find big changes.

    :param prev_frame_crops_vals: mean crop values of previous frame, shape=(crop_rows, crop_columns)
    :param curr_frame_crops_vals: mean crop values of current frame, shape=(crop_rows, crop_columns)
    :param thr_difference_crop_cut: the difference between the two frames is compared with the frame values
                                    multiplied by this parameter to find the cut
    :return:
    """
    diff = abs(prev_frame_crops_vals - curr_frame_crops_vals)
    pair_min = np.min([prev_frame_crops_vals, curr_frame_crops_vals], axis=0) * thr_difference_crop_cut
    changed_crops = np.greater(diff, pair_min)
    return changed_crops
