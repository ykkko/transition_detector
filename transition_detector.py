from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils import remove_short_ones, split_bins
from constants import (
    DISABLE_TQDM, RESIZED_FRAME_HEIGHT, RESIZED_FRAME_WIDTH, CROP_ROWS, CROP_COLUMNS,
    CROP_DOTS_Y0, CROP_DOTS_X0, CROP_DOTS_Y1, CROP_DOTS_X1,
    THR_DIFFERENCE_FRAME_FADE, THR_SHARE_OF_CHANGED_CROPS_CUT, THR_DIFFERENCE_CROP_CUT,
    MINIMUM_LENGTH_OF_SCENE, MINIMUM_GAP_BETWEEN_FADES, MINIMUM_LENGTH_OF_FADE, MINIMUM_BRIGHTNESS_OF_CROP
)


class TransitionDetector:

    def _make_frame_crops(self, frame: np.ndarray) -> np.ndarray:
        """
        Split frame to sequence by `self.lines`, `self.columns` and `self.downscale` parameters.

        Example:
            lines = 2, columns = 2, downscale = 2
            input: [[1 2 3 4]
                    [2 2 3 4]
                    [3 3 3 4]
                    [4 4 4 4]]
            output: [[[[1 2]
                       [2 2]]
                      [[3 4]
                       [3 4]]]
                     [[[3 3]
                       [4 4]]
                      [[3 4]
                       [4 4]]]]

        :param frame: image array from cv2.VideoCapture in BGR format
        :return: array with shape (self.lines, self.columns, crop_h, crop_w)
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (RESIZED_FRAME_WIDTH, RESIZED_FRAME_HEIGHT))

        crops = []
        for r in range(CROP_ROWS):
            row_crops = []
            for c in range(CROP_COLUMNS):
                row_crops.append(frame[CROP_DOTS_Y0[r]: CROP_DOTS_Y1[r], CROP_DOTS_X0[c]: CROP_DOTS_X1[c]])
            crops.append(row_crops)
        crops = np.array(crops)
        return crops

    def get_frame_crops_values(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculates mean value of each frame's crop.

        :param frame: image array from cv2.VideoCapture in BGR format
        :return: np.ndarray with crop's mean values
        """
        crops = self._make_frame_crops(frame)
        return np.mean(crops, axis=(2, 3))

    def _video2crops(self, video_path: str, video_slice: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Reads video's frames, makes sequence and calculates mean value of each crop.

        :param video_path: path to video file to process
        :param video_slice: tuple with start and end IDs for processing a specific piece of video
        :return: sequence with shape (frame_num, CROP_ROWS, CROP_COLUMNS),
                 each element of sequence is np.ndarray with crop's mean values
        """
        cap = cv2.VideoCapture(video_path)
        frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_crops_mean_values = []
        start_frame, end_frame = video_slice or (0, frames_num)

        if start_frame >= end_frame:
            raise AttributeError(f'1st element in `video_slice` must be less than 2nd element. '
                                 f'Got: {video_slice}.')
        if start_frame >= frames_num:
            raise AttributeError(f'1st element in `video_slice` must be less than the frames number in the video. '
                                 f'Video frame number: {frames_num}, got: {video_slice[0]}.')

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        for _ in tqdm(range(start_frame, end_frame), f'Read frames: {start_frame}-{end_frame}', disable=DISABLE_TQDM):
            ret, frame = cap.read()
            if not ret:
                break
            frames_crops_mean_values.append(self.get_frame_crops_values(frame))
        cap.release()
        return np.array(frames_crops_mean_values)

    def _get_fade(self, data: np.ndarray) -> int:
        """
        Detects if sequence of frames is fading. Length of sequence = 5.
        Compares all neighbors elements in `data`: 0th and 1th, 1th and 2nd, 2nd and 3rd ...

        Each pair is compared according to this rule:
            A is 1th element of pair, B is 2nd element
            if B - A > max(A, B) * thr
                sequence [A, B] is fadein
            if A - B > max(A, B) * thr:
                sequence [A, B] is fadeout
        If all pairs is fadein function returns int 2 for `data` sequence.
        If all pairs is fadeout function returns int 1 for `data` sequence.
        Otherwise function returns int 0 for `data` sequence.

        :param data: sequence, each element is mean value of frame
        :return: 2 - fadein, 1 - fadeout, 0 - nothing
        """
        counter_plus = sum([data[i + 1] - data[i] > max(data[i], data[i + 1]) * THR_DIFFERENCE_FRAME_FADE
                            for i in range(len(data) - 1)])
        counter_minus = sum([data[i] - data[i + 1] > max(data[i], data[i + 1]) * THR_DIFFERENCE_FRAME_FADE
                             for i in range(len(data) - 1)])

        # 4 is the number of all pairs to compare,
        # so if all pairs fades by itself it means that the whole sequence fades
        if counter_plus == 4:
            return 2
        elif counter_minus == 4:
            return 1
        else:
            return 0

    def _frame_changes(self, a: np.ndarray, b: np.ndarray, only_bright: bool = True) -> int:
        """
        Compares sequence of neighbors frames.

        Frames compares by crops. Each pairs of crops with the same indices
        is compared according to this rule:
            A is a crop of frame `a`, B is a crop of frame `b`
            if abs(A - B) > min(A, B) * thr
                crops pair is changed, that means that crop B from new scene
            if count of changed crops is more then threshold, then frame `b` is from new scene and cut is on frame `b`

        :param a: array with crops of 1st frame
        :param b: array with crops of 2nd frame
        :param only_bright: if True removes black sequence to compare only bright crops
        :return: 1 if cut is detected, 0 - otherwise
        """
        if only_bright:
            brighter_frame = np.max([a, b], axis=0)
            bright_crops = np.greater(brighter_frame, MINIMUM_BRIGHTNESS_OF_CROP)
            a = a[bright_crops]
            b = b[bright_crops]
            min_changed_crops_num = int(bright_crops.sum() * THR_SHARE_OF_CHANGED_CROPS_CUT)
        else:
            min_changed_crops_num = int(len(a) * THR_SHARE_OF_CHANGED_CROPS_CUT)
        if min_changed_crops_num == 0:
            return 0

        diff = abs(a - b)
        pair_min = np.min([a, b], axis=0) * THR_DIFFERENCE_CROP_CUT
        changed_crops_num = np.greater(diff, pair_min).sum()
        return int(changed_crops_num >= min_changed_crops_num)

    def _process_fade(self, seq: np.ndarray) -> np.ndarray:
        """
        Fills short groups of zeros with ones, and fills short groups of ones with zeros.

        :param seq: input sequence
        :return: processed sequence
        """
        seq = remove_short_ones(seq, minimal=MINIMUM_GAP_BETWEEN_FADES)
        seq = 1 - remove_short_ones(1 - seq, minimal=MINIMUM_LENGTH_OF_FADE)
        return seq

    def _find_transitions(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects where in a sequence of "frames" the crop, fadein, or fadeout.

        :param sequence: sequence of frames mean values
        :return: elements of tuple are lists (len == video frames count) named `cuts`, `fadein`, `fadeout`
                 where 1 in element of list with index `i` means that transition is in frame with same index `i`
        """
        cuts, fades = [0], []
        for i in range(len(sequence) - 1):
            frame_changed = self._frame_changes(sequence[i], sequence[i + 1])
            cuts.append(frame_changed)
            if i >= 4:
                seq_mean = np.mean(sequence[i - 4: i + 1], axis=(1, 2))
                fades.append(self._get_fade(seq_mean))

        cuts = remove_short_ones(np.array(cuts), minimal=MINIMUM_LENGTH_OF_SCENE)
        cuts = cuts[1:] - cuts[:-1]
        cuts = np.insert(cuts, 0, 0)
        cuts[cuts == -1] = 0

        # Add the first 4 elements to the `fades` sequence (`fades` are calculated from the 4th frame)
        fades = [fades[0]] * 4 + fades
        fadein = np.array([1 if x == 2 else 0 for x in fades])
        fadein = self._process_fade(fadein)
        fadeout = np.array([0 if x == 2 else x for x in fades])
        fadeout = self._process_fade(fadeout)
        return cuts, fadein, fadeout

    def get_transitions(self, crops_mean_values: np.ndarray, video_slice: Optional[Tuple[int, int]] = None) \
            -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Calculates where the transitions are.

        :param crops_mean_values: sequence with shape (frame_num, CROP_ROWS, CROP_COLUMNS),
                                  each element of sequence is np.ndarray with crop's mean values
        :param video_slice: tuple with start and end IDs for processing a specific piece of video
        :return: tuple,
            - `cut_frames`: np.ndarray with frame indexes where cut is
            - `fadein_frames`, `fadeout_frames`: list of tuples, 1st element in the tuple is the start of fading,
              2nd is the end of fading
        """
        cuts_sequence, fadein_sequence, fadeout_sequence = self._find_transitions(np.array(crops_mean_values))

        start_frame = video_slice[0] if video_slice is not None else 0
        if start_frame > 0:
            start_frame = video_slice[0]
            cuts_sequence = np.insert(cuts_sequence, 0, [0] * start_frame)
            fadein_sequence = np.insert(fadein_sequence, 0, [0] * start_frame)
            fadeout_sequence = np.insert(fadeout_sequence, 0, [0] * start_frame)

        cut_frames = np.argwhere(cuts_sequence == 1).flatten()
        fadein_frames = split_bins(fadein_sequence)
        fadeout_frames = split_bins(fadeout_sequence)
        return cut_frames, fadein_frames, fadeout_frames

    def run(self, video_path: str, video_slice: Optional[Tuple[int, int]] = None) -> \
            Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray]:
        """
        Runs the transition detection pipeline.

        :param video_path: path to video file to process
        :param video_slice: tuple with start and end IDs for processing a specific piece of video
        :return: tuple,
            - `cut_frames`: np.ndarray with frame indexes where cut is
            - `fadein_frames`, `fadeout_frames`: list of tuples, 1st element in the tuple is the start of fading,
              2nd is the end of fading
            - `frames_crops_mean_values`: sequence with shape (frame_num, CROP_ROWS, CROP_COLUMNS),
              each element of sequence is np.ndarray with crop's mean values
        """
        frames_crops_mean_values = self._video2crops(video_path, video_slice)
        cut_frames, fadein_frames, fadeout_frames = self.get_transitions(frames_crops_mean_values, video_slice)
        return cut_frames, fadein_frames, fadeout_frames, frames_crops_mean_values
