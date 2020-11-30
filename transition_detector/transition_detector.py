from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .utils import remove_short_ones, split_bins, get_changed_crops
from .visualize import visualize


class TransitionDetector:

    def __init__(self, resize_frame: Tuple[int, int] = (360, 640), crops_grid: Tuple[int, int] = (12, 12),
                 crop_downscale: int = 12, minimum_length_of_scene: int = 7, minimum_gap_between_fades: int = 7,
                 minimum_length_of_fade: int = 5, minimum_brightness_of_crop: float = 5,
                 thr_share_of_changed_crops_cut: float = 0.8, thr_difference_crop_cut: float = 0.08,
                 thr_difference_frame_fade: float = 0.05, disable_tqdm: bool = False):
        """
        Initiates instance of TransitionDetector.

        :param resize_frame: tuple, 1st element - new height, 2nd - new width
        :param crops_grid: 1st element: number of lines in crop grid, 2nd element: number of columns in crop grid
        :param crop_downscale: crop_height = frame_height // crop_downscale, same for width
        :param minimum_length_of_scene: minimum distance (in frames) between two cuts
        :param minimum_gap_between_fades: minimum distance (in frames) between two fades
        :param minimum_length_of_fade: minimum duration (in frames) of fadein or fadeout
        :param minimum_brightness_of_crop: minimum brightness of crop to remove black sequence from crops
                                           comparing, [0, 255]
        :param thr_share_of_changed_crops_cut: share of changed crops to say that a cut transition is detected, (0, 1]
        :param thr_difference_crop_cut: the difference between the two frames is compared with the frame values
                                        multiplied by this parameter to find the cut
        :param thr_difference_frame_fade: the difference between the two frames is compared with the frame values
                                          multiplied by this parameter to find the fade
        :param disable_tqdm: if True, tqdm will not show progress
        """
        self.resized_frame_height, self.resized_frame_width = resize_frame
        self.crop_rows, self.crop_columns = crops_grid
        self.crop_downscale = crop_downscale
        self.minimum_length_of_scene = minimum_length_of_scene
        self.minimum_gap_between_fades = minimum_gap_between_fades
        self.minimum_length_of_fade = minimum_length_of_fade
        self.minimum_brightness_of_crop = minimum_brightness_of_crop
        self.thr_share_of_changed_crops_cut = thr_share_of_changed_crops_cut
        self.thr_difference_crop_cut = thr_difference_crop_cut
        self.thr_difference_frame_fade = thr_difference_frame_fade
        self.disable_tqdm = disable_tqdm

        _crop_height = self.resized_frame_height // self.crop_downscale
        _crop_width = self.resized_frame_width // self.crop_downscale
        _h_pad = self.resized_frame_height - _crop_height
        _w_pad = self.resized_frame_width - _crop_width
        self.crop_dots_y0 = [int(_h_pad / (self.crop_rows - 1) * i) for i in range(self.crop_rows)]
        self.crop_dots_x0 = [int(_w_pad / (self.crop_columns - 1) * i) for i in range(self.crop_columns)]
        self.crop_dots_y1 = [y0 + _crop_height for y0 in self.crop_dots_y0]
        self.crop_dots_x1 = [x0 + _crop_width for x0 in self.crop_dots_x0]

        if self.resized_frame_height < self.crop_rows * _crop_height:
            raise ValueError(f'Height of frame ({self.resized_frame_height}) must be >= then '
                             f'`lines` * `crop_h` ({self.crop_rows * _crop_height})')
        if self.resized_frame_width < self.crop_columns * _crop_width:
            raise ValueError(f'Width of frame ({self.resized_frame_width}) must be >= then '
                             f'`columns` * `crop_w` ({self.crop_rows * _crop_width})')

    def _make_frame_crops(self, frame: np.ndarray) -> np.ndarray:
        """
        Split frame to sequence by `self.crop_rows`, `self.crop_columns` and `self.downscale` parameters.

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
        frame = cv2.resize(frame, (self.resized_frame_width, self.resized_frame_height))
        crops = []
        for r in range(self.crop_rows):
            row_crops = []
            for c in range(self.crop_columns):
                row_crops.append(frame[self.crop_dots_y0[r]: self.crop_dots_y1[r],
                                       self.crop_dots_x0[c]: self.crop_dots_x1[c]])
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
        :return: sequence with shape (frame_num, self.crop_rows, self.crop_columns),
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
        for _ in tqdm(range(start_frame, end_frame), f'Read frames: {start_frame}-{end_frame}',
                      disable=self.disable_tqdm):
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
        counter_plus = sum([data[i + 1] - data[i] > max(data[i], data[i + 1]) * self.thr_difference_frame_fade
                            for i in range(len(data) - 1)])
        counter_minus = sum([data[i] - data[i + 1] > max(data[i], data[i + 1]) * self.thr_difference_frame_fade
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
            bright_crops = np.greater(brighter_frame, self.minimum_brightness_of_crop)
            a = a[bright_crops]
            b = b[bright_crops]
            min_changed_crops_num = int(bright_crops.sum() * self.thr_share_of_changed_crops_cut)
        else:
            min_changed_crops_num = int(len(a) * self.thr_share_of_changed_crops_cut)
        if min_changed_crops_num == 0:
            return 0

        changed_crops_num = get_changed_crops(a, b, self.thr_difference_crop_cut).sum()
        return int(changed_crops_num >= min_changed_crops_num)

    def _process_fade(self, seq: np.ndarray) -> np.ndarray:
        """
        Fills short groups of zeros with ones, and fills short groups of ones with zeros.

        :param seq: input sequence
        :return: processed sequence
        """
        seq = remove_short_ones(seq, minimum_zeros_length=self.minimum_gap_between_fades)
        seq = 1 - remove_short_ones(1 - seq, minimum_zeros_length=self.minimum_length_of_fade)
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

        cuts = remove_short_ones(np.array(cuts), minimum_zeros_length=self.minimum_length_of_scene)
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

        :param crops_mean_values: sequence with shape (frame_num, self.crop_rows, self.crop_columns),
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
            - `frames_crops_mean_values`: sequence with shape (frame_num, self.crop_rows, self.crop_columns),
              each element of sequence is np.ndarray with crop's mean values
        """
        frames_crops_mean_values = self._video2crops(video_path, video_slice)
        cut_frames, fadein_frames, fadeout_frames = self.get_transitions(frames_crops_mean_values, video_slice)
        return cut_frames, fadein_frames, fadeout_frames, frames_crops_mean_values

    def visualize(self, video_path: str, cut_frames: np.ndarray, fadein_frames: List[Tuple[int, int]],
                  fadeout_frames: List[Tuple[int, int]], frames_crops_mean_values: Optional[np.ndarray] = None,
                  video_slice: Optional[Tuple[int, int]] = None, output_video_path: Optional[str] = None,
                  imshow: bool = False):
        """
        Interface for `visualize` from visualize.py. Shows frames or saves video with information
        about transitions on frames. If frames_crops_mean_values is not None, frames will be contained
        detailed information about all crops of frame.

        :param video_path: path to video file
        :param cut_frames: np.ndarray with frame indexes where cut is
        :param fadein_frames: list of tuples, 1st element in the tuple is the start of fading, 2nd is the end of fading
        :param fadeout_frames: same as `fadein_frames`
        :param frames_crops_mean_values: sequence with shape (frame_num, CROP_ROWS, CROP_COLUMNS), each element of sequence
                                         is np.ndarray with crop's mean values. It's used for visualize values of each crop.
                                         If None, then frames will not be contained detailed information
        :param video_slice: tuple with start and end IDs for visualize a specific piece of video
        :param output_video_path: path to video with information on frames, if None, then video will not be created
        :param imshow: call cv2.imshow if True
        """
        visualize(video_path=video_path, cut_frames=cut_frames, fadein_frames=fadein_frames,
                  fadeout_frames=fadeout_frames, frames_crops_mean_values=frames_crops_mean_values,
                  crops_grid=(self.crop_rows, self.crop_columns), crop_downscale=self.crop_downscale,
                  thr_difference_crop_cut=self.thr_difference_crop_cut,
                  video_slice=video_slice, output_video_path=output_video_path, imshow=imshow,
                  disable_tqdm=self.disable_tqdm)
