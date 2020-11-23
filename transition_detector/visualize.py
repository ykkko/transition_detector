from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .constants import DISABLE_TQDM, CROP_ROWS, CROP_COLUMNS, CROP_DOWNSCALE, THR_DIFFERENCE_CROP_CUT
from .utils import value_in_interval


def add_text(frame: np.ndarray, text: str, coord: Tuple[int, int], size: float,
             color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    Interface to call cv2.putText.

    :param frame: image array from cv2.VideoCapture in BGR format
    :param text: text content to put on frame
    :param coord: left bottom corner of text
    :param size: size of text
    :param color: color of text if BGR format
    :param thickness: thickness of text
    :return: frame with text on it
    """
    return cv2.putText(frame, text, coord, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, lineType=cv2.LINE_AA)


def add_detailed_info(frame: np.ndarray, prev_frame_crops_vals: np.ndarray,
                      curr_frame_crops_vals: np.ndarray, is_cut: bool,
                      crop_dots_x0: List[int], crop_dots_y0: List[int],
                      crop_dots_x1: List[int], crop_dots_y1: List[int]) -> np.ndarray:
    """
    Adds detailed information about all crops of frame.

    :param frame: image array from cv2.VideoCapture in BGR format
    :param prev_frame_crops_vals: mean crop values of previous frame, shape=(CROP_ROWS, CROP_COLUMNS)
    :param curr_frame_crops_vals: mean crop values of current frame, shape=(CROP_ROWS, CROP_COLUMNS)
    :param is_cut: True if current frame is first frame in a scene
    :param crop_dots_x0: left corners coordinates of crops
    :param crop_dots_y0: top corners coordinates of crops
    :param crop_dots_x1: rights corners coordinates of crops
    :param crop_dots_y1: bottom corners coordinates of crops
    :return: frame with detailed information on it
    """
    diff = abs(prev_frame_crops_vals - curr_frame_crops_vals)
    pair_min = np.min([prev_frame_crops_vals, curr_frame_crops_vals], axis=0) * THR_DIFFERENCE_CROP_CUT
    changed_crops_num = np.greater(diff, pair_min)
    for r in range(CROP_ROWS):
        for c in range(CROP_COLUMNS):
            x0, y0, x1, y1 = crop_dots_x0[c], crop_dots_y0[r], crop_dots_x1[c], crop_dots_y1[r]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 50, 170) if is_cut else (170, 50, 0), 1)
            crop_value = str(round(curr_frame_crops_vals[r][c]))
            color = (0, 0, 255) if changed_crops_num[r, c] else (255, 255, 255)
            frame = add_text(frame, crop_value, (x0, y1 - 5), 0.75, (0, 0, 0), 3)  # black stroke for text
            frame = add_text(frame, crop_value, (x0, y1 - 5), 0.75, color, 2)
    return frame


def visualize(video_path: str, cut_frames: np.ndarray, fadein_frames: List[Tuple[int, int]],
              fadeout_frames: List[Tuple[int, int]], frames_crops_mean_values: Optional[np.ndarray] = None,
              video_slice: Optional[Tuple[int, int]] = None,
              output_video_path: Optional[str] = None, imshow: bool = False):
    """
    Shows frames or saves video with information about transitions on frames. If frames_crops_mean_values
    is not None, frames will be contained detailed information about all crops of frame.

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
    if output_video_path is None and not imshow:
        raise AttributeError('If output_video_path is None and imshow is False, then this function will do nothing.')

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_video_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if frames_crops_mean_values is not None:
        crop_height = height // CROP_DOWNSCALE
        crop_width = width // CROP_DOWNSCALE
        _h_pad = height - crop_height
        _w_pad = width - crop_width
        crop_dots_y0 = [int(_h_pad / (CROP_ROWS - 1) * i) for i in range(CROP_ROWS)]
        crop_dots_x0 = [int(_w_pad / (CROP_COLUMNS - 1) * i) for i in range(CROP_COLUMNS)]
        crop_dots_y1 = [y0 + crop_height for y0 in crop_dots_y0]
        crop_dots_x1 = [x0 + crop_width for x0 in crop_dots_x0]

    start_frame, end_frame = video_slice or (0, frames_num)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    scene = 1
    prev_frame_crops_vals = None
    for i in tqdm(range(end_frame - start_frame), f'Read frames: {start_frame}-{end_frame}', disable=DISABLE_TQDM):
        ret, frame = cap.read()
        if ret:
            if frames_crops_mean_values is not None:
                if i == 0:
                    prev_frame_crops_vals = np.array(frames_crops_mean_values[i])
                curr_frame_crops_vals = np.array(frames_crops_mean_values[i])
                frame = add_detailed_info(frame, prev_frame_crops_vals, curr_frame_crops_vals, i in cut_frames,
                                          crop_dots_x0, crop_dots_y0, crop_dots_x1, crop_dots_y1)
                prev_frame_crops_vals = curr_frame_crops_vals
            if i + start_frame in cut_frames:
                scene += 1
            if value_in_interval(i + start_frame, fadein_frames):
                frame = add_text(frame, 'fadein', (10, height - 160), 3, (0, 255, 0), 2)
            if value_in_interval(i + start_frame, fadeout_frames):
                frame = add_text(frame, 'fadeout', (10, height - 200), 3, (0, 255, 0), 2)
            frame = add_text(frame, f'Cut {scene}', (10, height - 70), 3, (0, 255, 0), 2)
            frame = add_text(frame, f'frame {i + start_frame}', (15, height - 30), 1, (0, 255, 0), 2)

            if imshow:
                cv2.imshow('Frames', cv2.resize(frame, None, fx=0.5, fy=0.5))
            if output_video_path is not None:
                out.write(frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    if output_video_path is not None:
        out.release()
    cv2.destroyAllWindows()
