## How to use

#### Video-level interface (when you just need to find transitions in the video)

The `transition_detector/transition_detector.py` module finds transitions such as cut, fadein and fadeout in a video.
To call it use the next script:
```python
from transition_detector import TransitionDetector

video_path = 'path/to/video'
td = TransitionDetector()
cut_frames, fadein_frames, fadeout_frames, frames_crops_mean_values = td.run(video_path)
```
... it returns:
- `cut_frames`: np.ndarray with frame indexes where cut is
- `fadein_frames`, `fadeout_frames`: list of tuples, 1st element in the tuple is the start of fading,
  2nd is the end of fading
- `frames_crops_mean_values`: sequence with shape (frame_num, CROP_ROWS, CROP_COLUMNS),
  each element of sequence is np.ndarray with crop's mean values. This is useful for visualizing the results.

In case you need to analyze specific piece of video you can define this piece using attribute `video_slice`:
```python
_, _, _, _ = TransitionDetector().run(video_path, video_slice=(400, 600))
```
WARNING: This is done with `cap.set (cv2.CAP_PROP_POS_FRAMES, value)`, but it doesn't work correctly
with ALL videos in the world. If you see poor results, try to process the whole video (with `slice_video=None`).

#### Frame-level interface (when you read video frames from the outside and run a lot of frame processing programs)

Each frame must be processed by `TransitionDetector.get_frame_crops_values`. It returns crops values of frame.
You must append it to a list. After frames processing you must convert the list to np.ndarray and pass it to 
`TransitionDetector.get_transitions` method. It will return `cut_frames`, `fadein_frames` and `fadeout_frames` 
which described above.

Example:
```python
import cv2
import numpy as np

from transition_detector import TransitionDetector


video_path = 'path/to/video'

cap = cv2.VideoCapture(video_path)
td = TransitionDetector()

frames_values = []
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frames_values.append(td.get_frame_crops_values(frame))
cap.release()

frames_values = np.array(frames_values)
cut_frames, fadein_frames, fadeout_frames = td.get_transitions(frames_values)
```

---
### Visualization
Call `td.visualize` to see results or write them to video file:
```python
from transition_detector import TransitionDetector, visualize

video_path = 'path/to/video'
video_slice = 500, 1000

td = TransitionDetector()
cut_frames, fadein_frames, fadeout_frames, frames_crops_mean_values = td.run(video_path, video_slice=video_slice)
td.visualize(video_path=video_path, cut_frames=cut_frames, fadeout_frames=fadein_frames, fadein_frames=fadeout_frames,
             frames_crops_mean_values=frames_crops_mean_values,
             video_slice=video_slice, imshow=True, output_video_path='result.mp4')
```
`visualize` shows frames or saves video with information about transitions on frames.
- If pass `frames_crops_mean_values`, frames will be contained detailed information about all crops of frame.
- If you processed a video fragment using `video_slice`, define `video_slice` for visualize as well.
- `output_video_path`: path to video with information on frames. If None, then video will not be created.
- `imshow`: call cv2.imshow if True.

When visualization is running, press the "q" key to exit.
