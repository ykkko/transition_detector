## How to use
The `transition_detector.py` module finds transitions such as cut, fadein and fadeout in a video.

To call it use the next script:
```python
from transition_detector import TransitionDetector

video_path = 'path/to/video'
t = TransitionDetector()
result = t.run(video_path)
```

... it returns a dict:
- keys `cut_seq`, `fadein_seq`, `fadeout_seq`: sequence (len = video frame num), each element represents a frame
  with the corresponding index, and the value means if the transition is in this frame or not
- key `cut_frames`: list of frame ids where cut is
- keys `fadein_frames`, `fadeout_frames`: list of tuples, 1st element in the tuple is the start of fading,
  2nd is the end of fading

You also can call `visualize.py` to visualize results:
```python
from transition_detector import TransitionDetector
from visualize import visualize

video_path = 'path/to/video'
result = TransitionDetector().run(video_path)
visualize(video_path, result)
```
