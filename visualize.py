import cv2

from constants import RESULT_DICT_KEY_NAME_FADEIN_SEQUENCE, RESULT_DICT_KEY_NAME_FADEOUT_SEQUENCE, \
    RESULT_DICT_KEY_NAME_CUT_FRAMES


def visualize(video_path, data):
    times = data[RESULT_DICT_KEY_NAME_CUT_FRAMES]
    fadein = data[RESULT_DICT_KEY_NAME_FADEIN_SEQUENCE]
    fadeout = data[RESULT_DICT_KEY_NAME_FADEOUT_SEQUENCE]

    video = cv2.VideoCapture(video_path)

    i = -1
    scene = 1
    while True:
        ret, frame = video.read()
        if ret:
            i += 1
            if i in times:
                scene += 1
            if fadein[i]:
                cv2.putText(frame, 'fadein', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2,
                            lineType=cv2.LINE_AA)
            if fadeout[i]:
                cv2.putText(frame, 'fadeout', (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2,
                            lineType=cv2.LINE_AA)
            cv2.putText(frame, f'Cut {scene}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, f'Frame {i}', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        lineType=cv2.LINE_AA)

            cv2.imshow('1', cv2.resize(frame, None, fx=0.5, fy=0.5))

            if cv2.waitKey(1) == 27:
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
