import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock


@pytest.fixture
def sample_frame():
    """Фикстура для создания тестового кадра"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_crops_data():
    """Фикстура для создания тестовых данных кропов"""
    return np.random.rand(10, 3, 3)


@pytest.fixture
def mock_video_capture():
    """Фикстура для мокирования cv2.VideoCapture"""
    mock_cap = Mock()
    mock_cap.get.side_effect = lambda prop: {
        0: 640,   # CAP_PROP_FRAME_WIDTH
        1: 480,   # CAP_PROP_FRAME_HEIGHT
        2: 30.0,  # CAP_PROP_FPS
        7: 100    # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    return mock_cap


@pytest.fixture
def temp_video_file():
    """Фикстура для создания временного видео файла"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)