import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from transition_detector.visualize import add_text, add_detailed_info, visualize


class TestAddText:
    """Тесты для функции add_text"""
    
    @patch('cv2.putText')
    def test_add_text_basic(self, mock_put_text):
        """Тест базовой функциональности add_text"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_put_text.return_value = frame
        
        result = add_text(
            frame=frame,
            text="Test Text",
            coord=(10, 20),
            size=1.0,
            color=(255, 255, 255),
            thickness=2
        )
        
        mock_put_text.assert_called_once()
        assert isinstance(result, np.ndarray)
    
    @patch('cv2.putText')
    def test_add_text_parameters(self, mock_put_text):
        """Тест передачи правильных параметров в cv2.putText"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_put_text.return_value = frame
        
        add_text(
            frame=frame,
            text="Hello",
            coord=(50, 60),
            size=2.5,
            color=(0, 255, 0),
            thickness=3
        )
        
        # Проверяем, что cv2.putText вызван с правильными параметрами
        args, kwargs = mock_put_text.call_args
        assert args[1] == "Hello"  # text
        assert args[2] == (50, 60)  # coord
        assert args[4] == 2.5  # size
        assert args[5] == (0, 255, 0)  # color
        assert args[6] == 3  # thickness


class TestAddDetailedInfo:
    """Тесты для функции add_detailed_info"""
    
    @patch('cv2.rectangle')
    @patch('transition_detector.visualize.add_text')
    @patch('transition_detector.visualize.get_changed_crops')
    def test_add_detailed_info_basic(self, mock_get_changed, mock_add_text, mock_rectangle):
        """Тест базовой функциональности add_detailed_info"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        prev_crops = np.array([[10.0, 20.0], [30.0, 40.0]])
        curr_crops = np.array([[15.0, 25.0], [35.0, 45.0]])
        
        # Мокаем get_changed_crops
        mock_get_changed.return_value = np.array([[True, False], [False, True]])
        
        # Мокаем add_text
        mock_add_text.return_value = frame
        
        result = add_detailed_info(
            frame=frame,
            prev_frame_crops_vals=prev_crops,
            curr_frame_crops_vals=curr_crops,
            is_cut=True,
            crop_dots_x0=[0, 50],
            crop_dots_y0=[0, 50],
            crop_dots_x1=[50, 100],
            crop_dots_y1=[50, 100],
            thr_difference_crop_cut=0.1
        )
        
        assert isinstance(result, np.ndarray)
        mock_get_changed.assert_called_once()
        # Проверяем, что cv2.rectangle вызывается для каждого кропа
        assert mock_rectangle.call_count == 4  # 2x2 кропа
        # Проверяем, что add_text вызывается дважды для каждого кропа (черная обводка + цветной текст)
        assert mock_add_text.call_count == 8  # 2 вызова * 4 кропа
    
    @patch('cv2.rectangle')
    @patch('transition_detector.visualize.add_text')
    @patch('transition_detector.visualize.get_changed_crops')
    def test_add_detailed_info_cut_colors(self, mock_get_changed, mock_add_text, mock_rectangle):
        """Тест правильных цветов для cut и non-cut кадров"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crops = np.array([[10.0]])
        
        mock_get_changed.return_value = np.array([[False]])
        mock_add_text.return_value = frame
        
        # Тест с is_cut=True
        add_detailed_info(
            frame=frame,
            prev_frame_crops_vals=crops,
            curr_frame_crops_vals=crops,
            is_cut=True,
            crop_dots_x0=[0],
            crop_dots_y0=[0],
            crop_dots_x1=[50],
            crop_dots_y1=[50],
            thr_difference_crop_cut=0.1
        )
        
        # Проверяем цвет прямоугольника для cut кадра
        args, _ = mock_rectangle.call_args
        assert args[3] == (0, 50, 170)  # цвет для cut
        
        mock_rectangle.reset_mock()
        
        # Тест с is_cut=False
        add_detailed_info(
            frame=frame,
            prev_frame_crops_vals=crops,
            curr_frame_crops_vals=crops,
            is_cut=False,
            crop_dots_x0=[0],
            crop_dots_y0=[0],
            crop_dots_x1=[50],
            crop_dots_y1=[50],
            thr_difference_crop_cut=0.1
        )
        
        # Проверяем цвет прямоугольника для non-cut кадра
        args, _ = mock_rectangle.call_args
        assert args[3] == (170, 50, 0)  # цвет для non-cut


class TestVisualize:
    """Тесты для функции visualize"""
    
    def test_visualize_invalid_parameters(self):
        """Тест с неверными параметрами"""
        with pytest.raises(AttributeError) as excinfo:
            visualize(
                video_path="test.mp4",
                cut_frames=np.array([]),
                fadein_frames=[],
                fadeout_frames=[],
                output_video_path=None,
                imshow=False
            )
        assert "will do nothing" in str(excinfo.value)
    
    def test_visualize_missing_crop_parameters(self):
        """Тест с отсутствующими параметрами для кропов"""
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = Mock()
            mock_cap.get.return_value = 100
            mock_cap_class.return_value = mock_cap
            
            with pytest.raises(AttributeError) as excinfo:
                visualize(
                    video_path="test.mp4",
                    cut_frames=np.array([]),
                    fadein_frames=[],
                    fadeout_frames=[],
                    frames_crops_mean_values=np.array([[[1, 2], [3, 4]]]),
                    crops_grid=None,  # Отсутствует обязательный параметр
                    output_video_path="output.mp4",
                    imshow=False
                )
            assert "must be specified" in str(excinfo.value)
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.waitKey')
    @patch('transition_detector.visualize.add_text')
    def test_visualize_basic_functionality(self, mock_add_text, mock_waitkey, 
                                         mock_destroy, mock_writer_class, mock_cap_class):
        """Тест базовой функциональности visualize"""
        # Настройка мока для VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 640,   # CAP_PROP_FRAME_WIDTH
            1: 480,   # CAP_PROP_FRAME_HEIGHT  
            2: 30.0,  # CAP_PROP_FPS
            7: 50     # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        # Мокаем чтение кадров
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 5 + [(False, None)]
        mock_cap_class.return_value = mock_cap
        
        # Настройка мока для VideoWriter
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        # Настройка других моков
        mock_waitkey.return_value = ord('q')  # Эмулируем нажатие 'q'
        mock_add_text.return_value = test_frame
        
        # Вызываем функцию
        visualize(
            video_path="test.mp4",
            cut_frames=np.array([1, 3]),
            fadein_frames=[(0, 2)],
            fadeout_frames=[(4, 6)],
            output_video_path="output.mp4",
            imshow=False,
            video_slice=(0, 5),
            disable_tqdm=True
        )
        
        # Проверяем вызовы
        mock_cap_class.assert_called_once_with("test.mp4")
        mock_writer_class.assert_called_once()
        mock_destroy.assert_called_once()
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('transition_detector.visualize.add_text')
    @patch('transition_detector.visualize.add_detailed_info')
    def test_visualize_with_crops_info(self, mock_add_detailed, mock_add_text, 
                                     mock_destroy, mock_waitkey, mock_imshow, mock_cap_class):
        """Тест visualize с детальной информацией о кропах"""
        # Настройка мока для VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 640,   # CAP_PROP_FRAME_WIDTH
            1: 480,   # CAP_PROP_FRAME_HEIGHT
            7: 10     # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 3 + [(False, None)]
        mock_cap_class.return_value = mock_cap
        
        # Настройка других моков
        mock_waitkey.return_value = ord('q')
        mock_add_text.return_value = test_frame
        mock_add_detailed.return_value = test_frame
        
        # Вызываем функцию с информацией о кропах
        frames_crops_data = np.random.rand(3, 2, 2)
        visualize(
            video_path="test.mp4",
            cut_frames=np.array([1]),
            fadein_frames=[],
            fadeout_frames=[],
            frames_crops_mean_values=frames_crops_data,
            crops_grid=(2, 2),
            crop_downscale=10,
            thr_difference_crop_cut=0.1,
            imshow=True,
            video_slice=(0, 3),
            disable_tqdm=True
        )
        
        # Проверяем, что add_detailed_info вызывается
        assert mock_add_detailed.call_count >= 1
        # Проверяем, что imshow вызывается
        assert mock_imshow.call_count >= 1
    
    @patch('cv2.VideoCapture')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    @patch('transition_detector.visualize.add_text')
    @patch('transition_detector.visualize.value_in_interval')
    def test_visualize_transitions_display(self, mock_value_in_interval, mock_add_text, 
                                         mock_destroy, mock_waitkey, mock_cap_class):
        """Тест отображения переходов"""
        # Настройка мока для VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 7: 5
        }.get(prop, 0)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 5 + [(False, None)]
        mock_cap_class.return_value = mock_cap
        
        mock_waitkey.return_value = ord('q')
        mock_add_text.return_value = test_frame
        
        # Настраиваем value_in_interval для возврата разных значений
        mock_value_in_interval.side_effect = lambda frame, intervals: frame == 1
        
        visualize(
            video_path="test.mp4",
            cut_frames=np.array([2]),
            fadein_frames=[(1, 3)],
            fadeout_frames=[(3, 5)],
            imshow=False,
            output_video_path="output.mp4",
            disable_tqdm=True
        )
        
        # Проверяем, что add_text вызывается для добавления информации о переходах
        assert mock_add_text.call_count > 0
        
        # Проверяем, что value_in_interval вызывается для проверки fadein/fadeout
        assert mock_value_in_interval.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__])