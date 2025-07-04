import pytest
import numpy as np
from unittest.mock import patch, Mock
from transition_detector import TransitionDetector, visualize


class TestIntegration:
    """Интеграционные тесты для всего пакета"""
    
    def test_full_pipeline_mock(self, mock_video_capture):
        """Тест полного пайплайна с мокированными данными"""
        td = TransitionDetector(
            resize_frame=(120, 160),
            crops_grid=(3, 3),
            crop_downscale=10,
            disable_tqdm=True
        )
        
        # Создаем синтетические данные кропов
        frames_crops_data = np.random.rand(20, 3, 3)
        
        # Добавляем резкие изменения для имитации cuts
        frames_crops_data[5] = frames_crops_data[4] + 50  # cut в кадре 5
        frames_crops_data[15] = frames_crops_data[14] + 50  # cut в кадре 15
        
        # Создаем градиентные изменения для имитации fade
        for i in range(8, 12):  # fadein от кадра 8 до 12
            frames_crops_data[i] = frames_crops_data[7] + (i - 7) * 10
        
        cut_frames, fadein_frames, fadeout_frames = td.get_transitions(frames_crops_data)
        
        # Проверяем результаты
        assert isinstance(cut_frames, np.ndarray)
        assert isinstance(fadein_frames, list)
        assert isinstance(fadeout_frames, list)
    
    @patch('cv2.VideoCapture')
    def test_visualize_integration(self, mock_video_capture_class):
        """Тест интеграции visualize с основными функциями"""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 640, 1: 480, 2: 30.0, 7: 10
        }.get(prop, 0)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame)] * 5 + [(False, None)]
        mock_video_capture_class.return_value = mock_cap
        
        with patch('cv2.VideoWriter'), \
             patch('cv2.destroyAllWindows'), \
             patch('cv2.waitKey', return_value=ord('q')):
            
            visualize(
                video_path="test.mp4",
                cut_frames=np.array([2, 4]),
                fadein_frames=[(1, 3)],
                fadeout_frames=[(5, 7)],
                output_video_path="output.mp4",
                imshow=False,
                disable_tqdm=True
            )
    
    def test_parameter_consistency(self):
        """Тест согласованности параметров между компонентами"""
        td = TransitionDetector(
            resize_frame=(240, 320),
            crops_grid=(4, 5),
            crop_downscale=8
        )
        
        # Проверяем, что параметры кропов согласованы
        assert len(td.crop_dots_y0) == 4
        assert len(td.crop_dots_x0) == 5
        assert len(td.crop_dots_y1) == 4
        assert len(td.crop_dots_x1) == 5
        
        # Проверяем размеры кропов
        crop_height = 240 // 8
        crop_width = 320 // 8
        
        for i in range(4):
            assert td.crop_dots_y1[i] - td.crop_dots_y0[i] == crop_height
        for i in range(5):
            assert td.crop_dots_x1[i] - td.crop_dots_x0[i] == crop_width
    
    def test_edge_cases_small_video(self):
        """Тест граничных случаев с коротким видео"""
        td = TransitionDetector(disable_tqdm=True)
        
        # Данные достаточной длины для обработки fade (нужно минимум 5 кадров)
        short_data = np.random.rand(10, 3, 3)
        
        cut_frames, fadein_frames, fadeout_frames = td.get_transitions(short_data)
        
        # Проверяем, что функция не падает на коротких данных
        assert isinstance(cut_frames, np.ndarray)
        assert isinstance(fadein_frames, list)
        assert isinstance(fadeout_frames, list)
    
    def test_threshold_sensitivity(self):
        """Тест чувствительности к порогам"""
        # Тест с очень чувствительными порогами
        td_sensitive = TransitionDetector(
            thr_share_of_changed_crops_cut=0.1,
            thr_difference_crop_cut=0.01,
            disable_tqdm=True
        )
        
        # Тест с очень нечувствительными порогами
        td_insensitive = TransitionDetector(
            thr_share_of_changed_crops_cut=0.9,
            thr_difference_crop_cut=0.5,
            disable_tqdm=True
        )
        
        # Одинаковые данные с небольшими изменениями
        data = np.random.rand(10, 3, 3)
        data[5] = data[4] + 0.1  # небольшое изменение
        
        cuts_sensitive, _, _ = td_sensitive.get_transitions(data)
        cuts_insensitive, _, _ = td_insensitive.get_transitions(data)
        
        # Чувствительный детектор должен найти больше изменений
        assert len(cuts_sensitive) >= len(cuts_insensitive)


if __name__ == "__main__":
    pytest.main([__file__])