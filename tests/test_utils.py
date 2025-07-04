import pytest
import numpy as np
from transition_detector.utils import (
    remove_short_ones, split_bins, value_in_interval, get_changed_crops
)


class TestRemoveShortOnes:
    """Тесты для функции remove_short_ones"""
    
    def test_basic_functionality(self):
        """Тест базовой функциональности"""
        seq = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        result = remove_short_ones(seq, minimum_zeros_length=2)
        # Короткие группы нулей должны быть заполнены единицами
        assert isinstance(result, np.ndarray)
        assert len(result) == len(seq)
    
    def test_even_minimum_length(self):
        """Тест с четной минимальной длиной"""
        seq = np.array([1, 0, 0, 1, 1, 0, 1, 1])
        result = remove_short_ones(seq, minimum_zeros_length=4)
        assert len(result) == len(seq)
    
    def test_odd_minimum_length(self):
        """Тест с нечетной минимальной длиной"""
        seq = np.array([1, 0, 0, 1, 1, 0, 1, 1])
        result = remove_short_ones(seq, minimum_zeros_length=3)
        assert len(result) == len(seq)
    
    def test_empty_array(self):
        """Тест с пустым массивом"""
        seq = np.array([])
        # Ожидаем ValueError для пустого массива, так как numpy.convolve не работает с пустыми массивами
        with pytest.raises(ValueError):
            remove_short_ones(seq, minimum_zeros_length=2)
    
    def test_all_zeros(self):
        """Тест с массивом из всех нулей"""
        seq = np.array([0, 0, 0, 0, 0])
        result = remove_short_ones(seq, minimum_zeros_length=3)
        assert isinstance(result, np.ndarray)
    
    def test_all_ones(self):
        """Тест с массивом из всех единиц"""
        seq = np.array([1, 1, 1, 1, 1])
        result = remove_short_ones(seq, minimum_zeros_length=3)
        assert isinstance(result, np.ndarray)


class TestSplitBins:
    """Тесты для функции split_bins"""
    
    def test_basic_functionality(self):
        """Тест базовой функциональности"""
        conditional_array = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        result = split_bins(conditional_array)
        expected = [(2, 5), (6, 7)]
        assert result == expected
    
    def test_single_group(self):
        """Тест с одной группой единиц"""
        conditional_array = np.array([0, 0, 1, 1, 1, 0, 0])
        result = split_bins(conditional_array)
        expected = [(2, 5)]
        assert result == expected
    
    def test_continuous_ones(self):
        """Тест с непрерывными единицами"""
        conditional_array = np.array([1, 1, 1, 1, 1])
        result = split_bins(conditional_array)
        expected = [(0, 5)]
        assert result == expected
    
    def test_no_ones(self):
        """Тест без единиц"""
        conditional_array = np.array([0, 0, 0, 0, 0])
        result = split_bins(conditional_array)
        assert result == []
    
    def test_empty_array(self):
        """Тест с пустым массивом"""
        conditional_array = np.array([])
        result = split_bins(conditional_array)
        assert result == []
    
    def test_single_one(self):
        """Тест с одной единицей"""
        conditional_array = np.array([0, 1, 0])
        result = split_bins(conditional_array)
        expected = [(1, 2)]
        assert result == expected
    
    def test_alternating_pattern(self):
        """Тест с чередующимся паттерном"""
        conditional_array = np.array([1, 0, 1, 0, 1])
        result = split_bins(conditional_array)
        expected = [(0, 1), (2, 3), (4, 5)]
        assert result == expected


class TestValueInInterval:
    """Тесты для функции value_in_interval"""
    
    def test_value_in_single_interval(self):
        """Тест со значением в единственном интервале"""
        assert value_in_interval(5, [(2, 8)]) == True
        assert value_in_interval(2, [(2, 8)]) == True
        assert value_in_interval(7, [(2, 8)]) == True
    
    def test_value_not_in_single_interval(self):
        """Тест со значением вне единственного интервала"""
        assert value_in_interval(1, [(2, 8)]) == False
        assert value_in_interval(8, [(2, 8)]) == False
        assert value_in_interval(10, [(2, 8)]) == False
    
    def test_value_in_multiple_intervals(self):
        """Тест со значением в одном из нескольких интервалов"""
        intervals = [(1, 3), (5, 8), (10, 15)]
        assert value_in_interval(2, intervals) == True
        assert value_in_interval(6, intervals) == True
        assert value_in_interval(12, intervals) == True
    
    def test_value_not_in_multiple_intervals(self):
        """Тест со значением вне всех интервалов"""
        intervals = [(1, 3), (5, 8), (10, 15)]
        assert value_in_interval(4, intervals) == False
        assert value_in_interval(9, intervals) == False
        assert value_in_interval(16, intervals) == False
    
    def test_empty_intervals(self):
        """Тест с пустым списком интервалов"""
        assert value_in_interval(5, []) == False
    
    def test_edge_cases(self):
        """Тест граничных случаев"""
        intervals = [(0, 5), (10, 15)]
        assert value_in_interval(0, intervals) == True  # начало интервала
        assert value_in_interval(5, intervals) == False  # конец интервала (не включается)
        assert value_in_interval(10, intervals) == True  # начало второго интервала
        assert value_in_interval(15, intervals) == False  # конец второго интервала


class TestGetChangedCrops:
    """Тесты для функции get_changed_crops"""
    
    def test_basic_functionality(self):
        """Тест базовой функциональности"""
        prev_frame = np.array([[10.0, 20.0], [30.0, 40.0]])
        curr_frame = np.array([[15.0, 25.0], [35.0, 45.0]])
        threshold = 0.1
        
        result = get_changed_crops(prev_frame, curr_frame, threshold)
        assert result.shape == prev_frame.shape
        assert result.dtype == bool
    
    def test_no_changes(self):
        """Тест без изменений"""
        frame = np.array([[10.0, 20.0], [30.0, 40.0]])
        threshold = 0.1
        
        result = get_changed_crops(frame, frame, threshold)
        assert not result.any()  # все False
    
    def test_significant_changes(self):
        """Тест с существенными изменениями"""
        prev_frame = np.array([[10.0, 20.0], [30.0, 40.0]])
        curr_frame = np.array([[50.0, 60.0], [70.0, 80.0]])  # большие изменения
        threshold = 0.1
        
        result = get_changed_crops(prev_frame, curr_frame, threshold)
        assert result.all()  # все True
    
    def test_partial_changes(self):
        """Тест с частичными изменениями"""
        prev_frame = np.array([[10.0, 20.0], [30.0, 40.0]])
        curr_frame = np.array([[10.5, 50.0], [30.5, 40.5]])  # некоторые большие изменения
        threshold = 0.1
        
        result = get_changed_crops(prev_frame, curr_frame, threshold)
        # Проверяем, что есть и True, и False значения
        assert result.any() and not result.all()
    
    def test_different_thresholds(self):
        """Тест с разными порогами"""
        prev_frame = np.array([[10.0, 20.0]])
        curr_frame = np.array([[11.0, 21.0]])  # небольшие изменения
        
        # С низким порогом - изменения должны быть обнаружены
        result_low = get_changed_crops(prev_frame, curr_frame, 0.05)
        # С высоким порогом - изменения не должны быть обнаружены
        result_high = get_changed_crops(prev_frame, curr_frame, 0.2)
        
        assert result_low.any()
        assert not result_high.any()
    
    def test_zero_values(self):
        """Тест с нулевыми значениями"""
        prev_frame = np.array([[0.0, 10.0], [20.0, 0.0]])
        curr_frame = np.array([[5.0, 15.0], [25.0, 5.0]])
        threshold = 0.1
        
        result = get_changed_crops(prev_frame, curr_frame, threshold)
        assert result.shape == prev_frame.shape
    
    def test_single_crop(self):
        """Тест с одним кропом"""
        prev_frame = np.array([[10.0]])
        curr_frame = np.array([[20.0]])
        threshold = 0.1
        
        result = get_changed_crops(prev_frame, curr_frame, threshold)
        assert result.shape == (1, 1)
        assert result[0, 0] == True  # значительное изменение


if __name__ == "__main__":
    pytest.main([__file__])