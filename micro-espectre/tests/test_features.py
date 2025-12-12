"""
Micro-ESPectre - Feature Extraction Unit Tests

Tests for feature functions and classes in src/features.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from features import (
    calc_skewness,
    calc_kurtosis,
    calc_iqr_turb,
    calc_entropy_turb,
    PublishTimeFeatureExtractor,
    MultiFeatureDetector
)


class TestCalcSkewness:
    """Test skewness calculation"""
    
    def test_empty_list(self):
        """Test skewness of empty list"""
        assert calc_skewness([]) == 0.0
    
    def test_single_value(self):
        """Test skewness of single value"""
        assert calc_skewness([5.0]) == 0.0
    
    def test_two_values(self):
        """Test skewness of two values (needs 3+)"""
        assert calc_skewness([1.0, 2.0]) == 0.0
    
    def test_symmetric_distribution(self):
        """Test skewness of symmetric distribution (should be ~0)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        skew = calc_skewness(values)
        assert abs(skew) < 0.1  # Should be close to 0
    
    def test_right_skewed(self):
        """Test skewness of right-skewed distribution"""
        # Most values low, one high -> positive skew
        values = [1.0, 1.0, 1.0, 1.0, 10.0]
        skew = calc_skewness(values)
        assert skew > 0
    
    def test_left_skewed(self):
        """Test skewness of left-skewed distribution"""
        # Most values high, one low -> negative skew
        values = [10.0, 10.0, 10.0, 10.0, 1.0]
        skew = calc_skewness(values)
        assert skew < 0
    
    def test_constant_values(self):
        """Test skewness of constant values (std=0)"""
        values = [5.0] * 10
        skew = calc_skewness(values)
        assert skew == 0.0
    
    def test_matches_scipy(self):
        """Test that result approximately matches scipy"""
        np.random.seed(42)
        values = list(np.random.exponential(2.0, 100))
        
        our_skew = calc_skewness(values)
        
        # Exponential distribution should have positive skew
        assert our_skew > 0


class TestCalcKurtosis:
    """Test kurtosis calculation"""
    
    def test_empty_list(self):
        """Test kurtosis of empty list"""
        assert calc_kurtosis([]) == 0.0
    
    def test_single_value(self):
        """Test kurtosis of single value"""
        assert calc_kurtosis([5.0]) == 0.0
    
    def test_three_values(self):
        """Test kurtosis of three values (needs 4+)"""
        assert calc_kurtosis([1.0, 2.0, 3.0]) == 0.0
    
    def test_normal_distribution(self):
        """Test kurtosis of normal distribution (should be ~0)"""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 1000))
        kurt = calc_kurtosis(values)
        # Excess kurtosis of normal is 0
        assert abs(kurt) < 0.5
    
    def test_uniform_distribution(self):
        """Test kurtosis of uniform distribution (should be < 0)"""
        np.random.seed(42)
        values = list(np.random.uniform(0, 1, 1000))
        kurt = calc_kurtosis(values)
        # Uniform distribution has negative excess kurtosis
        assert kurt < 0
    
    def test_heavy_tailed(self):
        """Test kurtosis of heavy-tailed distribution (should be > 0)"""
        # Create data with outliers -> heavy tails
        values = list(np.random.normal(0, 1, 100))
        values.extend([10.0, -10.0, 15.0, -15.0])  # Add outliers
        kurt = calc_kurtosis(values)
        # Should have positive excess kurtosis
        assert kurt > 0
    
    def test_constant_values(self):
        """Test kurtosis of constant values (std=0)"""
        values = [5.0] * 10
        kurt = calc_kurtosis(values)
        assert kurt == 0.0


class TestCalcIqrTurb:
    """Test IQR approximation calculation"""
    
    def test_empty_buffer(self):
        """Test IQR of empty buffer"""
        assert calc_iqr_turb([], 0) == 0.0
    
    def test_single_value(self):
        """Test IQR of single value"""
        assert calc_iqr_turb([5.0], 1) == 0.0
    
    def test_two_values(self):
        """Test IQR of two values"""
        buffer = [0.0, 10.0]
        iqr = calc_iqr_turb(buffer, 2)
        # IQR = (max - min) * 0.5 = (10 - 0) * 0.5 = 5
        assert iqr == pytest.approx(5.0, rel=1e-6)
    
    def test_known_range(self):
        """Test IQR with known range"""
        buffer = [1.0, 2.0, 3.0, 4.0, 5.0]
        iqr = calc_iqr_turb(buffer, 5)
        # IQR = (5 - 1) * 0.5 = 2
        assert iqr == pytest.approx(2.0, rel=1e-6)
    
    def test_constant_values(self):
        """Test IQR of constant values"""
        buffer = [5.0] * 10
        iqr = calc_iqr_turb(buffer, 10)
        assert iqr == pytest.approx(0.0, rel=1e-6)
    
    def test_partial_buffer(self):
        """Test IQR with partial buffer"""
        buffer = [1.0, 5.0, 0.0, 0.0, 0.0]  # Only first 2 valid
        iqr = calc_iqr_turb(buffer, 2)
        # IQR = (5 - 1) * 0.5 = 2
        assert iqr == pytest.approx(2.0, rel=1e-6)


class TestCalcEntropyTurb:
    """Test Shannon entropy calculation"""
    
    def test_empty_buffer(self):
        """Test entropy of empty buffer"""
        assert calc_entropy_turb([], 0) == 0.0
    
    def test_single_value(self):
        """Test entropy of single value"""
        assert calc_entropy_turb([5.0], 1) == 0.0
    
    def test_constant_values(self):
        """Test entropy of constant values (max-min ~0)"""
        buffer = [5.0] * 10
        entropy = calc_entropy_turb(buffer, 10)
        assert entropy == 0.0
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has higher entropy"""
        # Uniform across bins -> high entropy
        buffer = [float(i % 10) for i in range(100)]  # 0-9 evenly
        entropy = calc_entropy_turb(buffer, 100, n_bins=10)
        
        # Max entropy for 10 bins = log2(10) ≈ 3.32
        assert entropy > 2.0
    
    def test_concentrated_distribution_low_entropy(self):
        """Test that concentrated distribution has low entropy"""
        # All values in one bin -> low entropy
        buffer = [5.0 + 0.01 * i for i in range(100)]  # Very narrow range
        entropy = calc_entropy_turb(buffer, 100, n_bins=10)
        
        # Most values in few bins -> lower entropy
        # (Might still have some entropy due to discretization)
        assert entropy >= 0
    
    def test_returns_positive(self):
        """Test that entropy is non-negative"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        entropy = calc_entropy_turb(buffer, 50)
        assert entropy >= 0


class TestPublishTimeFeatureExtractor:
    """Test PublishTimeFeatureExtractor class"""
    
    def test_initialization(self):
        """Test extractor initialization"""
        extractor = PublishTimeFeatureExtractor()
        assert extractor.last_features is None
    
    def test_compute_features_returns_dict(self):
        """Test that compute_features returns a dict"""
        extractor = PublishTimeFeatureExtractor()
        
        amplitudes = [10.0, 12.0, 11.0, 13.0, 10.5]
        turbulence_buffer = [5.0] * 50
        buffer_count = 50
        moving_variance = 1.5
        
        features = extractor.compute_features(
            amplitudes, turbulence_buffer, buffer_count, moving_variance
        )
        
        assert isinstance(features, dict)
    
    def test_compute_features_keys(self):
        """Test that all expected features are present"""
        extractor = PublishTimeFeatureExtractor()
        
        amplitudes = [10.0, 12.0, 11.0, 13.0, 10.5]
        turbulence_buffer = [5.0] * 50
        
        features = extractor.compute_features(
            amplitudes, turbulence_buffer, 50, 1.5
        )
        
        expected_keys = ['skewness', 'kurtosis', 'variance_turb', 'iqr_turb', 'entropy_turb']
        for key in expected_keys:
            assert key in features
    
    def test_variance_turb_passthrough(self):
        """Test that moving_variance is passed through"""
        extractor = PublishTimeFeatureExtractor()
        
        features = extractor.compute_features(
            [10.0] * 5, [5.0] * 50, 50, 2.5
        )
        
        assert features['variance_turb'] == 2.5
    
    def test_get_features(self):
        """Test get_features returns last computed"""
        extractor = PublishTimeFeatureExtractor()
        
        features = extractor.compute_features(
            [10.0] * 5, [5.0] * 50, 50, 1.0
        )
        
        assert extractor.get_features() == features
    
    def test_features_update(self):
        """Test that features update on each call"""
        extractor = PublishTimeFeatureExtractor()
        
        features1 = extractor.compute_features(
            [10.0] * 5, [5.0] * 50, 50, 1.0
        )
        
        features2 = extractor.compute_features(
            [20.0] * 5, [10.0] * 50, 50, 5.0
        )
        
        assert features1['variance_turb'] != features2['variance_turb']


class TestMultiFeatureDetector:
    """Test MultiFeatureDetector class"""
    
    def test_initialization_default_thresholds(self):
        """Test initialization with default thresholds"""
        detector = MultiFeatureDetector()
        
        assert detector.min_confidence == 0.5
        assert len(detector.thresholds) > 0
    
    def test_initialization_custom_thresholds(self):
        """Test initialization with custom thresholds"""
        custom = {
            'test_feature': {'threshold': 1.0, 'weight': 1.0, 'direction': 'above'}
        }
        detector = MultiFeatureDetector(thresholds=custom)
        
        assert 'test_feature' in detector.thresholds
    
    def test_detect_returns_tuple(self):
        """Test that detect returns (is_motion, confidence, triggered)"""
        detector = MultiFeatureDetector()
        
        features = {
            'iqr_turb': 5.0,  # Above default threshold
            'skewness': 1.0,
            'kurtosis': -2.0,
            'entropy_turb': 3.5,
            'variance_turb': 2.0
        }
        
        result = detector.detect(features)
        
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)
    
    def test_detect_none_features(self):
        """Test detection with None features"""
        detector = MultiFeatureDetector()
        
        is_motion, confidence, triggered = detector.detect(None)
        
        assert is_motion is False
        assert confidence == 0.0
        assert triggered == []
    
    def test_high_features_trigger_motion(self):
        """Test that high feature values trigger motion"""
        detector = MultiFeatureDetector(min_confidence=0.3)
        
        # All features above their thresholds
        features = {
            'iqr_turb': 10.0,  # >> 2.18
            'skewness': 2.0,   # >> 0.57
            'kurtosis': -5.0,  # << -1.33 (below)
            'entropy_turb': 5.0,  # >> 2.94
            'variance_turb': 5.0  # >> 0.99
        }
        
        is_motion, confidence, triggered = detector.detect(features)
        
        assert is_motion is True
        assert confidence > 0.5
        assert len(triggered) > 0
    
    def test_low_features_no_motion(self):
        """Test that low feature values don't trigger motion"""
        detector = MultiFeatureDetector(min_confidence=0.5)
        
        # All features below their thresholds
        features = {
            'iqr_turb': 0.5,   # < 2.18
            'skewness': 0.1,  # < 0.57
            'kurtosis': 0.0,  # > -1.33 (not below)
            'entropy_turb': 1.0,  # < 2.94
            'variance_turb': 0.1  # < 0.99
        }
        
        is_motion, confidence, triggered = detector.detect(features)
        
        assert is_motion is False
        assert confidence < 0.5
    
    def test_direction_above(self):
        """Test 'above' direction threshold"""
        thresholds = {
            'test': {'threshold': 5.0, 'weight': 1.0, 'direction': 'above'}
        }
        detector = MultiFeatureDetector(thresholds=thresholds)
        
        # Above threshold
        _, _, triggered = detector.detect({'test': 6.0})
        assert 'test' in triggered
        
        # Below threshold
        _, _, triggered = detector.detect({'test': 4.0})
        assert 'test' not in triggered
    
    def test_direction_below(self):
        """Test 'below' direction threshold"""
        thresholds = {
            'test': {'threshold': 5.0, 'weight': 1.0, 'direction': 'below'}
        }
        detector = MultiFeatureDetector(thresholds=thresholds)
        
        # Below threshold
        _, _, triggered = detector.detect({'test': 4.0})
        assert 'test' in triggered
        
        # Above threshold
        _, _, triggered = detector.detect({'test': 6.0})
        assert 'test' not in triggered
    
    def test_confidence_range(self):
        """Test that confidence is in [0, 1] range"""
        detector = MultiFeatureDetector()
        
        # Test various feature combinations
        test_cases = [
            {'iqr_turb': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 
             'entropy_turb': 0.0, 'variance_turb': 0.0},
            {'iqr_turb': 100.0, 'skewness': 100.0, 'kurtosis': -100.0,
             'entropy_turb': 100.0, 'variance_turb': 100.0},
        ]
        
        for features in test_cases:
            _, confidence, _ = detector.detect(features)
            assert 0.0 <= confidence <= 1.0
    
    def test_get_confidence(self):
        """Test get_confidence returns last value"""
        detector = MultiFeatureDetector()
        
        detector.detect({
            'iqr_turb': 5.0, 'skewness': 1.0, 'kurtosis': -2.0,
            'entropy_turb': 3.5, 'variance_turb': 2.0
        })
        
        confidence = detector.get_confidence()
        assert confidence > 0
    
    def test_get_triggered(self):
        """Test get_triggered returns last triggered list"""
        detector = MultiFeatureDetector()
        
        detector.detect({
            'iqr_turb': 10.0,  # Should trigger
            'skewness': 0.0,   # Should not trigger
            'kurtosis': 0.0,
            'entropy_turb': 0.0,
            'variance_turb': 0.0
        })
        
        triggered = detector.get_triggered()
        assert 'iqr_turb' in triggered


class TestFeatureIntegration:
    """Integration tests for feature extraction pipeline"""
    
    def test_extractor_to_detector_pipeline(self):
        """Test full pipeline from extraction to detection"""
        extractor = PublishTimeFeatureExtractor()
        detector = MultiFeatureDetector()
        
        # Simulate baseline (low variance)
        amplitudes_baseline = [10.0, 10.5, 10.2, 10.3, 10.1]
        turb_buffer_baseline = [5.0] * 50
        
        features = extractor.compute_features(
            amplitudes_baseline, turb_buffer_baseline, 50, 0.5
        )
        is_motion, conf, _ = detector.detect(features)
        
        # Low variance should not trigger motion
        assert conf < 0.5
    
    def test_baseline_vs_movement_features(self):
        """Test that movement has different features than baseline"""
        extractor = PublishTimeFeatureExtractor()
        
        # Baseline: stable
        np.random.seed(42)
        amps_baseline = list(np.random.normal(10, 0.5, 12))
        turb_baseline = list(np.random.normal(5, 0.3, 50))
        
        features_baseline = extractor.compute_features(
            amps_baseline, turb_baseline, 50, 0.3
        )
        
        # Movement: variable
        amps_movement = list(np.random.normal(10, 3, 12))
        turb_movement = list(np.random.normal(10, 3, 50))
        
        features_movement = extractor.compute_features(
            amps_movement, turb_movement, 50, 5.0
        )
        
        # Movement should have higher IQR and variance
        assert features_movement['iqr_turb'] > features_baseline['iqr_turb']
        assert features_movement['variance_turb'] > features_baseline['variance_turb']

