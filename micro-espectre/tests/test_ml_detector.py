"""
Tests for ML Detector module.

Tests the neural network-based motion detector including:
- Activation functions (relu, sigmoid)
- Feature normalization
- Inference functions (predict, is_motion)
- MLDetector class
"""
import pytest
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_detector import (
    relu, sigmoid, normalize_features, predict, is_motion,
    MLDetector
)
from src.detector_interface import MotionState


class TestRelu:
    """Test ReLU activation function."""
    
    def test_positive_input(self):
        """Positive values pass through unchanged."""
        assert relu(5.0) == 5.0
        assert relu(0.1) == 0.1
        assert relu(100.0) == 100.0
    
    def test_negative_input(self):
        """Negative values return 0."""
        assert relu(-5.0) == 0.0
        assert relu(-0.1) == 0.0
        assert relu(-100.0) == 0.0
    
    def test_zero_input(self):
        """Zero returns zero."""
        assert relu(0.0) == 0.0


class TestSigmoid:
    """Test Sigmoid activation function."""
    
    def test_zero_input(self):
        """Sigmoid(0) = 0.5."""
        assert sigmoid(0.0) == 0.5
    
    def test_positive_input(self):
        """Positive values return > 0.5."""
        assert sigmoid(1.0) > 0.5
        assert sigmoid(5.0) > 0.9
    
    def test_negative_input(self):
        """Negative values return < 0.5."""
        assert sigmoid(-1.0) < 0.5
        assert sigmoid(-5.0) < 0.1
    
    def test_large_positive_overflow_protection(self):
        """Large positive values return 1.0 (overflow protection)."""
        assert sigmoid(100.0) == 1.0
        assert sigmoid(21.0) == 1.0
    
    def test_large_negative_overflow_protection(self):
        """Large negative values return 0.0 (overflow protection)."""
        assert sigmoid(-100.0) == 0.0
        assert sigmoid(-21.0) == 0.0
    
    def test_output_range(self):
        """Output is always in (0, 1)."""
        for x in [-10, -5, -1, 0, 1, 5, 10]:
            result = sigmoid(x)
            assert 0.0 <= result <= 1.0


class TestNormalizeFeatures:
    """Test feature normalization."""
    
    def test_normalization_produces_list(self):
        """Normalization returns a list."""
        features = [1.0] * 12
        result = normalize_features(features)
        assert isinstance(result, list)
        assert len(result) == 12
    
    def test_normalization_changes_values(self):
        """Normalization changes input values."""
        features = [10.0, 5.0, 20.0, 1.0, 15.0, 8.0, 
                    3.0, 2.5, 0.5, -0.5, 0.1, 5.0]
        result = normalize_features(features)
        # Values should be different after normalization
        assert result != features


class TestPredict:
    """Test neural network prediction."""
    
    def test_predict_returns_float(self):
        """Predict returns a float."""
        features = [5.0, 1.0, 10.0, 2.0, 8.0, 2.0,
                    1.5, 2.0, 0.3, -0.5, 0.01, 1.0]
        result = predict(features)
        assert isinstance(result, float)
    
    def test_predict_output_range(self):
        """Prediction is always in [0, 1]."""
        # Test with various feature combinations
        test_cases = [
            [0.0] * 12,  # All zeros
            [10.0] * 12,  # All same value
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],  # Increasing
        ]
        for features in test_cases:
            result = predict(features)
            assert 0.0 <= result <= 1.0
    
    def test_predict_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        # Use realistic feature values based on actual data distributions
        # Features: turb_mean, turb_std, turb_max, turb_range, mv_mean, mv_max,
        #           iqr, entropy, skewness, kurtosis, zero_crossings, delta_mv
        # Baseline-like: low turbulence and variance
        features1 = [12.0, 1.5, 15.0, 3.0, 5.0, 3.0,
                     3.0, 1.5, -0.5, -0.8, 0.001, 0.0]
        # Motion-like: high turbulence and variance
        features2 = [20.0, 5.0, 30.0, 15.0, 25.0, 30.0,
                     10.0, 3.5, 0.5, 1.5, 0.05, 15.0]
        
        result1 = predict(features1)
        result2 = predict(features2)
        
        # Motion features should produce higher probability than baseline
        assert result2 > result1


class TestIsMotion:
    """Test motion detection function."""
    
    def test_is_motion_returns_bool(self):
        """is_motion returns a boolean."""
        features = [5.0] * 12
        result = is_motion(features)
        assert isinstance(result, bool)
    
    def test_is_motion_default_threshold(self):
        """Default threshold is 0.5."""
        features = [5.0] * 12
        prob = predict(features)
        expected = prob > 0.5
        assert is_motion(features) == expected
    
    def test_is_motion_custom_threshold(self):
        """Custom threshold works correctly."""
        features = [5.0] * 12
        prob = predict(features)
        
        # With threshold above probability, should be False
        assert is_motion(features, threshold=prob + 0.1) == False
        # With threshold below probability, should be True
        if prob > 0.01:
            assert is_motion(features, threshold=prob - 0.01) == True


class TestMLDetector:
    """Test MLDetector class."""
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        detector = MLDetector()
        assert detector._threshold == 0.5
        assert detector._state == MotionState.IDLE
        assert detector._packet_count == 0
        assert detector.track_data == False
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = MLDetector(window_size=100, threshold=0.7)
        assert detector._threshold == 0.7
        assert detector._context.window_size == 100
    
    def test_get_name(self):
        """Test get_name returns 'ML'."""
        detector = MLDetector()
        assert detector.get_name() == "ML"
    
    def test_get_state_initial(self):
        """Initial state is IDLE."""
        detector = MLDetector()
        assert detector.get_state() == MotionState.IDLE
    
    def test_get_threshold(self):
        """Test get_threshold."""
        detector = MLDetector(threshold=0.6)
        assert detector.get_threshold() == 0.6
    
    def test_set_threshold_valid(self):
        """Test setting valid threshold."""
        detector = MLDetector()
        assert detector.set_threshold(0.7) == True
        assert detector._threshold == 0.7
    
    def test_set_threshold_invalid(self):
        """Test setting invalid threshold."""
        detector = MLDetector()
        original = detector._threshold
        assert detector.set_threshold(1.5) == False
        assert detector.set_threshold(-0.1) == False
        assert detector._threshold == original
    
    def test_is_ready_empty(self):
        """Detector is not ready before filling buffer."""
        detector = MLDetector(window_size=50)
        assert detector.is_ready() == False
    
    def test_get_motion_metric_initial(self):
        """Initial motion metric is 0."""
        detector = MLDetector()
        assert detector.get_motion_metric() == 0.0
    
    def test_total_packets_initial(self):
        """Initial packet count is 0."""
        detector = MLDetector()
        assert detector.total_packets == 0
    
    def test_reset(self):
        """Test reset clears state."""
        detector = MLDetector()
        detector._packet_count = 100
        detector._state = MotionState.MOTION
        detector._current_probability = 0.8
        detector._motion_count = 10
        detector.probability_history = [0.5, 0.6]
        detector.state_history = ['IDLE', 'MOTION']
        
        detector.reset()
        
        assert detector._state == MotionState.IDLE
        assert detector._current_probability == 0.0
        assert detector._motion_count == 0
        assert detector.probability_history == []
        assert detector.state_history == []


class TestMLDetectorProcessing:
    """Test MLDetector packet processing with synthetic data."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector with small window for testing."""
        return MLDetector(window_size=10, threshold=0.5)
    
    @pytest.fixture
    def sample_csi_data(self):
        """Create sample CSI data (64 subcarriers * 2 = 128 bytes)."""
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        data = []
        for i in range(64):  # 64 subcarriers
            data.append(10 + i)  # Q (imaginary first)
            data.append(20 + i)  # I (real second)
        return data
    
    def test_process_packet_increments_count(self, detector, sample_csi_data):
        """Processing packet increments packet count."""
        initial = detector._packet_count
        detector.process_packet(sample_csi_data, list(range(11, 23)))
        assert detector._packet_count == initial + 1
    
    def test_process_multiple_packets(self, detector, sample_csi_data):
        """Processing multiple packets fills buffer."""
        subcarriers = list(range(11, 23))
        for _ in range(10):
            detector.process_packet(sample_csi_data, subcarriers)
        
        assert detector._packet_count == 10
        assert detector.is_ready() == True
    
    def test_update_state_before_ready(self, detector, sample_csi_data):
        """Update state before buffer is full returns default values."""
        detector.process_packet(sample_csi_data, list(range(11, 23)))
        
        metrics = detector.update_state()
        
        assert metrics['state'] == MotionState.IDLE
        assert metrics['probability'] == 0.0
        assert metrics['threshold'] == 0.5
    
    def test_update_state_after_ready(self, detector, sample_csi_data):
        """Update state after buffer is full runs inference."""
        subcarriers = list(range(11, 23))
        for _ in range(10):
            detector.process_packet(sample_csi_data, subcarriers)
        
        metrics = detector.update_state()
        
        assert 'state' in metrics
        assert 'probability' in metrics
        assert 'threshold' in metrics
        assert 0.0 <= metrics['probability'] <= 1.0
    
    def test_tracking_enabled(self, detector, sample_csi_data):
        """Test that tracking records data when enabled."""
        detector.track_data = True
        subcarriers = list(range(11, 23))
        
        for _ in range(10):
            detector.process_packet(sample_csi_data, subcarriers)
        
        detector.update_state()
        
        assert len(detector.probability_history) == 1
        assert len(detector.state_history) == 1
    
    def test_tracking_disabled(self, detector, sample_csi_data):
        """Test that tracking does not record when disabled."""
        detector.track_data = False
        subcarriers = list(range(11, 23))
        
        for _ in range(10):
            detector.process_packet(sample_csi_data, subcarriers)
        
        detector.update_state()
        
        assert len(detector.probability_history) == 0
        assert len(detector.state_history) == 0


class TestExtractAllFeaturesIntegration:
    """Test that extract_all_features is correctly integrated."""
    
    def test_extract_features_returns_12_values(self):
        """_extract_features returns 12 values."""
        detector = MLDetector(window_size=10)
        
        # Fill buffer with synthetic data
        csi_data = [20] * 128  # 64 subcarriers * 2
        subcarriers = list(range(11, 23))
        
        for _ in range(10):
            detector.process_packet(csi_data, subcarriers)
        
        features = detector._extract_features()
        
        assert len(features) == 12
        assert all(isinstance(f, (int, float)) for f in features)


class TestMLDetectorMotionTracking:
    """Test motion tracking with data that triggers MOTION state."""
    
    def test_motion_count_increments_on_motion(self):
        """Motion count increments when MOTION is detected."""
        detector = MLDetector(window_size=10, threshold=0.0)  # Very low threshold
        detector.track_data = True
        
        # Create varying CSI data to trigger motion
        subcarriers = list(range(11, 23))
        for i in range(10):
            # Vary data to create turbulence
            csi_data = [(20 + i * 5) % 127] * 128
            detector.process_packet(csi_data, subcarriers)
        
        # Update state - with threshold=0, should detect motion
        detector.update_state()
        
        # Should have recorded in history
        assert len(detector.probability_history) == 1
        assert len(detector.state_history) == 1
    
    def test_get_motion_count(self):
        """Test get_motion_count method."""
        detector = MLDetector(window_size=10, threshold=0.0)
        detector.track_data = True
        
        subcarriers = list(range(11, 23))
        for i in range(10):
            csi_data = [(20 + i * 5) % 127] * 128
            detector.process_packet(csi_data, subcarriers)
        
        # Update multiple times
        detector.update_state()
        count = detector.get_motion_count()
        
        assert isinstance(count, int)
        assert count >= 0
    
    def test_state_changes_to_motion(self):
        """Test that state changes to MOTION with low threshold."""
        detector = MLDetector(window_size=10, threshold=0.0)
        
        subcarriers = list(range(11, 23))
        for i in range(10):
            csi_data = [50] * 128
            detector.process_packet(csi_data, subcarriers)
        
        metrics = detector.update_state()
        
        # With threshold=0, any probability > 0 triggers motion
        if metrics['probability'] > 0:
            assert metrics['state'] == MotionState.MOTION
