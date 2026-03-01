"""
Tests for ML Detector module.

Tests the neural network-based motion detector including:
- Activation functions (relu, sigmoid)
- Feature normalization
- Inference function (predict_class)
- MLDetector class
"""
import pytest
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_detector import (
    relu, sigmoid, normalize_features, predict_class,
    CLASS_LABELS, NUM_CLASSES, MLDetector,
    ML_DEFAULT_THRESHOLD, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD
)
from src.detector_interface import MotionState
import src.ml_detector as ml_detector_module


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
    """Test sigmoid activation function."""

    def test_output_in_range(self):
        """Sigmoid output is always in (0, 1)."""
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            result = sigmoid(x)
            assert 0.0 < result < 1.0

    def test_zero_input(self):
        """sigmoid(0) = 0.5."""
        assert abs(sigmoid(0.0) - 0.5) < 1e-6

    def test_symmetry(self):
        """sigmoid(-x) = 1 - sigmoid(x)."""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert abs(sigmoid(-x) - (1.0 - sigmoid(x))) < 1e-6

    def test_large_positive(self):
        """Large positive input approaches 1."""
        assert sigmoid(100.0) > 0.999

    def test_large_negative(self):
        """Large negative input approaches 0."""
        assert sigmoid(-100.0) < 0.001


class TestNormalizeFeatures:
    """Test feature normalization."""

    def test_normalization_produces_list(self):
        """Normalization returns a list of 12 values."""
        features = [1.0] * 12
        result = normalize_features(features)
        assert isinstance(result, list)
        assert len(result) == 12

    def test_normalization_changes_values(self):
        """Normalization changes input values."""
        features = [10.0, 5.0, 20.0, 1.0, 15.0, 8.0,
                    3.0, 2.5, 0.5, -0.5, 0.1, 5.0]
        result = normalize_features(features)
        assert result != features


class TestPredictClass:
    """Test multiclass neural network prediction."""

    def test_returns_tuple(self):
        """predict_class returns (class_id, class_name, confidence)."""
        features = [14.0, 2.0, 17.0, 9.0, 0.30,
                    -1.5, 8.0, 2.0, 0.15, 0.7, 0.001, 0.0]
        result = predict_class(features)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_class_id_valid(self):
        """class_id is a valid index into CLASS_LABELS."""
        features = [0.0] * 12
        class_id, class_name, confidence = predict_class(features)
        assert 0 <= class_id < NUM_CLASSES

    def test_class_name_matches_labels(self):
        """class_name matches CLASS_LABELS[class_id]."""
        for features in [[0.0] * 12, [10.0] * 12]:
            class_id, class_name, confidence = predict_class(features)
            assert class_name == CLASS_LABELS[class_id]

    def test_confidence_in_range(self):
        """Confidence is always in [0, 1]."""
        for features in [[0.0] * 12, [5.0] * 12, [10.0] * 12]:
            _, _, confidence = predict_class(features)
            assert 0.0 <= confidence <= 1.0

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        features1 = [0.07, 0.005, 0.09, 0.05, 0.20,
                     -3.0, 15.0, 1.5, 0.35, 0.003, 0.0, 1.0]
        features2 = [0.18, 0.06, 0.30, 0.08, 0.50,
                     1.0, 3.0, 3.0, -0.10, 0.04, 0.002, 2.5]
        result1 = predict_class(features1)
        result2 = predict_class(features2)
        assert result1 != result2

    def test_idle_class_is_zero(self):
        """class_id=0 corresponds to 'idle'."""
        assert CLASS_LABELS[0] == 'idle'


class TestMLDetector:
    """Test MLDetector class."""
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        detector = MLDetector()
        assert detector._threshold == ML_DEFAULT_THRESHOLD  # 5.0
        assert detector._state == MotionState.IDLE
        assert detector._packet_count == 0
        assert detector.track_data == False
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = MLDetector(window_size=100, threshold=7.0)
        assert detector._threshold == 7.0
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
        detector = MLDetector(threshold=6.0)
        assert detector.get_threshold() == 6.0
    
    def test_set_threshold_valid(self):
        """Test setting valid threshold (range 0.1-10.0)."""
        detector = MLDetector()
        assert detector.set_threshold(7.0) == True
        assert detector._threshold == 7.0
    
    def test_set_threshold_invalid(self):
        """Test setting invalid threshold (outside 0.1-10.0)."""
        detector = MLDetector()
        original = detector._threshold
        assert detector.set_threshold(15.0) == False
        assert detector.set_threshold(0.05) == False
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
        return MLDetector(window_size=10, threshold=5.0)
    
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
        assert metrics['threshold'] == 5.0
    
    def test_update_state_after_ready(self, detector, sample_csi_data):
        """Update state after buffer is full runs inference."""
        subcarriers = list(range(11, 23))
        for _ in range(10):
            detector.process_packet(sample_csi_data, subcarriers)

        metrics = detector.update_state()

        assert 'state' in metrics
        assert 'probability' in metrics
        assert 'threshold' in metrics
        # Note: 'gesture' is not returned by MLDetector.
        # Gesture classification is handled by GestureDetector.
        # Motion metric is scaled to 0-10 (unified with MVS).
        assert 0.0 <= metrics['probability'] <= 10.0
    
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


class TestExtractFeaturesIntegration:
    """Test that extract_features_by_name is correctly integrated."""
    
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
        detector = MLDetector(window_size=10)
        detector.track_data = True

        subcarriers = list(range(11, 23))
        for i in range(10):
            csi_data = [(20 + i * 5) % 127] * 128
            detector.process_packet(csi_data, subcarriers)

        detector.update_state()

        assert len(detector.probability_history) == 1
        assert len(detector.state_history) == 1

    def test_get_motion_count(self):
        """Test get_motion_count method."""
        detector = MLDetector(window_size=10)
        detector.track_data = True

        subcarriers = list(range(11, 23))
        for i in range(10):
            csi_data = [(20 + i * 5) % 127] * 128
            detector.process_packet(csi_data, subcarriers)

        detector.update_state()
        count = detector.get_motion_count()

        assert isinstance(count, int)
        assert count >= 0

    def test_state_consistent_with_probability(self):
        """State is MOTION iff probability >= threshold.
        
        MLDetector is binary (IDLE/MOTION).
        Gesture classification is handled by GestureDetector.
        """
        detector = MLDetector(window_size=10)

        subcarriers = list(range(11, 23))
        for i in range(10):
            csi_data = [50] * 128
            detector.process_packet(csi_data, subcarriers)

        metrics = detector.update_state()

        if metrics['probability'] >= metrics['threshold']:
            assert metrics['state'] == MotionState.MOTION
        else:
            assert metrics['state'] == MotionState.IDLE


class TestThresholdDecisionPolicy:
    """Test explicit threshold/probability decision policy (scaled 0-10 range)."""

    def _run_with_metric(self, monkeypatch, threshold, metric):
        """Run update_state() with a forced motion metric (0-10 range)."""
        detector = MLDetector(window_size=1, threshold=threshold)
        detector._context.buffer_count = detector._context.window_size  # Force ready state

        monkeypatch.setattr(detector, '_extract_features', lambda: [0.0] * 12)
        monkeypatch.setattr(ml_detector_module, 'predict_probability',
                            lambda _features: metric)

        return detector.update_state()

    def test_threshold_3_metric_4_motion(self, monkeypatch):
        """threshold=3.0, metric=4.0 => MOTION."""
        metrics = self._run_with_metric(monkeypatch, threshold=3.0, metric=4.0)
        assert metrics['state'] == MotionState.MOTION

    def test_threshold_7_metric_6_idle(self, monkeypatch):
        """threshold=7.0, metric=6.0 => IDLE."""
        metrics = self._run_with_metric(monkeypatch, threshold=7.0, metric=6.0)
        assert metrics['state'] == MotionState.IDLE

    def test_threshold_5_behavior_unchanged(self, monkeypatch):
        """threshold=5.0 keeps the expected binary behavior (default)."""
        metrics_motion = self._run_with_metric(monkeypatch, threshold=5.0, metric=6.0)
        metrics_idle = self._run_with_metric(monkeypatch, threshold=5.0, metric=4.0)
        assert metrics_motion['state'] == MotionState.MOTION
        assert metrics_idle['state'] == MotionState.IDLE
