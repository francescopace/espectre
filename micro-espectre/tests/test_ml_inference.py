"""
Micro-ESPectre - ML Inference Validation Tests

Tests that the Python ML inference implementation produces correct results
when compared against the reference model outputs stored in ml_test_data.npz.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ml_detector import predict_class, predict_probability, CLASS_LABELS, NUM_CLASSES

# Test data path
MODELS_DIR = Path(__file__).parent.parent / 'models'
TEST_DATA_PATH = MODELS_DIR / 'ml_test_data.npz'


class TestMLInferenceAccuracy:
    """Test ML inference accuracy against reference model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load test data before each test."""
        if not TEST_DATA_PATH.exists():
            pytest.skip(f"Test data not found: {TEST_DATA_PATH}")

        self.test_data = np.load(TEST_DATA_PATH)
        self.features = self.test_data['features']
        # expected_outputs: float32, 1 - prob[idle] — same format as C++ MLDetector::predict()
        self.expected_outputs = self.test_data['expected_outputs']

    def test_inference_matches_reference(self):
        """Verify Python predict_probability() matches reference float outputs (tolerance 1e-4).

        Mirrors test_ml_inference_matches_reference in C++ test_ml_detector.cpp.
        """
        TOLERANCE = 1e-4
        num_samples = min(100, len(self.features))

        for i in range(num_samples):
            features = self.features[i].tolist()
            result = predict_probability(features)
            expected = float(self.expected_outputs[i])

            assert abs(result - expected) < TOLERANCE, (
                f"Sample {i}: expected {expected:.6f}, got {result:.6f} "
                f"(error={abs(result - expected):.2e})"
            )

        print(f"\nTested {num_samples} samples, all within tolerance {TOLERANCE}")

    def test_all_samples_match(self):
        """Verify all samples match reference outputs within tolerance."""
        TOLERANCE = 1e-4
        errors = 0

        for i in range(len(self.features)):
            features = self.features[i].tolist()
            result = predict_probability(features)
            expected = float(self.expected_outputs[i])
            if abs(result - expected) >= TOLERANCE:
                errors += 1

        accuracy = (len(self.features) - errors) / len(self.features) * 100
        print(f"\nAll {len(self.features)} samples tested:")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Errors:   {errors}")

        assert errors == 0, f"{errors} samples outside tolerance {TOLERANCE}"

    def test_confidence_in_range(self):
        """Verify confidence values are in valid probability range [0, 1]."""
        for i in range(len(self.features)):
            features = self.features[i].tolist()
            _, _, confidence = predict_class(features)

            assert 0.0 <= confidence <= 1.0, (
                f"Sample {i}: confidence {confidence} outside [0, 1] range"
            )


class TestMLInferencePerformance:
    """Benchmark ML inference performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load test data before each test."""
        if not TEST_DATA_PATH.exists():
            pytest.skip(f"Test data not found: {TEST_DATA_PATH}")

        self.test_data = np.load(TEST_DATA_PATH)
        self.features = self.test_data['features']

    def test_inference_speed(self):
        """Benchmark inference speed."""
        num_iterations = 1000

        for _ in range(10):
            predict_class(self.features[0].tolist())

        start_time = time.perf_counter()
        for i in range(num_iterations):
            predict_class(self.features[i % len(self.features)].tolist())
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        time_per_inference_us = (total_time_ms * 1000) / num_iterations
        inferences_per_second = num_iterations / (end_time - start_time)

        print(f"\nPerformance ({num_iterations} iterations):")
        print(f"  Total time:     {total_time_ms:.2f} ms")
        print(f"  Per inference:  {time_per_inference_us:.2f} us")
        print(f"  Rate:           {inferences_per_second:.0f} inferences/sec")

        assert inferences_per_second > 100, (
            f"Inference too slow: {inferences_per_second:.0f} inferences/sec"
        )


class TestMLDetectorIntegration:
    """Integration tests for MLDetector class."""

    def test_mldetector_import(self):
        """Test that MLDetector can be imported."""
        from ml_detector import MLDetector, ML_SUBCARRIERS

        assert MLDetector is not None
        assert len(ML_SUBCARRIERS) == 12

    def test_mldetector_initialization(self):
        """Test MLDetector initialization."""
        from ml_detector import MLDetector

        detector = MLDetector(window_size=50, threshold=0.5)
        assert detector is not None
        assert detector.get_name() == "ML"
        assert detector.get_threshold() == 0.5

    def test_mldetector_threshold_bounds(self):
        """Test threshold validation."""
        from ml_detector import MLDetector

        detector = MLDetector()

        assert detector.set_threshold(0.0)
        assert detector.set_threshold(1.0)
        assert detector.set_threshold(0.5)

        assert not detector.set_threshold(-0.1)
        assert not detector.set_threshold(1.1)

    def test_num_classes_matches_labels(self):
        """NUM_CLASSES matches length of CLASS_LABELS."""
        assert NUM_CLASSES == len(CLASS_LABELS)
        assert NUM_CLASSES >= 2

    def test_idle_is_class_zero(self):
        """CLASS_LABELS[0] is always 'idle'."""
        assert CLASS_LABELS[0] == 'idle'
