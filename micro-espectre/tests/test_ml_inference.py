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

from ml_detector import predict

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
        self.expected_outputs = self.test_data['expected_outputs']
    
    def test_inference_matches_reference(self):
        """Verify that inference matches reference model outputs."""
        max_error = 0.0
        num_samples = min(100, len(self.features))  # Test first 100 samples
        
        for i in range(num_samples):
            features = self.features[i].tolist()
            expected = self.expected_outputs[i]
            
            result = predict(features)
            error = abs(result - expected)
            max_error = max(max_error, error)
            
            # Allow small numerical error (1e-3) due to float32 precision
            # in manual MLP inference vs TensorFlow reference
            assert error < 1e-3, (
                f"Sample {i}: expected {expected:.6f}, got {result:.6f}, "
                f"error {error:.6f}"
            )
        
        print(f"\nTested {num_samples} samples, max error: {max_error:.2e}")
    
    def test_all_samples_match(self):
        """Verify all test samples match reference outputs."""
        errors = []
        
        for i in range(len(self.features)):
            features = self.features[i].tolist()
            expected = self.expected_outputs[i]
            result = predict(features)
            errors.append(abs(result - expected))
        
        errors = np.array(errors)
        max_error = errors.max()
        mean_error = errors.mean()
        
        print(f"\nAll {len(self.features)} samples tested:")
        print(f"  Max error:  {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        
        assert max_error < 1e-3, f"Max error {max_error:.2e} exceeds tolerance"
    
    def test_output_range(self):
        """Verify outputs are in valid probability range [0, 1]."""
        for i in range(len(self.features)):
            features = self.features[i].tolist()
            result = predict(features)
            
            assert 0.0 <= result <= 1.0, (
                f"Sample {i}: output {result} outside [0, 1] range"
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
        
        # Warm up
        for _ in range(10):
            predict(self.features[0].tolist())
        
        # Benchmark
        start_time = time.perf_counter()
        for i in range(num_iterations):
            predict(self.features[i % len(self.features)].tolist())
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        time_per_inference_us = (total_time_ms * 1000) / num_iterations
        inferences_per_second = num_iterations / (end_time - start_time)
        
        print(f"\nPerformance ({num_iterations} iterations):")
        print(f"  Total time:     {total_time_ms:.2f} ms")
        print(f"  Per inference:  {time_per_inference_us:.2f} us")
        print(f"  Rate:           {inferences_per_second:.0f} inferences/sec")
        
        # Should be fast enough for real-time (>100 inferences/sec)
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
        
        # Valid thresholds
        assert detector.set_threshold(0.0)
        assert detector.set_threshold(1.0)
        assert detector.set_threshold(0.5)
        
        # Invalid thresholds
        assert not detector.set_threshold(-0.1)
        assert not detector.set_threshold(1.1)
