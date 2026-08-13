# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - ML Inference Tests

Validation tests for ML inference behavior.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import numpy as np
import pytest

from tools.lib.repo_paths import generated_data_dir

from ml_detector import predict, ML_METRIC_SCALE
from tools.train_ml_model import (
    predict_probabilities_from_arrays,
    render_micropython_weights,
)

# Test data path
GENERATED_DATA_DIR = generated_data_dir()
TEST_DATA_PATH = GENERATED_DATA_DIR / 'ml_test_data.npz'
INFERENCE_TOLERANCE = 2e-3


def test_micropython_export_uses_inference_ready_weight_layout():
    source = render_micropython_weights(
        [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([0.25, -0.5], dtype=np.float32),
        ],
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        [2, 2],
        feature_names=["first", "second"],
        trained_at="2026-08-11 00:00:00",
    )
    namespace = {}
    exec(source, namespace)

    assert namespace["WEIGHTS_T"] == [[[1.0, 3.0], [2.0, 4.0]]]


def test_array_inference_uses_runtime_saturation_contract():
    layers = [
        (
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            True,
        )
    ]

    probabilities = predict_probabilities_from_arrays(
        np.asarray([[-21.0], [0.0], [21.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        layers,
    )

    np.testing.assert_array_equal(
        probabilities,
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )


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
    
    def test_all_samples_match(self):
        """Verify all test samples match reference outputs."""
        errors = []
        
        for i in range(len(self.features)):
            features = self.features[i].tolist()
            expected = self.expected_outputs[i] * ML_METRIC_SCALE
            result = predict(features)
            errors.append(abs(result - expected))
        
        errors = np.array(errors)
        max_error = errors.max()
        mean_error = errors.mean()
        
        print(f"\nAll {len(self.features)} samples tested:")
        print(f"  Max error:  {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        
        assert max_error < INFERENCE_TOLERANCE, (
            f"Max error {max_error:.2e} exceeds tolerance {INFERENCE_TOLERANCE:.2e}"
        )
    
    def test_output_range(self):
        """Verify outputs are in valid probability range [0, 1]."""
        for i in range(len(self.features)):
            features = self.features[i].tolist()
            result = predict(features)
            
            assert 0.0 <= result <= ML_METRIC_SCALE, (
                f"Sample {i}: output {result} outside [0, {ML_METRIC_SCALE}] range"
            )
