"""
ESPectre - Real Data Validation Tests

Validation tests using real CSI datasets.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
from functools import lru_cache

# ============================================================================
# Detector Constants (imported from config.py, matches C++ base_detector.h)
# ============================================================================
import numpy as np
import math

from filters import HampelFilter
from tools.lib.csi_analysis import calculate_spatial_turbulence
from tools.lib.performance_report import (
    STRESS_TARGET_FP_RATE,
    STRESS_TARGET_RECALL,
    compute_classic_dataset_result as _compute_classic_dataset_result,
    compute_classic_empty_fp_result as _compute_classic_empty_fp_result,
    compute_ml_dataset_result as _compute_ml_dataset_result,
    compute_ml_empty_fp_result as _compute_ml_empty_fp_result,
    evaluate_detector_packets,
    get_available_chip_types as _shared_get_available_chip_types,
    get_available_empty_datasets as _shared_get_available_empty_datasets,
    get_available_paired_datasets as _shared_get_available_paired_datasets,
    get_paired_dataset_role as _get_paired_dataset_role,
    is_low_rssi_paired_dataset as _is_low_rssi_paired_dataset,
    load_real_data_cached as _load_real_data_cached,
)
from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import (
    build_calibrated_classic_detector,
)
from config import (
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    EVALUATION_INTERVAL,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    SEG_WINDOW_SIZE as DETECTOR_DEFAULT_WINDOW_SIZE,
    HAMPEL_WINDOW,
    HAMPEL_THRESHOLD,
    LOWPASS_CUTOFF,
)
from classic_detector import ClassicDetector
from conftest import get_classic_fp_rate_target, get_classic_recall_target, record_performance
from threshold import StartupThresholdCalibrator, get_detector_auto_factor, get_detector_startup_gate


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_available_datasets():
    """Get explicit static-presence/motion pairs (HT20: 64 SC only)."""
    return [
        pytest.param(dataset, id=dataset[4])
        for dataset in _shared_get_available_paired_datasets()
    ]


def get_available_empty_datasets():
    """Get empty-room recordings for ML false-positive gates."""
    return [
        pytest.param(path, id=path.stem)
        for path in _shared_get_available_empty_datasets()
    ]


def get_available_chip_types():
    """Return the stable set of chips covered by the paired real-data datasets."""
    return _shared_get_available_chip_types()


ML_RESERVED_REPLAY_GUARDRAIL_RECALL = 90.0
ML_RESERVED_REPLAY_GUARDRAIL_FP_RATE = 10.0
ML_STRESS_REPLAY_GUARDRAIL_RECALL = 85.0
ML_STRESS_REPLAY_GUARDRAIL_FP_RATE = 15.0


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=get_available_datasets())
def dataset_config(request):
    """
    Parametrized fixture that provides dataset configuration.
    Tests using this fixture will run once per available dataset.
    
    Returns:
        tuple: (static_presence_path, motion_path, num_subcarriers, chip, dataset_id)
    """
    return request.param


@pytest.fixture
def real_data(dataset_config):
    """Load real CSI data from the current dataset.
    
    Matches C++ behavior (csi_test_data.h):
    - Baseline: all packets loaded, starting from packet 0
    - Movement: all packets loaded
    """
    static_presence_path, motion_path, num_sc, chip, dataset_id = dataset_config

    return _load_real_data_cached(static_presence_path, motion_path)


@pytest.fixture
def num_subcarriers(dataset_config):
    """Get number of subcarriers for current dataset"""
    _, _, num_sc, _, _ = dataset_config
    return num_sc


@pytest.fixture
def chip_type(dataset_config):
    """Get chip type for current dataset"""
    _, _, _, chip, _ = dataset_config
    return chip


@pytest.fixture
def dataset_id(dataset_config):
    """Get the stable dataset id for current static-presence/motion pair."""
    _, _, _, _, dataset_id_value = dataset_config
    return dataset_id_value


@pytest.fixture
def link_stress(dataset_config):
    """True when the pair is a real weak-link (`low_rssi`) stress capture."""
    static_presence_path, _, _, _, _ = dataset_config
    return _is_low_rssi_paired_dataset(static_presence_path)


@pytest.fixture
def dataset_role(dataset_config):
    """Return the normalized ML provenance role for the paired replay."""
    static_presence_path, _, _, _, _ = dataset_config
    role = _get_paired_dataset_role(static_presence_path)
    return role or "train"


@pytest.fixture
def window_size(chip_type):
    """Get optimal window size for chip type.
    
    All chips use the same window size for consistent behavior.
    This matches the production default DETECTOR_DEFAULT_WINDOW_SIZE.
    """
    return DETECTOR_DEFAULT_WINDOW_SIZE


@pytest.fixture(params=["fixed_default"])
def calibration_algorithm(request, chip_type):
    """
    Parametrized fixture for the fixed-subcarrier startup calibration path.
    """
    algo = request.param
    return algo


@pytest.fixture
def enable_hampel(chip_type):
    """Enable Hampel filter for chip type.
    
    Matches C++ get_enable_hampel(): true for all chips.
    """
    return True


# ============================================================================
# Shared Variance Calibration Tests
# ============================================================================

def run_fixed_subcarrier_calibration(static_presence_packets, num_subcarriers, hint_band=None, window_size_override=None):
    """
    Run fixed-subcarrier Classic startup calibration exactly as in production.

    Calibration starts from packet 0 and uses the first CALIBRATION_BUFFER_SIZE
    packets, matching live startup behavior.

    Args:
        static_presence_packets: List of baseline CSI packets
        num_subcarriers: Number of subcarriers
        hint_band: Optional subcarrier band override (defaults to fixed defaults).
        window_size_override: Optional detector window size for validation

    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    selected_band = hint_band if hint_band is not None else DEFAULT_SUBCARRIERS
    window_size = window_size_override or DETECTOR_DEFAULT_WINDOW_SIZE
    adaptive_threshold = run_classic_calibration(
        static_presence_packets,
        selected_band=tuple(selected_band),
        window_size=window_size,
    )
    return selected_band, adaptive_threshold


@lru_cache(maxsize=None)
def _load_packets_cached(path_value):
    """Load and cache packet dictionaries for a .npz dataset."""
    return tuple(load_npz_as_packets(path_value))


@lru_cache(maxsize=None)
def _compute_startup_calibration_result(
    static_presence_path,
    num_subcarriers,
    algorithm,
    hint_band,
    window_size_override,
):
    """Cache startup calibration results for repeated validation checks."""
    static_presence_packets = _load_packets_cached(str(static_presence_path))
    return run_calibration(
        static_presence_packets,
        num_subcarriers,
        algorithm,
        hint_band=hint_band,
        window_size_override=window_size_override,
    )


def run_calibration(static_presence_packets, num_subcarriers, algorithm="fixed_default", hint_band=None,
                    window_size_override=None):
    """
    Run startup calibration using fixed subcarriers.
    
    Args:
        static_presence_packets: List of baseline CSI packets
        num_subcarriers: Number of subcarriers
        algorithm: Calibration variant name (only "fixed_default" supported)
        hint_band: Optional fixed subcarrier band to use
        window_size_override: Optional shared variance window size for validation
    
    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    return run_fixed_subcarrier_calibration(
        static_presence_packets,
        num_subcarriers,
        hint_band=hint_band,
        window_size_override=window_size_override,
    )


def run_classic_calibration(static_presence_packets, selected_band, window_size):
    """Run startup calibration for the classic detector."""
    detector = ClassicDetector(
        window_size=window_size,
        threshold=1.0,
        enable_lowpass=ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=LOWPASS_CUTOFF,
        enable_hampel=ENABLE_HAMPEL_FILTER,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
    )
    calibrator = StartupThresholdCalibrator(
        CALIBRATION_BUFFER_SIZE,
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    packets_since_evaluation = 0
    for pkt in static_presence_packets:
        detector.process_packet(pkt["csi_data"], selected_band)
        packets_since_evaluation += 1
        if packets_since_evaluation < EVALUATION_INTERVAL:
            continue
        detector.update_state()
        calibrator.observe_detector(
            detector,
            packet_weight=packets_since_evaluation,
        )
        packets_since_evaluation = 0
        if calibrator.is_complete():
            break
    if not calibrator.is_successful():
        return 1.0
    threshold, _ = calibrator.calculate_threshold()
    return float(threshold)


# ============================================================================
# Hampel Filter Tests with Real Data
# ============================================================================

class TestHampelFilterRealData:
    """Test Hampel filter with real CSI turbulence data"""
    
    def test_hampel_reduces_spikes(self, real_data, default_subcarriers):
        """Test that Hampel filter reduces turbulence spikes"""
        static_presence_packets, motion_packets = real_data
        all_packets = static_presence_packets + motion_packets
        
        # Calculate raw turbulence
        raw_turbulence = []
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            raw_turbulence.append(turb)
        
        # Apply Hampel filter
        hf = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        filtered_turbulence = [hf.filter(t) for t in raw_turbulence]
        
        # Filtered should have lower max (spikes reduced)
        raw_max = max(raw_turbulence)
        filtered_max = max(filtered_turbulence)
        
        # If there were spikes, they should be reduced
        if raw_max > np.mean(raw_turbulence) * 3:
            assert filtered_max <= raw_max, "Hampel should not increase max value"
    
    def test_hampel_preserves_variance_separation(self, real_data, default_subcarriers):
        """Test that Hampel filter preserves baseline/movement separation"""
        static_presence_packets, motion_packets = real_data
        
        # Calculate filtered turbulence for baseline
        hf_baseline = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        static_presence_turb = []
        for pkt in static_presence_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            filtered = hf_baseline.filter(turb)
            static_presence_turb.append(filtered)
        
        # Calculate filtered turbulence for movement
        hf_movement = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        motion_turb = []
        for pkt in motion_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            filtered = hf_movement.filter(turb)
            motion_turb.append(filtered)
        
        # Movement should still have higher variance
        static_presence_var = np.var(static_presence_turb)
        motion_var = np.var(motion_turb)
        
        assert motion_var > static_presence_var, \
            f"Movement variance ({motion_var:.3f}) should be > baseline ({static_presence_var:.3f})"


# ============================================================================
# Performance Metrics Tests
# ============================================================================

class TestPerformanceMetrics:
    """Test that we achieve expected performance metrics with fixed subcarriers."""

    def test_classic_detection_accuracy(self, dataset_config, fp_rate_target, recall_target,
                                        chip_type, default_subcarriers, dataset_id,
                                        link_stress):
        """
        Test Classic motion detection accuracy with fixed production subcarriers.

        This is the promotion gate for the sole non-ML runtime detector.
        Normal-link pairs are gated at the production targets; real weak-link
        (`low_rssi`) pairs are stress diagnostics and stay report-only, because
        at very low RSSI the motion/static turbulence separation collapses and
        the threshold detector has no headroom left to trade.
        """
        static_presence_path, motion_path, _num_sc, _chip, _dataset_id = dataset_config
        cached_result = _compute_classic_dataset_result(
            static_presence_path,
            motion_path,
            tuple(default_subcarriers),
            DETECTOR_DEFAULT_WINDOW_SIZE,
        )
        assert cached_result is not None, "Classic startup calibration failed"
        adaptive_threshold, metrics = cached_result

        print("\n")
        print("=" * 70)
        print("                   CLASSIC DETECTION TEST SUMMARY")
        print("=" * 70)
        print(f"Dataset pair: {dataset_id}")
        print(f"Subcarriers: {default_subcarriers}")
        print(f"Threshold:   {adaptive_threshold:.6f}")
        print()
        print(f"CONFUSION MATRIX ({metrics['num_baseline']} baseline + {metrics['num_movement']} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {metrics['tn']:4d} (TN)  {metrics['fp']:4d} (FP)")
        print(f"    MOTION      {metrics['fn']:4d} (FN)  {metrics['tp']:4d} (TP)")
        print()
        print("MOTION DETECTION METRICS:")
        print(f"  * Recall:     {metrics['recall']:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {metrics['precision']:.1f}%")
        print(f"  * FP Rate:    {metrics['fp_rate']:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {metrics['f1']:.1f}%")
        print()
        print("=" * 70)

        record_performance(
            chip_type,
            "classic",
            metrics["recall"],
            metrics["fp_rate"],
            metrics["precision"],
            metrics["f1"],
            dataset_id=dataset_id,
        )

        assert 0.0 <= adaptive_threshold <= 1.0
        assert 0.0 <= metrics["recall"] <= 100.0
        assert 0.0 <= metrics["precision"] <= 100.0
        assert 0.0 <= metrics["fp_rate"] <= 100.0
        assert 0.0 <= metrics["f1"] <= 100.0

        if link_stress:
            print("Link class: weak (low_rssi) -> Classic stress replay, report-only")
            return
        assert metrics["recall"] > recall_target, (
            f"Classic Recall too low: {metrics['recall']:.1f}% (target: >{recall_target}%)"
        )
        if metrics["num_baseline"] > 0:
            assert metrics["fp_rate"] < fp_rate_target, (
                f"Classic FP Rate too high: {metrics['fp_rate']:.1f}% (target: <{fp_rate_target}%)"
            )

    def test_ml_detection_accuracy(self, dataset_config, num_subcarriers, ml_fp_rate_target, ml_recall_target,
                                   chip_type, dataset_id, link_stress, dataset_role):
        """
        Test ML (Neural Network) motion detection accuracy with real CSI data.
        
        ML uses a pre-trained MLP model for motion classification.
        No calibration needed - uses pre-trained weights.
        
        Note: ML model uses fixed subcarriers from config.DEFAULT_SUBCARRIERS regardless of chip type.
        ML uses the shared CV-normalized turbulence path.
        
        This per-recording replay is mostly diagnostic. Promotion gates live in
        the aggregate ML target tests below, split by provenance and link class:
        reserved normal-link replays measure honest generalization, training-role
        replays stay in-sample diagnostics, and real weak-link (`low_rssi`)
        captures stay stress diagnostics with relaxed targets. Individual
        reserved/stress replays still keep coarse anti-catastrophe guardrails so
        a single very bad capture does not disappear inside a good aggregate.
        """
        from config import DEFAULT_SUBCARRIERS

        static_presence_path, motion_path, _num_sc, _chip, _dataset_id = dataset_config
        if link_stress:
            ml_recall_target = STRESS_TARGET_RECALL
            ml_fp_rate_target = STRESS_TARGET_FP_RATE

        # ML model uses fixed subcarriers (must match training)
        ml_subcarriers = DEFAULT_SUBCARRIERS
        # ========================================
        # Initialize ML Detector (no calibration needed)
        # ========================================
        cached_metrics, _feature_payload = _compute_ml_dataset_result(
            static_presence_path,
            motion_path,
            tuple(ml_subcarriers),
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
            threshold=0.5,
        )

        print("\nML Detector initialized")
        print("  Threshold: 0.5")
        print(f"  Window size: {DETECTOR_DEFAULT_WINDOW_SIZE} (DETECTOR_DEFAULT_WINDOW_SIZE)")
        print(f"  Subcarriers: {ml_subcarriers} (fixed for ML)")
        print("  Turbulence: normalized runtime path")
        if link_stress:
            print("  Link class: weak (low_rssi) -> ML stress targets")
        else:
            print(f"  Provenance role: {dataset_role}")

        # ========================================
        # Print results
        # ========================================
        print("\n")
        print("=" * 70)
        print("                     ML DETECTION TEST SUMMARY")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({cached_metrics['num_baseline']} baseline + {cached_metrics['num_movement']} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {cached_metrics['tn']:4d} (TN)  {cached_metrics['fp']:4d} (FP)")
        print(f"    MOTION      {cached_metrics['fn']:4d} (FN)  {cached_metrics['tp']:4d} (TP)")
        print()
        print("METRICS:")
        print(f"  * Recall:     {cached_metrics['recall']:.1f}% (target: >{ml_recall_target}%)")
        print(f"  * Precision:  {cached_metrics['precision']:.1f}%")
        print(f"  * FP Rate:    {cached_metrics['fp_rate']:.1f}% (target: <{ml_fp_rate_target}%)")
        print(f"  * F1-Score:   {cached_metrics['f1']:.1f}%")
        print()
        print("=" * 70)
        
        # Record results for summary table
        record_performance(chip_type, 'ml', cached_metrics['recall'], cached_metrics['fp_rate'], cached_metrics['precision'], cached_metrics['f1'],
                           dataset_id=dataset_id)
        
        # ========================================
        # Diagnostic assertions
        # ========================================
        assert 0.0 <= cached_metrics["recall"] <= 100.0
        assert 0.0 <= cached_metrics["precision"] <= 100.0
        assert 0.0 <= cached_metrics["fp_rate"] <= 100.0
        assert 0.0 <= cached_metrics["f1"] <= 100.0

        if link_stress:
            assert cached_metrics["recall"] > ML_STRESS_REPLAY_GUARDRAIL_RECALL, (
                f"ML weak-link replay recall collapsed for {dataset_id}: "
                f"{cached_metrics['recall']:.1f}% "
                f"(guardrail: >{ML_STRESS_REPLAY_GUARDRAIL_RECALL}%)"
            )
            if cached_metrics["num_baseline"] > 0:
                assert cached_metrics["fp_rate"] < ML_STRESS_REPLAY_GUARDRAIL_FP_RATE, (
                    f"ML weak-link replay FP Rate exploded for {dataset_id}: "
                    f"{cached_metrics['fp_rate']:.1f}% "
                    f"(guardrail: <{ML_STRESS_REPLAY_GUARDRAIL_FP_RATE}%)"
                )
            return

        if dataset_role in {"selection", "holdout"}:
            assert cached_metrics["recall"] > ML_RESERVED_REPLAY_GUARDRAIL_RECALL, (
                f"ML reserved replay recall collapsed for {dataset_id}: "
                f"{cached_metrics['recall']:.1f}% "
                f"(guardrail: >{ML_RESERVED_REPLAY_GUARDRAIL_RECALL}%)"
            )
            if cached_metrics["num_baseline"] > 0:
                assert cached_metrics["fp_rate"] < ML_RESERVED_REPLAY_GUARDRAIL_FP_RATE, (
                    f"ML reserved replay FP Rate exploded for {dataset_id}: "
                    f"{cached_metrics['fp_rate']:.1f}% "
                    f"(guardrail: <{ML_RESERVED_REPLAY_GUARDRAIL_FP_RATE}%)"
                )

    @pytest.mark.parametrize("empty_dataset_path", get_available_empty_datasets())
    def test_ml_empty_false_positive_rate(self, empty_dataset_path):
        """Validate that empty-room recordings stay below the ML FP target."""
        from config import DEFAULT_SUBCARRIERS

        result = _compute_ml_empty_fp_result(
            empty_dataset_path,
            tuple(DEFAULT_SUBCARRIERS),
            DETECTOR_DEFAULT_WINDOW_SIZE,
            0.5,
        )
        fp_rate = result["fp_rate"]
        assert result["eval_count"] > 0
        fp_rate_target = 5.0
        assert fp_rate < fp_rate_target, (
            f"ML empty-room FP Rate too high for {empty_dataset_path.name}: "
            f"{fp_rate:.1f}% (target: <{fp_rate_target}%)"
        )

    @pytest.mark.parametrize("empty_dataset_path", get_available_empty_datasets())
    def test_classic_empty_false_positive_rate(self, empty_dataset_path):
        """Validate that empty-room recordings raise no Classic alarm.

        Empty rooms are the corpus ground truth for "nothing is moving", so this
        is the assertion that has to hold exactly. Static-presence recordings
        cannot serve the same purpose: a stationary person still breathes and
        shifts, and the detector sees it.
        """
        from config import DEFAULT_SUBCARRIERS

        result = _compute_classic_empty_fp_result(
            empty_dataset_path,
            tuple(DEFAULT_SUBCARRIERS),
        )
        assert result, f"Classic startup calibration failed for {empty_dataset_path.name}"
        assert result["eval_count"] > 0
        assert result["effective_alarms"] == 0, (
            f"Classic raised an empty-room alarm for {empty_dataset_path.name}: "
            f"{result['effective_alarms']}"
        )
        # Secondary regression guard on the raw per-evaluation rate. The corpus
        # maximum is 5.14%, so this bounds drift without tracking noise.
        fp_rate_target = 6.0
        assert result["fp_rate"] < fp_rate_target, (
            f"Classic empty-room FP Rate too high for {empty_dataset_path.name}: "
            f"{result['fp_rate']:.1f}% (target: <{fp_rate_target}%)"
        )


# ============================================================================
# Float32 Stability Tests (ESP32 Simulation)
# ============================================================================

class TestFloat32Stability:
    """
    Test numerical stability with float32 precision.
    These tests simulate ESP32 behavior where calculations use 32-bit floats.
    """
    
    def test_turbulence_float32_accuracy(self, real_data, default_subcarriers):
        """Test that float32 turbulence calculation is accurate"""
        static_presence_packets, _ = real_data
        
        max_rel_error = 0.0
        
        for pkt in static_presence_packets[:200]:
            csi_data = pkt['csi_data']
            
            # Float64 reference (Python default)
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            amplitudes_f64 = []
            for sc_idx in default_subcarriers:
                q_idx = sc_idx * 2      # Imaginary first
                i_idx = sc_idx * 2 + 1  # Real second
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amplitudes_f64.append(math.sqrt(I*I + Q*Q))
            turb_f64 = np.std(amplitudes_f64)
            
            # Float32 simulation (ESP32)
            amplitudes_f32 = []
            for sc_idx in default_subcarriers:
                q_idx = sc_idx * 2      # Imaginary first
                i_idx = sc_idx * 2 + 1  # Real second
                I = np.float32(float(csi_data[i_idx]))
                Q = np.float32(float(csi_data[q_idx]))
                amp = np.sqrt(I*I + Q*Q)
                amplitudes_f32.append(float(amp))
            turb_f32 = np.std(np.array(amplitudes_f32, dtype=np.float32))
            
            if turb_f64 > 0.01:  # Avoid division by near-zero
                rel_error = abs(turb_f32 - turb_f64) / turb_f64
                max_rel_error = max(max_rel_error, rel_error)
        
        # Float32 should be accurate within 0.1% for typical CSI values
        assert max_rel_error < 0.001, \
            f"Float32 turbulence error too high: {max_rel_error:.4%}"
    
    def test_variance_two_pass_vs_single_pass_float32(self, real_data, default_subcarriers):
        """Test that two-pass variance is more stable than single-pass with float32"""
        static_presence_packets, _ = real_data
        
        # Generate turbulence values
        turbulences = []
        for pkt in static_presence_packets[:100]:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            turbulences.append(turb)
        
        window = turbulences[:50]
        
        # Reference (float64)
        var_ref = np.var(window)
        
        # Two-pass with float32
        window_f32 = np.array(window, dtype=np.float32)
        mean_f32 = np.mean(window_f32)
        var_two_pass = np.mean((window_f32 - mean_f32) ** 2)
        
        # Single-pass with float32 (E[X²] - E[X]²)
        sum_x = np.float32(0.0)
        sum_sq = np.float32(0.0)
        for x in window_f32:
            sum_x += x
            sum_sq += x * x
        n = np.float32(len(window_f32))
        mean_single = sum_x / n
        var_single_pass = (sum_sq / n) - (mean_single * mean_single)
        
        # Both should be close to reference for normal CSI values
        error_two_pass = abs(var_two_pass - var_ref)
        error_single_pass = abs(var_single_pass - var_ref)
        
        # For normal CSI data, both methods should work
        assert error_two_pass < 0.01, f"Two-pass error too high: {error_two_pass}"
        assert error_single_pass < 0.01, f"Single-pass error too high: {error_single_pass}"
    
    def test_csi_range_values_float32_stable(self):
        """Test that float32 is stable within CSI amplitude range (0-200)"""
        # CSI amplitudes are typically 0-200 range - well within float32 precision
        csi_like_values = [30.0 + i * 0.1 for i in range(50)]  # Typical CSI turbulence
        
        # Reference (float64)
        var_ref = np.var(csi_like_values)
        
        # Two-pass with float32
        values_f32 = np.array(csi_like_values, dtype=np.float32)
        mean_f32 = np.mean(values_f32)
        var_two_pass = float(np.mean((values_f32 - mean_f32) ** 2))
        
        # Single-pass with float32
        sum_x = np.float32(0.0)
        sum_sq = np.float32(0.0)
        for x in values_f32:
            sum_x += x
            sum_sq += x * x
        n = np.float32(len(values_f32))
        mean_single = sum_x / n
        var_single_pass = float((sum_sq / n) - (mean_single * mean_single))
        
        # For CSI-range values, both methods should be accurate
        error_two_pass = abs(var_two_pass - var_ref) / var_ref if var_ref > 0 else 0
        error_single_pass = abs(var_single_pass - var_ref) / var_ref if var_ref > 0 else 0
        
        # Both should work for normal CSI values
        assert error_two_pass < 0.001, \
            f"Two-pass error too high for CSI range: {error_two_pass:.4%}"
        assert error_single_pass < 0.001, \
            f"Single-pass error too high for CSI range: {error_single_pass:.4%}"


# ============================================================================
# End-to-End Tests with Startup Calibration and Normalization
# ============================================================================

class TestEndToEndWithCalibration:
    """
    Test complete pipeline: fixed-band bootstrap -> calibrated Classic detector replay.
    
    These tests verify that the system works end-to-end with:
    - Fixed default subcarriers shared by Classic and ML
    - Adaptive threshold calibration from startup data
    - Production-aligned Classic detector replay achieving target performance
    """
    
    def test_band_calibration_produces_valid_band(self, dataset_config, num_subcarriers, calibration_algorithm, chip_type, default_subcarriers):
        """Test that startup calibration produces a valid fixed band and threshold."""
        
        from config import GUARD_BAND_LOW, GUARD_BAND_HIGH
        
        static_presence_path, _motion_path, _num_sc, _chip, _dataset_id = dataset_config
        
        # HT20 fixed guard bands (64 SC)
        guard_low = GUARD_BAND_LOW
        guard_high = GUARD_BAND_HIGH
        
        # Run calibration with the fixed default subcarriers.
        selected_band, adaptive_threshold = _compute_startup_calibration_result(
            static_presence_path,
            num_subcarriers,
            calibration_algorithm,
            tuple(default_subcarriers),
            None,
        )
        
        # Verify calibration results
        assert selected_band is not None, f"[{calibration_algorithm}] Band calibration failed"
        assert len(selected_band) == 12, f"[{calibration_algorithm}] Expected 12 subcarriers, got {len(selected_band)}"
        
        # All subcarriers should be valid (within valid range for this SC count)
        for sc in selected_band:
            assert guard_low <= sc <= guard_high, \
                f"[{calibration_algorithm}] Subcarrier {sc} outside valid range [{guard_low}-{guard_high}]"
        
        # Adaptive threshold should be valid
        assert adaptive_threshold > 0.0, f"[{calibration_algorithm}] Invalid adaptive threshold: {adaptive_threshold}"
        assert 0.0 <= adaptive_threshold <= 10.0, \
            f"[{calibration_algorithm}] Adaptive threshold out of range: {adaptive_threshold}"
        
        print(f"\n[{calibration_algorithm.upper()}] Startup Calibration Results:")
        print(f"  Selected band: {selected_band}")
        print(f"  Adaptive threshold: {adaptive_threshold:.4f}")
    
    def test_end_to_end_with_band_calibration_and_classic_path(self, dataset_config, real_data, num_subcarriers, window_size, fp_rate_target, recall_target, enable_hampel, calibration_algorithm, chip_type, default_subcarriers):
        """
        Test complete end-to-end flow: startup calibration -> Classic replay.

        This validates the actual production runtime path after startup
        calibration.

        Per-dataset promotion gates live in the aggregate Classic target test.
        This integration check only verifies that the calibrated pipeline
        produces meaningful class separation on each paired replay.
        """
        static_presence_packets, motion_packets = real_data
        static_presence_path, _motion_path, _num_sc, _chip, _dataset_id = dataset_config

        # ========================================
        # Step 1: Fixed-band bootstrap
        # ========================================
        print("\n" + "=" * 70)
        print(f"  END-TO-END TEST: Startup Calibration + Classic path ({num_subcarriers} SC, {calibration_algorithm.upper()})")
        print("=" * 70)
        
        print(f"\nStep 1: {calibration_algorithm.upper()} fixed-band bootstrap...")
        selected_band, adaptive_threshold = _compute_startup_calibration_result(
            static_presence_path,
            num_subcarriers,
            calibration_algorithm,
            tuple(default_subcarriers),
            window_size,
        )
        print(f"  Selected band: {selected_band}")

        assert selected_band is not None, f"[{calibration_algorithm}] Startup calibration failed for {num_subcarriers} SC"
        print(f"  Classic startup threshold: {adaptive_threshold:.4f}")
        
        # ========================================
        # Step 2: Build the production Classic detector with calibration state
        # ========================================
        print(f"\nStep 2: Build calibrated Classic detector (Hampel: {enable_hampel})...")
        calibrated = build_calibrated_classic_detector(
            static_presence_packets,
            selected_subcarriers=tuple(selected_band),
            enable_hampel=enable_hampel,
        )
        assert calibrated is not None, "Classic startup calibration failed"
        detector, calibrated_threshold = calibrated
        print(f"  Classic threshold: {calibrated_threshold:.4f}")

        # ========================================
        # Step 3: Replay baseline and movement through the detector
        # ========================================
        print("\nStep 3: Replay baseline and movement through Classic detector...")
        metrics = evaluate_detector_packets(
            detector,
            static_presence_packets,
            motion_packets,
            selected_band,
        )
        
        print()
        print("=" * 70)
        print("  END-TO-END RESULTS (Startup Calibration + Classic detector)")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({metrics['num_baseline']} baseline + {metrics['num_movement']} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {metrics['tn']:4d} (TN)  {metrics['fp']:4d} (FP)")
        print(f"    MOTION      {metrics['fn']:4d} (FN)  {metrics['tp']:4d} (TP)")
        print()
        print("METRICS:")
        print(f"  * Recall:     {metrics['recall']:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {metrics['precision']:.1f}%")
        print(f"  * FP Rate:    {metrics['fp_rate']:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {metrics['f1']:.1f}%")
        print()
        print("=" * 70)
        
        # ========================================
        # Assertions
        # ========================================
        # This test validates production wiring, not the stricter aggregate
        # promotion thresholds. Require meaningful separation between the idle
        # and motion replays so regressions in calibration or detector replay
        # still fail loudly.
        separation_margin = metrics["recall"] - metrics["fp_rate"]
        assert metrics["tp"] > metrics["fn"], (
            f"End-to-end replay favors IDLE on motion packets ({num_subcarriers} SC): "
            f"tp={metrics['tp']} fn={metrics['fn']}"
        )
        assert metrics["tn"] > metrics["fp"], (
            f"End-to-end replay favors MOTION on baseline packets ({num_subcarriers} SC): "
            f"tn={metrics['tn']} fp={metrics['fp']}"
        )
        assert separation_margin >= 10.0, (
            f"End-to-end separation too small ({num_subcarriers} SC): "
            f"recall={metrics['recall']:.1f}% fp={metrics['fp_rate']:.1f}% "
            f"margin={separation_margin:.1f}pp (target: >=10.0pp)"
        )


@pytest.mark.parametrize("chip", get_available_chip_types())
def test_classic_chip_aggregate_targets(chip):
    """Gate Classic on the aggregate normal-link metrics published in PERFORMANCE.md."""
    fp_rate_target = get_classic_fp_rate_target(chip)
    recall_target = get_classic_recall_target(chip)
    chip_pairs = []
    for static_path, motion_path, _num_sc, dataset_chip, dataset_id in (
        _shared_get_available_paired_datasets(synthetic=False)
    ):
        if str(dataset_chip).upper() != str(chip).upper():
            continue
        # Real weak-link pairs are stress diagnostics; the Classic promotion
        # aggregate covers normal-link sessions only.
        if _is_low_rssi_paired_dataset(static_path):
            continue
        chip_pairs.append((static_path, motion_path, dataset_id))

    assert chip_pairs, f"No paired datasets found for chip {chip}"

    total_tp = total_fn = total_fp = total_baseline = 0
    for static_path, motion_path, _dataset_id in chip_pairs:
        cached_result = _compute_classic_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            DETECTOR_DEFAULT_WINDOW_SIZE,
        )
        assert cached_result is not None, f"Classic startup calibration failed for {static_path.name}"
        _adaptive_threshold, metrics = cached_result
        total_tp += metrics["tp"]
        total_fn += metrics["fn"]
        total_fp += metrics["fp"]
        total_baseline += metrics["num_baseline"]

    recall = total_tp / (total_tp + total_fn) * 100.0 if (total_tp + total_fn) > 0 else 0.0
    fp_rate = total_fp / total_baseline * 100.0 if total_baseline > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) * 100.0 if (total_tp + total_fp) > 0 else 0.0
    f1 = (
        2 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall) > 0
        else 0.0
    )

    print("\n")
    print("=" * 70)
    print("              CLASSIC CHIP-AGGREGATE TEST SUMMARY")
    print("=" * 70)
    print(f"Chip:        {chip}")
    print(f"Datasets:    {len(chip_pairs)}")
    print(f"Recall:      {recall:.1f}% (target: >{recall_target}%)")
    print(f"Precision:   {precision:.1f}%")
    print(f"FP Rate:     {fp_rate:.1f}% (target: <{fp_rate_target}%)")
    print(f"F1-Score:    {f1:.1f}%")
    print("=" * 70)

    assert recall > recall_target, (
        f"Classic aggregate recall too low for {chip}: {recall:.1f}% (target: >{recall_target}%)"
    )
    assert fp_rate < fp_rate_target, (
        f"Classic aggregate FP Rate too high for {chip}: {fp_rate:.1f}% (target: <{fp_rate_target}%)"
    )


@pytest.mark.parametrize("chip", get_available_chip_types())
def test_ml_chip_aggregate_reserved_targets(chip):
    """Gate ML on aggregate reserved normal-link replays."""
    chip_pairs = []
    for static_path, motion_path, _num_sc, dataset_chip, dataset_id in (
        _shared_get_available_paired_datasets(synthetic=False)
    ):
        if str(dataset_chip).upper() != str(chip).upper():
            continue
        if _is_low_rssi_paired_dataset(static_path):
            continue
        dataset_role = _get_paired_dataset_role(static_path)
        if dataset_role not in {"selection", "holdout"}:
            continue
        chip_pairs.append((static_path, motion_path, dataset_id))

    if not chip_pairs:
        pytest.skip(f"No reserved normal-link paired datasets found for chip {chip}")

    total_tp = total_fn = total_fp = total_baseline = 0
    for static_path, motion_path, _dataset_id in chip_pairs:
        cached_metrics, _feature_payload = _compute_ml_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
            threshold=0.5,
        )
        total_tp += cached_metrics["tp"]
        total_fn += cached_metrics["fn"]
        total_fp += cached_metrics["fp"]
        total_baseline += cached_metrics["num_baseline"]

    total_motion = total_tp + total_fn
    assert total_motion > 0, f"No movement evaluations aggregated for reserved ML chip {chip}"
    recall = total_tp / total_motion * 100.0
    fp_rate = total_fp / total_baseline * 100.0 if total_baseline > 0 else 0.0
    ml_recall_target = 95.0
    ml_fp_rate_target = 5.0
    assert recall > ml_recall_target, (
        f"ML aggregate reserved recall too low for {chip}: {recall:.1f}% "
        f"(target: >{ml_recall_target}%)"
    )
    assert fp_rate < ml_fp_rate_target, (
        f"ML aggregate reserved FP Rate too high for {chip}: {fp_rate:.1f}% "
        f"(target: <{ml_fp_rate_target}%)"
    )


@pytest.mark.parametrize("chip", get_available_chip_types())
def test_ml_chip_aggregate_stress_targets(chip):
    """Gate ML weak-link stress replays on aggregate relaxed targets."""
    chip_pairs = []
    for static_path, motion_path, _num_sc, dataset_chip, dataset_id in (
        _shared_get_available_paired_datasets(synthetic=False)
    ):
        if str(dataset_chip).upper() != str(chip).upper():
            continue
        if not _is_low_rssi_paired_dataset(static_path):
            continue
        chip_pairs.append((static_path, motion_path, dataset_id))

    if not chip_pairs:
        pytest.skip(f"No weak-link paired datasets found for chip {chip}")

    total_tp = total_fn = total_fp = total_baseline = 0
    for static_path, motion_path, _dataset_id in chip_pairs:
        cached_metrics, _feature_payload = _compute_ml_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
            threshold=0.5,
        )
        total_tp += cached_metrics["tp"]
        total_fn += cached_metrics["fn"]
        total_fp += cached_metrics["fp"]
        total_baseline += cached_metrics["num_baseline"]

    total_motion = total_tp + total_fn
    assert total_motion > 0, f"No movement evaluations aggregated for weak-link ML chip {chip}"
    recall = total_tp / total_motion * 100.0
    fp_rate = total_fp / total_baseline * 100.0 if total_baseline > 0 else 0.0
    assert recall > STRESS_TARGET_RECALL, (
        f"ML aggregate weak-link recall too low for {chip}: {recall:.1f}% "
        f"(target: >{STRESS_TARGET_RECALL}%)"
    )
    assert fp_rate < STRESS_TARGET_FP_RATE, (
        f"ML aggregate weak-link FP Rate too high for {chip}: {fp_rate:.1f}% "
        f"(target: <{STRESS_TARGET_FP_RATE}%)"
    )
