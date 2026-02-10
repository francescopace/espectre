"""
Micro-ESPectre - Test Fixtures

Pytest fixtures for CSI data, configuration, and test utilities.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import math
import pytest
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add src and tools to path for imports
# src is inserted last (position 0) so it takes precedence for config imports
SRC_PATH = Path(__file__).parent.parent / 'src'
TOOLS_PATH = Path(__file__).parent.parent / 'tools'
sys.path.insert(0, str(TOOLS_PATH))
sys.path.insert(0, str(SRC_PATH))

# Data directory (shared between tests and tools)
DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_subcarriers(request):
    """
    Optimal subcarrier band for testing (HT20: 64 SC only).
    
    Matches C++ test configuration exactly (test_motion_detection.cpp).
    
    SUBCARRIERS_ESP32_64SC = {12, 13, 14, 17, 44, 45, 46, 48, 49, 50, 51, 52}
    SUBCARRIERS_C3_64SC = {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
    SUBCARRIERS_C6_64SC = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    SUBCARRIERS_S3_64SC = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}
    """
    try:
        chip_type = request.getfixturevalue('chip_type')
        if chip_type == 'C3':
            return [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        if chip_type == 'C6':
            return [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        if chip_type == 'S3':
            return [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        if chip_type == 'ESP32':
            return [12, 13, 14, 17, 44, 45, 46, 48, 49, 50, 51, 52]
        raise ValueError(f"Unknown chip type: {chip_type}. Add subcarrier config for this chip.")
    except pytest.FixtureLookupError:
        # No chip_type fixture available, use C6 default for backward compatibility
        return [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


@pytest.fixture
def segmentation_config():
    """Default segmentation configuration - matches C++ DETECTOR_DEFAULT_WINDOW_SIZE"""
    return {
        'window_size': 75,  # DETECTOR_DEFAULT_WINDOW_SIZE
        'threshold': 1.0,
        'enable_hampel': False,
        'hampel_window': 7,
        'hampel_threshold': 4.0,
    }


@pytest.fixture
def hampel_config():
    """Default Hampel filter configuration"""
    return {
        'window_size': 7,
        'threshold': 4.0,
    }


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def constant_values():
    """Constant value test data"""
    return [5.0] * 500


@pytest.fixture
def linear_ramp():
    """Linear ramp test data"""
    return [float(i) for i in range(500)]


@pytest.fixture
def sine_wave():
    """Sine wave test data"""
    return [math.sin(i * 0.1) * 10 + 50 for i in range(500)]


@pytest.fixture
def random_uniform():
    """Random uniform distribution test data"""
    np.random.seed(42)  # Reproducible
    return list(np.random.uniform(0, 100, 500))


@pytest.fixture
def random_normal():
    """Random normal distribution test data"""
    np.random.seed(42)  # Reproducible
    return list(np.random.normal(50, 15, 500))


@pytest.fixture
def step_function():
    """Step function test data"""
    return [10.0] * 250 + [90.0] * 250


@pytest.fixture
def impulse_data():
    """Impulse/spike test data"""
    return [50.0] * 200 + [200.0] + [50.0] * 299


@pytest.fixture
def synthetic_turbulence_baseline():
    """Simulated baseline turbulence (low variance)"""
    np.random.seed(42)
    return list(np.random.normal(5.0, 0.5, 500))


@pytest.fixture
def synthetic_turbulence_movement():
    """Simulated movement turbulence (high variance)"""
    np.random.seed(42)
    return list(np.random.normal(10.0, 3.0, 500))


# ============================================================================
# CSI Data Fixtures
# ============================================================================

@pytest.fixture
def synthetic_csi_packet():
    """Generate a synthetic CSI packet (64 subcarriers, I/Q pairs)"""
    np.random.seed(42)
    # Generate I/Q values as int8 (range -128 to 127)
    iq_data = np.random.randint(-50, 50, size=128, dtype=np.int8)
    return iq_data


@pytest.fixture
def synthetic_csi_baseline_packets():
    """Generate synthetic baseline CSI packets (stable signal)"""
    np.random.seed(42)
    packets = []
    for i in range(100):
        # Stable signal with small variations
        base_amplitude = 30
        iq_data = np.zeros(128, dtype=np.int8)
        for sc in range(64):
            I = int(base_amplitude + np.random.normal(0, 2))
            Q = int(base_amplitude * 0.3 + np.random.normal(0, 2))
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            iq_data[sc * 2] = np.clip(Q, -127, 127)      # Imaginary first
            iq_data[sc * 2 + 1] = np.clip(I, -127, 127)  # Real second
        packets.append({'csi_data': iq_data, 'label': 'baseline'})
    return packets


@pytest.fixture
def synthetic_csi_movement_packets():
    """Generate synthetic movement CSI packets (variable signal)"""
    np.random.seed(43)
    packets = []
    for i in range(100):
        # Variable signal with larger variations
        base_amplitude = 25 + np.random.uniform(-10, 10)
        iq_data = np.zeros(128, dtype=np.int8)
        for sc in range(64):
            I = int(base_amplitude + np.random.normal(0, 8))
            Q = int(base_amplitude * 0.3 + np.random.normal(0, 8))
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            iq_data[sc * 2] = np.clip(Q, -127, 127)      # Imaginary first
            iq_data[sc * 2 + 1] = np.clip(I, -127, 127)  # Real second
        packets.append({'csi_data': iq_data, 'label': 'movement'})
    return packets


# ============================================================================
# Real CSI Data Fixtures (optional - skip if not available)
# ============================================================================

@pytest.fixture
def real_csi_data_available():
    """Check if real CSI data files are available"""
    from csi_utils import find_dataset
    try:
        find_dataset(chip='C6')
        return True
    except FileNotFoundError:
        return False


@pytest.fixture
def real_baseline_packets(real_csi_data_available):
    """Load real baseline CSI packets (skip if not available)"""
    if not real_csi_data_available:
        pytest.skip("Real CSI data not available")
    
    from csi_utils import load_baseline_and_movement
    baseline, _ = load_baseline_and_movement()
    return baseline


@pytest.fixture
def real_movement_packets(real_csi_data_available):
    """Load real movement CSI packets (skip if not available)"""
    if not real_csi_data_available:
        pytest.skip("Real CSI data not available")
    
    from csi_utils import load_baseline_and_movement
    _, movement = load_baseline_and_movement()
    return movement


@pytest.fixture
def real_turbulence_values(real_csi_data_available, default_subcarriers):
    """Calculate turbulence values from real CSI data"""
    if not real_csi_data_available:
        pytest.skip("Real CSI data not available")
    
    from csi_utils import load_baseline_and_movement, calculate_spatial_turbulence
    
    baseline, movement = load_baseline_and_movement()
    turbulence_values = []
    
    for packet in baseline:
        turbulence = calculate_spatial_turbulence(packet['csi_data'], default_subcarriers)
        turbulence_values.append(float(turbulence))
    
    for packet in movement:
        turbulence = calculate_spatial_turbulence(packet['csi_data'], default_subcarriers)
        turbulence_values.append(float(turbulence))
    
    return turbulence_values


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def tolerance():
    """Standard tolerance for floating point comparisons"""
    return 1e-6


# ============================================================================
# Performance Results Collection (for summary table)
# ============================================================================

import json
import tempfile
import os

# Use a temp file to share results between test module and conftest hook
_PERF_RESULTS_FILE = os.path.join(tempfile.gettempdir(), 'espectre_perf_results.json')


def record_performance(chip: str, algorithm: str, recall: float, fp_rate: float,
                       precision: float = 0.0, f1: float = 0.0):
    """
    Record performance metrics for the summary table.
    
    Args:
        chip: Chip type (C3, C6, ESP32, S3)
        algorithm: Algorithm name (mvs_optimal, mvs_nbvi, ml)
        recall: Recall percentage
        fp_rate: False positive rate percentage
        precision: Precision percentage
        f1: F1-score percentage
    """
    # Load existing results
    results = {}
    if os.path.exists(_PERF_RESULTS_FILE):
        try:
            with open(_PERF_RESULTS_FILE, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError):
            results = {}
    
    # Add new result
    if chip not in results:
        results[chip] = {}
    results[chip][algorithm] = {
        'recall': recall,
        'fp_rate': fp_rate,
        'precision': precision,
        'f1': f1
    }
    
    # Save
    with open(_PERF_RESULTS_FILE, 'w') as f:
        json.dump(results, f)


def pytest_configure(config):
    """Clear performance results at the start of test session."""
    if os.path.exists(_PERF_RESULTS_FILE):
        os.remove(_PERF_RESULTS_FILE)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print performance summary table at the end of test session."""
    if not os.path.exists(_PERF_RESULTS_FILE):
        return
    
    try:
        with open(_PERF_RESULTS_FILE, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError):
        return
    
    if not results:
        return
    
    terminalreporter.write_line("")
    terminalreporter.write_line("=" * 105)
    terminalreporter.write_line("                              PERFORMANCE SUMMARY TABLE (Python)")
    terminalreporter.write_line("=" * 105)
    terminalreporter.write_line("")
    terminalreporter.write_line("| Chip   | MVS Optimal             | MVS + NBVI              | ML                      |")
    terminalreporter.write_line("|--------|-------------------------|-------------------------|-------------------------|")
    
    # Sort chips for consistent output
    for chip in ['C3', 'C6', 'ESP32', 'S3']:
        if chip not in results:
            continue
        
        chip_results = results[chip]
        
        # MVS Optimal
        if 'mvs_optimal' in chip_results:
            mvs_opt = chip_results['mvs_optimal']
            mvs_opt_str = f"{mvs_opt['recall']:.1f}% R, {mvs_opt['fp_rate']:.1f}% FP"
        else:
            mvs_opt_str = "N/A"
        
        # MVS + NBVI
        if 'mvs_nbvi' in chip_results:
            mvs = chip_results['mvs_nbvi']
            mvs_str = f"{mvs['recall']:.1f}% R, {mvs['fp_rate']:.1f}% FP"
        else:
            mvs_str = "N/A"
        
        # ML
        if 'ml' in chip_results:
            ml = chip_results['ml']
            ml_str = f"{ml['recall']:.1f}% R, {ml['fp_rate']:.1f}% FP"
        else:
            ml_str = "N/A"
        
        terminalreporter.write_line(f"| {chip:<6} | {mvs_opt_str:<23} | {mvs_str:<23} | {ml_str:<23} |")
    
    terminalreporter.write_line("")
    terminalreporter.write_line("Legend: R = Recall, FP = False Positive Rate")
    terminalreporter.write_line("Targets: MVS Recall >97%, ML Recall >93%, FP Rate <10%")
    terminalreporter.write_line("=" * 105)
    
    # Detailed table for PERFORMANCE.md
    terminalreporter.write_line("")
    terminalreporter.write_line("                         DETAILED METRICS (for PERFORMANCE.md)")
    terminalreporter.write_line("-" * 105)
    terminalreporter.write_line("| Chip   | Algorithm   | Recall  | Precision | FP Rate | F1-Score |")
    terminalreporter.write_line("|--------|-------------|---------|-----------|---------|----------|")
    
    for chip in ['C3', 'C6', 'ESP32', 'S3']:
        if chip not in results:
            continue
        
        chip_results = results[chip]
        
        for algo_key, algo_name in [('mvs_optimal', 'MVS Optimal'), ('mvs_nbvi', 'MVS + NBVI'), ('ml', 'ML')]:
            if algo_key in chip_results:
                r = chip_results[algo_key]
                terminalreporter.write_line(
                    f"| {chip:<6} | {algo_name:<11} | {r['recall']:>6.1f}% | {r.get('precision', 0):>8.1f}% | {r['fp_rate']:>6.1f}% | {r.get('f1', 0):>7.1f}% |"
                )
    
    terminalreporter.write_line("-" * 105)
    
    # Cleanup
    os.remove(_PERF_RESULTS_FILE)

