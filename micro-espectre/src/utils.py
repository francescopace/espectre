"""
Micro-ESPectre - Utility Functions

Shared utility functions used across multiple modules.
Mirrors utils.h from ESPectre C++ implementation.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math

HT20_CSI_LEN = 128
HT20_CSI_LEN_SHORT = 114
HT20_CSI_LEN_SHORT_DOUBLE = 228
HT20_CSI_SHORT_LEFT_PAD = 8
HT20_CSI_SHORT_COPY_END = HT20_CSI_SHORT_LEFT_PAD + HT20_CSI_LEN_SHORT
HT20_CSI_SHORT_RIGHT_PAD = HT20_CSI_LEN - HT20_CSI_SHORT_COPY_END
HT20_CSI_SHORT_LEFT_ZEROS = b"\x00" * HT20_CSI_SHORT_LEFT_PAD
HT20_CSI_SHORT_RIGHT_ZEROS = b"\x00" * HT20_CSI_SHORT_RIGHT_PAD


def normalize_ht20_csi_payload(csi_data, expected_len=128, chip_type=None, remap_buffer=None):
    """
    Normalize CSI payload length to HT20 expected layout.

    Handles:
    - Doubled HT payload (2x expected): keep first HT-LTF block
    - Doubled short HT payload (228 bytes): collapse to 114 bytes
    - Short HT payload (114 bytes): remap to 128 bytes with guard padding

    Args:
        csi_data: Raw CSI payload (bytes/bytearray)
        expected_len: Target HT20 payload length (default: 128)
        chip_type: Optional chip hint ('C5', 'ESP32-C5', etc.)
        remap_buffer: Optional pre-allocated bytearray(expected_len) reused by caller

    Returns:
        tuple: (normalized_payload, raw_len, remap_tag)
            - normalized_payload: bytes-like object or None if unsupported length
            - raw_len: original payload length
            - remap_tag: None | 'double_ht20' | 'double_ht57_and_remap' | 'ht57_to_64'
    """
    raw_len = len(csi_data)
    input_len = raw_len

    # STBC workaround: two HT-LTF blocks, keep first 64 SC (HT20).
    if raw_len == expected_len * 2:
        return csi_data[:expected_len], input_len, 'double_ht20'

    if raw_len == expected_len:
        return csi_data, input_len, None

    # Best-effort fallback for short HT estimates:
    # - 228 bytes can represent 2 x 114-byte short HT blocks (collapse to first block)
    # - 114 bytes represents 57 complex samples that we remap to HT20 (128 bytes)
    if expected_len != HT20_CSI_LEN:
        return None, input_len, None

    short_double_collapsed = False
    if raw_len == HT20_CSI_LEN_SHORT_DOUBLE:
        csi_data = csi_data[:HT20_CSI_LEN_SHORT]
        raw_len = HT20_CSI_LEN_SHORT
        short_double_collapsed = True

    if raw_len == HT20_CSI_LEN_SHORT:
        if remap_buffer is None or len(remap_buffer) != expected_len:
            remap_buffer = bytearray(expected_len)

        # Clear left/right guard bins and copy payload in the middle.
        remap_buffer[:HT20_CSI_SHORT_LEFT_PAD] = HT20_CSI_SHORT_LEFT_ZEROS
        remap_buffer[HT20_CSI_SHORT_COPY_END:] = HT20_CSI_SHORT_RIGHT_ZEROS
        remap_buffer[HT20_CSI_SHORT_LEFT_PAD:HT20_CSI_SHORT_COPY_END] = csi_data
        if short_double_collapsed:
            return remap_buffer, input_len, 'double_ht57_and_remap'
        return remap_buffer, input_len, 'ht57_to_64'

    return None, input_len, None


def to_signed_int8(value):
    """
    Convert unsigned byte to signed int8.
    
    Used for CSI I/Q values and AGC/FFT gain data, which are
    stored as unsigned bytes but represent signed values.
    
    Args:
        value: Unsigned byte value (0-255)
    
    Returns:
        int: Signed value (-128 to 127)
    """
    return value if value < 128 else value - 256


def calculate_median(values):
    """
    Calculate median of a list.
    
    Note: This function sorts the input list in-place for efficiency
    (avoids memory allocation on MicroPython).
    
    Args:
        values: List of numeric values (will be sorted in-place)
    
    Returns:
        Median value (0 if empty, integer for int lists)
    """
    if not values:
        return 0
    values.sort()
    n = len(values)
    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) // 2
    return values[n // 2]


def insertion_sort(arr, n):
    """
    In-place insertion sort for small arrays.
    
    Faster than Python's Timsort for N < 10-15 elements
    due to lower overhead (no function calls, cache-friendly).
    Also used for N=50 on ESP32 where it's acceptable.
    
    Args:
        arr: Array to sort (modified in place)
        n: Number of elements to sort
    """
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def calculate_percentile(values, percentile):
    """
    Calculate percentile value from a list.
    
    Uses linear interpolation between adjacent values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        float: Percentile value (0.0 if list is empty, matching C++ NBVICalibrator)
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    p = percentile / 100.0
    k = int((n - 1) * p)
    
    if k >= n - 1:
        return sorted_values[-1]
    
    # Linear interpolation
    frac = (n - 1) * p - k
    return sorted_values[k] * (1 - frac) + sorted_values[k + 1] * frac


def calculate_variance(values):
    """
    Calculate variance using two-pass algorithm (numerically stable).
    
    Two-pass algorithm: variance = sum((x - mean)^2) / n
    More stable than single-pass E[X²] - E[X]² for float arithmetic.
    
    Args:
        values: List of numeric values
    
    Returns:
        float: Variance (0.0 if empty)
    """
    if not values:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance


def calculate_std(values):
    """
    Calculate standard deviation.
    
    Args:
        values: List of numeric values
    
    Returns:
        float: Standard deviation (0.0 if empty)
    """
    var = calculate_variance(values)
    return math.sqrt(var) if var > 0 else 0.0


def calculate_magnitude(i, q):
    """
    Calculate magnitude (amplitude) from I/Q components.
    
    Args:
        i: In-phase component (int8)
        q: Quadrature component (int8)
    
    Returns:
        float: Magnitude = sqrt(I² + Q²)
    """
    fi = float(i)
    fq = float(q)
    return math.sqrt(fi * fi + fq * fq)


def calculate_spatial_turbulence(magnitudes, band, use_cv_normalization=True):
    """
    Calculate spatial turbulence from magnitudes.
    
    Two modes:
    - CV normalization (std/mean): gain-invariant, used when gain is NOT locked.
      If AGC scales all amplitudes by factor k, std(kA)/mean(kA) = std(A)/mean(A).
    - Raw std: better sensitivity for contiguous bands, used when gain IS locked.
    
    Args:
        magnitudes: List of magnitude values (one per subcarrier)
        band: List of subcarrier indices to use
        use_cv_normalization: True = std/mean, False = raw std (default: True)
    
    Returns:
        float: Turbulence value (0.0 if no valid subcarriers)
    """
    band_mags = [magnitudes[sc] for sc in band if sc < len(magnitudes)]
    
    if not band_mags:
        return 0.0
    
    std = calculate_std(band_mags)
    if use_cv_normalization:
        mean = sum(band_mags) / len(band_mags)
        return std / mean if mean > 0 else 0.0
    else:
        return std


def calculate_moving_variance(values, window_size=75):
    """
    Calculate moving variance series.
    
    For each position, calculates variance of the previous window_size values.
    
    Args:
        values: List of numeric values
        window_size: Size of sliding window (default: 75, matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)
    
    Returns:
        list: Moving variance series (length = len(values) - window_size)
    """
    if len(values) < window_size:
        return []
    
    variances = []
    for i in range(window_size, len(values)):
        window = values[i-window_size:i]
        variances.append(calculate_variance(window))
    
    return variances


# =============================================================================
# CSI I/Q Parsing Functions
# =============================================================================

def extract_amplitude(csi_data, sc_idx):
    """
    Extract amplitude for a single subcarrier from CSI data.
    
    Uses Espressif CSI format: [Imaginary, Real, ...] per subcarrier.
    CSI values are signed int8 stored as uint8.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
        sc_idx: Subcarrier index (0-63 for HT20)
    
    Returns:
        float: Amplitude (magnitude) value, or 0.0 if invalid index
    """
    i_idx = sc_idx * 2 + 1  # Real (In-phase) is second
    q_idx = sc_idx * 2      # Imaginary (Quadrature) is first
    
    if q_idx + 1 >= len(csi_data):
        return 0.0
    
    # Convert to signed int8
    I = to_signed_int8(csi_data[i_idx])
    Q = to_signed_int8(csi_data[q_idx])
    
    return math.sqrt(float(I * I + Q * Q))


def extract_amplitudes(csi_data, subcarriers=None):
    """
    Extract amplitudes for multiple subcarriers from CSI data.
    
    Uses Espressif CSI format: [Imaginary, Real, ...] per subcarrier.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
        subcarriers: List of subcarrier indices (default: all 64)
    
    Returns:
        list: Amplitude values for each subcarrier
    """
    if subcarriers is None:
        # Use all available subcarriers (HT20: 64 max)
        max_sc = min(64, len(csi_data) // 2)
        subcarriers = range(max_sc)
    
    amplitudes = []
    for sc_idx in subcarriers:
        amp = extract_amplitude(csi_data, sc_idx)
        if amp > 0.0 or sc_idx * 2 + 1 < len(csi_data):
            amplitudes.append(amp)
    
    return amplitudes


def extract_all_magnitudes(csi_data):
    """
    Extract magnitudes for ALL subcarriers from CSI data.
    
    Returns a list indexed by subcarrier number (0-63 for HT20).
    This is useful when you need to access magnitudes by subcarrier index.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
    
    Returns:
        list: Magnitudes indexed by subcarrier (length = num_subcarriers)
    """
    num_sc = min(64, len(csi_data) // 2)
    magnitudes = [0.0] * num_sc
    
    for sc_idx in range(num_sc):
        magnitudes[sc_idx] = extract_amplitude(csi_data, sc_idx)
    
    return magnitudes


def extract_phase(csi_data, sc_idx):
    """
    Extract phase for a single subcarrier from CSI data.
    
    Uses Espressif CSI format: [Imaginary, Real, ...] per subcarrier.
    CSI values are signed int8 stored as uint8.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
        sc_idx: Subcarrier index (0-63 for HT20)
    
    Returns:
        float: Phase value in radians (-pi to pi), or 0.0 if invalid index
    """
    i_idx = sc_idx * 2 + 1  # Real (In-phase) is second
    q_idx = sc_idx * 2      # Imaginary (Quadrature) is first
    
    if q_idx + 1 >= len(csi_data):
        return 0.0
    
    I = to_signed_int8(csi_data[i_idx])
    Q = to_signed_int8(csi_data[q_idx])
    
    return math.atan2(float(Q), float(I))


def extract_phases(csi_data, subcarriers=None):
    """
    Extract phases for multiple subcarriers from CSI data.
    
    Uses Espressif CSI format: [Imaginary, Real, ...] per subcarrier.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
        subcarriers: List of subcarrier indices (default: all 64)
    
    Returns:
        list: Phase values in radians for each subcarrier
    """
    if subcarriers is None:
        # Use all available subcarriers (HT20: 64 max)
        max_sc = min(64, len(csi_data) // 2)
        subcarriers = range(max_sc)
    
    phases = []
    for sc_idx in subcarriers:
        if sc_idx * 2 + 1 < len(csi_data):
            phases.append(extract_phase(csi_data, sc_idx))
    
    return phases


def extract_amplitudes_and_phases(csi_data, subcarriers=None):
    """
    Extract both amplitudes and phases for subcarriers from CSI data.
    
    More efficient than calling extract_amplitudes and extract_phases separately
    since it only parses I/Q once per subcarrier.
    
    Args:
        csi_data: Raw CSI data (bytes or list of uint8)
        subcarriers: List of subcarrier indices (default: all 64)
    
    Returns:
        tuple: (amplitudes, phases) lists
    """
    if subcarriers is None:
        max_sc = min(64, len(csi_data) // 2)
        subcarriers = range(max_sc)
    
    amplitudes = []
    phases = []
    
    for sc_idx in subcarriers:
        i_idx = sc_idx * 2 + 1  # Real (In-phase) is second
        q_idx = sc_idx * 2      # Imaginary (Quadrature) is first
        
        if q_idx + 1 >= len(csi_data):
            continue
        
        I = float(to_signed_int8(csi_data[i_idx]))
        Q = float(to_signed_int8(csi_data[q_idx]))
        
        amplitudes.append(math.sqrt(I * I + Q * Q))
        phases.append(math.atan2(Q, I))
    
    return amplitudes, phases
