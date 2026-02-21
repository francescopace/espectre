"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts statistical features from turbulence buffer and subcarrier amplitudes
for ML-based motion detection.

Feature design principles:
  - No redundant features (each adds unique information)
  - Turbulence-based statistical features from sliding window (stable estimates)
  - Turbulence-based temporal features (multi-lag autocorrelation from Wi-Limb paper)
  - Amplitude features from current packet (cross-subcarrier entropy)
  - Phase features from current packet (phase difference variance)
  - Energy features (FFT-based) from CSI-F paper (experimental)
  - Spectral features (centroid, flatness) from CSI-HC paper (experimental)
  - MicroPython compatible (no numpy at runtime)
  - Optimized for motion detection separability

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.utils import insertion_sort
except ImportError:
    from utils import insertion_sort


# ============================================================================
# FFT Implementation (Pure Python, MicroPython compatible)
# ============================================================================

def _fft_real(x):
    """
    Compute FFT of real-valued signal using Cooley-Tukey algorithm.
    
    Returns magnitude spectrum (positive frequencies only).
    Pads input to next power of 2 for efficiency.
    
    Args:
        x: List of real values
    
    Returns:
        list: Magnitude values for positive frequencies (length n//2)
    """
    n = len(x)
    if n < 2:
        return [0.0]
    
    # Pad to next power of 2
    n_fft = 1
    while n_fft < n:
        n_fft *= 2
    
    # Zero-pad input
    x_padded = list(x) + [0.0] * (n_fft - n)
    
    # Bit-reversal permutation
    real = x_padded[:]
    imag = [0.0] * n_fft
    
    j = 0
    for i in range(n_fft):
        if i < j:
            real[i], real[j] = real[j], real[i]
        k = n_fft // 2
        while k >= 1 and j >= k:
            j -= k
            k //= 2
        j += k
    
    # Cooley-Tukey FFT
    m = 1
    while m < n_fft:
        m2 = m * 2
        theta = -math.pi / m
        w_real = 1.0
        w_imag = 0.0
        w_step_real = math.cos(theta)
        w_step_imag = math.sin(theta)
        
        for k in range(m):
            for i in range(k, n_fft, m2):
                j = i + m
                t_real = w_real * real[j] - w_imag * imag[j]
                t_imag = w_real * imag[j] + w_imag * real[j]
                real[j] = real[i] - t_real
                imag[j] = imag[i] - t_imag
                real[i] = real[i] + t_real
                imag[i] = imag[i] + t_imag
            
            # Update twiddle factor
            new_w_real = w_real * w_step_real - w_imag * w_step_imag
            w_imag = w_real * w_step_imag + w_imag * w_step_real
            w_real = new_w_real
        
        m = m2
    
    # Compute magnitudes (positive frequencies only)
    n_pos = n_fft // 2
    magnitudes = [0.0] * n_pos
    for i in range(n_pos):
        magnitudes[i] = math.sqrt(real[i] * real[i] + imag[i] * imag[i])
    
    return magnitudes


# ============================================================================
# Turbulence Buffer Features
# ============================================================================

def calc_skewness(values, count, mean, std):
    """
    Calculate Fisher's skewness (third standardized moment).
    
    Skewness measures asymmetry of the distribution:
    - gamma1 > 0: Right-skewed (tail on right)
    - gamma1 < 0: Left-skewed (tail on left)
    - gamma1 = 0: Symmetric
    
    Args:
        values: List of values
        count: Number of valid values
        mean: Pre-computed mean
        std: Pre-computed standard deviation
    
    Returns:
        float: Skewness coefficient
    """
    if count < 3 or std < 1e-10:
        return 0.0
    
    # Third central moment
    m3 = 0.0
    for i in range(count):
        diff = values[i] - mean
        m3 += diff * diff * diff
    m3 /= count
    
    return m3 / (std * std * std)


def calc_kurtosis(values, count, mean, std):
    """
    Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
    
    Kurtosis measures "tailedness" of the distribution:
    - gamma2 > 0: Leptokurtic (heavy tails, sharp peak)
    - gamma2 < 0: Platykurtic (light tails, flat peak)
    - gamma2 = 0: Mesokurtic (normal distribution)
    
    Args:
        values: List of values
        count: Number of valid values
        mean: Pre-computed mean
        std: Pre-computed standard deviation
    
    Returns:
        float: Excess kurtosis coefficient
    """
    if count < 4 or std < 1e-10:
        return 0.0
    
    # Fourth central moment
    m4 = 0.0
    for i in range(count):
        diff = values[i] - mean
        diff2 = diff * diff
        m4 += diff2 * diff2
    m4 /= count
    
    # Excess kurtosis (subtract 3 for normal distribution baseline)
    std4 = std * std * std * std
    return (m4 / std4) - 3.0


def calc_entropy_turb(turbulence_buffer, buffer_count, n_bins=10):
    """
    Calculate Shannon entropy of turbulence distribution.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        n_bins: Number of histogram bins
    
    Returns:
        float: Shannon entropy in bits
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max
    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    # Create histogram
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for i in range(buffer_count):
        val = turbulence_buffer[i]
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    # Calculate entropy
    entropy = 0.0
    log2 = math.log(2)
    for count in bins:
        if count > 0:
            p = count / buffer_count
            entropy -= p * math.log(p) / log2
    
    return entropy


def calc_zero_crossing_rate(turbulence_buffer, buffer_count, mean=None):
    """
    Calculate zero-crossing rate around the mean.
    
    Counts the fraction of consecutive samples where the signal crosses
    the mean value. High ZCR indicates rapid oscillations (motion),
    low ZCR indicates stable signal (idle).
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        mean: Pre-computed mean (optional, computed if not provided)
    
    Returns:
        float: Zero-crossing rate (0.0 to 1.0)
    """
    if buffer_count < 2:
        return 0.0
    
    # Calculate mean if not provided
    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count
    
    # Count zero crossings (around mean)
    crossings = 0
    prev_above = turbulence_buffer[0] >= mean
    
    for i in range(1, buffer_count):
        curr_above = turbulence_buffer[i] >= mean
        if curr_above != prev_above:
            crossings += 1
        prev_above = curr_above
    
    return crossings / (buffer_count - 1)


def calc_autocorrelation(turbulence_buffer, buffer_count, mean=None, variance=None, lag=1):
    """
    Calculate lag-k autocorrelation coefficient.
    
    Measures temporal correlation between values separated by 'lag' samples.
    Higher lag captures longer-term temporal patterns in motion.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        mean: Pre-computed mean (optional)
        variance: Pre-computed variance (optional)
        lag: Number of samples to lag (default: 1)
    
    Returns:
        float: Autocorrelation coefficient (-1.0 to 1.0)
    """
    if buffer_count < lag + 2:
        return 0.0
    
    # Calculate mean if not provided
    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count
    
    # Calculate variance if not provided
    if variance is None:
        variance = 0.0
        for i in range(buffer_count):
            diff = turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= buffer_count
    
    if variance < 1e-10:
        return 0.0
    
    # Calculate lag-k autocovariance
    autocovariance = 0.0
    for i in range(buffer_count - lag):
        autocovariance += (turbulence_buffer[i] - mean) * (turbulence_buffer[i + lag] - mean)
    autocovariance /= (buffer_count - lag)
    
    return autocovariance / variance


def calc_mad(turbulence_buffer, buffer_count):
    """
    Calculate Median Absolute Deviation (MAD).
    
    Robust measure of variability, less sensitive to outliers than std.
    Uses an approximate median via the middle element of a sorted copy.
    
    On MicroPython (ESP32), buffer_count is typically 50, so a simple
    insertion sort is acceptable.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: MAD value
    """
    if buffer_count < 2:
        return 0.0
    
    # Copy values for sorting (don't modify original buffer)
    sorted_vals = [0.0] * buffer_count
    for i in range(buffer_count):
        sorted_vals[i] = turbulence_buffer[i]
    
    insertion_sort(sorted_vals, buffer_count)
    
    # Median
    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]
    
    # Calculate absolute deviations
    abs_devs = [0.0] * buffer_count
    for i in range(buffer_count):
        abs_devs[i] = abs(turbulence_buffer[i] - median)
    
    insertion_sort(abs_devs, buffer_count)
    
    # Median of absolute deviations
    if buffer_count % 2 == 0:
        mad = (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    else:
        mad = abs_devs[mid]
    
    return mad


# ============================================================================
# Energy Features (FFT-based) - From CSI-F paper
# ============================================================================

def calc_energy_features(turbulence_buffer, buffer_count, sample_rate=100.0):
    """
    Calculate energy-based features using FFT.
    
    Based on CSI-F paper: energy features help distinguish idle/motion states
    and are direction-independent.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        sample_rate: Sample rate in Hz (default: 100 Hz)
    
    Returns:
        tuple: (total_energy, low_freq_energy, mid_freq_energy, 
                high_freq_energy, energy_ratio_low, dominant_freq)
    """
    if buffer_count < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Get values as list
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    # Compute FFT magnitudes
    magnitudes = _fft_real(values)
    n_freqs = len(magnitudes)
    
    if n_freqs < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Frequency resolution
    freq_resolution = sample_rate / (2 * n_freqs)
    
    # Define frequency bands (human movement: 0.5-10 Hz)
    # Low: 0-3 Hz (slow movements, breathing)
    # Mid: 3-10 Hz (walking, gestures)
    # High: >10 Hz (noise, rapid movements)
    low_cutoff = int(3.0 / freq_resolution) if freq_resolution > 0 else n_freqs // 4
    mid_cutoff = int(10.0 / freq_resolution) if freq_resolution > 0 else n_freqs // 2
    
    low_cutoff = min(low_cutoff, n_freqs)
    mid_cutoff = min(mid_cutoff, n_freqs)
    
    # Calculate energy in each band (sum of squared magnitudes)
    total_energy = 0.0
    low_freq_energy = 0.0
    mid_freq_energy = 0.0
    high_freq_energy = 0.0
    
    max_mag = 0.0
    dominant_idx = 0
    
    for i in range(n_freqs):
        mag_sq = magnitudes[i] * magnitudes[i]
        total_energy += mag_sq
        
        if i < low_cutoff:
            low_freq_energy += mag_sq
        elif i < mid_cutoff:
            mid_freq_energy += mag_sq
        else:
            high_freq_energy += mag_sq
        
        # Track dominant frequency (skip DC component at i=0)
        if i > 0 and magnitudes[i] > max_mag:
            max_mag = magnitudes[i]
            dominant_idx = i
    
    # Energy ratio (low freq / total) - high ratio = idle, low ratio = motion
    energy_ratio_low = low_freq_energy / (total_energy + 1e-10)
    
    # Dominant frequency in Hz
    dominant_freq = dominant_idx * freq_resolution if freq_resolution > 0 else 0.0
    
    return (total_energy, low_freq_energy, mid_freq_energy, 
            high_freq_energy, energy_ratio_low, dominant_freq)


# ============================================================================
# Spectral Features - From CSI-HC paper
# ============================================================================

def calc_spectral_centroid(turbulence_buffer, buffer_count, sample_rate=100.0):
    """
    Calculate spectral centroid (center of mass of spectrum).
    
    Indicates where the "center" of the spectrum is located.
    Higher centroid = higher frequency content = more rapid movement.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        sample_rate: Sample rate in Hz
    
    Returns:
        float: Spectral centroid in Hz
    """
    if buffer_count < 4:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    magnitudes = _fft_real(values)
    n_freqs = len(magnitudes)
    
    if n_freqs < 2:
        return 0.0
    
    freq_resolution = sample_rate / (2 * n_freqs)
    
    # Spectral centroid = sum(f * mag) / sum(mag)
    weighted_sum = 0.0
    total_mag = 0.0
    
    for i in range(n_freqs):
        freq = i * freq_resolution
        weighted_sum += freq * magnitudes[i]
        total_mag += magnitudes[i]
    
    return weighted_sum / (total_mag + 1e-10)


def calc_spectral_flatness(turbulence_buffer, buffer_count):
    """
    Calculate spectral flatness (Wiener entropy).
    
    Measures how "flat" vs "peaked" the spectrum is.
    - Flatness close to 1.0: White noise (uniform spectrum)
    - Flatness close to 0.0: Tonal signal (peaked spectrum)
    
    Motion typically produces more peaked spectra (lower flatness).
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: Spectral flatness (0.0 to 1.0)
    """
    if buffer_count < 4:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    magnitudes = _fft_real(values)
    n_freqs = len(magnitudes)
    
    if n_freqs < 2:
        return 0.0
    
    # Filter out zero/near-zero values for log calculation
    valid_mags = [m for m in magnitudes if m > 1e-10]
    if len(valid_mags) < 2:
        return 0.0
    
    # Geometric mean = exp(mean(log(x)))
    log_sum = 0.0
    for m in valid_mags:
        log_sum += math.log(m)
    geometric_mean = math.exp(log_sum / len(valid_mags))
    
    # Arithmetic mean
    arithmetic_mean = sum(valid_mags) / len(valid_mags)
    
    return geometric_mean / (arithmetic_mean + 1e-10)


def calc_spectral_rolloff(turbulence_buffer, buffer_count, rolloff_percent=0.85, sample_rate=100.0):
    """
    Calculate spectral rolloff frequency.
    
    The frequency below which rolloff_percent of the total spectral energy lies.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        rolloff_percent: Percentage threshold (default: 0.85 = 85%)
        sample_rate: Sample rate in Hz
    
    Returns:
        float: Rolloff frequency in Hz
    """
    if buffer_count < 4:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    magnitudes = _fft_real(values)
    n_freqs = len(magnitudes)
    
    if n_freqs < 2:
        return 0.0
    
    freq_resolution = sample_rate / (2 * n_freqs)
    
    # Calculate total energy
    total_energy = sum(m * m for m in magnitudes)
    if total_energy < 1e-10:
        return 0.0
    
    # Find rolloff frequency
    threshold = rolloff_percent * total_energy
    cumulative_energy = 0.0
    
    for i in range(n_freqs):
        cumulative_energy += magnitudes[i] * magnitudes[i]
        if cumulative_energy >= threshold:
            return i * freq_resolution
    
    return (n_freqs - 1) * freq_resolution


# ============================================================================
# Multi-Lag Autocorrelation - From Wi-Limb paper
# ============================================================================

def calc_autocorrelation_lag(turbulence_buffer, buffer_count, lag, mean=None, variance=None):
    """
    Calculate autocorrelation at a specific lag.
    
    Multi-lag autocorrelation helps capture temporal structure at different scales.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        lag: Lag value (1, 2, 5, etc.)
        mean: Pre-computed mean (optional)
        variance: Pre-computed variance (optional)
    
    Returns:
        float: Autocorrelation coefficient at specified lag (-1.0 to 1.0)
    """
    if buffer_count < lag + 2:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    n = len(values)
    
    # Calculate mean if not provided
    if mean is None:
        mean = sum(values) / n
    
    # Calculate variance if not provided
    if variance is None:
        variance = sum((x - mean) ** 2 for x in values) / n
    
    if variance < 1e-10:
        return 0.0
    
    # Calculate lag-k autocovariance
    autocovariance = 0.0
    for i in range(n - lag):
        autocovariance += (values[i] - mean) * (values[i + lag] - mean)
    autocovariance /= (n - lag)
    
    return autocovariance / variance


def calc_periodicity_strength(turbulence_buffer, buffer_count):
    """
    Calculate strength of the dominant periodic component.
    
    Uses autocorrelation to detect periodic patterns in the signal.
    High periodicity = repetitive motion (walking, waving).
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: Periodicity strength (0.0 to 1.0)
    """
    if buffer_count < 10:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    
    if variance < 1e-10:
        return 0.0
    
    # Find peak in autocorrelation (excluding lag 0)
    max_autocorr = 0.0
    
    # Search for peaks in reasonable lag range (10-50% of buffer)
    min_lag = max(2, n // 10)
    max_lag = min(n // 2, 40)
    
    for lag in range(min_lag, max_lag):
        autocorr = calc_autocorrelation_lag(values, n, lag, mean, variance)
        if autocorr > max_autocorr:
            max_autocorr = autocorr
    
    return max(0.0, max_autocorr)


# ============================================================================
# Cross-Subcarrier Correlation Features - From Wi-Limb paper
# ============================================================================

def calc_subcarrier_correlation(amplitude_buffer):
    """
    Calculate correlation between subcarrier amplitudes across time.
    
    Requires a buffer of amplitude arrays (multiple packets).
    High correlation = stable signal (idle).
    Low correlation = varying signal (motion).
    
    Args:
        amplitude_buffer: List of amplitude arrays, shape (n_packets, n_subcarriers)
    
    Returns:
        tuple: (mean_correlation, correlation_std)
    """
    if amplitude_buffer is None or len(amplitude_buffer) < 2:
        return 0.0, 0.0
    
    n_packets = len(amplitude_buffer)
    n_subcarriers = len(amplitude_buffer[0]) if amplitude_buffer[0] else 0
    
    if n_subcarriers < 2:
        return 0.0, 0.0
    
    # Calculate correlation between consecutive packets
    correlations = []
    
    for i in range(n_packets - 1):
        amp1 = amplitude_buffer[i]
        amp2 = amplitude_buffer[i + 1]
        
        if len(amp1) != n_subcarriers or len(amp2) != n_subcarriers:
            continue
        
        # Pearson correlation between the two amplitude vectors
        mean1 = sum(amp1) / n_subcarriers
        mean2 = sum(amp2) / n_subcarriers
        
        numerator = 0.0
        var1 = 0.0
        var2 = 0.0
        
        for j in range(n_subcarriers):
            d1 = amp1[j] - mean1
            d2 = amp2[j] - mean2
            numerator += d1 * d2
            var1 += d1 * d1
            var2 += d2 * d2
        
        denominator = math.sqrt(var1 * var2)
        if denominator > 1e-10:
            correlations.append(numerator / denominator)
    
    if not correlations:
        return 0.0, 0.0
    
    # Mean and std of correlations
    mean_corr = sum(correlations) / len(correlations)
    
    if len(correlations) > 1:
        var_corr = sum((c - mean_corr) ** 2 for c in correlations) / len(correlations)
        std_corr = math.sqrt(var_corr)
    else:
        std_corr = 0.0
    
    return mean_corr, std_corr


# ============================================================================
# Phase-based Features
# ============================================================================

def calc_phase_diff_variance(phases):
    """
    Calculate variance of phase differences between adjacent subcarriers.
    
    Phase difference is more robust to phase offset than raw phase.
    High variance = inconsistent phase pattern (motion).
    Low variance = stable phase relationship (idle).
    
    Args:
        phases: List of phase values in radians (one per subcarrier)
    
    Returns:
        float: Variance of phase differences
    """
    if phases is None or len(phases) < 3:
        return 0.0
    
    # Calculate phase differences between adjacent subcarriers
    phase_diffs = []
    for i in range(len(phases) - 1):
        # Wrap difference to [-pi, pi]
        diff = phases[i + 1] - phases[i]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        phase_diffs.append(diff)
    
    # Calculate variance
    n = len(phase_diffs)
    mean_diff = sum(phase_diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in phase_diffs) / n
    
    return variance


def calc_phase_std(phases):
    """
    Calculate standard deviation of phases across subcarriers.
    
    Higher std indicates more dispersed phase values (motion).
    Lower std indicates coherent phase pattern (idle).
    
    Args:
        phases: List of phase values in radians (one per subcarrier)
    
    Returns:
        float: Standard deviation of phases
    """
    if phases is None or len(phases) < 2:
        return 0.0
    
    n = len(phases)
    mean_phase = sum(phases) / n
    variance = sum((p - mean_phase) ** 2 for p in phases) / n
    
    return math.sqrt(variance)


def calc_phase_entropy(phases, n_bins=5):
    """
    Calculate entropy of phase distribution across subcarriers.
    
    Higher entropy indicates more uniform/random phase distribution.
    Motion tends to create varied phase patterns (higher entropy).
    
    Args:
        phases: List of phase values in radians (one per subcarrier)
        n_bins: Number of histogram bins (default: 5, suitable for 12 samples)
    
    Returns:
        float: Shannon entropy in bits
    """
    if phases is None or len(phases) < 2:
        return 0.0
    
    n = len(phases)
    
    # Find range (phases are in [-pi, pi])
    min_p = min(phases)
    max_p = max(phases)
    
    if max_p <= min_p:
        return 0.0
    
    # Build histogram
    bin_width = (max_p - min_p) / n_bins
    counts = [0] * n_bins
    
    for p in phases:
        bin_idx = int((p - min_p) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        counts[bin_idx] += 1
    
    # Calculate entropy
    entropy = 0.0
    for count in counts:
        if count > 0:
            prob = count / n
            entropy -= prob * math.log(prob)
    
    return entropy


def calc_phase_range(phases):
    """
    Calculate range of phases across subcarriers (max - min).
    
    Larger range indicates more phase variation (motion).
    Smaller range indicates stable phase pattern (idle).
    
    Args:
        phases: List of phase values in radians (one per subcarrier)
    
    Returns:
        float: Phase range in radians
    """
    if phases is None or len(phases) < 2:
        return 0.0
    
    return max(phases) - min(phases)


# ============================================================================
# Cross-Subcarrier Features (from amplitude array)
# ============================================================================

def calc_amp_entropy(amplitudes, n_bins=5):
    """
    Calculate entropy of amplitude distribution across subcarriers.
    
    Higher entropy indicates more uniform amplitude distribution.
    Motion tends to create non-uniform patterns (lower entropy).
    
    Args:
        amplitudes: List of subcarrier amplitudes (typically 12 values)
        n_bins: Number of histogram bins (default: 5, suitable for 12 samples)
    
    Returns:
        float: Shannon entropy in bits
    """
    if amplitudes is None:
        return 0.0
    n = len(amplitudes)
    if n < 2:
        return 0.0
    
    min_val = min(amplitudes)
    max_val = max(amplitudes)
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for val in amplitudes:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    entropy = 0.0
    log2 = math.log(2)
    for count in bins:
        if count > 0:
            p = count / n
            entropy -= p * math.log(p) / log2
    
    return entropy


# ============================================================================
# Feature Registry and Configurable Extraction
# ============================================================================

# Default feature set (12 features: 11 turbulence + 1 cross-subcarrier)
# Note: turb_delta was replaced by amp_entropy based on SHAP importance analysis.
# See tools/10_train_ml_model.py for full SHAP feature importance table.
# Default feature set for inference (12 features)
# Ordered by category: statistical, temporal (multi-lag autocorr), amplitude
DEFAULT_FEATURES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_zcr',
    'turb_skewness', 'turb_kurtosis', 'turb_entropy', 'turb_autocorr', 'turb_mad',
    'turb_slope', 'amp_entropy'
]

# All available features (for documentation and experimentation)
# To experiment with different features, edit TRAINING_FEATURES in 10_train_ml_model.py
ALL_AVAILABLE_FEATURES = [
    # === TURBULENCE STATISTICAL FEATURES ===
    'turb_mean',              # Mean turbulence (central tendency)
    'turb_std',               # Standard deviation (spread)
    'turb_max',               # Maximum value in window
    'turb_min',               # Minimum value in window
    'turb_zcr',               # Zero-crossing rate around mean
    'turb_skewness',          # Asymmetry (3rd moment)
    'turb_kurtosis',          # Tailedness (4th moment)
    'turb_entropy',           # Shannon entropy (randomness)
    'turb_mad',               # Median absolute deviation
    'turb_slope',             # Linear regression slope
    'turb_delta',             # Start-to-end change
    
    # === TURBULENCE TEMPORAL FEATURES ===
    'turb_autocorr',          # Lag-1 autocorrelation (10ms @ 100Hz)
    'turb_autocorr_lag2',     # Lag-2 autocorrelation (20ms @ 100Hz)
    'turb_autocorr_lag5',     # Lag-5 autocorrelation (50ms @ 100Hz)
    'turb_periodicity',       # Strength of periodic component
    
    # === AMPLITUDE FEATURES ===
    'amp_entropy',            # Entropy of amplitude distribution
    'amp_range',              # Amplitude range
    'amp_skewness',           # Amplitude skewness
    'amp_kurtosis',           # Amplitude kurtosis
    
    # === PHASE FEATURES ===
    'phase_diff_var',         # Variance of phase differences between subcarriers
    'phase_std',              # Standard deviation of phases
    'phase_entropy',          # Shannon entropy of phase distribution
    'phase_range',            # Range of phases (max - min)
    
    # === ENERGY FEATURES (FFT-based, from CSI-F paper) ===
    'fft_total_energy',       # Total spectral energy
    'fft_low_energy',         # Energy 0-3 Hz (slow movements)
    'fft_mid_energy',         # Energy 3-10 Hz (gestures, walking)
    'fft_high_energy',        # Energy >10 Hz (noise, rapid movements)
    'fft_energy_ratio_low',   # Low freq ratio (idle indicator)
    'fft_dominant_freq',      # Dominant frequency Hz
    
    # === SPECTRAL FEATURES (from CSI-HC paper) ===
    'spectral_centroid',      # Center of mass of spectrum
    'spectral_flatness',      # Wiener entropy (noise vs tonal)
    'spectral_rolloff',       # Frequency at 85% energy
]


def extract_features_by_name(turbulence_buffer, buffer_count, amplitudes=None, 
                              feature_names=None, sample_rate=100.0, phases=None):
    """
    Extract specified features from turbulence buffer, amplitudes, and phases.
    
    This is the main feature extraction function that supports configurable
    feature selection. Use this for training experiments with different
    feature sets.
    
    Args:
        turbulence_buffer: List/buffer of turbulence values
        buffer_count: Number of valid values in buffer
        amplitudes: List of subcarrier amplitudes (needed for amp_* features)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
        sample_rate: Sample rate in Hz for FFT features (default: 100.0)
        phases: List of subcarrier phases in radians (needed for phase_* features)
    
    Returns:
        list: Feature values in the order specified by feature_names
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    
    if buffer_count < 2:
        return [0.0] * len(feature_names)
    
    # Convert to list if needed
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        turb_list = list(turbulence_buffer)[:buffer_count]
    else:
        turb_list = turbulence_buffer[:buffer_count]
    
    n = len(turb_list)
    if n < 2:
        return [0.0] * len(feature_names)
    
    # Pre-compute common turbulence statistics (computed once, reused)
    turb_mean = sum(turb_list) / n
    turb_var = sum((x - turb_mean) ** 2 for x in turb_list) / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    turb_min = min(turb_list)
    turb_max = max(turb_list)
    
    # Pre-compute slope (needed for turb_slope)
    mean_i = (n - 1) / 2.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        diff_i = i - mean_i
        diff_x = turb_list[i] - turb_mean
        numerator += diff_i * diff_x
        denominator += diff_i * diff_i
    turb_slope = numerator / denominator if denominator > 0 else 0.0
    
    # Lazy computation cache for expensive FFT features
    _fft_cache = {}
    
    def _get_energy_features():
        if 'energy' not in _fft_cache:
            _fft_cache['energy'] = calc_energy_features(turb_list, n, sample_rate)
        return _fft_cache['energy']
    
    # Feature calculation lookup table
    feature_calculators = {
        # Turbulence buffer features (11 used in DEFAULT_FEATURES)
        'turb_mean': lambda: turb_mean,
        'turb_std': lambda: turb_std,
        'turb_max': lambda: turb_max,
        'turb_min': lambda: turb_min,
        'turb_zcr': lambda: calc_zero_crossing_rate(turb_list, n, mean=turb_mean),
        'turb_skewness': lambda: calc_skewness(turb_list, n, turb_mean, turb_std),
        'turb_kurtosis': lambda: calc_kurtosis(turb_list, n, turb_mean, turb_std),
        'turb_entropy': lambda: calc_entropy_turb(turb_list, n),
        'turb_autocorr': lambda: calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var),
        'turb_mad': lambda: calc_mad(turb_list, n),
        'turb_slope': lambda: turb_slope,
        
        # Cross-subcarrier feature (1 used in DEFAULT_FEATURES)
        'amp_entropy': lambda: calc_amp_entropy(amplitudes),
        
        # Energy features (CSI-F paper) - FFT-based
        'fft_total_energy': lambda: _get_energy_features()[0],
        'fft_low_energy': lambda: _get_energy_features()[1],
        'fft_mid_energy': lambda: _get_energy_features()[2],
        'fft_high_energy': lambda: _get_energy_features()[3],
        'fft_energy_ratio_low': lambda: _get_energy_features()[4],
        'fft_dominant_freq': lambda: _get_energy_features()[5],
        
        # Spectral features (CSI-HC paper)
        'spectral_centroid': lambda: calc_spectral_centroid(turb_list, n, sample_rate),
        'spectral_flatness': lambda: calc_spectral_flatness(turb_list, n),
        'spectral_rolloff': lambda: calc_spectral_rolloff(turb_list, n, 0.85, sample_rate),
        
        # Multi-lag autocorrelation (Wi-Limb paper)
        'turb_autocorr_lag2': lambda: calc_autocorrelation_lag(turb_list, n, 2, turb_mean, turb_var),
        'turb_autocorr_lag5': lambda: calc_autocorrelation_lag(turb_list, n, 5, turb_mean, turb_var),
        'turb_periodicity': lambda: calc_periodicity_strength(turb_list, n),
        
        # Legacy features (kept for training experiments)
        'turb_delta': lambda: turb_list[-1] - turb_list[0],
        'amp_range': lambda: (max(amplitudes) - min(amplitudes)) if amplitudes and len(amplitudes) >= 2 else 0.0,
        'amp_skewness': lambda: _calc_amp_moment(amplitudes, 3),
        'amp_kurtosis': lambda: _calc_amp_moment(amplitudes, 4),
        
        # Phase-based features
        'phase_diff_var': lambda: calc_phase_diff_variance(phases),
        'phase_std': lambda: calc_phase_std(phases),
        'phase_entropy': lambda: calc_phase_entropy(phases),
        'phase_range': lambda: calc_phase_range(phases),
    }
    
    # Extract requested features
    features = []
    for name in feature_names:
        if name not in feature_calculators:
            raise ValueError(f"Unknown feature: {name}. Available: {list(feature_calculators.keys())}")
        features.append(feature_calculators[name]())
    
    return features


def _calc_amp_moment(amplitudes, moment):
    """Calculate skewness (moment=3) or kurtosis (moment=4) of amplitudes."""
    if amplitudes is None:
        return 0.0
    n = len(amplitudes)
    if n < moment:
        return 0.0
    
    mean = sum(amplitudes) / n
    variance = sum((x - mean) ** 2 for x in amplitudes) / n
    std = math.sqrt(variance) if variance > 0 else 0.0
    
    if std < 1e-10:
        return 0.0
    
    m = sum((x - mean) ** moment for x in amplitudes) / n
    result = m / (std ** moment)
    return result - 3.0 if moment == 4 else result  # Excess kurtosis


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
