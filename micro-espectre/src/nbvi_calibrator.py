"""
NBVI (Normalized Baseline Variability Index) Calibrator

Automatic subcarrier selection based on baseline variability analysis.
Identifies optimal subcarriers for motion detection using statistical analysis.

Algorithm:
1. Collect baseline CSI packets (quiet room)
2. Find candidate baseline windows using percentile-based detection
3. For each candidate, calculate NBVI for all subcarriers
4. Select 12 subcarriers with lowest NBVI and spectral spacing
5. Validate using MVS false positive rate

Output: (selected_band, mv_values)
- selected_band: List of 12 optimal subcarrier indices
- mv_values: Moving variance values for adaptive threshold calculation

Adaptive threshold is calculated externally using threshold.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import gc
import os
from array import array

try:
    from src.config import (
        NUM_SUBCARRIERS, EXPECTED_CSI_LEN,
        GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER, BAND_SIZE,
        SEG_WINDOW_SIZE, CALIBRATION_BUFFER_SIZE,
        ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD,
        ENABLE_LOWPASS_FILTER, LOWPASS_CUTOFF
    )
    from src.utils import (
        to_signed_int8, calculate_percentile, insertion_sort
    )
    from src.segmentation import SegmentationContext
except ImportError:
    from config import (
        NUM_SUBCARRIERS, EXPECTED_CSI_LEN,
        GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER, BAND_SIZE,
        SEG_WINDOW_SIZE, CALIBRATION_BUFFER_SIZE,
        ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD,
        ENABLE_LOWPASS_FILTER, LOWPASS_CUTOFF
    )
    from utils import (
        to_signed_int8, calculate_percentile, insertion_sort
    )
    from segmentation import SegmentationContext

# Constants
BUFFER_FILE = '/nbvi_buffer.bin'

# Threshold for null subcarrier detection (mean amplitude below this = null)
NULL_SUBCARRIER_THRESHOLD = 1.0

# Adaptive validation threshold parameters (aligned with runtime threshold mode AUTO)
VALIDATION_ADAPTIVE_PERCENTILE = 95
VALIDATION_ADAPTIVE_FACTOR = 1.1


def cleanup_buffer_file():
    """Remove any leftover buffer file from previous interrupted runs."""
    try:
        os.remove(BUFFER_FILE)
        print("NBVI: Cleaned up leftover buffer file")
    except OSError:
        pass


class NBVICalibrator:
    """
    Automatic NBVI calibrator with percentile-based baseline detection
    
    Collects CSI packets at boot and automatically selects optimal subcarriers
    using multi-strategy NBVI with percentile-based baseline detection.
    
    Uses file-based storage to avoid RAM limitations. Magnitudes stored as
    uint8 (max CSI magnitude ~181 fits in 1 byte).
    
    After subcarrier selection, calculates adaptive threshold using Pxx * factor.
    """
    
    def __init__(self, buffer_size=None, mvs_window_size=None,
                 percentile=5, alpha=0.75, min_spacing=1, noise_gate_percentile=15):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: CALIBRATION_BUFFER_SIZE from config)
            mvs_window_size: MVS window size for validation (default: SEG_WINDOW_SIZE from config)
            percentile: Percentile for baseline window detection (default: 5)
            alpha: NBVI weighting factor (default: 0.75)
            min_spacing: Minimum spacing between subcarriers (default: 1)
            noise_gate_percentile: Percentile for noise gate (default: 15)
        """
        self.buffer_size = buffer_size if buffer_size is not None else CALIBRATION_BUFFER_SIZE
        self._buffer_file = BUFFER_FILE
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._initialized = False
        
        # Keep batch writes under ~4 KB to avoid long SPIFFS stalls during CSI collection.
        self._write_batch_size = 50
        self._write_buf = bytearray(self._write_batch_size * NUM_SUBCARRIERS)
        self._write_buf_idx = 0
        
        # Remove old buffer file if exists
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        # Open file for writing
        self._file = open(BUFFER_FILE, 'wb')
        
        # NBVI parameters
        self.mvs_window_size = mvs_window_size if mvs_window_size is not None else SEG_WINDOW_SIZE
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = noise_gate_percentile
        self.hint_fp_tolerance = 0.0
        self.prefer_hint_on_tie = False
        # False: raw std (gain lock active), True: CV std/mean (gain lock absent)
        self.use_cv_normalization = False
        self._validation_ctx = None
        self._valid_subcarriers = tuple(
            sc for sc in range(NUM_SUBCARRIERS)
            if GUARD_BAND_LOW <= sc <= GUARD_BAND_HIGH and sc != DC_SUBCARRIER
        )
        self._valid_subcarrier_pairs = tuple(
            (sc, sc * 2, sc * 2 + 1) for sc in self._valid_subcarriers
        )
        self._zero_packet = b"\x00" * NUM_SUBCARRIERS
        self._analysis_window_size = 200
        self._analysis_window_bytes = self._analysis_window_size * NUM_SUBCARRIERS
        self._validation_chunk_packets = 64
        self._validation_chunk_bytes = self._validation_chunk_packets * NUM_SUBCARRIERS
        self._analysis_window_buf = None
        self._candidate_turbulence = None
        self._window_values = None
        self._window_sorted_values = None
        self._window_abs_devs = None
        self._window_bins = None
        self._validation_chunk_buf = None
        self._mean_by_sc = None
        self._classic_by_sc = None
        self._entropy_by_sc = None
        self._mad_by_sc = None

    def set_cv_normalization(self, enabled):
        """Enable or disable CV normalization for turbulence calculations."""
        self.use_cv_normalization = bool(enabled)

    def set_hint_fp_tolerance(self, tolerance):
        """Set max FP degradation allowed when keeping hint band."""
        self.hint_fp_tolerance = float(tolerance)

    def set_prefer_hint_on_tie(self, enabled):
        """If False, hint band must be strictly better than best candidate."""
        self.prefer_hint_on_tie = bool(enabled)
    
    # ========================================================================
    # Buffer management
    # ========================================================================
    
    def _prepare_for_reading(self):
        """Flush remaining buffer, close write mode and reopen for reading."""
        if self._file:
            # Flush any remaining packets in batch buffer
            if self._write_buf_idx > 0:
                remaining = self._write_buf_idx * NUM_SUBCARRIERS
                self._file.write(memoryview(self._write_buf)[:remaining])
                self._write_buf_idx = 0
            self._file.flush()
            self._file.close()
        # Free write buffer — no longer needed after collection phase
        self._write_buf = None
        gc.collect()
        self._file = open(self._buffer_file, 'rb')

    def _ensure_analysis_resources(self):
        """Allocate heavy analysis buffers only after packet collection is complete."""
        if self._analysis_window_buf is None:
            self._analysis_window_buf = bytearray(self._analysis_window_bytes)
        if self._candidate_turbulence is None:
            self._candidate_turbulence = self._make_float_array(0.0, self._analysis_window_size)
        if self._window_values is None:
            self._window_values = bytearray(self._analysis_window_size)
        if self._window_sorted_values is None:
            self._window_sorted_values = bytearray(self._analysis_window_size)
        if self._window_abs_devs is None:
            self._window_abs_devs = bytearray(self._analysis_window_size)
        if self._window_bins is None:
            self._window_bins = [0] * 10
        if self._validation_chunk_buf is None:
            self._validation_chunk_buf = bytearray(self._validation_chunk_bytes)

        _INF = float('inf')
        if self._mean_by_sc is None:
            self._mean_by_sc = self._make_float_array(0.0, NUM_SUBCARRIERS)
        if self._classic_by_sc is None:
            self._classic_by_sc = self._make_float_array(_INF, NUM_SUBCARRIERS)
        if self._entropy_by_sc is None:
            self._entropy_by_sc = self._make_float_array(_INF, NUM_SUBCARRIERS)
        if self._mad_by_sc is None:
            self._mad_by_sc = self._make_float_array(_INF, NUM_SUBCARRIERS)
        if self._validation_ctx is None:
            self._validation_ctx = SegmentationContext(
                window_size=self.mvs_window_size,
                threshold=1.0,
                enable_lowpass=ENABLE_LOWPASS_FILTER,
                lowpass_cutoff=LOWPASS_CUTOFF,
                enable_hampel=ENABLE_HAMPEL_FILTER,
                hampel_window=HAMPEL_WINDOW,
                hampel_threshold=HAMPEL_THRESHOLD,
            )

        gc.collect()
    
    def free_buffer(self):
        """Free resources after calibration is complete."""
        if self._file:
            self._file.close()
            self._file = None
        
        # Free batch buffer
        self._write_buf = None
        
        try:
            os.remove(self._buffer_file)
        except OSError:
            pass

        self._analysis_window_buf = None
        self._candidate_turbulence = None
        self._window_values = None
        self._window_sorted_values = None
        self._window_abs_devs = None
        self._window_bins = None
        self._validation_chunk_buf = None
        self._mean_by_sc = None
        self._classic_by_sc = None
        self._entropy_by_sc = None
        self._mad_by_sc = None
        self._validation_ctx = None
    
    def get_packet_count(self):
        """Get the number of packets currently in the buffer."""
        return self._packet_count
    
    def is_buffer_full(self):
        """Check if the buffer has collected enough packets."""
        return self._packet_count >= self.buffer_size
    
    # ========================================================================
    # Packet collection
    # ========================================================================
        
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer (file-based)
        
        HT20 only: expects 128 bytes (64 subcarriers x 2 I/Q).
        
        Args:
            csi_data: CSI data array (128 bytes for HT20)
        
        Returns:
            int: Current buffer size (progress indicator)
        """
        if self._packet_count >= self.buffer_size:
            return self.buffer_size
        
        # STBC packets (256 bytes) are truncated upstream before reaching here.
        # See GitHub issue #76, espressif/esp-csi#238 for details.
        if len(csi_data) != EXPECTED_CSI_LEN:
            self._filtered_count += 1
            if self._filtered_count % 50 == 1:
                print(f'[WARN] Filtered {self._filtered_count} packets with wrong SC count (got {len(csi_data)} bytes)')
            return self._packet_count
        
        # Initialize on first packet
        if not self._initialized:
            self._initialized = True
            print(f'NBVI: HT20 mode, {NUM_SUBCARRIERS} SC, guard [{GUARD_BAND_LOW}-{GUARD_BAND_HIGH}], DC={DC_SUBCARRIER}')
        
        # Extract magnitudes into batch buffer (avoids per-packet flash write)
        # Guard band and DC subcarriers are zeroed without computing sqrt —
        # they are excluded from NBVI selection anyway (marked inf in calibrate()).
        # Cache math.sqrt locally to avoid 42 global+attr lookups per packet.
        # I*I integer arithmetic avoids float() conversions (exact for I ∈ [-127,127]).
        _sqrt = math.sqrt
        buf_offset = self._write_buf_idx * NUM_SUBCARRIERS
        write_buf = self._write_buf
        write_buf[buf_offset:buf_offset + NUM_SUBCARRIERS] = self._zero_packet

        for sc, q_idx, i_idx in self._valid_subcarrier_pairs:
            q = csi_data[q_idx]
            i = csi_data[i_idx]

            # Inline signed int8 conversion to keep the collection hot path cheap.
            if q >= 128:
                q -= 256
            if i >= 128:
                i -= 256

            write_buf[buf_offset + sc] = int(_sqrt(i * i + q * q))
        
        self._write_buf_idx += 1
        self._packet_count += 1
        
        # Batch write when buffer full (reduces flash writes from 750 to ~8)
        if self._write_buf_idx >= self._write_batch_size:
            self._file.write(self._write_buf)
            self._write_buf_idx = 0
        
        return self._packet_count
    
    # ========================================================================
    # File I/O helpers
    # ========================================================================
    
    def _read_packet(self, packet_idx):
        """Read a single packet from file"""
        self._file.seek(packet_idx * NUM_SUBCARRIERS)
        data = self._file.read(NUM_SUBCARRIERS)
        return data if data else None

    def _read_packets_into(self, start_idx, packet_count, buffer_obj):
        """Read packet_count packets into a preallocated buffer."""
        total_bytes = packet_count * NUM_SUBCARRIERS
        if total_bytes > len(buffer_obj):
            return None

        self._file.seek(start_idx * NUM_SUBCARRIERS)
        try:
            bytes_read = self._file.readinto(memoryview(buffer_obj)[:total_bytes])
            if bytes_read != total_bytes:
                return None
            return buffer_obj
        except AttributeError:
            data = self._file.read(total_bytes)
            if not data or len(data) != total_bytes:
                return None
            memoryview(buffer_obj)[:total_bytes] = data
            return buffer_obj
    
    def _packet_turbulence(self, data, band, offset=0, packet_len=NUM_SUBCARRIERS):
        """Calculate spatial turbulence from raw packet bytes.

        Uses raw standard deviation by default. When CV normalization is enabled,
        uses std/mean to maintain gain invariance when gain lock is not active.
        """
        total = 0.0
        count = 0
        max_index = offset + packet_len

        for sc in band:
            idx = offset + sc
            if idx >= max_index:
                continue
            total += data[idx]
            count += 1

        if count == 0:
            return 0.0

        mean_mag = total / count
        variance_sum = 0.0
        for sc in band:
            idx = offset + sc
            if idx >= max_index:
                continue
            diff = data[idx] - mean_mag
            variance_sum += diff * diff

        variance = variance_sum / count
        std = math.sqrt(variance) if variance > 0 else 0.0
        if self.use_cv_normalization:
            return std / mean_mag if mean_mag > 1e-6 else 0.0
        return std

    def _reset_validation_context(self, ctx):
        """Reset SegmentationContext in place to avoid re-allocation churn."""
        buf = ctx.turbulence_buffer
        for i in range(len(buf)):
            buf[i] = 0.0

        ctx.state = ctx.STATE_IDLE
        ctx.packet_index = 0
        ctx.buffer_index = 0
        ctx.buffer_count = 0
        ctx.current_moving_variance = 0.0
        ctx.last_turbulence = 0.0
        ctx.last_amplitudes = None

        if ctx.lowpass_filter is not None:
            ctx.lowpass_filter.reset()
        if ctx.hampel_filter is not None:
            ctx.hampel_filter.reset()

    def _get_validation_context(self):
        """Create or reuse the SegmentationContext used in validation."""
        if self._validation_ctx is None:
            self._validation_ctx = SegmentationContext(
                window_size=self.mvs_window_size,
                threshold=1.0,
                enable_lowpass=ENABLE_LOWPASS_FILTER,
                lowpass_cutoff=LOWPASS_CUTOFF,
                enable_hampel=ENABLE_HAMPEL_FILTER,
                hampel_window=HAMPEL_WINDOW,
                hampel_threshold=HAMPEL_THRESHOLD,
            )
        else:
            self._reset_validation_context(self._validation_ctx)

        self._validation_ctx.threshold = 1.0
        self._validation_ctx.use_cv_normalization = self.use_cv_normalization
        return self._validation_ctx
    
    # ========================================================================
    # Calibration algorithm
    # ========================================================================
    
    def _find_candidate_windows(self, current_band, window_size=200, step=50):
        """
        Find all candidate baseline windows using percentile-based detection.
        Reads one full window at a time into a reusable buffer to reduce I/O overhead.
        
        NO absolute threshold - adapts automatically to environment.
        """
        if self._packet_count < window_size:
            return []
        if window_size > self._analysis_window_size:
            return []
        self._ensure_analysis_resources()
        
        window_results = []
        turbulence_values = self._candidate_turbulence
        window_data = self._analysis_window_buf
        
        for i in range(0, self._packet_count - window_size + 1, step):
            if self._read_packets_into(i, window_size, window_data) is None:
                continue
            sum_turb = 0.0
            count = window_size
            for idx in range(window_size):
                offset = idx * NUM_SUBCARRIERS
                turbulence = self._packet_turbulence(
                    window_data, current_band, offset=offset, packet_len=NUM_SUBCARRIERS
                )
                turbulence_values[idx] = turbulence
                sum_turb += turbulence

            if count == 0:
                continue

            mean_turb = sum_turb / count
            sum_sq = 0.0
            for idx in range(count):
                turbulence = turbulence_values[idx]
                diff = turbulence - mean_turb
                sum_sq += diff * diff
            
            window_results.append((i, sum_sq / count))
            
            if i % 200 == 0:
                gc.collect()
        
        if not window_results:
            return []
        
        variances = [w[1] for w in window_results]
        p_threshold = calculate_percentile(variances, self.percentile)
        
        candidates = [w for w in window_results if w[1] <= p_threshold]
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _calculate_nbvi_from_stats(self, mean, std, mad=0.0, entropy=0.0):
        """
        Calculate multiple NBVI scores to evaluate different candidate bands.
        """
        if mean < 1e-6:
            return {
                'nbvi_classic': float('inf'), 'nbvi_entropy': float('inf'),
                'nbvi_mad': float('inf'),
                'mean': mean, 'std': std,
            }

        cv = std / mean
        nbvi_energy = std / (mean * mean)
        base_score = self.alpha * nbvi_energy + (1 - self.alpha) * cv

        # Entropy-rewarded score
        entropy_factor = max(0.5, entropy)
        entropy_score = base_score / entropy_factor

        # MAD-based robust score
        robust_std = mad * 1.4826 if mad > 1e-6 else std
        cv_mad = robust_std / mean
        energy_mad = robust_std / (mean * mean)
        mad_score = self.alpha * energy_mad + (1 - self.alpha) * cv_mad

        return {
            'nbvi_classic': base_score,
            'nbvi_entropy': entropy_score,
            'nbvi_mad': mad_score,
            'mean': mean,
            'std': std,
            'mad': mad,
            'entropy': entropy,
        }
    
    def _apply_noise_gate(self, subcarrier_metrics):
        """Apply Noise Gate: exclude weak subcarriers and those with infinite NBVI"""
        # Collect valid means (exclude infinite NBVI, matching C++ implementation)
        valid_means = [m['mean'] for m in subcarrier_metrics 
                       if m['mean'] > 1.0 and m['nbvi'] != float('inf')]
        
        if not valid_means:
            print("NBVI: Noise Gate - no valid subcarriers found")
            return []
        
        threshold = calculate_percentile(valid_means, self.noise_gate_percentile)
        # Filter by mean threshold AND exclude infinite NBVI (matching C++)
        filtered = [m for m in subcarrier_metrics 
                if m['mean'] >= threshold and m['nbvi'] != float('inf')]
        
        return filtered
    
    def _select_with_spacing_strict(self, sorted_metrics, k=12):
        valid_candidates = [c for c in sorted_metrics if c['nbvi'] != float('inf')]
        for current_spacing in range(self.min_spacing, -1, -1):
            selected = []
            for candidate in valid_candidates:
                if len(selected) >= k:
                    break
                sc = candidate['subcarrier']
                if selected and min(abs(sc - s) for s in selected) < current_spacing:
                    continue
                selected.append(sc)
            if len(selected) >= k:
                selected.sort()
                return selected
        selected = [c['subcarrier'] for c in valid_candidates[:k]]
        selected.sort()
        return selected

    def _select_with_spacing(self, sorted_metrics, k=12):
        """Original clustered strategy for backward compatibility"""
        selected = []
        for m in sorted_metrics:
            if len(selected) >= 5:
                break
            if m['nbvi'] != float('inf'):
                selected.append(m['subcarrier'])
        
        for candidate in sorted_metrics[5:]:
            if len(selected) >= k:
                break
            sc = candidate['subcarrier']
            if min(abs(sc - s) for s in selected) >= self.min_spacing:
                selected.append(sc)
        
        if len(selected) < k:
            for candidate in sorted_metrics:
                if len(selected) >= k:
                    break
                sc = candidate['subcarrier']
                if sc not in selected:
                    selected.append(sc)
        
        selected.sort()
        return selected

    def _select_indices_with_spacing_strict(self, sorted_subcarriers, k=12):
        """Spacing selector for pre-sorted subcarrier index lists."""
        for current_spacing in range(self.min_spacing, -1, -1):
            selected = []
            for sc in sorted_subcarriers:
                if len(selected) >= k:
                    break
                if selected and min(abs(sc - s) for s in selected) < current_spacing:
                    continue
                selected.append(sc)
            if len(selected) >= k:
                selected.sort()
                return selected
        selected = list(sorted_subcarriers[:k])
        selected.sort()
        return selected

    def _select_indices_with_spacing(self, sorted_subcarriers, k=12):
        """Clustered spacing selector for pre-sorted subcarrier index lists."""
        selected = []
        for sc in sorted_subcarriers:
            if len(selected) >= 5:
                break
            selected.append(sc)

        for sc in sorted_subcarriers[5:]:
            if len(selected) >= k:
                break
            if min(abs(sc - s) for s in selected) >= self.min_spacing:
                selected.append(sc)

        if len(selected) < k:
            for sc in sorted_subcarriers:
                if len(selected) >= k:
                    break
                if sc not in selected:
                    selected.append(sc)

        selected.sort()
        return selected

    @staticmethod
    def _calculate_percentile_compact(values, percentile):
        """Calculate percentile without allocating a large Python list copy."""
        if not values:
            return 0.0

        sorted_values = array('f', values)
        insertion_sort(sorted_values, len(sorted_values))
        n = len(sorted_values)
        p = percentile / 100.0
        k = int((n - 1) * p)

        if k >= n - 1:
            return sorted_values[-1]

        frac = (n - 1) * p - k
        return sorted_values[k] * (1 - frac) + sorted_values[k + 1] * frac

    @staticmethod
    def _collect_filtered_subcarriers(valid_subcarriers, mean_by_sc, score_by_sc, threshold, inf_value):
        """Collect filtered subcarriers into a compact bytearray."""
        filtered = bytearray()
        for sc in valid_subcarriers:
            if mean_by_sc[sc] >= threshold and score_by_sc[sc] != inf_value:
                filtered.append(sc)
        return filtered

    @staticmethod
    def _sort_subcarriers_by_score(subcarriers, score_by_sc):
        """Sort compact subcarrier ids by a parallel score array."""
        sorted_subcarriers = bytearray(subcarriers)
        n = len(sorted_subcarriers)
        for i in range(1, n):
            key_sc = sorted_subcarriers[i]
            key_score = score_by_sc[key_sc]
            j = i - 1
            while j >= 0 and score_by_sc[sorted_subcarriers[j]] > key_score:
                sorted_subcarriers[j + 1] = sorted_subcarriers[j]
                j -= 1
            sorted_subcarriers[j + 1] = key_sc
        return sorted_subcarriers

    @staticmethod
    def _make_float_array(fill_value, size):
        """Create a compact float array without first allocating a Python list."""
        values = array('f')
        for _ in range(size):
            values.append(fill_value)
        return values

    @staticmethod
    def _fill_float_array(values, fill_value):
        """Reset a compact float array in place."""
        for i in range(len(values)):
            values[i] = fill_value
    
    def _validate_subcarriers(self, band):
        """
        Validate subcarriers by running MVS on entire buffer.

        Uses the runtime detector path for filtering and moving variance:
        turbulence -> SegmentationContext.add_turbulence() -> update_state()

        Returns:
            tuple: (fp_rate, mv_values) where mv_values is a compact float array
        """
        if self._packet_count < self.mvs_window_size:
            return 0.0, array('f')

        self._ensure_analysis_resources()
        
        ctx = self._get_validation_context()

        total_packets = 0
        # Subsample mv_values at 1:5 for the adaptive threshold (P95).
        # The 750-packet buffer is needed for band selection quality, but P95
        # is statistically stable with ~140 samples. A contiguous list of 700
        # floats (2700 bytes) exceeds the available heap on ESP32-C3 after the
        # NBVI streaming phase, while 140 floats (560 bytes) fits comfortably.
        MV_SUBSAMPLE = 5
        mv_values = array('f')
        pkt_idx = 0
        chunk_packets = self._validation_chunk_packets
        chunk_data = self._validation_chunk_buf

        for start_idx in range(0, self._packet_count, chunk_packets):
            count = min(chunk_packets, self._packet_count - start_idx)
            if self._read_packets_into(start_idx, count, chunk_data) is None:
                break

            for local_idx in range(count):
                offset = local_idx * NUM_SUBCARRIERS
                turbulence = self._packet_turbulence(
                    chunk_data, band, offset=offset, packet_len=NUM_SUBCARRIERS
                )
                ctx.add_turbulence(turbulence)

                if pkt_idx >= self.mvs_window_size:
                    metrics = ctx.update_state()
                    mv_variance = metrics['moving_variance']
                    if total_packets % MV_SUBSAMPLE == 0:
                        mv_values.append(mv_variance)

                    total_packets += 1

                pkt_idx += 1

        if not mv_values:
            return 0.0, array('f')

        adaptive_thr = self._calculate_percentile_compact(
            mv_values, VALIDATION_ADAPTIVE_PERCENTILE
        ) * VALIDATION_ADAPTIVE_FACTOR
        motion_count = 0
        for mv in mv_values:
            if mv > adaptive_thr:
                motion_count += 1
        fp_rate = motion_count / len(mv_values)
        return fp_rate, mv_values
    
    def calibrate(self, hint_band=None):
        """
        Calibrate using NBVI Weighted with percentile-based detection.
        
        Args:
            hint_band: Optional band to use for candidate window search.
                       If provided, uses this band to calculate turbulence
                       when finding baseline candidate windows.
                       Matches C++ start_calibration(current_band) behavior.
        
        Returns:
            tuple: (selected_band, mv_values) or (None, []) if failed
        """
        window_size = self._analysis_window_size
        step = 50
        
        if self._packet_count < self.mvs_window_size + 10:
            print("NBVI: Not enough packets for calibration")
            return None, []
        
        self._prepare_for_reading()
        self._ensure_analysis_resources()
        
        # Use hint_band if provided, otherwise use default band for finding candidate windows
        # This matches C++ behavior where start_calibration() receives current_band as hint
        if hint_band is not None:
            search_band = hint_band
        else:
            search_band = list(range(GUARD_BAND_LOW, GUARD_BAND_LOW + BAND_SIZE))
        candidates = self._find_candidate_windows(search_band, window_size, step)
        
        if not candidates:
            print("NBVI: Failed to find candidate windows")
            return None, []
        
        print(f"NBVI: Found {len(candidates)} candidate windows")
        
        best_fp_rate = 1.0
        best_band = None
        best_mv_values = array('f')
        best_avg_nbvi = 0.0
        best_avg_mean = 0.0
        best_window_idx = 0
        
        for idx, (start_idx, window_variance) in enumerate(candidates):
            count = min(window_size, self._packet_count - start_idx)
            
            if count == 0:
                continue

            values = self._window_values
            sorted_values = self._window_sorted_values
            abs_devs = self._window_abs_devs
            bins = self._window_bins
            window_data = self._analysis_window_buf
            _sqrt = math.sqrt
            _log2 = math.log2
            _INF = float('inf')
            mean_by_sc = self._mean_by_sc
            classic_by_sc = self._classic_by_sc
            entropy_by_sc = self._entropy_by_sc
            mad_by_sc = self._mad_by_sc
            self._fill_float_array(mean_by_sc, 0.0)
            self._fill_float_array(classic_by_sc, _INF)
            self._fill_float_array(entropy_by_sc, _INF)
            self._fill_float_array(mad_by_sc, _INF)
            if self._read_packets_into(start_idx, count, window_data) is None:
                continue
            
            for sc in self._valid_subcarriers:
                total = 0
                min_v = 255
                max_v = 0
                for val_idx in range(count):
                    value = window_data[val_idx * NUM_SUBCARRIERS + sc]
                    values[val_idx] = value
                    total += value
                    if value < min_v:
                        min_v = value
                    if value > max_v:
                        max_v = value

                mean = total / count
                var_sum = 0.0
                range_v = max_v - min_v
                entropy = 0.0
                if range_v > 0:
                    bin_w = range_v / 10
                    for bin_idx in range(10):
                        bins[bin_idx] = 0
                    for val_idx in range(count):
                        value = values[val_idx]
                        diff = value - mean
                        var_sum += diff * diff
                        b = int((value - min_v) / bin_w)
                        if b == 10:
                            b = 9
                        bins[b] += 1
                    for b in bins:
                        if b > 0:
                            p = b / count
                            entropy -= p * _log2(p)
                else:
                    for val_idx in range(count):
                        value = values[val_idx]
                        diff = value - mean
                        var_sum += diff * diff

                var = var_sum / count
                std = _sqrt(var) if var > 0 else 0.0

                # MAD
                for val_idx in range(count):
                    sorted_values[val_idx] = values[val_idx]
                insertion_sort(sorted_values, count)
                median = sorted_values[count // 2]
                for val_idx in range(count):
                    diff = values[val_idx] - median
                    abs_devs[val_idx] = diff if diff >= 0 else -diff
                insertion_sort(abs_devs, count)
                mad = abs_devs[count // 2]

                metrics = self._calculate_nbvi_from_stats(mean, std, mad=mad, entropy=entropy)
                mean_by_sc[sc] = metrics['mean']
                if metrics['mean'] >= NULL_SUBCARRIER_THRESHOLD:
                    classic_by_sc[sc] = metrics['nbvi_classic']
                    entropy_by_sc[sc] = metrics['nbvi_entropy']
                    mad_by_sc[sc] = metrics['nbvi_mad']

            valid_means = array('f')
            for sc in self._valid_subcarriers:
                if classic_by_sc[sc] != _INF:
                    valid_means.append(mean_by_sc[sc])
            if not valid_means:
                print("NBVI: Noise Gate - no valid subcarriers found")
                continue

            threshold = self._calculate_percentile_compact(valid_means, self.noise_gate_percentile)
            filtered_subcarriers = self._collect_filtered_subcarriers(
                self._valid_subcarriers, mean_by_sc, classic_by_sc, threshold, _INF
            )

            sorted_entropy = self._sort_subcarriers_by_score(filtered_subcarriers, entropy_by_sc)
            band_entropy = self._select_indices_with_spacing_strict(sorted_entropy, k=BAND_SIZE)

            sorted_mad = self._sort_subcarriers_by_score(filtered_subcarriers, mad_by_sc)
            band_mad = self._select_indices_with_spacing(sorted_mad, k=BAND_SIZE)

            sorted_classic = self._sort_subcarriers_by_score(filtered_subcarriers, classic_by_sc)
            band_classic_spaced = self._select_indices_with_spacing_strict(sorted_classic, k=BAND_SIZE)
            band_classic = self._select_indices_with_spacing(sorted_classic, k=BAND_SIZE)

            candidates_to_eval = []

            def append_unique_candidate(candidate_band, score_table):
                if len(candidate_band) != BAND_SIZE:
                    return
                for existing_band, _ in candidates_to_eval:
                    if existing_band == candidate_band:
                        return
                candidates_to_eval.append((candidate_band, score_table))

            if len(band_entropy) == BAND_SIZE:
                append_unique_candidate(band_entropy, entropy_by_sc)
            if len(band_mad) == BAND_SIZE:
                append_unique_candidate(band_mad, mad_by_sc)
            if len(band_classic_spaced) == BAND_SIZE:
                append_unique_candidate(band_classic_spaced, classic_by_sc)
            if len(band_classic) == BAND_SIZE:
                append_unique_candidate(band_classic, classic_by_sc)
            
            for candidate_band, score_by_sc in candidates_to_eval:
                if len(candidate_band) != BAND_SIZE:
                    continue
                
                avg_nbvi = sum(score_by_sc[sc] for sc in candidate_band) / len(candidate_band)
                avg_mean = sum(mean_by_sc[sc] for sc in candidate_band) / len(candidate_band)
                fp_rate, mv_values = self._validate_subcarriers(candidate_band)
                
                override = False
                
                if best_band is None:
                    override = True
                elif fp_rate <= 0.05:
                    if best_fp_rate > 0.05:
                        override = True
                else:
                    if fp_rate < best_fp_rate:
                        override = True

                if override:
                    best_fp_rate = fp_rate
                    best_band = candidate_band
                    best_mv_values = mv_values
                    best_window_idx = idx
                    best_avg_nbvi = avg_nbvi
                    best_avg_mean = avg_mean

            gc.collect()
        
        if best_band is None:
            print("NBVI: All candidate windows failed - using default subcarriers")
            
            # Run validation on search_band (hint_band or default) to get MV values
            _, mv_values = self._validate_subcarriers(search_band)
            
            print(f"NBVI: Fallback to default band")
            
            if self._filtered_count > 0:
                print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
            
            return search_band, mv_values
        
        HINT_FP_TOLERANCE = self.hint_fp_tolerance
        FP_COMPARE_EPSILON = 1e-6
        use_hint_band = False
        hint_fp_rate = 1.0
        hint_mv_values = array('f')
        if hint_band is not None and len(hint_band) == BAND_SIZE:
            hint_fp_rate, hint_mv_values = self._validate_subcarriers(hint_band)

            best_fp_acceptable = best_fp_rate <= 0.05
            hint_fp_acceptable = hint_fp_rate <= 0.05
            acceptable_best_cmp = best_fp_rate + HINT_FP_TOLERANCE + FP_COMPARE_EPSILON
            strict_best_cmp = best_fp_rate + HINT_FP_TOLERANCE
            if best_fp_acceptable and hint_fp_acceptable:
                if hint_fp_rate <= acceptable_best_cmp:
                    use_hint_band = True
                else:
                    print(f"NBVI: Keeping candidate band with FP {best_fp_rate*100:.1f}% "
                          f"vs hint {hint_fp_rate*100:.1f}% (acceptable target <5.0%)")
            elif not best_fp_acceptable:
                if self.prefer_hint_on_tie:
                    hint_fp_ok = hint_fp_rate <= acceptable_best_cmp
                else:
                    hint_fp_ok = (hint_fp_rate + FP_COMPARE_EPSILON) < strict_best_cmp
                
                if hint_fp_ok:
                    use_hint_band = True
                else:
                    print(f"NBVI: Hint FP ({hint_fp_rate*100:.1f}%) not better than "
                          f"candidate ({best_fp_rate*100:.1f}%) - keeping NBVI band")
            else:
                print(f"NBVI: Keeping candidate band with FP {best_fp_rate*100:.1f}% "
                      f"(target <5.0%, hint {hint_fp_rate*100:.1f}% not acceptable)")
        
        if use_hint_band:
            best_band = list(hint_band)
            best_mv_values = hint_mv_values
            print(
                f"NBVI: Using hint band (FP {hint_fp_rate * 100:.1f}% "
                f"vs best {best_fp_rate * 100:.1f}%, tol {HINT_FP_TOLERANCE * 100:.1f}%, "
                f"tie={'prefer' if self.prefer_hint_on_tie else 'strict'})"
            )
        
        print(f"NBVI: Selected window {best_window_idx + 1}/{len(candidates)} with FP rate {best_fp_rate * 100:.1f}%")
        
        print(f"NBVI: Band selection successful")
        print(f"  Band: {best_band}")
        print(f"  Avg NBVI: {best_avg_nbvi:.6f}")
        print(f"  Avg magnitude: {best_avg_mean:.2f}")
        print(f"  Est. FP rate: {best_fp_rate * 100:.1f}%")
        
        if self._filtered_count > 0:
            print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
        
        return best_band, best_mv_values
