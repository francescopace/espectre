"""
ESPectre - Segmentation Tests

Unit tests for shared turbulence-context helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import numpy as np
from config import SEGMENTATION_WINDOW_SIZE_MS
from runtime_policy import derive_detector_timing, nominal_packet_interval_us
from segmentation import SegmentationContext

DEFAULT_WINDOW_PACKETS = derive_detector_timing(
    nominal_packet_interval_us(100), SEGMENTATION_WINDOW_SIZE_MS
)["window_packets"]


class TestSegmentationContextInit:
    """Test SegmentationContext initialization"""

    def test_default_parameters(self):
        """Test default parameters (matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)"""
        ctx = SegmentationContext()
        assert ctx.window_size == DEFAULT_WINDOW_PACKETS
        assert ctx.buffer_count == 0

    def test_custom_parameters(self):
        """Test custom parameters"""
        ctx = SegmentationContext(
            window_size=100,
            enable_hampel=False
        )
        assert ctx.window_size == 100
        assert ctx.hampel_filter is None

    def test_buffer_pre_allocation(self):
        """Test that turbulence buffer is pre-allocated"""
        ctx = SegmentationContext(window_size=DEFAULT_WINDOW_PACKETS)
        assert len(ctx.turbulence_buffer) == DEFAULT_WINDOW_PACKETS

    def test_hampel_enabled_by_default(self):
        """Test that Hampel filter is enabled by default"""
        ctx = SegmentationContext()
        assert ctx.hampel_filter is not None

    def test_hampel_enabled(self):
        """Test Hampel filter initialization when enabled"""
        ctx = SegmentationContext(
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        assert ctx.hampel_filter is not None

    def test_lowpass_disabled_by_default(self):
        """Test that low-pass filter is disabled by default"""
        ctx = SegmentationContext()
        assert ctx.lowpass_filter is None

    def test_lowpass_enabled(self):
        """Test low-pass filter initialization when enabled"""
        ctx = SegmentationContext(
            enable_lowpass=True,
            lowpass_cutoff=11.5
        )
        assert ctx.lowpass_filter is not None
        assert ctx.lowpass_filter.cutoff_hz == 11.5


class TestComputeVarianceTwoPass:
    """Test the static two-pass variance calculation"""

    def test_empty_list(self):
        """Test variance of empty list"""
        result = SegmentationContext.compute_variance_two_pass([])
        assert result == 0.0

    def test_single_value(self):
        """Test variance of single value"""
        result = SegmentationContext.compute_variance_two_pass([5.0])
        assert result == 0.0

    def test_constant_values(self):
        """Test variance of constant values"""
        result = SegmentationContext.compute_variance_two_pass([10.0] * 100)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_known_variance(self):
        """Test with known variance"""
        # Values 1, 2, 3, 4, 5 have variance = 2.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = SegmentationContext.compute_variance_two_pass(values)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_matches_numpy(self):
        """Test that result matches numpy variance"""
        np.random.seed(42)
        values = list(np.random.normal(50, 15, 100))

        result = SegmentationContext.compute_variance_two_pass(values)
        expected = np.var(values)

        assert result == pytest.approx(expected, rel=1e-6)


class TestComputeSpatialTurbulence:
    """Test the static spatial turbulence calculation"""

    def test_empty_data(self):
        """Test with empty CSI data"""
        turb, amps = SegmentationContext.compute_spatial_turbulence([])
        assert turb == 0.0
        assert amps == []

    def test_minimal_data(self):
        """Test with minimal CSI data"""
        turb, amps = SegmentationContext.compute_spatial_turbulence([0])
        assert turb == 0.0

    def test_single_subcarrier(self):
        """Test with single subcarrier (I, Q)"""
        # I=3, Q=4 -> amplitude = 5
        turb, amps = SegmentationContext.compute_spatial_turbulence([3, 4])
        assert len(amps) == 1
        assert amps[0] == pytest.approx(5.0, rel=1e-6)

    def test_multiple_subcarriers(self):
        """Test with multiple subcarriers"""
        # 4 subcarriers with I/Q pairs
        csi_data = [3, 4, 6, 8, 5, 12, 8, 15]  # Amplitudes: 5, 10, 13, 17
        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data)

        assert len(amps) == 4
        assert amps[0] == pytest.approx(5.0, rel=1e-6)
        assert amps[1] == pytest.approx(10.0, rel=1e-6)

    def test_selected_subcarriers(self):
        """Test with selected subcarriers only"""
        # 8 subcarriers, select only indices 0, 2, 3
        csi_data = [3, 4, 6, 8, 5, 12, 8, 15, 0, 0, 0, 0, 0, 0, 0, 0]
        selected = [0, 2, 3]

        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data, selected)

        assert len(amps) == 3

    def test_turbulence_is_std(self):
        """Test that turbulence equals standard deviation of amplitudes"""
        # Create data with known std
        # I=10, Q=0 for all -> all amplitudes = 10
        csi_data = [10, 0] * 10
        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data)

        # All amplitudes equal -> std = 0
        assert turb == pytest.approx(0.0, abs=1e-6)


class TestAddTurbulence:
    """Test the add_turbulence buffer path"""

    def test_buffer_filling(self):
        """Test that buffer fills correctly"""
        ctx = SegmentationContext(window_size=10)

        for i in range(10):
            ctx.add_turbulence(float(i))

        assert ctx.buffer_count == 10

    def test_circular_buffer(self):
        """Test circular buffer behavior"""
        ctx = SegmentationContext(window_size=5)

        # Fill with initial values
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)

        # Add more values - should overwrite oldest
        for v in [6.0, 7.0]:
            ctx.add_turbulence(v)

        # Buffer should now contain [6, 7, 3, 4, 5] in some order
        assert ctx.buffer_count == 5

    def test_no_normalization_applied(self):
        """Test that turbulence is NOT normalized"""
        ctx = SegmentationContext(window_size=5, enable_hampel=False)

        # Add turbulence - should NOT be scaled
        ctx.add_turbulence(5.0)

        # last_turbulence should be 5.0 (no normalization)
        assert ctx.last_turbulence == pytest.approx(5.0, rel=1e-6)


class TestReset:
    """Test reset functionality"""

    def test_soft_reset(self):
        """Test soft reset (keep buffer)"""
        ctx = SegmentationContext(window_size=5)

        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)

        ctx.reset(full=False)

        # Buffer should still have data
        assert ctx.buffer_count == 5

    def test_full_reset(self):
        """Test full reset (clear buffer)"""
        ctx = SegmentationContext(window_size=5)

        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)

        ctx.reset(full=True)

        assert ctx.buffer_count == 0
        assert ctx.last_turbulence == 0.0


class TestHampelIntegration:
    """Test integration with Hampel filter"""

    def test_hampel_filters_outliers(self):
        """Test that Hampel filter removes outliers"""
        ctx = SegmentationContext(
            window_size=10,
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )

        # Add values with some variance (needed for MAD calculation)
        for v in [5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 4.9]:
            ctx.add_turbulence(v)

        # Add extreme outlier
        ctx.add_turbulence(100.0)

        # Outlier should be filtered (replaced with median ~5.0)
        assert ctx.last_turbulence < 100.0


class TestLowPassIntegration:
    """Test integration with low-pass filter"""

    def test_lowpass_smooths_signal(self):
        """Test that low-pass filter smooths high-frequency noise"""
        ctx = SegmentationContext(
            window_size=50,
            enable_lowpass=True,
            lowpass_cutoff=10.0
        )

        # Generate noisy signal: base + high-freq noise
        np.random.seed(42)
        baseline = 5.0
        noise = np.random.randn(50) * 2.0
        signal = baseline + noise

        for v in signal:
            ctx.add_turbulence(v)

        # The filtered value should be closer to baseline than the noisy input
        assert 3.0 < ctx.last_turbulence < 7.0

    def test_lowpass_preserves_dc(self):
        """Test that low-pass filter preserves DC component"""
        ctx = SegmentationContext(
            window_size=50,
            enable_lowpass=True,
            lowpass_cutoff=10.0
        )

        # Feed constant value
        for _ in range(30):
            ctx.add_turbulence(5.0)

        # Should pass through unchanged
        assert ctx.last_turbulence == pytest.approx(5.0, rel=0.01)

    def test_filter_chain_order(self):
        """Test that filter chain applies: hampel → lowpass (no normalization)"""
        ctx = SegmentationContext(
            window_size=10,
            enable_lowpass=True,
            lowpass_cutoff=10.0,
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )

        # Feed values to initialize filter
        for v in [3.0, 3.1, 2.9, 3.0, 3.2]:
            ctx.add_turbulence(v)

        # Feed normal value (no normalization, just filtering)
        ctx.add_turbulence(3.0)

        # Output should be around 3.0 (slightly smoothed by lowpass)
        assert 2.5 < ctx.last_turbulence < 3.5


class TestCalculateSpatialTurbulence:
    """Test the instance method calculate_spatial_turbulence"""

    def test_shared_packet_frame_matches_direct_csi_extraction(self):
        csi_data = np.asarray(
            [((index * 37 + 91) % 255) - 127 for index in range(128)],
            dtype=np.int8,
        )
        selected = (4, 8, 12, 16, 20, 24, 28, 36, 40, 48, 56, 60)
        frame = [0.0] * 64
        count = SegmentationContext.fill_subcarrier_energy_buffer(
            csi_data,
            frame,
        )
        SegmentationContext.energies_to_amplitudes_in_place(frame, count)

        for aggregation_width in (None, 5):
            direct = SegmentationContext(
                adjacent_aggregation_width=aggregation_width,
            )
            shared = SegmentationContext(
                adjacent_aggregation_width=aggregation_width,
            )

            direct_value, direct_amplitudes = direct.calculate_spatial_turbulence(
                csi_data,
                selected,
                return_amplitudes=True,
            )
            shared_value = (
                shared.calculate_spatial_turbulence_from_subcarrier_amplitudes(
                    frame,
                    count,
                    selected,
                )
            )

            assert shared_value == pytest.approx(direct_value, abs=1e-12)
            assert shared._amplitude_buffer[:shared._amplitude_count] == (
                pytest.approx(direct_amplitudes, abs=1e-12)
            )

    def test_stores_amplitudes(self, synthetic_csi_packet, default_subcarriers):
        """Test that amplitudes are available when explicitly requested"""
        ctx = SegmentationContext()

        turb, amps = ctx.calculate_spatial_turbulence(
            synthetic_csi_packet, default_subcarriers, return_amplitudes=True
        )

        assert len(amps) == len(default_subcarriers)
        assert ctx._amplitude_count == len(default_subcarriers)
        assert turb >= 0.0

    def test_w5_adjacent_aggregation_averages_live_bin_magnitudes(self):
        csi_data = np.zeros(128, dtype=np.int8)
        for subcarrier in range(64):
            csi_data[subcarrier * 2 + 1] = subcarrier
        ctx = SegmentationContext(adjacent_aggregation_width=5)

        _, amplitudes = ctx.calculate_spatial_turbulence(
            csi_data,
            (4, 28, 36, 60),
            return_amplitudes=True,
        )

        assert amplitudes == pytest.approx([6.0, 28.0, 36.0, 58.0])


class TestEndToEnd:
    """End-to-end turbulence-buffer integration tests"""

    def test_baseline_fills_buffer(self, synthetic_csi_baseline_packets, default_subcarriers):
        """Test that baseline packets populate the turbulence buffer"""
        ctx = SegmentationContext(window_size=50)

        for pkt in synthetic_csi_baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            ctx.add_turbulence(turb)

        assert ctx.buffer_count == min(50, len(synthetic_csi_baseline_packets))
        assert ctx.last_turbulence >= 0.0

    def test_movement_fills_buffer(self, synthetic_csi_movement_packets, default_subcarriers):
        """Test that movement packets populate the turbulence buffer"""
        ctx = SegmentationContext(window_size=50)

        for pkt in synthetic_csi_movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            ctx.add_turbulence(turb)

        assert ctx.buffer_count == min(50, len(synthetic_csi_movement_packets))
        assert ctx.last_turbulence >= 0.0
