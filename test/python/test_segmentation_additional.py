"""
ESPectre - Additional Segmentation Tests

Additional unit tests for segmentation helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from unittest.mock import MagicMock

from config import SEG_WINDOW_SIZE
from segmentation import SegmentationContext


class TestSegmentationFilterErrors:
    """Test error handling for filter initialization failures"""

    def test_lowpass_init_success(self):
        """Test that lowpass filter initializes correctly"""
        ctx = SegmentationContext(enable_lowpass=True)
        # Should have lowpass filter initialized
        assert ctx.lowpass_filter is not None
        assert ctx.window_size == SEG_WINDOW_SIZE  # Matches C++ DETECTOR_DEFAULT_WINDOW_SIZE

    def test_hampel_init_success(self):
        """Test that hampel filter initializes correctly"""
        ctx = SegmentationContext(enable_hampel=True)
        assert ctx.hampel_filter is not None


class TestSegmentationFilterChainErrors:
    """Test error handling in filter chain during add_turbulence"""

    def test_lowpass_filter_exception(self):
        """Test handling of lowpass filter exception during add_turbulence"""
        ctx = SegmentationContext(enable_lowpass=True)

        if ctx.lowpass_filter is not None:
            # Force filter to raise exception
            ctx.lowpass_filter.filter = MagicMock(side_effect=Exception("Filter error"))

            # Should not raise exception, should pass through raw value
            ctx.add_turbulence(5.0)

            # Value should still be stored (raw or normalized)
            assert ctx.last_turbulence >= 0

    def test_hampel_filter_exception(self):
        """Test handling of hampel filter exception during add_turbulence"""
        ctx = SegmentationContext(enable_hampel=True)

        if ctx.hampel_filter is not None:
            # Force filter to raise exception
            ctx.hampel_filter.filter = MagicMock(side_effect=Exception("Filter error"))

            # Should not raise exception
            ctx.add_turbulence(5.0)

            # Value should still be stored
            assert ctx.last_turbulence >= 0


class TestSegmentationResetWithFilters:
    """Test reset with filters enabled"""

    def test_full_reset_with_lowpass(self):
        """Test full reset resets lowpass filter"""
        ctx = SegmentationContext(enable_lowpass=True)

        # Add some values to warm up filter
        for i in range(10):
            ctx.add_turbulence(float(i))

        # Full reset
        ctx.reset(full=True)

        assert ctx.buffer_count == 0
        # Filter should be reset (internal state cleared)
        if ctx.lowpass_filter:
            assert ctx.lowpass_filter.initialized is False

    def test_full_reset_with_hampel(self):
        """Test full reset resets hampel filter"""
        ctx = SegmentationContext(enable_hampel=True)

        # Add some values
        for i in range(10):
            ctx.add_turbulence(float(i))

        # Full reset
        ctx.reset(full=True)

        assert ctx.buffer_count == 0
        # Hampel filter should be reset
        if ctx.hampel_filter:
            assert ctx.hampel_filter.count == 0


class TestSegmentationEdgeCases:
    """Test edge cases in segmentation"""

    def test_compute_spatial_turbulence_with_single_subcarrier(self):
        """Test spatial turbulence with only one subcarrier selected"""
        csi_data = [3, 4] * 64  # All same amplitude
        selected = [0]  # Only one subcarrier

        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data, selected)

        assert len(amps) == 1
        # Turbulence should be 0 (can't compute std of single value)
        assert turb == 0.0

    def test_partial_buffer_keeps_count(self):
        """Test buffer count before the window is full"""
        ctx = SegmentationContext(window_size=5)

        ctx.add_turbulence(5.0)

        assert ctx.buffer_count == 1
        assert ctx.last_turbulence == 5.0
