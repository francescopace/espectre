"""
Micro-ESPectre - Device Utils

Minimal allocation-conscious helpers used by the MicroPython runtime.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

try:
    from src.serial_sequence import SerialSequenceTracker
except ImportError:
    from serial_sequence import SerialSequenceTracker

HT20_CSI_LEN = 128
HT20_CSI_LEN_SHORT = 114
HT20_CSI_LEN_SHORT_DOUBLE = 228
HT20_CSI_SHORT_LEFT_PAD = 8
HT20_CSI_SHORT_COPY_END = HT20_CSI_SHORT_LEFT_PAD + HT20_CSI_LEN_SHORT
HT20_CSI_SHORT_RIGHT_PAD = HT20_CSI_LEN - HT20_CSI_SHORT_COPY_END
HT20_CSI_SHORT_LEFT_ZEROS = b"\x00" * HT20_CSI_SHORT_LEFT_PAD
HT20_CSI_SHORT_RIGHT_ZEROS = b"\x00" * HT20_CSI_SHORT_RIGHT_PAD

_CSI_READ_SUPPORTS_REUSE = None


class CsiFrameTimestampFilter:
    """Reject duplicate or stale CSI frames using the Wi-Fi RX timestamp."""

    def __init__(self):
        self._tracker = SerialSequenceTracker()

    def accept(self, frame):
        if frame is None or len(frame) <= 4:
            return True
        timestamp = frame[4]
        if timestamp is None or timestamp == 0:
            return True
        try:
            return self._tracker.observe(timestamp) >= 0
        except (TypeError, ValueError):
            return True

    def reset(self):
        self._tracker.reset()


def csi_read_frame(wlan, reuse_frame=None):
    """Read one CSI frame, reusing the previous result when supported."""
    global _CSI_READ_SUPPORTS_REUSE

    if _CSI_READ_SUPPORTS_REUSE is not False:
        try:
            frame = wlan.csi_read(reuse_frame)
            _CSI_READ_SUPPORTS_REUSE = True
            return frame
        except TypeError:
            _CSI_READ_SUPPORTS_REUSE = False

    return wlan.csi_read()


def is_ht20_sensing_frame(frame):
    """Return True when a CSI frame matches the HT20 sensing contract.

    MicroPython ``wlan.csi_read()`` list layout:
    index 7 = ``sig_mode`` (0=legacy, 1=HT, 3=VHT), index 9 = ``cwb``
    (0=20 MHz, 1=40 MHz). Frames without those fields are treated as HT20 for
    older firmware compatibility (same as host NPZ without PHY metadata).
    """
    if frame is None:
        return False
    try:
        length = len(frame)
    except TypeError:
        return False
    if length <= 9:
        return True
    return frame[7] == 1 and frame[9] == 0


def normalize_ht20_csi_payload(csi_data, expected_len=128, remap_buffer=None):
    """Normalize supported CSI payload layouts to one HT20 payload."""
    raw_len = len(csi_data)
    input_len = raw_len

    if raw_len == expected_len * 2:
        if remap_buffer is not None and len(remap_buffer) == expected_len:
            remap_buffer[:] = memoryview(csi_data)[:expected_len]
            return remap_buffer, input_len, "double_ht20"
        return csi_data[:expected_len], input_len, "double_ht20"

    if raw_len == expected_len:
        return csi_data, input_len, None

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

        remap_buffer[:HT20_CSI_SHORT_LEFT_PAD] = HT20_CSI_SHORT_LEFT_ZEROS
        remap_buffer[HT20_CSI_SHORT_COPY_END:] = HT20_CSI_SHORT_RIGHT_ZEROS
        remap_buffer[HT20_CSI_SHORT_LEFT_PAD:HT20_CSI_SHORT_COPY_END] = csi_data
        if short_double_collapsed:
            return remap_buffer, input_len, "double_ht57_and_remap"
        return remap_buffer, input_len, "ht57_to_64"

    return None, input_len, None


def to_signed_int8(value):
    """Convert an unsigned byte to its signed int8 value."""
    return value if value < 128 else value - 256


def insertion_sort(arr, n):
    """Sort the first ``n`` elements of a small pre-allocated list in place."""
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def calculate_variance(values):
    """Calculate population variance with a numerically stable two-pass loop."""
    n = len(values)
    if n == 0:
        return 0.0

    mean = sum(values) / n
    variance_sum = 0.0
    for value in values:
        diff = value - mean
        variance_sum += diff * diff
    return variance_sum / n
