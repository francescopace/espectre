# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Console Output

Shared console formatting helpers for live motion detection output.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

def _clamp_unit_interval(value):
    """Clamp a metric onto the shared 0-1 display scale."""
    numeric = float(value)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _threshold_marker_index(threshold, width):
    """Map a 0-1 threshold onto a bar index, or -1 when no marker should be shown."""
    if threshold <= 0.0 or width < 1:
        return -1
    pos = int(threshold * width + 0.5)
    if pos >= width:
        return width - 1
    if pos < 0:
        return 0
    return pos


def format_progress_bar(
    progress,
    width=20,
    threshold_pos=-1,
    *,
    filled_char="█",
    empty_char="░",
    threshold_char="|",
    include_percent=False,
):
    """Format the runtime-style progress bar for console output.

    ``progress`` fills the bar on a 0-1 scale of ``width``. ``threshold_pos`` overlays a marker at that character index; pass a negative value to hide it.
    """
    if width < 1:
        width = 1
    elif width > 20:
        width = 20
    show_threshold = threshold_pos >= 0
    if show_threshold and threshold_pos >= width:
        threshold_pos = width - 1

    filled = int(progress * width)
    filled = max(0, min(filled, width))

    bar = "["
    for idx in range(width):
        if show_threshold and idx == threshold_pos:
            bar += threshold_char
        elif idx < filled:
            bar += filled_char
        else:
            bar += empty_char
    bar += "]"

    if include_percent:
        percent = int(progress * 100)
        return f"{bar} {percent:>3d}%"
    return bar


def _format_drop_text(*, packet_count=None, dropped_count=None):
    """Format the shared drop-rate suffix when packet counters are available."""
    if packet_count is None or dropped_count is None:
        return ""
    total_expected = max(int(packet_count) + int(dropped_count), 1)
    drop_rate = (float(dropped_count) / float(total_expected)) * 100.0
    return f" | drop {drop_rate:.1f}%"


def format_detection_publish_line(
    *,
    packet_count=None,
    dropped_count=None,
    pps,
    motion_metric,
    threshold,
    effective_state,
    device_label=None,
    width=20,
    filled_char="█",
    empty_char="░",
    threshold_char="|",
):
    """Build the shared runtime-style live publish log line."""
    progress_bar = format_progress_bar(
        _clamp_unit_interval(motion_metric),
        width=width,
        threshold_pos=_threshold_marker_index(threshold, width),
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
    )
    state_str = "MOTION" if effective_state == 1 else "IDLE"
    drop_text = _format_drop_text(packet_count=packet_count, dropped_count=dropped_count)
    line = (
        f"{progress_bar} | mvmt:{motion_metric:.6f} "
        f"thr:{threshold:.6f} | {state_str} | {pps} pkt/s{drop_text}"
    )
    if device_label:
        return f"{device_label} | {line}"
    return line


def format_calibration_status_line(
    *,
    progress,
    pps,
    packet_count=None,
    dropped_count=None,
    motion_metric=None,
    calibration_packets=None,
    calibration_target_packets=None,
    effective_state_label="CALIBRATING",
    device_label=None,
    width=20,
    threshold_pos=-1,
    filled_char="█",
    empty_char="░",
    threshold_char="|",
):
    """Build a shared calibration progress line."""
    progress_bar = format_progress_bar(
        progress,
        width=width,
        threshold_pos=threshold_pos,
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
        include_percent=True,
    )
    packets_text = ""
    if calibration_packets is not None and calibration_target_packets is not None:
        packets_text = f" pkt:{calibration_packets}/{calibration_target_packets}"
    drop_text = _format_drop_text(packet_count=packet_count, dropped_count=dropped_count)

    line = f"{progress_bar} |{packets_text} | {effective_state_label} | {pps} pkt/s{drop_text}"
    if device_label:
        return f"{device_label} | {line}"
    return line


def format_waiting_status_line(
    *,
    device_label,
    pps_placeholder="--",
    metric_placeholder="--",
    threshold_placeholder="--",
    state_label="WAITING",
    width=20,
    threshold_pos=-1,
    filled_char="█",
    empty_char="░",
    threshold_char="|",
):
    """Build a placeholder line that matches the standard status layout."""
    progress_bar = format_progress_bar(
        0.0,
        width=width,
        threshold_pos=threshold_pos,
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
    )
    return (
        f"{device_label} | {progress_bar} | "
        f"mvmt:{metric_placeholder} thr:{threshold_placeholder} | "
        f"{state_label} | {pps_placeholder} pkt/s"
    )
