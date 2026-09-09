# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Console Output

Shared console formatting helpers for live motion detection output.

Author: Francesco Pace <francesco.pace@gmail.com>
"""


def print_log(level, message):
    """Print one operational console line with a stable ``[LEVEL]`` prefix.

    Use ``INFO``, ``WARN``, or ``ERROR``. Keep banners, calibration ceremony, and status bars on plain ``print()``.
    """
    print("[{}] {}".format(level, message))


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
    filled_char="#",
    empty_char="-",
    threshold_char="|",
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

    return bar


def _lookup_value(diagnostics, key, default=None):
    if diagnostics is None:
        return default
    if isinstance(diagnostics, dict):
        return diagnostics.get(key, default)
    return getattr(diagnostics, key, default)


def _format_metric_value(value, *, placeholder="--"):
    if value is None:
        return placeholder
    return f"{float(value):.6f}"


def _format_integer_value(value, *, placeholder="--"):
    if value is None:
        return placeholder
    return str(int(value))


def _format_status_fields(diagnostics, *, placeholders=False):
    def rate(key):
        value = None if placeholders else _lookup_value(diagnostics, key, None)
        return "--" if value is None else "{:.1f}".format(float(value))

    occupancy = None if placeholders else _lookup_value(diagnostics, "csi_occupancy", None)
    channel = None if placeholders else _lookup_value(diagnostics, "wifi_channel", None)
    rssi = None if placeholders else _lookup_value(diagnostics, "wifi_rssi_dbm", None)
    occupancy_text = "--" if occupancy is None else str(int(float(occupancy) * 100.0 + 0.5))
    return "tx:{} cb:{} accepted:{} hwerr:{} occ:{}% | ch:{} rssi:{}".format(
        rate("traffic_tx_pps"),
        rate("csi_callback_pps"),
        rate("csi_accepted_pps"),
        # Older firmware does not expose the dedicated hardware counter.
        rate("csi_hw_error_pps"),
        occupancy_text,
        _format_integer_value(channel if channel else None),
        _format_integer_value(rssi),
    )


def _format_status_line(
    *,
    progress,
    threshold_pos,
    motion_metric,
    threshold,
    state_label,
    diagnostics,
    device_label=None,
    width=20,
    filled_char="#",
    empty_char="-",
    threshold_char="|",
    placeholder_metrics=False,
):
    progress_bar = format_progress_bar(
        progress,
        width=width,
        threshold_pos=threshold_pos,
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
    )
    display_motion_metric = None if placeholder_metrics else motion_metric
    display_threshold = None if placeholder_metrics else threshold
    line = "{} | mvmt:{} thr:{} | {} | {}".format(
        progress_bar,
        _format_metric_value(display_motion_metric),
        _format_metric_value(display_threshold),
        state_label,
        _format_status_fields(diagnostics, placeholders=placeholder_metrics),
    )
    if device_label:
        return f"{device_label} | {line}"
    return line


def format_detection_publish_line(
    *,
    diagnostics=None,
    motion_metric,
    threshold,
    effective_state,
    device_label=None,
    width=20,
    filled_char="#",
    empty_char="-",
    threshold_char="|",
):
    """Build the shared runtime-style live publish log line."""
    state_str = "MOTION" if effective_state == 1 else "IDLE"
    return _format_status_line(
        progress=_clamp_unit_interval(motion_metric),
        threshold_pos=_threshold_marker_index(threshold, width),
        motion_metric=motion_metric,
        threshold=threshold,
        state_label=state_str,
        diagnostics=diagnostics,
        device_label=device_label,
        width=width,
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
    )


def format_calibration_status_line(
    *,
    progress,
    motion_metric,
    threshold,
    diagnostics=None,
    effective_state_label="CALIBRATING",
    device_label=None,
    width=20,
    filled_char="#",
    empty_char="-",
    threshold_char="|",
):
    """Build a shared calibration progress line."""
    return _format_status_line(
        progress=progress,
        threshold_pos=-1,
        motion_metric=motion_metric,
        threshold=threshold,
        state_label=effective_state_label,
        diagnostics=diagnostics,
        device_label=device_label,
        width=width,
        filled_char=filled_char,
        empty_char=empty_char,
        threshold_char=threshold_char,
    )


def format_waiting_status_line(
    *,
    device_label=None,
    metric_placeholder="--",
    threshold_placeholder="--",
    state_label="WAITING",
    width=20,
    threshold_pos=-1,
    filled_char="#",
    empty_char="-",
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
    line = "{} | mvmt:{} thr:{} | {} | {}".format(
        progress_bar,
        metric_placeholder,
        threshold_placeholder,
        state_label,
        _format_status_fields(None, placeholders=True),
    )
    if device_label:
        return f"{device_label} | {line}"
    return line
