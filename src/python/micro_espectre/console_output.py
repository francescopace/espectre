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
    placeholder = "--" if placeholders else "0"
    admitted = _lookup_value(diagnostics, "csi_admitted_pps", 0.0)
    accepted = _lookup_value(diagnostics, "csi_accepted_pps", 0.0)
    traffic = _lookup_value(diagnostics, "traffic_tx_pps", 0.0)
    occupancy = _lookup_value(diagnostics, "csi_occupancy", 0.0)
    missing = _lookup_value(diagnostics, "csi_missing_slots_pps", 0.0)
    excess = _lookup_value(diagnostics, "csi_excess_pps", 0.0)
    stale = _lookup_value(diagnostics, "csi_stale_pps", 0.0)
    out_of_order = _lookup_value(diagnostics, "csi_out_of_order_pps", 0.0)
    channel = _lookup_value(diagnostics, "wifi_channel", 0)
    rssi = _lookup_value(diagnostics, "wifi_rssi_dbm", None)

    occupancy_text = placeholder
    if not placeholders:
        occupancy_text = str(int(float(occupancy) * 100.0 + 0.5))

    return (
        f"csi:{_format_integer_value(admitted if not placeholders else None, placeholder=placeholder)}/"
        f"{_format_integer_value(accepted if not placeholders else None, placeholder=placeholder)} "
        f"tx:{_format_integer_value(traffic if not placeholders else None, placeholder=placeholder)} "
        f"occ:{occupancy_text}% "
        f"miss:{_format_integer_value(missing if not placeholders else None, placeholder=placeholder)} "
        f"excess:{_format_integer_value(excess if not placeholders else None, placeholder=placeholder)} "
        f"stale:{_format_integer_value(stale if not placeholders else None, placeholder=placeholder)} "
        f"ooo:{_format_integer_value(out_of_order if not placeholders else None, placeholder=placeholder)} "
        f"| ch:{_format_integer_value(channel if not placeholders else None, placeholder=placeholder)} "
        f"rssi:{_format_integer_value(rssi if not placeholders else None, placeholder=placeholder)}"
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
    line = (
        f"{progress_bar} | mvmt:{_format_metric_value(None if placeholder_metrics else motion_metric)} "
        f"thr:{_format_metric_value(None if placeholder_metrics else threshold)} | {state_label} | "
        f"{_format_status_fields(diagnostics, placeholders=placeholder_metrics)}"
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
    line = (
        f"{progress_bar} | mvmt:{metric_placeholder} thr:{threshold_placeholder} | "
        f"{state_label} | {_format_status_fields(None, placeholders=True)}"
    )
    if device_label:
        return f"{device_label} | {line}"
    return line
