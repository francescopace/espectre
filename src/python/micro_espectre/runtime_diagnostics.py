# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Runtime Diagnostics

Rate and link diagnostics derived from cumulative runtime counters.
Mirrors `src/cpp/runtime/runtime_diagnostics.cpp` for MQTT `stats`.
"""
import time

STATS_DIAGNOSTIC_KEYS = (
    "traffic_tx_pps",
    "csi_callback_pps",
    "csi_accepted_pps",
    "csi_admitted_pps",
    "csi_filtered_pps",
    "csi_missing_slots_pps",
    "csi_excess_pps",
    "csi_stale_pps",
    "csi_out_of_order_pps",
    "csi_occupancy",
    "wifi_channel",
    "wifi_rssi_dbm",
)


def _ticks_diff(new, old):
    diff_fn = getattr(time, "ticks_diff", None)
    return diff_fn(new, old) if diff_fn is not None else new - old


def _counter_delta(current, previous):
    current = int(current)
    previous = int(previous)
    return current - previous if current >= previous else current


def _packets_per_second(delta, elapsed_ms):
    if elapsed_ms <= 0:
        return 0.0
    return (float(delta) * 1000.0) / float(elapsed_ms)


def empty_diagnostics_sample(wifi_channel=0, wifi_rssi_dbm=None):
    """Return the MQTT `stats` CSI/Wi-Fi fields with zero rates."""
    return {
        "traffic_tx_pps": 0.0,
        "csi_callback_pps": 0.0,
        "csi_accepted_pps": 0.0,
        "csi_admitted_pps": 0.0,
        "csi_filtered_pps": 0.0,
        "csi_missing_slots_pps": 0.0,
        "csi_excess_pps": 0.0,
        "csi_stale_pps": 0.0,
        "csi_out_of_order_pps": 0.0,
        "csi_occupancy": 0.0,
        "wifi_channel": int(wifi_channel or 0),
        "wifi_rssi_dbm": wifi_rssi_dbm,
    }


def wifi_rssi_dbm(wlan):
    """Return association RSSI in dBm, or None when unavailable."""
    if wlan is None:
        return None
    try:
        isconnected = getattr(wlan, "isconnected", None)
        if callable(isconnected) and not isconnected():
            return None
        status = getattr(wlan, "status", None)
        if not callable(status):
            return None
        rssi = status("rssi")
        if rssi is None:
            return None
        return int(rssi)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def wifi_csi_dropped(wlan):
    """Return the cumulative native CSI ring-buffer drop count."""
    if wlan is None:
        return 0
    try:
        get_dropped = getattr(wlan, "csi_dropped", None)
        if not callable(get_dropped):
            return 0
        return max(0, int(get_dropped()))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def collect_runtime_diagnostics_snapshot(
    traffic_generator=None,
    callback_total=0,
    accepted_total=0,
    admitted_total=0,
    filtered_total=0,
    missing_slots_total=0,
    excess_total=0,
    stale_total=0,
    out_of_order_total=0,
    occupancy_slots=0,
    window_slots=0,
    wifi_channel=0,
    rssi_dbm=None,
):
    """Build the cumulative counter snapshot consumed by the rate sampler."""
    traffic_packets_total = 0
    if traffic_generator is not None:
        get_count = getattr(traffic_generator, "get_packet_count", None)
        if callable(get_count):
            try:
                traffic_packets_total = int(get_count())
            except (TypeError, ValueError):
                traffic_packets_total = 0
    return {
        "traffic_packets_total": int(traffic_packets_total),
        "csi_callbacks_total": int(callback_total),
        "csi_accepted_total": int(accepted_total),
        "csi_admitted_total": int(admitted_total),
        "csi_filtered_total": int(filtered_total),
        "csi_missing_slots_total": int(missing_slots_total),
        "csi_excess_total": int(excess_total),
        "csi_stale_total": int(stale_total),
        "csi_out_of_order_total": int(out_of_order_total),
        "csi_occupancy_slots": int(occupancy_slots),
        "csi_window_slots": int(window_slots),
        "wifi_channel": int(wifi_channel or 0),
        "wifi_rssi_dbm": rssi_dbm,
    }


def apply_diagnostics_sample(payload, sample, wifi_channel=0, rssi_dbm=None):
    """Copy MQTT diagnostic keys into `payload`, filling zeros for missing ones."""
    defaults = empty_diagnostics_sample(wifi_channel=wifi_channel, wifi_rssi_dbm=rssi_dbm)
    source = sample if isinstance(sample, dict) else defaults
    for key in STATS_DIAGNOSTIC_KEYS:
        if key in source:
            payload[key] = source[key]
        else:
            payload[key] = defaults[key]
    return payload


class RuntimeDiagnosticsSampler:
    """Convert cumulative counters into rates over the interval between reads."""

    def __init__(self):
        self._previous = None
        self._previous_ms = 0
        self._baseline_ready = False

    def reset(self, snapshot, now_ms):
        self._previous = snapshot
        self._previous_ms = int(now_ms)
        self._baseline_ready = True

    def sample(self, snapshot, now_ms):
        now_ms = int(now_ms)
        result = empty_diagnostics_sample(
            wifi_channel=snapshot.get("wifi_channel", 0),
            wifi_rssi_dbm=snapshot.get("wifi_rssi_dbm"),
        )
        if not self._baseline_ready:
            self.reset(snapshot, now_ms)
            return result

        elapsed_ms = _ticks_diff(now_ms, self._previous_ms)
        if elapsed_ms <= 0:
            return result

        previous = self._previous
        result["traffic_tx_pps"] = _packets_per_second(
            _counter_delta(snapshot["traffic_packets_total"], previous["traffic_packets_total"]),
            elapsed_ms,
        )
        result["csi_callback_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_callbacks_total"], previous["csi_callbacks_total"]),
            elapsed_ms,
        )
        result["csi_accepted_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_accepted_total"], previous["csi_accepted_total"]),
            elapsed_ms,
        )
        result["csi_admitted_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_admitted_total"], previous["csi_admitted_total"]),
            elapsed_ms,
        )
        result["csi_filtered_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_filtered_total"], previous["csi_filtered_total"]),
            elapsed_ms,
        )
        result["csi_missing_slots_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_missing_slots_total"], previous["csi_missing_slots_total"]),
            elapsed_ms,
        )
        result["csi_excess_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_excess_total"], previous["csi_excess_total"]),
            elapsed_ms,
        )
        result["csi_stale_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_stale_total"], previous["csi_stale_total"]),
            elapsed_ms,
        )
        result["csi_out_of_order_pps"] = _packets_per_second(
            _counter_delta(snapshot["csi_out_of_order_total"], previous["csi_out_of_order_total"]),
            elapsed_ms,
        )
        window_slots = int(snapshot.get("csi_window_slots", 0) or 0)
        occupancy_slots = int(snapshot.get("csi_occupancy_slots", 0) or 0)
        result["csi_occupancy"] = (
            occupancy_slots / float(window_slots) if window_slots > 0 else 0.0
        )
        self.reset(snapshot, now_ms)
        return result


class RuntimeDebugTelemetry:
    """Aggregate optional loop, detector, and heap benchmark diagnostics."""

    LOG_INTERVAL_MS = 10_000

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self._minimum_heap_free = None
        self.reset()

    def reset(self):
        """Clear the current timing window while retaining the heap low-water mark."""
        self._window_start_ms = None
        self._loop_busy_us = 0
        self._loop_duration_sum_us = 0
        self._loop_duration_max_us = 0
        self._loop_samples = 0
        self._detection_duration_sum_us = 0
        self._detection_duration_min_us = 0
        self._detection_duration_max_us = 0
        self._detection_samples = 0
        self._packet_duration_sum_us = 0
        self._packet_duration_min_us = 0
        self._packet_duration_max_us = 0
        self._packet_samples = 0

    def is_due(self, now_ms):
        """Return whether the next call to format_if_due() will emit a sample."""
        if not self.enabled or self._window_start_ms is None:
            return False
        return _ticks_diff(int(now_ms), self._window_start_ms) >= self.LOG_INTERVAL_MS

    def record_loop_duration(self, duration_us):
        """Record one measured main-loop body duration."""
        if not self.enabled:
            return
        duration_us = max(0, int(duration_us))
        self._loop_busy_us += duration_us
        self._loop_duration_sum_us += duration_us
        self._loop_duration_max_us = max(self._loop_duration_max_us, duration_us)
        self._loop_samples += 1

    def record_detection_duration(self, duration_us):
        """Record one detector evaluation duration."""
        if not self.enabled:
            return
        duration_us = max(0, int(duration_us))
        self._detection_duration_sum_us += duration_us
        if self._detection_samples == 0:
            self._detection_duration_min_us = duration_us
        else:
            self._detection_duration_min_us = min(
                self._detection_duration_min_us,
                duration_us,
            )
        self._detection_duration_max_us = max(
            self._detection_duration_max_us,
            duration_us,
        )
        self._detection_samples += 1

    def record_packet_duration(self, duration_us):
        """Record detector packet-processing time, excluding state evaluation."""
        if not self.enabled:
            return
        duration_us = max(0, int(duration_us))
        self._packet_duration_sum_us += duration_us
        if self._packet_samples == 0:
            self._packet_duration_min_us = duration_us
        else:
            self._packet_duration_min_us = min(
                self._packet_duration_min_us,
                duration_us,
            )
        self._packet_duration_max_us = max(
            self._packet_duration_max_us,
            duration_us,
        )
        self._packet_samples += 1

    def format_if_due(self, now_ms, heap_free, heap_free_post_gc=None, gc_pause_us=None):
        """Return a C++-compatible telemetry payload when the window is due."""
        if not self.enabled:
            return None
        now_ms = int(now_ms)
        heap_free = max(0, int(heap_free))
        if self._minimum_heap_free is None:
            self._minimum_heap_free = heap_free
        else:
            self._minimum_heap_free = min(self._minimum_heap_free, heap_free)
        if self._window_start_ms is None:
            self._window_start_ms = now_ms
            return None
        elapsed_ms = _ticks_diff(now_ms, self._window_start_ms)
        if elapsed_ms < self.LOG_INTERVAL_MS:
            return None

        elapsed_us = max(1, int(elapsed_ms) * 1000)
        runtime_load = min(100.0, self._loop_busy_us * 100.0 / elapsed_us)
        loop_average = (
            self._loop_duration_sum_us // self._loop_samples
            if self._loop_samples
            else 0
        )
        detection_average = (
            self._detection_duration_sum_us // self._detection_samples
            if self._detection_samples
            else 0
        )
        packet_average = (
            self._packet_duration_sum_us // self._packet_samples
            if self._packet_samples
            else 0
        )
        payload = (
            "[telemetry] heap_free={} heap_min={} runtime_load={:.2f}% "
            "loop_avg_us={} loop_max_us={} detection_samples={} "
            "detection_sum_us={} detection_avg_us={} detection_min_us={} "
            "detection_max_us={} packet_samples={} packet_sum_us={} "
            "packet_avg_us={} packet_min_us={} packet_max_us={}"
        ).format(
            heap_free,
            self._minimum_heap_free,
            runtime_load,
            loop_average,
            self._loop_duration_max_us,
            self._detection_samples,
            self._detection_duration_sum_us,
            detection_average,
            self._detection_duration_min_us,
            self._detection_duration_max_us,
            self._packet_samples,
            self._packet_duration_sum_us,
            packet_average,
            self._packet_duration_min_us,
            self._packet_duration_max_us,
        )
        if heap_free_post_gc is not None:
            payload += " heap_free_post_gc={}".format(max(0, int(heap_free_post_gc)))
        if gc_pause_us is not None:
            payload += " gc_pause_us={}".format(max(0, int(gc_pause_us)))
        self.reset()
        self._window_start_ms = now_ms
        return payload
