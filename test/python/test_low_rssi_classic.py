"""
ESPectre - Classic low-RSSI regression test

Validates the production Classic detector on the real C3 low-RSSI pair.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import config

from tools.lib.dataset_metadata import load_dataset_info, resolve_entry_path
from tools.lib.performance_report import compute_classic_packet_result, load_real_data_cached


def test_production_classic_handles_real_low_rssi_pair():
    files = load_dataset_info()["files"]
    static_entry = next(
        entry
        for entry in files["static_presence"]
        if entry.get("low_rssi") and entry.get("chip") == "C3"
    )
    motion_entry = next(
        entry
        for entry in files["motion"]
        if entry.get("low_rssi") and entry.get("chip") == "C3"
    )
    static_packets, motion_packets = load_real_data_cached(
        resolve_entry_path("static_presence", static_entry),
        resolve_entry_path("motion", motion_entry),
    )

    result = compute_classic_packet_result(
        static_packets,
        motion_packets,
        tuple(config.DEFAULT_SUBCARRIERS),
        config.SEG_WINDOW_SIZE,
    )

    assert result is not None
    _threshold, metrics = result
    assert metrics["recall"] >= 85.0
    assert metrics["fp_rate"] <= 5.0
    assert metrics["effective_alarms"] == 0
