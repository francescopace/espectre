# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark parsing and runtime-window contracts."""

from __future__ import annotations

from tools import benchmark_firmware as bench


IDF_SIZE_LOG = """
Bootloader binary size 0x51e0 bytes. 0x2ae20 bytes (89%) free.
espectre-native.bin binary size 0x15a8c0 bytes. Smallest app partition is 0x1e0000 bytes. 0x85640 bytes (27%) free.
"""


def _status_line(timestamp_ms: int, state: str = "IDLE") -> str:
    return (
        f"I ({timestamp_ms}) espectre.runtime: {state} | csi:84/100 | occ:84% | "
        "mvmt:0.01 thr:0.50\n"
    )


def _telemetry_line(timestamp_ms: int, heap_free: int, *, detection_samples: int = 4) -> str:
    return (
        f"D ({timestamp_ms}) espectre: [telemetry] heap_free={heap_free} heap_min=90000 "
        "heap_largest=114688 cpu_mhz=160 runtime_load=2.50% loop_avg_us=200 loop_max_us=800 "
        f"detection_samples={detection_samples} detection_sum_us=4000 detection_avg_us=1000 "
        "detection_min_us=24 detection_max_us=1200\n"
    )


def _runtime_log(heap_by_offset_ms: dict[int, int], *, status_first_ms: int = 10_000) -> str:
    lines: list[str] = []
    for offset in range(0, 60_000, 1_000):
        timestamp_ms = status_first_ms + offset
        lines.append(_status_line(timestamp_ms))
        if offset in heap_by_offset_ms:
            lines.append(_telemetry_line(timestamp_ms, heap_by_offset_ms[offset]))
    return "".join(lines)


def test_parse_build_metrics_uses_app_image_not_bootloader():
    metrics = bench.parse_build_metrics(IDF_SIZE_LOG)

    assert metrics.firmware_size_bytes == 0x15A8C0
    assert metrics.partition_total_bytes == 0x1E0000
    assert metrics.partition_free_bytes == 0x85640
    assert metrics.partition_used_bytes == 0x1E0000 - 0x85640
    assert metrics.partition_free_percent == 27.0


def test_parse_build_metrics_prefers_application_binary_file(tmp_path):
    firmware = tmp_path / "espectre-native.bin"
    firmware.write_bytes(b"\x00" * 1_419_776)

    metrics = bench.parse_build_metrics(IDF_SIZE_LOG, firmware)

    assert metrics.firmware_size_bytes == 1_419_776


def test_heap_decline_ignores_telemetry_during_startup_grace():
    _metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {
                0: 150_000,
                10_000: 141_000,
                20_000: 140_500,
                50_000: 140_000,
            }
        )
    )

    assert "free heap declined by more than 5% after startup settled" not in reasons


def test_heap_decline_still_fails_after_startup_grace():
    _metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {
                10_000: 150_000,
                20_000: 141_000,
                50_000: 140_000,
            }
        )
    )

    assert "free heap declined by more than 5% after startup settled" in reasons


def test_runtime_expected_counts_use_status_span_not_boot_time():
    metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {offset: 140_000 for offset in range(0, 60_000, 10_000)},
            status_first_ms=70_000,
        )
    )

    assert metrics.status_samples == 60
    assert metrics.status_expected_samples == 60
    assert metrics.telemetry_samples == 6
    assert metrics.telemetry_expected_samples == 6
    assert "free heap declined by more than 5% after startup settled" not in reasons
    assert not any("expected detector status" in reason for reason in reasons)
    assert not any("expected shared debug telemetry" in reason for reason in reasons)
