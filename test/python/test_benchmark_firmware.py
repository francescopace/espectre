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


def test_cases_include_esphome_high_accuracy_after_lightweight():
    labels = [case.label for case in bench.CASES]

    assert labels.index("ESPHome Lightweight") < labels.index("ESPHome High Accuracy")


def test_detect_esphome_api_host_prefers_sta_over_ap():
    text = (
        "[C][wifi:984]: Setting up AP:\n"
        "  IP Address: 192.168.4.1\n"
        "[C][wifi:1259]:   IP Address: 192.168.1.50\n"
    )

    assert bench.detect_esphome_api_host_from_text(text) == "192.168.1.50"


def test_detect_esphome_api_host_uses_last_sta_address():
    text = "IP Address: 192.168.1.10\nIP Address: 192.168.1.20\n"

    assert bench.detect_esphome_api_host_from_text(text) == "192.168.1.20"


def test_esphome_api_hosts_fall_back_to_mdns():
    assert bench.esphome_api_hosts("") == [bench.ESPHOME_MDNS_HOST]
    assert bench.esphome_api_hosts("IP Address: 192.168.1.50\n") == [
        "192.168.1.50",
        bench.ESPHOME_MDNS_HOST,
    ]


def test_find_esphome_detector_select_matches_object_id():
    from aioesphomeapi.model import SelectInfo

    entities = [
        SelectInfo(object_id="csi_traffic_ownership", key=2, name="CSI Traffic Ownership", options=["internal"]),
        SelectInfo(
            object_id="detection_profile",
            key=7,
            name="Detection Profile",
            options=["lightweight", "high_accuracy"],
        ),
    ]

    select = bench.find_esphome_detector_select(entities)

    assert select is not None
    assert select.key == 7
    assert select.object_id == "detection_profile"


def test_esphome_benchmark_logger_keeps_default_uart():
    source = (
        "logger:\n"
        "  level: INFO\n"
        "  logs:\n"
        "    sensor: INFO\n"
        "api:\n"
    )

    updated = bench.apply_esphome_benchmark_logger(source)

    assert "level: DEBUG" in updated
    assert "hardware_uart:" not in updated
    assert "logs:\n    sensor: INFO" in updated


def test_esphome_benchmark_logger_does_not_override_explicit_uart0():
    source = (
        "logger:\n"
        "  level: INFO\n"
        "  hardware_uart: UART0\n"
        "api:\n"
    )

    updated = bench.apply_esphome_benchmark_logger(source)

    assert "level: DEBUG" in updated
    assert "hardware_uart: UART0" in updated


def test_status_stream_is_stable_requires_consecutive_one_hertz_samples():
    too_few = "".join(_status_line(20_000 + offset) for offset in range(0, 4_000, 1_000))
    gapped = "".join(_status_line(10_000 + offset) for offset in range(0, 5_000, 1_000))
    gapped += _status_line(28_570)
    stable = "".join(_status_line(20_000 + offset) for offset in range(0, 5_000, 1_000))

    assert not bench.status_stream_is_stable(too_few)
    assert not bench.status_stream_is_stable(gapped)
    assert bench.status_stream_is_stable(stable)
