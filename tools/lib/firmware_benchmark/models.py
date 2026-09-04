# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark models owner."""

from __future__ import annotations

from dataclasses import dataclass, field


CHIP_LABELS = {
    "esp32": "ESP32",
    "c3": "ESP32-C3",
    "c5": "ESP32-C5",
    "c6": "ESP32-C6",
    "s3": "ESP32-S3",
    "s2": "ESP32-S2",
}

FRONTEND_LABELS = {
    "esphome": "ESPHome",
    "matter": "Matter",
    "micro": "Micro-ESPectre",
    "native": "Native",
}

DETECTOR_LABELS = {
    "lightweight": "Lightweight",
    "collect": "Collect",
    "default": "Default",
    "high_accuracy": "High Accuracy",
}

@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str
    benchmark_mode: str = "runtime"

    @property
    def label(self) -> str:
        return f"{FRONTEND_LABELS[self.frontend]} {DETECTOR_LABELS[self.detector]}"

@dataclass(frozen=True)
class RepositoryState:
    revision: str
    worktree_dirty: bool
    source_fingerprint: str

@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    output: str
    reached_timeout: bool = False
    line_elapsed_seconds: list[float] = field(default_factory=list)
    analysis_start_line: int = 0

@dataclass
class BuildMetrics:
    firmware_size_bytes: int | None = None
    firmware_sha256: str | None = None
    deployed_source_bytes: int | None = None
    partition_used_bytes: int | None = None
    partition_total_bytes: int | None = None
    partition_free_bytes: int | None = None
    partition_free_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None

@dataclass
class RuntimeMetrics:
    status_samples: int = 0
    direct_request_attempts: int = 0
    direct_request_failures: int = 0
    direct_request_censored: int = 0
    packet_rate_samples: int = 0
    status_expected_samples: int = 0
    status_first_timestamp_ms: int | None = None
    status_last_timestamp_ms: int | None = None
    status_interval_mean_ms: float | None = None
    status_interval_max_ms: int | None = None
    status_gap_count: int = 0
    serial_framing_anomalies: int = 0
    device_reboots: int = 0
    motion_samples: int = 0
    motion_expected_samples: int = 0
    startup_state: str | None = None
    boot_marker_seen: bool = False
    device_ip: str | None = None
    pps_mean: float | None = None
    pps_min: int | None = None
    pps_max: int | None = None
    pps_stddev: float | None = None
    occupancy_samples: int = 0
    occupancy_mean: float | None = None
    occupancy_min: int | None = None
    occupancy_max: int | None = None
    dominant_motion_state: str | None = None
    motion_transitions: int = 0
    dominant_state_share_percent: float | None = None
    secondary_status_samples: int = 0
    secondary_dominant_motion_state: str | None = None
    secondary_dominant_state_share_percent: float | None = None
    heap_free_last: int | None = None
    heap_free_post_gc_last: int | None = None
    heap_free_settled_first: int | None = None
    heap_free_settled_last: int | None = None
    heap_free_settled_delta: int | None = None
    heap_free_settled_delta_percent: float | None = None
    heap_min: int | None = None
    heap_largest_last: int | None = None
    runtime_load_mean: float | None = None
    loop_avg_us_mean: float | None = None
    loop_max_us_max: int | None = None
    detection_samples: int = 0
    detection_avg_us_mean: float | None = None
    detection_min_us: int | None = None
    detection_max_us: int | None = None
    packet_processing_samples: int = 0
    packet_processing_avg_us_mean: float | None = None
    packet_processing_min_us: int | None = None
    packet_processing_max_us: int | None = None
    gc_pause_us_mean: float | None = None
    gc_pause_us_max: int | None = None
    stream_motion_samples: int = 0
    stream_csi_ap_mean: float | None = None
    stream_udp_rx_mean: float | None = None
    stream_udp_tx_mean: float | None = None
    stream_fresh_mean: float | None = None
    stream_tx_backpressure_total: int | None = None
    collect_devices_observed: int = 0
    collect_packets_seen: int = 0
    verified_detector: str | None = None

@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    status: str = "NOT RUN"
    reasons: list[str] = field(default_factory=list)
    build: CommandResult | None = None
    deploy: CommandResult | None = None
    flash: CommandResult | None = None
    monitor: CommandResult | None = None
    collect: CommandResult | None = None
    build_metrics: BuildMetrics = field(default_factory=BuildMetrics)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    direct_samples: list[dict[str, object]] = field(default_factory=list)
    direct_events: list[dict[str, object]] = field(default_factory=list)
    direct_attempts: list[dict[str, object]] = field(default_factory=list)
    transport_evidence: dict[str, object] = field(default_factory=dict)


CASES = tuple(
    [
        BenchmarkCase("native", "lightweight"),
        BenchmarkCase("native", "high_accuracy"),
        BenchmarkCase("esphome", "lightweight"),
        BenchmarkCase("esphome", "high_accuracy"),
        BenchmarkCase("matter", "lightweight"),
        BenchmarkCase("matter", "high_accuracy"),
        BenchmarkCase("micro", "lightweight"),
    ]
)

LEGACY_REPORT_CASE_LABELS = frozenset({"Micro-ESPectre High Accuracy"})

def clone_prebuilt_result(case: BenchmarkCase, source: BenchmarkResult) -> BenchmarkResult:
    return BenchmarkResult(
        case=case,
        build=source.build,
        build_metrics=BuildMetrics(**vars(source.build_metrics)),
    )
