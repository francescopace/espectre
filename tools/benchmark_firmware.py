#!/usr/bin/env python3
"""Build, flash, and benchmark ESPectre firmware on connected hardware."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Iterator, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.espectre_cli.common import detect_chip_type, get_serial_port
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, IDF_FRONTENDS


MONITOR_DURATION_SECONDS = 180
STREAMER_COLLECT_DURATION_SECONDS = 120
STREAMER_IP_WAIT_SECONDS = 45
EXPECTED_PPS_MIN = 90
EXPECTED_PPS_MAX = 110
MIN_STATUS_SAMPLES = 120
MIN_TELEMETRY_SAMPLES = 12
MIN_STREAMER_COLLECT_SAMPLES = 60
MOTION_WARMUP_SAMPLES = 3

SUPPORTED_CHIPS = tuple(sorted(set(ESPHOME_CONFIGS) & set(IDF_FRONTENDS["native"]["targets"])))
CHIP_LABELS = {
    "esp32": "ESP32",
    "c3": "ESP32-C3",
    "c5": "ESP32-C5",
    "c6": "ESP32-C6",
    "s3": "ESP32-S3",
}

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STATUS_RE = re.compile(r"\b(?P<state>MOTION|IDLE)\s*\|\s*(?P<pps>\d+)\s+pkt/s\b")
TELEMETRY_RE = re.compile(r"\[telemetry\]\s+(?P<fields>[^\r\n]+)")
KEY_VALUE_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>-?[0-9]+(?:\.[0-9]+)?)(?:%|\b)")
FATAL_PATTERNS = (
    "Guru Meditation Error",
    "abort() was called",
    "panic'ed",
    "Stack smashing protect failure",
)
MATTER_BOOT_MARKER = "ESPectre Matter firmware started on endpoint"
MATTER_STARTUP_STATE_RE = re.compile(r"ESPectre Matter CSI services:\s*(?P<state>[^\r\n]+)")
MATTER_VALID_STARTUP_STATES = {"armed", "waiting for commissioning"}
STREAMER_IP_RE = re.compile(r"Wi-Fi connected: ip=(?P<ip>\d+\.\d+\.\d+\.\d+)")
STREAMER_STATE_RE = re.compile(r"\[STATE\]\s+\S+\s+->\s+(?P<state>\S+)\s+\(")
STREAMER_TELEMETRY_RE = re.compile(
    r"csi_ap=(?P<csi_ap>\d+(?:\.\d+)?)"
    r"(?:\s+csi_filt=(?P<csi_filt>\d+(?:\.\d+)?))?"
    r"(?:\s+valid=(?P<valid>\d+))?"
    r"(?:\s+bad_sc=(?P<bad_sc>\d+))?"
    r"\s+udp_rx=(?P<udp_rx>\d+(?:\.\d+)?)"
    r"\s+udp_tx=(?P<udp_tx>\d+(?:\.\d+)?)"
    r"\s+fresh=(?P<fresh>\d+(?:\.\d+)?)"
    r"(?:\s+tx_err=(?P<tx_err_rate>\d+(?:\.\d+)?)/(?P<tx_err_total>\d+))?"
    r"(?:\s+tx_bp=(?P<tx_bp_rate>\d+(?:\.\d+)?)/(?P<tx_bp_total>\d+))?"
    r"\s+age_ms=(?P<age_ms>\d+)"
)
COLLECT_DETAIL_RE = re.compile(
    r"ip=(?P<ip>\S+)\s+chip=(?P<chip>\S+)\s+ch=(?P<channel>\S+)\s+rssi=(?P<rssi>\S+)"
    r"(?:\s+\[(?P<detector>[^\]]+)\])?\s+\|.*?\|\s+mvmt:(?P<motion_metric>-?[0-9.]+)"
    r"\s+thr:(?P<threshold>-?[0-9.]+)\s+\|\s+(?P<state>MOTION|IDLE)\s+\|\s+(?P<pps>\d+)\s+pkt/s"
)


@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str
    benchmark_mode: str = "runtime"

    @property
    def label(self) -> str:
        return f"{self.frontend.capitalize()} {self.detector.capitalize()}"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    output: str
    reached_timeout: bool = False


@dataclass
class BuildMetrics:
    firmware_size_bytes: int | None = None
    partition_used_bytes: int | None = None
    partition_total_bytes: int | None = None
    partition_free_bytes: int | None = None
    partition_free_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None


@dataclass
class RuntimeMetrics:
    status_samples: int = 0
    telemetry_samples: int = 0
    startup_state: str | None = None
    boot_marker_seen: bool = False
    device_ip: str | None = None
    pps_mean: float | None = None
    pps_min: int | None = None
    pps_max: int | None = None
    pps_stddev: float | None = None
    dominant_motion_state: str | None = None
    motion_transitions: int = 0
    dominant_state_share_percent: float | None = None
    secondary_status_samples: int = 0
    secondary_dominant_motion_state: str | None = None
    secondary_dominant_state_share_percent: float | None = None
    heap_free_last: int | None = None
    heap_min: int | None = None
    heap_largest_last: int | None = None
    runtime_load_mean: float | None = None
    loop_avg_us_mean: float | None = None
    loop_max_us_max: int | None = None
    detection_samples: int = 0
    detection_avg_us_mean: float | None = None
    detection_min_us: int | None = None
    detection_max_us: int | None = None
    stream_telemetry_samples: int = 0
    stream_csi_ap_mean: float | None = None
    stream_udp_rx_mean: float | None = None
    stream_udp_tx_mean: float | None = None
    stream_fresh_mean: float | None = None
    stream_tx_backpressure_total: int | None = None
    collect_devices_observed: int = 0
    collect_packets_seen: int = 0


@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    status: str = "NOT RUN"
    reasons: list[str] = field(default_factory=list)
    build: CommandResult | None = None
    flash: CommandResult | None = None
    monitor: CommandResult | None = None
    collect: CommandResult | None = None
    build_metrics: BuildMetrics = field(default_factory=BuildMetrics)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)


CASES = tuple(
    [
        *(BenchmarkCase(frontend, detector) for frontend in ("esphome", "native") for detector in ("classic", "ml")),
        BenchmarkCase("matter", "classic", benchmark_mode="smoke"),
        BenchmarkCase("streamer", "collect", benchmark_mode="stream"),
    ]
)


def select_cases(frontend: str | None = None, detector: str | None = None) -> tuple[BenchmarkCase, ...]:
    """Return the benchmark cases matching the optional CLI filters."""
    return tuple(
        case
        for case in CASES
        if (frontend is None or case.frontend == frontend)
        and (detector is None or case.detector == detector)
    )


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def format_duration(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60.0)
    if minutes < 1:
        return f"{seconds:.1f}s"
    return f"{int(minutes)}m {remaining:.1f}s"


def format_bytes(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,} bytes ({value / 1024.0:.1f} KiB)"


def format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            process.wait()


def child_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    resolved = (env or os.environ).copy()
    # Preserve the virtualenv symlink location; resolving it would point back to
    # the host interpreter and hide virtualenv-installed commands such as ESPHome.
    interpreter_bin = str(Path(sys.executable).parent)
    path_entries = resolved.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in path_entries:
        resolved["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])
    if sys.prefix != sys.base_prefix:
        resolved["VIRTUAL_ENV"] = sys.prefix
    return resolved


def run_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    timeout_is_success: bool = False,
    output_prefix: str = "",
) -> CommandResult:
    display_command = " ".join(str(part) for part in command)
    print(f"\n{output_prefix}$ {display_command}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        env=child_environment(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    output_lines: list[str] = []

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(f"{output_prefix}{line}", end="", flush=True)

    relay_thread = threading.Thread(target=_relay_output, daemon=True)
    relay_thread.start()
    reached_timeout = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reached_timeout = True
        _terminate_process(process)
        returncode = 0 if timeout_is_success else process.returncode or 1
    except KeyboardInterrupt:
        _terminate_process(process)
        raise
    finally:
        relay_thread.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()

    return CommandResult(
        command=[str(part) for part in command],
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        output="".join(output_lines),
        reached_timeout=reached_timeout,
    )


def parse_build_metrics(output: str, firmware_path: Path | None = None) -> BuildMetrics:
    text = strip_ansi(output)
    metrics = BuildMetrics()
    if firmware_path is not None and firmware_path.is_file():
        metrics.firmware_size_bytes = firmware_path.stat().st_size

    ram_match = re.search(
        r"RAM:.*?\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)",
        text,
        flags=re.IGNORECASE,
    )
    if ram_match:
        metrics.ram_used_bytes = int(ram_match.group(1))
        metrics.ram_total_bytes = int(ram_match.group(2))

    flash_match = re.search(
        r"Flash:.*?\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)",
        text,
        flags=re.IGNORECASE,
    )
    if flash_match:
        metrics.partition_used_bytes = int(flash_match.group(1))
        metrics.partition_total_bytes = int(flash_match.group(2))
        metrics.partition_free_bytes = metrics.partition_total_bytes - metrics.partition_used_bytes
        metrics.partition_free_percent = metrics.partition_free_bytes / metrics.partition_total_bytes * 100.0

    native_size_match = re.search(r"binary size\s+0x([0-9a-f]+)\s+bytes", text, flags=re.IGNORECASE)
    if metrics.firmware_size_bytes is None and native_size_match:
        metrics.firmware_size_bytes = int(native_size_match.group(1), 16)

    native_partition_match = re.search(
        r"Smallest app partition is\s+0x([0-9a-f]+)\s+bytes.*?"
        r"0x([0-9a-f]+)\s+bytes\s+\((\d+)%\)\s+free",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if native_partition_match:
        metrics.partition_total_bytes = int(native_partition_match.group(1), 16)
        metrics.partition_free_bytes = int(native_partition_match.group(2), 16)
        metrics.partition_free_percent = float(native_partition_match.group(3))
        metrics.partition_used_bytes = metrics.partition_total_bytes - metrics.partition_free_bytes

    return metrics


def _parse_telemetry_samples(text: str) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for match in TELEMETRY_RE.finditer(strip_ansi(text)):
        fields = {
            item.group("key"): float(item.group("value"))
            for item in KEY_VALUE_RE.finditer(match.group("fields"))
        }
        if fields:
            samples.append(fields)
    return samples


def _append_common_monitor_reasons(
    metrics: RuntimeMetrics,
    telemetry: Sequence[dict[str, float]],
    reasons: list[str],
    *,
    require_detection_timing: bool,
) -> None:
    if len(telemetry) < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {len(telemetry)} shared debug telemetry samples were logged")
    if metrics.heap_free_last is not None and telemetry:
        heap_free_first = telemetry[0].get("heap_free")
        if heap_free_first is not None and metrics.heap_free_last < heap_free_first * 0.95:
            reasons.append("free heap declined by more than 5% during monitoring")
    if require_detection_timing and metrics.detection_samples == 0:
        reasons.append("detector timing was not logged")


def _collect_values(samples: Sequence[dict[str, float]], key: str) -> list[float]:
    return [sample[key] for sample in samples if key in sample]


def _apply_state_series(
    metrics: RuntimeMetrics,
    states: Sequence[str],
    *,
    secondary: bool = False,
) -> None:
    if not states:
        return
    dominant_state = max(set(states), key=states.count)
    dominant_share_percent = states.count(dominant_state) / len(states) * 100.0
    if secondary:
        metrics.secondary_status_samples = len(states)
        metrics.secondary_dominant_motion_state = dominant_state
        metrics.secondary_dominant_state_share_percent = dominant_share_percent
        return
    metrics.status_samples = len(states)
    metrics.dominant_motion_state = dominant_state
    metrics.dominant_state_share_percent = dominant_share_percent


def _parse_streamer_telemetry_samples(text: str) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for match in STREAMER_TELEMETRY_RE.finditer(strip_ansi(text)):
        sample: dict[str, float] = {}
        for key, value in match.groupdict().items():
            if value is None:
                continue
            sample[key] = float(value)
        if sample:
            samples.append(sample)
    return samples


def _parse_collect_output(text: str) -> RuntimeMetrics:
    metrics = RuntimeMetrics()
    detector_states: dict[str, list[str]] = {}
    pps_values: list[int] = []
    observed_ips: set[str] = set()
    packet_counts: list[int] = []
    for line in strip_ansi(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("STATUS: COLLECTING"):
            packet_match = re.search(r"packets\s+(?P<packets>\d+)", line)
            observed_match = re.search(r"COLLECTING\s+(?P<observed>\d+)/(?P<required>\d+)", line)
            if packet_match:
                packet_counts.append(int(packet_match.group("packets")))
            if observed_match:
                metrics.collect_devices_observed = max(
                    metrics.collect_devices_observed,
                    int(observed_match.group("observed")),
                )
            continue

        match = COLLECT_DETAIL_RE.search(line)
        if match is None:
            continue
        detector = (match.group("detector") or "classic").strip().lower()
        state = match.group("state")
        pps = int(match.group("pps"))
        detector_states.setdefault(detector, []).append(state)
        pps_values.append(pps)
        observed_ips.add(match.group("ip"))

    metrics.collect_devices_observed = max(metrics.collect_devices_observed, len(observed_ips))
    metrics.collect_packets_seen = max(packet_counts) if packet_counts else 0
    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    primary_states = detector_states.get("classic") or detector_states.get("ml") or []
    secondary_states = detector_states.get("ml") if primary_states is detector_states.get("classic") else []
    _apply_state_series(metrics, primary_states)
    _apply_state_series(metrics, secondary_states, secondary=True)
    return metrics


def analyze_monitor_output(output: str, benchmark_mode: str = "runtime") -> tuple[RuntimeMetrics, list[str]]:
    text = strip_ansi(output)
    status_matches = list(STATUS_RE.finditer(text))
    pps_values = [int(match.group("pps")) for match in status_matches if int(match.group("pps")) > 0]
    states = [match.group("state") for match in status_matches if int(match.group("pps")) > 0]
    observed_states = states[MOTION_WARMUP_SAMPLES:]
    telemetry = _parse_telemetry_samples(text)

    metrics = RuntimeMetrics(
        status_samples=len(pps_values),
        telemetry_samples=len(telemetry),
    )
    reasons: list[str] = []

    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    if observed_states:
        metrics.motion_transitions = sum(
            current != previous for previous, current in zip(observed_states, observed_states[1:])
        )
        metrics.dominant_motion_state = max(set(observed_states), key=observed_states.count)
        metrics.dominant_state_share_percent = (
            observed_states.count(metrics.dominant_motion_state) / len(observed_states) * 100.0
        )

    heap_free = _collect_values(telemetry, "heap_free")
    heap_min = _collect_values(telemetry, "heap_min")
    heap_largest = _collect_values(telemetry, "heap_largest")
    runtime_load = _collect_values(telemetry, "runtime_load")
    loop_avg = _collect_values(telemetry, "loop_avg_us")
    loop_max = _collect_values(telemetry, "loop_max_us")
    detection_windows = [sample for sample in telemetry if sample.get("detection_samples", 0) > 0]
    detection_samples = int(sum(sample["detection_samples"] for sample in detection_windows))
    detection_sum_us = sum(
        sample.get("detection_sum_us", sample.get("detection_avg_us", 0) * sample["detection_samples"])
        for sample in detection_windows
    )

    metrics.heap_free_last = int(heap_free[-1]) if heap_free else None
    metrics.heap_min = int(min(heap_min)) if heap_min else None
    metrics.heap_largest_last = int(heap_largest[-1]) if heap_largest else None
    metrics.runtime_load_mean = statistics.fmean(runtime_load) if runtime_load else None
    metrics.loop_avg_us_mean = statistics.fmean(loop_avg) if loop_avg else None
    metrics.loop_max_us_max = int(max(loop_max)) if loop_max else None
    metrics.detection_samples = detection_samples
    metrics.detection_avg_us_mean = detection_sum_us / detection_samples if detection_samples else None
    metrics.detection_min_us = (
        int(min(sample["detection_min_us"] for sample in detection_windows)) if detection_windows else None
    )
    metrics.detection_max_us = (
        int(max(sample["detection_max_us"] for sample in detection_windows)) if detection_windows else None
    )

    if benchmark_mode == "runtime":
        if len(pps_values) < MIN_STATUS_SAMPLES:
            reasons.append(f"only {len(pps_values)} motion/packet-rate samples were logged")
        elif metrics.pps_mean is None or not EXPECTED_PPS_MIN <= metrics.pps_mean <= EXPECTED_PPS_MAX:
            reasons.append(
                f"mean packet rate {metrics.pps_mean:.2f} pps is outside "
                f"{EXPECTED_PPS_MIN}-{EXPECTED_PPS_MAX} pps"
            )
        _append_common_monitor_reasons(metrics, telemetry, reasons, require_detection_timing=True)
    elif benchmark_mode == "smoke":
        startup_state_match = MATTER_STARTUP_STATE_RE.search(text)
        metrics.startup_state = startup_state_match.group("state").strip() if startup_state_match else None
        metrics.boot_marker_seen = MATTER_BOOT_MARKER in text
        if not metrics.boot_marker_seen:
            reasons.append("Matter firmware boot marker was not logged")
        if metrics.startup_state is None:
            reasons.append("Matter startup state was not logged")
        elif metrics.startup_state.lower() not in MATTER_VALID_STARTUP_STATES:
            reasons.append(f"unexpected Matter startup state: {metrics.startup_state}")
        _append_common_monitor_reasons(metrics, telemetry, reasons, require_detection_timing=False)
    elif benchmark_mode == "stream":
        state_matches = list(STREAMER_STATE_RE.finditer(text))
        ip_match = STREAMER_IP_RE.search(text)
        stream_samples = _parse_streamer_telemetry_samples(text)
        metrics.startup_state = state_matches[-1].group("state") if state_matches else None
        metrics.device_ip = ip_match.group("ip") if ip_match else None
        metrics.stream_telemetry_samples = len(stream_samples)
        stream_csi_ap = _collect_values(stream_samples, "csi_ap")
        stream_udp_rx = _collect_values(stream_samples, "udp_rx")
        stream_udp_tx = _collect_values(stream_samples, "udp_tx")
        stream_fresh = _collect_values(stream_samples, "fresh")
        stream_tx_bp_totals = _collect_values(stream_samples, "tx_bp_total")
        metrics.stream_csi_ap_mean = statistics.fmean(stream_csi_ap) if stream_csi_ap else None
        metrics.stream_udp_rx_mean = statistics.fmean(stream_udp_rx) if stream_udp_rx else None
        metrics.stream_udp_tx_mean = statistics.fmean(stream_udp_tx) if stream_udp_tx else None
        metrics.stream_fresh_mean = statistics.fmean(stream_fresh) if stream_fresh else None
        metrics.stream_tx_backpressure_total = int(max(stream_tx_bp_totals)) if stream_tx_bp_totals else 0
        if metrics.device_ip is None:
            reasons.append("streamer Wi-Fi IP was not logged")
        if metrics.startup_state is None:
            reasons.append("streamer workflow state was not logged")
        elif metrics.startup_state != "STREAMING":
            reasons.append(f"streamer did not reach STREAMING state (last state: {metrics.startup_state})")
        _append_common_monitor_reasons(metrics, telemetry, reasons, require_detection_timing=False)
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")

    for pattern in FATAL_PATTERNS:
        if pattern in text:
            reasons.append(f"fatal firmware log detected: {pattern}")

    return metrics, reasons


def _latest_firmware_artifact(frontend: str) -> Path | None:
    if frontend == "esphome":
        candidates = list((REPO_ROOT / "examples" / ".esphome").glob("build/*/.pioenvs/*/firmware.bin"))
    else:
        app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
        build_dir = os.environ.get("ESPECTRE_IDF_BUILD_DIR", "build")
        preferred = app_dir / build_dir / f"espectre-{frontend}.bin"
        if preferred.is_file():
            candidates = [preferred]
        else:
            candidates = [
                path
                for path in (app_dir / build_dir).glob("*.bin")
                if path.name not in {"bootloader.bin", "partition-table.bin", "ota_data_initial.bin"}
            ]
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: (path.stat().st_size, path.stat().st_mtime)) if existing else None


@contextmanager
def esphome_case_config(chip: str, detector: str) -> Iterator[Path]:
    source_path = Path(ESPHOME_CONFIGS[chip]["dev"])
    content = source_path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"^(\s*detection_algorithm:\s*)(?:classic|ml)(\s*(?:#.*)?)$",
        rf"\g<1>{detector}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError(f"could not set detector in {source_path}")

    temporary_path = source_path.parent / f".espectre-benchmark-{chip}-{detector}.yaml"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark config already exists: {temporary_path}")
    try:
        temporary_path.write_text(updated, encoding="utf-8")
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def idf_case_environment(frontend: str, chip: str, detector: str) -> Iterator[dict[str, str]]:
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    defaults = [app_dir / "sdkconfig.defaults"]
    target_defaults = app_dir / f"sdkconfig.defaults.{idf_target}"
    if target_defaults.is_file():
        defaults.append(target_defaults)

    if frontend == "native":
        native_wifi = app_dir / "sdkconfig.wifi"
        streamer_wifi = REPO_ROOT / "src" / "cpp" / "frontend" / "streamer" / "app" / "sdkconfig.wifi"
        wifi_defaults = native_wifi if native_wifi.is_file() else streamer_wifi
        if not wifi_defaults.is_file():
            raise RuntimeError("native Wi-Fi defaults are missing (expected native or streamer sdkconfig.wifi)")
        defaults.append(wifi_defaults)
    elif frontend == "streamer":
        streamer_wifi = app_dir / "sdkconfig.wifi"
        if streamer_wifi.is_file():
            defaults.append(streamer_wifi)

    classic_enabled = detector == "classic"
    override_lines = [
        "# Generated temporary firmware benchmark overrides.",
        "CONFIG_LOG_DEFAULT_LEVEL_DEBUG=y",
        "CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y",
    ]
    if frontend in {"native", "matter"}:
        override_lines.extend(
            [
                (
                    "CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC=y"
                    if classic_enabled
                    else "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC is not set"
                ),
                (
                    "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML is not set"
                    if classic_enabled
                    else "CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML=y"
                ),
            ]
        )
    override_lines.append("")
    override = "\n".join(override_lines)
    temporary_path = app_dir / f".espectre-benchmark-{chip}-{detector}.defaults"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark defaults already exist: {temporary_path}")
    try:
        temporary_path.write_text(override, encoding="utf-8")
        defaults.append(temporary_path)
        env = os.environ.copy()
        env["SDKCONFIG_DEFAULTS"] = ";".join(str(path.resolve()) for path in defaults)
        yield env
    finally:
        temporary_path.unlink(missing_ok=True)


def update_native_sdkconfig_detector(detector: str) -> None:
    """Select a detector in the generated sdkconfig for an incremental build."""
    sdkconfig = Path(IDF_FRONTENDS["native"]["app_dir"]) / "sdkconfig"
    if not sdkconfig.is_file():
        raise RuntimeError(f"incremental Native build requires {sdkconfig}")

    selections = {
        "classic": (
            "CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC=y",
            "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML is not set",
        ),
        "ml": (
            "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC is not set",
            "CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML=y",
        ),
    }
    classic_line, ml_line = selections[detector]
    content = sdkconfig.read_text(encoding="utf-8")
    content, classic_replacements = re.subn(
        r"^(?:# )?CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC(?:=y| is not set)$",
        classic_line,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    content, ml_replacements = re.subn(
        r"^(?:# )?CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML(?:=y| is not set)$",
        ml_line,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if classic_replacements != 1 or ml_replacements != 1:
        raise RuntimeError(f"could not select {detector} detector in {sdkconfig}")
    sdkconfig.write_text(content, encoding="utf-8")


def _commands_for_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    config: Path | None = None,
    *,
    clean: bool,
) -> tuple[list[str], list[str], list[str]]:
    launcher = str(REPO_ROOT / "espectre")
    if case.frontend == "esphome":
        assert config is not None
        config_value = str(config)
        build_command = [launcher, "esphome", "build", "--config", config_value]
        if clean:
            build_command.append("--clean")
        return (
            build_command,
            [launcher, "esphome", "flash", "--config", config_value, "--device", port],
            [launcher, "esphome", "monitor", "--config", config_value, "--device", port],
        )
    build_command = [launcher, case.frontend, "build", "--chip", chip]
    if clean:
        build_command.append("--clean")
    return (
        build_command,
        [launcher, case.frontend, "flash", "--port", port],
        [launcher, "monitor", "--port", port],
    )


@contextmanager
def case_context(
    case: BenchmarkCase,
    chip: str,
    *,
    clean: bool,
) -> Iterator[tuple[dict[str, str] | None, Path | None]]:
    if case.frontend == "esphome":
        with esphome_case_config(chip, case.detector) as config:
            yield None, config
    else:
        with idf_case_environment(case.frontend, chip, case.detector) as env:
            if case.frontend == "native" and not clean:
                update_native_sdkconfig_detector(case.detector)
            yield env, None


def build_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    output_prefix: str = "",
) -> BenchmarkResult:
    result = BenchmarkResult(case=case)
    try:
        with case_context(case, chip, clean=clean) as (env, config):
            build_command, _flash_command, _monitor_command = _commands_for_case(
                case,
                chip,
                port,
                config,
                clean=clean,
            )
            result.build = run_command(build_command, env=env, output_prefix=output_prefix)
            result.build_metrics = parse_build_metrics(
                result.build.output,
                _latest_firmware_artifact(case.frontend),
            )
            if result.build.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"build exited with status {result.build.returncode}")
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def _run_background_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    output_prefix: str = "",
    line_callback: Callable[[str], None] | None = None,
) -> tuple[subprocess.Popen[str], list[str], threading.Thread, float]:
    display_command = " ".join(str(part) for part in command)
    print(f"\n{output_prefix}$ {display_command}", flush=True)
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        env=child_environment(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    output_lines: list[str] = []
    started = time.monotonic()

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(f"{output_prefix}{line}", end="", flush=True)
            if line_callback is not None:
                line_callback(line)

    relay_thread = threading.Thread(target=_relay_output, daemon=True)
    relay_thread.start()
    return process, output_lines, relay_thread, started


def _finalize_background_command(
    process: subprocess.Popen[str],
    output_lines: list[str],
    relay_thread: threading.Thread,
    started: float,
    command: Sequence[str],
) -> CommandResult:
    relay_thread.join(timeout=5)
    if process.stdout is not None:
        process.stdout.close()
    returncode = process.returncode if process.returncode is not None else 0
    if returncode in {-signal.SIGINT, 130, 143}:
        returncode = 0
    return CommandResult(
        command=[str(part) for part in command],
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        output="".join(output_lines),
        reached_timeout=False,
    )


def run_streamer_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
) -> BenchmarkResult:
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = build_case(case, chip, port, clean=clean)
    if result.build is None or result.build.returncode != 0:
        return result

    launcher = str(REPO_ROOT / "espectre")
    try:
        with case_context(case, chip, clean=clean) as (env, config):
            _build_command, flash_command, monitor_command = _commands_for_case(
                case,
                chip,
                port,
                config,
                clean=clean,
            )
            result.flash = run_command(flash_command, env=env)
            if result.flash.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"flash exited with status {result.flash.returncode}")
                return result

            device_ip_event = threading.Event()
            device_ip_holder = {"value": None}

            def _capture_device_ip(line: str) -> None:
                match = STREAMER_IP_RE.search(strip_ansi(line))
                if match is not None:
                    device_ip_holder["value"] = match.group("ip")
                    device_ip_event.set()

            monitor_process, monitor_output, relay_thread, monitor_started = _run_background_command(
                monitor_command,
                env=env,
                output_prefix="[stream] ",
                line_callback=_capture_device_ip,
            )
            try:
                if not device_ip_event.wait(timeout=STREAMER_IP_WAIT_SECONDS):
                    _terminate_process(monitor_process)
                    monitor_process.wait(timeout=5)
                    result.monitor = _finalize_background_command(
                        monitor_process,
                        monitor_output,
                        relay_thread,
                        monitor_started,
                        monitor_command,
                    )
                    result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                        result.monitor.output,
                        benchmark_mode=case.benchmark_mode,
                    )
                    result.reasons.extend(analysis_reasons)
                    result.reasons.append("timed out waiting for streamer Wi-Fi IP")
                    result.status = "FAIL"
                    return result

                device_ip = device_ip_holder["value"]
                collect_command = [
                    launcher,
                    "collect",
                    "--no-save",
                    "--duration",
                    str(STREAMER_COLLECT_DURATION_SECONDS),
                    "--target",
                    str(device_ip),
                    "--detector",
                    "classic,ml",
                ]
                result.collect = run_command(collect_command)
            finally:
                if monitor_process.poll() is None:
                    _terminate_process(monitor_process)
                    monitor_process.wait(timeout=5)
                result.monitor = _finalize_background_command(
                    monitor_process,
                    monitor_output,
                    relay_thread,
                    monitor_started,
                    monitor_command,
                )

            if result.collect is None or result.collect.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(
                    f"collect exited with status {result.collect.returncode if result.collect else 'N/A'}"
                )
                return result

            result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                result.monitor.output,
                benchmark_mode=case.benchmark_mode,
            )
            collect_metrics = _parse_collect_output(result.collect.output)
            if result.runtime_metrics.device_ip is None:
                result.runtime_metrics.device_ip = device_ip
            result.runtime_metrics.collect_devices_observed = collect_metrics.collect_devices_observed
            result.runtime_metrics.collect_packets_seen = collect_metrics.collect_packets_seen
            result.runtime_metrics.pps_mean = collect_metrics.pps_mean
            result.runtime_metrics.pps_min = collect_metrics.pps_min
            result.runtime_metrics.pps_max = collect_metrics.pps_max
            result.runtime_metrics.pps_stddev = collect_metrics.pps_stddev
            result.runtime_metrics.status_samples = collect_metrics.status_samples
            result.runtime_metrics.dominant_motion_state = collect_metrics.dominant_motion_state
            result.runtime_metrics.dominant_state_share_percent = collect_metrics.dominant_state_share_percent
            result.runtime_metrics.secondary_status_samples = collect_metrics.secondary_status_samples
            result.runtime_metrics.secondary_dominant_motion_state = collect_metrics.secondary_dominant_motion_state
            result.runtime_metrics.secondary_dominant_state_share_percent = (
                collect_metrics.secondary_dominant_state_share_percent
            )
            result.reasons.extend(analysis_reasons)
            if result.runtime_metrics.collect_devices_observed < 1:
                result.reasons.append("host collect did not observe any streamer device")
            if result.runtime_metrics.status_samples < MIN_STREAMER_COLLECT_SAMPLES:
                result.reasons.append(
                    f"only {result.runtime_metrics.status_samples} classic host collect samples were logged"
                )
            if result.runtime_metrics.secondary_status_samples < MIN_STREAMER_COLLECT_SAMPLES:
                result.reasons.append(
                    f"only {result.runtime_metrics.secondary_status_samples} ml host collect samples were logged"
                )
            if result.runtime_metrics.pps_mean is None or not EXPECTED_PPS_MIN <= result.runtime_metrics.pps_mean <= EXPECTED_PPS_MAX:
                result.reasons.append(
                    f"host collect mean packet rate {result.runtime_metrics.pps_mean:.2f} pps is outside "
                    f"{EXPECTED_PPS_MIN}-{EXPECTED_PPS_MAX} pps"
                    if result.runtime_metrics.pps_mean is not None
                    else "host collect packet rate was not logged"
                )
            result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def run_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    prebuilt: BenchmarkResult | None = None,
    overlap_build: BenchmarkCase | None = None,
) -> tuple[BenchmarkResult, BenchmarkResult | None]:
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = prebuilt or build_case(case, chip, port, clean=clean)
    overlapped_result: BenchmarkResult | None = None

    if result.build is None or result.build.returncode != 0:
        return result, None

    try:
        with case_context(case, chip, clean=clean) as (env, config):
            _build_command, flash_command, monitor_command = _commands_for_case(
                case,
                chip,
                port,
                config,
                clean=clean,
            )
            result.flash = run_command(flash_command, env=env)
            if result.flash.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"flash exited with status {result.flash.returncode}")
                return result, None

            executor: ThreadPoolExecutor | None = None
            build_future = None
            if overlap_build is not None:
                print(f"\nStarting {overlap_build.label} build during {case.label} monitoring.", flush=True)
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firmware-build")
                build_future = executor.submit(
                    build_case,
                    overlap_build,
                    chip,
                    port,
                    clean=False,
                    output_prefix="[ML build] ",
                )
            try:
                result.monitor = run_command(
                    monitor_command,
                    env=env,
                    timeout=MONITOR_DURATION_SECONDS,
                    timeout_is_success=True,
                )
                if build_future is not None:
                    overlapped_result = build_future.result()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            if result.monitor.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"monitor exited with status {result.monitor.returncode}")
                return result, overlapped_result
            result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                result.monitor.output,
                benchmark_mode=case.benchmark_mode,
            )
            result.reasons.extend(analysis_reasons)
            result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result, overlapped_result


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def render_report(
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    expected_cases: Sequence[BenchmarkCase] = CASES,
) -> str:
    chip_label = CHIP_LABELS[chip]
    overall = (
        "PASS"
        if len(results) == len(expected_cases) and all(result.status == "PASS" for result in results)
        else "FAIL"
    )

    def format_summary_bytes(value: int | None) -> str:
        if value is None:
            return "N/A"
        if value < 1024:
            return f"{value:,} bytes"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} KiB"
        return f"{value / (1024 * 1024):.2f} MiB"

    def format_summary_partition_free(bytes_value: int | None, percent_value: float | None) -> str:
        if bytes_value is None:
            return "N/A"
        formatted = format_summary_bytes(bytes_value)
        if percent_value is not None:
            formatted += f" ({percent_value:.1f}%)"
        return formatted

    lines = [
        "<!-- Generated file. Do not edit manually. -->",
        "",
        f"# {chip_label} Firmware Performance",
        "",
        f"Generated by: `tools/benchmark_firmware.py --chip {chip}`",
        f"Git revision: `{_git_revision()}`",
        f"Run started: `{started_at.astimezone().isoformat(timespec='seconds')}`",
        f"Monitor duration per firmware: `{MONITOR_DURATION_SECONDS} seconds`",
        f"Overall result: **{overall}**",
        "",
        "## Summary",
        "",
        "| Frontend | Detector | Result | Binary size | Partition free | CPU load | Min free heap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        frontend_label, detector_label = result.case.label.rsplit(" ", 1)
        lines.append(
            "| "
            + " | ".join(
                [
                    frontend_label,
                    detector_label,
                    f"**{result.status}**",
                    format_summary_bytes(build.firmware_size_bytes),
                    format_summary_partition_free(build.partition_free_bytes, build.partition_free_percent),
                    format_number(runtime.runtime_load_mean, "%"),
                    format_summary_bytes(runtime.heap_min),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Results", ""])

    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        detail_rows = [f"| Benchmark mode | {result.case.benchmark_mode} |"]

        if result.build:
            detail_rows.append(f"| Build duration | {format_duration(result.build.duration_seconds)} |")
        if result.flash:
            detail_rows.append(f"| Flash duration | {format_duration(result.flash.duration_seconds)} |")
        if result.monitor:
            detail_rows.append(f"| Monitor duration | {format_duration(result.monitor.duration_seconds)} |")
        if result.collect:
            detail_rows.append(f"| Collect duration | {format_duration(result.collect.duration_seconds)} |")

        if build.firmware_size_bytes is not None:
            detail_rows.append(f"| Firmware binary | {format_bytes(build.firmware_size_bytes)} |")
        if build.partition_used_bytes is not None:
            detail_rows.append(f"| Application partition used | {format_bytes(build.partition_used_bytes)} |")
        if build.partition_free_bytes is not None:
            detail_rows.append(f"| Application partition free | {format_bytes(build.partition_free_bytes)} |")
        if build.ram_used_bytes is not None:
            detail_rows.append(f"| Build RAM used | {format_bytes(build.ram_used_bytes)} |")

        if runtime.startup_state is not None:
            detail_rows.append(f"| Startup state | {runtime.startup_state} |")

        if result.case.benchmark_mode in {"runtime", "stream"} and runtime.status_samples > 0:
            detail_rows.append(f"| Packet-rate samples | {runtime.status_samples} |")
            detail_rows.append(
                f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                f"{format_number(runtime.pps_stddev)} standard deviation |"
            )

        if runtime.telemetry_samples > 0:
            detail_rows.append(f"| Telemetry samples | {runtime.telemetry_samples} |")
        if runtime.heap_free_last is not None:
            detail_rows.append(f"| Last free heap | {format_bytes(runtime.heap_free_last)} |")
        if runtime.heap_min is not None:
            detail_rows.append(f"| Minimum free heap | {format_bytes(runtime.heap_min)} |")
        if runtime.heap_largest_last is not None:
            detail_rows.append(f"| Last largest heap block | {format_bytes(runtime.heap_largest_last)} |")
        if runtime.runtime_load_mean is not None:
            detail_rows.append(f"| Runtime load | {format_number(runtime.runtime_load_mean, '%')} mean |")
        if runtime.loop_avg_us_mean is not None:
            detail_rows.append(f"| Loop average | {format_number(runtime.loop_avg_us_mean, ' us')} |")
        if runtime.loop_max_us_max is not None:
            detail_rows.append(f"| Loop maximum | {format_number(runtime.loop_max_us_max, ' us')} |")

        if result.case.benchmark_mode == "runtime" and result.monitor:
            detail_rows.append(f"| Detection samples | {runtime.detection_samples} |")
            if runtime.detection_avg_us_mean is not None:
                detail_rows.append(f"| Detection average | {format_number(runtime.detection_avg_us_mean, ' us')} |")
            if runtime.detection_min_us is not None:
                detail_rows.append(f"| Detection minimum | {format_number(runtime.detection_min_us, ' us')} |")
            if runtime.detection_max_us is not None:
                detail_rows.append(f"| Detection maximum | {format_number(runtime.detection_max_us, ' us')} |")

        if result.case.benchmark_mode == "stream":
            if runtime.stream_telemetry_samples > 0:
                detail_rows.append(f"| Stream telemetry samples | {runtime.stream_telemetry_samples} |")
            if runtime.stream_csi_ap_mean is not None:
                detail_rows.append(f"| Stream CSI accepted | {format_number(runtime.stream_csi_ap_mean, ' pps')} |")
            if runtime.stream_udp_rx_mean is not None:
                detail_rows.append(f"| Stream UDP RX | {format_number(runtime.stream_udp_rx_mean, ' pps')} |")
            if runtime.stream_udp_tx_mean is not None:
                detail_rows.append(f"| Stream UDP TX | {format_number(runtime.stream_udp_tx_mean, ' pps')} |")
            if runtime.stream_fresh_mean is not None:
                detail_rows.append(f"| Stream fresh records | {format_number(runtime.stream_fresh_mean, ' pps')} |")
            detail_rows.append(f"| Stream TX backpressure total | {format_number(runtime.stream_tx_backpressure_total)} |")
            detail_rows.append(f"| Host collect devices | {runtime.collect_devices_observed} |")
            detail_rows.append(f"| Host collect packets | {runtime.collect_packets_seen} |")

        lines.extend(
            [
                f"### {result.case.label}",
                "",
                f"Result: **{result.status}**",
                "",
                "| Metric | Value |",
                "|---|---:|",
                *detail_rows,
                "",
            ]
        )
        if result.reasons:
            lines.extend(["Failure reasons:", ""])
            lines.extend(f"- {reason}" for reason in result.reasons)
            lines.append("")

    lines.extend(
        [
            "## Pass Criteria",
            "",
            "- all builds and flashes complete successfully",
            f"- at least {MIN_TELEMETRY_SAMPLES} shared debug telemetry samples are logged",
            "- free heap does not decline by more than 5% during monitoring",
            "- ESPHome and Native runtime benchmarks log at least "
            f"{MIN_STATUS_SAMPLES} valid motion states with non-zero packet rates",
            f"- ESPHome and Native mean packet rate remains between {EXPECTED_PPS_MIN} and {EXPECTED_PPS_MAX} pps",
            "- ESPHome and Native detector timing is present",
            "- Matter smoke benchmarks log a boot marker and the commissioning startup state",
            "- Streamer benchmarks log the device IP, reach STREAMING, and sustain host collect around the target packet rate",
            f"- Streamer host collect logs at least {MIN_STREAMER_COLLECT_SAMPLES} classic and ML samples",
            "- no fatal firmware log is observed",
            "",
        ]
    )
    return "\n".join(lines)


def report_path_for_chip(chip: str) -> Path:
    return REPO_ROOT / "docs" / "performance" / f"{CHIP_LABELS[chip]}.md"


def write_report(
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    expected_cases: Sequence[BenchmarkCase] = CASES,
) -> Path:
    destination = report_path_for_chip(chip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(chip, port, started_at, results, expected_cases),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, flash, and benchmark ESPHome/Native runtime variants, Matter smoke, and Streamer host collect for one chip.",
    )
    parser.add_argument("--chip", required=True, choices=SUPPORTED_CHIPS, help="Connected ESP32 target")
    parser.add_argument(
        "--frontend",
        choices=("esphome", "native", "matter", "streamer"),
        help="Run only cases for one frontend",
    )
    parser.add_argument(
        "--detector",
        choices=("classic", "ml", "collect"),
        help="Run only cases for one detector or the streamer collect workflow",
    )
    args = parser.parse_args()

    selected_cases = select_cases(args.frontend, args.detector)
    if not selected_cases:
        parser.error("the selected frontend and detector do not define a benchmark case")

    port = get_serial_port(None)
    detected_chip = detect_chip_type(port)
    if detected_chip is not None and detected_chip != args.chip:
        parser.error(
            f"connected device is {CHIP_LABELS.get(detected_chip, detected_chip)}, "
            f"but --chip selects {CHIP_LABELS[args.chip]}"
        )
    started_at = datetime.now().astimezone()
    results: list[BenchmarkResult] = []
    report_path = report_path_for_chip(args.chip)
    print(f"Chip:     {CHIP_LABELS[args.chip]}")
    print(f"Port:     {port}")
    print(f"Report:   {report_path.relative_to(REPO_ROOT)}")
    print(f"Matrix:   {', '.join(case.label for case in selected_cases)}")

    try:
        for frontend in ("esphome", "native"):
            frontend_cases = tuple(case for case in selected_cases if case.frontend == frontend)
            if not frontend_cases:
                continue
            classic_case = BenchmarkCase(frontend, "classic")
            ml_case = BenchmarkCase(frontend, "ml")
            if classic_case not in frontend_cases:
                ml_result, _unused = run_case(ml_case, args.chip, port, clean=True)
                results.append(ml_result)
                write_report(args.chip, port, started_at, results, selected_cases)
                continue

            overlap_ml = ml_case if ml_case in frontend_cases else None
            classic_result, ml_build = run_case(
                classic_case,
                args.chip,
                port,
                clean=True,
                overlap_build=overlap_ml,
            )
            results.append(classic_result)
            write_report(args.chip, port, started_at, results, selected_cases)

            if ml_case not in frontend_cases:
                continue

            if ml_build is None:
                classic_build_succeeded = (
                    classic_result.build is not None and classic_result.build.returncode == 0
                )
                ml_result, _unused = run_case(
                    ml_case,
                    args.chip,
                    port,
                    clean=not classic_build_succeeded,
                )
            else:
                ml_result, _unused = run_case(
                    ml_case,
                    args.chip,
                    port,
                    clean=False,
                    prebuilt=ml_build,
                )
            results.append(ml_result)
            write_report(args.chip, port, started_at, results, selected_cases)

        matter_case = BenchmarkCase("matter", "classic", benchmark_mode="smoke")
        if matter_case in selected_cases:
            matter_result, _unused = run_case(
                matter_case,
                args.chip,
                port,
                clean=True,
            )
            results.append(matter_result)
            write_report(args.chip, port, started_at, results, selected_cases)

        streamer_case = BenchmarkCase("streamer", "collect", benchmark_mode="stream")
        if streamer_case in selected_cases:
            streamer_result = run_streamer_case(
                streamer_case,
                args.chip,
                port,
                clean=True,
            )
            results.append(streamer_result)
            write_report(args.chip, port, started_at, results, selected_cases)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; writing the partial report.", file=sys.stderr)
        write_report(args.chip, port, started_at, results, selected_cases)
        return 130

    destination = write_report(args.chip, port, started_at, results, selected_cases)
    passed = all(result.status == "PASS" for result in results) and len(results) == len(selected_cases)
    print(f"\nWrote {destination}")
    print(f"Overall result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
