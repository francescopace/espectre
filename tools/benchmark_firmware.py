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
from typing import Iterator, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.espectre_cli.common import detect_chip_type, get_serial_port
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, IDF_FRONTENDS


MONITOR_DURATION_SECONDS = 180
EXPECTED_PPS_MIN = 90
EXPECTED_PPS_MAX = 110
MIN_STATUS_SAMPLES = 120
MIN_TELEMETRY_SAMPLES = 12
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


@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str

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
    pps_mean: float | None = None
    pps_min: int | None = None
    pps_max: int | None = None
    pps_stddev: float | None = None
    dominant_motion_state: str | None = None
    motion_transitions: int = 0
    dominant_state_share_percent: float | None = None
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


@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    status: str = "NOT RUN"
    reasons: list[str] = field(default_factory=list)
    build: CommandResult | None = None
    flash: CommandResult | None = None
    monitor: CommandResult | None = None
    build_metrics: BuildMetrics = field(default_factory=BuildMetrics)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)


CASES = tuple(
    BenchmarkCase(frontend, detector)
    for frontend in ("esphome", "native")
    for detector in ("classic", "ml")
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


def analyze_monitor_output(output: str) -> tuple[RuntimeMetrics, list[str]]:
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

    def values(key: str) -> list[float]:
        return [sample[key] for sample in telemetry if key in sample]

    heap_free = values("heap_free")
    heap_min = values("heap_min")
    heap_largest = values("heap_largest")
    runtime_load = values("runtime_load")
    loop_avg = values("loop_avg_us")
    loop_max = values("loop_max_us")
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

    if len(pps_values) < MIN_STATUS_SAMPLES:
        reasons.append(f"only {len(pps_values)} motion/packet-rate samples were logged")
    elif metrics.pps_mean is None or not EXPECTED_PPS_MIN <= metrics.pps_mean <= EXPECTED_PPS_MAX:
        reasons.append(
            f"mean packet rate {metrics.pps_mean:.2f} pps is outside "
            f"{EXPECTED_PPS_MIN}-{EXPECTED_PPS_MAX} pps"
        )

    if len(telemetry) < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {len(telemetry)} shared telemetry samples were logged")
    if heap_free and heap_free[-1] < heap_free[0] * 0.95:
        reasons.append("free heap declined by more than 5% during monitoring")
    if detection_samples == 0:
        reasons.append("detector timing was not logged")
    for pattern in FATAL_PATTERNS:
        if pattern in text:
            reasons.append(f"fatal firmware log detected: {pattern}")

    return metrics, reasons


def _latest_firmware_artifact(frontend: str) -> Path | None:
    if frontend == "esphome":
        candidates = list((REPO_ROOT / "examples" / ".esphome").glob("build/*/.pioenvs/*/firmware.bin"))
    else:
        app_dir = Path(IDF_FRONTENDS["native"]["app_dir"])
        build_dir = os.environ.get("ESPECTRE_IDF_BUILD_DIR", "build")
        candidates = [app_dir / build_dir / "espectre-native.bin"]
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


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
def native_case_environment(chip: str, detector: str) -> Iterator[dict[str, str]]:
    app_dir = Path(IDF_FRONTENDS["native"]["app_dir"])
    idf_target = IDF_FRONTENDS["native"]["targets"][chip]
    defaults = [app_dir / "sdkconfig.defaults"]
    target_defaults = app_dir / f"sdkconfig.defaults.{idf_target}"
    if target_defaults.is_file():
        defaults.append(target_defaults)

    native_wifi = app_dir / "sdkconfig.wifi"
    streamer_wifi = REPO_ROOT / "src" / "cpp" / "frontend" / "streamer" / "app" / "sdkconfig.wifi"
    wifi_defaults = native_wifi if native_wifi.is_file() else streamer_wifi
    if not wifi_defaults.is_file():
        raise RuntimeError("native Wi-Fi defaults are missing (expected native or streamer sdkconfig.wifi)")
    defaults.append(wifi_defaults)

    classic_enabled = detector == "classic"
    override = "\n".join(
        [
            "# Generated temporary firmware benchmark overrides.",
            "CONFIG_LOG_DEFAULT_LEVEL_DEBUG=y",
            "CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y",
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
            "",
        ]
    )
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
    build_command = [launcher, "native", "build", "--chip", chip]
    if clean:
        build_command.append("--clean")
    return (
        build_command,
        [launcher, "native", "flash", "--port", port],
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
        with native_case_environment(chip, case.detector) as env:
            if not clean:
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
            result.runtime_metrics, analysis_reasons = analyze_monitor_output(result.monitor.output)
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


def render_report(chip: str, port: str, started_at: datetime, results: Sequence[BenchmarkResult]) -> str:
    chip_label = CHIP_LABELS[chip]
    overall = "PASS" if len(results) == len(CASES) and all(result.status == "PASS" for result in results) else "FAIL"
    lines = [
        "<!-- Generated file. Do not edit manually. -->",
        "",
        f"# {chip_label} Firmware Performance",
        "",
        f"Generated by: `tools/benchmark_firmware.py --chip {chip}`",
        f"Git revision: `{_git_revision()}`",
        f"Run started: `{started_at.astimezone().isoformat(timespec='seconds')}`",
        f"Serial port: `{port}`",
        f"Monitor duration per firmware: `{MONITOR_DURATION_SECONDS} seconds`",
        f"Overall result: **{overall}**",
        "",
        "## Summary",
        "",
        "| Firmware | Result | Binary size | Partition free | Mean PPS | Motion samples | Detection average |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        partition_free = format_bytes(build.partition_free_bytes)
        if build.partition_free_percent is not None:
            partition_free += f" ({build.partition_free_percent:.1f}%)"
        lines.append(
            "| "
            + " | ".join(
                [
                    result.case.label,
                    f"**{result.status}**",
                    format_bytes(build.firmware_size_bytes),
                    partition_free,
                    format_number(runtime.pps_mean, " pps"),
                    format_number(runtime.status_samples),
                    format_number(runtime.detection_avg_us_mean, " us"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Results", ""])

    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        lines.extend(
            [
                f"### {result.case.label}",
                "",
                f"Result: **{result.status}**",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Build duration | {format_duration(result.build.duration_seconds) if result.build else 'N/A'} |",
                f"| Flash duration | {format_duration(result.flash.duration_seconds) if result.flash else 'N/A'} |",
                f"| Monitor duration | {format_duration(result.monitor.duration_seconds) if result.monitor else 'N/A'} |",
                f"| Firmware binary | {format_bytes(build.firmware_size_bytes)} |",
                f"| Application partition used | {format_bytes(build.partition_used_bytes)} |",
                f"| Application partition free | {format_bytes(build.partition_free_bytes)} |",
                f"| Build RAM used | {format_bytes(build.ram_used_bytes)} |",
                f"| Packet-rate samples | {runtime.status_samples} |",
                (
                    f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                    f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                    f"{format_number(runtime.pps_stddev)} standard deviation |"
                ),
                f"| Dominant motion state | {runtime.dominant_motion_state or 'N/A'} |",
                f"| Motion transitions | {runtime.motion_transitions} |",
                f"| Dominant state share | {format_number(runtime.dominant_state_share_percent, '%')} |",
                f"| Telemetry samples | {runtime.telemetry_samples} |",
                f"| Last free heap | {format_bytes(runtime.heap_free_last)} |",
                f"| Minimum free heap | {format_bytes(runtime.heap_min)} |",
                f"| Last largest heap block | {format_bytes(runtime.heap_largest_last)} |",
                f"| Runtime load | {format_number(runtime.runtime_load_mean, '%')} mean |",
                f"| Loop average | {format_number(runtime.loop_avg_us_mean, ' us')} |",
                f"| Loop maximum | {format_number(runtime.loop_max_us_max, ' us')} |",
                f"| Detection samples | {runtime.detection_samples} |",
                f"| Detection average | {format_number(runtime.detection_avg_us_mean, ' us')} |",
                f"| Detection minimum | {format_number(runtime.detection_min_us, ' us')} |",
                f"| Detection maximum | {format_number(runtime.detection_max_us, ' us')} |",
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
            f"- at least {MIN_STATUS_SAMPLES} valid motion states with non-zero packet rates are logged",
            f"- mean packet rate remains between {EXPECTED_PPS_MIN} and {EXPECTED_PPS_MAX} pps",
            "- motion transitions are informational and do not affect the result",
            f"- at least {MIN_TELEMETRY_SAMPLES} shared debug telemetry samples are logged",
            "- free heap does not decline by more than 5% during monitoring",
            "- detector timing is present, and no fatal firmware log is observed",
            "",
        ]
    )
    return "\n".join(lines)


def report_path_for_chip(chip: str) -> Path:
    return REPO_ROOT / "docs" / "performance" / f"{CHIP_LABELS[chip]}.md"


def write_report(chip: str, port: str, started_at: datetime, results: Sequence[BenchmarkResult]) -> Path:
    destination = report_path_for_chip(chip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(chip, port, started_at, results), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, flash, and benchmark every ESPHome and Native detector variant for one chip.",
    )
    parser.add_argument("--chip", required=True, choices=SUPPORTED_CHIPS, help="Connected ESP32 target")
    args = parser.parse_args()

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
    print(f"Matrix:   {', '.join(case.label for case in CASES)}")

    try:
        for frontend in ("esphome", "native"):
            classic_case = BenchmarkCase(frontend, "classic")
            ml_case = BenchmarkCase(frontend, "ml")
            classic_result, ml_build = run_case(
                classic_case,
                args.chip,
                port,
                clean=True,
                overlap_build=ml_case,
            )
            results.append(classic_result)
            write_report(args.chip, port, started_at, results)

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
            write_report(args.chip, port, started_at, results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; writing the partial report.", file=sys.stderr)
        write_report(args.chip, port, started_at, results)
        return 130

    destination = write_report(args.chip, port, started_at, results)
    passed = all(result.status == "PASS" for result in results) and len(results) == len(CASES)
    print(f"\nWrote {destination}")
    print(f"Overall result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
