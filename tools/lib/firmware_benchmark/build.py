# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark build owner."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
from typing import Callable, Iterator
from src.python.espectre_cli.idf import cached_sdkconfig_path, resolve_idf_build_dir_name
from src.python.espectre_cli.micro import deployment_files
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, IDF_FRONTENDS

from tools.lib.firmware_benchmark.analysis import strip_ansi
from tools.lib.firmware_benchmark.models import BenchmarkCase, BenchmarkResult, BuildMetrics
from tools.lib.firmware_benchmark.process import parse_json_object_from_output, run_command
from tools.lib.firmware_benchmark.settings import (
    MICRO_SOURCE_DIR,
    REPO_ROOT,
    benchmark_setting,
    benchmark_setting_int,
    require_benchmark_setting,
)

def parse_build_metrics(output: str, firmware_path: Path | None = None) -> BuildMetrics:
    text = strip_ansi(output)
    metrics = BuildMetrics()

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

    app_image_match = re.search(
        r"(?:binary size\s+0x(?P<app_size>[0-9a-f]+)\s+bytes\.\s+)?"
        r"Smallest app partition is\s+0x(?P<part_total>[0-9a-f]+)\s+bytes\.\s+"
        r"0x(?P<part_free>[0-9a-f]+)\s+bytes\s+\((?P<part_free_pct>\d+)%\)\s+free",
        text,
        flags=re.IGNORECASE,
    )
    if app_image_match:
        app_size = app_image_match.group("app_size")
        if app_size is not None:
            metrics.firmware_size_bytes = int(app_size, 16)
        if metrics.partition_used_bytes is None:
            metrics.partition_total_bytes = int(app_image_match.group("part_total"), 16)
            metrics.partition_free_bytes = int(app_image_match.group("part_free"), 16)
            metrics.partition_free_percent = float(app_image_match.group("part_free_pct"))
            metrics.partition_used_bytes = metrics.partition_total_bytes - metrics.partition_free_bytes

    if firmware_path is not None and firmware_path.is_file():
        metrics.firmware_size_bytes = firmware_path.stat().st_size
        digest = hashlib.sha256()
        with firmware_path.open("rb") as firmware:
            for chunk in iter(lambda: firmware.read(1024 * 1024), b""):
                digest.update(chunk)
        metrics.firmware_sha256 = digest.hexdigest()

    return metrics


TRAFFIC_GENERATOR_CONFIGS = {
    "ping": "CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_PING",
    "dns": "CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS",
    "dns_tcp": "CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS_TCP",
}


def configured_traffic_generator_mode(frontend: str, chip: str) -> str:
    """Return the production traffic mode owned by the frontend configuration."""
    if frontend == "micro":
        config_path = MICRO_SOURCE_DIR / "config.py"
        match = re.search(
            r'''(?m)^TRAFFIC_GENERATOR_MODE\s*=\s*["'](ping|dns|dns_tcp)["'](?:\s*#.*)?$''',
            config_path.read_text(encoding="utf-8"),
        )
        if match is None:
            raise RuntimeError(
                f"Micro config does not select a traffic generator mode: {config_path}"
            )
        return match.group(1)
    if frontend == "esphome":
        config_path = Path(ESPHOME_CONFIGS[chip])
        match = re.search(
            r"(?m)^\s+traffic_generator_mode:\s*(ping|dns|dns_tcp)(?:\s|#|$)",
            config_path.read_text(encoding="utf-8"),
        )
        if match is None:
            raise RuntimeError(f"ESPHome config does not select a traffic generator mode: {config_path}")
        return match.group(1)
    if frontend not in IDF_FRONTENDS:
        raise RuntimeError(f"unsupported benchmark frontend: {frontend}")

    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    selected = "ping"
    for defaults in (
        app_dir / "sdkconfig.defaults",
        app_dir / f"sdkconfig.defaults.{idf_target}",
    ):
        if not defaults.is_file():
            continue
        content = defaults.read_text(encoding="utf-8")
        for mode, symbol in TRAFFIC_GENERATOR_CONFIGS.items():
            if re.search(rf"(?m)^{re.escape(symbol)}=y\s*$", content):
                selected = mode
    return selected

def build_artifact_from_output(
    output: str,
    *,
    frontend: str,
    chip: str,
) -> tuple[Path, dict[str, object]]:
    """Return the artifact declared by a delegated CLI build command."""
    metadata = parse_json_object_from_output(output)
    if metadata.get("command") not in {"build", "flash"}:
        raise RuntimeError("delegated CLI command did not emit build artifact metadata")
    if metadata.get("frontend") != frontend or metadata.get("chip") != chip:
        raise RuntimeError("delegated CLI build metadata does not match the benchmark case")
    artifact_value = metadata.get("artifact")
    if not isinstance(artifact_value, str) or not artifact_value:
        raise RuntimeError("delegated CLI build metadata did not contain an artifact")
    artifact = Path(artifact_value)
    if not artifact.is_file():
        raise RuntimeError(f"delegated CLI build artifact does not exist: {artifact}")
    return artifact, metadata

def render_micro_benchmark_config() -> str:
    """Configure only the connectivity that Micro cannot provision at runtime."""
    values: list[tuple[str, object]] = [
        ("WIFI_SSID", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")),
        ("WIFI_PASSWORD", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")),
        ("WIFI_BSSID", benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")),
        ("WIFI_CHANNEL", benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)),
    ]
    lines = [
        "# Generated temporary Micro-ESPectre laboratory environment overrides.",
        *(f"{name} = {value!r}" for name, value in values),
        "",
    ]
    return "\n".join(lines)

@contextmanager
def micro_case_config(chip: str, detector: str) -> Iterator[Path]:
    """Yield an isolated config deployed through the production Micro CLI."""
    temporary_path = MICRO_SOURCE_DIR / f".espectre-benchmark-{chip}-{detector}.py"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark config already exists: {temporary_path}")
    try:
        temporary_path.write_text(
            render_micro_benchmark_config(),
            encoding="utf-8",
        )
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)

def micro_deployed_source_size(config_path: Path) -> int:
    """Return the exact source footprint selected by the production deploy manifest."""
    return sum(Path(source).stat().st_size for source, _destination in deployment_files(config_path))

def _commands_for_case(
    case: BenchmarkCase,
    chip: str,
    port: str | None,
    config: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    launcher = str(REPO_ROOT / "espectre")
    # Always use the shared serial monitor and request an explicit hard reset so
    # one-shot boot markers (especially Matter onboarding data) are captured.
    monitor_command = [
        launcher,
        "monitor",
        "--chip",
        chip,
        "--frontend",
        case.frontend,
        "--reset",
    ]
    if port:
        monitor_command.extend(["--port", port])
    if case.frontend == "esphome":
        build_command = [
            launcher,
            "esphome",
            "build",
            "--chip",
            chip,
            "--json",
        ]
        flash_command = [
            launcher,
            "esphome",
            "flash",
            "--chip",
            chip,
            "--erase",
        ]
        if port:
            flash_command.extend(["--device", port])
        return build_command, flash_command, monitor_command
    build_command = [
        launcher,
        case.frontend,
        "build",
        "--chip",
        chip,
        "--backend",
        "local",
        "--json",
    ]
    flash_command = [launcher, case.frontend, "flash", "--chip", chip]
    flash_command.append("--erase")
    if case.frontend == "matter":
        flash_command.append("--json")
    if port:
        flash_command.extend(["--port", port])
    return build_command, flash_command, monitor_command

@contextmanager
def case_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
) -> Iterator[tuple[dict[str, str] | None, Path | None]]:
    del case, chip, port
    yield None, None


def validate_idf_benchmark_sdkconfig(frontend: str, chip: str) -> None:
    """Reject a reusable local build that does not contain production defaults."""
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    build_dir_name = resolve_idf_build_dir_name(app_dir, idf_target)
    path = cached_sdkconfig_path(app_dir, build_dir_name)
    using_default_sdkconfig = path is None
    if using_default_sdkconfig:
        path = app_dir / "sdkconfig"
    if path is None or not path.is_file():
        raise RuntimeError("could not inspect the resolved ESP-IDF configuration")
    content = path.read_text(encoding="utf-8")
    if using_default_sdkconfig and f'CONFIG_IDF_TARGET="{idf_target}"' not in content:
        raise RuntimeError("could not inspect the resolved ESP-IDF configuration")
    expected_traffic_mode = configured_traffic_generator_mode(frontend, chip)
    required_lines = (
        "CONFIG_ESPECTRE_DETECTION_ALGORITHM_LIGHTWEIGHT=y",
        "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_HIGH_ACCURACY is not set",
        "CONFIG_ESPECTRE_CSI_TARGET_PPS=100",
        "CONFIG_ESPECTRE_CSI_TRAFFIC_MODE_INTERNAL=y",
    )
    required_lines += tuple(
        f"{symbol}=y" if mode == expected_traffic_mode else f"# {symbol} is not set"
        for mode, symbol in TRAFFIC_GENERATOR_CONFIGS.items()
    )
    if frontend == "native":
        required_lines += (
            'CONFIG_ESPECTRE_WIFI_SSID=""',
            'CONFIG_ESPECTRE_WIFI_PASSWORD=""',
            'CONFIG_ESPECTRE_WIFI_BSSID=""',
            'CONFIG_ESPECTRE_DEVICE_LABEL=""',
            "# CONFIG_ESPECTRE_MQTT_ENABLED is not set",
            'CONFIG_ESPECTRE_MQTT_HOST=""',
            'CONFIG_ESPECTRE_MQTT_USERNAME=""',
            'CONFIG_ESPECTRE_MQTT_PASSWORD=""',
        )
    missing = [line for line in required_lines if line not in content]
    if missing:
        raise RuntimeError(
            f"resolved {frontend} firmware does not use production defaults: "
            + ", ".join(missing)
        )


def _build_case_in_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    env: dict[str, str] | None,
    config: Path | None,
    output_prefix: str = "",
) -> BenchmarkResult:
    result = BenchmarkResult(case=case)
    build_command, _flash_command, _monitor_command = _commands_for_case(
        case,
        chip,
        port,
        config,
    )
    result.build = run_command(build_command, env=env, output_prefix=output_prefix)
    artifact = None
    artifact_metadata: dict[str, object] = {}
    if result.build.returncode == 0:
        artifact, artifact_metadata = build_artifact_from_output(
            result.build.output,
            frontend=case.frontend,
            chip=chip,
        )
    result.build_metrics = parse_build_metrics(result.build.output, artifact)
    if artifact_metadata:
        if artifact_metadata.get("firmware_size_bytes") != result.build_metrics.firmware_size_bytes:
            raise RuntimeError("delegated CLI build size does not match the artifact")
        if artifact_metadata.get("firmware_sha256") != result.build_metrics.firmware_sha256:
            raise RuntimeError("delegated CLI build hash does not match the artifact")
    if case.frontend in IDF_FRONTENDS and result.build.returncode == 0:
        validate_idf_benchmark_sdkconfig(case.frontend, chip)
    if result.build.returncode != 0:
        result.status = "FAIL"
        result.reasons.append(f"build exited with status {result.build.returncode}")
        return result
    return result

def _flash_prebuilt_cpp_case_in_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
    result: BenchmarkResult,
    *,
    env: dict[str, str] | None,
    config: Path | None,
    line_callback: Callable[[str], None] | None = None,
    output_redactor: Callable[[str], str] | None = None,
) -> bool:
    _build_command, flash_command, _monitor_command = _commands_for_case(
        case,
        chip,
        port,
        config,
    )
    result.flash = run_command(
        flash_command,
        env=env,
        line_callback=line_callback,
        output_redactor=output_redactor,
    )
    if result.flash.returncode != 0:
        result.reasons.append(f"flash exited with status {result.flash.returncode}")
        result.status = "FAIL"
        return False
    return True

def run_cpp_build_flash_case(case: BenchmarkCase, chip: str, port: str) -> BenchmarkResult:
    """Build and flash one C++ smoke case without opening a scored transport."""
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    try:
        with case_context(case, chip, port) as (env, config):
            result = _build_case_in_context(
                case,
                chip,
                port,
                env=env,
                config=config,
            )
            if result.build is None or result.build.returncode != 0:
                return result
            if not _flash_prebuilt_cpp_case_in_context(
                case,
                chip,
                port,
                result,
                env=env,
                config=config,
            ):
                return result
    except (OSError, RuntimeError) as exc:
        return BenchmarkResult(case=case, status="FAIL", reasons=[str(exc)])
    result.status = "PASS"
    result.transport_evidence = {"transport": "flash-only"}
    return result
