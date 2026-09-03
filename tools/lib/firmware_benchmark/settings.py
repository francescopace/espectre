# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark settings owner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence
from dotenv import dotenv_values
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, IDF_FRONTENDS
from tools.lib.temporal_csi_sampler import (
    MINIMUM_COVERAGE_DENOMINATOR,
    MINIMUM_COVERAGE_NUMERATOR,
)
from src.python.espectre_cli.device_transport import (
    DEFAULT_DIRECT_ORIGIN,
)

from tools.lib.firmware_benchmark.models import BenchmarkCase

SCRIPT_DIR = Path(__file__).resolve().parents[2]

REPO_ROOT = SCRIPT_DIR.parent

BENCHMARK_LOCAL_ENV_PATH = SCRIPT_DIR / "benchmark_firmware.local.env"

BENCHMARK_LOCAL_ENV = dotenv_values(BENCHMARK_LOCAL_ENV_PATH) if BENCHMARK_LOCAL_ENV_PATH.is_file() else {}

BENCHMARK_ARTIFACT_ROOT = REPO_ROOT / "data" / "untracked" / "firmware_benchmarks"

MONITOR_DURATION_SECONDS = 60

WIFI_CONNECT_WAIT_SECONDS = 60

DIRECT_DISCOVERY_TIMEOUT_SECONDS = 45

DIRECT_SAMPLE_INTERVAL_SECONDS = 1.0

DIRECT_SAMPLE_PHASE_OFFSET_SECONDS = 0.125

DIRECT_MINIMUM_REQUEST_INTERVAL_SECONDS = 0.075

# Avoid sampling in lockstep with Micro's one-second cached diagnostics refresh.
# A half-second phase offset prevents adjacent 4 s/6 s snapshot deltas.
MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS = 4.5

# Micro serves diagnostics from a snapshot refreshed on a one-second cadence.
# Allow that quantization without weakening the C++ frontend cadence check.
MICRO_RUNTIME_STATUS_GAP_TOLERANCE_MS = 1000

DIRECT_STABLE_SAMPLE_COUNT = 5

CPP_DIRECT_RUNTIME_MINIMUM_UPTIME_SECONDS = 30

DIRECT_READINESS_MARGIN_SECONDS = 2.0

DIRECT_EVENT_OPEN_ATTEMPTS = 3

MICRO_DIRECT_PREPARE_ATTEMPTS = 3

DIRECT_ORIGIN = DEFAULT_DIRECT_ORIGIN

MINIMUM_OCCUPANCY_PERCENT = 100.0 * MINIMUM_COVERAGE_NUMERATOR / MINIMUM_COVERAGE_DENOMINATOR

STARTUP_GRACE_SECONDS = 10

HEAP_STABILITY_WINDOW_SECONDS = 10

HEAP_STABILITY_MAX_DECLINE_PERCENT = 5.0

MIN_MOTION_SAMPLES = 5

MICRO_SOURCE_DIR = REPO_ROOT / "src/python/micro_espectre"

STATUS_STABLE_WAIT_SECONDS = 30

BENCHMARK_CONTROL_TIMEOUT_SECONDS = 30.0

RUNTIME_STATUS_GAP_TOLERANCE_MS = 500

RUNTIME_STATUS_BOUNDARY_TOLERANCE_SAMPLES = 1

MINIMUM_BENCHMARK_CSI_TARGET_PPS = 100

SUPPORTED_CHIPS = tuple(sorted(set(ESPHOME_CONFIGS) & set(IDF_FRONTENDS["native"]["targets"])))

def benchmark_setting(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    value = BENCHMARK_LOCAL_ENV.get(name)
    if value is None:
        return default
    return str(value)

def benchmark_setting_int(name: str, default: int) -> int:
    value = benchmark_setting(name)
    if value is None or value == "":
        return default
    return int(value)

def require_benchmark_setting(name: str) -> str:
    value = benchmark_setting(name)
    if value is None or value == "":
        raise RuntimeError(
            f"missing required benchmark setting {name}; "
            f"configure {BENCHMARK_LOCAL_ENV_PATH.relative_to(REPO_ROOT)} or export the variable"
        )
    return value

def require_benchmark_prerequisites(cases: Sequence[BenchmarkCase]) -> None:
    if cases:
        require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
        require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
    if (
        any(case.frontend in {"native", "esphome", "matter", "micro"} for case in cases)
        and benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0) > 0
        and not benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")
    ):
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_WIFI_CHANNEL requires "
            "ESPECTRE_BENCHMARK_WIFI_BSSID so the benchmark can pin and verify one access point"
        )
