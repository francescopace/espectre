# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Matter controller automation for firmware benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from tools.lib.firmware_benchmark.models import CommandResult
from tools.lib.firmware_benchmark.process import run_command
from tools.lib.firmware_benchmark.settings import (
    REPO_ROOT,
    benchmark_setting,
    benchmark_setting_int,
    require_benchmark_setting,
)


CONNECTEDHOMEIP_REVISION_PATTERN = re.compile(r"works with commit \[([0-9a-f]{10,40})\]")
DEFAULT_MATTER_NODE_ID = 0xE5C30001
DEFAULT_MATTER_COMMISSIONING_TIMEOUT_SECONDS = 180
DEFAULT_MATTER_COMMISSIONING_ATTEMPTS = 2


@dataclass(frozen=True)
class MatterOnboardingData:
    qr_payload: str
    manual_code: str


@dataclass(frozen=True)
class MatterCommissioningEvidence:
    result: CommandResult
    controller: str
    controller_revision: str
    node_id: int


class MatterOnboardingCapture:
    """Capture onboarding data while redacting it from delegated command output."""

    def __init__(self) -> None:
        self._qr_payload = ""
        self._manual_code = ""

    def feed(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or event.get("event") != "matter_onboarding":
            return
        qr_payload = event.get("qr_payload")
        manual_code = event.get("manual_code")
        if isinstance(qr_payload, str):
            self._qr_payload = qr_payload
        if isinstance(manual_code, str):
            self._manual_code = manual_code

    def redact(self, line: str) -> str:
        redacted = line
        for sensitive in (self._qr_payload, self._manual_code):
            if sensitive:
                redacted = redacted.replace(sensitive, "<redacted>")
        return redacted

    def require_data(self) -> MatterOnboardingData:
        if not self._qr_payload or not self._manual_code:
            raise RuntimeError("Matter flash did not expose complete onboarding data")
        return MatterOnboardingData(self._qr_payload, self._manual_code)


def expected_connectedhomeip_revision() -> str:
    readme = (
        REPO_ROOT
        / "src/cpp/frontend/matter/app/managed_components/espressif__esp_matter/README.md"
    )
    try:
        content = readme.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("could not inspect the esp-matter connectedhomeip revision") from exc
    match = CONNECTEDHOMEIP_REVISION_PATTERN.search(content)
    if match is None:
        raise RuntimeError("esp-matter does not declare a connectedhomeip revision")
    return match.group(1)


def resolve_chip_tool() -> tuple[Path, str]:
    """Return a revision-compatible installed CHIP Tool executable."""
    configured = benchmark_setting("ESPECTRE_BENCHMARK_CHIP_TOOL", "") or ""
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(resolved) if (resolved := shutil.which("chip-tool")) else None,
        Path.home() / ".local" / "bin" / "chip-tool",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            resolved_candidate = candidate.resolve()
            configured_revision = benchmark_setting("ESPECTRE_BENCHMARK_CHIP_TOOL_REVISION", "") or ""
            revision_path = resolved_candidate.with_name(f"{resolved_candidate.name}.revision")
            if configured_revision:
                revision = configured_revision.strip()
            elif revision_path.is_file():
                revision = revision_path.read_text(encoding="utf-8").strip()
            else:
                raise RuntimeError(
                    "chip-tool revision is unknown; install chip-tool.revision beside the binary or "
                    "set ESPECTRE_BENCHMARK_CHIP_TOOL_REVISION"
                )
            expected = expected_connectedhomeip_revision()
            if not revision.startswith(expected):
                raise RuntimeError(
                    f"chip-tool revision {revision[:10]} does not match esp-matter revision {expected}"
                )
            return resolved_candidate, revision
    raise RuntimeError(
        "chip-tool is required for Matter benchmarks; install it or set "
        "ESPECTRE_BENCHMARK_CHIP_TOOL"
    )


def matter_node_id() -> int:
    raw = benchmark_setting("ESPECTRE_BENCHMARK_MATTER_NODE_ID", hex(DEFAULT_MATTER_NODE_ID))
    try:
        node_id = int(raw or "", 0)
    except ValueError as exc:
        raise RuntimeError("ESPECTRE_BENCHMARK_MATTER_NODE_ID must be an integer") from exc
    if node_id <= 0 or node_id > 0xFFFFFFEFFFFFFFFF:
        raise RuntimeError("ESPECTRE_BENCHMARK_MATTER_NODE_ID is outside the operational range")
    return node_id


def commission_matter_device(onboarding: MatterOnboardingData) -> MatterCommissioningEvidence:
    """Commission one erased Matter device through BLE and production Wi-Fi credentials."""
    chip_tool, controller_revision = resolve_chip_tool()
    ssid = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
    password = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
    ssid_argument = f"hex:{ssid.encode('utf-8').hex()}"
    password_argument = f"hex:{password.encode('utf-8').hex()}"
    node_id = matter_node_id()
    timeout_seconds = benchmark_setting_int(
        "ESPECTRE_BENCHMARK_MATTER_COMMISSIONING_TIMEOUT_SECONDS",
        DEFAULT_MATTER_COMMISSIONING_TIMEOUT_SECONDS,
    )
    if timeout_seconds <= 0:
        raise RuntimeError("ESPECTRE_BENCHMARK_MATTER_COMMISSIONING_TIMEOUT_SECONDS must be positive")
    attempts = benchmark_setting_int(
        "ESPECTRE_BENCHMARK_MATTER_COMMISSIONING_ATTEMPTS",
        DEFAULT_MATTER_COMMISSIONING_ATTEMPTS,
    )
    if attempts <= 0:
        raise RuntimeError("ESPECTRE_BENCHMARK_MATTER_COMMISSIONING_ATTEMPTS must be positive")

    with tempfile.TemporaryDirectory(prefix="espectre-matter-controller-") as storage_directory:
        command = [
            str(chip_tool),
            "pairing",
            "code-wifi",
            str(node_id),
            ssid_argument,
            password_argument,
            onboarding.manual_code,
            "--storage-directory",
            storage_directory,
            "--timeout",
            str(timeout_seconds),
        ]
        for attempt in range(1, attempts + 1):
            result = run_command(
                command,
                timeout=float(timeout_seconds + 15),
                redactions=(
                    ssid,
                    password,
                    ssid_argument,
                    password_argument,
                    onboarding.qr_payload,
                    onboarding.manual_code,
                ),
            )
            if result.returncode == 0:
                break
            if attempt < attempts:
                print(
                    f"Matter commissioning attempt {attempt}/{attempts} failed; retrying...",
                    flush=True,
                )
                time.sleep(2.0)
        else:
            if (
                sys.platform == "darwin"
                and "GATT write characteristic operation failed" in result.output
            ):
                raise RuntimeError(
                    "Matter commissioning failed because macOS rejected the BLE GATT write; "
                    "install the Bluetooth Central Matter Client Developer Mode profile, "
                    "restart macOS, and rerun the benchmark"
                )
            raise RuntimeError(
                f"Matter commissioning failed after {attempts} attempts "
                f"(last status {result.returncode})"
            )
    return MatterCommissioningEvidence(result, chip_tool.name, controller_revision, node_id)
