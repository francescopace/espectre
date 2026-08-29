# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Matter benchmark controller contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tools.lib.firmware_benchmark import matter as bench
from tools.lib.firmware_benchmark.models import CommandResult
from tools.lib.firmware_benchmark.process import run_command


def test_onboarding_capture_retains_codes_and_redacts_output():
    capture = bench.MatterOnboardingCapture()
    qr_line = "  QR payload:  MT:Y.K90-C714FGCO6MZ00\n"
    manual_line = "  Manual code: 12704227053\n"

    capture.feed(qr_line)
    capture.feed(manual_line)

    assert capture.require_data() == bench.MatterOnboardingData(
        "MT:Y.K90-C714FGCO6MZ00",
        "12704227053",
    )
    assert "MT:Y.K90-C714FGCO6MZ00" not in capture.redact(qr_line)
    assert "12704227053" not in capture.redact(manual_line)


def test_resolve_chip_tool_requires_matching_revision(tmp_path, monkeypatch):
    chip_tool = tmp_path / "chip-tool"
    chip_tool.write_text("binary", encoding="utf-8")
    chip_tool.chmod(0o755)
    chip_tool.with_name("chip-tool.revision").write_text(
        "cf84d0360c48dbc194c48b47b09169f302a9745b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ESPECTRE_BENCHMARK_CHIP_TOOL", str(chip_tool))
    monkeypatch.delenv("ESPECTRE_BENCHMARK_CHIP_TOOL_REVISION", raising=False)
    monkeypatch.setattr(bench, "expected_connectedhomeip_revision", lambda: "cf84d0360c")

    resolved, revision = bench.resolve_chip_tool()

    assert resolved == chip_tool
    assert revision == "cf84d0360c48dbc194c48b47b09169f302a9745b"


def test_commission_matter_device_uses_ephemeral_storage_and_redactions(
    tmp_path, monkeypatch
):
    chip_tool = tmp_path / "chip-tool"
    chip_tool.write_text("binary", encoding="utf-8")
    chip_tool.chmod(0o755)
    monkeypatch.setattr(
        bench,
        "resolve_chip_tool",
        lambda: (chip_tool, "cf84d0360c48dbc194c48b47b09169f302a9745b"),
    )
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "Matter Lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "matter-secret")
    observed: dict[str, object] = {}

    def fake_run_command(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = kwargs
        storage = Path(command[command.index("--storage-directory") + 1])
        observed["storage"] = storage
        assert storage.is_dir()
        return CommandResult(list(command), 0, 1.0, "commissioned")

    monkeypatch.setattr(bench, "run_command", fake_run_command)

    evidence = bench.commission_matter_device(
        bench.MatterOnboardingData("MT:TESTPAYLOAD", "12704227053")
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[1:3] == ["pairing", "code-wifi"]
    assert "hex:4d6174746572204c6162" in command
    assert "hex:6d61747465722d736563726574" in command
    assert "12704227053" in command
    assert not Path(observed["storage"]).exists()
    assert "12704227053" in observed["kwargs"]["redactions"]
    assert evidence.controller == "chip-tool"
    assert evidence.controller_revision.startswith("cf84d0360c")


def test_commission_matter_device_retries_transient_controller_failure(
    tmp_path, monkeypatch
):
    chip_tool = tmp_path / "chip-tool"
    chip_tool.write_text("binary", encoding="utf-8")
    chip_tool.chmod(0o755)
    monkeypatch.setattr(bench, "resolve_chip_tool", lambda: (chip_tool, "cf84d0360c"))
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    results = iter(
        (
            CommandResult(["chip-tool"], 1, 1.0, "transient"),
            CommandResult(["chip-tool"], 0, 1.0, "commissioned"),
        )
    )
    calls = []

    def fake_run_command(command, **_kwargs):
        calls.append(command)
        return next(results)

    monkeypatch.setattr(bench, "run_command", fake_run_command)

    evidence = bench.commission_matter_device(
        bench.MatterOnboardingData("MT:TESTPAYLOAD", "12704227053")
    )

    assert len(calls) == 2
    assert evidence.result.returncode == 0


def test_commission_matter_device_explains_missing_macos_profile(tmp_path, monkeypatch):
    chip_tool = tmp_path / "chip-tool"
    chip_tool.write_text("binary", encoding="utf-8")
    chip_tool.chmod(0o755)
    monkeypatch.setattr(bench, "resolve_chip_tool", lambda: (chip_tool, "cf84d0360c"))
    monkeypatch.setattr(bench.sys, "platform", "darwin")
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setattr(
        bench,
        "run_command",
        lambda *_args, **_kwargs: CommandResult(
            ["chip-tool"],
            1,
            1.0,
            "GATT write characteristic operation failed",
        ),
    )

    with pytest.raises(RuntimeError, match="Bluetooth Central Matter Client"):
        bench.commission_matter_device(
            bench.MatterOnboardingData("MT:TESTPAYLOAD", "12704227053")
        )


def test_run_command_redacts_command_and_output(capsys):
    result = run_command(
        [sys.executable, "-c", "print('matter-secret')"],
        redactions=("matter-secret",),
    )

    assert result.returncode == 0
    assert "matter-secret" not in " ".join(result.command)
    assert "matter-secret" not in result.output
    assert "matter-secret" not in capsys.readouterr().out
