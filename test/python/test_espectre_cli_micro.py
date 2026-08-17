# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Micro Tests

Tests for espectre_cli.micro host-side helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from espectre_cli import micro


def _make_args(**overrides) -> argparse.Namespace:
    args = {
        "port": None,
        "chip": "c3",
        "erase": False,
        "firmware": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _make_verify_args(**overrides) -> argparse.Namespace:
    args = {"port": None}
    args.update(overrides)
    return argparse.Namespace(**args)


class _FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.headers = {"content-length": str(sum(len(chunk) for chunk in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _create_micro_src_tree(base_dir: Path) -> None:
    files = [
        "__init__.py",
        "branding.py",
        "config.py",
        "config_local.py",
        "device_utils.py",
        "utils.py",
        "threshold.py",
        "filters.py",
        "csi_features.py",
        "segmentation.py",
        "detector_interface.py",
        "runtime_policy.py",
        "runtime_diagnostics.py",
        "lightweight_detector.py",
        "high_accuracy_detector.py",
        "ml_weights.py",
        "traffic_generator.py",
        "console_output.py",
        "main.py",
        "mqtt/__init__.py",
        "mqtt/handler.py",
        "mqtt/commands.py",
        "mqtt/home_assistant.py",
    ]
    for rel_path in files:
        target = base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# test\n", encoding="utf-8")


def test_calculate_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"firmware-bytes"
    firmware = tmp_path / "firmware.bin"
    firmware.write_bytes(payload)

    assert micro._calculate_sha256(firmware) == hashlib.sha256(payload).hexdigest()


def test_device_sources_avoid_unsupported_future_annotations() -> None:
    for rel_path in micro.MICRO_DEVICE_RELATIVE_FILES:
        source = micro.PYTHON_SRC_DIR / rel_path
        if not source.exists():
            # config_local.py only exists after local setup; check the
            # shipped template instead.
            source = source.with_name(source.name + ".example")
            assert source.exists(), rel_path
        assert "from __future__ import annotations" not in source.read_text(
            encoding="utf-8"
        ), rel_path


def test_require_mpremote_accepts_installed_binary(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    micro._require_mpremote()

    assert calls == [["mpremote", "--version"]]


def test_require_mpremote_exits_when_binary_missing(monkeypatch) -> None:
    def fake_run(cmd, capture_output, check):
        raise FileNotFoundError()

    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        micro._require_mpremote()


def test_reset_device_reports_completion_even_on_exception(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)

    def fake_run(cmd, timeout, capture_output):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(micro.subprocess, "run", fake_run)
    micro._reset_device("/dev/cu.usbmodem1")

    def fake_run_fail(cmd, timeout, capture_output):
        raise RuntimeError("busy")

    monkeypatch.setattr(micro.subprocess, "run", fake_run_fail)
    micro._reset_device("/dev/cu.usbmodem1")

    out = capsys.readouterr().out
    assert calls == [["mpremote", "connect", "/dev/cu.usbmodem1", "exec", "import machine; machine.reset()"]]
    assert out.count("ESP32 reset completed") == 2


def test_download_firmware_uses_verified_cache(tmp_path: Path, monkeypatch) -> None:
    payload = b"cached-ok"
    firmware = tmp_path / "ESP32_CSI_C3.bin"
    firmware.write_bytes(payload)
    monkeypatch.setattr(micro, "FIRMWARE_HASHES", {"ESP32_CSI_C3.bin": hashlib.sha256(payload).hexdigest()})
    monkeypatch.setattr(micro.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no download")))

    assert micro.download_firmware("c3", tmp_path) == firmware


def test_download_firmware_redownloads_when_cached_hash_mismatches(tmp_path: Path, monkeypatch) -> None:
    stale = b"stale"
    fresh = b"fresh-firmware"
    firmware = tmp_path / "ESP32_CSI_C3.bin"
    firmware.write_bytes(stale)
    monkeypatch.setattr(micro, "FIRMWARE_HASHES", {"ESP32_CSI_C3.bin": hashlib.sha256(fresh).hexdigest()})
    monkeypatch.setattr(micro.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse([fresh]))

    result = micro.download_firmware("c3", tmp_path)

    assert result == firmware
    assert firmware.read_bytes() == fresh


def test_download_firmware_supports_unknown_chip_without_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(micro, "FIRMWARE_HASHES", {})
    monkeypatch.setattr(micro.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse([b"raw-fw"]))

    result = micro.download_firmware("h2", tmp_path)

    assert result.name == "ESP32_CSI_H2.bin"
    assert result.read_bytes() == b"raw-fw"


def test_download_firmware_rejects_bad_download_hash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(micro, "FIRMWARE_HASHES", {"ESP32_CSI_C3.bin": "deadbeef"})
    monkeypatch.setattr(micro.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse([b"bad-fw"]))

    with pytest.raises(SystemExit):
        micro.download_firmware("c3", tmp_path)

    assert not (tmp_path / "ESP32_CSI_C3.bin").exists()


def test_download_firmware_handles_network_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(micro, "FIRMWARE_HASHES", {})

    def fake_urlopen(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr(micro.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit):
        micro.download_firmware("c3", tmp_path)


def test_flash_firmware_raises_when_esptool_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "esptool":
            raise ImportError("missing esptool")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit):
        micro.flash_firmware(_make_args())


def test_flash_firmware_retries_after_failed_write_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    firmware = tmp_path / "fw.bin"
    firmware.write_bytes(b"fw")
    calls: list[list[str]] = []

    class FakeEsptool:
        def __init__(self):
            self._attempt = 0

        def main(self, cmd):
            calls.append(cmd)
            if cmd[-1] == str(firmware) and self._attempt == 0:
                self._attempt += 1
                raise RuntimeError("temporary failure")

    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(micro, "prompt_chip_type", lambda: "c5")
    monkeypatch.setattr(micro, "download_firmware", lambda chip, cache_dir: firmware)
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)
    monkeypatch.setitem(sys.modules, "esptool", FakeEsptool())

    micro.flash_firmware(_make_args(chip=None, erase=True))

    assert calls[0] == ["--chip", "esp32c5", "--port", "/dev/cu.usbmodem1", "--baud", "460800", "erase-flash"]
    assert any("0x2000" in cmd for cmd in calls[1:])
    assert len(calls) == 3


def test_flash_firmware_rejects_missing_custom_firmware(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "esptool", SimpleNamespace(main=lambda _cmd: None))
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")

    with pytest.raises(SystemExit):
        micro.flash_firmware(_make_args(firmware="/tmp/does-not-exist.bin"))


def test_flash_firmware_exits_when_chip_cannot_be_selected(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "esptool", SimpleNamespace(main=lambda _cmd: None))
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(micro, "prompt_chip_type", lambda: None)

    with pytest.raises(SystemExit):
        micro.flash_firmware(_make_args(chip=None))


def test_flash_firmware_exits_after_exhausting_retries(tmp_path: Path, monkeypatch) -> None:
    firmware = tmp_path / "fw.bin"
    firmware.write_bytes(b"fw")
    attempts: list[list[str]] = []

    def always_fail(cmd):
        attempts.append(cmd)
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "esptool", SimpleNamespace(main=always_fail))
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro, "download_firmware", lambda chip, cache_dir: firmware)
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit):
        micro.flash_firmware(_make_args(chip="c3"))

    assert len(attempts) == 3


def test_deploy_code_requires_config_local(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")

    with pytest.raises(SystemExit):
        micro.deploy_code(_make_args())


def test_deploy_code_uploads_files_to_device(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]:
            return SimpleNamespace(returncode=0, stdout="MP_OK", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    micro.deploy_code(_make_args())

    mkdir_calls = [cmd for cmd in calls if "mkdir" in cmd]
    cp_calls = [cmd for cmd in calls if "cp" in cmd]
    assert len(mkdir_calls) == 2
    assert len(cp_calls) == len(micro.MICRO_DEVICE_RELATIVE_FILES)
    assert any(cmd[-1] == ":src/" for cmd in cp_calls)
    assert any(cmd[-1] == ":src/mqtt/" for cmd in cp_calls)
    assert any(cmd[-2].endswith("console_output.py") for cmd in cp_calls)
    assert any(cmd[-2].endswith("branding.py") for cmd in cp_calls)
    assert any(cmd[-2].endswith("lightweight_detector.py") for cmd in cp_calls)
    assert any(cmd[-2].endswith("runtime_diagnostics.py") for cmd in cp_calls)


def test_deploy_code_rejects_invalid_healthcheck(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="bad boot")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        micro.deploy_code(_make_args())


def test_deploy_code_rejects_incomplete_source_tree(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)
    (src_dir / "device_utils.py").unlink()
    calls = []

    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(SystemExit):
        micro.deploy_code(_make_args())

    assert calls == []


def test_deploy_code_exits_on_copy_failure(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]:
            return SimpleNamespace(returncode=0, stdout="MP_OK", stderr="")
        if "cp" in cmd:
            raise subprocess.CalledProcessError(2, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        micro.deploy_code(_make_args())


def test_run_application_starts_mpremote_process(monkeypatch) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        def wait(self):
            return 0

    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "Popen", lambda cmd: started.append(cmd) or FakeProcess())

    micro.run_application(_make_args())

    assert started == [[
        "mpremote",
        "connect",
        "/dev/cu.usbmodem1",
        "exec",
        "from src.main import main; main()",
    ]]


def test_run_application_handles_keyboard_interrupt_and_resets_device(monkeypatch) -> None:
    events: list[str] = []

    class FakeProcess:
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired(cmd="mpremote", timeout=timeout)

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "Popen", lambda _cmd: FakeProcess())
    monkeypatch.setattr(micro, "_reset_device", lambda port: events.append(f"reset:{port}"))

    micro.run_application(_make_args())

    assert events == ["terminate", "kill", "reset:/dev/cu.usbmodem1"]


def test_run_application_exits_on_subprocess_error(monkeypatch) -> None:
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(
        micro.subprocess,
        "Popen",
        lambda _cmd: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["mpremote"])),
    )

    with pytest.raises(SystemExit):
        micro.run_application(_make_args())


def test_verify_installation_passes_when_all_checks_succeed(monkeypatch) -> None:
    results = [
        SimpleNamespace(stdout="csi_start,csi_stop\n", stderr=""),
        SimpleNamespace(stdout="(1, 24, 0)\n", stderr=""),
        SimpleNamespace(
                stdout="['__init__.py', 'branding.py', 'config.py', 'config_local.py', 'device_utils.py', 'utils.py', 'threshold.py', 'filters.py', "
            "'csi_features.py', 'segmentation.py', 'detector_interface.py', 'runtime_policy.py', 'runtime_diagnostics.py', 'lightweight_detector.py', "
            "'high_accuracy_detector.py', 'ml_weights.py', 'traffic_generator.py', 'console_output.py', 'main.py', 'mqtt']\n",
            stderr="",
        ),
        SimpleNamespace(stdout="['__init__.py', 'handler.py', 'commands.py', 'home_assistant.py']\n", stderr=""),
        SimpleNamespace(stdout="True\n", stderr=""),
    ]

    def fake_run(cmd, capture_output, text, check):
        return results.pop(0)

    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    micro.verify_installation(_make_verify_args())


def test_verify_installation_raises_when_required_checks_fail(monkeypatch) -> None:
    calls = [
        SimpleNamespace(stdout="NONE\n", stderr=""),
        subprocess.CalledProcessError(1, ["mpremote"], stderr="version error"),
        subprocess.CalledProcessError(1, ["mpremote"], stderr="missing src"),
        subprocess.CalledProcessError(1, ["mpremote"], stderr="missing mqtt"),
        subprocess.CalledProcessError(1, ["mpremote"], stderr="config missing"),
    ]

    def fake_run(cmd, capture_output, text, check):
        result = calls.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        micro.verify_installation(_make_verify_args())
