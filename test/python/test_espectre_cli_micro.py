# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Micro Tests

Tests for espectre_cli.micro host-side helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import ast
import builtins
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from espectre_cli import micro
from espectre_cli import micro_firmware


def _make_args(**overrides) -> argparse.Namespace:
    args = {
        "port": None,
        "chip": "c3",
        "erase": False,
        "firmware": None,
        "clean": False,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _make_verify_args(**overrides) -> argparse.Namespace:
    args = {"port": None}
    args.update(overrides)
    return argparse.Namespace(**args)


def _create_micro_src_tree(base_dir: Path) -> None:
    for rel_path in micro.MICRO_DEVICE_RELATIVE_FILES:
        target = base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# test\n", encoding="utf-8")


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


def test_deploy_manifest_contains_local_runtime_imports() -> None:
    deployed = set(micro.MICRO_DEVICE_RELATIVE_FILES)
    missing: set[str] = set()
    for rel_path in deployed:
        if rel_path == "config_local.py":
            continue
        source_path = micro.PYTHON_SRC_DIR / rel_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=rel_path)
        parent = Path(rel_path).parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "src":
                required = (f"{alias.name.replace('.', '/')}.py" for alias in node.names)
            elif node.module and node.module.startswith("src."):
                required = (f"{node.module[4:].replace('.', '/')}.py",)
            elif node.level == 1 and node.module:
                required = (str(parent / f"{node.module.replace('.', '/')}.py"),)
            else:
                continue
            missing.update(required_path for required_path in required if required_path not in deployed)

    assert missing == set()


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


def test_require_mpy_cross_accepts_installed_binary(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    micro._require_mpy_cross()

    assert calls == [["mpy-cross-v6.3", "--version"]]


def test_require_mpy_cross_exits_when_binary_missing(monkeypatch) -> None:
    def fake_run(cmd, capture_output, check):
        raise FileNotFoundError()

    monkeypatch.setattr(micro.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        micro._require_mpy_cross()


def test_reset_device_reports_command_result(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)

    def fake_run(cmd, timeout, capture_output, text, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(micro.subprocess, "run", fake_run)
    assert micro._reset_device("/dev/cu.usbmodem1") is True

    def fake_run_fail(cmd, timeout, capture_output, text, check):
        raise subprocess.CalledProcessError(1, cmd, stderr="busy")

    monkeypatch.setattr(micro.subprocess, "run", fake_run_fail)
    assert micro._reset_device("/dev/cu.usbmodem1") is False

    assert calls == [["mpremote", "connect", "/dev/cu.usbmodem1", "exec", "import machine; machine.reset()"]]


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
    monkeypatch.setattr(
        micro,
        "build_project_firmware_image",
        lambda *, chip, clean, backend, pull_policy: firmware,
    )
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


@pytest.mark.parametrize(
    ("chip", "esptool_chip", "offset"),
    (
        ("esp32", "esp32", "0x1000"),
        ("c3", "esp32c3", "0x0"),
        ("s2", "esp32s2", "0x1000"),
        ("c5", "esp32c5", "0x2000"),
        ("c6", "esp32c6", "0x0"),
        ("s3", "esp32s3", "0x0"),
    ),
)
def test_flash_firmware_uses_project_build_for_supported_chips(
    chip: str, esptool_chip: str, offset: str, tmp_path: Path, monkeypatch
) -> None:
    firmware = tmp_path / "project.bin"
    firmware.write_bytes(b"project")
    calls: list[list[str]] = []

    monkeypatch.setitem(
        sys.modules,
        "esptool",
        SimpleNamespace(main=lambda command: calls.append(command)),
    )
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(
        micro,
        "build_project_firmware_image",
        lambda *, chip, clean, backend, pull_policy: firmware,
    )

    micro.flash_firmware(_make_args(chip=chip))

    assert calls[-1][:2] == ["--chip", esptool_chip]
    assert calls[-1][-2:] == [offset, str(firmware)]


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
    monkeypatch.setattr(
        micro,
        "build_project_firmware_image",
        lambda *, chip, clean, backend, pull_policy: firmware,
    )
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit):
        micro.flash_firmware(_make_args(chip="c6"))

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
    exec_scripts = [cmd[-1] for cmd in calls if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]]
    assert len(mkdir_calls) == 1
    assert mkdir_calls[0][-1] == ":src.stage"
    assert len(cp_calls) == len(micro.MICRO_DEVICE_RELATIVE_FILES)
    assert any(cmd[-1] == ":src.stage/main.mpy" for cmd in cp_calls)
    assert not any(cmd[-1].startswith(":src.stage/mqtt/") for cmd in cp_calls)
    assert any(cmd[-2].endswith("console_output.mpy") for cmd in cp_calls)
    assert any(cmd[-2].endswith("branding.mpy") for cmd in cp_calls)
    assert any(cmd[-2].endswith("lightweight_detector.mpy") for cmd in cp_calls)
    assert any(cmd[-2].endswith("runtime_diagnostics.mpy") for cmd in cp_calls)
    assert any(cmd[-2].endswith("protocol.mpy") for cmd in cp_calls)
    assert all(cmd[-2].endswith(".mpy") for cmd in cp_calls)
    assert all(cmd[-1].startswith(":src.stage/") for cmd in cp_calls)
    assert any("remove_tree('/src.stage')" in script for script in exec_scripts)
    assert any("os.rename('/src.stage', '/src')" in script for script in exec_scripts)
    assert any("os.rename('/src.previous', '/src')" in script for script in exec_scripts)

    compile_calls = [
        cmd
        for cmd in calls
        if cmd and cmd[0] == micro.MPY_CROSS_COMMAND and micro.MPY_OPTIMIZATION_LEVEL in cmd
    ]
    assert len(compile_calls) == len(micro.MICRO_DEVICE_RELATIVE_FILES)
    assert all(micro.MPY_OPTIMIZATION_LEVEL in cmd for cmd in compile_calls)


def test_project_manifest_does_not_freeze_application(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.py"

    micro_firmware._write_manifest(manifest)

    assert manifest.read_text(encoding="utf-8") == (
        'freeze("$(PORT_DIR)/modules", ("_boot.py", "flashbdev.py", "inisetup.py"))\n'
    )


def test_project_firmware_rejects_unsupported_chip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported project firmware chip: h2"):
        micro_firmware.build_project_firmware(tmp_path, chip="h2", cache_dir=tmp_path)


def test_project_firmware_aligns_idf_55_lockfile(tmp_path: Path) -> None:
    micropython_dir = tmp_path / "micropython"
    lockfile = (
        micropython_dir
        / "ports"
        / "esp32"
        / "lockfiles"
        / "dependencies.lock.esp32s3"
    )
    lockfile.parent.mkdir(parents=True)
    lockfile.write_text(
        "dependencies:\n  idf:\n    source:\n      type: idf\n    version: 5.5.2\n",
        encoding="utf-8",
    )
    micro_firmware._align_idf_lockfile(micropython_dir, "s3", "5.5.5")

    assert "    version: 5.5.5\n" in lockfile.read_text(encoding="utf-8")


def test_project_firmware_disables_legacy_csi_capture(tmp_path: Path) -> None:
    source_path = tmp_path / "ports" / "esp32" / "network_wlan_csi.c"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "wifi_csi_config_t config = {\n    .acquire_csi_legacy = 1,\n};\n",
        encoding="utf-8",
    )

    micro_firmware._configure_project_csi_capture(tmp_path)
    micro_firmware._configure_project_csi_capture(tmp_path)

    source = source_path.read_text(encoding="utf-8")
    assert ".acquire_csi_legacy = 0," in source
    assert ".acquire_csi_legacy = 1," not in source


def test_project_firmware_exposes_dual_band_mode_configuration(tmp_path: Path) -> None:
    source_path = tmp_path / "ports" / "esp32" / "network_wlan.c"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """                    case MP_QSTR_bandwidth: {
                        esp_exceptions(esp_wifi_set_bandwidth(self->if_id, mp_obj_get_int(kwargs->table[i].value)));
                        break;
                    }
    { MP_ROM_QSTR(MP_QSTR_BANDWIDTH_20), MP_ROM_INT(WIFI_BW20) },
""",
        encoding="utf-8",
    )

    micro_firmware._configure_project_wifi_band_mode(tmp_path)
    micro_firmware._configure_project_wifi_band_mode(tmp_path)

    source = source_path.read_text(encoding="utf-8")
    assert source.count("case MP_QSTR_band_mode:") == 1
    assert source.count("esp_wifi_set_band_mode") == 1
    assert source.count("MP_QSTR_BAND_MODE_2G_ONLY") == 1


def test_project_boards_use_one_shared_profile_and_only_esp32_override() -> None:
    boards_dir = micro.PYTHON_SRC_DIR / "firmware" / "boards"

    for board in micro_firmware.PROJECT_FIRMWARE_BOARDS.values():
        board_cmake = (boards_dir / board / "mpconfigboard.cmake").read_text(
            encoding="utf-8"
        )
        assert "../micro_espectre.cmake" in board_cmake

    overrides = sorted(
        path.relative_to(boards_dir).as_posix()
        for path in boards_dir.rglob("sdkconfig.override")
    )
    assert overrides == ["ESP32_MICRO_ESPECTRE/sdkconfig.override"]

    common_header = (boards_dir / "mpconfigboard_common.h").read_text(encoding="utf-8")
    assert "MICROPY_HW_ENABLE_MDNS_RESPONDER (1)" in common_header

    native_cmake = (
        micro.PYTHON_SRC_DIR / "firmware" / "native_components" / "micropython.cmake"
    ).read_text(encoding="utf-8")
    assert "native_direct.c" in native_cmake
    assert "native_traffic.c" in native_cmake
    assert "native_mqtt.c" not in native_cmake
    assert "idf::mqtt" not in native_cmake


def test_device_manifest_is_lightweight_direct_only() -> None:
    deployed = set(micro.MICRO_DEVICE_RELATIVE_FILES)

    assert {"lightweight_detector.py", "direct_api.py", "protocol.py"} <= deployed
    assert "high_accuracy_detector.py" not in deployed
    assert "ml_feature_trackers.py" not in deployed
    assert "ml_weights.py" not in deployed
    assert not any(path.startswith("mqtt/") for path in deployed)


def test_deploy_code_uses_selected_config_as_device_override(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)
    benchmark_config = tmp_path / "benchmark_config.py"
    benchmark_config.write_text("CSI_TARGET_PPS = 80\n", encoding="utf-8")
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

    micro.deploy_code(_make_args(config=benchmark_config))

    config_compile = next(
        cmd
        for cmd in calls
        if cmd and cmd[0] == micro.MPY_CROSS_COMMAND and cmd[-1] == str(benchmark_config)
    )
    assert config_compile[3] == "src/config_local.py"
    assert any(cmd[-1] == ":src.stage/config_local.mpy" for cmd in calls if "cp" in cmd)


def test_deploy_code_retries_healthcheck_while_micropython_starts(monkeypatch, tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    _create_micro_src_tree(src_dir)
    health_attempts = 0

    def fake_run(cmd, **kwargs):
        nonlocal health_attempts
        if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]:
            if "MP_OK" in cmd[-1]:
                health_attempts += 1
                if health_attempts == 1:
                    return SimpleNamespace(returncode=1, stdout="", stderr="port is not ready")
                return SimpleNamespace(returncode=0, stdout="MP_OK", stderr="")
            return SimpleNamespace(returncode=0, stdout="NONE", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(micro, "PYTHON_SRC_DIR", src_dir)
    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "run", fake_run)
    monkeypatch.setattr(micro.time, "sleep", lambda _seconds: None)

    micro.deploy_code(_make_args())

    assert health_attempts == 2


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
    monkeypatch.setattr(micro, "MICROPYTHON_READY_TIMEOUT_SECONDS", 0.0)

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

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
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

    exec_scripts = [cmd[-1] for cmd in calls if cmd[:4] == ["mpremote", "connect", "/dev/cu.usbmodem1", "exec"]]
    assert not any("os.rename('/src.stage', '/src')" in script for script in exec_scripts)


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


def test_run_application_propagates_mpremote_failure(monkeypatch) -> None:
    class FakeProcess:
        def wait(self):
            return 2

    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "Popen", lambda _cmd: FakeProcess())

    with pytest.raises(SystemExit, match="2"):
        micro.run_application(_make_args())


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
    monkeypatch.setattr(micro, "_reset_device", lambda port: events.append(f"reset:{port}") or True)

    micro.run_application(_make_args())

    assert events == ["terminate", "kill", "reset:/dev/cu.usbmodem1"]


def test_run_application_exits_when_interrupt_reset_fails(monkeypatch) -> None:
    class FakeProcess:
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda _port: "/dev/cu.usbmodem1")
    monkeypatch.setattr(micro.subprocess, "Popen", lambda _cmd: FakeProcess())
    monkeypatch.setattr(micro, "_reset_device", lambda _port: False)

    with pytest.raises(SystemExit):
        micro.run_application(_make_args())


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
    src_listing = [
        Path(rel_path).with_suffix(".mpy").name
        for rel_path in micro.MICRO_DEVICE_RELATIVE_FILES
        if "/" not in rel_path
    ]
    results = [
        SimpleNamespace(stdout="csi_start,csi_stop\n", stderr=""),
        SimpleNamespace(stdout="(1, 24, 0)\n", stderr=""),
        SimpleNamespace(stdout=f"{src_listing!r}\n", stderr=""),
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
