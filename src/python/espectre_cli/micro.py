# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Micro

MicroPython device workflow commands.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import ast
import ipaddress
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

from .common import (
    Fore,
    MICRO_CHIP_CHOICES,
    PYTHON_SRC_DIR,
    Style,
    detect_chip_type,
    get_serial_port,
    prompt_chip_type,
    remember_serial_port_identity,
    resolve_serial_port,
    serial_console_mode,
    cli_command,
    copy_config_command,
    print_box_banner,
)
from .build_artifacts import build_artifact_metadata, print_build_artifact_metadata
from .device_discovery import ESPECTRE_DIRECT_PORT
from .device_transport import direct_endpoint_from_device_url
from .idf import idf_flash_baud, run_idf_flash_lifecycle


MICRO_DEVICE_RELATIVE_FILES = [
    "__init__.py",
    "branding.py",
    "config.py",
    "config_local.py",
    "device_utils.py",
    "serial_sequence.py",
    "threshold.py",
    "detector_interface.py",
    "runtime_motion_policy.py",
    "runtime_diagnostics.py",
    "runtime_main.py",
    "temporal_csi_sampler.py",
    "wifi_bootstrap.py",
    "lightweight_detector.py",
    "traffic_generator.py",
    "console_output.py",
    "protocol.py",
    "direct_api.py",
    "main.py",
]

MICRO_WIFI_CONNECTED_PATTERN = re.compile(
    r"WiFi connected - IP:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\b"
)
MPY_CROSS_COMMAND = "mpy-cross-v6.3"
MPY_OPTIMIZATION_LEVEL = "-O3"
MICROPYTHON_READY_TIMEOUT_SECONDS = 15.0
MICROPYTHON_HEALTHCHECK_TIMEOUT_SECONDS = 5.0
MICROPYTHON_READY_RETRY_SECONDS = 1.0


def _resolve_config_local_path(config_path: str | Path | None = None) -> Path:
    """Return the selected runtime override, defaulting to config_local.py."""
    return Path(config_path) if config_path is not None else PYTHON_SRC_DIR / "config_local.py"


def deployment_files(config_local_path: Path) -> List[Tuple[str, str]]:
    """Return the MicroPython source manifest compiled by `micro deploy`."""
    files: List[Tuple[str, str]] = []
    for rel_path in MICRO_DEVICE_RELATIVE_FILES:
        if rel_path == "config_local.py":
            files.append((str(config_local_path), ":src/config_local.py"))
            continue
        src_path = PYTHON_SRC_DIR / rel_path
        files.append((str(src_path), ":src/"))
    return files


def compile_deployment_files(
    config_local_path: Path,
    output_dir: Path,
) -> List[Tuple[str, str]]:
    """Compile the complete device manifest to optimized portable bytecode."""
    compiled: List[Tuple[str, str]] = []
    source_files = deployment_files(config_local_path)
    for rel_path, (source, _source_destination) in zip(
        MICRO_DEVICE_RELATIVE_FILES,
        source_files,
        strict=True,
    ):
        output_relative = Path(rel_path).with_suffix(".mpy")
        output_path = output_dir / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        device_source = (Path("src") / rel_path).as_posix()
        command = [
            MPY_CROSS_COMMAND,
            MPY_OPTIMIZATION_LEVEL,
            "-s",
            device_source,
            "-o",
            str(output_path),
            source,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(
                f"Failed to compile {rel_path}: {detail}"
            ) from exc
        compiled.append(
            (str(output_path), f":src/{output_relative.as_posix()}")
        )
    return compiled


def build_project_firmware_image(
    *,
    chip: str = "esp32",
    clean: bool = False,
    backend: str = "auto",
    pull_policy: str = "ask",
) -> Path:
    """Build the pinned lean firmware for a supported Micro-ESPectre chip."""
    from .micro_firmware import build_project_firmware

    return build_project_firmware(
        PYTHON_SRC_DIR,
        chip=chip,
        clean=clean,
        backend=backend,
        pull_policy=pull_policy,
    )


def build_project_firmware_command(args) -> None:
    """Build a project MicroPython image for a supported chip."""
    chip = getattr(args, "chip", "esp32")
    _require_supported_micro_chip(chip)
    print_box_banner("Building Micro-ESPectre Firmware")
    try:
        firmware_path = build_project_firmware_image(
            chip=chip,
            clean=bool(getattr(args, "clean", False)),
            backend=getattr(args, "backend", "auto"),
            pull_policy=getattr(args, "pull", "ask"),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"{Fore.RED}❌ Project firmware build failed: {exc}{Style.RESET_ALL}")
        raise SystemExit(1) from exc
    print()
    print(f"{Fore.GREEN}✅ Project firmware built successfully{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Firmware: {firmware_path}{Style.RESET_ALL}")
    if bool(getattr(args, "json", False)):
        print_build_artifact_metadata(
            frontend="micro",
            chip=chip,
            artifact=firmware_path,
        )


def _wait_for_micropython(port: str) -> tuple[bool, str]:
    """Wait briefly for the MicroPython REPL after a flash or hard reset."""
    deadline = time.monotonic() + MICROPYTHON_READY_TIMEOUT_SECONDS
    last_detail = ""
    while True:
        remaining_seconds = max(0.1, deadline - time.monotonic())
        try:
            health = subprocess.run(
                ["mpremote", "connect", port, "exec", 'print("MP_OK")'],
                capture_output=True,
                text=True,
                timeout=min(MICROPYTHON_HEALTHCHECK_TIMEOUT_SECONDS, remaining_seconds),
            )
            if health.returncode == 0 and "MP_OK" in (health.stdout or ""):
                return True, ""
            last_detail = "\n".join(
                part.strip()
                for part in (health.stdout or "", health.stderr or "")
                if part.strip()
            )
        except subprocess.TimeoutExpired:
            last_detail = "MicroPython readiness probe timed out"
        except OSError as exc:
            last_detail = str(exc)

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return False, last_detail
        time.sleep(min(MICROPYTHON_READY_RETRY_SECONDS, remaining_seconds))


def _require_mpremote() -> None:
    try:
        subprocess.run(["mpremote", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Fore.RED}❌ mpremote not found. Install it with:{Style.RESET_ALL}")
        print("   pip install mpremote")
        raise SystemExit(1)


def _require_mpy_cross() -> None:
    try:
        subprocess.run(
            [MPY_CROSS_COMMAND, "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Fore.RED}❌ {MPY_CROSS_COMMAND} not found. Install dependencies with:{Style.RESET_ALL}")
        print("   pip install -r requirements.txt")
        raise SystemExit(1)


def _subprocess_error_detail(exc: BaseException) -> str:
    """Return readable captured output from a subprocess exception."""
    parts = []
    for attribute in ("stderr", "stdout"):
        value = getattr(exc, attribute, None)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            text = "\n".join(
                str(value).replace("\x00", "").splitlines()
            ).strip()
            if text and text not in parts:
                parts.append(text)
    return "\n".join(parts) or str(exc).strip()


def _reset_device(port: str) -> bool:
    """Reset one MicroPython device and report whether the command succeeded."""
    time.sleep(0.5)
    try:
        subprocess.run(
            ["mpremote", "connect", port, "exec", "import machine; machine.reset()"],
            timeout=5,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"{Fore.GREEN}ESP32 reset completed{Style.RESET_ALL}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # A hard reset can leave mpremote attached to the newly booted REPL
        # until its command times out. Treat a reachable REPL as success.
        ready, readiness_detail = _wait_for_micropython(port)
        if ready:
            print(f"{Fore.GREEN}ESP32 reset completed{Style.RESET_ALL}")
            return True
        detail = readiness_detail or _subprocess_error_detail(exc)
        print(f"{Fore.RED}❌ ESP32 reset failed{Style.RESET_ALL}")
        if detail:
            print(detail)
        return False
    except OSError as exc:
        print(f"{Fore.RED}❌ ESP32 reset failed{Style.RESET_ALL}")
        print(str(exc).strip())
        return False


def _require_supported_micro_chip(chip: str | None) -> None:
    """Reject chips outside the maintained Micro-ESPectre matrix."""
    if chip is not None and chip not in MICRO_CHIP_CHOICES:
        print(f"{Fore.RED}❌ Unsupported Micro-ESPectre chip: {chip}{Style.RESET_ALL}")
        raise SystemExit(1)


def flash_firmware(args) -> None:
    """Flash MicroPython firmware to ESP32 using esptool."""
    try:
        import esptool
    except ImportError:
        print(f"{Fore.RED}❌ esptool not found. Install it with:{Style.RESET_ALL}")
        print("   pip install esptool")
        raise SystemExit(1)

    chip = args.chip
    _require_supported_micro_chip(chip)
    discovery_port = None
    if not chip or args.port:
        discovery_port = get_serial_port(
            args.port,
            chip=chip,
            frontend="micro",
            purpose="flash",
        )
        remember_serial_port_identity(discovery_port)
    if not chip:
        assert discovery_port is not None
        chip = detect_chip_type(discovery_port)
        if not chip:
            print(f"\n{Fore.YELLOW}💡 Tip: If the chip is not responding, try:{Style.RESET_ALL}")
            print("   1. Hold the BOOT button on your ESP32")
            print("   2. Press and release the RESET button (while holding BOOT)")
            print("   3. Release the BOOT button")
            print("   4. Try flashing again")
            print()
            chip = prompt_chip_type(MICRO_CHIP_CHOICES)
            if not chip:
                raise SystemExit(1)
    _require_supported_micro_chip(chip)

    if args.firmware:
        firmware_path = Path(args.firmware)
        if not firmware_path.exists():
            print(f"{Fore.RED}❌ Firmware not found: {firmware_path}{Style.RESET_ALL}")
            raise SystemExit(1)
    else:
        try:
            firmware_path = build_project_firmware_image(
                chip=chip,
                clean=bool(getattr(args, "clean", False)),
                backend=getattr(args, "backend", "auto"),
                pull_policy=getattr(args, "pull", "ask"),
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"{Fore.RED}❌ Project firmware build failed: {exc}{Style.RESET_ALL}")
            raise SystemExit(1) from exc

    port = resolve_serial_port(
        discovery_port,
        chip=chip,
        frontend="micro",
        purpose="flash",
    )
    if serial_console_mode(chip, port) == "usb_cdc":
        port = resolve_serial_port(
            port,
            chip=chip,
            frontend="micro",
            purpose="flash",
            require_firmware_download=True,
        )

    print_box_banner("Flashing MicroPython Firmware")
    print()
    print(f"{Fore.CYAN}Chip:     {chip.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Port:     {port}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Firmware: {firmware_path.name}{Style.RESET_ALL}")
    print()

    chip_name_map = {
        "esp32": "esp32",
        "c3": "esp32c3",
        "s2": "esp32s2",
        "s3": "esp32s3",
        "c5": "esp32c5",
        "c6": "esp32c6",
    }
    chip_name = chip_name_map.get(chip, "esp32")
    flash_baud = idf_flash_baud(chip_name)
    before = "no-reset" if serial_console_mode(chip, port) == "usb_cdc" else "default-reset"

    try:
        print(f"{Fore.YELLOW}Flashing firmware...{Style.RESET_ALL}")
        flash_offset_map = {
            "esp32": "0x1000",
            "c3": "0x0",
            "s2": "0x1000",
            "s3": "0x0",
            "c5": "0x2000",
            "c6": "0x0",
        }
        flash_offset = flash_offset_map.get(chip, "0x0")
        flash_command = [
            "--chip",
            chip_name,
            "--port",
            port,
            "--baud",
            flash_baud,
            "--before",
            "no-reset" if args.erase else before,
            "--after",
            "no-reset",
            "write-flash",
            "--flash-mode",
            "dio",
            "--flash-freq",
            "40m",
            "--flash-size",
            "detect",
            flash_offset,
            str(firmware_path),
        ]
        run_idf_flash_lifecycle(
            flash_command,
            port,
            erase=bool(args.erase),
            before=before,
        )
        print()
        print(f"{Fore.GREEN}✅ Firmware flashed successfully!{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}Next steps:{Style.RESET_ALL}")
        print(f"  1. {copy_config_command()}")
        print("  2. Edit src/python/micro_espectre/config_local.py with your credentials")
        print(f"  3. {Fore.GREEN}{cli_command('micro', 'deploy')}{Style.RESET_ALL}")
        print(f"  4. {Fore.GREEN}{cli_command('micro', 'run')}{Style.RESET_ALL}")
        print()
        if bool(getattr(args, "json", False)):
            metadata = build_artifact_metadata(
                frontend="micro",
                chip=chip,
                artifact=firmware_path,
            )
            metadata.update({"command": "flash", "port": port})
            print(json.dumps(metadata, sort_keys=True))
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error flashing firmware: {e}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Troubleshooting tips:{Style.RESET_ALL}")
        print("  1. Try holding the BOOT button while connecting")
        print("  2. Use a different USB cable (data cable, not charge-only)")
        print("  3. Try a different USB port (preferably USB 2.0)")
        print("  4. Ensure no other programs are using the serial port")
        print(f"  5. Try with --erase flag: {Fore.GREEN}{cli_command('micro', 'flash', '--erase')}{Style.RESET_ALL}")
        print()
        raise SystemExit(1)


def deploy_code(args) -> None:
    """Compile and deploy optimized MicroPython bytecode using mpremote."""
    _require_supported_micro_chip(getattr(args, "chip", None))
    _require_mpremote()
    port = get_serial_port(
        args.port,
        chip=getattr(args, "chip", None),
        frontend="micro",
        purpose="deploy",
    )

    config_local_path = _resolve_config_local_path(getattr(args, "config", None))
    if not config_local_path.exists():
        print(f"{Fore.RED}❌ MicroPython config not found: {config_local_path}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Create it from the template:{Style.RESET_ALL}")
        print(f"  {copy_config_command()}")
        print("  # Then edit src/python/micro_espectre/config_local.py with your credentials")
        print()
        raise SystemExit(1)

    files_to_upload = deployment_files(config_local_path)
    missing_files = [src for src, _dst in files_to_upload if not Path(src).exists()]
    if missing_files:
        print(f"{Fore.RED}Cannot deploy: required source files are missing:{Style.RESET_ALL}")
        for missing in missing_files:
            print(f"  {missing}")
        raise SystemExit(1)
    _require_mpy_cross()

    print_box_banner("Deploying Code to Device")
    print()
    print(f"{Fore.CYAN}Port: {port}{Style.RESET_ALL}")
    print()

    try:
        with tempfile.TemporaryDirectory(prefix="espectre-mpy-") as build_dir:
            ready, health_detail = _wait_for_micropython(port)
            if not ready:
                print(f"{Fore.RED}❌ Device is not running a valid MicroPython firmware{Style.RESET_ALL}")
                if health_detail:
                    print(f"{Fore.YELLOW}   Probe output: {health_detail}{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Recommended fix:{Style.RESET_ALL}")
                print(f"  {Fore.GREEN}{cli_command('micro', 'flash', '--erase')}{Style.RESET_ALL}")
                print(f"  {Fore.GREEN}{cli_command('micro', 'deploy')}{Style.RESET_ALL}")
                print()
                raise SystemExit(1)

            print(f"{Fore.YELLOW}⚙️  Compiling optimized bytecode ({MPY_OPTIMIZATION_LEVEL})...{Style.RESET_ALL}")
            compiled_files = compile_deployment_files(
                config_local_path,
                Path(build_dir),
            )

            stage_dir = ":src.stage"
            prepare_script = (
                "import os\n"
                "def exists(path):\n"
                "    try:\n"
                "        os.stat(path)\n"
                "        return True\n"
                "    except OSError:\n"
                "        return False\n"
                "def remove_tree(path):\n"
                "    try:\n"
                "        names = os.listdir(path)\n"
                "    except OSError:\n"
                "        return\n"
                "    for name in names:\n"
                "        child = path + '/' + name\n"
                "        try:\n"
                "            os.remove(child)\n"
                "        except OSError:\n"
                "            remove_tree(child)\n"
                "    os.rmdir(path)\n"
                "remove_tree('/src.stage')\n"
                "if exists('/src.previous'):\n"
                "    if exists('/src'):\n"
                "        remove_tree('/src.previous')\n"
                "    else:\n"
                "        os.rename('/src.previous', '/src')"
            )
            subprocess.run(
                ["mpremote", "connect", port, "exec", prepare_script],
                check=True,
                capture_output=True,
            )
            print(f"{Fore.YELLOW}📁 Creating staging directory...{Style.RESET_ALL}")
            subprocess.run(
                ["mpremote", "connect", port, "mkdir", stage_dir],
                check=True,
                capture_output=True,
            )

            print(f"{Fore.YELLOW}📤 Uploading optimized bytecode...{Style.RESET_ALL}")
            for src, dst in compiled_files:
                staged_dst = dst.replace(":src/", f"{stage_dir}/", 1)
                print(f"  {Path(src).name} → {staged_dst}")
                subprocess.run(
                    ["mpremote", "connect", port, "cp", src, staged_dst],
                    check=True,
                    capture_output=True,
                )

            print(f"{Fore.YELLOW}🔄 Activating staged bytecode...{Style.RESET_ALL}")
            expected_names = tuple(
                Path(rel_path).with_suffix(".mpy").name
                for rel_path in MICRO_DEVICE_RELATIVE_FILES
            )
            activate_script = (
                "import os\n"
                "def exists(path):\n"
                "    try:\n"
                "        os.stat(path)\n"
                "        return True\n"
                "    except OSError:\n"
                "        return False\n"
                "def remove_tree(path):\n"
                "    if not exists(path):\n"
                "        return\n"
                "    for name in os.listdir(path):\n"
                "        child = path + '/' + name\n"
                "        try:\n"
                "            os.remove(child)\n"
                "        except OSError:\n"
                "            remove_tree(child)\n"
                "    os.rmdir(path)\n"
                f"expected = {expected_names!r}\n"
                "present = os.listdir('/src.stage')\n"
                "missing = [name for name in expected if name not in present]\n"
                "if missing:\n"
                "    raise OSError('staged deployment is incomplete: ' + ','.join(missing))\n"
                "had_src = exists('/src')\n"
                "if had_src:\n"
                "    os.rename('/src', '/src.previous')\n"
                "try:\n"
                "    os.rename('/src.stage', '/src')\n"
                "except BaseException:\n"
                "    if had_src and exists('/src.previous'):\n"
                "        os.rename('/src.previous', '/src')\n"
                "    raise\n"
                "remove_tree('/src.previous')\n"
                "for path in ('/config_local.py', '/config_local.mpy'):\n"
                "    try:\n"
                "        os.remove(path)\n"
                "    except OSError:\n"
                "        pass"
            )
            subprocess.run(
                ["mpremote", "connect", port, "exec", activate_script],
                check=True,
                capture_output=True,
            )

        print()
        print(f"{Fore.GREEN}✅ Deployment complete!{Style.RESET_ALL}")
        print()
        print(f"{Fore.CYAN}To run the application:{Style.RESET_ALL}")
        print(f"  {cli_command('micro', 'run')}")
        print()
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}❌ Error during deployment: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)


def run_application(args) -> None:
    """Run the MicroPython application on ESP32."""
    _require_supported_micro_chip(getattr(args, "chip", None))
    _require_mpremote()
    port = get_serial_port(
        args.port,
        chip=getattr(args, "chip", None),
        frontend="micro",
        purpose="run",
    )

    print_box_banner("Running MicroPython Application")
    print()
    print(f"{Fore.YELLOW}🚀 Starting application...{Style.RESET_ALL}")
    print()

    process = None
    try:
        command = [
            "mpremote",
            "connect",
            port,
            "exec",
            "from src.main import main; main()",
        ]
        json_output = bool(getattr(args, "json", False))
        process = subprocess.Popen(
            command,
            **(
                {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                }
                if json_output
                else {}
            ),
        )
        if json_output:
            assert process.stdout is not None
            endpoint_emitted = False
            for line in process.stdout:
                print(line, end="", flush=True)
                if endpoint_emitted:
                    continue
                match = MICRO_WIFI_CONNECTED_PATTERN.search(line)
                if match is None:
                    continue
                address = str(ipaddress.IPv4Address(match.group("ip")))
                print(
                    json.dumps(
                        {
                            "chip": getattr(args, "chip", None),
                            "endpoint": direct_endpoint_from_device_url(
                                f"http://{address}:{ESPECTRE_DIRECT_PORT}"
                            ),
                            "event": "direct_ready",
                            "frontend": "micro",
                            "port": port,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                endpoint_emitted = True
        returncode = process.wait()
        if returncode != 0:
            raise SystemExit(returncode)
    except subprocess.CalledProcessError as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Application stopped - cleaning up ESP32...{Style.RESET_ALL}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        if not _reset_device(port):
            raise SystemExit(1)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        raise SystemExit(1)


def verify_installation(args) -> None:
    """Verify MicroPython firmware and deployed code."""
    _require_supported_micro_chip(getattr(args, "chip", None))
    port = get_serial_port(
        args.port,
        chip=getattr(args, "chip", None),
        frontend="micro",
        purpose="verify",
    )
    print_box_banner("Verifying Installation")
    print()

    all_ok = True

    print(f"{Fore.YELLOW}🔍 Checking CSI firmware support...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            [
                "mpremote",
                "connect",
                port,
                "exec",
                "import network; wlan = network.WLAN(network.STA_IF); "
                "csi_methods = [m for m in dir(wlan) if m.startswith('csi_')]; "
                "print(','.join(csi_methods) if csi_methods else 'NONE')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        csi_methods = result.stdout.strip()
        if csi_methods and csi_methods != "NONE":
            print(f"{Fore.GREEN}✅ CSI methods available: {csi_methods}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ CSI methods not found in firmware{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Hint: Flash the CSI-enabled firmware:{Style.RESET_ALL}")
            print(f"   {cli_command('micro', 'flash', '--erase')}")
            all_ok = False
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ Failed to check CSI support: {e.stderr.strip()}{Style.RESET_ALL}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking ESPectre core module...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            [
                "mpremote",
                "connect",
                port,
                "exec",
                "import espectre_native_features as core; "
                "print(core.BACKEND, hasattr(core, 'Detector'), "
                "hasattr(core, 'TemporalCsiSampler'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip() != "espectre_core True True":
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr="incompatible espectre core module",
            )
        print(f"{Fore.GREEN}✅ ESPectre core detector and sampler available{Style.RESET_ALL}")
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()
        print(f"{Fore.RED}❌ ESPectre core module unavailable or incompatible{Style.RESET_ALL}")
        if detail:
            print(f"{Fore.YELLOW}   {detail}{Style.RESET_ALL}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking MicroPython version...{Style.RESET_ALL}")
    try:
        result = subprocess.run(
            ["mpremote", "connect", port, "exec", "import sys; print(sys.implementation.version)"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"{Fore.GREEN}✅ MicroPython version: {result.stdout.strip()}{Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Failed to get MicroPython version{Style.RESET_ALL}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking application modules...{Style.RESET_ALL}")
    try:
        src_result = subprocess.run(
            ["mpremote", "connect", port, "exec", 'import os; print(os.listdir("/src"))'],
            capture_output=True,
            text=True,
            check=True,
        )
        expected_src = {
            Path(rel).with_suffix(".mpy").name
            for rel in MICRO_DEVICE_RELATIVE_FILES
            if "/" not in rel
        }
        src_present = set(ast.literal_eval(src_result.stdout.strip()))
        missing_src = sorted(expected_src - src_present)
        if missing_src:
            print(f"{Fore.RED}❌ Missing deployed bytecode detected{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Missing in /src: {', '.join(missing_src)}{Style.RESET_ALL}")
            all_ok = False
        else:
            print(f"{Fore.GREEN}✅ Required bytecode found in /src{Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.RED}❌ Application modules not found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Hint: Deploy the code first:{Style.RESET_ALL}")
        print(f"   {cli_command('micro', 'deploy')}")
        all_ok = False
    print()

    print(f"{Fore.YELLOW}🔍 Checking configuration...{Style.RESET_ALL}")
    try:
        config_probe = 'import os; print("config_local.mpy" in os.listdir("/src"))'
        result = subprocess.run(
            ["mpremote", "connect", port, "exec", config_probe],
            capture_output=True,
            text=True,
            check=True,
        )
        if "True" in result.stdout:
            print(f"{Fore.GREEN}✅ config_local.mpy found{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️  config_local.mpy not found (will use defaults from config.py){Style.RESET_ALL}")
    except subprocess.CalledProcessError:
        print(f"{Fore.YELLOW}⚠️  Could not check config_local.mpy{Style.RESET_ALL}")
    print()

    if all_ok:
        print_box_banner("Installation Verified Successfully", color=Fore.GREEN)
        print()
        print(f"{Fore.CYAN}You can now run the application:{Style.RESET_ALL}")
        print(f"  {cli_command('micro', 'run')}")
        print()
        return

    print_box_banner("Installation Checks Failed", color=Fore.RED)
    print()
    print(f"{Fore.YELLOW}Please fix the issues above and try again.{Style.RESET_ALL}")
    print()
    raise SystemExit(1)
