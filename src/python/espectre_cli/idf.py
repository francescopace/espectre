# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI IDF

Thin wrappers around idf.py for ESPectre frontends.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from .common import Fore, REPO_ROOT, Style, cli_command, detect_chip_type, get_serial_port
from .idf_container import DockerBackendError, IDF_VERSION, ensure_docker_backend, run_idf_container
from .targets import IDF_FRONTENDS, resolve_idf_target


MATTER_QR_PATTERN = re.compile(r"MATTER_QR=(MT:[A-Z0-9.\-]+)")
MATTER_MANUAL_CODE_PATTERN = re.compile(r"MATTER_MANUAL_CODE=([0-9]+)")
IDF_TARGET_CONFIG_PATTERN = re.compile(r'^CONFIG_IDF_TARGET="([^"]+)"$', re.MULTILINE)
ESPHOME_IDF_STAMP_FILE = ".esphome.stamp.json"


@dataclass(frozen=True)
class ResolvedIdfEnvironment:
    """Describe how the CLI will launch idf.py on the current host."""

    mode: str
    source: str
    install_dir: Path | None = None
    export_script: Path | None = None
    export_kind: str | None = None
    idf_path_entry: str | None = None
    python_executable: Path | None = None
    process_env: dict[str, str] | None = None


@dataclass(frozen=True)
class ResolvedIdfBuildBackend:
    """Selected local or Docker backend for an ESP-IDF build."""

    mode: str
    idf_environment: ResolvedIdfEnvironment | None = None
    docker: str | None = None


def remove_idf_artifacts(app_path: Path, artifact_names: list[str]) -> None:
    """Remove the selected ESP-IDF artifacts relative to the app directory."""
    removed: list[str] = []
    seen: set[str] = set()

    for artifact_name in artifact_names:
        if artifact_name in seen:
            continue
        seen.add(artifact_name)
        artifact_path = app_path / artifact_name
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
            removed.append(artifact_name)
        elif artifact_path.exists():
            artifact_path.unlink()
            removed.append(artifact_name)

    if removed:
        print(f"{Fore.CYAN}Cleaned:   {', '.join(removed)}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Cleaned:   nothing to remove{Style.RESET_ALL}")


def clean_idf_build_artifacts(app_path: Path, build_dir_name: str | None = None) -> None:
    """Remove only the selected ESP-IDF build directory before rebuilding."""
    remove_idf_artifacts(app_path, [build_dir_name or "build"])


def clean_all_idf_build_artifacts(app_path: Path) -> None:
    """Remove all ESP-IDF build directories and shared frontend artifacts."""
    artifact_names = [
        entry.name
        for entry in app_path.iterdir()
        if entry.name == "build" or (entry.is_dir() and entry.name.startswith("build-"))
    ]
    if build_dir_name := os.environ.get("ESPECTRE_IDF_BUILD_DIR"):
        artifact_names.append(build_dir_name)
    artifact_names.extend(["sdkconfig", "sdkconfig.old", "dependencies.lock"])
    remove_idf_artifacts(app_path, artifact_names)


def resolve_sdkconfig_defaults(app_path: Path, idf_target: str | None = None) -> str:
    """Resolve SDKCONFIG defaults from the environment or local app defaults."""
    env_defaults = os.environ.get("SDKCONFIG_DEFAULTS")
    if env_defaults:
        return env_defaults

    sdkconfig_defaults = ["sdkconfig.defaults"]
    if idf_target:
        target_defaults = app_path / f"sdkconfig.defaults.{idf_target}"
        if target_defaults.exists():
            sdkconfig_defaults.append(target_defaults.name)
    if (app_path / "sdkconfig.wifi").exists():
        sdkconfig_defaults.append("sdkconfig.wifi")
    return ";".join(sdkconfig_defaults)


def build_idf_base_command(build_dir_name: str | None) -> list[str]:
    """Build the shared idf.py command prefix."""
    command = ["idf.py"]
    if build_dir_name:
        command.extend(["-B", build_dir_name])
    return command


def sdkconfig_matches_target(app_path: Path, idf_target: str, sdkconfig_path: Path | None = None) -> bool:
    """Return whether the generated sdkconfig already selects the requested target."""
    sdkconfig = sdkconfig_path or app_path / "sdkconfig"
    if not sdkconfig.is_file():
        return False
    try:
        content = sdkconfig.read_text(encoding="utf-8")
    except OSError:
        return False
    return f'CONFIG_IDF_TARGET="{idf_target}"' in content


def cached_sdkconfig_path(app_path: Path, build_dir_name: str | None) -> Path | None:
    """Return the sdkconfig path retained by one CMake build cache, if any."""
    cache_path = app_path / (build_dir_name or "build") / "CMakeCache.txt"
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("SDKCONFIG:") and "=" in line:
            value = line.split("=", 1)[1]
            if not value:
                return None
            path = Path(value)
            return path.resolve() if path.is_absolute() else (app_path / path).resolve()
    return None


def resolve_configured_idf_target(app_path: Path) -> str | None:
    """Read the configured ESP-IDF target from sdkconfig when present."""
    sdkconfig = app_path / "sdkconfig"
    if not sdkconfig.is_file():
        return None
    try:
        content = sdkconfig.read_text(encoding="utf-8")
    except OSError:
        return None
    match = IDF_TARGET_CONFIG_PATTERN.search(content)
    return match.group(1) if match else None


def resolve_idf_build_dir_name(
    app_path: Path,
    idf_target: str | None = None,
    *,
    container: bool = False,
    prefer_existing_default: bool = False,
) -> str | None:
    """Resolve the ESP-IDF build directory, honoring overrides and per-target defaults."""
    if build_dir_name := os.environ.get("ESPECTRE_IDF_BUILD_DIR"):
        return build_dir_name

    configured_target = idf_target or resolve_configured_idf_target(app_path)
    if not configured_target:
        return None

    target_build_dir_name = f"build-{configured_target}{'-docker' if container else ''}"
    if not prefer_existing_default:
        return target_build_dir_name

    legacy_build_dir = app_path / "build"
    target_build_dir = app_path / target_build_dir_name
    if target_build_dir.exists() or not legacy_build_dir.exists():
        return target_build_dir_name
    return None


def resolve_flash_idf_build_dir_name(frontend: str, app_path: Path, port: str) -> str | None:
    """Resolve the ESP-IDF build directory for flashing, preferring the connected chip."""
    if build_dir_name := os.environ.get("ESPECTRE_IDF_BUILD_DIR"):
        return build_dir_name

    detected_chip = detect_chip_type(port)
    if detected_chip:
        try:
            _, detected_target = resolve_idf_target(frontend, detected_chip)
        except ValueError:
            print(
                f"{Fore.RED}❌ Connected chip {detected_chip} is not supported by the {frontend} frontend.{Style.RESET_ALL}"
            )
            raise SystemExit(1)
        return resolve_idf_build_dir_name(app_path, detected_target, prefer_existing_default=True)

    return resolve_idf_build_dir_name(app_path, prefer_existing_default=True)


def is_windows_host() -> bool:
    """Return True when the current host is Windows."""
    return os.name == "nt"


def format_idf_host_label() -> str:
    """Return a compact host label for user-facing output."""
    return "Windows" if is_windows_host() else "macOS/Linux"


def is_idf_environment_active() -> bool:
    """Best-effort check for an already prepared ESP-IDF shell."""
    return bool(shutil.which("idf.py") and os.environ.get("IDF_PATH") and os.environ.get("IDF_PYTHON_ENV_PATH"))


def iter_idf_install_candidates() -> list[tuple[str, Path]]:
    """Return likely ESP-IDF installation directories for the current host."""
    candidates: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    env_idf_path = os.environ.get("IDF_PATH")
    if env_idf_path:
        candidates.append(("IDF_PATH", Path(env_idf_path).expanduser()))

    home = Path(os.environ.get("USERPROFILE") or Path.home()) if is_windows_host() else Path.home()
    candidates.append(("standard ESP-IDF install", home / "esp" / "esp-idf"))
    if is_windows_host():
        candidates.append(("standard ESP-IDF install", home / "esp" / "v5.5" / "esp-idf"))

    unique_candidates: list[tuple[str, Path]] = []
    for source, candidate in candidates:
        try:
            normalized = candidate.expanduser().resolve(strict=False)
        except OSError:
            normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append((source, candidate.expanduser()))
    return unique_candidates


def resolve_idf_export_script_for_install(install_dir: Path) -> tuple[Path, str] | None:
    """Find a usable export script inside an ESP-IDF installation directory."""
    if is_windows_host():
        candidates = [
            (install_dir / "export.bat", "bat"),
            (install_dir / "export.ps1", "ps1"),
        ]
    else:
        candidates = [(install_dir / "export.sh", "sh")]

    for script_path, script_kind in candidates:
        if script_path.is_file():
            return script_path, script_kind
    return None


def get_esphome_idf_tools_path() -> Path | None:
    """Return ESPHome's native ESP-IDF cache path when ESPHome is available."""
    try:
        from esphome.espidf.framework import get_idf_tools_path
    except ImportError:
        return None

    try:
        return Path(get_idf_tools_path())
    except (OSError, RuntimeError, ValueError):
        return None


def build_esphome_idf_process_environment(
    framework_path: Path, python_env_path: Path
) -> dict[str, str]:
    """Build the subprocess environment for an ESPHome-managed ESP-IDF install."""
    from esphome.core import CORE
    from esphome.espidf.framework import get_framework_env

    previous_build_path = CORE.build_path
    CORE.build_path = REPO_ROOT
    try:
        return get_framework_env(framework_path, python_env_path, env=os.environ.copy())
    finally:
        CORE.build_path = previous_build_path


def repair_esphome_managed_idf_install() -> tuple[Path, Path]:
    """Complete an interrupted ESPHome-managed ESP-IDF installation."""
    from esphome.espidf.framework import check_esp_idf_install

    framework_path, python_env_path = check_esp_idf_install(IDF_VERSION)
    return Path(framework_path), Path(python_env_path)


def resolve_esphome_managed_idf_environment() -> ResolvedIdfEnvironment | None:
    """Resolve the pinned native ESP-IDF toolchain already managed by ESPHome."""
    tools_path = get_esphome_idf_tools_path()
    if tools_path is None:
        return None

    framework_path = tools_path / "frameworks" / IDF_VERSION
    python_env_path = tools_path / "penvs" / IDF_VERSION
    idf_py = framework_path / "tools" / "idf.py"
    python_executable = python_env_path / ("Scripts/python.exe" if is_windows_host() else "bin/python")
    if not idf_py.is_file():
        return None

    environment_complete = python_executable.is_file() and (
        python_env_path / ESPHOME_IDF_STAMP_FILE
    ).is_file()
    if not environment_complete:
        try:
            framework_path, python_env_path = repair_esphome_managed_idf_install()
        except (ImportError, OSError, RuntimeError, ValueError):
            return None
        idf_py = framework_path / "tools" / "idf.py"
        python_executable = python_env_path / (
            "Scripts/python.exe" if is_windows_host() else "bin/python"
        )
        if not idf_py.is_file() or not python_executable.is_file():
            return None

    try:
        process_env = build_esphome_idf_process_environment(framework_path, python_env_path)
    except (ImportError, OSError, RuntimeError, ValueError):
        return None

    return ResolvedIdfEnvironment(
        mode="esphome",
        source="ESPHome-managed native toolchain",
        install_dir=framework_path,
        idf_path_entry=str(idf_py),
        python_executable=python_executable,
        process_env=process_env,
    )


def resolve_idf_environment() -> ResolvedIdfEnvironment:
    """Resolve how the CLI should launch idf.py on this host."""
    if is_idf_environment_active():
        return ResolvedIdfEnvironment(
            mode="active",
            source="active ESP-IDF shell",
            install_dir=Path(os.environ["IDF_PATH"]).expanduser(),
        )

    for source, install_dir in iter_idf_install_candidates():
        export_script = resolve_idf_export_script_for_install(install_dir)
        if export_script is None:
            continue
        script_path, script_kind = export_script
        return ResolvedIdfEnvironment(
            mode="export",
            source=source,
            install_dir=install_dir,
            export_script=script_path,
            export_kind=script_kind,
        )

    if esphome_env := resolve_esphome_managed_idf_environment():
        return esphome_env

    idf_on_path = shutil.which("idf.py")
    if idf_on_path:
        return ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry=idf_on_path)

    raise FileNotFoundError("idf.py")


def describe_idf_environment(env: ResolvedIdfEnvironment) -> str:
    """Return a user-facing description of the resolved ESP-IDF environment."""
    if env.mode == "active":
        return f"using active shell environment at {env.install_dir}"
    if env.mode == "export":
        return f"auto-loading {env.source} at {env.install_dir}"
    if env.mode == "esphome":
        return f"using {env.source} at {env.install_dir}"
    return f"using idf.py from PATH at {env.idf_path_entry}"


def resolve_idf_build_backend(
    requested_backend: str = "auto",
    pull_policy: str = "ask",
) -> ResolvedIdfBuildBackend:
    """Select the common local-first or pinned-Docker ESP-IDF backend."""
    if requested_backend not in {"auto", "local", "docker"}:
        raise ValueError(f"Unsupported ESP-IDF build backend: {requested_backend}")

    if requested_backend != "docker":
        try:
            return ResolvedIdfBuildBackend(
                mode="local",
                idf_environment=resolve_idf_environment(),
            )
        except FileNotFoundError:
            if requested_backend == "local":
                raise

    return ResolvedIdfBuildBackend(
        mode="docker",
        docker=ensure_docker_backend(pull_policy),
    )


def print_idf_recovery_instructions() -> None:
    """Print concise, platform-aware recovery guidance."""
    print(f"{Fore.YELLOW}Try one of these setup paths, then rerun {cli_command('doctor')}.{Style.RESET_ALL}")
    print(f"  1. {cli_command('esphome', 'build', '--chip', 'c3')}")
    if is_windows_host():
        print('  2. . "$env:USERPROFILE\\esp\\esp-idf\\export.ps1"')
        print(f"     {cli_command('doctor')}")
    else:
        print("  2. source ~/esp/esp-idf/export.sh")
        print(f"     {cli_command('doctor')}")


def quote_powershell_literal(value: str) -> str:
    """Quote a string as a PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def prepare_idf_subprocess_command(
    command: list[str], env: ResolvedIdfEnvironment
) -> tuple[list[str], Path | None]:
    """Prepare the subprocess command for the resolved ESP-IDF environment."""
    if env.mode == "esphome":
        assert env.python_executable is not None
        assert env.idf_path_entry is not None
        return [str(env.python_executable), env.idf_path_entry, *command[1:]], None
    if env.mode != "export":
        return command, None

    assert env.export_script is not None
    assert env.export_kind is not None

    if env.export_kind == "sh":
        shell = shutil.which("bash") or shutil.which("zsh") or "/bin/sh"
        shell_command = f". {shlex.quote(str(env.export_script))} >/dev/null && {shlex.join(command)}"
        return [shell, "-lc", shell_command], env.export_script

    if env.export_kind == "bat":
        shell = os.environ.get("COMSPEC") or shutil.which("cmd") or "cmd.exe"
        shell_command = f'call "{env.export_script}" >NUL && {subprocess.list2cmdline(command)}'
        return [shell, "/d", "/c", shell_command], env.export_script

    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        raise FileNotFoundError("powershell")
    quoted_command = " ".join(quote_powershell_literal(part) for part in command)
    shell_command = f"& {{ . {quote_powershell_literal(str(env.export_script))}; & {quoted_command} }}"
    return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", shell_command], env.export_script


def run_in_idf_environment(
    command: list[str],
    env: ResolvedIdfEnvironment,
    *,
    cwd: Path,
) -> None:
    """Run an arbitrary toolchain command in one resolved ESP-IDF environment."""
    if env.mode == "export" or (command and command[0] == "idf.py"):
        subprocess_command, _ = prepare_idf_subprocess_command(command, env)
    else:
        subprocess_command = command
    if env.process_env is None:
        subprocess.run(subprocess_command, cwd=cwd, check=True)
    else:
        subprocess.run(subprocess_command, cwd=cwd, check=True, env=env.process_env)


def prepare_idf_subprocess_command_sequence(
    commands: list[list[str]], env: ResolvedIdfEnvironment
) -> tuple[list[str], Path | None]:
    """Prepare a single subprocess command that runs multiple idf.py commands."""
    if not commands:
        raise ValueError("at least one idf.py command is required")
    if len(commands) == 1 or env.mode != "export":
        return prepare_idf_subprocess_command(commands[0], env)

    assert env.export_script is not None
    assert env.export_kind is not None

    if env.export_kind == "sh":
        shell = shutil.which("bash") or shutil.which("zsh") or "/bin/sh"
        joined_commands = " && ".join(shlex.join(command) for command in commands)
        shell_command = f". {shlex.quote(str(env.export_script))} >/dev/null && {joined_commands}"
        return [shell, "-lc", shell_command], env.export_script

    if env.export_kind == "bat":
        shell = os.environ.get("COMSPEC") or shutil.which("cmd") or "cmd.exe"
        joined_commands = " && ".join(subprocess.list2cmdline(command) for command in commands)
        shell_command = f'call "{env.export_script}" >NUL && {joined_commands}'
        return [shell, "/d", "/c", shell_command], env.export_script

    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        raise FileNotFoundError("powershell")
    joined_commands = " ; ".join(
        "& " + " ".join(quote_powershell_literal(part) for part in command) for command in commands
    )
    shell_command = f"& {{ . {quote_powershell_literal(str(env.export_script))}; {joined_commands} }}"
    return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", shell_command], env.export_script


def run_idf_doctor(_args) -> int:
    """Validate the local ESP-IDF environment used by the repository CLI."""
    print(f"{Fore.CYAN}Host:     {format_idf_host_label()}{Style.RESET_ALL}")
    try:
        env = resolve_idf_environment()
    except FileNotFoundError:
        print(f"{Fore.RED}❌ No usable ESP-IDF installation was auto-detected.{Style.RESET_ALL}")
        print_idf_recovery_instructions()
        raise SystemExit(1)

    print(f"{Fore.CYAN}ESP-IDF:  {describe_idf_environment(env)}{Style.RESET_ALL}")
    command = ["idf.py", "--version"]
    subprocess_command, export_script = prepare_idf_subprocess_command(command, env)
    if export_script is not None:
        print(f"{Fore.CYAN}Export:   {export_script}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Command:  {' '.join(command)}{Style.RESET_ALL}")
    try:
        if env.process_env is None:
            subprocess.run(subprocess_command, check=True)
        else:
            subprocess.run(subprocess_command, check=True, env=env.process_env)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ The resolved ESP-IDF launcher could not be started.{Style.RESET_ALL}")
        print_idf_recovery_instructions()
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ ESP-IDF validation failed with exit code {e.returncode}.{Style.RESET_ALL}")
        print_idf_recovery_instructions()
        raise SystemExit(e.returncode)
    print(f"{Fore.GREEN}✅ ESP-IDF is ready for repository build commands.{Style.RESET_ALL}")
    return 0


def read_matter_onboarding(port: str, timeout_seconds: float = 20.0) -> bool:
    """Reset a Matter device and print its persisted onboarding codes."""
    try:
        import serial
    except ImportError:
        print(f"{Fore.RED}❌ pyserial is required to read the Matter QR code.{Style.RESET_ALL}")
        return False

    print(f"{Fore.CYAN}Matter QR: waiting for {port}; press RESET if needed...{Style.RESET_ALL}")
    try:
        from .serial_monitor import hard_reset_serial

        with serial.Serial(port, baudrate=115200, timeout=1.0) as connection:
            hard_reset_serial(connection)

            deadline = time.monotonic() + timeout_seconds
            qr_payload = None
            manual_code = None
            while time.monotonic() < deadline:
                line = connection.readline().decode("utf-8", errors="replace")
                if qr_match := MATTER_QR_PATTERN.search(line):
                    qr_payload = qr_match.group(1)
                if manual_match := MATTER_MANUAL_CODE_PATTERN.search(line):
                    manual_code = manual_match.group(1)
                if qr_payload and manual_code:
                    print(f"{Fore.GREEN}✅ Matter onboarding data{Style.RESET_ALL}")
                    print(f"  QR payload:  {qr_payload}")
                    print(f"  Manual code: {manual_code}")
                    return True
    except (OSError, serial.SerialException) as exc:
        print(f"{Fore.RED}❌ Cannot read Matter onboarding data: {exc}{Style.RESET_ALL}")
        return False

    print(f"{Fore.YELLOW}Matter onboarding data was not received. Reset the board and retry with "
          f"{cli_command('matter', 'qr', '--port', port)}.{Style.RESET_ALL}")
    return False


def run_idf_command(frontend: str, args) -> None:
    """Run an IDF workflow for the given frontend."""
    chip = getattr(args, "chip", None)
    try:
        if args.idf_command == "build":
            app_dir, idf_target = resolve_idf_target(frontend, chip)
        else:
            app_dir = IDF_FRONTENDS[frontend]["app_dir"]
            idf_target = None
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    print(f"{Fore.CYAN}Frontend: {frontend}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}App dir:   {app_dir}{Style.RESET_ALL}")

    app_path = Path(app_dir)
    build_dir_name = None
    if args.idf_command == "qr":
        port = get_serial_port(args.port)
        if not read_matter_onboarding(port):
            raise SystemExit(1)
        return
    idf_env = None
    docker = None
    build_backend = "local"
    if args.idf_command == "build":
        requested_backend = getattr(args, "backend", "auto")
        try:
            resolved_backend = resolve_idf_build_backend(
                requested_backend,
                getattr(args, "pull", "ask"),
            )
        except FileNotFoundError:
            print(f"{Fore.RED}❌ No usable local ESP-IDF installation was auto-detected.{Style.RESET_ALL}")
            print_idf_recovery_instructions()
            raise SystemExit(1)
        except DockerBackendError as exc:
            print(f"{Fore.RED}❌ {exc}{Style.RESET_ALL}")
            raise SystemExit(1)
        build_backend = resolved_backend.mode
        idf_env = resolved_backend.idf_environment
        docker = resolved_backend.docker
        if build_backend == "docker":
            print(f"{Fore.CYAN}Build env: Docker with the pinned ESP-IDF image{Style.RESET_ALL}")
        build_dir_name = resolve_idf_build_dir_name(
            app_path,
            idf_target,
            container=build_backend == "docker",
        )
    clean_requested = getattr(args, "clean", False) or getattr(args, "clean_all", False)
    if args.idf_command == "build":
        if getattr(args, "clean_all", False):
            clean_all_idf_build_artifacts(app_path)
        elif getattr(args, "clean", False):
            clean_idf_build_artifacts(app_path, build_dir_name)

    sdkconfig_defaults = resolve_sdkconfig_defaults(app_path, idf_target)
    defaults_arg = f"-DSDKCONFIG_DEFAULTS={sdkconfig_defaults}"
    custom_sdkconfig = os.environ.get("ESPECTRE_IDF_SDKCONFIG")
    sdkconfig_path = Path(custom_sdkconfig).resolve() if custom_sdkconfig else None
    cmake_args = [defaults_arg]
    if sdkconfig_path is not None:
        cmake_args.append(f"-DSDKCONFIG={sdkconfig_path}")
    if frontend == "native" and args.idf_command == "build":
        ota_channel = getattr(args, "ota_channel", None)
        if ota_channel:
            cmake_args.append(f"-DNATIVE_OTA_CHANNEL={ota_channel}")
            print(f"{Fore.CYAN}OTA channel: {ota_channel}{Style.RESET_ALL}")

    commands = []
    flash_port = None
    if args.idf_command == "build":
        base_command = build_idf_base_command(build_dir_name)
        cached_sdkconfig = cached_sdkconfig_path(app_path, build_dir_name)
        if sdkconfig_path is None and cached_sdkconfig not in {None, (app_path / "sdkconfig").resolve()}:
            cmake_args.append(f"-DSDKCONFIG={(app_path / 'sdkconfig').resolve()}")
        commands = []
        if clean_requested or not sdkconfig_matches_target(app_path, idf_target, sdkconfig_path):
            commands.append([*base_command, *cmake_args, "set-target", idf_target])
        commands.append([*base_command, *cmake_args, "build"])
    elif args.idf_command == "flash":
        port = get_serial_port(args.port)
        flash_port = port
        build_dir_name = resolve_flash_idf_build_dir_name(frontend, app_path, port)
        base_command = build_idf_base_command(build_dir_name)
        cached_sdkconfig = cached_sdkconfig_path(app_path, build_dir_name)
        if sdkconfig_path is None and cached_sdkconfig not in {None, (app_path / "sdkconfig").resolve()}:
            cmake_args.append(f"-DSDKCONFIG={(app_path / 'sdkconfig').resolve()}")
        commands = [[*base_command, *cmake_args[1:], "-p", port, "flash"]]

    if args.idf_command == "build" and build_backend == "docker":
        for command in commands:
            print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
        try:
            run_idf_container(
                frontend=frontend,
                app_path=app_path,
                commands=commands,
                repo_root=REPO_ROOT,
                sdkconfig_defaults=sdkconfig_defaults,
                pull_policy=getattr(args, "pull", "ask"),
                docker=docker,
            )
        except DockerBackendError as exc:
            print(f"{Fore.RED}❌ {exc}{Style.RESET_ALL}")
            raise SystemExit(1)
        return

    try:
        env = idf_env or resolve_idf_environment()
    except FileNotFoundError:
        print(f"{Fore.RED}❌ No usable local ESP-IDF installation was auto-detected.{Style.RESET_ALL}")
        print_idf_recovery_instructions()
        raise SystemExit(1)

    print(f"{Fore.CYAN}ESP-IDF: {describe_idf_environment(env)}{Style.RESET_ALL}")
    try:
        if env.mode == "export" and len(commands) > 1:
            for command in commands:
                print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
            subprocess_command, export_script = prepare_idf_subprocess_command_sequence(commands, env)
            assert export_script is not None
            print(f"{Fore.CYAN}Export:  {export_script}{Style.RESET_ALL}")
            if env.process_env is None:
                subprocess.run(subprocess_command, cwd=app_dir, check=True)
            else:
                subprocess.run(subprocess_command, cwd=app_dir, check=True, env=env.process_env)
        else:
            fallback_notice_printed = False
            for command in commands:
                print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
                subprocess_command, export_script = prepare_idf_subprocess_command(command, env)
                if export_script is not None and not fallback_notice_printed:
                    print(f"{Fore.CYAN}Export:  {export_script}{Style.RESET_ALL}")
                    fallback_notice_printed = True
                if env.process_env is None:
                    subprocess.run(subprocess_command, cwd=app_dir, check=True)
                else:
                    subprocess.run(subprocess_command, cwd=app_dir, check=True, env=env.process_env)
        if frontend == "matter" and args.idf_command == "flash" and flash_port is not None:
            read_matter_onboarding(flash_port)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ The resolved ESP-IDF launcher could not be started.{Style.RESET_ALL}")
        print_idf_recovery_instructions()
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ idf.py command failed with exit code {e.returncode}{Style.RESET_ALL}")
        raise SystemExit(e.returncode)
