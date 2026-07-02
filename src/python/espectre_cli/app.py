"""Main parser and dispatcher for the ESPectre repository CLI."""

from __future__ import annotations

import argparse

from .common import MICRO_CHIP_CHOICES, add_mqtt_connection_args, build_mqtt_namespace
from .esphome import run_esphome_command
from .host import collect_csi_data, detect_live_motion, open_web_ui
from .idf import run_idf_command
from .micro import deploy_code, flash_firmware, run_application, verify_installation
from .mqtt_shell import EspectreMQTTShell
from .targets import ESPHOME_CONFIGS, IDF_FRONTENDS


def run_mqtt_shell(args) -> int:
    shell = EspectreMQTTShell(build_mqtt_namespace(args))
    shell.start()
    return 0


def _add_micro_namespace(subparsers) -> None:
    micro_parser = subparsers.add_parser(
        "micro",
        help="MicroPython and host-side R&D workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_mqtt_connection_args(micro_parser)
    micro_subparsers = micro_parser.add_subparsers(dest="micro_command", help="Micro workflow commands")

    flash_parser = micro_subparsers.add_parser("flash", help="Flash MicroPython firmware to ESP32")
    flash_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, help="ESP32 chip type (auto-detected if not specified)")
    flash_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    flash_parser.add_argument("--erase", action="store_true", help="Erase flash before flashing (recommended)")
    flash_parser.add_argument("--firmware", help="Custom firmware path (optional)")
    flash_parser.set_defaults(handler=flash_firmware)

    deploy_parser = micro_subparsers.add_parser("deploy", help="Deploy code to MicroPython device")
    deploy_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    deploy_parser.set_defaults(handler=deploy_code)

    run_parser = micro_subparsers.add_parser("run", help="Run application on ESP32")
    run_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    run_parser.set_defaults(handler=run_application)

    verify_parser = micro_subparsers.add_parser("verify", help="Verify installation")
    verify_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    verify_parser.set_defaults(handler=verify_installation)

    ui_parser = micro_subparsers.add_parser("ui", help="Open a web UI in the browser")
    ui_parser.add_argument(
        "interface",
        nargs="?",
        choices=["mqtt", "ble", "theremin"],
        default="mqtt",
        help="Web UI to open (default: mqtt)",
    )
    ui_parser.set_defaults(handler=lambda args: open_web_ui(args.interface))

    collect_parser = micro_subparsers.add_parser("collect", help="Collect labeled CSI data for training")
    collect_parser.add_argument("--label", "-l", help="Label for collected data (e.g., static_presence, motion, empty, wave)")
    collect_parser.add_argument(
        "--samples",
        "--count",
        "-n",
        dest="samples",
        type=int,
        default=1,
        help="Number of timed collections/samples to record (default: 1)",
    )
    collect_parser.add_argument("--duration", "-d", type=float, default=2.0, help="Duration per sample in seconds (default: 2.0)")
    collect_parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Delay before starting collection in seconds (default: 0.0)",
    )
    collect_parser.add_argument("--info", "-i", action="store_true", help="Show dataset statistics")
    collect_parser.add_argument("--interactive", action="store_true", help="Interactive mode (press ENTER for each sample)")
    collect_parser.add_argument("--udp-port", type=int, default=5001, help="UDP port for CSI reception (default: 5001)")
    collect_parser.add_argument("--bind-ip", default=None, help="Local IP/interface for UDP bind (default: auto-detect)")
    collect_parser.add_argument("--streamer-ip", help="IPv4 address of the streamer device to stimulate")
    collect_parser.add_argument("--stimulus-port", type=int, default=9999, help="UDP port used by the streamer listener (default: 9999)")
    collect_parser.add_argument("--stimulus-rate", type=int, default=100, help="Stimulus packets per second sent to the streamer (default: 100)")
    collect_parser.add_argument("--reference-every", type=int, default=0, help="Mark every Nth stimulus packet as reference (default: 0 = measurement only)")
    collect_parser.add_argument("--contributor", "-c", help="GitHub username of the contributor")
    collect_parser.add_argument("--description", help="Description for the collected samples")
    collect_parser.set_defaults(handler=collect_csi_data)

    detect_parser = micro_subparsers.add_parser("detect", help="Run live ML motion detection from CSI UDP stream")
    detect_parser.add_argument("--udp-port", type=int, default=5001, help="UDP port for CSI reception (default: 5001)")
    detect_parser.add_argument("--bind-ip", default=None, help="Local IP/interface for UDP bind (default: auto-detect)")
    detect_parser.add_argument("--streamer-ip", required=True, help="IPv4 address of the streamer device to stimulate")
    detect_parser.add_argument("--stimulus-port", type=int, default=9999, help="UDP port used by the streamer listener (default: 9999)")
    detect_parser.add_argument("--stimulus-rate", type=int, default=100, help="Stimulus packets per second sent to the streamer (default: 100)")
    detect_parser.add_argument("--reference-every", type=int, default=0, help="Mark every Nth stimulus packet as reference (default: 0 = measurement only)")
    detect_parser.add_argument("--log-features", action="store_true", help="Print the 8 ML features after each published sample")
    detect_parser.add_argument("--log-turbulence", action="store_true", help="Print raw/filtered turbulence and recent buffer tail after each publish")
    detect_parser.add_argument("--log-only-motion", action="store_true", help="Only print publish lines when the effective state is MOTION")
    detect_parser.add_argument("--window-tail", type=int, default=16, help="Number of latest turbulence samples to print with --log-turbulence")
    detect_parser.add_argument("--capture-label", help="Also save received raw CSI as this dataset label")
    detect_parser.add_argument("--capture-duration", type=float, help="Stop and save capture after N seconds of received CSI")
    detect_parser.add_argument("--contributor", "-c", help="GitHub username of the contributor for saved captures")
    detect_parser.add_argument("--description", help="Description for the saved live-detect capture")
    detect_parser.set_defaults(handler=detect_live_motion)

    mqtt_parser = micro_subparsers.add_parser("mqtt", help="Start the interactive MQTT shell")
    mqtt_parser.set_defaults(handler=run_mqtt_shell)


def _add_esphome_namespace(subparsers) -> None:
    esphome_parser = subparsers.add_parser("esphome", help="ESPHome frontend workflow")
    esphome_subparsers = esphome_parser.add_subparsers(dest="esphome_command", required=True, help="ESPHome commands")

    for command_name, help_text in {
        "build": "Build the selected ESPHome firmware",
        "flash": "Build and flash the selected ESPHome firmware",
        "config": "Validate and render the selected ESPHome config",
        "logs": "Open serial logs for the selected ESPHome config",
    }.items():
        command_parser = esphome_subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--chip", choices=sorted(ESPHOME_CONFIGS.keys()), help="Target chip family")
        command_parser.add_argument("--dev", action="store_true", help="Use the *-dev example config")
        command_parser.add_argument("--config", help="Explicit ESPHome YAML path override")
        command_parser.add_argument("--device", help="Serial device for flash/logs when needed")
        command_parser.set_defaults(handler=run_esphome_command)


def _add_idf_namespace(subparsers, frontend: str) -> None:
    parser = subparsers.add_parser(frontend, help=f"{frontend.capitalize()} ESP-IDF frontend workflow")
    idf_subparsers = parser.add_subparsers(dest="idf_command", required=True, help=f"{frontend.capitalize()} commands")

    for command_name, help_text in {
        "build": "Configure target and build firmware",
        "flash": "Flash firmware with idf.py",
        "monitor": "Open idf.py monitor",
    }.items():
        command_parser = idf_subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--chip", choices=sorted(IDF_FRONTENDS[frontend]["targets"].keys()), required=True, help="ESP-IDF target chip")
        if command_name in {"flash", "monitor"}:
            command_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
        if command_name == "monitor":
            command_parser.add_argument(
                "--print-filter",
                dest="print_filter",
                help="Forward an ESP-IDF monitor print filter such as '*:W espectre.matter:I'",
            )
        command_parser.set_defaults(handler=lambda args, current_frontend=frontend: run_idf_command(current_frontend, args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ESPectre CLI - repository orchestrator for micro, esphome, native, matter, and streamer workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./espectre micro flash --erase
  ./espectre micro deploy
  ./espectre micro
  ./espectre esphome build --chip c3 --dev
  ./espectre native build --chip c3
  ./espectre matter build --chip c3
  ./espectre streamer monitor --chip s3 --port /dev/cu.usbmodemXXXX
""",
    )
    subparsers = parser.add_subparsers(dest="namespace", help="Available namespaces")
    _add_micro_namespace(subparsers)
    _add_esphome_namespace(subparsers)
    _add_idf_namespace(subparsers, "native")
    _add_idf_namespace(subparsers, "matter")
    _add_idf_namespace(subparsers, "streamer")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.namespace is None:
        parser.print_help()
        return 0

    if args.namespace == "micro" and getattr(args, "micro_command", None) is None:
        return run_mqtt_shell(args)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("missing command")
    result = handler(args)
    return 0 if result is None else int(result)
