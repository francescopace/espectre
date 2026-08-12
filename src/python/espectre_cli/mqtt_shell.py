# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI MQTT Shell

Interactive MQTT shell for ESPectre.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict

import paho.mqtt.client as mqtt
import yaml
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle

try:
    from paho.mqtt.enums import CallbackAPIVersion

    PAHO_V2 = True
except ImportError:
    CallbackAPIVersion = None
    PAHO_V2 = False

from .common import (
    CompactDumper,
    Fore,
    Style,
)
from .host import open_web_ui
from micro_espectre.branding import ASCII_BANNER


def _make_mqtt_client(username: str | None, password: str | None) -> mqtt.Client:
    """Build a paho client with the repo's compatibility settings."""
    if PAHO_V2:
        client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1)
    else:
        client = mqtt.Client()
    if username:
        client.username_pw_set(username, password if password else None)
    return client


def _mqtt_topic_bindings(args: argparse.Namespace) -> tuple[str, str, list[str], str]:
    """Return base, command, response topics, and info topic."""
    device_id = (args.device_id or "").strip()
    if not device_id:
        raise ValueError("MQTT device id is required for non-interactive commands")
    topic_prefix = args.topic_prefix.rstrip("/")
    base_topic = f"{topic_prefix}/{device_id}"
    return (
        base_topic,
        f"{base_topic}/commands/request",
        [
            f"{base_topic}/commands/accepted",
            f"{base_topic}/commands/rejected",
        ],
        f"{base_topic}/info",
    )


def _wait_for_subscription_ready(
    client: mqtt.Client,
    topics: list[str],
    *,
    timeout_s: float,
    error_holder: dict[str, str],
) -> threading.Event:
    """Subscribe to topics and return an event that signals all SUBACKs arrived."""
    if not topics:
        ready_event = threading.Event()
        ready_event.set()
        return ready_event

    ready_event = threading.Event()
    subscribed_mids: set[int] = set()
    expected_mids: set[int] = set()

    def on_subscribe(_client, _userdata, mid, granted_qos, properties=None):
        del granted_qos, properties
        subscribed_mids.add(int(mid))
        if subscribed_mids >= expected_mids:
            ready_event.set()

    client.on_subscribe = on_subscribe
    for topic in topics:
        result = client.subscribe(topic)
        if isinstance(result, tuple):
            rc = int(result[0])
            mid = int(result[1]) if len(result) > 1 else None
            if rc != mqtt.MQTT_ERR_SUCCESS:
                error_holder["subscribe"] = f"MQTT subscribe failed for {topic} with return code {rc}"
                ready_event.set()
                continue
            if mid is not None:
                expected_mids.add(mid)
        else:
            # Fallback for clients that do not expose Paho's subscribe tuple.
            ready_event.set()
    if not expected_mids and "subscribe" not in error_holder:
        ready_event.set()
    return ready_event


def send_mqtt_command_and_wait(
    args: argparse.Namespace,
    cmd_data: Dict[str, Any],
    *,
    timeout_s: float = 10.0,
    observe_request_echo: bool = False,
) -> Dict[str, Any]:
    """Publish one MQTT command and wait for the matching response."""
    _base_topic, topic_cmd, topic_responses, _info_topic = _mqtt_topic_bindings(args)
    command = dict(cmd_data)
    command.setdefault("protocol_version", "1.0")
    command_id = str(command.get("command_id") or f"cmd-{uuid.uuid4().hex[:12]}")
    command["command_id"] = command_id

    client = _make_mqtt_client(args.username, args.password)
    connected_event = threading.Event()
    response_event = threading.Event()
    request_echo_event = threading.Event()
    response_holder: dict[str, Dict[str, Any]] = {}
    error_holder: dict[str, str] = {}
    subscription_event: threading.Event | None = None

    def on_connect(client, userdata, flags, rc, properties=None):
        del userdata, flags, properties
        if rc == 0:
            nonlocal subscription_event
            subscription_event = _wait_for_subscription_ready(
                client,
                [*topic_responses, *([topic_cmd] if observe_request_echo else [])],
                timeout_s=timeout_s,
                error_holder=error_holder,
            )
        else:
            error_holder["connect"] = f"MQTT connect failed with return code {rc}"
        connected_event.set()

    def on_message(client, userdata, msg):
        del client, userdata
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
        except Exception:
            return
        topic = msg.topic.decode() if isinstance(msg.topic, bytes) else msg.topic
        if observe_request_echo and topic == topic_cmd and data.get("command_id") == command_id:
            request_echo_event.set()
            return
        if data.get("command_id") != command_id:
            return
        response_holder["payload"] = data
        response_event.set()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
        client.loop_start()
        if not connected_event.wait(timeout=timeout_s):
            raise RuntimeError(f"timed out connecting to MQTT broker {args.broker}:{args.port}")
        if "connect" in error_holder:
            raise RuntimeError(error_holder["connect"])
        if subscription_event is not None and not subscription_event.wait(timeout=timeout_s):
            raise RuntimeError("timed out waiting for MQTT subscriptions")
        if "subscribe" in error_holder:
            raise RuntimeError(error_holder["subscribe"])

        publish_result = client.publish(topic_cmd, json.dumps(command))
        if getattr(publish_result, "rc", mqtt.MQTT_ERR_UNKNOWN) != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"failed to publish MQTT command to {topic_cmd} "
                f"(rc={getattr(publish_result, 'rc', 'unknown')})"
            )
        if not response_event.wait(timeout=timeout_s):
            request_echo = "yes" if request_echo_event.is_set() else "no"
            raise RuntimeError(
                f"timed out waiting for MQTT response to {command_id} "
                f"(topic={topic_cmd} request_echo={request_echo})"
            )
        return response_holder["payload"]
    finally:
        client.loop_stop()
        client.disconnect()


def request_mqtt_info_and_wait(
    args: argparse.Namespace,
    *,
    timeout_s: float = 10.0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Request MQTT info and wait for both the command ack and info payload."""
    _base_topic, topic_cmd, topic_responses, info_topic = _mqtt_topic_bindings(args)
    command_id = f"cmd-{uuid.uuid4().hex[:12]}"
    command = {
        "protocol_version": "1.0",
        "command_id": command_id,
        "command": "info",
    }

    client = _make_mqtt_client(args.username, args.password)
    connected_event = threading.Event()
    command_event = threading.Event()
    info_event = threading.Event()
    command_holder: dict[str, Dict[str, Any]] = {}
    info_holder: dict[str, Dict[str, Any]] = {}
    error_holder: dict[str, str] = {}
    subscription_event: threading.Event | None = None

    def on_connect(client, userdata, flags, rc, properties=None):
        del userdata, flags, properties
        if rc == 0:
            nonlocal subscription_event
            subscription_event = _wait_for_subscription_ready(
                client,
                [*topic_responses, info_topic],
                timeout_s=timeout_s,
                error_holder=error_holder,
            )
        else:
            error_holder["connect"] = f"MQTT connect failed with return code {rc}"
        connected_event.set()

    def on_message(client, userdata, msg):
        del client, userdata
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
        except Exception:
            return
        topic = msg.topic.decode() if isinstance(msg.topic, bytes) else msg.topic
        if topic == info_topic:
            info_holder["payload"] = data
            info_event.set()
            return
        if data.get("command_id") == command_id:
            command_holder["payload"] = data
            command_event.set()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
        client.loop_start()
        if not connected_event.wait(timeout=timeout_s):
            raise RuntimeError(f"timed out connecting to MQTT broker {args.broker}:{args.port}")
        if "connect" in error_holder:
            raise RuntimeError(error_holder["connect"])
        if subscription_event is not None and not subscription_event.wait(timeout=timeout_s):
            raise RuntimeError("timed out waiting for MQTT subscriptions")
        if "subscribe" in error_holder:
            raise RuntimeError(error_holder["subscribe"])

        publish_result = client.publish(topic_cmd, json.dumps(command))
        if getattr(publish_result, "rc", mqtt.MQTT_ERR_UNKNOWN) != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"failed to publish MQTT command to {topic_cmd} "
                f"(rc={getattr(publish_result, 'rc', 'unknown')})"
            )
        if not command_event.wait(timeout=timeout_s):
            raise RuntimeError(f"timed out waiting for MQTT response to {command_id}")
        if not info_event.wait(timeout=timeout_s):
            raise RuntimeError(f"timed out waiting for MQTT info payload after {command_id}")
        return command_holder["payload"], info_holder["payload"]
    finally:
        client.loop_stop()
        client.disconnect()


class EspectreMQTTShell:
    """Interactive MQTT CLI for runtime commands."""

    DISCOVERY_TIMEOUT_S = 2.0

    def __init__(self, args):
        self.broker = args.broker
        self.port = args.port
        self.topic_prefix = args.topic_prefix.rstrip("/")
        self.device_id = args.device_id or None
        self.base_topic = ""
        self.username = args.username
        self.password = args.password

        self.topic_cmd = ""
        self.topic_responses: list[str] = []
        self.discovery_info_topic = f"{self.topic_prefix}/+/info"
        self.discovery_status_topic = f"{self.topic_prefix}/+/status"
        self.discovered_devices: dict[str, dict[str, Any]] = {}
        self.discovery_active = self.device_id is None
        self._set_active_device(self.device_id)
        self.client = _make_mqtt_client(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.running = True

        hist_file = os.path.join(os.path.expanduser("~"), ".espectre_cli_history")
        completer_dict = {
            "set_threshold": None,
            "info": None,
            "stats": None,
            "ota_status": None,
            "ota_check": None,
            "ota_start": None,
            "clear": None,
            "help": None,
            "exit": None,
        }
        prompt_style = PromptStyle.from_dict({"prompt": "#00aa00 bold"})
        self.session = PromptSession(
            history=FileHistory(hist_file),
            completer=NestedCompleter.from_nested_dict(completer_dict),
            style=prompt_style,
            complete_while_typing=True,
            enable_history_search=True,
        )

    def _set_active_device(self, device_id: str | None) -> None:
        """Update topic bindings for the selected device."""
        self.device_id = device_id
        if not device_id:
            self.base_topic = ""
            self.topic_cmd = ""
            self.topic_responses = []
            return
        self.base_topic = f"{self.topic_prefix}/{device_id}"
        self.topic_cmd = f"{self.base_topic}/commands/request"
        self.topic_responses = [
            f"{self.base_topic}/commands/accepted",
            f"{self.base_topic}/commands/rejected",
        ]

    def _subscribe_selected_device(self, client) -> None:
        """Subscribe to command responses for the selected device."""
        print(f"{Fore.BLUE}Command topic: {self.topic_cmd}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Listening on: {', '.join(self.topic_responses)}{Style.RESET_ALL}")
        for topic in self.topic_responses:
            client.subscribe(topic)

    def _extract_device_id_from_topic(self, topic: str) -> str | None:
        """Extract the ESPectre device id from a topic under the configured prefix."""
        prefix = f"{self.topic_prefix}/"
        if not topic.startswith(prefix):
            return None
        remainder = topic[len(prefix) :]
        parts = remainder.split("/")
        if len(parts) < 2 or not parts[0]:
            return None
        if parts[1] not in {"info", "status"}:
            return None
        return parts[0]

    def _record_discovered_device(self, topic: str, payload: bytes | str) -> None:
        """Track devices seen through info/status broadcasts during discovery."""
        device_id = self._extract_device_id_from_topic(topic)
        if not device_id:
            return
        try:
            body = payload.decode() if isinstance(payload, bytes) else payload
            data = json.loads(body)
        except Exception:
            return

        device = self.discovered_devices.setdefault(device_id, {"device_id": device_id})
        if "device_id" in data and data["device_id"]:
            device["device_id"] = data["device_id"]
        if topic.endswith("/info"):
            for key in ("device_name", "device_label", "frontend", "chip"):
                if data.get(key):
                    device[key] = data[key]
        elif topic.endswith("/status"):
            if "online" in data:
                device["online"] = bool(data["online"])
        if "timestamp_ms" in data:
            device["timestamp_ms"] = data["timestamp_ms"]

    def _print_discovered_devices(self) -> list[dict[str, Any]]:
        """Render the devices discovered during the MQTT scan."""
        devices = sorted(self.discovered_devices.values(), key=lambda item: item["device_id"])
        print()
        print(f"{Fore.CYAN}Discovered MQTT devices:{Style.RESET_ALL}")
        for index, device in enumerate(devices, start=1):
            label = device.get("device_label") or device.get("device_name") or "unnamed"
            frontend = device.get("frontend", "unknown")
            online = "online" if device.get("online") else "offline/unknown"
            print(f"  {index}. {device['device_id']} | {label} | {frontend} | {online}")
        print()
        return devices

    def _prompt_for_device_choice(self, devices: list[dict[str, Any]]) -> str | None:
        """Prompt the user to select a discovered device or enter one manually."""
        if len(devices) == 1:
            selected = devices[0]["device_id"]
            print(f"{Fore.GREEN}Selected device: {selected}{Style.RESET_ALL}")
            return selected

        prompt = f"{Fore.CYAN}Select device (1-{len(devices)}) or enter a device id: {Style.RESET_ALL}"
        while True:
            try:
                response = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
                return None
            if not response:
                print(f"{Fore.RED}Please choose a device or enter a device id{Style.RESET_ALL}")
                continue
            if response.isdigit():
                choice = int(response)
                if 1 <= choice <= len(devices):
                    selected = devices[choice - 1]["device_id"]
                    print(f"{Fore.GREEN}Selected device: {selected}{Style.RESET_ALL}")
                    return selected
            if "/" not in response and "+" not in response and "#" not in response:
                print(f"{Fore.GREEN}Selected device: {response}{Style.RESET_ALL}")
                return response
            print(f"{Fore.RED}Invalid choice: {response}{Style.RESET_ALL}")

    def _activate_selected_device(self, device_id: str) -> None:
        """Switch from discovery mode to the selected device topics."""
        unsubscribe = getattr(self.client, "unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe(self.discovery_info_topic)
            unsubscribe(self.discovery_status_topic)
        self.discovery_active = False
        self._set_active_device(device_id)
        self._subscribe_selected_device(self.client)

    def select_device(self) -> bool:
        """Discover active devices over MQTT and prompt the user to choose one."""
        print(
            f"{Fore.YELLOW}Scanning MQTT for devices on {self.discovery_info_topic} "
            f"and {self.discovery_status_topic} for {self.DISCOVERY_TIMEOUT_S:.1f}s...{Style.RESET_ALL}"
        )
        time.sleep(self.DISCOVERY_TIMEOUT_S)

        devices = self._print_discovered_devices() if self.discovered_devices else []
        if not devices:
            try:
                manual_device_id = input(
                    f"{Fore.CYAN}No devices discovered. Enter a device id manually: {Style.RESET_ALL}"
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
                return False
            if not manual_device_id:
                print(f"{Fore.RED}No device selected{Style.RESET_ALL}")
                return False
            selected = manual_device_id
        else:
            selected = self._prompt_for_device_choice(devices)
            if not selected:
                return False

        self._activate_selected_device(selected)
        return True

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"{Fore.BLUE}Connected to: {self.broker}:{self.port}{Style.RESET_ALL}")
            if self.discovery_active:
                print(f"{Fore.BLUE}Discovery info topic: {self.discovery_info_topic}{Style.RESET_ALL}")
                print(f"{Fore.BLUE}Discovery status topic: {self.discovery_status_topic}{Style.RESET_ALL}")
                client.subscribe(self.discovery_info_topic)
                client.subscribe(self.discovery_status_topic)
            else:
                self._subscribe_selected_device(client)
        else:
            print(f"{Fore.RED}Failed to connect, return code {rc}{Style.RESET_ALL}")

    def on_message(self, client, userdata, msg):
        topic = getattr(msg, "topic", "")
        topic = topic.decode() if isinstance(topic, bytes) else topic
        if self.discovery_active:
            self._record_discovered_device(topic, msg.payload)
            return
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print()
            formatted_yaml = yaml.dump(data, Dumper=CompactDumper, sort_keys=False, default_flow_style=False, width=1000)
            print(f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} Received:")
            print_formatted_text(
                FormattedText([("class:pygments", formatted_yaml)]),
                style=PromptStyle.from_dict({"pygments": "#ansiwhite"}),
            )
            print()
        except Exception as e:
            print(f"\nError parsing message: {e}")

    def send_command(self, cmd_data: Dict[str, Any]):
        try:
            payload = json.dumps(cmd_data)
            self.client.publish(self.topic_cmd, payload)
        except Exception as e:
            print(f"{Fore.RED}Error sending command: {e}{Style.RESET_ALL}")

    def start(self):
        print(f"{Fore.MAGENTA}{ASCII_BANNER}")
        print(f"{Style.RESET_ALL}")

        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            time.sleep(0.5)
            if self.discovery_active and not self.select_device():
                return
            print(f"\n{Fore.YELLOW}Type 'help' for commands, 'exit' to quit{Style.RESET_ALL}\n")
            print(f"{Fore.YELLOW}Tip: Use TAB for autocompletion, Ctrl+R to search history{Style.RESET_ALL}\n")
            while self.running:
                try:
                    user_input = self.session.prompt(HTML("<prompt>espectre></prompt> "))
                    self.process_input(user_input)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            print("\nExiting...")

    def process_input(self, user_input):
        if not user_input.strip():
            return

        parts = shlex.split(user_input)
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["exit", "quit", "q"]:
            self.running = False
            return
        if cmd in ["help", "h"]:
            self.show_help()
            return
        if cmd in ["about", "a"]:
            self.show_about()
            return
        if cmd in ["webui", "web"]:
            open_web_ui()
            return
        if cmd in ["clear", "cls"]:
            os.system("cls" if os.name == "nt" else "clear")
            return

        try:
            if cmd in ["set_threshold", "st"]:
                self.cmd_set_threshold(args)
            elif cmd in ["info", "i"]:
                self.send_command({"command": "info"})
            elif cmd in ["stats", "s"]:
                self.send_command({"command": "stats"})
            elif cmd in ["ota_status", "os"]:
                self.send_command({"command": "ota_status"})
            elif cmd in ["ota_check", "oc"]:
                self.cmd_ota_check(args)
            elif cmd in ["ota_start", "ou"]:
                self.cmd_ota_start(args)
            else:
                print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error executing command: {e}{Style.RESET_ALL}")

    def cmd_set_threshold(self, args):
        if not args:
            print(f"{Fore.RED}Usage: set_threshold <threshold>{Style.RESET_ALL}")
            return
        self.send_command({"command": "set_threshold", "threshold": float(args[0])})

    def cmd_ota_check(self, args):
        if args:
            print(f"{Fore.RED}Usage: ota_check{Style.RESET_ALL}")
            return
        self.send_command({"command": "ota_check"})

    def cmd_ota_start(self, args):
        if args:
            print(f"{Fore.RED}Usage: ota_start{Style.RESET_ALL}")
            return
        self.send_command({"command": "ota_start"})

    def show_help(self):
        help_text = HTML(
            """
<ansibrightcyan><b>ESPectre MQTT Shell Commands</b></ansibrightcyan>

<ansiyellow><b>Configuration Commands:</b></ansiyellow>
  <ansigreen>set_threshold|st</ansigreen> &lt;val&gt;               Set session threshold (0.0-1.0)

<ansiyellow><b>System Commands:</b></ansiyellow>
  <ansigreen>info|i</ansigreen>                              Show current configuration
  <ansigreen>stats|s</ansigreen>                             Show runtime statistics (memory, loop time)
  <ansigreen>ota_status|os</ansigreen>                       Show OTA state
  <ansigreen>ota_check|oc</ansigreen>                        Check GitHub Releases for an update
  <ansigreen>ota_start|ou</ansigreen>                        Install the update from GitHub Releases

<ansiyellow><b>Utility Commands:</b></ansiyellow>
  <ansigreen>webui|web</ansigreen>                           Open the MQTT web UI
  <ansigreen>about|a</ansigreen>                             Show shell information
  <ansigreen>clear|cls</ansigreen>                           Clear screen
  <ansigreen>help|h</ansigreen>                              Show this help message
  <ansigreen>exit|quit|q</ansigreen>                         Exit interactive mode
"""
        )
        print()
        print_formatted_text(help_text)
        print()

    def show_about(self):
        """Display a compact shell summary."""
        print()
        print("ESPectre MQTT Shell")
        print("Interactive MQTT control surface for ESPectre Protocol devices.")
        if self.device_id:
            print(f"Selected device: {self.device_id}")
        print()
