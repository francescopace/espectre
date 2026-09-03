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
import sys
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
from micro_espectre.branding import ASCII_BANNER
_SHELL_ALIASES = {
    "s": "read_diagnostics",
    "st": "update_sensing",
    "oc": "check_ota",
    "ou": "start_ota",
}

_LOCAL_UTILITIES = {
    "help": None,
    "h": None,
    "about": None,
    "a": None,
    "clear": None,
    "cls": None,
    "exit": None,
    "quit": None,
    "q": None,
}

_MQTT_COMMANDS = {
    "update_device",
    "update_sensing",
    "recalibrate",
    "read_diagnostics",
    "check_ota",
    "start_ota",
}


def _mqtt_commands_from_catalog(payload: Dict[str, Any]) -> list[str]:
    """Return command names from a canonical capabilities payload."""
    raw = payload.get("operations")
    if isinstance(raw, list):
        return [
            str(item["name"])
            for item in raw
            if isinstance(item, dict) and item.get("name") in _MQTT_COMMANDS
        ]
    if isinstance(raw, dict):
        return [str(name) for name in raw if name in _MQTT_COMMANDS]
    return []


_OTA_CHANNELS = {"release": None, "preview": None, "develop": None}
_OTA_CHANNEL_NAMES = tuple(_OTA_CHANNELS)


def _mqtt_completer_dict(commands: list[str]) -> Dict[str, Any]:
    """Build tab-completion entries from device commands plus local utilities."""
    completer: Dict[str, Any] = {}
    for name in commands:
        completer[name] = dict(_OTA_CHANNELS) if name in {"check_ota", "start_ota"} else None
    for alias, target in _SHELL_ALIASES.items():
        if target in completer:
            completer[alias] = completer[target]
    completer.update(_LOCAL_UTILITIES)
    return completer


def _coerce_command_token(token: str) -> Any:
    """Coerce a shell token into a JSON-friendly command field."""
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    body = token[1:] if token.startswith("-") else token
    if body.isdigit():
        return int(token)
    try:
        return float(token)
    except ValueError:
        return token


def _mqtt_command_payload(command: str, args: list[str]) -> tuple[Dict[str, Any] | None, str | None]:
    """Build a protocol command payload from shell tokens.

    Named ``field=value`` tokens after the command are copied through. A single
    positional after ``update_sensing`` is stored as ``threshold``. A single
    positional after an OTA command is stored as ``channel``. The command name
    itself is never a ``key=value`` token.
    """
    fields: Dict[str, Any] = {"command": command}
    positionals: list[str] = []
    for arg in args:
        key, sep, value = arg.partition("=")
        if sep and key.isidentifier():
            fields[key] = _coerce_command_token(value)
        else:
            positionals.append(arg)
    if len(positionals) == 1 and command == "update_sensing":
        fields["threshold"] = _coerce_command_token(positionals[0])
        positionals = []
    if len(positionals) == 1 and command in {"check_ota", "start_ota"}:
        fields["channel"] = str(positionals[0])
        positionals = []
    if positionals:
        joined = " ".join(positionals)
        return None, f"unexpected argument: {joined}"
    if command in {"check_ota", "start_ota"} and "channel" in fields:
        channel = str(fields["channel"])
        if channel not in _OTA_CHANNEL_NAMES:
            return None, "invalid ota channel (accepted: release, preview, and develop)"
        fields["channel"] = channel
    return fields, None


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
    """Return base, command, response topics, and device topic."""
    device_id = (args.device_id or "").strip()
    if not device_id:
        raise ValueError("MQTT device id is required for non-interactive commands")
    topic_prefix = args.topic_prefix.rstrip("/")
    base_topic = f"{topic_prefix}/{device_id}"
    return (
        base_topic,
        f"{base_topic}/commands/request",
        [f"{base_topic}/commands/result"],
        f"{base_topic}/device",
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
    subscription_lock = threading.Lock()
    registration_complete = False

    def on_subscribe(_client, _userdata, mid, granted_qos, properties=None):
        del granted_qos, properties
        with subscription_lock:
            subscribed_mids.add(int(mid))
            if registration_complete and subscribed_mids >= expected_mids:
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
                with subscription_lock:
                    expected_mids.add(mid)
        else:
            # Fallback for clients that do not expose Paho's subscribe tuple.
            ready_event.set()
    with subscription_lock:
        registration_complete = True
        if subscribed_mids >= expected_mids and "subscribe" not in error_holder:
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


def request_mqtt_diagnostics_and_wait(
    args: argparse.Namespace,
    *,
    timeout_s: float = 10.0,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Request MQTT diagnostics and return the correlated result and data."""
    _base_topic, topic_cmd, topic_responses, _info_topic = _mqtt_topic_bindings(args)
    command_id = f"cmd-{uuid.uuid4().hex[:12]}"
    command = {
        "command_id": command_id,
        "command": "read_diagnostics",
    }

    client = _make_mqtt_client(args.username, args.password)
    connected_event = threading.Event()
    command_event = threading.Event()
    command_holder: dict[str, Dict[str, Any]] = {}
    error_holder: dict[str, str] = {}
    subscription_event: threading.Event | None = None

    def on_connect(client, userdata, flags, rc, properties=None):
        del userdata, flags, properties
        if rc == 0:
            nonlocal subscription_event
            subscription_event = _wait_for_subscription_ready(
                client,
                topic_responses,
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
        result = command_holder["payload"]
        info = result.get("data")
        if not isinstance(info, dict):
            raise RuntimeError(f"MQTT diagnostics response to {command_id} has no data object")
        return result, info
    finally:
        client.loop_stop()
        client.disconnect()


class EspectreMQTTShell:
    """Interactive MQTT CLI for runtime commands."""

    DISCOVERY_TIMEOUT_S = 2.0
    COMMAND_ACK_TIMEOUT_S = 10.0
    PROMPT_DISPLAY = "espectre> "
    _PAYLOAD_LABELS = {
        "read_diagnostics": "diagnostics",
    }

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
        self.discovery_info_topic = f"{self.topic_prefix}/+/device"
        self.discovery_status_topic = f"{self.topic_prefix}/+/health"
        self.discovered_devices: dict[str, dict[str, Any]] = {}
        self.discovery_active = self.device_id is None
        self._device_commands: list[str] = []
        self._quiet_command_ids: set[str] = set()
        self._suppress_catalog_payload = False
        self._set_active_device(self.device_id)
        self.client = _make_mqtt_client(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.running = True
        self._typed_line: str | None = None
        self._pending_lock = threading.Lock()
        self._discovery_lock = threading.Lock()
        self._pending_command_id = ""
        self._pending_command = ""
        self._pending_payload_label = ""
        self._pending_result: dict[str, Any] | None = None
        self._pending_payload: tuple[str, Dict[str, Any]] | None = None
        self._pending_result_event = threading.Event()
        self._pending_payload_event = threading.Event()

        hist_file = os.path.join(os.path.expanduser("~"), ".espectre_cli_history")
        prompt_style = PromptStyle.from_dict({"prompt": "#00aa00 bold"})
        self.session = PromptSession(
            history=FileHistory(hist_file),
            completer=NestedCompleter.from_nested_dict(_mqtt_completer_dict([])),
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
            f"{self.base_topic}/commands/result",
            f"{self.base_topic}/capabilities",
            f"{self.base_topic}/device",
            f"{self.base_topic}/health",
            f"{self.base_topic}/sensing",
            f"{self.base_topic}/wifi",
            f"{self.base_topic}/ota",
        ]

    def _subscribe_selected_device(self, client) -> None:
        """Subscribe to command responses and payload topics for the selected device."""
        print(f"{Fore.BLUE}Command topic: {self.topic_cmd}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Listening on: {', '.join(self.topic_responses)}{Style.RESET_ALL}")
        for topic in self.topic_responses:
            client.subscribe(topic)

    def _update_completer(self) -> None:
        """Refresh tab completion from the current device command catalog."""
        completer = NestedCompleter.from_nested_dict(_mqtt_completer_dict(self._device_commands))
        if getattr(self, "session", None) is not None:
            self.session.completer = completer

    def _apply_device_commands(self, commands: list[str]) -> None:
        """Replace the local command catalog used for help and completion."""
        self._device_commands = list(commands)
        self._update_completer()

    def _apply_catalog_payload(self, payload: Dict[str, Any]) -> None:
        """Adopt command names from a capabilities payload."""
        commands = _mqtt_commands_from_catalog(payload)
        if commands:
            self._apply_device_commands(commands)

    def _request_command_catalog(self) -> None:
        """Capabilities are retained and arrive after subscription."""
        return

    def _topic_label(self, topic: str) -> str:
        """Return the selected-device topic suffix used in received-message output."""
        prefix = f"{self.base_topic}/" if self.base_topic else ""
        if prefix and topic.startswith(prefix):
            return topic[len(prefix) :]
        return topic

    def _print_command_result(self, accepted: bool, data: Dict[str, Any]) -> None:
        """Render a compact command ACK on the line after the prompt."""
        command = str(data.get("command") or "command")
        print()
        if accepted:
            print(f"{Fore.GREEN}✓ {command}{Style.RESET_ALL}")
            return
        reason = str(data.get("message") or "").strip()
        suffix = f": {reason}" if reason else ""
        print(f"{Fore.RED}✗ {command}{suffix}{Style.RESET_ALL}")

    def _can_annotate_typed_command(self, typed: str | None) -> bool:
        """Return True when the ACK mark can be appended to the submitted prompt line."""
        if not typed or not sys.stdout.isatty():
            return False
        try:
            width = os.get_terminal_size().columns
        except OSError:
            return False
        return len(self.PROMPT_DISPLAY) + len(typed) + 2 < width

    def _annotate_typed_command(self, typed: str, accepted: bool, reason: str = "") -> None:
        """Append the ACK mark to the just-submitted prompt line."""
        mark = f"{Fore.GREEN}✓{Style.RESET_ALL}" if accepted else f"{Fore.RED}✗{Style.RESET_ALL}"
        extra = f" {Fore.RED}{reason}{Style.RESET_ALL}" if reason and not accepted else ""
        column = len(self.PROMPT_DISPLAY) + len(typed) + 1
        sys.stdout.write(f"\033[A\033[{column}G {mark}{extra}\n")
        sys.stdout.flush()

    def _show_command_ack(self, accepted: bool, data: Dict[str, Any]) -> None:
        """Place the ACK on the typed command line when possible, otherwise on the next line."""
        typed = self._typed_line
        reason = str(data.get("message") or "").strip()
        if self._can_annotate_typed_command(typed) and typed is not None:
            self._annotate_typed_command(typed, accepted, reason)
            return
        self._print_command_result(accepted, data)

    def _print_payload(self, data: Dict[str, Any], label: str) -> None:
        """Dump a JSON payload topic as compact YAML."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print()
        formatted_yaml = yaml.dump(data, Dumper=CompactDumper, sort_keys=False, default_flow_style=False, width=1000)
        received = f"Received on {label}:" if label else "Received:"
        print(f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {received}")
        print_formatted_text(
            FormattedText([("class:pygments", formatted_yaml)]),
            style=PromptStyle.from_dict({"pygments": "#ansiwhite"}),
        )
        print()

    def _clear_pending_command(self) -> None:
        """Drop in-flight command wait state."""
        with self._pending_lock:
            self._pending_command_id = ""
            self._pending_command = ""
            self._pending_payload_label = ""
            self._pending_result = None
            self._pending_payload = None
            self._pending_result_event.clear()
            self._pending_payload_event.clear()

    def _matches_pending_result(self, data: Dict[str, Any]) -> bool:
        """Return True when an ACK belongs to the in-flight shell command."""
        if not self._pending_command_id:
            return False
        incoming_id = str(data.get("command_id") or "")
        if incoming_id == self._pending_command_id:
            return True
        if incoming_id:
            return False
        incoming_command = str(data.get("command") or "")
        if incoming_command == self._pending_command:
            return True
        # Current firmware may drop command_id on parse failure and echo command=unknown.
        return incoming_command in {"", "unknown"}

    def _extract_device_id_from_topic(self, topic: str) -> str | None:
        """Extract the ESPectre device id from a topic under the configured prefix."""
        prefix = f"{self.topic_prefix}/"
        if not topic.startswith(prefix):
            return None
        remainder = topic[len(prefix) :]
        parts = remainder.split("/")
        if len(parts) < 2 or not parts[0]:
            return None
        if parts[1] not in {"device", "health"}:
            return None
        return parts[0]

    def _record_discovered_device(self, topic: str, payload: bytes | str) -> None:
        """Track devices seen through device/health broadcasts during discovery."""
        device_id = self._extract_device_id_from_topic(topic)
        if not device_id:
            return
        try:
            body = payload.decode() if isinstance(payload, bytes) else payload
            data = json.loads(body)
        except Exception:
            return

        with self._discovery_lock:
            device = self.discovered_devices.setdefault(
                device_id,
                {"device_id": device_id},
            )
            if "device_id" in data and data["device_id"]:
                device["device_id"] = data["device_id"]
            if topic.endswith("/device"):
                for key in ("name", "label", "frontend", "chip"):
                    if data.get(key):
                        device[key] = data[key]
            elif topic.endswith("/health") and "online" in data:
                device["online"] = bool(data["online"])
            if "timestamp_ms" in data:
                device["timestamp_ms"] = data["timestamp_ms"]

    def _print_discovered_devices(self) -> list[dict[str, Any]]:
        """Render the devices discovered during the MQTT scan."""
        with self._discovery_lock:
            devices = [dict(device) for device in self.discovered_devices.values()]
        devices.sort(key=lambda item: item["device_id"])
        print()
        print(f"{Fore.CYAN}Discovered MQTT devices:{Style.RESET_ALL}")
        for index, device in enumerate(devices, start=1):
            label = device.get("label") or device.get("name") or "unnamed"
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

        devices = self._print_discovered_devices()
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
            label = self._topic_label(topic)
            if label == "commands/result":
                accepted = data.get("accepted") is True
                command_id = str(data.get("command_id") or "")
                with self._pending_lock:
                    if command_id in self._quiet_command_ids:
                        self._quiet_command_ids.discard(command_id)
                        if not accepted:
                            self._suppress_catalog_payload = False
                        return
                    if self._matches_pending_result(data):
                        self._pending_result = {"accepted": accepted, "data": data}
                        self._pending_result_event.set()
                        return
                self._print_command_result(accepted, data)
                return
            if label == "capabilities":
                self._apply_catalog_payload(data)
                with self._pending_lock:
                    if self._suppress_catalog_payload:
                        self._suppress_catalog_payload = False
                        if self._pending_payload_label == label:
                            self._pending_payload = (label, data)
                            self._pending_payload_event.set()
                            return
                        return
            with self._pending_lock:
                if self._pending_payload_label and label == self._pending_payload_label:
                    self._pending_payload = (label, data)
                    self._pending_payload_event.set()
                    return
            self._print_payload(data, label)
        except Exception as e:
            print(f"\nError parsing message: {e}")

    def send_command(self, cmd_data: Dict[str, Any], *, timeout_s: float | None = None):
        command = dict(cmd_data)
        command_id = str(command.get("command_id") or f"cmd-{uuid.uuid4().hex[:12]}")
        command["command_id"] = command_id
        payload_label = self._PAYLOAD_LABELS.get(str(command.get("command") or ""))
        wait_s = self.COMMAND_ACK_TIMEOUT_S if timeout_s is None else timeout_s

        with self._pending_lock:
            self._pending_result_event.clear()
            self._pending_payload_event.clear()
            self._pending_command_id = command_id
            self._pending_command = str(command.get("command") or "")
            self._pending_payload_label = payload_label or ""
            self._pending_result = None
            self._pending_payload = None

        try:
            try:
                self.client.publish(self.topic_cmd, json.dumps(command))
            except Exception as e:
                print(f"{Fore.RED}Error sending command: {e}{Style.RESET_ALL}")
                return
            if not self._pending_result_event.wait(timeout=wait_s):
                self._show_command_ack(
                    False,
                    {"command": command.get("command"), "message": "timed out waiting for device"},
                )
                return
            with self._pending_lock:
                result = self._pending_result or {}
            accepted = bool(result.get("accepted"))
            data = result.get("data") or {"command": command.get("command")}
            self._show_command_ack(accepted, data)
            if accepted and isinstance(data, dict) and isinstance(data.get("data"), dict):
                if command.get("command") == "capabilities":
                    self._apply_catalog_payload(data["data"])
                self._print_payload(data["data"], str(command.get("command")))
            if not accepted or not payload_label or (isinstance(data, dict) and isinstance(data.get("data"), dict)):
                return
            if not self._pending_payload_event.wait(timeout=wait_s):
                print(f"{Fore.RED}timed out waiting for {payload_label}{Style.RESET_ALL}")
                return
            with self._pending_lock:
                payload = self._pending_payload
            if payload is not None:
                if payload[0] == "capabilities":
                    self._apply_catalog_payload(payload[1])
                self._print_payload(payload[1], payload[0])
        finally:
            self._clear_pending_command()

    def start(self):
        print(f"{Fore.MAGENTA}{ASCII_BANNER}")
        print(f"{Style.RESET_ALL}")

        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            time.sleep(0.5)
            if self.discovery_active and not self.select_device():
                return
            self._request_command_catalog()
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
        self._typed_line = user_input

        try:
            if cmd in ["exit", "quit", "q"]:
                self.running = False
                return
            if cmd in ["help", "h"]:
                self.show_help()
                return
            if cmd in ["about", "a"]:
                self.show_about()
                return
            if cmd in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                return

            command_name = _SHELL_ALIASES.get(cmd, cmd)
            payload, error = _mqtt_command_payload(command_name, args)
            if payload is None:
                print(f"{Fore.RED}{error}{Style.RESET_ALL}")
                return
            self.send_command(payload)
        except Exception as e:
            print(f"{Fore.RED}Error executing command: {e}{Style.RESET_ALL}")
        finally:
            self._typed_line = None

    def show_help(self):
        lines = [
            "",
            "<ansibrightcyan><b>ESPectre MQTT Shell</b></ansibrightcyan>",
            "",
            "MQTT commands are forwarded to the selected device. Unknown or unsupported commands are rejected by the device.",
        ]
        if self._device_commands:
            lines.append("")
            lines.append("<ansiyellow><b>Device commands</b></ansiyellow> (from MQTT <ansigreen>commands</ansigreen>):")
            for name in self._device_commands:
                aliases = [alias for alias, target in _SHELL_ALIASES.items() if target == name]
                label = "|".join([name, *aliases])
                lines.append(f"  <ansigreen>{label}</ansigreen>")
            lines.append("")
            lines.append("Write values after the command name: <ansigreen>st 0.35</ansigreen>, <ansigreen>check_ota preview</ansigreen>, or <ansigreen>update_sensing detector=lightweight motion_on_hits=4 motion_off_hits=3</ansigreen>.")
        else:
            lines.append("")
            lines.append("Device command names appear after the device answers MQTT <ansigreen>commands</ansigreen>.")
        lines.extend(
            [
                "",
                "<ansiyellow><b>Utility commands:</b></ansiyellow>",
                "  <ansigreen>about|a</ansigreen>                             Show shell information",
                "  <ansigreen>clear|cls</ansigreen>                           Clear screen",
                "  <ansigreen>help|h</ansigreen>                              Show this help message",
                "  <ansigreen>exit|quit|q</ansigreen>                         Exit interactive mode",
            ]
        )
        print()
        print_formatted_text(HTML("\n".join(lines)))
        print()

    def show_about(self):
        """Display a compact shell summary."""
        print()
        print("ESPectre MQTT Shell")
        print("Interactive MQTT control surface for ESPectre Protocol devices.")
        if self.device_id:
            print(f"Selected device: {self.device_id}")
        print()
