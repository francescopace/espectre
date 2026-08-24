# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Compatibility helpers for Streamer consumers of shared device discovery."""

from __future__ import annotations

from .common import Fore, Style
from .device_discovery import (
    COLLECT_DISCOVERY_QUIET_WINDOW_S,
    DISCOVERY_TIMEOUT_S,
    ESPECTRE_SERVICE_TYPE,
    DeviceDiscoveryError,
    DiscoveredDevice,
    _DeviceListener,
    choose_device_interactively,
    discover_devices,
    print_device_list,
)

STREAMER_SERVICE_TYPE = ESPECTRE_SERVICE_TYPE


StreamerDiscoveryError = DeviceDiscoveryError
StreamerDiscoveryRecord = DiscoveredDevice
_StreamerListener = _DeviceListener


def discover_streamer_devices(timeout_s: float = DISCOVERY_TIMEOUT_S) -> list[StreamerDiscoveryRecord]:
    return discover_devices(
        frontend="streamer",
        timeout_s=timeout_s,
        quiet_window_s=COLLECT_DISCOVERY_QUIET_WINDOW_S,
    )


def choose_streamer_device_interactively(records: list[StreamerDiscoveryRecord]) -> StreamerDiscoveryRecord:
    return choose_device_interactively(records, frontend_label="Streamer")


def print_streamer_device_list(records: list[StreamerDiscoveryRecord]) -> None:
    if not records:
        print()
        print(f"{Fore.YELLOW}No Streamer devices discovered via mDNS.{Style.RESET_ALL}")
        return
    print_device_list(records, heading="Discovered Streamer devices")
