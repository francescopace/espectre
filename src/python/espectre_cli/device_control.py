# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""CLI handlers for Improv Serial provisioning and Direct requests."""

from __future__ import annotations

import getpass
import json
import os

from .common import get_serial_port
from .device_discovery import choose_device_interactively, discover_devices
from .device_transport import (
    DEFAULT_DIRECT_ORIGIN,
    DirectClient,
    ImprovSerialClient,
    direct_endpoint_from_device_url,
)


def run_improv_provision_command(args) -> int:
    password = os.environ.get(args.password_env)
    if password is None:
        password = getpass.getpass(f"Wi-Fi password ({args.password_env} is unset): ")
    port = get_serial_port(args.port)
    try:
        with ImprovSerialClient(port) as client:
            result = client.provision(args.ssid, password, timeout=args.timeout)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Improv provisioning failed: {exc}")
        return 1
    print("Improv provisioning completed.")
    print(f"Device endpoint: {direct_endpoint_from_device_url(result.endpoint)}")
    return 0


def _resolve_direct_endpoint(args) -> str:
    if args.endpoint:
        return direct_endpoint_from_device_url(args.endpoint)
    records = discover_devices(frontend=args.frontend, timeout_s=args.discovery_timeout)
    if not records:
        raise RuntimeError("no matching Direct device was discovered")
    return choose_device_interactively(records, frontend_label=args.frontend).endpoint


def run_direct_request_command(args) -> int:
    try:
        params = json.loads(args.params)
        if not isinstance(params, dict):
            raise ValueError("--params must decode to a JSON object")
        endpoint = _resolve_direct_endpoint(args)
        with DirectClient(endpoint, origin=args.origin, timeout=args.timeout) as client:
            result = client.request(args.method, params)
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"Direct request failed: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_DIRECT_ORIGIN",
    "run_direct_request_command",
    "run_improv_provision_command",
]
