# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""CLI handlers for Improv Serial provisioning and Direct requests."""

from __future__ import annotations

import getpass
import json
import os
import sys
from contextlib import nullcontext, redirect_stdout

from .common import resolve_serial_port
from .device_discovery import discover_devices, select_discovered_device
from .device_transport import (
    DEFAULT_DIRECT_ORIGIN,
    DirectClient,
    ImprovSerialClient,
    direct_endpoint_from_device_url,
)


def run_improv_provision_command(args) -> int:
    json_output = bool(getattr(args, "json", False))
    diagnostic_stream = sys.stderr if json_output else sys.stdout
    chip = getattr(args, "chip", None)
    frontend = getattr(args, "frontend", "native")
    try:
        output_context = redirect_stdout(diagnostic_stream) if json_output else nullcontext()
        with output_context:
            port = resolve_serial_port(
                args.port,
                chip=chip,
                frontend=frontend,
                purpose="improv",
            )
        password = os.environ.get(args.password_env)
        if password is None:
            password = getpass.getpass(f"Wi-Fi password ({args.password_env} is unset): ")
        with ImprovSerialClient(port) as client:
            result = client.provision(args.ssid, password, timeout=args.timeout)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Improv provisioning failed: {exc}", file=diagnostic_stream)
        return 1
    endpoint = direct_endpoint_from_device_url(result.endpoint)
    if json_output:
        print(
            json.dumps(
                {
                    "chip": chip,
                    "device_info": list(result.device_info),
                    "endpoint": endpoint,
                    "frontend": frontend,
                    "port": port,
                    "states": list(result.states),
                },
                sort_keys=True,
            )
        )
        return 0
    print("Improv provisioning completed.")
    print(f"Device endpoint: {endpoint}")
    return 0


def _resolve_direct_endpoint(args) -> str:
    if args.endpoint:
        if getattr(args, "chip", None):
            raise ValueError("--chip requires --frontend discovery")
        return direct_endpoint_from_device_url(args.endpoint)
    records = discover_devices(frontend=args.frontend, timeout_s=args.discovery_timeout)
    return select_discovered_device(
        records,
        frontend_label=args.frontend,
        chip=getattr(args, "chip", None),
    ).endpoint


def run_direct_request_command(args) -> int:
    try:
        params = json.loads(args.data)
        if not isinstance(params, dict):
            raise ValueError("--data must decode to a JSON object")
        endpoint = _resolve_direct_endpoint(args)
        with DirectClient(endpoint, origin=args.origin, timeout=args.timeout) as client:
            if args.http_method != "get" or args.resource.strip("/") != "capabilities":
                client.negotiate()
            result = client.request(args.http_method, args.resource, params)
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
