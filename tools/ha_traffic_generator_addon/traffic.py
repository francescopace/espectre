#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Read Home Assistant options and run the shared external traffic generator."""

import ipaddress
import json

if __package__:
    from .espectre_traffic_generator import DSCP, MULTICAST_TTL, run
else:
    from espectre_traffic_generator import DSCP, MULTICAST_TTL, run


def load_options(path="/data/options.json"):
    """Load and validate the Home Assistant add-on configuration."""
    with open(path, encoding="utf-8") as options_file:
        options = json.load(options_file)

    if not isinstance(options["targets"], list):
        raise ValueError("targets must be a list of IPv4 addresses")
    targets = []
    for raw_target in options["targets"]:
        target = ipaddress.ip_address(str(raw_target).strip())
        if target.version != 4:
            raise ValueError("targets must contain only IPv4 addresses")
        if (target.is_unspecified or target.is_loopback or target.is_link_local
                or target.is_reserved):
            raise ValueError(f"target is not routable: {target}")
        targets.append(str(target))
    if not targets:
        raise ValueError("at least one target is required")

    port = options["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be in the 1-65535 range")

    rate_pps = options["rate_pps"]
    if type(rate_pps) is not int or not 1 <= rate_pps <= 1000:
        raise ValueError("rate_pps must be in the 1-1000 range")

    multicast_ttl = options.get("multicast_ttl", MULTICAST_TTL)
    if type(multicast_ttl) is not int or not 1 <= multicast_ttl <= 255:
        raise ValueError("multicast_ttl must be an integer in the 1-255 range")

    dscp = options.get("dscp", DSCP)
    if type(dscp) is not int or not 0 <= dscp <= 63:
        raise ValueError("dscp must be an integer in the 0-63 range")

    source_ip = str(options.get("source_ip", "")).strip()
    if source_ip:
        source = ipaddress.ip_address(source_ip)
        if (source.version != 4 or source.is_unspecified or source.is_multicast
                or source.is_reserved):
            raise ValueError("source_ip must be a unicast IPv4 address")
        source_ip = str(source)

    return {"targets": targets, "port": port, "rate_pps": rate_pps,
            "source_ip": source_ip or None, "multicast_ttl": multicast_ttl, "dscp": dscp}


def main():
    run(**load_options())


if __name__ == "__main__":
    main()
