# CSI acquisition and traffic

ESPectre senses motion from channel state information (CSI): a per-packet measurement of how the Wi-Fi signal travels through the room. The shared C++ runtime needs a steady flow of packets to measure. It captures CSI from those packets, checks its quality, and converts it to one common layout for the detector.

For settings and defaults, see [shared sensing options](SDK.md#shared-sensing-options). For external traffic and data formats, see [external CSI traffic](API.md#external-csi-traffic). For timing, see [detector timing](ALGORITHMS.md#detector-timing).

## Traffic sources

`csi_traffic_mode` chooses who sends the traffic: the device (`internal`) or another host (`external`). `csi_target_pps` sets the packet rate and the detector timing in both cases.

The runtime never switches source on its own. Low occupancy is reported as a diagnostic; it does not change the rate or the source.

### Internal generators

| Mode | Traffic | Destination must |
|------|---------|------------------|
| `ping` (default) | ICMP Echo Requests | Reply to ping |
| `dns` | DNS root queries over UDP | Be a resolver on port `53` |
| `dns_tcp` | DNS queries over a persistent TCP connection | Accept TCP queries on port `53` |
| `wifi_raw` | Experimental 802.11 Null Data frames | Be the access point, with usable ACK CSI on this chip |

`ping`, `dns`, and `dns_tcp` send to `traffic_generator_target_ip`, or to the Wi-Fi gateway when it is empty. The address is refreshed after every reconnect. See [traffic destination](SDK.md#traffic-destination) for valid values.

`wifi_raw` sends Null Data frames to the access point at 6 Mbps and measures CSI on the ACKs that come back. If the driver rejects that rate, `wifi_raw` does not start. It ignores the IP destination and follows the access point after roaming. Check the [compatibility limits](#compatibility-limits) before using it.

#### Transmit rate

`CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` is a build-time setting for all station traffic, including Direct and MQTT:

| Value | Rate | Requires |
|-------|------|----------|
| `"0"` | Automatic (default on every chip except classic ESP32) | — |
| `"6.5"` | HT20 MCS0, long GI (default on classic ESP32) | An 802.11n access point |
| `"6"` | Legacy OFDM | An 802.11g/n access point, or 5 GHz |

If the access point does not support the selected rate, or TX A-MPDU is enabled, the firmware falls back to automatic rates. It checks again after each connection or roam. If the driver rejects the rate, connected services do not start and the error is logged. The setting does not change the access point's rate or the packet rate. Micro-ESPectre follows the same rules; see its [CSI acquisition](../src/python/micro_espectre/README.md#csi-acquisition) section.

### External sources

In `external` mode the internal generator stops, and another host sends the packets and sets the pace. The ESP-IDF frontends accept UDP markers, by unicast or on the configured multicast group, and unicast ICMP Echo Requests addressed to the device. An empty multicast group disables the multicast join.

On 64-bit Home Assistant OS, the **ESPectre Traffic Generator** add-on sends this traffic and adds a panel to switch ESPHome and Native MQTT devices between internal and external mode. Match its `rate_pps` to the device's `csi_target_pps`. Stop the add-on before running `./espectre collect` on the same devices, because collection starts its own generator. See the [add-on documentation](../tools/ha_traffic_generator_addon/DOCS.md).

The host generator sends multicast with TTL 8, so it can cross up to seven routers where multicast routing is set up. Set `multicast_ttl` to 1 to keep it on the local network. See [switches, VLANs, and routed networks](../tools/ha_traffic_generator_addon/DOCS.md#switches-vlans-and-routed-networks).

Broadcast traffic does not give reliable HT20 CSI; use unicast or multicast. Ports and markers are defined in [external CSI traffic](API.md#external-csi-traffic), and host commands in [`collect`](CLI.md#collect).

### Pacing

The device and host generators send packets at an even rate and avoid bursts after delays. Delivery errors can lower the measured rate, but the runtime keeps your target and source.

IP traffic is marked with DSCP 46 by default. The host generator lets you set another value from 0 to 63; see [DSCP marking](../tools/ha_traffic_generator_addon/DOCS.md#advanced-dscp-marking). Better delivery is not guaranteed. The measurements behind this choice are in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md).

## Wi-Fi and capture lifecycle

Sensing starts once Wi-Fi has an IPv4 address and stops on disconnect. After a reconnect or roam, the runtime restarts traffic and CSI capture. A roam that keeps the same address restores sensing, discovery, and Direct as soon as the new association is confirmed.

If traffic flows but no CSI arrives at startup, the runtime tries one recovery scan on the current channel. If CSI still does not arrive, see [no CSI or insufficient input](TROUBLESHOOTING.md#no-csi-or-insufficient-input).

ESPectre captures CSI only while connected to an access point and never uses promiscuous mode. Use it only on networks you are allowed to use; having the Wi-Fi password is not consent to sensing.

### Capture profiles

A capture profile decides which part of the Wi-Fi frame is measured. Set it at build time with `csi_capture_profile` in ESPHome YAML, the Kconfig option, or `RuntimeConfig::csi_capture_policy`. It cannot be changed at runtime. The read-only `csi_profile` diagnostic shows the profile in use.

| Setting | With internal `wifi_raw` | On a 5 GHz VHT link | Otherwise |
|---------|--------------------------|---------------------|-----------|
| `auto` (default) | `lltf20` | `vht20` | `ht20` |
| `lltf` | `lltf20` | `lltf20` | `lltf20` |
| `ht-vht` | not allowed | `vht20` | `ht20` |

- `wifi_raw` needs `auto` or `lltf`. A saved `wifi_raw` selection that does not match the profile is ignored at startup.
- `lltf20` also measures ACKs addressed to the device, whatever the traffic source. Other profiles drop ACKs.
- With `auto`, switching away from `wifi_raw` returns to `ht20` or `vht20`. With `lltf`, the profile never changes.
- A profile change clears pending samples and detector history. Wi-Fi stays connected.
- Changing the traffic mode restarts Lightweight calibration. High Accuracy keeps its threshold.

The band is set separately to `2g`, `5g`, or `auto`. `auto`, the default, uses the bands the radio has, so it means 2.4 GHz on single-band chips. `5g` needs a dual-band chip such as ESP32-C5. Published ESP32-C5 firmware uses `auto`. Bandwidth is always 20 MHz. Only 2.4 GHz is validated; 5 GHz detection quality has not been measured yet. The reasons for this design are in [2026-07-23-adopt-classifier-first-ht20-sensing-contract.md](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

## Capture quality

Packets flagged with hardware errors are dropped before sensing and collection:

- On every chip: a nonzero `rx_state`.
- On C5 and C6 also: a nonzero `rxend_state` or a cleared `rx_channel_estimate_info_vld`.

Per-reason counters are in the frontend diagnostics. They include background traffic, so they do not tell which source caused an error.

The `first_word_invalid` flag marks the first four bytes as invalid. Full-width frames with a known layout are kept: the affected values are zeros in raw output and the flag is preserved. In classic layout this hits the DC bin and subcarrier +1, so raw-data users must treat +1 as missing. The detector fills +1 from +2 on its own copy. Short or ambiguous flagged frames are dropped.

Many callbacks do not mean usable input. Check accepted packets, occupancy, and readiness as described in [check the sensing input](TROUBLESHOOTING.md#check-the-sensing-input).

## Normalization

The detector works on one layout: 64 subcarriers over 20 MHz, centered. The runtime accepts the `lltf20`, `ht20`, and `vht20` profiles and maps each known layout onto it. HE20 and wider channels are rejected. The current dataset validates only 2.4 GHz HT20.

| Input | Raw size | Mapping | Output |
|-------|----------|---------|--------|
| HT20 | `128 B = 64 SC` | unchanged | `64 SC / 128 B` |
| Short HT estimate | `114 B = 57 SC` | pad 4 left, copy 57, pad 3 right | `64 SC / 128 B` |
| Double HT20 | `256 B = 2 x 64 SC` | keep one half | `64 SC / 128 B` |
| Double short HT | `228 B = 2 x 57 SC` | keep one half, pad 4 left and 3 right | `64 SC / 128 B` |
| Compact LLTF | `106 B = 53 SC`, ordered `-26..+26` | pad 6 left, 5 right; DC lands on bin 32 | `64 SC / 128 B` |

Compact LLTF also needs legacy LLTF capture; full-width LLTF is accepted too. Missing bins stay zero. Samples must be 8-bit: packed 12-bit samples are not decoded, and C5 LLTF capture selects 8-bit mode. The LLTF ordering was confirmed on C5 hardware; see the [C5/C6 investigation](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation).

## Detector input and raw collection

Sensing and raw collection receive the same validated packets. The detector then works on its own copy, where missing LLTF edge tones are filled from -26/+26 and an invalid +1 tone from +2. Real zero values are kept. C++ and Python prepare data the same way.

Raw collection keeps the normalized data and hardware flags without those fills. It also keeps extra packets that the detector skips to hold its rate. Each record holds at most 128 CSI bytes. See [CSI collection](API.md#csi-collection) for framing and drop counters, and the [data collection guide](ML_DATA_COLLECTION.md) for how to collect.

## Compatibility limits

`wifi_raw` is experimental and not available on ESP32-C6 in any frontend. A saved `wifi_raw` selection on C6 is ignored. Use `ping`, `dns`, or `dns_tcp` instead.

On ESP32-C6 revision 0.1 with ESP-IDF 5.5.5, ACK CSI estimates were invalid ([esp-idf#19062](https://github.com/espressif/esp-idf/issues/19062)). The restriction applies to all C6 revisions until a working setup is validated.

The hardware trials are in the [C5/C6 investigation](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation). Detection results are in the [performance report](performance/README.md).
