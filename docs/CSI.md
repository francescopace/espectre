# CSI acquisition and traffic

The shared C++ sensing runtime obtains channel state information (CSI) from Wi-Fi frames, validates the capture, and normalizes it before passing samples to the detector or raw collection.

Use [SDK.md](SDK.md#shared-sensing-options) for configuration defaults and ranges, [API.md](API.md#external-csi-traffic) for external traffic and collection formats, and [ALGORITHMS.md](ALGORITHMS.md#detector-timing) for detector timing.

## Traffic sources

`csi_traffic_mode` selects whether the device generates traffic (`internal`) or receives it from an external source (`external`). `csi_target_pps` sets the managed-traffic target and detector slot cadence independently of that selection. It never enables or disables traffic.

The runtime preserves an explicit source selection across ordinary delivery problems. Occupancy is diagnostic and never changes the target or selects a fallback protocol.

### Internal generators

| Mode | Traffic | Destination requirement |
|------|---------|-------------------------|
| `ping` | ICMP Echo Requests | A host that replies to ping |
| `dns` | Connectionless DNS root queries over UDP | A resolver on port `53` |
| `dns_tcp` | Length-prefixed DNS queries over a persistent, non-blocking TCP connection | A resolver that accepts TCP queries on port `53` |
| `wifi_raw` | Experimental 802.11 Null Data frames | The associated AP, with usable ACK CSI on the selected device and driver |

The default is `ping`. The IP-based generators use `traffic_generator_target_ip`, or the current Wi-Fi gateway when it is empty. The runtime uses the same resolved address for sending and identifying IP responses, and refreshes it after reconnection. [SDK.md](SDK.md#traffic-destination) describes address validation and startup configuration.

The build-time station TX rate defaults to HT20 MCS0 with long GI (6.5 Mbps) on classic ESP32 and Auto on other targets. `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` accepts `"0"` (Auto), `"6"`, or `"6.5"`; see [SDK.md](SDK.md#shared-sensing-options). This rate applies to station traffic, including Direct and MQTT, independently of sensing or traffic mode. It does not set the AP's downlink rate or change the requested packet cadence.

A fixed 6.5-Mbps rate requires an 802.11n AP. Fixed OFDM 6 Mbps requires a 5 GHz AP or support for 802.11g or 802.11n. Firmware uses automatic rates when the AP lacks the selected PHY or TX A-MPDU is enabled, and reevaluates the choice after association or roaming. The AP must also allow the selected rate in its configuration. A driver error prevents connected services from starting and is logged; it does not select another rate automatically. Micro uses the same policy; see its [README.md](../src/python/micro_espectre/README.md#csi-acquisition).

`wifi_raw` sends Null Data frames to the associated AP at legacy OFDM 6 Mbps and obtains CSI from its ACKs. The configured IP destination has no effect on this mode. The target BSSID is refreshed after roaming, even when IP and channel stay the same. A rejected raw-rate setting prevents this generator from starting. See [Compatibility limits](#compatibility-limits) before selecting it.

### External sources

In `external` mode, the internal generator stops. The ESP-IDF frontends accept UDP markers or unicast ICMP Echo Requests addressed to the device. The host owns pacing. UDP can use unicast or the configured multicast group; the listener joins that group unless the setting is empty.

For continuous external traffic on 64-bit Home Assistant OS, use the **ESPectre Traffic Generator** add-on. It runs the shared UDP generator and provides an Ingress panel for traffic ownership and automatically updated diagnostics through existing ESPHome or Native MQTT entities in Home Assistant. Set each device to `csi_traffic_mode: external` and match the add-on's `rate_pps` to the device's `csi_target_pps`; mode changes require an explicit action. See [DOCS.md](../tools/ha_traffic_generator_addon/DOCS.md) for installation, requirements, and configuration. Stop the add-on before running `./espectre collect` for the same devices, because collection starts its own generator.

The host UDP generator defaults to multicast TTL 8, allowing up to seven router hops when multicast forwarding is configured in the network. Set `multicast_ttl` to 1 to keep multicast local; Layer 2 switches do not consume TTL. Unicast uses the operating system's normal TTL. See [DOCS.md](../tools/ha_traffic_generator_addon/DOCS.md#switches-vlans-and-routed-networks) for cross-VLAN setup and interface selection.

Subnet and limited broadcast do not produce reliable HT20 CSI. Use the destination, port, and marker contract in [API.md](API.md#external-csi-traffic). Host generation and collection workflows are in [CLI.md](CLI.md#collect).

### Pacing

The device and host generators pace traffic at the configured rate and avoid bursts after scheduling delays. Delivery errors can reduce the observed rate; the runtime keeps your selected target and traffic source.

Internal IP traffic requests DSCP 46 treatment. The host UDP generator uses the same default, with a configurable `dscp` codepoint from 0 to 63; see [DOCS.md](../tools/ha_traffic_generator_addon/DOCS.md#advanced-dscp-marking). A particular WMM priority or improvement in delivery is not guaranteed. [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md) records the pacing decision and measurements.

## Wi-Fi and capture lifecycle

Sensing starts after Wi-Fi has a usable IPv4 address and stops on disconnect. Reconnection or roaming restarts the traffic source and refreshes CSI capture. A roaming transition can retain its address; firmware restores sensing, discovery, and Direct services after checking the current association, once the retained address is available. Initial connections and ordinary reconnects wait for a valid address.

If startup traffic continues without CSI callbacks, the runtime can attempt one recovery scan on the associated channel. If usable CSI still does not arrive, follow [TROUBLESHOOTING.md](TROUBLESHOOTING.md#no-csi-or-insufficient-input).

### Capture profiles

The C++ runtime accepts a build-time `csi_capture_profile` selection through ESPHome YAML, the shared Kconfig capture-profile choice, or `RuntimeConfig`: `auto`, `lltf`, or `ht-vht`. The default `auto` follows the policy below. `lltf` always selects LLTF20. `ht-vht` selects VHT20 on a VHT-capable 5 GHz link and HT20 otherwise, including on ESP32 and ESP32-S2; it also supports automatic band selection on ESP32-C5. `wifi_raw` requires `auto` or `lltf`, including when selected through existing runtime traffic controls. A persisted source incompatible with the configured profile is ignored at startup, preserving the configured source. The configured policy is not persisted or writable at runtime. The read-only `csi_profile` diagnostic reports the effective physical profile: `lltf20`, `ht20`, or `vht20`.

The frontend or SDK integrator explicitly selects `2g`, `5g`, or `auto`; `2g` is the validated band, while `5g` and `auto` are available only on dual-band targets. The lifecycle applies that band mode first and pins 20 MHz bandwidth on the selected band or bands. With `auto`, the runtime selects `lltf20` after association for internal `wifi_raw`, `vht20` on a VHT-capable 5 GHz link, and `ht20` otherwise, including on ESP32 and ESP32-S2; HE capture remains disabled. The shared C++ `lltf20` capture profile enables ACK dumping and admits valid 802.11 ACKs addressed to the local station, independently of the selected traffic generator; IP traffic retains its configured provenance filter. Other capture profiles reject ACKs. See [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

Published ESP32-C5 firmware defaults to automatic band selection; single-band targets use 2.4 GHz. Detection quality on 5 GHz remains uncharacterized.

Selecting internal `wifi_raw` switches to LLTF20 and ACK capture. A runtime source change stops the generator and reconfigures CSI only if the effective capture profile changes; Wi-Fi stays associated. With `auto`, leaving `wifi_raw` restores the chip/band profile, including when traffic ownership changes to `external`; explicit `lltf` keeps LLTF20 selected. Profile changes clear pending samples and detector history. Active traffic-mode changes recalibrate Lightweight while preserving High Accuracy thresholds. Explicit `lltf` keeps the same capture profile across generator changes on every supported chip.

First-party firmware captures CSI while associated with a Wi-Fi access point and keeps promiscuous mode disabled. Use a network you are authorized to access; having credentials does not establish consent to sensing.

## Capture quality

Packets with hardware-quality errors are rejected before sensing or raw collection. A nonzero `rx_state` rejects a packet on every supported chip. C5 and C6 also reject nonzero `rxend_state` or a cleared `rx_channel_estimate_info_vld`. Per-reason counters are available in C++ frontend diagnostics. These counters include background traffic and do not identify which generator caused an error.

`first_word_invalid` marks the first four source bytes. Full-width frames can still be used when their layout is known: the affected pairs appear as zeros in raw output, which retains the flag. In classic ordering, the affected bins are DC and physical subcarrier +1; raw-data consumers must treat +1 as missing. The detector fills that missing tone from +2 in its private input. Compact or ambiguous flagged frames are rejected.

A high callback rate does not establish usable CSI input. Check accepted packets, temporal occupancy, and sensing readiness; without enough valid CSI, sensing is unavailable. Follow the diagnostic sequence in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#no-csi-or-insufficient-input).

## Normalization

Production detectors consume a canonical centered 64-subcarrier, 20 MHz view. The runtime admits the named `lltf20`, `ht20`, and `vht20` capture profiles and normalizes recognized layouts onto that view. The current detection corpus validates only 2.4 GHz HT20 with HT-LTF; 5 GHz VHT20 detection quality is not characterized yet. HE20 and wider layouts are not accepted. The PHY rationale lives in [2026-07-23-adopt-classifier-first-ht20-sensing-contract.md](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

Supported HT20 payload variants are normalized onto the same internal 64-subcarrier index grid before fixed-subcarrier extraction. Short estimates are centered so the HT20 midpoint remains aligned, and doubled payloads are collapsed to one HT20 half.

| Input case | Raw layout | Mapping to HT20 | Output |
|------------|------------|-----------------|--------|
| Native HT20 | `128 B = 64 SC` | pass-through | `64 SC / 128 B` |
| Short HT estimate | `114 B = 57 SC` | zero-pad `4` SC left, copy `57` SC, zero-pad `3` SC right | `64 SC / 128 B` |
| Double HT20 payload | `256 B = 2 x 64 SC` | collapse to one `128 B` half | `64 SC / 128 B` |
| Double short HT estimate | `228 B = 2 x 57 SC` | collapse to one `57 SC` half, then pad `4` left and `3` right | `64 SC / 128 B` |

Normalization supports compact 106-byte LLTF estimates (53 signed 8-bit I/Q pairs) as well as full-width LLTF. Compact estimates require legacy LLTF admission. Their centered ordering is `-26..+26`, with DC at pair 26. Normalization pads six bins on the left and five on the right, placing DC at bin 32 in the existing 128-byte payload and leaving absent bins zero-filled. The detector applies its separate LLTF edge-tone imputation.

This mapping accepts 8-bit components; it does not decode packed 12-bit samples. C5 LLTF capture selects 8-bit mode. The ordering was confirmed on C5 hardware; evidence and the limits of that observation are in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation).

## Detector input and raw collection

Sensing and raw collection accept packets from the selected traffic source after capture validation. The detector uses a private normalized view: missing LLTF edge tones are filled from -26/+26, and an invalid classic +1 tone is filled from +2. Valid zero values are retained. C++ and Python use the same preparation for calibration and detection.

Raw collection preserves the normalized data and hardware flags before those detector-specific replacements. It also keeps accepted packets that exceed the detector's temporal sampling cadence. A raw record holds at most 128 CSI bytes. See [API.md](API.md#csi-collection) for framing and drop counters, [ALGORITHMS.md](ALGORITHMS.md#detector-timing) for detector timing, and [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) for collection instructions.

## Compatibility limits

`wifi_raw` is experimental and unavailable on ESP32-C6 in every frontend. Firmware ignores an older saved `wifi_raw` selection on C6 and keeps the configured source. Use `ping`, `dns`, or `dns_tcp` instead.

ESP32-C6 revision 0.1 with ESP-IDF 5.5.5 produced invalid ACK CSI estimates, tracked in [esp-idf#19062](https://github.com/espressif/esp-idf/issues/19062). The restriction remains in place for all C6 revisions until a working configuration is validated.

The hardware trials, including frame-length and PHY comparisons, remain in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation). Detector accuracy and benchmark results are tracked separately in [README.md](performance/README.md).
