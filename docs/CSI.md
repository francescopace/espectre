# CSI acquisition and traffic

The sensing runtime obtains channel state information (CSI) from Wi-Fi frames, validates the capture, and normalizes it before passing samples to the detector or raw collection. This reference covers the shared C++ acquisition path and [Micro-ESPectre's acquisition differences](#micro-espectre-acquisition). Micro's deployment settings are documented in its [README.md](../src/python/micro_espectre/README.md).

[SDK.md](SDK.md#shared-sensing-options) owns configuration defaults and ranges. [API.md](API.md#external-csi-traffic) owns the external traffic contract and collection format; [ALGORITHMS.md](ALGORITHMS.md#detector-timing) owns temporal admission and detector processing.

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

`wifi_raw` sends 24-byte non-QoS Null Data frames using the station MAC, the associated AP BSSID, and driver-managed sequence numbers. It refreshes the BSSID when the generator restarts after association, including roaming with unchanged IP and channel. The configured IP destination has no effect on this mode.

### External sources

In `external` mode, the internal generator stops. The ESP-IDF frontends accept UDP markers or unicast ICMP Echo Requests addressed to the device. The host owns pacing. UDP can use unicast or the configured multicast group; the listener joins that group unless the setting is empty.

Subnet and limited broadcast do not produce reliable HT20 CSI. Use the destination, port, and marker contract in [API.md](API.md#external-csi-traffic). Host generation and collection workflows are in [CLI.md](CLI.md#collect).

### Pacing

The C++ generator, Micro-ESPectre native generator, and host UDP generator preserve the configured send phase through ordinary scheduler jitter. If the next deadline would fall less than half a period after a send, they restart the phase from that send time to avoid a close catch-up pair. Socket failures use local backoff; the runtime does not chase occupancy by changing the target.

Internal IP traffic and the host UDP generator request DSCP 46 treatment. A particular WMM priority or improvement in delivery is not guaranteed. [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md) records the pacing decision and measurements.

## Wi-Fi and capture lifecycle

`WiFiLifecycleManager` owns the CSI-specific ESP-IDF radio policy for every frontend. It applies the protocol and HT20 bandwidth policy synchronously on `WIFI_EVENT_STA_START`, before the first association, then completes the CSI prerequisites when `IP_EVENT_STA_GOT_IP` is drained from the runtime loop. ESPHome, Native, and Matter must not apply these radio settings in their frontend code.

Association changes are also drained by `WiFiLifecycleManager` from `WIFI_EVENT_STA_CONNECTED`. A reassociation while the session is active, or a roaming disconnect followed by association with valid retained IPv4 state, reuses the shared disconnect/connect callbacks to stop traffic, refresh the CSI receive path, and restart sensing. This also refreshes the `wifi_raw` BSSID target when IP and channel are unchanged, without requiring a new `GOT_IP` event. Initial association and ordinary reconnects still wait for `GOT_IP`; a later duplicate IP notification does not restart the session again.

The `GOT_IP` payload supplies the local address, netmask, and gateway used during service startup. An empty traffic destination follows that gateway; a configured destination overrides it for the internal IP generators and response filtering. Disconnect processing clears readiness, and the runtime repeats startup after reconnection.

`runtime/csi_traffic_service` owns the platform-neutral traffic policy: configuration projection, internal-versus-external selection, lifecycle, callbacks, and counters. It depends only on the `ICsiTrafficGenerator` and `ICsiTrafficIngress` boundaries. The ESP-IDF layer implements those boundaries with `TrafficGeneratorManager` and `UDPListener`; the latter delegates socket creation, binding, multicast membership, and datagram reads to `UdpDatagramSocketEspIdf`. This keeps lwIP and FreeRTOS below the shared runtime contract and lets host tests inject deterministic in-memory adapters.

### Capture profiles

The C++ runtime accepts a build-time `csi_capture_profile` selection through ESPHome YAML, the shared Kconfig capture-profile choice, or `RuntimeConfig`: `auto`, `lltf`, or `ht-vht`. The default `auto` follows the policy below. `lltf` always selects LLTF20. `ht-vht` selects VHT20 on a VHT-capable 5 GHz link and HT20 otherwise, including on ESP32 and ESP32-S2; it also supports automatic band selection on ESP32-C5. `wifi_raw` requires `auto` or `lltf`, including when selected through existing runtime traffic controls. A persisted source incompatible with the configured profile is ignored at startup, preserving the configured source. The configured policy is not persisted or writable at runtime. The read-only `csi_profile` diagnostic reports the effective physical profile: `lltf20`, `ht20`, or `vht20`.

The frontend or SDK integrator explicitly selects `2g`, `5g`, or `auto`; `2g` is the validated band, while `5g` and `auto` are available only on dual-band targets. The lifecycle applies that band mode first and pins 20 MHz bandwidth on the selected band or bands. With `auto`, the runtime selects `lltf20` after association for internal `wifi_raw` or on ESP32 and ESP32-S2, `vht20` on a VHT-capable 5 GHz link, and `ht20` otherwise; HE capture remains disabled. The shared C++ `lltf20` capture profile enables ACK dumping and admits valid 802.11 ACKs addressed to the local station, independently of the selected traffic generator; IP traffic retains its configured provenance filter. Other capture profiles reject ACKs. Fixed-band policies use the single-band ESP-IDF APIs, while AUTO uses the per-band APIs. See [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

Published ESP32-C5 firmware defaults to automatic band selection; single-band targets use 2.4 GHz. Detection quality on 5 GHz remains uncharacterized.

Selecting internal `wifi_raw` switches to LLTF20 and ACK capture. A runtime source change stops the generator and reconfigures CSI only if the effective capture profile changes; Wi-Fi stays associated. With `auto`, leaving `wifi_raw` restores the chip/band profile, including when traffic ownership changes to `external`; explicit `lltf` keeps LLTF20 selected. Profile changes clear pending samples and detector history. Active traffic-mode changes recalibrate Lightweight while preserving High Accuracy thresholds. With `auto` or `lltf`, ESP32 and ESP32-S2 already use LLTF20, so switching their generator does not cycle CSI.

Supported first-party firmware uses an associated Wi-Fi station and does not enable promiscuous mode. Standalone ESP-IDF startup explicitly keeps promiscuous mode disabled, and the shared CSI pipeline filters frames against the local device identity where the relevant metadata is available. This is an intentional responsible-use boundary: a protected network requires valid credentials, which raises the barrier against passive collection by an unaffiliated device. It is not an authorization mechanism or proof of consent; open networks need no password, credentials can be misused, and downstream open-source builds can change the radio policy.

## Capture quality

Capture validates hardware quality before normalization, calibration, sensing, or collection, independently of the traffic generator. A nonzero `rx_state` rejects the packet on every supported chip. On HE-capable chips (C5/C6), a nonzero `rxend_state` or a cleared `rx_channel_estimate_info_vld` also rejects it. Unsupported metadata fields are not read on classic chips. Hardware estimate length is not used as a validity gate; payload length and layout validation still apply to the original CSI buffer.

MicroPython includes native hardware-quality rejections in `csi_filtered_total` and `csi_filtered_pps`, and counts those callbacks in `csi_callbacks_total`. The dedicated native `csi_hw_errors()` counter counts only hardware-quality rejections, including faults accompanied by malformed buffers, once per callback. Buffer-only failures remain in the overall filtered count, and native ring overflow stays separate. Per-reason quality counters are available on the C++ frontends.

`first_word_invalid` identifies the first four source bytes, not the first four normalized bytes. A frame is retained only when its full-width centered ordering can be identified independently of those bytes: the invalid guard pairs are zeroed in a private buffer. Compact, classic-order, and ambiguous flagged frames are rejected because their invalid pairs may affect live tones. No replacement live tones are synthesized. Quality-drop counters cover all capture callbacks, including background traffic, and do not by themselves identify generator-specific failures. Quality errors take priority over missing or malformed payloads and do not participate in the format-drop reset streak; valid-stream gaps remain subject to temporal admission. RSSI and sequence metadata are not additional validity gates.

Missing valid CSI makes sensing unavailable. A high callback rate does not establish usable input, and collection traffic itself can generate additional ACKs. Compare accepted input, temporal occupancy, and readiness when evaluating a generator. [TROUBLESHOOTING.md](TROUBLESHOOTING.md#no-csi-or-insufficient-input) gives the diagnostic sequence.

## Normalization

Production detectors consume a canonical centered 64-subcarrier, 20 MHz view. The runtime admits the named `lltf20`, `ht20`, and `vht20` capture profiles and normalizes recognized layouts onto that view. The current detection corpus validates only 2.4 GHz HT20 with HT-LTF; 5 GHz VHT20 detection quality is not characterized yet. HE20 and wider layouts are not accepted. The PHY rationale lives in [2026-07-23-adopt-classifier-first-ht20-sensing-contract.md](adr/2026-07-23-adopt-classifier-first-ht20-sensing-contract.md).

Supported HT20 payload variants are normalized onto the same internal 64-subcarrier index grid before fixed-subcarrier extraction. Short estimates are centered so the HT20 midpoint remains aligned, and doubled payloads are collapsed to one HT20 half.

| Input case | Raw layout | Mapping to HT20 | Output |
|------------|------------|-----------------|--------|
| Native HT20 | `128 B = 64 SC` | pass-through | `64 SC / 128 B` |
| Short HT estimate | `114 B = 57 SC` | zero-pad `4` SC left, copy `57` SC, zero-pad `3` SC right | `64 SC / 128 B` |
| Double HT20 payload | `256 B = 2 x 64 SC` | collapse to one `128 B` half | `64 SC / 128 B` |
| Double short HT estimate | `228 B = 2 x 57 SC` | collapse to one `57 SC` half, then pad `4` left and `3` right | `64 SC / 128 B` |

The shared C++ and Python normalizers support compact 106-byte LLTF estimates (53 signed 8-bit I/Q pairs) as well as full-width LLTF. Compact estimates require legacy LLTF admission. Their centered ordering is `-26..+26`, with DC at pair 26. Normalization pads six bins on the left and five on the right, placing DC at bin 32 in the existing 128-byte payload and leaving absent bins zero-filled. The detector applies its separate LLTF edge-tone imputation.

This mapping accepts 8-bit components; it does not decode packed 12-bit samples. C5 LLTF capture selects 8-bit mode. The ordering was confirmed on C5 hardware; evidence and the limits of that observation are in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation).

## Detector input and raw collection

The CSI callback classifies packet provenance against the traffic mode before data reaches sensing or collection. Accepted sensing frames pass through `TemporalCsiSampler`. Accepted raw frames bypass temporal sampling and enter a preallocated SPSC ring drained by a dedicated task-notified HTTP worker. Raw collection therefore retains packets that the detector may classify as excess within a temporal slot.

[ALGORITHMS.md](ALGORITHMS.md#detector-timing) defines admission, window geometry, and readiness. [API.md](API.md#csi-collection) defines collection sessions, framing, and observable drop accounting; [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) covers dataset collection.

## Micro-ESPectre acquisition

Micro-ESPectre reuses the native managed `ping`, `dns`, and `dns_tcp` generators. It does not expose `wifi_raw`, runtime traffic changes, a UDP marker listener, or multicast membership. `TRAFFIC_GENERATOR_TARGET_IP` applies the destination rules in [SDK.md](SDK.md#traffic-destination); an empty value follows the gateway. `TRAFFIC_GENERATOR_ENABLED = False` requires an external source of Wi-Fi traffic without the managed UDP ingress used by the C++ frontends.

Micro requests automatic band selection when the MicroPython station API supports it and otherwise configures 2.4 GHz. Its profile selector uses `lltf20` on ESP32 and ESP32-S2, `vht20` on a 5 GHz ESP32-C5 association, and `ht20` otherwise. The same limits on uncharacterized 5 GHz sensing apply. Classic-MAC payloads are rotated from Espressif's `0~31, -32~-1` ordering to the centered convention before detector processing.

`CSI_BUFFER_SIZE` sets the number of native ring records. `CSI_CAPTURE_MAX_DATA_LEN` sets their fixed stride: the default `256` bytes accommodates doubled HT20 payloads. A `128`-byte stride is suitable only when every captured frame fits the canonical payload; larger frames are truncated before normalization. Normalization and temporal admission then prepare the detector input as described in [ALGORITHMS.md](ALGORITHMS.md#detector-timing).

When no frame arrives for `CSI_LINK_RECOVERY_TIMEOUT_MS`, Micro first rearms CSI. If the stall persists, it stops Direct, reconnects Wi-Fi, creates a fresh capture ring, recalibrates, and republishes discovery. This recovery is implemented in [runtime_main.py](../src/python/micro_espectre/runtime_main.py) and [wifi_bootstrap.py](../src/python/micro_espectre/wifi_bootstrap.py).

## Compatibility limits

`wifi_raw` is experimental. Some device, driver, and AP combinations produce usable ACK CSI; others do not. The tested ESP32-C6 revision 0.1 with ESP-IDF 5.5.5 produced invalid ACK estimates, while the paired C5 control produced valid estimates. Use `ping` for that failing C6 configuration. This result does not establish behavior on other revisions or SDK releases.

The hardware trials, including frame-length and PHY comparisons, remain in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation). Detector accuracy and benchmark results are tracked separately in [README.md](performance/README.md).
