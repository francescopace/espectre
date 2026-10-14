# CSI acquisition and traffic

ESPectre senses motion from channel state information (CSI): a per-packet measurement of how the Wi-Fi signal travels through the room. The shared C++ runtime needs a steady flow of packets to measure. It captures CSI from those packets, checks its quality, and converts it to one common layout for the detector.

For settings and defaults, see [shared sensing options](SDK.md#shared-sensing-options). For external traffic and data formats, see [external CSI traffic](API.md#external-csi-traffic). For timing, see [detector timing](ALGORITHMS.md#detector-timing).

## Traffic sources

`traffic_generator_mode` chooses who sends the traffic: the device, with one of the packet types below, or another host (`external`). `csi_target_pps` sets the packet rate and the detector timing in both cases.

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

In `external` mode the internal generator stops, and another host sends the packets and sets the pace. The ESP-IDF frontends accept UDP markers, by unicast or on the configured multicast group, unicast ICMP Echo Requests, direct non-QoS Null Data frames, and unencrypted unicast ESP-NOW frames addressed to the device. An empty multicast group disables the multicast join. See [external CSI traffic](API.md#external-csi-traffic) for frame validation and addressing.

An ESP32 transmitter can send Null Data or ESP-NOW directly to the receiver's station MAC on the same channel. These frames supply CSI on reception; they do not use the ACK capture path of internal `wifi_raw`. The default `auto` policy selects `ht20` on a 2.4 GHz link in `external` mode. Both sources work at HT20 MCS0 with this profile. Legacy OFDM at 6 Mbps instead requires a receiver built with the `lltf` capture policy. ESP-NOW's default 1 Mbps rate does not supply the required OFDM training field. The capture policy does not change automatically with the sender.

For direct Null Data at MCS0, configure the transmitter bandwidth explicitly with `esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20)`, and use `FF:FF:FF:FF:FF:FF` as the BSSID in address 3. Keep address 1 equal to the receiver's station MAC, address 2 equal to the transmitter's station MAC, and `ToDS = FromDS = 0`. The destination remains unicast. ESP-NOW uses the same [connectionless MAC addressing](https://docs.espressif.com/projects/esp-idf/en/v5.5.5/esp32c3/api-reference/network/esp_now.html#frame-format), and its per-peer PHY configuration selects HT20 independently of the interface's default bandwidth. Set the raw rate with `esp_wifi_config_80211_tx_rate()` before starting Wi-Fi, as specified by [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/v5.5.5/esp32c3/api-reference/network/esp_wifi.html).

The following acquisition checks used an ESP32-C3 transmitter, an ESP32-S3 Native receiver, ESP-IDF 5.5.5, and no promiscuous reception. They cover CSI acquisition; detector accuracy across environments remains unmeasured.

| Source | TX rate | Receiver profile | Result |
|--------|---------|------------------|--------|
| Unicast ESP-NOW v2 | HT20 MCS0 | `ht20` | 2,500 successful sends, approximately 100 accepted CSI samples/s, and zero CSI hardware errors |
| Unicast ESP-NOW v2, 300-byte payload | OFDM 6 Mbps | `lltf20` | 1,500 successful sends; multi-element frames admitted with zero CSI hardware errors |
| Direct non-QoS Null Data | OFDM 6 Mbps | `lltf20` | 1,500 successful sends, approximately 100 accepted CSI samples/s, and zero CSI hardware errors |
| Direct non-QoS Null Data, wildcard BSSID | HT20 MCS0 | `ht20` | 2,500 successful sends with the final transmitter, approximately 100 accepted CSI samples/s, and zero CSI hardware errors |

Both transmitter settings are needed on this C3-to-S3 setup: with a wildcard BSSID but the default HT40 interface bandwidth, all 1,500 sends failed; with explicit HT20 but the AP's BSSID, all 750 sends failed. With both settings corrected, Null Data supplies HT20 CSI without changing ESPectre's receiver profile or registering an ESP-NOW peer.

Raw Null Data collection delivered 996 HT20 records at MCS0 and 1,002 legacy LLTF20 records at 6 Mbps in separate runs of approximately 10 seconds. Every record was fresh, nonzero, and contained 64 subcarriers, with no sequence gaps, device queue drops, or stream backpressure.

On 64-bit Home Assistant OS, the **ESPectre Traffic Generator** add-on sends UDP marker traffic and adds a panel to switch ESPHome and Native MQTT devices between internal and external mode. Match its `rate_pps` to the device's `csi_target_pps`. Stop the add-on before running `./espectre collect` on the same devices, because collection starts its own generator. See the [add-on documentation](../tools/ha_traffic_generator_addon/DOCS.md).

The host generator sends multicast with TTL 8, so it can cross up to seven routers where multicast routing is set up. Set `multicast_ttl` to 1 to keep it on the local network. See [switches, VLANs, and routed networks](../tools/ha_traffic_generator_addon/DOCS.md#switches-vlans-and-routed-networks).

Broadcast traffic does not give reliable HT20 CSI; use unicast or multicast. Ports and markers are defined in [external CSI traffic](API.md#external-csi-traffic), and host commands in [`collect`](CLI.md#collect).

### Pacing

The device and host generators send packets at an even rate and avoid bursts after delays. Delivery errors can lower the measured rate, but the runtime keeps your target and source.

IP traffic is marked with DSCP 46 by default. The host generator lets you set another value from 0 to 63; see [DSCP marking](../tools/ha_traffic_generator_addon/DOCS.md#advanced-dscp-marking). Better delivery is not guaranteed. The measurements behind this choice are in [2026-08-23-standardize-managed-csi-traffic-sources.md](adr/2026-08-23-standardize-managed-csi-traffic-sources.md).

### Network impact and airtime

Sensing needs continuous packet exchanges while enabled. Small payloads still consume airtime through PHY preambles, MAC headers, and acknowledgments. Disabling Wi-Fi power save keeps the receiver awake and increases device power consumption; listening alone does not occupy the channel. Disabling A-MPDU affects aggregation on the ESP32's link, not aggregation between the AP and other clients.

The estimates below compare all traffic sources under one explicit radio model. They are calculated exchange budgets, not measured channel utilization, detector occupancy, or a guarantee for a particular network. They include PHY packet durations, the 2.4-GHz signal extension, and the SIFS interval before each Wi-Fi ACK. They exclude channel-access waiting time and unrelated traffic.

#### Calculation assumptions

- **Topology and cadence:** one AP on a 2.4-GHz, 20-MHz channel, with `100` generator cycles per second per device. Internal IP destinations and external IP senders are on the AP itself or its wired LAN. Direct Null Data and ESP-NOW senders transmit to the device without the AP. No additional wireless forwarding or mesh backhaul is counted. All requests succeed and receive the expected response.
- **PHY:** IP data frames in both directions use HT20 MCS0 at `6.5 Mbps`, with a mixed-format preamble, one spatial stream, BCC coding, long GI, and no STBC. Internal `wifi_raw` Null Data and all Wi-Fi ACKs use legacy OFDM at `6 Mbps`. Direct Null Data and ESP-NOW rows show both validated rates, legacy OFDM at `6 Mbps` and HT20 MCS0 at `6.5 Mbps`. This is a low-rate example, not the default rate of every link: see [transmit rate](#transmit-rate). The AP chooses its own downlink rate.
- **Framing:** IPv4 headers are `20 B`, UDP headers are `8 B`, and established TCP headers are `20 B`, without options. Unicast IP packets add `54 B`: a `26 B` QoS MAC header, `8 B` LLC/SNAP, `8 B` CCMP header, `8 B` CCMP MIC, and `4 B` FCS. True multicast uses a `24 B` non-QoS MAC header, so its overhead is `52 B`. There is no aggregation, fragmentation, RTS/CTS, CTS-to-self, or retransmission.
- **DNS:** the query is `17 B`. Each response is assumed to contain `R = 100 B` of DNS data, before UDP headers or the TCP length prefix. This is an illustrative response size, not a measured average or a fixed resolver behavior. DNS over TCP adds a `2 B` length prefix to each message. Each query and response occupies one TCP segment on an already established connection.
- **Acknowledgments:** each unicast data or Null Data frame gets one `14 B` Wi-Fi ACK after `10 µs` SIFS. True multicast gets no Wi-Fi ACK or application reply. TCP acknowledgments are separate from Wi-Fi ACKs: the TCP range allows zero to two additional standalone TCP ACK segments per query/response cycle, depending on piggybacking and delayed ACKs. Connection setup, teardown, and recovery are excluded.
- **External traffic:** UDP uses the current [four-byte marker](API.md#external-csi-traffic). The ICMP example uses an `8 B` Echo header with no extra payload, matching the internal ping size; a host ping tool may send a larger payload. The multicast row assumes one AP transmission reaches all devices at the stated HT rate, without conversion to unicast. Direct Null Data is a `28 B` frame without payload. Unencrypted ESP-NOW with `P` data bytes in one element is `43 + P B`: a `24 B` MAC header, an `8 B` action header, a `7 B` element header, the data, and a `4 B` FCS. The example uses `P = 1`.

#### Comparison at 100 cycles per second

Frame lengths below include the MAC header, encryption overhead where applicable, and FCS, but exclude PHY preambles and Wi-Fi ACK frames. `+` separates frames in one cycle. The airtime columns include their Wi-Fi ACKs and SIFS intervals. The ten-device column is an arithmetic projection on the same channel, not a validated deployment size.

| Source | Data or Null Data frame lengths per cycle | Exchange time per cycle | Airtime, 1 device | Airtime, 10 devices |
|--------|-----------------------------------------|-------------------------|-------------------|--------------------|
| Internal `ping` | `82 + 82 B` | `420 µs` | `4.20%` | `42.00%` |
| Internal `dns`, `R = 100 B` | `99 + 182 B` | `560 µs` | `5.60%` | `56.00%` |
| Internal `dns_tcp`, `R = 100 B` | `113 + 196 B`, plus `0–2 × 94 B` TCP ACKs | `596–1,040 µs` | `5.96–10.40%` | `59.60–104.00%` |
| Internal `wifi_raw` | `28 B` | `130 µs` | `1.30%` | `13.00%` |
| External UDP unicast | `86 B` per device | `214 µs` | `2.14%` | `21.40%` |
| External UDP true multicast | One shared `84 B` frame | `150 µs` | `1.50%` | `1.50%` |
| External ICMP unicast, no extra payload | `82 + 82 B` | `420 µs` | `4.20%` | `42.00%` |
| External direct Null Data, OFDM 6 Mbps or HT20 MCS0 | `28 B` | `130–142 µs` | `1.30–1.42%` | `13.00–14.20%` |
| External ESP-NOW, `P = 1`, OFDM 6 Mbps or HT20 MCS0 | `44 B` | `150–162 µs` | `1.50–1.62%` | `15.00–16.20%` |

A projection at or above `100%` cannot fit on the channel even before contention and other traffic. The TCP range describes the specified segment and ACK patterns, not a worst-case upper bound. Larger DNS responses, TCP options, split responses, and reconnects add traffic. `wifi_raw` has no IP, LLC/SNAP, or CCMP payload: its `24 B` Null Data header plus `4 B` FCS gets only a Wi-Fi ACK. Its smaller exchange does not remove the [chip and capture restrictions](#compatibility-limits).

#### Reproducing the estimates

For a frame of `L` bytes including FCS, use the following durations in microseconds. The formulas round up to whole OFDM symbols; dividing bytes by the nominal bit rate alone misses this padding and the preamble. The HT structure and coding fields are described in [HT PPDU structure](https://www.mathworks.com/help/wlan/gs/ht-ppdu-structure.html); the [ns-3 HT PHY model](https://github.com/nsnam/ns-3-dev-git/blob/master/src/wifi/model/ht/ht-phy.cc) and [OFDM PHY model](https://github.com/nsnam/ns-3-dev-git/blob/master/src/wifi/model/non-ht/ofdm-phy.cc) implement the corresponding duration calculations.

```text
T_HT(L)   = 36 + 4 × ceil((16 + 8 × L + 6) / 26) + 6
T_OFDM(L) = 20 + 4 × ceil((16 + 8 × L + 6) / 24) + 6
T_ACK     = T_OFDM(14) = 50
U(L)      = T_HT(L) + 10 + T_ACK

ping / external ICMP = 2 × U(82)
dns                 = U(99) + U(82 + R)
dns_tcp             = U(113) + U(96 + R) + k × U(94), k = 0..2
wifi_raw            = T_OFDM(28) + 10 + T_ACK
external UDP unicast = U(86)
external multicast  = T_HT(84)
direct Null Data    = T(28) + 10 + T_ACK
ESP-NOW             = T(43 + P) + 10 + T_ACK
T                   = T_OFDM at 6 Mbps, or T_HT at MCS0

airtime (%) = cycles_per_second × exchange_time_us / 10,000
```

The `36` and `20 µs` terms are the HT mixed-format and legacy preambles including signaling. Both PHYs use `4 µs` symbols here, carrying `26` and `24` data bits respectively; `16` service bits and `6` BCC tail bits are added before rounding. The final `6 µs` is the 2.4-GHz signal extension. For example, ping has `28 B` IP packets, or `82 B` MAC frames: `T_HT(82) = 150 µs`, giving `2 × (150 + 10 + 50) = 420 µs` per cycle and `4.20%` at `100` cycles/s. For external ICMP with `P` extra payload bytes, replace `82` with `82 + P` in both directions.

#### Applying the model to a network

Channel access also requires idle interframe spacing and backoff. For an isolated short-slot DCF example, DIFS is `28 µs` and the mean initial backoff with `CWmin = 15` is `7.5 × 9 = 67.5 µs`, adding `95.5 µs` per independently contended data frame. These are waiting times, not RF transmissions, and are excluded from the table. ESPectre's DSCP marking, AP-advertised WMM parameters, TXOPs, competing transmitters, and retries change access timing, so that DCF example is not a prediction of the active queues. See the [ns-3 channel-access model](https://www.nsnam.org/docs/models/html/wifi-design.html#channel-access).

Scale each per-device row by `rate / 100` and by the device count. True multicast scales with the number of AP transmissions, not the number of listeners: multicast-to-unicast conversion instead approaches the unicast row, and multiple APs on the same channel add transmissions. A Wi-Fi-connected external IP sender adds its uplink to the AP, while direct Null Data and ESP-NOW do not pass through it; same-channel mesh forwarding adds further airtime. A separate SSID on the same channel does not create more airtime.

Actual PHY and ACK rates, DNS response sizes, and retransmissions must be measured for a deployment estimate. Faster data rates reduce serialization time but leave preamble and acknowledgment costs. Multicast often uses a legacy basic rate; the HT multicast row applies only when the AP actually supplies usable HT CSI, and does not make broadcast a supported alternative. Compare channel utilization, retries, and existing sensor reliability with sensing enabled and disabled before expanding a congested network. Changing `csi_target_pps` also changes [detector timing](ALGORITHMS.md#detector-timing); reducing traffic requires validating detection again.

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
- `lltf20` measures an ACK only when it answers a frame that `wifi_raw` just sent, at most one per frame. ACKs for the device's own traffic, such as the CSI stream, are dropped in every mode. Other profiles drop all ACKs.
- An access point retransmission of a frame already measured is dropped, so a weak link cannot multiply the samples.
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

Direct Null Data and ESP-NOW acquisition in `external` mode is validated only with an ESP32-S3 receiver. Other receiver chips may report the frame body differently; ESPectre then rejects ESP-NOW frames instead of misreading them, so [check the sensing input](TROUBLESHOOTING.md#check-the-sensing-input) before relying on these sources.

The hardware trials are in the [C5/C6 investigation](adr/2026-08-23-standardize-managed-csi-traffic-sources.md#c5c6-short-frame-csi-investigation). Detection results are in the [performance report](performance/README.md).
