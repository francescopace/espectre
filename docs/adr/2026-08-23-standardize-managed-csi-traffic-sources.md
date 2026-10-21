# ADR: standardize managed CSI traffic sources and their admission

- Status: Accepted
- Date: 2026-08-23
- Updated: 2026-10-21

## Context

Motion detection needs a steady supply of HT20 or LLTF packets to fill its fixed temporal grid. Diagnostics kept showing average occupancy around 85% even when the generator reported about 100 pps. The raw callback rate cannot tell missing packets apart from AP bursts, device send bursts, retries, legacy-PHY delivery, same-slot excess, or processing backlog.

ESPectre can drive traffic in several ways: internal ICMP ping, internal DNS over UDP or TCP, raw Wi-Fi Null Data that makes the AP (or a peer) answer with ACKs, external UDP through the AP, and unicast ICMP Echo Requests or direct frames from another host. These paths behave differently in protocol state, QoS, chip-specific Wi-Fi handling, and missed-deadline behavior.

Other local traffic produces valid CSI too: Direct HTTP requests, SSE clients, and ambient packets. Whether that incidental traffic could stand in for part of the generator was tested on an ESP32-S3 and rejected (see Evidence).

Packet time also needs a precise meaning. The PHY estimates CSI from the training field when the frame arrives. Later callback or loop handling does not move that observation in time, so the grid must use the Wi-Fi RX timestamp, and software clocks can only measure processing backlog.

## Decision

**Generators.** Internal traffic has explicit modes, with no automatic fallback between them:

- `ping`: stateless ICMP echo. It is the schema default and the default in published Native, Matter, ESPHome, and Micro-ESPectre configurations.
- `dns`: connectionless UDP/53 queries. Needs a resolver on the gateway.
- `dns_tcp`: length-prefixed queries over one persistent non-blocking TCP connection with `TCP_NODELAY` and reconnect backoff. Needs TCP/53 on the gateway.
- `wifi_raw`: non-QoS Null Data with ACK capture. Toward the AP it sends legacy OFDM 6 Mbps and captures LLTF ACKs. Toward an explicit peer it sends MCS0 with two zero padding bytes: HT20 on 2.4 GHz, VHT20 on C5 5 GHz. The peer captures the matching profile in `external` while the sender keeps capturing ACKs. Not available on ESP32-C6, whose ACK CSI is unusable.
- `external`: no internal generator. The device accepts the canonical UDP markers, unicast ICMP Echo Requests, direct Null Data, or unicast ESP-NOW. The external filter accepts the standard 28-byte Null Data form and, with HT20 or VHT20 metadata only, a 30-byte form with exactly two zero body bytes.

A saved choice is kept as is. If a mode does not work with a given chip, driver, AP, or resolver, the operator picks another one.

**Admission.** The detector only admits traffic that matches the configured source: in `external`, the accepted frame types above; with an internal generator, only replies to that generator. Incidental Direct, SSE, and ambient traffic is rejected before temporal admission. In `lltf20`, an 802.11 ACK is admitted only as the answer to a frame the device just injected: a valid, error-free 14-byte frame addressed to the device's MAC and matched to an ACK credit. `wifi_raw` arms the credit right before each Null Data frame, and one ACK within 5 ms consumes it; other modes never arm it, so they reject every ACK. ACKs carry no transmitter address, so the credit matches them by timing, not identity. Other control frames and ACKs in `ht20` or `vht20` are rejected. Every admitted data frame also goes through a retransmission check: a frame with the Retry bit and the same transmitter, sequence number, and length as the last admitted one is dropped and counted as a provenance rejection, unless the original never produced a callback.

**Pacing.** Internal traffic and the standalone external tool request DSCP 46, without assuming any particular WMM TID or occupancy gain. Generators keep their send phase through normal scheduler jitter, but restart from the actual send time when the next deadline would be less than half a period away, so no generator sends a close catch-up pair. The same rule lives in the shared C++ generator, the Micro-ESPectre native generator, and `tools/ha_traffic_generator_addon/espectre_traffic_generator.py`. Pacing multicast stays on the local link, and unicast or the joined group is preferred over broadcast. Occupancy is diagnostic only: the send rate never chases it.

**Station TX rate.** `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` defaults to Auto (`0`) on every target except classic ESP32, which uses HT20 MCS0 with long GI (`6.5`); legacy OFDM `6` is also available. Raw injection to the AP stays at OFDM 6 Mbps regardless. One shared rate component applies the station rate before connected services start and re-checks it after reconnects and roams; Micro-ESPectre reuses it through a native binding. If the AP cannot use the selected rate, or TX A-MPDU is on, the firmware uses automatic rates. A driver error is not hidden: if the driver rejects the station rate, connected services do not start, and `wifi_raw` fails to start on a raw-rate error.

**Timing.** CSI goes onto the detector grid by its Wi-Fi RX timestamp (`rx_ctrl.timestamp`). Callback-to-loop queue age is measured in `esp_timer` and only that duration is translated into the MAC clock domain; the two absolute clocks are never compared. Raw HTTP keeps every classified frame, including those the sensing view treats as same-slot excess.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-08-23 | Ping and UDP/53 DNS as interchangeable generators | Ping kept; UDP/53 dropped after the C3 capture showed legacy PHY and a very high retry share |
| 2026-08-23 | Mark managed traffic for low latency | Kept as a request; DSCP 46 changed the WMM category but not retry or timing on the tested AP |
| 2026-08-23 | Skip only fully missed deadlines in host pacing | Replaced by the half-period reset rule |
| 2026-08-23 | Compare Wi-Fi RX time directly with the processing clock | Replaced by same-clock queue age translated into the MAC domain |
| 2026-08-23 | Persistent non-blocking DNS/TCP, one pacing policy for firmware and host | Kept as the optional `dns_tcp` |
| 2026-08-28 | Admit any valid local IPv4 callback | Rejected: mostly more same-slot excess, plus a stability risk |
| 2026-08-28 | Lower the generator to 90, then 85 pps, and let incidental traffic fill in | Rejected: 90 pps gave 70.95% occupancy; 85 pps not run |
| 2026-08-29 | External ICMP as its own traffic mode | Rejected; `external` accepts both UDP markers and Echo Requests |
| 2026-08-29 | Ban DNS/UDP everywhere after the C3 result | Replaced by explicit `ping`, `dns`, and `dns_tcp`, since classic ESP32 DNS/UDP was stable |
| 2026-08-30 | Micro-ESPectre ping-only | Replaced by the shared native generator with all three modes |
| 2026-08-31 | Classic ESP32, S2, and Micro-ESPectre default to `dns` | Replaced; published configurations use `ping` |
| 2026-09-06 | Admit local 802.11 ACKs in `lltf20` | Accepted for `wifi_raw`; IP gates unchanged |
| 2026-09-06 | S3 ping from Video to Voice | Kept DSCP 46: about 0.8 points of RX regularity, occupancy did not revert |
| 2026-09-06 | Switch S3 to DNS/TCP | Kept explicit selection: lower CSI supply, mixed tails, occupancy did not revert |
| 2026-09-06 | S3 static32 TX buffers | Kept dynamic32: no occupancy gain, about 54 KiB less free heap |
| 2026-09-06 | Bypass S3 sensing, or disable CSI, to regularize the uplink | Kept the sensing path: most irregularity arises after driver handoff |
| 2026-09-14 | Fixed OFDM 6 Mbps station rate on every target | Replaced the same day by Auto, with HT20 MCS0 on classic ESP32 |
| 2026-09-14 | `wifi_raw` on ESP32-C6 | Disabled: ACK CSI is invalid |
| 2026-10-03 | Only credited ACKs; drop retransmissions of the last admitted frame | Accepted: on an ESP32 at −73 to −77 dBm, ACKs to the CSI stream pushed `wifi_raw` collection above 300 pps, and AP retries in `external` raised fresh records to 217/s for 100 sent packets until the stream dropped 60% of records and stalled |
| 2026-10-15 | Direct Null Data and unicast ESP-NOW in `external` | Accepted: they are addressed to the device, not incidental |
| 2026-10-17 | `wifi_raw` stable on supported chips, with an optional explicit peer | Accepted after operational validation |
| 2026-10-18 | Two zero padding bytes on peer HT20/VHT20 Null Data | Accepted: C6 needs 30-byte frames for valid HT20 CSI |
| 2026-10-21 | Extend the C5 raw VHT20 driver workaround to ESP-IDF 5.5.6 | Accepted: 5.5.6 still rejects the public configuration |

## Evidence

The measurements below come from single labs, APs, and chips. They explain the decisions above; they are not thresholds to copy to other setups.

### Generator behavior on C3 and classic ESP32

Five monitor-mode captures on 2026-08-23 (channel 2, one C3 station, one AP, 100 pps target) compared the generators. Retry share here is the share of observed frames with the 802.11 Retry bit, not an AP counter.

| Generator | Direction | Retry share | HT share | QoS | Median gap |
| --- | --- | ---: | ---: | --- | ---: |
| ping | C3 → AP | 2.81% | 99.74% | TID 5 / VI | 10.0 ms |
| DNS/UDP | C3 → AP | 76.49% | 5.32% | TID 7 / VO | 19.9 ms |
| DNS/TCP | C3 → AP | 4.05% | 99.78% | TID 5 / VI | 9.9 ms |
| external UDP, DSCP 46 | AP → C3 | 9.73% | 99.93% | TID 6 / VO | 10.0 ms |
| external UDP, best effort | AP → C3 | 9.29% | 99.93% | TID 3 / BE | 10.0 ms |

On the C3, DNS/UDP went to legacy 1 or 6 Mbps with TID 7, RTS/CTS, and about 27 times the ping retry share; DNS/TCP brought it back to HT and 10 ms spacing. DSCP 46 moved external downlink to the Voice category but did not change retries or spacing, so QoS marking is a request, not a guarantee. The receive direction stayed almost all HT for every mode, centered near 10 ms. Device logs agreed: about 94% occupancy with ping, 93–99% with DNS/TCP, 96–98% with external UDP.

On classic ESP32 (2026-08-29, channel 10), DNS/UDP also went out as legacy 6 Mbps with TID 7, but with a 3.08% retry share, 10 ms spacing, and about 95% occupancy. Legacy PHY and TID 7 are exposure factors, not sufficient causes, so the right generator depends on device, driver, AP, and resolver.

### Incidental traffic on S3

On 2026-08-28 a test firmware kept all RF, BSSID, channel, frame-format, and destination checks, but turned the UDP marker from a gate into a label, so any valid local IPv4 packet reached the detector. Native Lightweight ran with one SSE client and one new Direct `diagnostics` connection per second, measured over 60 s after settling.

| Admission | Generator | Callbacks | Admitted | Occupancy | Missing slots | Excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Marker only | 100 pps | 117.69 pps | 91.27 pps | 91.35% | 8.71/s | 8.74 pps |
| Any local IPv4 | 100 pps | 117.58 pps | 91.85 pps | 91.95% | 8.15/s | 25.61 pps |
| Any local IPv4 | 90 pps | 109.89 pps | 70.73 pps | 70.95% (36–90%) | 29.25/s | 29.33 pps |

Opening the filter added 0.6 occupancy points and 17 pps of excess: incidental packets arrive in bursts, so they mostly land in slots that are already taken. At 90 pps the device fell well below the 80% acceptance bar, so the planned 85 pps step was skipped. Two earlier open-filter runs ended in an intermittent `espectre_native` stack overflow and reboot; a later run did not reproduce it, so that is a risk signal, not a proven cause.

### S3 uplink timing investigation

A series of bounded A/B/A trials on one S3 (2026-09-06 to 2026-09-07, ping at 100 pps, a passive C6 observer) looked for the source of irregular uplink spacing:

- **Voice instead of Video QoS**: about 0.8 points better RX regularity; occupancy stayed up after switching back.
- **DNS/TCP instead of ping, static instead of dynamic TX buffers**: no repeatable occupancy gain; static32 cost about 54 KiB of free heap.
- **Minimal CSI callback**: callback time fell from about 224 to 42 µs with no uplink gain. Application-to-driver handoff was regular (p95 about 0.3 ms).
- **CSI disabled**: the improvement persisted after re-enabling CSI, so it was not caused by CSI.
- **TX-completion timing**: with exact packet joins, regularity was 99.8% at handoff and about 74% on air. Completion and radio intervals correlated at 0.999, placing the variation after driver handoff.
- **AP and channel**: moving a temporary AP to a quiet channel raised completion regularity from about 75–81% to 96–98%, twice, and it dropped again on return.
- **Driver internals**: the MAC queue was enabled promptly with normal contention settings, and the inspected PM/coexistence TX block never asserted during covered delays.
- **Calibrated PHY airtime counter**: for packets delayed over 2 ms, measured occupied airtime explained a median 90.7% of the wait, mostly 1 Mbps beacons from six nearby BSSIDs.

The dominant cause is medium-access deferral behind real radio activity, especially slow beacons, not a software queue. Three mitigations were tried without changing the 100 Hz grid. Beacon-relative pacing at 10.24 ms improved on-air regularity from 77% to 92% but not occupancy. Fixed 125 pps oversupply shortened gaps but differing controls prevent crediting it with the occupancy rise. Fixed 200 pps and randomized 9/10/11 ms spacing did not robustly improve occupancy. Production pacing, tasks, and buffers stay unchanged.

### C5/C6 short-frame CSI investigation

Starting on 2026-09-07, with ESP-IDF 5.5.5 and the same AP, ESP32-C5 rev 1.0 produced valid ACK CSI and ESP32-C6 rev 0.1 did not: 7,213/7,213 valid against 0/6,615 in the first sweep. C5 LLTF also turned out to be centered (its only zero I/Q pair at index 26), which differs from the Espressif documentation.

Nothing tried on the C6 recovered ACK CSI: scale and acquisition flags, promiscuous filters, raw TX rates up to 54 Mbps, HT and HE requests, other frame subtypes, ESP-NOW toward a C5, 802.11k Neighbor Reports, disabling 11b rates, setting the raw rate before association, and PHY dump register sweeps in a standalone ESP-IDF app that used only public Wi-Fi APIs. Over 25,000 ACKs stayed invalid. A driver-level probe showed the estimates are already invalid at the driver boundary and that part of the CSI buffer is never refreshed, so the problem lies below the public API. Probe responses and Neighbor Reports arrive at 1 Mbps DSSS and give no CSI either. Espressif tracks related C6 reports in [esp-idf#14271](https://github.com/espressif/esp-idf/issues/14271) and [esp-idf#19062](https://github.com/espressif/esp-idf/issues/19062).

The failure follows received frame length, not frame type. Data and Null Data frames below a PHY-dependent length were invalid on the C6, and longer ones were valid. A C5-to-C6 padding sweep on 2026-10-18, with subtype, addressing, and cadence held fixed, found these minimums for 100% valid CSI:

| PHY / capture | Minimum zero padding | Received length with FCS |
| --- | ---: | ---: |
| HT20 MCS0 or MCS7, long or short GI / HT20 | 2 bytes | 30 bytes |
| OFDM 6 Mbps / LLTF | 3 bytes | 31 bytes |
| OFDM 12 or 24 Mbps / LLTF | 6 bytes | 34 bytes |
| OFDM 54 Mbps / LLTF | 22 bytes | 50 bytes |

At 54 Mbps validity was mixed between 18 and 21 bytes of padding, so there is no sharp cutoff there. These results do not establish the hardware mechanism or a rule for other chips or rates. Padding the request cannot lengthen the AP's 14-byte ACK, so C6 `wifi_raw` stays disabled; the two-byte HT20 form was adopted for direct peer traffic.

### C5 raw VHT20 driver compatibility

ESP-IDF 5.5.5 rejects raw VHT20 configuration on C5 with `ESP_FAIL` and `invalid phymode`, even on a confirmed 802.11ac association. Both automatic band selection and a fixed 5 GHz build reproduced the failure. Inspection of the linked driver showed that `esp_wifi_config_80211_tx()` reads the interface's 2.4 GHz PHY field when validating the requested PHY; the 5 GHz configuration is stored separately. A temporary probe using the setter called by the public API's successful path transmitted padded VHT20 Null Data and produced valid CSI on the receiving C5, with ACK/LLTF on the sender.

ESP-IDF 5.5.6 still has the bug. On 2026-10-21 a C5 rev 1.0 on a 5 GHz channel 44 association ran Native builds that differed only in this workaround. With an explicit peer and the workaround disabled, the generator did not start (0 pps). With the workaround enabled, it sent about 100 pps; the `auto` destination also sent about 100 pps. The peer MAC was a placeholder, so this check confirms raw transmission only; the CSI validation above was done on 5.5.5.

The compatibility path is restricted to C5 with ESP-IDF 5.5.5 or 5.5.6, an explicit peer on a 5 GHz channel, a public API result of `ESP_FAIL`, and a confirmed VHT20 association. It applies the fixed MCS0, long-GI configuration through the driver's `ic_set_80211_tx_rate_config` symbol. Other errors, PHYs, bands, and SDK versions retain the public API result. This is a dependency on an undocumented driver ABI; remove the workaround when a driver with corrected validation is adopted.

### Station TX-rate comparison

Between 2026-09-13 and 2026-09-14, Native compared Auto with fixed OFDM 6 Mbps for every generator on S3, C3, C5, and C6 (60 s cells, 100 pps, one SSE client, one HTTP request pair per second). All 3,840 scored HTTP requests succeeded and the fixed rate worked everywhere, but occupancy changes were mixed: S3 `dns` went from 75% to 97%, while the other cells moved by a few points either way, and RSSI also differed between arms by up to 4 dB. C6 `wifi_raw` stayed at zero occupancy at either rate. These single sequential samples do not show a general gain.

### External radio and peer sensing validation

Hardware checks, all without promiscuous mode:

| Hardware | Traffic source | Result |
| --- | --- | --- |
| C3 transmitter to S3 receiver | Direct Null Data and unicast ESP-NOW | Valid CSI with HT20 and LLTF profiles |
| C5 mini and standard C5, 2.4 GHz | Direct Null Data and unicast ESP-NOW | Valid external HT20 with both sources, and ACK/LLTF on the Null Data generator |
| C5 mini and standard C5, 5 GHz | Direct Null Data, VHT20 MCS0 with two zero padding bytes | Valid external VHT20 and sender ACK/LLTF in both directions; source-off controls admit no samples |
| C5 transmitter to C6 receiver | Direct Null Data and unicast ESP-NOW | Valid HT20 with ESP-NOW and with Null Data padded by two zero bytes; unpadded Null Data is invalid |
| C6 transmitter to C5 receiver | Direct Null Data and unicast ESP-NOW | Valid external HT20 on C5 with standard and padded Null Data and ESP-NOW; returning ACKs are invalid on C6 |

With two C5 boards about three meters apart on the same AP, each detector ran 30-second still and walking phases in both directions (4 hits to enter motion, 3 to leave). Both Lightweight and High Accuracy reported motion in 0% of about 117 motion events per still phase and 100% per walking phase, on both boards, with no request errors, hardware errors, or queue drops. ESP-NOW from C6 to C5 did not recover C6 ACK CSI either, whatever the payload.

At 5 GHz on channel 40, the mini and standard C5 supported both roles with the version-bound driver compatibility path above. Radio checks and two-minute High Accuracy comparisons in each direction ran with serial monitors closed, regular acquisition, and no new hardware errors, invalid estimates, or queue drops. The participant confirmed alternating stillness and walking with the boards fixed. Both reported broadly matching idle and motion patterns; the ACK/LLTF side spent more time in motion than the external VHT20 side, including after reversing roles. Without timestamped activity labels, these comparisons do not establish detector accuracy or which input detects real movement more reliably. Disabling the generator stopped accepted input on both devices.

Both C5 boards also retained valid ACK/LLTF acquisition with `wifi_raw` targeting `auto` on the AP's 2.4 GHz and 5 GHz radios. The configured PHY was OFDM 6 Mbps (802.11g and 802.11a, respectively); raw CSI confirmed legacy LLTF at 20 MHz. A BSSID-only band switch updated the raw destination while retaining `auto`. One-minute checks on each band remained ready without new CSI errors or queue drops, with serial monitors closed.

With padding in production `wifi_raw`, C5 → C6 and C6 → C5 (from a temporary C6 transmitter) both ran at about 99–101 CSI packets/s with no invalid estimates, hardware errors, or queue drops; source-off windows admitted nothing. C6 Native slowdowns during these runs came from the USB console: the USB Serial/JTAG interrupt mask included Start of Frame, which the driver does not service, causing up to about 430,000 interrupts/s. The console now sets the mask before installing the driver and writes logs and Improv output with zero timeout. After that fix, five minutes of C5 → C6 traffic held about 100 packets/s, and C5 ACK/LLTF and C6 padded HT20 showed matching idle and motion patterns (relative agreement, not labeled accuracy).

Still unverified: external LLTF reception of Null Data and ESP-NOW on C5 and C6. Setup instructions are in [CSI acquisition](../CSI.md#external-sources).

## Alternatives considered

- **DNS/UDP as the universal default.** On the C3 it triggered legacy PHY, RTS/CTS, retries, and broken cadence; on classic ESP32 it was fine. Neither result makes it portable.
- **DNS/TCP as the universal default.** Ping is stateless and needs no TCP resolver. DNS/TCP stays as an option with its connection costs.
- **External UDP only.** Works well for controlled tests, but needs another always-on host and correct routing. Micro-ESPectre has no external UDP listener.
- **Adapt send rate to occupancy.** Occupancy is capped by the sampler and mixes supply with AP scheduling and slot placement; C3 and ESP32 trials did not beat a fixed cadence.
- **Admit all valid local traffic.** Valid is not the same as useful: on the S3, extra callbacks mostly became excess instead of new slots, and the generator could not drop to 90 pps.
- **Admit every received packet.** A burst would fill a window without covering its physical time. Raw HTTP still keeps the excess for research.
- **Timestamp by AP send time or processing time.** The device does not know the AP's send time, and processing time measures software delay after the PHY observation.

## Consequences

Users can pick stateless ICMP, DNS over UDP, or DNS over TCP without any extra service, plus `wifi_raw` and `external` where the hardware supports them. Each target can follow its measured behavior without removing options elsewhere. Generators no longer emit catch-up bursts, every production path uses the same physical-time grid, and raw capture loses nothing to admission except counted ring drops. The detector sees one explicit, deterministic source: in `lltf20`, ACKs count only when they answer an injected `wifi_raw` frame, so the device's own Direct or MQTT traffic and AP retransmissions cannot add samples. In `external`, the sender is responsible for how it mixes the accepted frame types.

The limits: DNS/TCP needs TCP/53 and keeps socket state; DNS/UDP can hit a legacy-PHY retry path on some chips; close responses can still land in the same slot; DSCP-to-WMM mapping depends on direction, sender, AP, and driver; external ICMP adds Echo Replies from the device; and the AP can still queue, retry, or burst frames after the generator paced them. Monitor-mode retry shares and pcap gaps are observer measurements, not device truth. New devices and APs need the same validation rather than copied numbers.

## Related

- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [CSI acquisition](../CSI.md), [troubleshooting](../TROUBLESHOOTING.md), [setup](../SETUP.md), [algorithms](../ALGORITHMS.md), [CLI guide](../CLI.md), and [API reference](../API.md)
- Commits `bab99c24` (pacing), `02e3c531` (explicit generators), `c2837c7b` (external ICMP), `3f027508` (Micro generator), `6f729fc2` (`ping` default), `12075d57` (station rate policy), `d8721300` (C6 `wifi_raw` disabled), `ca2a6a60` (explicit peer), `5d789e9c` (`wifi_raw` ACK capture), `b27f5432` (ACK credit and retry check), `8985ecb7` (Null Data and ESP-NOW in `external`)
