# ADR: standardize managed CSI traffic sources

- Status: Accepted
- Date: 2026-08-23
- Updated: 2026-08-25

## Context

ESPectre needs a regular supply of HT20/HT-LTF packets so motion detection can cover its fixed temporal grid. Runtime diagnostics repeatedly showed average occupancy around 85% even when a traffic generator reported approximately `100 pps`. Raw callback rate alone could not distinguish missing packets from AP scheduling bursts, device-side send bursts, retransmissions, legacy-PHY delivery, same-slot excess, or processing backlog.

The project supports internal ICMP ping, internal DNS, and external UDP delivered through the AP. These paths initially differed in protocol, QoS treatment, and missed-deadline behavior. DNS originally used UDP port `53`; the standalone external tool also lacked the fixed-phase behavior used by the internal generator.

Motion sensing also requires a precise definition of packet time. The CSI delivered by the Wi-Fi callback is estimated by the PHY from the packet training field at RF reception. Later callback or loop handling does not move that channel observation forward in time. The fixed admission grid must therefore use the Wi-Fi RX timestamp carried with the frame, while software clocks may measure only processing backlog.

## Evidence

### Capture method

Five monitor-mode captures were recorded on 2026-08-23 with a Mac on channel 2 (`2417 MHz`), an ESP32-C3 station at `ac:eb:e6:4a:e7:08`, and BSSID `e6:fa:c4:20:19:de`. Internal traffic crossed the gateway MAC `e4:fa:c4:b0:19:f8`; external traffic came from the Home Assistant host at `2c:cf:67:9b:80:2d`. The controlled target was `100 pps`.

The analysis decoded radiotap and IEEE 802.11 headers directly. It counted data-frame direction, Retry flags, QoS TID, HT versus legacy PHY, and gaps between observed non-Retry data frames. Control ACK, RTS, CTS, beacon, and unrelated data frames were excluded from the protocol tables. Because WPA3 payloads remained encrypted, protocol attribution comes from the controlled generator mode active during each capture rather than payload inspection.

The pcaps are local laboratory artifacts rather than versioned repository inputs. Their identities are retained here so an archived copy can be verified later.

| Mode | Local filename | Frames | Duration | SHA-256 prefix |
| --- | --- | ---: | ---: | --- |
| optimized ping | `espectre_air_ch2.pcap` | 31,286 | 57.159 s | `9bdc5a1bbff7` |
| former DNS/UDP | `espectre_air_dns_ch2.pcap` | 142,452 | 111.721 s | `31479d6ae622` |
| external UDP, DSCP 46 | `espectre_air_udp_external_ch2.pcap` | 68,568 | 189.611 s | `9b2f9035f343` |
| external UDP, best effort | `espectre_air_udp_be_ch2.pcap` | 100,112 | 291.025 s | `6f131aef9a0c` |
| DNS/TCP | `espectre_air_dns_tcp_ch2.pcap` | 86,860 | 155.657 s | `70450799fbca` |

### Generator-direction findings

`Retry share` is the fraction of observed data transmissions whose IEEE 802.11 Retry bit was set. It is not an AP retry counter: an independent sniffer can miss an original attempt or a retry. The large cross-mode differences remain useful, but the percentages are specific to this capture position and AP.

| Generator path | Measured direction | Observed data transmissions | Retry share | HT share | Dominant QoS mapping | Median non-Retry gap |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| ping | C3 → AP | 6,080 | 2.81% | 99.74% | TID 5 / AC_VI | 10.009 ms |
| DNS/UDP | C3 → AP | 18,037 | 76.49% | 5.32% | TID 7 / AC_VO | 19.950 ms |
| DNS/TCP | C3 → AP | 17,760 | 4.05% | 99.78% | TID 5 / AC_VI | 9.941 ms |
| external UDP, DSCP 46 | AP → C3 | 22,188 | 9.73% | 99.93% | TID 6 / AC_VO | 9.978 ms |
| external UDP, best effort | AP → C3 | 33,621 | 9.29% | 99.93% | TID 3 / AC_BE | 9.988 ms |

The former DNS/UDP path was the only severe anomaly. Approximately 94.7% of its observed uplink transmissions used legacy PHY, predominantly `1` or `6 Mbit/s`, and approximately 94.7% carried TID 7. Its Retry-frame share was about twenty-seven times the optimized ping value. RTS/CTS exchanges and repeated attempts were visible around the legacy data frames. DNS/TCP removed this behavior: almost every observed uplink frame returned to HT, its spacing returned to approximately `10 ms`, and its Retry share became close to ping.

DSCP 46 changed the external downlink mapping from this AP's TID 3 / AC_BE to predominantly TID 6 / AC_VO, but it did not materially reduce observed Retry share or median spacing on the otherwise healthy link. QoS marking is therefore a latency request, not a portable guarantee of higher occupancy. The exact TID selected for the same DSCP also differed by sender implementation: the C3 mapped its marked ping and DNS/TCP uplink mainly to TID 5 / AC_VI, while the AP mapped marked external downlink mainly to TID 6 / AC_VO.

### C3 receive-direction findings

The receive direction is the one that produces CSI on the C3. Every healthy path remained almost entirely HT in this direction.

| Mode | Observed AP → C3 Retry share | HT share | Median gap | 95th-percentile gap | Gaps below 5 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ping | 7.62% | 99.86% | 9.976 ms | 15.845 ms | 12.59% |
| DNS/UDP | 9.92% | 99.93% | 9.465 ms | 21.512 ms | 21.82% |
| DNS/TCP | 8.46% | 99.91% | 9.951 ms | 15.076 ms | 9.84% |
| external UDP, DSCP 46 | 9.73% | 99.93% | 9.978 ms | 13.244 ms | 9.15% |
| external UDP, best effort | 9.29% | 99.93% | 9.988 ms | 13.500 ms | 9.42% |

The AP did not emit a perfectly uniform stream, but the paced healthy paths remained centered near `10 ms`. DNS/UDP had both the highest close-gap share and the longest 95th-percentile gap, which is consistent with its uplink retry behavior producing less regular downlink opportunities. The encrypted capture establishes correlation, not per-query causality. External DSCP and best-effort traffic were nearly indistinguishable in retry and spacing on this AP despite their different QoS access categories.

Short device-log observations agreed with the air captures. Optimized ping produced roughly 94% median occupancy. After the DNS/TCP firmware was flashed, the C3 reported `99-100 tx/s`, mostly `93-99%` steady-state occupancy with a median around 96%, and zero stale or out-of-order packets during the observed interval. External paced UDP reached approximately `96-98%` occupancy. These are diagnostic observations, not detector performance gates.

### Timing and admission findings

The packet's useful sensing instant is reception by the device, when the PHY estimates CSI from HT-LTF. AP transmit time is not directly available to the device, and callback processing time describes software latency rather than the RF channel sample. A CSI result cannot be deferred arbitrarily and recalculated later without retaining the original RF samples; the callback exposes the estimate already associated with that received packet.

ESP-IDF processing therefore uses `rx_ctrl.timestamp` for temporal slots, gaps, and occupancy. The Wi-Fi timestamp belongs to the MAC clock domain, not the `esp_timer` domain. The runtime records `esp_timer` at callback acceptance, computes callback-to-loop queue age only within that software clock, and translates the elapsed duration into the RX timestamp domain for backlog rejection. Directly comparing the two absolute clocks is invalid and was the reason target guards existed in the earlier implementation.

Packets counted as same-slot excess still contain valid CSI, but they do not add a new temporal position to the fixed sensing grid. Sensing frontends admit the candidate nearest the slot center rather than letting a burst fill a physical-time window. Raw HTTP preserves classified timestamped CSI before temporal admission so research capture does not discard those frames; live detectors and derived sensing views apply the same admission as deployed sensing frontends.

## Decision

Standardize managed CSI traffic as follows:

- keep stateless ICMP ping as the universal internal default;
- keep DNS as an optional runtime mode, but send length-prefixed queries over one persistent, non-blocking TCP connection to gateway port `53`, with `TCP_NODELAY` and reconnect backoff;
- do not use UDP/53 for managed sensing traffic;
- request DSCP 46 treatment for internal traffic and the standalone external UDP tool, without treating a particular WMM TID or occupancy improvement as guaranteed;
- preserve the configured send phase through ordinary scheduler jitter, but restart from the actual send time when the next phase deadline would be less than half a period away, so no generator emits a close catch-up pair;
- apply that fixed-phase rule in the shared C++ generator, the Micro-ESPectre native generator, and `tools/espectre_traffic_generator.py`;
- limit pacing multicast to the local link and prefer unicast or the joined multicast group over subnet or limited broadcast;
- keep occupancy diagnostic-only and never make device send rate chase admitted occupancy;
- place CSI on the detector grid using the device Wi-Fi RX timestamp, with processing time used only for same-domain queue-age measurement; and
- keep raw HTTP records even when the sensing view classifies additional same-slot records as excess.

Internal DNS/TCP requires a gateway resolver that accepts TCP queries on port `53`. If it does not, operators should use ping or an external paced source rather than silently falling back to UDP/53.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-08-23 | Use ICMP ping or UDP/53 DNS as interchangeable internal generators | Ping retained; UDP/53 rejected after the C3 capture showed legacy PHY and extreme observed retry share |
| 2026-08-23 | Mark managed traffic for low latency | Retained as a request; external DSCP 46 changed WMM category but did not materially change retry or timing on the tested AP |
| 2026-08-23 | Skip only deadlines already fully missed on host pacing | Replaced by the half-period reset rule so late wake-up cannot produce a close packet pair |
| 2026-08-23 | Compare Wi-Fi RX time directly with the processing wall clock on selected targets | Replaced by measuring queue age in `esp_timer` and translating only that duration into the MAC timestamp domain |
| 2026-08-23 | Use persistent non-blocking DNS/TCP and one pacing policy across firmware and host tools | Accepted |

## Alternatives Considered

### Keep DNS over UDP and tune its QoS

Rejected. UDP/53 triggered a target-specific Wi-Fi handling path on the tested C3 that changed QoS, forced predominantly legacy transmission, added RTS/CTS and retries, and disrupted the response cadence. Explicit DSCP cannot reliably override such stack classification.

### Make DNS/TCP the universal default

Rejected. Ping is stateless, broadly available, and does not require a TCP-capable gateway resolver. DNS/TCP remains valuable as an alternative traffic shape and performed well on the tested gateway, but it has connection lifecycle and compatibility costs.

### Use only external UDP

Rejected. External pacing performed well and is useful for controlled experiments, but it requires another always-on host and correct routing. Micro-ESPectre intentionally has no external UDP listener. Device-local ping remains the most portable default.

### Adapt send rate from occupancy

Rejected. Occupancy is capped by the sampler and mixes traffic supply with AP scheduling and slot placement. Earlier bounded C3 and classic ESP32 trials did not outperform fixed cadence. Raw HTTP reports transport backpressure but never changes the external generator rate.

### Admit every received CSI packet

Rejected for live sensing. A burst would manufacture a full detector window without covering the corresponding physical time. Valid excess CSI remains available through raw HTTP capture when research needs it.

### Timestamp by AP transmit time or processing time

Rejected. AP transmit time is not available as the device's authoritative sensing timestamp, while processing time measures software scheduling after the PHY observation. The Wi-Fi RX timestamp best represents when the measured channel existed.

## Consequences

Benefits:

- the accepted sources avoid the measured C3 DNS/UDP legacy-PHY failure mode;
- generator-side scheduler delay no longer creates avoidable catch-up bursts;
- ping, DNS/TCP, and external UDP have explicit and comparable roles;
- all production paths use the same physical-time interpretation for motion sensing;
- DNS/TCP avoids the measured UDP/53 failure mode while preserving the AP as the downlink packet source; and
- raw HTTP capture remains lossless with respect to temporal admission decisions, except for explicitly counted bounded-ring drops.

Trade-offs and limits:

- DNS/TCP maintains socket state and reconnect logic and depends on gateway TCP/53 support;
- TCP acknowledgments and DNS responses may create harmless same-slot excess even when occupancy improves;
- DSCP-to-WMM mapping is direction-, sender-, AP-, and driver-dependent;
- the AP may still queue, aggregate, retry, or burst frames after the generator has paced them;
- monitor-mode Retry shares are observer measurements, not authoritative device or AP counters;
- pcap gap statistics use the monitor's observation timestamps, not the device's authoritative `rx_ctrl.timestamp`;
- encrypted captures cannot prove each frame's L4 protocol without correlating the controlled test mode; and
- the numeric capture results characterize one C3, one AP, channel 2, and one laboratory interval, so future devices and APs require the same validation rather than copied thresholds.

## Related

- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- [`../TUNING.md`](../TUNING.md)
- [`../SETUP.md`](../SETUP.md)
- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`../CLI.md`](../CLI.md)
- [`../ESPECTRE_PROTOCOL.md`](../ESPECTRE_PROTOCOL.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
