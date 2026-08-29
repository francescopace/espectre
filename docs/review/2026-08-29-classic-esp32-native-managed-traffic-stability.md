# Review: classic ESP32 managed-traffic stability

- Review date: 2026-08-29
- Status: Concluded
- Scope: Classic ESP32, primarily Native, with an ESPHome comparison
- Outcome: Internal ping and DNS/TCP were rejected on the tested setup; DNS/UDP passed the controlled Native screening and became the classic ESP32 default

## Purpose

This review preserves the investigation into simultaneous CSI loss and Direct HTTP stalls on the classic ESP32. It is a dated experiment record, not the owner of the traffic-source contract. The durable policy remains in [the managed-traffic ADR](../adr/2026-08-23-standardize-managed-csi-traffic-sources.md).

The investigation began with a broad suspicion that Direct HTTP, asynchronous response handling, or general memory pressure was failing under Wi-Fi load. The completed tests instead found a protocol-dependent loss of forward progress in the classic ESP32 Wi-Fi/lwIP path. Internal ICMP, externally supplied ICMP, and persistent DNS/TCP all reproduced long Direct pauses or CSI collapse. Connectionless DNS/UDP did not: three clean Native boots completed every control request and every diagnostic sample, and the observed radio cadence remained close to the configured `100 pps`.

That result does not make DNS/UDP a universal source. A separate C3 investigation had found severe retries and cadence disruption with the same protocol. ESPectre therefore exposes `ping`, `dns`, and `dns_tcp`, keeps `ping` as the shared default, and selects `dns` only in validated classic ESP32 product profiles.

## Evidence boundary

The controlled tests used one classic ESP32, the Lightweight detector unless stated otherwise, one pinned and verified AP BSSID and channel, and a target traffic rate of `100 pps`. A standard replica used a clean reset, a `25 s` warm-up, and a `60 s` scored window. The final ping candidate used three independent boots and a `300 s` scored window per boot.

The Direct forensic client opened a new TCP connection for every request, alternated `status` and `diagnostics`, and measured connect, request send, first byte, headers, and body separately. SSE was disabled in the discriminating Native runs. Timeouts were retained as censored failures: they contribute to the failure rate, but their configured timeout is not treated as an exact latency sample.

The campaign included host-side TCP captures and one 802.11 monitor-mode observation. They could show retransmissions, delayed SYN-ACKs, header-to-body gaps, radio retries, PHY mode, and packet cadence. They could not expose the internal state of the closed Wi-Fi driver.

The source tree and benchmark harness changed during the investigation. Early runs that did not verify the selected BSSID, provisioning attempts that did not reach the scored window, and interrupted captures are retained only as discovery evidence. Quantitative comparisons use completed runs with the intended firmware profile and verified association.

## Early benchmark evidence and BSSID control

The first full-corpus runs were intermittent. On 2026-08-26, Native and ESPHome produced a mixture of Direct timeouts, diagnostic gaps, and low occupancy; immediate reruns could pass with Native occupancy around `82–83%`. On 2026-08-27, both ESPHome detectors completed `60/60` samples at roughly `83–86%` occupancy. After later runtime and benchmark changes, failures became frequent again: some cases timed out during Direct preparation, while others entered the scored window and then lost diagnostics for tens of seconds. These early results established intermittence, but they were not suitable for configuration comparisons. The Matter cases in those runs were flash-only smoke checks with no sensing samples, and the Micro cases used a different traffic and Direct implementation while also encountering provisioning failures; neither group contributes evidence to this review.

One firmware change persisted a newly selected BSSID without the reboot previously used after reassociation. On the classic ESP32, live CSI rearming could report success while callbacks remained sterile. Some early low-RSSI attempts had also associated with the wrong AP. The benchmark was therefore changed to set the BSSID through the public API, reboot, and verify the active BSSID and channel before each scored run. Attempts that failed this check were excluded. This fixed a separate provisioning and CSI-rearm problem; it did not remove the managed-traffic stalls once the correct AP was pinned.

The first phase-aware Native baseline then recorded 54 Direct attempts, one censored `diagnostics` failure before the first byte, only `26/60` expected diagnostic samples, and an `11.84 s` maximum diagnostic interval. The paired High Accuracy case failed its Direct preflight after a request send of less than a millisecond and roughly 31 seconds without a first response byte. This was the starting point for the focused investigation below.

## Direct HTTP and asynchronous-path probes

The non-persistent client ruled out HTTP keep-alive and session rearming: failures occurred on the first request of a fresh TCP connection. Disabling SSE did not prevent them, so the shared response/SSE worker could amplify latency but was not necessary for the failure. The main Direct mutex did not cover network sends, and its short acquisition timeout did not reveal a seconds-long application critical section.

Reducing centralized diagnostics removed roughly one kilobyte of BSS and shortened the response, but did not remove the failure. The reduced diagnostics body still crossed the `1440`-byte TCP MSS and could stall after its headers. The much smaller `status` response fit below one MSS and also stalled in later ping runs. Payload size changed exposure, not the underlying condition.

The response-side probes showed that application writes could complete quickly because a response of a few kilobytes fit inside the configured `5760`-byte TCP send buffer. That return did not prove that lwIP or the Wi-Fi driver had transmitted and acknowledged the queued segments. The theoretical ESP-IDF `send() == 0` spin was reviewed as an edge case, but the observed failures did not establish it as the cause.

TCP observations placed the long pauses below ordinary HTTP parsing and dispatch. Some requests were accepted and answered through the headers before the body stopped; later in the same degraded interval, request delivery, SYN-ACK generation, or the first response could also be delayed. Recurrent pauses close to `22.5 s` matched four cumulative retransmission timeouts with the configured `1500 ms` initial RTO: `1.5 + 3 + 6 + 12`. In several header-to-body stalls, no corresponding retransmission was visible at the host, which is consistent with loss of progress between lwIP queuing and radio transmission rather than an HTTP worker deadlock.

The classic ESP32 profile raised the Direct HTTPD task from priority 1 to priority 4 before the controlled traffic-source matrix. It remained enabled in all later Native and ESPHome runs. Priority 4 was not sufficient to prevent ping and DNS/TCP collapse, and no otherwise identical priority `1` versus `4` comparison was completed with the stable DNS/UDP source. It is therefore part of the tested profile, not a demonstrated cause of the final improvement.

## Internal ICMP versus external UDP

The most discriminating early comparison used three replicas per arm at `100 pps`, with the same device, AP, detector, Direct probe, warm-up, and scored duration.

| Traffic source | Direct attempts | Censored failures | CSI samples | Effective rate | Maximum diagnostics interval | `ENOMEM` events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| External UDP, 3 replicas | 360 | 0 | 180 / 180 | 96.02–98.83 pps | 1.033 s | 0 |
| Internal ICMP, 3 replicas | 40 | 2 | 18 / 180 | 68.72–74.88 pps where measurable | 23.966 s | 71 |

All three external replicas completed 120 Direct attempts and received all 60 expected CSI samples. The TCP observations contained no SYN retry, request retransmission, response retransmission, header-only response, missing response, or header-to-body gap above one second. The maximum diagnostics header-to-body gap was `33.692 ms`, the maximum first-response latency was `63.277 ms`, and the maximum SYN-ACK latency was `52.694 ms`.

The internal arm degraded in every replica. The first two replicas received only 7 and 11 of 60 CSI samples, with maximum diagnostic intervals of `13.906 s` and `23.966 s`. The third received no scored CSI samples and produced two censored Direct failures. Across the three runs, diagnostics header-to-body gaps reached `22.184 s`; `status` also stalled, reaching a `4.340 s` first-response delay in the third replica.

This comparison proved that CSI sensing and Direct HTTP could coexist under externally delivered traffic. It did not make external UDP a production option because it requires another service on the user's network, and it did not distinguish ICMP from other device-originated protocols.

## ICMP pacing and QoS probes

The original internal generator used open-loop pacing and DSCP 46. Changing the marking to DSCP 0 did not remove the failure. The valid DSCP 0 runs then reduced the number of outstanding Echo Requests without changing the AP, BSSID, target rate, or Direct workload.

| Profile | Completed replicas | Direct attempts | CSI samples | Maximum diagnostics interval | `ENOMEM` events | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DSCP 0, open loop | 1 | 50 | 25 / 60 | 24.677 s | 42 | Unstable |
| DSCP 0, window 4 | 1 | 54 | 27 / 60 | 16.105 s | 0 | Unstable |
| DSCP 0, window 1 | 2 | 184 | 92 / 120 | 11.456 s | 0 | Unstable |

Limiting outstanding requests reduced pressure, removed `ENOMEM` from the valid windowed runs, and improved average availability. Long CSI gaps and zero-sample windows remained. `ENOMEM` was therefore a pressure signal, not a necessary condition for collapse. Incomplete replicas were excluded from the totals.

## Dynamic Wi-Fi TX buffers

The next controlled comparison changed `CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM` from 32 to 64 while retaining internal ping and the same benchmark shape.

| TX buffers | Replicas | Direct attempts | Censored failures | CSI samples | `ENOMEM` events | Maximum body gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3 | 58 | 2 | 27 / 180 | 72 | 22.441 s |
| 64 | 3 | 98 | 1 | 48 / 180 | 60 | 22.542 s |

TX64 improved aggregate availability and reduced the number of long body gaps, but it preserved the same approximately `22.5 s` worst-case signature and included a catastrophic replica with only 2 of 60 CSI samples. More queue depth increased margin; it did not restore forward progress. TX128 was not tested because it would consume more memory and could defer the same failure without testing a different mechanism.

## Static Wi-Fi TX buffers

The allocation-model probe replaced dynamic TX buffers with 16 statically reserved buffers. A clean build verified `CONFIG_ESP_WIFI_STATIC_TX_BUFFER=y`, `CONFIG_ESP_WIFI_STATIC_TX_BUFFER_NUM=16`, and `CONFIG_ESP_WIFI_TX_BUFFER_TYPE=0`; dynamic TX buffers and TX A-MPDU were disabled, and the TCP/IP receive mailbox remained at 32 entries. The firmware also exposed free, minimum free, and largest free `MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA` heap.

The first boot completed its `180 s` scored window but failed decisively: 5 of 20 Direct attempts were censored, only 5 of 180 expected diagnostic samples arrived, the maximum diagnostic interval was `75.71 s`, and the serial log contained 79 ping `ENOMEM` events. The BSSID and channel were correct, RSSI remained between approximately `-49` and `-54 dBm`, and there were no `csi:0/0` windows. Internal DMA-capable heap remained substantial: free heap was at least `108.82 KiB`, its cumulative low-water mark was `80.62 KiB`, and the largest free block was at least `96 KiB`.

The next two clean boots failed the Direct preflight before scoring. Both associated with the correct BSSID on channel 10 at approximately `-56` to `-57 dBm`, armed CSI, and started ping. The host connected and received the first response byte in less than `160 ms`, but the response did not complete within approximately `30.22 s`. A final probe on the third boot returned the sub-MSS `status` response in `0.37 s`, while `diagnostics` timed out at `5 s`.

Static reservation therefore neither removed ping allocation failures nor restored multi-segment Direct progress. The longer promotion gate and the proposed static-buffer-plus-A-MPDU profile were not run after this rejection.

## Combined ping candidate

The final ping candidate combined the remaining mitigations:

- 64 dynamic Wi-Fi TX buffers;
- a TCP/IP receive mailbox of 64 entries;
- at most one outstanding ICMP request;
- a `250 ms` ICMP reply timeout;
- DSCP 46;
- SSE disabled; and
- three boots with `25 s` warm-up and `300 s` scored duration each.

| Replica | Direct attempts | Censored failures | CSI samples | Effective rate | Maximum diagnostics interval | `csi:0/0` windows | Minimum free heap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 118 | 5 | 54 / 300 | 76.57 pps | 67.284 s | 7 | 104,563 bytes |
| 2 | 68 | 6 | 28 / 300 | 61.72 pps | 43.793 s | 7 | 104,479 bytes |
| 3 | 68 | 4 | 30 / 300 | 56.00 pps | 91.907 s | 6 | 104,007 bytes |
| Total | 254 | 15 | 112 / 900 | — | 91.907 s | 20 | 104,007 bytes |

No replica passed. The run produced no `ENOMEM`, fatal event, or reboot, yet Direct and CSI both degraded. At least one zero-sample window occurred well inside each scored soak. TCP observation covered the latter part of the first replica and all of the next two: among 240 HTTP flows, 50 header-to-body gaps exceeded one second, 19 exceeded five seconds, 10 responses stopped after their headers, and the maximum body gap was `46.457 s`. The approximately `22.5 s` signature remained and sometimes doubled.

The larger mailbox was tested only inside this combined candidate. Because the profile failed decisively and DNS/UDP later provided a protocol-level discriminator, no separate mailbox-only matrix was run.

## ESPHome comparison

A canonical ESPHome Lightweight run was built, erased, provisioned through Improv Serial, pinned through `set_wifi_bssid`, and verified on the requested BSSID and channel. The firmware initially calibrated normally, but occupancy declined from the `80–90%` range to approximately `30–60%`, followed by repeated ping `ENOMEM` events. The benchmark stopped after `54.6 s` because Direct could not obtain five consecutive ready, non-zero CSI samples, so the planned `300 s` scored window never began.

This run does not quantify ESPHome steady state. It does show that the ping collapse is not specific to Native orchestration: ESPHome uses the same runtime, CSI capture service, traffic generator, and ESP-IDF Wi-Fi/lwIP path while adding its own API and component workload. DNS/UDP was not subsequently soaked on ESPHome hardware during this campaign.

## DNS over persistent TCP

Native was then tested with the source now named `dns_tcp`: length-prefixed DNS queries sent through one persistent, non-blocking TCP connection to gateway port 53. Each replica persisted and verified the BSSID, reset the device, observed RSSI between approximately `-51` and `-56 dBm`, kept SSE disabled, warmed up for `25 s`, and scored for `60 s`.

| Replica | Direct attempts | Censored failures | CSI samples | Mean admitted rate | Mean occupancy | Maximum diagnostics interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 1 | 5 / 60 | 11.31 pps | 20.2% | 12.83 s |
| 2 | 42 | 0 | 21 / 60 | 40.34 pps | 43.3% | 12.02 s |
| 3 | 36 | 1 | 17 / 60 | 22.97 pps | 28.1% | 33.68 s |
| Total | 90 | 2 | 43 / 180 | 30.10 pps across received samples | 34.6% across received samples | 33.68 s |

The serial evidence contained 59 `csi:0/0` windows, 103 DNS send failures with `EAGAIN`, 31 TCP connection-establishment events, and no DNS `ENOMEM`. Calibration did not complete usefully in any replica. DNS/TCP therefore failed independently of low RSSI and general heap allocation failure, and it was rejected as the classic ESP32 default on this AP.

## External ICMP

A forensic build extended the external CSI provenance filter to admit unicast ICMP Echo Requests addressed to the ESP32. The host sent requests at `100 pps`; the internal traffic generator remained stopped, the normal lwIP echo responder transmitted the replies, SSE remained disabled, and the BSSID and channel were verified for every boot. The bounded packet shape was later retained as a supported diagnostic input to `external`, alongside the UDP marker.

| Replica | Direct attempts | Censored failures | CSI samples | Mean admitted rate | Mean occupancy | Maximum diagnostics interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98 | 0 | 49 / 60 | 82.48 pps | 82.80% | 12.64 s |
| 2 | 100 | 0 | 50 / 60 | 83.56 pps | 83.72% | 11.30 s |
| 3 | 8 | 2 | 2 / 60 | 85.97 pps where measurable | 87.00% where measurable | Two requests exceeded 30 s |
| Total | 206 | 2 | 101 / 180 | — | — | — |

No replica was clean. The first two reproduced approximately 11–13-second Direct pauses; the third collapsed to two diagnostic samples and two censored failures. There were no generator `ENOMEM` events because the internal generator was not running.

This rules out the ESPectre ping task and its raw-socket request path as necessary causes. It also separates external UDP from external ICMP: ingress-only UDP was stable, while Echo Requests forced the ESP32 to transmit one reply for every received packet and reproduced the failure. At that point, however, the test still could not distinguish ICMP-specific handling from generic high-rate device transmission.

## DNS over UDP

The final protocol probe used `dns`, now defined as connectionless DNS/UDP to the gateway resolver. It retained the same `100 pps` target, verified BSSID, clean boots, `25 s` warm-up, `60 s` scored window, alternating non-persistent `status` and `diagnostics` requests, and disabled SSE.

| Replica | Direct attempts | Censored failures | Diagnostics samples | Mean admitted rate | Mean occupancy | Maximum diagnostics interval | Slowest request |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120 | 0 | 60 / 60 | 85.57 pps | 85.73% | 1.023 s | 86.962 ms |
| 2 | 120 | 0 | 60 / 60 | 86.31 pps | 86.42% | 1.017 s | 81.182 ms |
| 3 | 120 | 0 | 60 / 60 | 82.75 pps | 82.87% | 1.025 s | 71.465 ms |
| Total | 360 | 0 | 180 / 180 | — | — | 1.025 s | 86.962 ms |

All 360 Direct requests completed. The runs contained no censored failure, `csi:0/0` window, generator error, watchdog, or reboot. This was the first internally generated source to pass every replica on the classic ESP32 while the Direct control plane was exercised continuously.

The monitor-mode observation covered almost two minutes of steady traffic. ESP32 uplink remained predominantly legacy `6 Mbit/s` with TID 7, but only `3.08%` of observed transmissions carried the Retry flag. Non-retry uplink spacing had a `10.022 ms` median and a `98.91 pps` rate. AP-to-device traffic was `99.97%` HT, with a `2.63%` Retry share and a `9.998 ms` median gap. No observed radio gap exceeded `61.3 ms`. A diagnostic window immediately afterward reported `100.9` generated packets/s, `101.9` CSI callbacks/s, `94.9` admitted packets/s, and `95%` occupancy.

This result narrowed the failure again. High-rate device-originated traffic is not sufficient to trigger the collapse, because DNS/UDP also originates at the ESP32. Legacy PHY and TID 7 are not sufficient either: both were present without the retry storm or cadence failure previously measured on the C3. The remaining behavior is protocol-, target-, driver-, AP-, and resolver-dependent.

## Findings

The completed tests support these conclusions:

- HTTP keep-alive, SSE, and a seconds-long application mutex hold are not necessary for the failure.
- Diagnostics size increases exposure but is not the root cause; both multi-segment `diagnostics` and sub-MSS `status` stalled under ping.
- HTTPD priority 4 did not prevent ping or DNS/TCP failure. Its contribution to the stable DNS/UDP result was not isolated.
- The recurring `22.5 s` timing and the progression from body stalls to delayed SYN-ACKs place the dominant failure below ordinary HTTP parsing and dispatch.
- DSCP and WMM treatment are not primary causes.
- General heap exhaustion is not required. The combined candidate failed with more than 104 KiB free and no `ENOMEM` events.
- Measured internal DMA-capable heap remained substantial during the static-buffer failure, so its `ENOMEM` events do not establish exhaustion of the general DMA-capable heap.
- Smaller ICMP windows, more dynamic TX buffers, and a larger TCP/IP mailbox are mitigations only. Static TX allocation is not a solution on the tested device.
- The ping failure is not specific to Native; canonical ESPHome failed its readiness gate on the same classic ESP32 and pinned AP.
- External ICMP reproduces the failure without the ESPectre generator task. External UDP does not, but it requires another network service and remains a laboratory control.
- Persistent DNS/TCP is not a viable fallback on the tested setup.
- DNS/UDP passed three controlled Native replicas and maintained steady radio cadence. It is the validated classic ESP32 source for this device and AP, not a universal replacement for other targets.

The strongest remaining inference is a classic-ESP32-specific loss of forward progress in the shared Wi-Fi/lwIP path for some high-rate bidirectional traffic patterns. The evidence does not locate the defect precisely enough to assign it to lwIP, the Wi-Fi driver, radio retransmission behavior, or their interaction. It does rule out further HTTPD, payload-size, SSE, DSCP, and queue-depth tuning as the primary path to a fix.

## Product decision and remaining validation

ESPectre now exposes three explicit internal generators: `ping`, `dns` for UDP, and `dns_tcp`. The shared default remains `ping`; classic ESP32 Native and Matter builds, and the classic ESP32 ESPHome example, select `dns`. The runtime does not fall back automatically because the successful source depends on the device, driver, AP, and resolver.

The runtime and target defaults were compiled successfully for Native and Matter, and the ESPHome classic configuration resolved successfully with `dns`. The hardware soak in this campaign covered Native only. Matter and ESPHome therefore inherit the classic default from the shared runtime and configuration evidence, but still need their own DNS/UDP hardware soak before claiming frontend-specific runtime validation.

External UDP remains a controlled laboratory source. External ICMP remains useful for field diagnosis because it requires no ESPectre-specific service, but its instability prevents it from serving as evidence of a production fix.

One scheduling uncertainty remains: all controlled late-stage runs used Direct HTTPD priority 4. A clean DNS/UDP comparison at priority 1 and 4 would determine whether the classic override still earns its place. Removing it without that comparison would move outside the tested profile; retaining it does not imply that it caused the successful result.

TX128, static TX16 plus A-MPDU, an ESP-IDF 5.5.4 versus 5.5.5 comparison, the proposed `80 pps` classic profile, and the minimal upstream reproducer were not run. DNS/UDP supplied a stronger production result before those experiments became necessary. They remain diagnostic options only if the selected source regresses on a future classic ESP32 environment.

## Related records

- [ADR: standardize managed CSI traffic sources](../adr/2026-08-23-standardize-managed-csi-traffic-sources.md)
- [ADR: retain provenance-filtered CSI admission](../adr/2026-08-28-retain-provenance-filtered-csi-admission.md)
