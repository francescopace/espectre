# Review: classic ESP32 Native managed-traffic stability

- Review date: 2026-08-29
- Status: Concluded
- Scope: Classic ESP32 running the Native and ESPHome frontends
- Outcome: The tested tuning profiles and DNS/TCP fallback were rejected; the production decision remains open

## Purpose

This review preserves the evidence collected while investigating simultaneous CSI loss and Direct HTTP stalls on the classic ESP32. It records a dated experiment, not a durable architecture decision. The managed-traffic policy remains owned by [the existing ADR](../adr/2026-08-23-standardize-managed-csi-traffic-sources.md); that ADR now retains bounded external ICMP as a diagnostic input but does not select it as the production fallback.

The central result is that device-originated managed traffic can drive the classic ESP32 into a degraded Wi-Fi/TCP state even when the general heap remains healthy. DSCP, a smaller ICMP outstanding window, more dynamic Wi-Fi TX buffers, a larger TCP/IP receive mailbox, and statically reserved Wi-Fi TX buffers changed the frequency or timing of failures but did not remove them. The same failure class appeared with ESPHome, and DNS over persistent TCP performed worse than ping. External UDP remained stable in the controlled comparison, but it is a laboratory control rather than a deployable solution because it requires another host on the network.

## Test boundary

The tests used one classic ESP32, primarily the Native frontend, the Lightweight detector, one pinned and verified AP BSSID and channel, and a target traffic rate of `100 pps`. The forensic Direct probe used non-persistent TCP connections, alternated `status` and `diagnostics`, timed the connect, request, first-byte, headers, and body phases separately, and kept SSE disabled. A standard replica used a clean device reset, a `25 s` warm-up, and a `60 s` scored window. The final candidate used three independent boots and a `300 s` scored window per boot. A later canonical ESPHome run used its normal SSE readiness path and failed before scoring.

Timeouts were retained as censored failures: they contribute to the failure rate, but their configured timeout is not treated as an exact latency sample. The host capture was a normal host-side Ethernet capture, not an 802.11 monitor-mode capture. It therefore shows TCP progress on the wire but cannot distinguish Wi-Fi MAC retries from loss or delayed progress inside lwIP and the closed Wi-Fi driver.

The source tree and benchmark tooling evolved during the investigation. Comparisons below include only completed runs with the intended BSSID and configuration, but the campaign is not a clean-revision release benchmark. Incomplete or contaminated attempts are explicitly excluded.

## Internal ICMP versus external UDP

The most discriminating comparison used three replicas per arm at `100 pps`, with the same device, AP, detector, Direct probe, warm-up, and scored duration.

| Traffic source | Direct attempts | Censored failures | CSI samples | Effective rate | Maximum diagnostics interval | `ENOMEM` events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| External UDP, 3 replicas | 360 | 0 | 180 / 180 | 96.02–98.83 pps | 1.033 s | 0 |
| Internal ICMP, 3 replicas | 40 | 2 | 18 / 180 | 68.72–74.88 pps where measurable | 23.966 s | 71 |

All three external replicas completed 120 Direct attempts and received all 60 expected CSI samples. Across their PCAP flows there were no SYN retries, request retransmissions, response retransmissions, header-only responses, missing responses, or header-to-body gaps above one second. The maximum diagnostics header-to-body gap was `33.692 ms`, the maximum first-response latency was `63.277 ms`, and the maximum SYN-ACK latency was `52.694 ms`.

The internal arm degraded in every replica. The first two replicas received only 7 and 11 of 60 CSI samples, with maximum diagnostic intervals of `13.906 s` and `23.966 s`. The third received no scored CSI samples and produced two censored Direct failures. Its capture included SYN retries, request retransmissions, one header-only diagnostics response, and one request with no response. Across the three internal replicas, diagnostics header-to-body gaps reached `22.184 s`; status also stalled, reaching a `4.340 s` first-response delay in the third replica.

This comparison isolates device-originated managed traffic from externally delivered traffic. It does not make external UDP a production candidate.

## External ICMP

A forensic build extended only the external CSI provenance filter to admit unicast ICMP Echo Requests addressed to the ESP32. The Mac sent ping requests at `100 pps`; the ESP32 internal traffic generator remained stopped, the normal lwIP echo responder transmitted the replies, SSE remained disabled, and the BSSID and channel were verified for every boot. The experimental filter was initially removed; the subsequent product decision retained the same bounded packet shape as a supported input to `external`, alongside the existing UDP marker.

| Replica | Direct attempts | Censored failures | CSI samples | Mean admitted rate | Mean occupancy | Maximum diagnostics interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98 | 0 | 49 / 60 | 82.48 pps | 82.80% | 12.64 s |
| 2 | 100 | 0 | 50 / 60 | 83.56 pps | 83.72% | 11.30 s |
| 3 | 8 | 2 | 2 / 60 | 85.97 pps where measurable | 87.00% where measurable | Two requests exceeded 30 s |
| Total | 206 | 2 | 101 / 180 | — | — | — |

No scored replica was clean. The first two reproduced approximately 11–13-second Direct pauses; the third collapsed to two diagnostics samples and two censored failures. There were no ping-generator `ENOMEM` events because that generator was not running. Each serial log contained one `csi:0/0` startup sample before the host ping began and outside the scored window.

This rules out the ESPectre ping-generator task and its raw-socket request path as necessary causes. It does not yet distinguish an ICMP-specific defect from a more general problem caused by high-rate IP transmission from the ESP32: unlike the stable external UDP control, every external Echo Request makes lwIP transmit an Echo Reply. External ICMP is therefore an intermediate result: materially better than internal ping on two boots, but consistently worse than ingress-only external UDP and catastrophic on the third.

## Focused parameter probes

DSCP and ICMP outstanding-window probes used internal ping and the correct pinned BSSID. Only completed replicas are included.

| Profile | Completed replicas | Direct attempts | CSI samples | Maximum diagnostics interval | `ENOMEM` events | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DSCP 0, open loop | 1 | 50 | 25 / 60 | 24.677 s | 42 | Unstable |
| DSCP 0, window 4 | 1 | 54 | 27 / 60 | 16.105 s | 0 | Unstable |
| DSCP 0, window 1 | 2 | 184 | 92 / 120 | 11.456 s | 0 | Unstable |

Changing DSCP did not remove the failure. Limiting outstanding ICMP requests reduced pressure, eliminated `ENOMEM` in these valid runs, and improved average availability, but long CSI gaps and zero-sample windows remained. `ENOMEM` is therefore a useful pressure indicator, not a necessary condition for collapse.

The incomplete second open-loop and window-4 replicas, and the incomplete third window-1 replica, are not used in the totals or conclusions.

## Dynamic Wi-Fi TX buffers

The next controlled comparison changed `CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM` from 32 to 64 while retaining internal ping and the same benchmark shape.

| TX buffers | Replicas | Direct attempts | Censored failures | CSI samples | `ENOMEM` events | Maximum PCAP body gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3 | 58 | 2 | 27 / 180 | 72 | 22.441 s |
| 64 | 3 | 98 | 1 | 48 / 180 | 60 | 22.542 s |

TX64 improved aggregate availability and reduced the number of long body gaps, but it retained the same approximately `22.5 s` worst-case signature and included a catastrophic replica with only 2 of 60 CSI samples. More queue depth increased margin; it did not restore reliable forward progress. TX128 was not pursued because it would consume more memory and could defer the same failure without addressing its mechanism.

## Static Wi-Fi TX buffers

The final queue-allocation probe replaced dynamic TX buffers with 16 statically reserved buffers. A clean build verified `CONFIG_ESP_WIFI_STATIC_TX_BUFFER=y`, `CONFIG_ESP_WIFI_STATIC_TX_BUFFER_NUM=16`, and `CONFIG_ESP_WIFI_TX_BUFFER_TYPE=0`; dynamic TX buffers and TX A-MPDU were disabled, and the TCP/IP receive mailbox remained at 32 entries. The firmware also exposed classic-ESP32-only measurements for free, minimum free, and largest free `MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA` heap.

The first boot completed its `180 s` scored window but failed decisively: 5 of 20 Direct attempts were censored, only 5 of 180 expected diagnostics samples arrived, the maximum diagnostics interval was `75.71 s`, and the serial log contained 79 ping `ENOMEM` events. The pinned BSSID and channel were correct, RSSI remained between approximately `-49` and `-54 dBm` in the received diagnostics, and there were no `csi:0/0` windows. The sampled internal DMA-capable heap remained substantial: free heap was at least `108.82 KiB`, the cumulative low-water mark was `80.62 KiB`, and the largest free block was at least `96 KiB`.

The next two clean boots failed the Direct preflight before scoring. Both associated with the correct BSSID on channel 10 at approximately `-56` to `-57 dBm`, armed CSI, and started ping. The host connected and received the first response byte in less than `160 ms`, but the response did not complete within approximately `30.22 s`. A final probe on the third boot returned the sub-MSS `status` response in `0.37 s`, while `diagnostics` timed out at `5 s`.

Static reservation therefore neither removed ping allocation failures nor restored multi-segment Direct progress. The `3 × 300 s` promotion gate and static-buffer-plus-A-MPDU profile were not run. The target defaults were restored to the previous dynamic TX64 profile after the rejection.

## Final candidate

The final candidate combined all remaining plausible mitigations:

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

No replica passed. The run produced no `ENOMEM`, fatal event, or reboot, yet Direct and CSI both degraded. At least one zero-sample window occurred well inside the scored soak rather than only during startup.

The PCAP covers part of replica 1 and all of replicas 2 and 3. It contains 240 HTTP flows, 50 header-to-body gaps above one second, 19 above five seconds, 10 header-only responses, and a maximum header-to-body gap of `46.456923 s`. The common approximately `22.5 s` delay remained and sometimes doubled. The candidate was rejected on all three boots.

## Frontend comparison

A canonical ESPHome Lightweight run was then built, erased, provisioned through Improv Serial, pinned through `set_wifi_bssid`, and verified on the requested BSSID and channel. The firmware initially calibrated normally, but occupancy declined from the 80–90% range to approximately 30–60%, followed by repeated ping `ENOMEM` events. The benchmark stopped after `54.6 s` because Direct could not obtain five consecutive ready, non-zero CSI samples, so the planned `300 s` scored window never began.

This run does not quantify steady-state ESPHome performance, but it is sufficient to show that the collapse is not specific to Native frontend orchestration. ESPHome uses the same runtime, CSI capture service, traffic generator, and ESP-IDF Wi-Fi/lwIP path while adding its own API and component workload.

## DNS over persistent TCP

Native was restored and tested with the production DNS mode selected through `set_traffic_generator_mode`. The mode sends length-prefixed DNS queries through one persistent, non-blocking TCP connection to gateway port 53. Each replica explicitly persisted the BSSID through `set_wifi_bssid`, reset the device, verified the active BSSID and channel after boot, observed RSSI between approximately `-51` and `-56 dBm`, kept SSE disabled, warmed up for `25 s`, and scored for `60 s`.

| Replica | Direct attempts | Censored failures | CSI samples | Mean admitted rate | Mean occupancy | Maximum diagnostics gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 1 | 5 / 60 | 11.31 pps | 20.2% | 12.83 s |
| 2 | 42 | 0 | 21 / 60 | 40.34 pps | 43.3% | 12.02 s |
| 3 | 36 | 1 | 17 / 60 | 22.97 pps | 28.1% | 33.68 s |
| Total | 90 | 2 | 43 / 180 | 30.10 pps across received samples | 34.6% across received samples | 33.68 s |

The serial evidence contains 59 `csi:0/0` windows, 103 DNS send failures with `EAGAIN`, 31 TCP connection-establishment events, and no DNS `ENOMEM`. Calibration did not complete usefully in any replica, and detector timing remained unavailable. DNS/TCP therefore fails independently of low RSSI and general heap allocation failure. It is not a viable fallback on the tested AP and implementation.

## Findings

The evidence supports these conclusions:

- HTTP keep-alive is not required for the failure because the timed client opened a new TCP connection for every request.
- SSE is not required because it was disabled throughout the discriminating runs.
- Diagnostics size increases exposure but is not the root cause. The diagnostics body spans multiple TCP segments, while `status` fits below the measured MSS; both paths nevertheless stalled under internal traffic.
- A control-plane application deadlock does not explain the captures. In many failures, HTTP headers reached the host promptly and the body then stopped progressing. Later failures also delayed SYN-ACKs, request delivery, or the first response.
- DSCP and WMM treatment are not the primary cause.
- General heap exhaustion is not required. The final candidate failed with more than 104 KiB of free heap and no `ENOMEM` events.
- Internal DMA-capable heap exhaustion or fragmentation was not visible in the static TX16 samples; substantial free space and a 96 KiB largest block coexisted with 79 ping `ENOMEM` events and Direct stalls.
- More Wi-Fi TX buffers, a larger TCP/IP mailbox, and a one-packet ICMP window are mitigations only.
- Statically reserving Wi-Fi TX buffers is not a solution on the tested device.
- The failure is not specific to Native; canonical ESPHome failed its readiness gate on the same classic ESP32 and pinned AP.
- Persistent DNS/TCP does not avoid the failure and performed worse than ping in three valid replicas.
- External UDP demonstrates that CSI sensing and Direct HTTP can coexist on this device under comparable externally delivered traffic. It does not satisfy the production constraint of requiring no additional network service.
- External ICMP reproduces Direct stalls without the ESPectre traffic-generator task, while still requiring the ESP32 to transmit an IP reply for every request.

The strongest remaining inference is a classic-ESP32-specific loss of forward progress in the shared Wi-Fi/lwIP path when CSI and device-originated managed traffic are active. The available captures do not locate the boundary precisely enough to assign the defect to lwIP, the Wi-Fi driver, radio retransmission behavior, or their interaction. They do, however, rule out further HTTPD, payload, SSE, DSCP, and queue-depth tuning as justified next steps.

## Decision boundary

This review does not replace internal ping or choose the production fallback. The public traffic-source decision remains owned by the managed-traffic ADR, which now permits external ICMP as a dependency-free diagnostic input to `external` alongside the existing UDP marker. The evidence is sufficient to reject DNS/TCP on this setup and stop tuning the failed profile.

External UDP may remain a laboratory control, but it must not be presented as a production requirement. External ICMP is available for field diagnostics without an additional network service; its observed instability means it is not evidence of a production fix. Any production fallback still requires multiple clean boots, a soak longer than the observed collapse time, no censored Direct failures, no `csi:0/0` windows, and no body gap above one second.

## Artifact provenance

Raw captures and run logs remain under `data/untracked/firmware_benchmarks/` and are intentionally not versioned. The key results and hashes are recorded here so a local artifact can be authenticated.

| Experiment | Local artifact | SHA-256 |
| --- | --- | --- |
| Internal ICMP versus external UDP | `20260829-esp32-ab-100pps-3pairs/valid2/network.pcap` | `b574f955fc3dd45ba77b13f1a612f8586459ab29d5c69a9c47ca0d81a1b21346` |
| TX32 | `20260829-esp32-txbuf-ab/tx32/network.pcap` | `fc1d55edd5b3b243e24ef08828acf3d450fbe741514d6c51d540f9b91a517abd` |
| TX64 | `20260829-esp32-txbuf-ab/tx64/network.pcap` | `61d4bba49abfd921cff9c916e6fc25a7aef4e07ade5c04c036cf47152f99761b` |
| Final candidate | `20260829-esp32-final-candidate/network.pcap` | `920eb8757d6cf92400bd8fd0665afa92539528a8bc7dc875674712bcba619cbe` |
| Static TX16, scored boot | `20260829-esp32-native-static-tx16-3x180/replica-1-internal-analysis.json` | `3686add0e610cefa83c7184712f715dab21a7ad7cce7113163163b3fd68af681` |
| Static TX16, second-boot preflight | `20260829-esp32-native-static-tx16-3x180/replica-2-internal-radio-pin.json` | `f55374af227e89cc2e1f1040c025793d369f21b76db610d8d3dd2ab5f8266f5e` |
| Static TX16, third-boot preflight | `20260829-esp32-native-static-tx16-3x180/replica-3-internal-radio-pin.json` | `adc43796955b5167cca7c1d7890b585adc6f5650c0fd95c5f0c0f66b319ef713` |
| External ICMP, replica 1 | `20260829-esp32-native-external-icmp-3x60-valid/replica-1-external_icmp-analysis.json` | `ce4a858d3bb861c423bd57c17f831c16620b8b625d921b27eedfc553f4aeebd7` |
| External ICMP, replica 2 | `20260829-esp32-native-external-icmp-3x60-valid/replica-2-external_icmp-analysis.json` | `22f60ba04b8a0a8c8733d497f9e46dfc6564dcc26ef2b06c0e566c92394da623` |
| External ICMP, replica 3 | `20260829-esp32-native-external-icmp-3x60-valid/replica-3-external_icmp-analysis.json` | `9a371ea8d9522cbb4c024390ebe81a359a4b139483f1a42eaca73afd88fb9883` |

The machine-readable final profile and verdict are stored locally as `20260829-esp32-final-candidate/profile.json` and `20260829-esp32-final-candidate/summary.json`.

The ESPHome readiness failure is stored as `20260829T152132+0200-esp32-f2df6f5bc6bd/esphome-lightweight/analysis.json`; its firmware SHA-256 is `5be1605919f4bff397d61199c805412e38567ce09ed3b3be2c0a140f24fbe16b`. The three DNS/TCP results and serial logs are stored under `20260829-esp32-native-dns-tcp-3rep-valid/`. No PCAP was collected for those two follow-up experiments.

## Follow-up

Further work should start from a clean firmware revision and a fresh baseline. The next product-oriented experiment is a classic-ESP32-specific `80 pps` profile, with both the generator and temporal CSI grid set to 80 pps. If it passes the full corpus and hardware gates, the managed-traffic ADR can evaluate that target-specific cadence. The remaining diagnostic choices are an ingress UDP test that makes the ESP32 transmit one UDP reply per packet, which would separate ICMP from generic high-rate IP TX, or an otherwise identical ESP-IDF 5.5.4 versus 5.5.5 comparison. A minimal classic-ESP32 reproducer follows if those comparisons are inconclusive. No further queue-depth tuning is justified. A monitor-mode 802.11 capture is optional forensic work if distinguishing over-the-air retry behavior from local driver or lwIP stalls becomes necessary.

## Related records

- [ADR: standardize managed CSI traffic sources](../adr/2026-08-23-standardize-managed-csi-traffic-sources.md)
- [ADR: retain provenance-filtered CSI admission](../adr/2026-08-28-retain-provenance-filtered-csi-admission.md)
