# ADR: Defer ESP32-C6 sensing from access-point Block ACKs

- Status: Accepted
- Date: 2026-11-21

## Decision

Keep `wifi_raw` disabled on ESP32-C6. Preserve the associated-AP Block ACK experiment as a candidate for a future C6 traffic source. The diagnostic achieved 100 usable CSI reports/s on one AP with MQTT and HTTP active. Three constraints limit its use in the product: it measures the AP-to-C6 link, it depends on private ESP-IDF 6.1.1 Wi-Fi driver details, and the integrated firmware has yet to demonstrate the result on hardware.

An ordinary `wifi_raw` sender can direct a Null Data frame to another associated client on the same channel. That peer can measure the incoming frame in `external` mode. The C6 receives an ordinary 14-byte ACK in return, which yielded no usable CSI in this experiment. Entering the peer's MAC does not establish a direct QoS aggregation agreement between the clients. For an existing C6 deployment that needs to sense a second ESP directly, the demonstrated arrangement is to put the C6 in `external`/HT20 and have the other ESP send it padded MCS0 Null Data. An ESP acting as the C6's AP would be a separate hardware trial.

## Evidence

The diagnostic ran on an ESP32-C6 revision 0.1, ESP-IDF 6.1.1, and one associated 2.4 GHz HT20 AP. The AP configuration stayed fixed and promiscuous mode stayed off. There was no gateway request or second ESP in the measurement exchange. MQTT and HTTP were independent load generators.

| Test | Observed result |
| --- | --- |
| Ordinary ACK bootstrap | 71 addressed 14-byte ACKs, zero valid CSI estimates. All 71 identified the same station receive slot. |
| AP response rate | With automatic uplink rate, 0/1,054 Block ACKs at 24 Mbps had valid CSI. With MCS1 uplink, 0/605 at 12 Mbps were valid. With MCS0 uplink, 645/645 at 6 Mbps were valid. The AP chose these rates. |
| Minimum submission size | Every tested size from 280 down to 14 bytes sustained a 100/s average. At 14 bytes, four 30-second windows, including a fresh association, yielded 12,000 usable reports from 12,120 successful submissions. Fourteen bytes is a complete Ethernet header with zero payload. |
| Attribution control | With probes off and application traffic also driven at MCS0, the two 8-second control windows produced 412 and 402 valid Block ACKs on TID 0. The TID 1 measurement filter accepted zero. Rate, addresses, and CSI validity alone would have counted application traffic. |
| Probes and both services | Two scored 30-second windows produced 3,000 usable TID 1 reports each from 3,073 and 3,071 submissions. The filter excluded 3,086 other Block ACKs. All 60 seconds had both MQTT and HTTP activity, with 1,659 QoS 1 PUBACKs and 1,167 successful HTTP transactions inside the scored windows. |

The mixed run averaged 100 usable reports/s, with 98–102 in individual one-second bins and a longest gap of 31.6 ms. These counts describe radio callbacks after frame and CSI validation. Detector slot occupancy, motion accuracy, and sustained operation of a production frontend require their own measurements.

## Reimplementation notes

### Establish the measurement link

1. Gate the backend and its linker options on C6, **exactly ESP-IDF 6.1.1**, and TX A-MPDU. The lab build used `CONFIG_ESP_WIFI_CSI_ENABLED=y`, `CONFIG_ESP_WIFI_AMPDU_TX_ENABLED=y`, `CONFIG_ESP_WIFI_TX_BA_WIN=12`, and `CONFIG_FREERTOS_HZ=1000`. It also enabled RX A-MPDU. The compiled frontend candidate kept RX A-MPDU disabled, so repeat the hardware test with the actual product defaults. Enabling TX aggregation also affects ordinary station traffic and its fixed TX rate policy.
2. Use an associated station, power save off, 2.4 GHz 11b/g/n, and 20 MHz bandwidth. Read the actual BSSID with `esp_wifi_sta_get_ap_info()`, the local MAC with `esp_wifi_get_mac(WIFI_IF_STA, ...)`, and verify `WIFI_PHY_MODE_HT20` with `esp_wifi_sta_get_negotiated_phymode()`. Record chip revision, SDK version, channel, RSSI, and BSSID continuity for each window.
3. Submit an Ethernet-II frame through `esp_wifi_internal_tx(WIFI_IF_STA, frame, 14)`. Bytes 0–5 are the associated BSSID, bytes 6–11 the station MAC, and bytes 12–13 are `88 b5`, the experimental EtherType in network order. There is no payload, IP address, UDP port, or marker. The Wi-Fi driver supplies the QoS MAC framing, security overhead as applicable, and FCS. It negotiates aggregation with the AP and transmits QoS data on TID 1. One compressed Block ACK can acknowledge several MPDUs.

### Mark the probes and choose their uplink rate

The diagnostic used GNU linker wrappers for three symbols from the C6 Wi-Fi libraries. Call each original function first. The following byte offsets belong to the 32-bit C6 libraries shipped with **6.1.1**. Recheck the call sites, layouts, linked symbols, and hardware controls before porting to another SDK.

| Linker option | Operation verified in the diagnostic |
| --- | --- |
| `--wrap=ic_ebuf_alloc` | Call `__real_ic_ebuf_alloc(src, kind, len)`. Match the exact buffer being submitted by the probe task, length 14, kind 1 or 9, both MACs, and EtherType. Read the descriptor pointer from byte offset 52 of the returned ebuf using `memcpy`. For a non-null descriptor with original priority 15 (automatic) or 0, set `descriptor[4] = (descriptor[4] & 0xf0) \| 1`. Clear the submission marker on every return from TX. |
| `--wrap=rcGetRate` | After `__real_rcGetRate(control, tx)`, restrict the override to an active session, non-null arguments, `(tx[4] & 0x0f) == 1`, and an existing HT MCS0–7 selection with long or short GI. Write `WIFI_PHY_RATE_MCS0_LGI` to `tx[12]`. |
| `--wrap=rcGetAmpduSched` | Apply the same check and rate after `__real_rcGetAmpduSched(control, tx)`. Also call `rcSetTxAmpduLimit(control, rx11NRate2AMPDULimit(WIFI_PHY_RATE_MCS0_LGI))` so the aggregate limit agrees with the selected rate. |

MCS0 with long GI is **6.5 Mbps on the uplink**. The successful **6 Mbps downlink** was legacy OFDM, as reported in the Block ACK's RX metadata. Scope the override to probe traffic. The diagnostic's MQTT and HTTP sockets used TID 0. A product must also handle applications that explicitly request TID 1. The reverted candidate redirected conflicting station traffic to TID 2 in the same WMM background category. Validate explicit priority and IPv4/IPv6 traffic-class mapping on hardware before claiming exclusive attribution. Keep the TX hooks suitable for their Wi-Fi execution context, and synchronize submission and session state across the worker and callback.

### Deliver addressed Block ACKs to the CSI callback

The MAC could process Block ACKs while the CSI callback saw none. Setting `WIFI_PROMIS_CTRL_FILTER_MASK_BA` in the stored control filter with promiscuous mode off did not change the effective receive-slot mask. Reapplying CSI configuration did not fix it either.

1. Register the CSI callback and enable legacy CSI acquisition. The diagnostic also requested HT acquisition for observation and set `dump_ack_en` for bootstrap. For the bootstrap only, configure raw TX with `esp_wifi_config_80211_tx()` for 11g at `WIFI_PHY_RATE_6M`. Send addressed 24-byte Null Data frames to the associated AP with `esp_wifi_80211_tx(WIFI_IF_STA, frame, 24, true)`. The diagnostic used 70 sends spaced 12 ms apart.
2. Inspect ordinary ACK headers **before** rejecting invalid CSI estimates. Require ACK subtype (`hdr[0] == 0xd4`), receiver address equal to the station, and exactly one of `rx_ctrl.rxmatch0` and `rx_ctrl.rxmatch1`. The diagnostic required more than ten observations on one slot and zero on the other. Stop if the slot is ambiguous. Slot 0 happened to match on this board.
3. Read `mask = hal_get_dump_ctrl_frame_cfg(slot)` and call `hal_set_dump_ctrl_frame_cfg(slot, mask | (1U << 9))` on that slot only. Check the effective masks and confirm `esp_wifi_get_promiscuous()` remains false. Track whether this session added the bit. On exit, clear only an owned bit from the current mask, preserving the other subtype bits.
4. To diagnose a silent callback, optionally wrap `hal_mac_tx_get_blockack(queue, result)` and observe successful original calls. The 6.1.1 result layout was `uint8_t tid, pad`, `uint16_t ssn`, then two `uint32_t` bitmap words (12 bytes total). A changing sequence/bitmap establishes MAC-level receipt. Compare that counter to addressed BA callbacks, valid estimates, and admitted samples to locate the drop. The diagnostic hook is unnecessary in the sensing path.

### Admit one usable report and pace the next probes

Read `wifi_csi_info_t.hdr` only after checking the pointer and `rx_ctrl.sig_len`. The accepted single-TID compressed BA is **32 bytes including FCS**. The separate CSI buffer contains **128 bytes of I/Q data**.

| Field | Acceptance check |
| --- | --- |
| Frame identity | `hdr[0] == 0x94`, station receiver address at offset 4, and associated AP transmitter address at offset 10. |
| BA control | At offset 16, require single-TID compressed BA. The diagnostic checked `(hdr[16] & 6) == 4`, `(hdr[17] & 15) == 0`, and `(hdr[17] >> 4) == 1`. The integrated candidate used the stricter `(hdr[16] & 0xfe) == 4` and `hdr[17] == 0x10` for this format. |
| Freshness | Require a nonzero 8-byte bitmap at offsets 20–27. Compare the 10 bytes at offsets 18–27 (starting sequence control plus bitmap) with the last admitted report. Drop an identical repeat. Reset the key for each association. |
| PHY and quality | Require `cur_bb_format == RX_BB_FORMAT_11G`, `rate == WIFI_PHY_RATE_6M`, `rx_channel_estimate_info_vld`, zero `rx_state` and `rxend_state`, non-null `buf`, and `len == 128`. Skip the first four bytes if `first_word_invalid`. In the remaining I/Q pairs require at least 16 with either component nonzero and at least four differing from the first usable pair. |

Use a phase-preserving 10 ms timer for the 100/s target. At tick `i` (starting at zero), take a synchronized snapshot of the cumulative fresh usable report count and submit `clamp((i + 1) - usable_reports, 0, 2)` probes using signed arithmetic. Count each successful submission separately. Advance `usable_reports` only after all checks pass. Reset both counters together at the start of a measurement window. The existing temporal sampler should still use the Wi-Fi RX timestamp for admission to the detector grid.

## Reopening the decision

- Start with a full flash backup of the identified C6, verify it against the device, and use a temporary diagnostic build. After testing, restore the original flash and verify it byte-for-byte or by device digest. The original 8 MiB image passed the restore verification in this experiment.
- Reproduce the ordinary ACK and rate controls, then run probe-only, application-only at the same rate, and mixed-load windows. Count successes inside the same timed windows as CSI. Record TID histograms, effective RX rates, RX errors, repeated reports, per-second bins, maximum gaps, MQTT PUBACKs, and HTTP successes. A useful reproduction gets zero measurement admissions with the probes off, and approximately 100 fresh reports/s with both services active.
- Integrate in the [shared traffic generator](../../src/cpp/runtime/esp_idf/traffic_generator_service.cpp), the [early CSI callback](../../src/cpp/runtime/esp_idf/csi_capture_service.cpp), and the [frame identity filter](../../src/cpp/runtime/esp_idf/csi_frame_identity.cpp). Keep the existing LLTF quality, normalization, and detector timing paths. Stop and release the slot on worker exit and partial startup failure. On disconnect, roam, or target change, discard the old addresses, slot, counters, and duplicate key before starting again.
- Gate the source and private linker options in the [SDK source groups](../../src/cpp/espectre_sources.cmake) and [sensing Kconfig](../../src/cpp/runtime/esp_idf/espectre_config/Kconfig.projbuild). Verify Native, Matter, and the upstream ESPHome validator, select options, and SDK configuration against the same version and aggregation requirements. If an explicit MAC remains editable, specify separately what the associated BSSID and another station MAC do. A capability advertised to MQTT, Direct HTTP, or the website must agree with the linked firmware.
- The reverted candidate passed host checks, ESPHome code generation, frontend configuration checks, and a C6 link test with the three wrappers present. Only the diagnostic ran on silicon. Before changing the production compatibility decision, measure the integrated detector input under load, exercise stop/restart and roaming, repeat on additional APs, and verify the existing Null Data path on other chips.

## Related

- [Managed CSI traffic sources and the original short-frame investigation](2026-08-23-standardize-managed-csi-traffic-sources.md)
- [Current CSI compatibility limits](../CSI.md#compatibility-limits)
- [Temporal CSI admission](2026-08-15-use-fixed-temporal-csi-admission.md)
