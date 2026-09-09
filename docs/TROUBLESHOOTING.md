# Troubleshooting

Start with CSI occupancy and traffic, then check placement and detector behavior. Use the browser connectivity section if you cannot reach the device controls. These procedures cover the maintained C++ frontends; Micro-ESPectre users can apply the shared diagnostic checks, with configuration and controls described in its [README.md](../src/python/micro_espectre/README.md).

## Contents

- [Low occupancy](#low-occupancy)
- [Bluetooth reduces CSI occupancy](#bluetooth-reduces-csi-occupancy)
- [No CSI or insufficient input](#no-csi-or-insufficient-input)
- [LAN traffic blocked](#lan-traffic-blocked)
- [Check the sensing input](#check-the-sensing-input)
- [Mesh Wi-Fi instability](#mesh-wi-fi-instability)
- [Calibration stalls or startup quality is poor](#calibration-stalls-or-startup-quality-is-poor)
- [Too many false positives](#too-many-false-positives)
- [Missing movements](#missing-movements)
- [Slow response or flickering](#slow-response-or-flickering)
- [Tuning essentials](#tuning-essentials)
- [Device not reachable](#device-not-reachable)

## Low occupancy

Occupancy measures how much of the detector window contains valid CSI input. Detection requires at least 70% valid occupancy. A high packet rate can still leave the detector unready when packets arrive in bursts.

1. Compare traffic, callback, accepted-packet, and occupancy readings using the [log example](#check-the-sensing-input).
2. If traffic or accepted input is missing, follow [No CSI or insufficient input](#no-csi-or-insufficient-input).
3. If packets arrive but occupancy stays low, check loss, burst delivery, [Bluetooth activity](#bluetooth-reduces-csi-occupancy), and [LAN restrictions](#lan-traffic-blocked). Recheck placement using [SETUP.md](SETUP.md#sensor-placement) before compensating with detector parameters. On a mesh network, check whether the device is [roaming between access points](#mesh-wi-fi-instability).
4. Repeat the quiet-and-motion test once occupancy is stable and the detector is ready.

Keep `csi_target_pps` at its production default of `100` while repairing the traffic path. Changing it alters detector timing and requires validation at the chosen cadence; [ALGORITHMS.md](ALGORITHMS.md#detector-timing) explains that constraint. Lowering the detection threshold cannot repair missing input.

## Bluetooth reduces CSI occupancy

Bluetooth and Wi-Fi share radio time. BLE scanning can leave CSI packets concentrated in bursts: accepted input may remain near or above the target rate while admitted input and occupancy fall. Compare with Bluetooth disabled before changing detector thresholds. Callbacks that are all filtered, with no accepted input, require the separate [capture checks](#no-csi-or-insufficient-input); the measurements below do not explain that failure.

For an advertisement-only ESPHome Bluetooth proxy, try short passive scan windows if BLE must remain enabled. The ESPHome [README.md](../src/cpp/frontend/esphome/README.md#bluetooth-proxy-and-csi-occupancy) provides the tested YAML and rebuild procedure. In ESPHome 2026.8.2 with ESP-IDF 5.5.5, setting the tracker's `software_coexistence: false` alone left ESP-IDF software coexistence enabled, including after a clean build. Explicitly disabling `CONFIG_ESP_COEX_SW_COEXIST_ENABLE` produced the largest occupancy improvement in this experiment. This disables software arbitration, not Bluetooth or radio contention.

### ESP32-S3 measurements

The investigation for [issue #165](https://github.com/francescopace/espectre/issues/165) included 18 runs on September 9, 2026, using one ESP32-S3 at 240 MHz, ESPHome 2026.8.2, ESP-IDF 5.5.5, and local ESPectre `develop` builds. The board position and AP BSSID were fixed, with internal `wifi_raw` traffic at 100 pps. Each run lasted 60–120 seconds; occupancy averages exclude its first 15 seconds. Repeated runs are listed separately within each row.

| Scan window / interval | Scan type | Tracker software coexistence | ESP-IDF software coexistence | Mean CSI occupancy per run |
|------------------------|-----------|------------------------------|------------------------------|----------------------------|
| BLE disabled | — | — | — | 94.1%, 88.8%, 94.0% |
| 320 / 320 ms | Active | Enabled | Enabled | 52.3% |
| 30 / 320 ms | Passive | Enabled | Enabled | 53.2% |
| 10 / 100 ms | Passive | Enabled | Enabled | 53.3%, 54.4% |
| 5 / 100 ms | Passive | Enabled | Enabled | 53.6%, 51.9% |
| 10 / 100 ms | Active | Enabled | Enabled | 54.0% |
| 10 / 100 ms | Passive | Disabled | Enabled | 56.7% |
| 5 / 100 ms | Passive | Disabled | Enabled | 56.5% |
| 30 / 320 ms | Passive | Disabled | Explicitly disabled | 84.4% |
| 10 / 100 ms | Passive | Disabled | Explicitly disabled | 88.5%, 87.6% |
| 5 / 100 ms | Passive | Disabled | Explicitly disabled | 93.4%, 91.1%, 92.6% |

Passive scanning for 5 ms every 100 ms, with both software coexistence settings disabled, had the lowest measured CSI impact among the BLE configurations. The final two-minute run used a fresh boot with static YAML: calibration completed, scored occupancy ranged from 89% to 96%, and the proxy received 433 advertisements from 26 distinct addresses over the full run. The 10 ms window received roughly twice as many advertisements in this environment, at the cost of lower CSI occupancy. The nominal window-to-interval ratio alone did not predict the impact.

These are short measurements on one board and network, not a general BLE compatibility or motion-accuracy result. Active GATT proxy connections were not tested. Keep the workaround experimental, compare BLE reception as well as CSI occupancy on the target installation, and repeat quiet-and-motion checks after input becomes stable.

## No CSI or insufficient input

1. Check the Wi-Fi connection, selected traffic source, and CSI-enabled build configuration. If protocol or bandwidth appears as `unavailable` in logs, check packet counters before concluding that capture failed.
2. For internal traffic, verify that the selected destination replies to the chosen protocol. For external traffic, verify that the host source is running and its packets can reach the device.
3. If the source is active but traffic does not reach its destination, check [LAN restrictions](#lan-traffic-blocked).
4. If callbacks arrive but few packets are accepted, inspect hardware-quality errors and frame identity filtering. Use [CSI.md](CSI.md#capture-quality) for the validation rules.

Start with internal `ping`. If you selected experimental `wifi_raw` and usable input disappears, select another generator explicitly; the runtime has no automatic fallback. [CSI.md](CSI.md#compatibility-limits) records the tested hardware limits.

## LAN traffic blocked

Check the network path used by the selected source:

- For internal traffic, check whether the destination accepts the selected ICMP or DNS protocol and whether network rules block or rate-limit it. An empty traffic destination uses the Wi-Fi gateway; [SDK.md](SDK.md#traffic-destination) describes how to select another reachable host.
- For external traffic, check client isolation, guest-network restrictions, and firewall rules between the sending host and the ESP32. Devices on different VLANs need a permitted path for the selected traffic.
- If multicast delivery fails, compare unicast traffic to the device's current IP. [API.md](API.md#external-csi-traffic) defines the accepted destinations, UDP port, and marker.

Check sensing occupancy after each network change. Discovery and sensing use different traffic: blocked mDNS can prevent discovery while Direct HTTP remains reachable by IP. [DISCOVERY.md](DISCOVERY.md#client-validation-and-fallback) describes discovery restrictions.

## Check the sensing input

Open Monitor or the device logs and check calibration, movement score, threshold, and motion state. CSI means channel state information; the detector needs a steady supply of valid CSI to produce usable motion data.

Read the packet rates in sequence:

| Observation | Check next |
|-------------|------------|
| No traffic | Wi-Fi connection and the selected traffic source |
| Traffic without CSI callbacks | Capture configuration and radio state |
| Callbacks without accepted packets | Hardware-quality errors and frame identity filtering |
| Accepted packets with low occupancy | Packet loss, bursts, Bluetooth activity, and sensor placement |
| Stable input with unstable output | Threshold, motion-hit settings, and detector profile |

Accepted packets have passed capture and identity checks. Admitted packets are the detector input after temporal admission. Occupancy measures how much of the detector window contains valid input; high packet rates can still leave gaps when packets arrive in bursts.

The periodic sensing log includes:

```text
mvmt:0.012000 thr:0.500000 | IDLE | tx:100.0 cb:120.0 accepted:99.0 hwerr:2.0 occ:93% | ch:6 rssi:-55
```

`tx`, `cb`, `accepted`, and `hwerr` are traffic, callback, accepted-packet, and hardware-rejection rates per second. `occ` is detector-window occupancy, and `ch` and `rssi` describe the Wi-Fi link. Missing values appear as `--`. Use [API.md](API.md#diagnostics) for counter definitions and [CSI.md](CSI.md#capture-quality) for capture validation.

## Mesh Wi-Fi instability

Roaming between access points can change the radio path, channel, and packet delivery. If these changes cause unstable detection, pin the device to a specific BSSID through its advertised Wi-Fi controls:

1. Open [Device settings](https://espectre.dev/tools/device-settings/) and connect to the device.
2. Refresh the access-point list, select the BSSID, and save. The station reconnects when it changes access point.
3. Check occupancy and repeat the motion test after sensing becomes ready.

Choose automatic access-point selection to clear a stale pin after replacing or removing an access point. This keeps the SSID and password. [CLI.md](CLI.md#access-point-selection) covers the equivalent commands; each frontend README describes persistence and recovery.

After a Wi-Fi channel change, allow the runtime to reset detector history and collect fresh valid coverage before evaluating the result. Prefer a fixed access-point channel when possible.

## Calibration stalls or startup quality is poor

Check occupancy, improve the radio path if needed, and boot Lightweight with the room quiet. Missing or burst-concentrated input extends startup because it supplies too little valid evidence. Wait for calibration and readiness before testing motion.

High Accuracy skips quiet-room threshold calibration but still needs CSI readiness and feature-window warmup. A profile change resets the threshold; switching to Lightweight starts calibration.

## Too many false positives

Try in this order:

1. Check for environmental movement, such as fans, curtains, or pets.
2. Verify occupancy and placement.
3. Raise the threshold.
4. Increase `motion_on_hits` if only short bursts become alarms.
5. At the default 100 PPS cadence, enable or tune the low-pass filter if the score remains noisy.
6. For Lightweight, recalibrate in a quiet room.

Repeat the same quiet-and-motion test after each change.

## Missing movements

Check packet flow, occupancy, and placement first. If the movement score responds but does not cross the threshold, lower the threshold. If it crosses the threshold but the motion state changes too slowly, reduce `motion_on_hits`. Compare High Accuracy when its additional CPU and memory cost fits the device.

## Slow response or flickering

With stable occupancy, raise the threshold if the score repeatedly crosses it in a quiet room. Increase `motion_on_hits` to reject brief motion readings, or increase `motion_off_hits` to keep short idle readings from clearing motion. Reduce the corresponding hit count when confirmation is too slow.

Tune these controls before changing `evaluation_interval_ms` or the detector window. If the score itself remains noisy, try the low-pass filter at the default 100 PPS cadence.

## Tuning essentials

Change one setting at a time and repeat the same test. The frontend README describes how to apply settings, and [SDK.md](SDK.md#shared-sensing-options) lists the shared defaults and ranges.

### Detection profile

Choose Lightweight when the surrounding firmware needs lower detector CPU and working-memory use. Choose High Accuracy when detection quality is the priority and the additional cost fits. The maintained C++ frontends persist an accepted runtime profile selection. Current measurements and limitations are in [README.md](performance/README.md).

### Threshold

Raise the threshold to reduce false positives; lower it to catch missed movement. A manual threshold applies to the current session. After reboot, Lightweight calibrates again and High Accuracy restores its trained default. A Lightweight override suspends automatic threshold lowering until recalibration or explicit adaptive-threshold application.

### Filters and timing

Keep Hampel filtering enabled unless a controlled comparison shows it removes useful motion detail. Try the low-pass filter only after checking input quality, placement, threshold, and motion hits. A lower cutoff adds smoothing and can hide fast motion; a higher cutoff preserves more short-term variation.

Keep the production `1000 ms` detector window and `100 pps` target for routine tuning. Other settings change the feature timing and need detector validation. [ALGORITHMS.md](ALGORITHMS.md#signal-conditioning) explains filter behavior and [ALGORITHMS.md](ALGORITHMS.md#motion-hit-filtering) describes evaluation and confirmation timing.

### Recalibration

Recalibrate after a material placement or radio-environment change when the frontend offers the control. Lightweight starts a fresh threshold calibration; keep the room quiet. High Accuracy immediately restores its trained threshold without collecting a quiet-room window.

## Device not reachable

The published ESP-IDF frontends expose Direct HTTP on the local network. If Device settings or Monitor cannot connect:

1. Confirm that the device and browser are on the same LAN.
2. Try the current private IPv4 address if the `.local` hostname does not resolve.
3. Grant the browser's local-network permission when prompted.
4. Use a desktop Chromium browser listed in the current [browser support matrix](https://espectre.dev/guides/setup/#setup-native-discovery) when another browser blocks hosted HTTPS-to-local-HTTP access.
5. Confirm that the hosted page uses `https://espectre.dev`, `https://www.espectre.dev`, or `https://test.espectre.dev`. A local website preview also requires firmware that accepts the corresponding loopback origin.

Device settings and Monitor accept a private IP, device name, full 16-character device ID, or the last 6 characters of that ID. A full ID maps to the device's unique local address; a name or short ID uses the same bounded discovery as the **Auto-discovery** button. One match connects directly, while multiple matches require an explicit selection. Names, short IDs, and `.local` addresses depend on working mDNS. If discovery fails, use `./espectre devices`, enter the current IP, or check the router's DHCP lease table. Remove a stale remembered endpoint before entering a replacement address. [DISCOVERY.md](DISCOVERY.md#browser-bootstrap) owns the peer-discovery contract.

Origin, mixed-content, and local-network permission errors come from the browser boundary rather than the detector. Grant local-network access only to the ESPectre portal, confirm that the device remains on the same trusted LAN, and retry with a browser in the support matrix. Once Direct connects, return to the [occupancy checks](#low-occupancy).
