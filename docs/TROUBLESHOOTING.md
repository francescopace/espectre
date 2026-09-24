# Troubleshooting

Start with the sensing input, then check placement, then tune the detector. If you cannot open the device controls, start with [Device not reachable](#device-not-reachable). These steps apply to the ESPHome, Native, and Matter frontends.

## Contents

- [Check the sensing input](#check-the-sensing-input)
- [Low occupancy](#low-occupancy)
- [No CSI or insufficient input](#no-csi-or-insufficient-input)
- [LAN traffic blocked](#lan-traffic-blocked)
- [Bluetooth reduces CSI occupancy](#bluetooth-reduces-csi-occupancy)
- [Mesh Wi-Fi instability](#mesh-wi-fi-instability)
- [Calibration stalls or startup quality is poor](#calibration-stalls-or-startup-quality-is-poor)
- [Too many false positives](#too-many-false-positives)
- [Missing movements](#missing-movements)
- [Slow response or flickering](#slow-response-or-flickering)
- [Tuning essentials](#tuning-essentials)
- [Device not reachable](#device-not-reachable)

## Check the sensing input

The detector needs a steady supply of valid CSI (channel state information, the Wi-Fi measurement it works on). Open Monitor or the device logs and read the packet rates from left to right:

| Observation | Check next |
|-------------|------------|
| No traffic | Wi-Fi connection and the selected traffic source |
| Traffic but no CSI callbacks | Capture configuration and radio state |
| Callbacks but no accepted packets | Hardware errors and packet filtering |
| Accepted packets but low occupancy | Packet loss, bursts, Bluetooth, and placement |
| Stable input but unstable output | Threshold, motion hits, and detection profile |

The periodic log line looks like this:

```text
mvmt:0.012000 thr:0.500000 | IDLE | tx:100.0 cb:120.0 accepted:99.0 hwerr:2.0 occ:93% | ch:6 rssi:-55
```

- `tx`, `cb`, `accepted`, and `hwerr`: traffic, callbacks, accepted packets, and hardware rejections per second.
- `occ`: **occupancy**, the share of the detector window filled with valid input. Detection needs at least 70%. A high packet rate can still give low occupancy when packets arrive in bursts.
- `ch` and `rssi`: the Wi-Fi channel and signal strength.

Missing values appear as `--`. Field definitions are in [API diagnostics](API.md#diagnostics).

## Low occupancy

1. If traffic or accepted packets are missing, see [No CSI or insufficient input](#no-csi-or-insufficient-input).
2. If packets arrive but occupancy stays low, check [Bluetooth](#bluetooth-reduces-csi-occupancy), [LAN restrictions](#lan-traffic-blocked), and [roaming on mesh networks](#mesh-wi-fi-instability).
3. Recheck [placement](SETUP.md#sensor-placement) before changing detector settings.
4. Once occupancy is stable, repeat the quiet-and-motion test.

Keep `csi_target_pps` at its default of `100`. Changing it changes detector timing (see [detector timing](ALGORITHMS.md#detector-timing)). Lowering the threshold does not fix missing input.

## No CSI or insufficient input

1. Check the Wi-Fi connection and the selected traffic source.
2. Internal traffic: check that the destination replies to the selected protocol. External traffic: check that the sender is running and its packets reach the device.
3. If the traffic does not arrive, see [LAN traffic blocked](#lan-traffic-blocked).
4. If callbacks arrive but few packets are accepted, check the hardware error counters; see [capture quality](CSI.md#capture-quality).

Start with the default `ping` source. If you selected the experimental `wifi_raw` source and input disappears, switch to another source; there is no automatic fallback. See [compatibility limits](CSI.md#compatibility-limits).

A protocol or bandwidth shown as `unavailable` in logs does not mean capture failed; check the packet counters first.

## LAN traffic blocked

- **Internal traffic:** check that the destination accepts ICMP or DNS and that no firewall blocks or rate-limits it. By default the destination is the Wi-Fi gateway; see [traffic destination](SDK.md#traffic-destination) to use another host.
- **External traffic:** check client isolation, guest-network rules, and firewalls between the sender and the ESP32. Devices on different VLANs need a route for this traffic.
- **Multicast:** if multicast does not arrive, try unicast to the device IP. Ports and markers are in [external CSI traffic](API.md#external-csi-traffic).

Discovery uses different traffic from sensing. Blocked mDNS can hide the device while its IP still works; see [discovery fallback](DISCOVERY.md#client-validation-and-fallback).

## Bluetooth reduces CSI occupancy

Bluetooth and Wi-Fi share the same radio. BLE scanning can push CSI packets into bursts, so occupancy drops even when the packet rate looks fine. Compare with Bluetooth disabled before changing the threshold.

If you run an ESPHome Bluetooth proxy, short passive scan windows and disabled software coexistence restore most of the occupancy. See [Bluetooth proxy and CSI occupancy](../src/cpp/frontend/esphome/README.md#bluetooth-proxy-and-csi-occupancy) for the tested YAML and measurements.

## Mesh Wi-Fi instability

Roaming between access points changes the radio path and can make detection unstable. Pin the device to one access point:

1. Open [Device settings](https://espectre.dev/tools/device-settings/) and connect to the device.
2. Refresh the access-point list, select one, and save. The device reconnects.
3. Wait for sensing to be ready, then check occupancy and repeat the motion test.

To remove the pin, choose automatic access-point selection. The SSID and password are kept. The [access-point selection](CLI.md#access-point-selection) commands do the same from the CLI.

After a channel change, the detector restarts its history. Wait for it to be ready before judging the result. A fixed access-point channel helps.

## Calibration stalls or startup quality is poor

Lightweight calibrates on quiet-room input. Missing or bursty input slows it down. Improve occupancy first, then restart the device with the room quiet and wait for calibration to finish.

High Accuracy does not calibrate but still waits for enough valid input. Switching to Lightweight starts a new calibration.

## Too many false positives

Try these in order, one at a time:

1. Look for things that move: fans, curtains, or pets.
2. Check occupancy and placement.
3. Raise the threshold.
4. Increase `motion_on_hits` if short bursts trigger motion.
5. Enable the low-pass filter if the score is still noisy.
6. For Lightweight, recalibrate in a quiet room.

## Missing movements

Check packet flow, occupancy, and placement first. Then:

- If the score moves but stays below the threshold, lower the threshold.
- If motion is detected too late, reduce `motion_on_hits`.
- Try High Accuracy if the device has the CPU and memory for it.

## Slow response or flickering

- If the score crosses the threshold in a quiet room, raise the threshold.
- If motion turns on from brief spikes, increase `motion_on_hits`.
- If motion turns off too early, increase `motion_off_hits`.
- If confirmation is too slow, reduce the matching hit count.

Change these before touching `evaluation_interval_ms` or the detector window. If the score itself is noisy, try the low-pass filter.

## Tuning essentials

Change one setting at a time and repeat the same test. Each frontend guide shows how to apply settings; defaults and ranges are in [shared sensing options](SDK.md#shared-sensing-options).

### Detection profile

- **Lightweight** uses less CPU and memory. Choose it when sensing shares the chip with other work.
- **High Accuracy** detects better and skips calibration, but costs more CPU and memory.

A noisy-link warning after calibration means the metric stays high even in a still room, usually because of a weak signal or interference. Lightweight then misses weaker movement. Improve the signal or switch to High Accuracy.

The frontends remember the selected profile across reboots. Measured results are in the [performance report](performance/README.md).

### Threshold

Raise the threshold to reduce false positives; lower it to catch missed movement. A manual threshold lasts until reboot: then Lightweight calibrates again and High Accuracy returns to its trained default. While a manual threshold is set, Lightweight stops lowering it automatically until the next calibration.

### Filters and timing

Keep the Hampel filter on. Try the low-pass filter only after checking input, placement, threshold, and motion hits. A lower cutoff smooths more but can hide fast motion.

Keep the default `1000 ms` window and `100 pps` rate. Other values change detector timing and have not been validated. See [signal conditioning](ALGORITHMS.md#signal-conditioning) and [motion-hit filtering](ALGORITHMS.md#motion-hit-filtering).

### Recalibration

Recalibrate after changing the room layout. Lightweight collects a new quiet-room baseline in about 10 seconds, so keep the room still. A brief movement extends calibration up to 30 seconds; if the room stays busy, calibration fails and keeps the current threshold. After moving the device, restart it instead: the old threshold no longer describes the room. High Accuracy simply restores its trained threshold.

## Device not reachable

If Device settings or Monitor cannot connect:

1. Check that the device and the browser are on the same network.
2. Use the device IP if the `.local` name does not work.
3. Allow local-network access when the browser asks.
4. Use a desktop browser from the [browser support list](https://espectre.dev/guides/setup/#setup-native-discovery).
5. Check that the page address is `https://espectre.dev`, `https://www.espectre.dev`, or `https://test.espectre.dev`. A local preview of the website needs firmware that accepts that local address.

You can connect with the device IP, its name, its 16-character ID, or the last 6 characters of the ID. Names and short IDs rely on mDNS. If discovery fails, run `./espectre devices`, enter the IP, or look it up in your router's DHCP list. Remove an old saved address before entering a new one. See [browser bootstrap](DISCOVERY.md#browser-bootstrap) for how discovery works.

Once connected, go back to [Check the sensing input](#check-the-sensing-input).
