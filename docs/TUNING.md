# Tuning Guide

Use this operational guide after a device is installed and producing motion data. It explains what to change, in what order, and what result to expect. Detailed detector formulas and validation evidence remain in [ALGORITHMS.md](ALGORITHMS.md) and the generated [performance report](performance/README.md).

Inline snippets use ESPHome YAML as a concrete example. CSI means channel state information. Raw accepted `pps` is the capture supply used by traffic control; admitted `pps` is the detector input after temporal slot admission.

## Quick Start

### 1. Boot In A Quiet Room

For the default `lightweight` detector, startup quality matters.

Current startup behavior:

1. CSI capture starts with AGC active
2. the runtime builds a quiet anchor
3. if a clean `quiet -> motion -> quiet` pattern appears, startup may finish early
4. otherwise the detector falls back internally to the quiet-only path

With the default `segmentation_window_size_ms: 1000` and `csi_target_pps: 100`, the startup budget is ten seconds of clean, ready slot coverage after temporal warmup. Missing or burst-concentrated slots extend wall-clock calibration instead of counting as evidence; ten seconds is required valid coverage, not a fixed wait.

Practical rule:

- stay quiet immediately after boot
- after the first quiet phase, one short motion can help `lightweight` converge faster, but it is optional
- repeated movement during startup still hurts calibration quality

`high_accuracy` does not use startup threshold calibration and becomes active as soon as CSI capture is ready and its feature window has filled.

### 2. Watch The Runtime Surface

Use whatever your frontend exposes:

- logs or serial monitor
- live motion state
- movement score
- current threshold
- calibration state, when available

### 3. Test Real Movement

Walk in the monitored area and confirm:

- `MOTION` while moving
- `IDLE` while still

### 4. Tune Only One Knob At A Time

Start with threshold. If needed, then adjust window size or filters.

## Main Parameters

### Threshold

The threshold is selected automatically at startup. Lightweight adapts it from the observed quiet room, while High Accuracy uses the value validated with the exported model. Where a frontend exposes writable threshold control, both remain adjustable for the current session. Recalibration recomputes the quiet-room threshold for Lightweight; for High Accuracy, it immediately restores the trained default without collecting a quiet-room window. Matter currently exposes no writable sensing controls. Native MQTT, Micro MQTT, and ESPHome now expose the same threshold, motion-hit debounce, and recalibration control family; Micro keeps every runtime write session-only. Native BLE is setup and recovery only and does not accept those writes.

Both detectors expose a `0.0-1.0` probability threshold.

Rules of thumb:

- too many false positives: raise the threshold
- missed movement: lower the threshold

Runtime threshold changes are session-only and are recalculated at boot. ESPHome and Native can switch between `lightweight` and `high_accuracy` at runtime and persist the selection. Matter supports either detector as a build-time choice but exposes no runtime detector control; published Matter firmware uses `lightweight`. Streamer does not run a detector.

### Detection Profile

```yaml
espectre:
  detection_algorithm: lightweight  # or high_accuracy
```

| Profile | Accuracy and cost | Startup behavior |
|-----------|-------------------|------------------|
| **Lightweight Detection** (`lightweight`) | Lower detector CPU and working-memory cost, with lower accuracy and robustness than High Accuracy | About 10 seconds of clean, ready quiet-room coverage after temporal warmup; longer in wall time when occupancy is insufficient |
| **High-Accuracy Detection** (`high_accuracy`) | Higher accuracy and generalization, with additional feature state and neural-inference work | No threshold calibration; waits for CSI readiness and feature-window warmup |

Choose `lightweight` when the device must reserve resources for other firmware features or when its compute and memory budget is tight. Choose `high_accuracy` when detection quality is the priority and the additional runtime cost fits the product budget. On ESPHome and Native you can compare them at runtime; on Matter the choice is made at build time. Current measurements and known limits are documented in [ALGORITHMS.md](ALGORITHMS.md#known-limits) and the [performance report](performance/README.md).

### Window Size

```yaml
espectre:
  segmentation_window_size_ms: 1000
```

The setting is elapsed time. Together with `csi_target_pps`, it defines a fixed temporal grid: `window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)`. The runtime admits at most one packet per slot, preserves missing slots, and requires at least 70% valid occupancy before detection is ready. A burst cannot fill the window early, and arrival jitter never reconstructs the detector.

The temporal-admission contract belongs in [ALGORITHMS.md](ALGORITHMS.md#detector-timing) and its [ADR](adr/2026-08-15-use-fixed-temporal-csi-admission.md). Historical recordings without target provenance, or with sustained occupancy below 70%, cannot claim exact runtime-parity evidence and must be recollected for binding performance validation.

Rules of thumb:

- `1000 ms`: the default and the interval used by runtime, replay, validation, and training
- larger interval: steadier and slower to react
- occupancy below 70% of the configured slots: repair traffic pacing or packet supply; if the path cannot sustain the cadence, lower `csi_target_pps` explicitly and rerun performance validation rather than letting runtime jitter change it

Start with `1000 ms` unless you have a measured reason to change it.

### Traffic Rate

For frontends that expose the shared internal traffic generator:

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: internal
```

`csi_target_pps` is both the detector's temporal grid and the managed-traffic target. `csi_traffic_mode` separately chooses who supplies traffic. Internal traffic always uses a fixed send cadence at that target. Local socket send backoff still applies on `ENOMEM`; occupancy does not change the send rate. If occupancy stays below 70%, repair the traffic path or lower `csi_target_pps` explicitly and revalidate.

Runtime traffic controls follow one family across ESPHome, Native MQTT, and Micro MQTT:

- `csi_traffic_mode`: `internal`, `external`, `pacing`, or `disabled`
- `traffic_generator_mode`: `ping` or `dns`

Micro-ESPectre is the only exception: it rejects `pacing`, because it does not expose the UDP pacing listener used by the ESP-IDF runtime. Native and ESPHome persist accepted traffic-control changes; Micro keeps them session-only.

Host `espectre collect` slows only on sustained firmware TX backpressure, then recovers toward `--pps`. Occupancy remains telemetry. `--pps` stays the detector grid; `--fixed` holds a constant send rate for A/B experiments, especially on Streamer where pacing is host-owned and does not require a firmware reflash.

Rules of thumb:

- `100 pps`: default and recommended CSI target
- lower values: less overhead, less temporal detail
- higher values: more detail, more CPU and Wi-Fi cost

### Publish Interval

```yaml
espectre:
  publish_interval_ms: 1000
```

This controls the periodic status log and diagnostics sampling from the runtime's monotonic clock. Canonical MQTT telemetry and Home Assistant Movement Score follow `evaluation_interval_ms` instead. Motion state edges are handled separately, and neither heartbeat deadlines nor state-edge publication force detector evaluation.

### Evaluation Interval And Hit Filtering

```yaml
espectre:
  evaluation_interval_ms: 250
  motion_on_hits: 4
  motion_off_hits: 3
```

The detector processes every admitted CSI packet into its sliding window, but the published motion state updates only on a coarser cadence:

1. every `evaluation_interval_ms` of packet arrival time, the runtime evaluates the detector and gets a raw `IDLE` or `MOTION` reading; there is no packet-count fallback, so live input and supported replay datasets must provide advancing timestamps
2. Native MQTT, Micro MQTT, and ESPHome publish canonical telemetry and Movement Score from that evaluation once `ready_to_publish` is true
3. that raw reading must repeat for `motion_on_hits` consecutive evaluations before the published state becomes `MOTION`
4. leaving motion requires `motion_off_hits` consecutive `IDLE` evaluations

These hits are consecutive evaluation ticks, not detector windows (`segmentation_window_size_ms`). One opposing reading resets the pending count.

ESPHome, Native MQTT, and Micro MQTT all expose these runtime motion-hit settings. Native and ESPHome persist accepted updates; Micro applies them only for the current session. Native BLE does not accept motion-hit writes.

With the default `evaluation_interval_ms = 250`:

| Transition | Hits | Evaluation period | Minimum hold before publish |
|------------|------|-------------------|-----------------------------|
| `IDLE -> MOTION` | `4` | `0.25 s` | about `1.0 s` of sustained raw motion |
| `MOTION -> IDLE` | `3` | `0.25 s` | about `0.75 s` of sustained raw idle |

So a brief burst that crosses the detector threshold for one or two evaluations does not become a published motion alarm. That is the intended debounce: fewer false edges, at the cost of a short confirmation delay.

Rules of thumb:

- more hit filtering: steadier state changes, slower transitions
- expected publish latency is roughly `0.25 s * motion_on_hits`, and it no longer depends on the packet rate: a link running at `80 pps` confirms motion in the same wall-clock time as one at `100`
- increasing `evaluation_interval_ms` reduces evaluation frequency and lengthens the confirmation delay proportionally

## Filters

### Hampel Filter

Default: enabled

```yaml
espectre:
  hampel_enabled: true
  hampel_window: 7
  hampel_threshold: 5.0
```

Use it to suppress short outlier spikes. It applies to both `lightweight` and `high_accuracy`.

Disable it only if:

- you need maximum sensitivity in a clean environment, or
- you suspect it is suppressing useful low-SNR motion detail

### Low-Pass Filter

Default: disabled

```yaml
espectre:
  lowpass_enabled: true
  lowpass_cutoff: 11.0
```

Use it when the environment is noisy and false positives persist after threshold tuning.

Rules of thumb:

- lower cutoff: more smoothing, more risk of missing fast motion
- higher cutoff: less smoothing, more reactivity

## Sensor Placement

Placement still matters more than parameter tuning.

Recommended operating range:

| Distance to AP | Typical RSSI | Practical reading |
|----------------|--------------|-------------------|
| too close | above `-40 dBm` | more saturation risk |
| best range | `-40` to `-70 dBm` | good CSI headroom |
| too far | below `-80 dBm` | weaker signal, more noise |

Practical advice:

- keep the node roughly `3-8 m` from the AP when possible
- face the device toward the AP or router, not side-on to the link
- avoid putting it behind heavy obstacles if you want strong motion contrast
- if the node is too close to the AP, move it away before retuning thresholds

## Troubleshooting

### Too Many False Positives

Try in this order:

1. raise the threshold
2. enable or tune the low-pass filter
3. keep Hampel enabled
4. increase the window size slightly
5. inspect interference sources such as fans, curtains, pets, Bluetooth, or microwave activity
6. when using Lightweight, rerun calibration in a quiet room

### Missing Movements

Try in this order:

1. lower the threshold
2. reduce the window size
3. verify placement and packet flow
4. confirm the traffic source is active

### Calibration Stalls Or Startup Quality Is Poor

Usual causes:

- movement during the initial quiet phase
- sparse packet flow
- sensor too close to the AP
- chaotic RF environment at boot

Try:

1. boot again with a quieter room
2. move the sensor further from the AP
3. verify that packet flow is healthy
4. let startup complete before judging steady-state quality

### Unstable Detection Or Flickering

Try:

1. raise the threshold
2. increase the window size
3. enable the low-pass filter
4. increase hit filtering if your frontend exposes it

### No CSI Packets

Check:

1. Wi-Fi connection status
2. traffic generation path
3. CSI-enabled build/configuration
4. router compatibility and packet flow

If logs say protocol or bandwidth is `unavailable`, do not assume CSI is broken. Judge health from actual packet flow and calibration progress.

### False Positives After Wi-Fi Channel Change

If your AP changes channel often:

1. prefer a fixed router channel
2. reduce local interference
3. allow the runtime to reset and restabilize

## Recalibration

`lightweight` can recompute its threshold without changing firmware. For `high_accuracy`, the same control restores the trained default immediately and does not start a quiet-room calibration window.

Use the recalibration control when your frontend exposes one. ESPHome provides a calibration entity, Native exposes the shared control through MQTT and Home Assistant Discovery, and Micro MQTT exposes the same `recalibrate` action plus the Home Assistant switch. Matter currently exposes no writable sensing controls. Native BLE does not accept recalibration.

When recalibrating:

- expect the control surface to be briefly busy
- with Lightweight, keep the room quiet and treat it like a fresh startup calibration
- with High Accuracy, expect the trained default threshold to be restored without a collection phase

## Monitoring Checklist

Whatever frontend you use, keep an eye on:

- motion state
- movement score
- threshold
- readiness or calibration progress
- packet flow

### Firmware Performance Check

Use a `DEBUG` build when comparing firmware variants. Record the binary size and free application-partition space from the build summary, then monitor the device for several minutes after startup has settled.

For the repository hardware benchmark, connect one supported board and run `python tools/benchmark_firmware.py --chip <chip>`. It samples Native Lightweight and High Accuracy, ESPHome Lightweight, Matter's build-time default, and Streamer collection. This is representative benchmark coverage, not a capability matrix: ESPHome, Native, and Matter support both profiles, while only ESPHome and Native switch at runtime. See [tools/README.md](../tools/README.md#firmware-benchmark) for prerequisites and report behavior.

The shared ESP-IDF runtime emits a `[telemetry]` line approximately every 10 seconds at `DEBUG` level. Check that:

- packet flow stays close to the configured rate, normally around `100 pps`
- motion state remains stable when the environment is still
- `heap_free`, `heap_min`, and `heap_largest` settle instead of declining continuously
- `runtime_load`, `loop_avg_us`, and `loop_max_us` remain reasonably stable
- `detection_avg_us`, `detection_min_us`, and `detection_max_us` remain stable when comparing detector variants

For `high_accuracy`, detector timing includes feature extraction, inference, and state update. `runtime_load` measures the ESPectre runtime loop only; it is not whole-system CPU utilization. Compare results on the same target, Wi-Fi setup, traffic rate, and log level.

## Short Version

1. start with `lightweight`, `segmentation_window_size_ms: 1000`, and no low-pass filter
2. boot in a quiet room
3. tune threshold first
4. touch filters only when threshold alone is not enough
5. fix placement before chasing small parameter tweaks

## Related Docs

- [`README.md`](../README.md)
- [`SETUP.md`](SETUP.md)
- [`ALGORITHMS.md`](ALGORITHMS.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`docs/performance`](performance/README.md)
- the README of your selected frontend
