# Tuning Guide

Shared operational tuning guide for ESPectre.

This document is the main operational reference for startup behavior, thresholds, filters, placement, and troubleshooting.

Inline snippets use `ESPHome` YAML only as a concrete example.

## Quick Start

### 1. Boot In A Quiet Room

For the default `classic` detector, startup quality matters.

Current startup behavior:

1. CSI capture starts with AGC active
2. the runtime builds a quiet anchor
3. if a clean `quiet -> motion -> quiet` pattern appears, startup may finish early
4. otherwise the detector falls back internally to the quiet-only path

With the default `segmentation_window_size_ms: 1000`, the startup budget is ten seconds of clean CSI coverage. That resolves to about 800 packets at `80 pps` or 1000 packets at `100 pps`; it is a maximum, not a fixed wait.

Practical rule:

- stay quiet immediately after boot
- after the first quiet phase, one short motion can help `classic` converge faster, but it is optional
- repeated movement during startup still hurts calibration quality

`ml` does not use startup threshold calibration and becomes active as soon as CSI capture is ready.

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

The threshold is selected automatically at startup. Classic adapts its trained probability threshold from quiet session logits, while ML starts from the threshold validated with the exported model. Both remain adjustable from the frontend for the current session; recalibration restores the automatic value.

Both detectors expose a `0.0-1.0` probability threshold.

Rules of thumb:

- too many false positives: raise the threshold
- missed movement: lower the threshold

Runtime threshold changes are session-only and are recalculated at boot. ESPHome and Native persist runtime detector selections; Matter uses its fixed frontend default of `classic`, and Streamer does not run a detector.

### Detection Algorithm

```yaml
espectre:
  detection_algorithm: classic  # or ml
```

| Algorithm | Best for | Startup behavior |
|-----------|----------|------------------|
| `classic` | default adaptive non-ML path | startup threshold calibration |
| `ml` | calibration-free startup, highest recall | fixed probability threshold |

`classic` keeps false positives low on the maintained normal-link paired corpus, but its recall varies by chip and drops on weak links. Its long quiet C6 replay also exceeds the published `<5%` false-positive target; that suite is currently a diagnostic rather than a binding Classic gate. If you miss real motion and the false positives are already rare, switch to `ml` before tuning anything else: on the project corpus it recovers most of that gap without giving the false positives back. See the Known Limits section in [ALGORITHMS.md](ALGORITHMS.md) and the per-chip tables in [README.md](performance/README.md).

### Window Size

```yaml
espectre:
  segmentation_window_size_ms: 1000
```

The setting is elapsed time. The runtime measures the clean CSI cadence and rounds the sample count up so the window covers at least the requested duration. The default `1000 ms` interval resolves to `80` samples at `80 pps`, `100` at `100 pps`, and `120` at `120 pps`. The feature lags remain at the fitted `10:1` packet offsets; changing those definitions still requires a Classic refit and ML retraining. Below `80 pps`, the detector does not have enough samples for the supported feature contract and should remain on hold until the stream recovers.

The current augmented ML model was rechecked on 60-second prefixes of 22 normal-link pairs, reducing stable streams to `80 pps` by removing packets and preserving a clean `12.5 ms` cadence. This avoids the invalid shortcut of merely replaying a `100 pps` stream through a smaller window:

| replay contract | aggregate recall | aggregate false positives | worst-session recall |
| --- | ---: | ---: | ---: |
| fixed 100 samples | `99.546%` | `0.041%` | not used for the floor decision |
| temporal 1000 ms at 80 pps (80 samples) | `98.844%` | `0.019%` | `92.797%` |

The aggregate result supports `80 pps`, but the localized worst-session recall remains below the `95%` target. This is why `80 pps` is a supported floor rather than evidence that lower rates are safe. On the explicit high-rate C3 regression pair, stable decimation to `120`, `100`, and `80 pps` with matching one-second windows keeps ML at `100%` recall and `0%` false positives at every rate; Classic reaches `99.1%` recall and `0%` false positives at `80 pps`.

Correct native-cadence replay also replaces two optimistic Classic measurements that rounded `~92-93 pps` captures back to a synthetic `100 pps`. The affected normal-link C3 training replay moves from `93.64%` to `91.64%` recall, and the affected weak-link S3 replay moves from `85.06%` to `83.62%`; both remain at `0-4.2%` false positives, and the normal-link chip aggregates remain above their production targets. These are localized Classic limitations, not evidence for restoring packet-count windows.

Training follows the same rule. Packet-rate augmentation creates a lower-rate stable stream with advancing timestamps and contiguous sequence numbers, then feature extraction resolves the one-second window from that augmented cadence. Packet loss and burst loss remain separate contamination augmentations and do not masquerade as a stable rate change.

Rules of thumb:

- `1000 ms`: the default and the interval used by runtime, replay, validation, and training
- larger interval: steadier and slower to react
- fewer than `80` clean samples per second: repair the CSI supply instead of shortening the window

Start with `1000 ms` unless you have a measured reason to change it.

### Traffic Rate

For frontends that expose the shared internal traffic generator:

```yaml
espectre:
  traffic_generator_rate: 100
  traffic_generator_adaptive: true
```

The rate is the target for valid local CSI callbacks, not a fixed network send rate. By default, the shared C++ runtime and Micro-ESPectre use the same adaptive policy: send pacing can rise toward about `125%` of the target when CSI is short, backs off by `15%` on sustained socket send errors or sustained CSI oversupply, never drops below `70%` of the target, and waits three control windows between reductions. A severe CSI deficit below `50%` holds the current send rate rather than cutting it; on the original ESP32 the runtime now reports sustained low-supply windows as passive telemetry instead of trying to rearm CSI capture. Set `traffic_generator_adaptive: false` (or `TRAFFIC_GENERATOR_ADAPTIVE = False` in Micro-ESPectre) only when you need a fixed network send rate for an experiment.

Rules of thumb:

- `100 pps`: default and recommended CSI target
- lower values: less overhead, less temporal detail
- higher values: more detail, more CPU and Wi-Fi cost

### Publish Interval

```yaml
espectre:
  publish_interval_ms: 1000
```

This controls periodic movement-score reporting from the runtime's monotonic clock. Motion state edges are handled separately, and neither heartbeat deadlines nor state-edge publication force detector evaluation.

### Evaluation Interval And Hit Filtering

```yaml
espectre:
  evaluation_interval_ms: 250
  motion_on_hits: 4
  motion_off_hits: 3
```

The detector still processes every CSI packet into its sliding window, but the published motion state updates only on a coarser cadence:

1. every `evaluation_interval_ms` of packet arrival time, the runtime evaluates the detector and gets a raw `IDLE` or `MOTION` reading; there is no packet-count fallback, so live input and supported replay datasets must provide advancing timestamps
2. that raw reading must repeat for `motion_on_hits` consecutive evaluations before the published state becomes `MOTION`
3. leaving motion requires `motion_off_hits` consecutive `IDLE` evaluations

These hits are consecutive evaluation ticks, not detector windows (`segmentation_window_size_ms`). One opposing reading resets the pending count.

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

Use it to suppress short outlier spikes. It applies to both `classic` and `ml`.

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
| too close | above `-40 dB` | more saturation risk |
| best range | `-40` to `-70 dB` | good CSI headroom |
| too far | below `-80 dB` | weaker signal, more noise |

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
6. rerun calibration in a quiet room

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

`classic` can recompute its threshold without changing firmware.

Use the recalibration control exposed by your frontend, for example:

- ESPHome calibration entity
- Matter writable recalibration attribute
- native BLE or other frontend-specific control surface

When recalibrating:

- keep the room quiet
- expect the control surface to be briefly busy
- treat it like a fresh startup calibration

## Monitoring Checklist

Whatever frontend you use, keep an eye on:

- motion state
- movement score
- threshold
- readiness or calibration progress
- packet flow

### Firmware Performance Check

Use a `DEBUG` build when comparing firmware variants. Record the binary size and free application-partition space from the build summary, then monitor the device for several minutes after startup has settled.

For the repository hardware benchmark, connect one supported board and run `python tools/benchmark_firmware.py --chip <chip>`. It tests the ESPHome Dev and Native Debug frontends with both `classic` and `ml`, then writes the generated chip report under `docs/performance/`. The Classic build starts clean for each frontend, and the following ML build reuses its build directory. The ML build runs concurrently with Classic monitoring; firmware flashes and monitoring windows remain ordered and do not overlap.

The shared ESP-IDF runtime emits a `[telemetry]` line approximately every 10 seconds at `DEBUG` level. Check that:

- packet flow stays close to the configured rate, normally around `100 pps`
- motion state remains stable when the environment is still
- `heap_free`, `heap_min`, and `heap_largest` settle instead of declining continuously
- `runtime_load`, `loop_avg_us`, and `loop_max_us` remain reasonably stable
- `detection_avg_us`, `detection_min_us`, and `detection_max_us` remain stable when comparing detector variants

For `ml`, detector timing includes feature extraction, inference, and state update. `runtime_load` measures the ESPectre runtime loop only; it is not whole-system CPU utilization. Compare results on the same target, Wi-Fi setup, traffic rate, and log level.

## Short Version

1. start with `classic`, `segmentation_window_size_ms: 1000`, and no low-pass filter
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
