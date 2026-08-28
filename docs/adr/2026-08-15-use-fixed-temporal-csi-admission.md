# ADR: use fixed temporal CSI admission

- Status: Accepted
- Date: 2026-08-15
- Updated: 2026-08-25

## Context

The deployed detector window was first a fixed packet count. Its default of 100 samples represented one second only at the nominal `100 pps`; a slower generator widened the physical interval, while a faster generator shortened it. Runtime evaluation and periodic publishing had already moved to elapsed-time schedules, leaving the analysis window as the remaining public packet-rate-dependent setting.

The first temporal contract replaced that packet count with `segmentation_window_size_ms` and resolved sample capacity from a short-term measured packet rate. That restored a physical-time public setting, but ordinary delivery jitter could cross the resize dead band, reconstruct the detector, discard its history, and restart Lightweight startup calibration. It also allowed a burst of closely spaced packets to fill a nominal one-second window in milliseconds even though those packets did not provide one second of independent motion evidence.

Motion sensing needs stable physical-time geometry rather than a detector that follows network delivery bursts. The traffic source, the raw receive rate, and the detector sampling cadence are separate concerns. Raw collection must retain timestamped CSI for reproducible research while downstream sensing replays use the same admission behavior as deployed detectors.

This record incorporates the millisecond-window lineage. That file is removed rather than marked superseded because this ADR is the cumulative current detector-timing decision.

## Decision

Keep the public analysis window in milliseconds and admit CSI onto a fixed temporal slot grid before feature processing:

- the public setting is `segmentation_window_size_ms`; the default is `1000 ms`, and the supported configuration range is `1000-2000 ms`;
- `csi_target_pps` is a positive sensing target and defines slot width; it does not enable or disable traffic;
- `csi_traffic_mode` selects either internal or external traffic ownership;
- `window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)`;
- the minimum valid occupancy is seven tenths of `window_slots`, rounded up, with the ratio defined once in each production language;
- at most one packet is admitted per slot; the sampler retains the candidate nearest the slot center until a packet reaches a later slot, and counts every other same-slot candidate as excess;
- the minimum distance between consecutive admitted candidates is half a target slot, derived from `csi_target_pps`, so candidates on opposite sides of a boundary cannot create an arbitrarily short detector interval;
- duplicate, backward, and stale packets are rejected, timestamp wrap is handled explicitly, and a gap spanning a detector window invalidates temporal history immediately, even while the first post-gap candidate remains pending until a later slot or an explicit flush;
- missing slots remain missing: statistics use valid samples, while lagged and adjacent features use only pairs at their exact slot offsets;
- evaluation and startup calibration consume the same admitted stream;
- detector changes and calibration boundaries clear pending and admitted detector-window data without changing the timestamp-grid phase; only a CSI session discontinuity starts a new temporal epoch;
- detector instances and slot capacity remain fixed until an explicit configuration or detector-profile change; measured receive rate never reconstructs a detector;
- raw receive counters, detector-admitted counters, drop reasons, missing slots, and occupancy remain distinct diagnostics;
- live and replay inputs require advancing packet timestamps; there is no packet-count timing fallback;
- internal traffic generation uses the configured fixed send cadence; occupancy never changes the send rate; and
- host `espectre collect` may slow on sustained firmware TX backpressure and recover toward `--pps`, without occupancy trials.

Production feature offsets remain fitted slot geometry: the L1 displacement ratio stays `10:1`, and turbulence autocorrelation stays at lag `1`. Physical-time windows and fixed slot-offset features are deliberate parts of the same timing contract; changing an offset requires a Lightweight refit, a High Accuracy retrain, and the normal promotion gates.

Stable packet-rate augmentation is distinct from loss. It selects packets across a source interval and rewrites sequence counters and device/Wi-Fi timestamps to a clean lower cadence. Under the fixed grid that lower cadence appears as missing slots, not as a smaller window. Packet loss, burst loss, stutter, drift, and feature jitter remain separate augmentation effects.

Python has one CPython reference implementation in `tools/lib/temporal_csi_sampler.py`, while `src/python/micro_espectre/temporal_csi_sampler.py` is a thin MicroPython facade over the C++ core sampler. Collector, replay, training, validation, and integration tests use the CPython reference directly. C++ has one frontend-agnostic production implementation in `src/cpp/core/temporal_csi_sampler.h` and `.cpp`; Micro-ESPectre, runtime, replay support, benchmarks, utilities, and integration tests reuse it directly or validate against it. Detector-only unit tests may supply already-admitted samples, but runtime-equivalence and performance gates must use timestamped production admission. Identical timestamp sequences must produce identical Python and C++ decisions and counters.

Closing a slot is driven by the next packet timestamp, not by the processing-loop wall clock. The live runtime therefore keeps one fixed payload buffer for the current candidate; when a later-slot packet arrives, the previous payload is consumed before the current packet can replace the buffer. MicroPython uses two preallocated payload arrays for the same transition. Finite replay and controlled shutdown may explicitly flush the last buffered candidate.

Raw HTTP v2 transports every provenance-classified CSI frame before temporal admission. The collector's `--pps` value controls only its external UDP traffic generator and supplies the target for its live detector and derived sensing view. HTTP applies no pacing or temporal decimation; occupancy remains telemetry and never resizes the collector grid.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-25 | Derive windows, feature lags, and schedules from measured packet rate | Retained for elapsed-time schedules but rejected for fitted feature offsets |
| 2026-07-28 | Keep production feature lags at nominal packet offsets | Retained as slot offsets under the later admission grid |
| 2026-08-10 | Configure the analysis window in milliseconds and resolve sample capacity from measured cadence | Kept the public millisecond window; rejected measured-rate resize after live jitter reconstructed detectors |
| 2026-08-10 | Treat `80 pps` as the live floor by shrinking the resolved sample count, and hold below it | Replaced by the occupancy floor on a fixed `csi_target_pps` grid |
| 2026-08-15 | Admit CSI onto a fixed temporal slot grid and never reconstruct the detector from measured receive rate | Accepted |
| 2026-08-16 | Occupancy floor of four fifths | Moved to seven tenths after reserved idle-versus-motion AUC stayed close to full occupancy at 70% and collapsed around 50% |
| 2026-08-16 | Opt-in adaptive traffic that maximizes temporally admitted CSI rather than raw accepted callbacks | Rejected after C3 and classic ESP32 A/B; occupancy-adaptive send-rate trials did not beat fixed cadence, and `tx_bp` did not fire on the benches |
| 2026-08-23 | Re-anchor the temporal grid after detector or calibration resets | Rejected after C3 runtime switches changed occupancy for an unchanged RX stream; software-only resets now preserve the active grid phase |

A 60-second sweep over 22 normal-link pairs, under the earlier measured-rate window, removed packets from stable streams to create a genuine `80 pps` cadence. With the default temporal window resolved to 80 samples, aggregate ML recall was `98.844%`, aggregate false positives were `0.019%`, and worst-session recall was `92.797%`. The fixed 100-sample control produced `99.546%` recall and `0.041%` false positives. That evidence still argues against supporting arbitrarily slow sources by shrinking the configured duration; current readiness uses occupancy on the target grid instead of a second rate-derived window size.

On the explicit high-rate C3 regression pair, stable decimation to `120`, `100`, and `80 pps` with matching one-second windows kept ML at `100%` recall and `0%` false positives at every rate. Classic reached `99.1%` recall and `0%` false positives at `80 pps`.

The occupancy floor moved from four fifths to seven tenths after a reserved idle-versus-motion AUC sweep showed that the production features remain close to full occupancy at 70% and collapse only around 50%, where consecutive-pair statistics such as `turb_zcr` and `turb_autocorr` lose adjacent samples. The admission grid, slot selection, gap reset, and missing-slot contract are unchanged.

That denser occupancy can complete one four-hit Lightweight debounce burst on a short empty S3 recording. High Accuracy still requires zero empty-room alarms. Lightweight sequential empty tests now allow at most one effective alarm per recording rather than forcing a shared occupancy increase or a more expensive Lightweight feature set.

## Alternatives Considered

### Keep a fixed 100-sample window

Rejected. It makes the physical analysis interval depend on generator throughput and prevents deployment, replay, and packet-rate augmentation from sharing one temporal contract.

### Reconstruct the detector from measured packet rate

Rejected. It makes detector ownership and calibration depend on short-term network jitter and repeatedly discards valid state. This was the 2026-08-10 duration-to-sample conversion.

### Support slower sources by shrinking the configured window

Rejected. Localized recall already dropped at an 80-sample measured-rate window. Holding on occupancy is more explicit than emitting results outside the validated envelope or silently violating the configured duration.

### Keep a packet-count ring after dropping closely spaced packets

Rejected. Missing slots would be compacted, so fixed packet offsets would silently represent longer physical lags and the configured window could span too much time.

### Interpolate missing CSI slots or resample every stream to 100 pps

Rejected. Decimation can remove excess samples but cannot create missing information on a slower source, and interpolation would introduce synthetic CSI values into both training and evaluation.

### Feed admitted PPS into adaptive traffic control as supply feedback

Rejected. Admission caps its own output at the target, so chasing admitted PPS as if it were missing raw supply can drive a positive oversupply loop when traffic is bursty. Bounded occupancy trials that keep a send-rate step only when temporally admitted CSI improves were also rejected after ESP32-C3 and classic ESP32 A/B: they did not beat a fixed `csi_target_pps` cadence. Device traffic therefore stays at the configured rate. Host collect may still slow on sustained firmware TX backpressure. The configured `csi_target_pps` grid stays fixed.

## Consequences

Benefits:

- detector latency and physical coverage no longer depend on generator packet rate;
- detector lifetime and calibration are stable under delivery jitter;
- a network burst cannot manufacture a full sensing window;
- feature offsets keep their physical-time interpretation;
- device, collector, replay, training, and validation share one contract;
- packet-rate augmentation can train clean lower-cadence intervals without treating them as loss; and
- raw research captures remain reproducible.

Trade-offs:

- detectors must track missing-slot validity in addition to feature values;
- datasets without trustworthy timestamps cannot claim exact runtime parity;
- target-rate or window-duration changes require an explicit sampler and detector reset;
- detection receives the selected payload with a delay of at most one active slot, while one fixed CSI payload remains buffered;
- unsupported occupancy produces an explicit hold state instead of an unvalidated detector result; and
- the C++ and MicroPython implementations require a deterministic parity gate.

## Related

- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`2026-07-26-recover-the-startup-threshold-once-a-session-settles.md`](2026-07-26-recover-the-startup-threshold-once-a-session-settles.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
