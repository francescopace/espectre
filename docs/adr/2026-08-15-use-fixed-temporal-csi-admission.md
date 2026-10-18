# ADR: use fixed temporal CSI admission

- Status: Accepted
- Date: 2026-08-15
- Updated: 2026-10-02

## Context

The detector window used to be a packet count: 100 samples, which is one second only at 100 pps. A slower generator stretched it, a faster one shrank it. Evaluation and publishing had already moved to elapsed time, so the window was the last setting that depended on packet rate.

The first fix made the window a duration (`segmentation_window_size_ms`) and sized it from the measured packet rate. Normal delivery jitter then kept crossing the resize threshold, rebuilding the detector, throwing away its history, and restarting Lightweight calibration. A burst of closely spaced packets could also fill a "one-second" window in a few milliseconds without giving a second's worth of motion evidence.

Sensing needs stable physical-time geometry, not a detector that follows network bursts. Traffic source, raw receive rate, and detector sampling are separate concerns. Raw collection must keep timestamped CSI for research, while sensing replays use the same admission as the device.

This ADR also covers the earlier millisecond-window decision, whose file was removed.

## Decision

The window is set in milliseconds, and CSI is admitted onto a fixed grid of time slots before any feature is computed.

**Grid**

- `segmentation_window_size_ms` defaults to 1000 ms; the allowed range is 1000–2000 ms.
- `csi_target_pps` sets the slot width. It does not turn traffic on or off; `traffic_generator_mode` picks the source.
- `window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)`.
- A window is valid with at least seven tenths of its slots filled, rounded up. The ratio is defined once per language.

**Admission**

- At most one packet per slot: the one nearest the slot center, kept until a packet lands in a later slot. A candidate that loses its slot may take the free neighbour slot on its side if that slot's center is within three quarters of a slot; otherwise it counts as excess.
- Two admitted candidates are at least half a slot apart, so packets on either side of a boundary cannot create an arbitrarily short interval.
- Duplicate, backward, and stale packets are rejected, timestamp wrap is handled, and a gap longer than a window resets temporal history at once.
- Missing slots stay missing. Statistics use the valid samples; lag and adjacent features only use pairs at their exact slot offset.
- Evaluation and startup calibration read the same admitted stream.
- Detector switches and calibration clear window data but keep the grid phase. Only a CSI session break starts a new epoch.
- Detector and slot capacity change only on an explicit config or profile change, never because of measured receive rate.
- Raw counters, admitted counters, drop reasons, missing slots, and occupancy are reported separately.
- Live and replay input must carry advancing timestamps. There is no packet-count fallback.
- Internal traffic runs at the configured cadence; occupancy never changes the send rate. Host `espectre collect` may slow down on sustained firmware TX backpressure and then recover toward `--pps`.

Feature offsets are fixed in slots: the L1 displacement ratio stays at 10:1 and turbulence autocorrelation at lag 1. Changing an offset means a Lightweight refit, a High Accuracy retrain, and the normal promotion gates.

Packet-rate augmentation is not packet loss. It picks packets from a source interval and rewrites sequence numbers and timestamps to a clean lower cadence, which the grid then sees as missing slots, not as a shorter window. Loss, burst loss, stutter, drift, and jitter stay separate augmentation effects.

**Implementations**

C++ has one production sampler in `src/cpp/core/temporal_csi_sampler.{h,cpp}`, reused by Micro-ESPectre, runtime, replay, benchmarks, and tests. Python has one CPython reference in `tools/lib/temporal_csi_sampler.py` for collector, replay, training, and validation, and `src/python/micro_espectre/temporal_csi_sampler.py` is a thin wrapper over the C++ sampler. The same timestamps must give the same decisions and counters in both languages. Detector unit tests may feed pre-admitted samples, but runtime-equivalence and performance gates must go through real admission.

A slot closes when the next packet's timestamp arrives, not on a wall-clock tick. The runtime keeps one payload buffer for the current candidate (MicroPython uses two preallocated arrays). Replay and shutdown may flush the last candidate explicitly.

Raw HTTP sends every provenance-classified frame before admission. The collector's `--pps` drives only its UDP generator and the target of its live detector view; HTTP does no pacing or decimation, and occupancy never resizes the collector grid.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-25 | Derive windows, lags, and schedules from measured packet rate | Kept for elapsed-time schedules, rejected for feature offsets |
| 2026-07-28 | Keep feature lags at nominal packet offsets | Kept, now as slot offsets |
| 2026-08-10 | Millisecond window, capacity from measured cadence | Millisecond window kept; resizing dropped after jitter kept rebuilding detectors |
| 2026-08-10 | 80 pps live floor by shrinking the sample count, hold below it | Replaced by the occupancy floor on a fixed grid |
| 2026-08-15 | Fixed temporal slot grid; never rebuild the detector from measured rate | Accepted |
| 2026-08-16 | Occupancy floor of four fifths | Lowered to seven tenths: idle-vs-motion AUC stays near full at 70% and collapses around 50% |
| 2026-08-16 | Adaptive traffic that maximizes admitted CSI | Rejected after C3 and ESP32 A/B: no gain over fixed cadence, and `tx_bp` never fired |
| 2026-08-23 | Re-anchor the grid after detector or calibration resets | Rejected: on C3 it changed occupancy for the same RX stream |
| 2026-10-02 | Losing candidate fills a free neighbour slot | Accepted: more occupancy, no systematic detection change |
| 2026-10-02 | Earliest free slot within three quarters of a slot | Rejected: even more occupancy, but larger timing error and lower Lightweight worst recall under jitter |

## Evidence

- **Shrinking the window for slow sources.** A 60 s sweep over 22 normal-link pairs, decimated to a real 80 pps with an 80-sample window, gave ML 98.84% recall, 0.019% FP, and 92.80% worst-session recall, against 99.55% and 0.041% for the 100-sample control. Recall loss is why slow sources now hold on occupancy instead.
- **Stable decimation.** On the high-rate C3 regression pair at 120, 100, and 80 pps with one-second windows, ML stayed at 100% recall and 0% FP; Classic reached 99.1% recall and 0% FP at 80 pps.
- **Occupancy floor.** Features hold up to 70% occupancy and collapse near 50%, where pair statistics like `turb_zcr` and `turb_autocorr` lose their adjacent samples.
- **Neighbour spill.** On 87 captures at about 99.6 pps raw, median occupancy rose from 92.3% to 93.9% and adjacent valid pairs from 83.9% to 88.1%, with p95 slot-center error 4.2 ms instead of 4.1 ms. On 14 reserved pairs, occupancy rose from 92.5% to 95.0% (about +4.5 points under 2–3 ms jitter). Median recall and FP changes were zero. High Accuracy worst-pair recall improved in five of seven conditions; Lightweight's dropped 1.7–4.3 points at 3 ms jitter while its worst-pair FP fell from 6.6% to 4.3%. Earliest-free-slot reached 95.4% occupancy but 5.4 ms p95 error, and Lightweight worst recall fell from 95.5% to 88.1% at 3 ms.

Denser occupancy can complete one four-hit Lightweight debounce burst on a short empty S3 recording. Lightweight's empty-room budget is therefore one alarm per recording; alarm budgets are owned by the [host-side validation ADR](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md).

## Alternatives considered

- **Fixed 100-sample window.** The physical interval depends on generator throughput, and device, replay, and augmentation cannot share one timing contract.
- **Rebuild the detector from measured rate.** Detector and calibration follow network jitter and keep losing valid state (the 2026-08-10 approach).
- **Shrink the window for slower sources.** Recall already dropped at 80 samples. Holding on occupancy is more honest than reporting outside the validated range.
- **Packet-count ring after dropping close packets.** Missing slots would be compacted, so fixed offsets would silently mean longer lags.
- **Interpolate or resample to 100 pps.** Resampling cannot recover data a slow source never sent, and interpolation puts synthetic CSI into training and evaluation.
- **Use admitted PPS as feedback for traffic control.** Admission caps its own output, so chasing it can drive an oversupply loop on bursty traffic, and bounded occupancy trials did not beat fixed cadence on C3 and ESP32. Device traffic stays at the configured rate.

## Consequences

Latency and physical coverage no longer depend on packet rate, the detector and its calibration survive jitter, a burst cannot fake a full window, and feature offsets keep their physical meaning. Device, collector, replay, training, and validation share one contract, rate augmentation trains clean lower cadences without calling them loss, and raw captures stay reproducible.

The costs: detectors track slot validity as well as feature values; datasets without trustworthy timestamps cannot claim exact parity; changing the target rate or window needs an explicit reset; detection sees each payload up to one slot late, with one payload buffered; low occupancy produces an explicit hold instead of a result; and C++ and MicroPython need a deterministic parity gate.

## Related

- [Algorithms](../ALGORITHMS.md)
- [`2026-07-08-define-the-lightweight-detector.md`](2026-07-08-define-the-lightweight-detector.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- Commits `b40c0807` (millisecond window), `d63e6222` (fixed grid), `aee0b969` (fixed traffic cadence), `ac7af68c` (grid phase kept across resets), `32e6da34` (neighbour spill), `dda85970` (`traffic_generator_mode`)
