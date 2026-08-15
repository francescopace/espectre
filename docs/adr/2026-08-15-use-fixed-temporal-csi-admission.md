# ADR: use fixed temporal CSI admission

- Status: Accepted
- Date: 2026-08-15
- Supersedes: `2026-08-10-configure-detector-windows-in-milliseconds.md`

## Context

The millisecond detector-window decision derived detector storage from a short-term measured packet rate. Ordinary delivery jitter could therefore cross the resize dead band, reconstruct the detector, discard its history, and restart Lightweight startup calibration. It also allowed a burst of closely spaced packets to fill a nominal one-second window in milliseconds even though those packets did not provide one second of independent motion evidence.

Motion sensing needs stable physical-time geometry rather than a detector that follows network delivery bursts. The traffic source, the raw receive rate, and the detector sampling cadence are separate concerns. Streamer must also retain raw timestamped CSI for reproducible research while its downstream sensing replays use the same admission behavior as deployed detectors.

## Decision

Use a fixed temporal slot grid before feature processing:

- `csi_target_pps` is a positive sensing target and defines slot width; it does not enable or disable traffic;
- `csi_traffic_mode` alone selects internal, external, paced, or disabled traffic ownership;
- `window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)`;
- the minimum valid occupancy is four fifths of `window_slots`, rounded up, with the ratio defined once in each production language;
- at most one packet is admitted per slot, and excess packets in the same slot are discarded before feature processing;
- duplicate, backward, and stale packets are rejected, timestamp wrap is handled explicitly, and a gap spanning a detector window invalidates temporal history;
- missing slots remain missing: statistics use valid samples, while lagged and adjacent features use only pairs at their exact slot offsets;
- evaluation and startup calibration consume the same admitted stream;
- detector instances and slot capacity remain fixed until an explicit configuration or detector-profile change; measured receive rate never reconstructs a detector;
- raw receive counters, detector-admitted counters, drop reasons, missing slots, and occupancy remain distinct diagnostics; and
- internal traffic generation uses the configured fixed send cadence by default because unrelated application traffic also produces raw CSI;
- adaptive traffic control is opt-in and observes pre-admission capture supply, never the sampler output.

Python has one MicroPython-compatible production implementation in `src/python/micro_espectre/temporal_csi_sampler.py`. Micro-ESPectre, collector, replay, training, validation, and integration tests import it directly. C++ has one frontend-agnostic production implementation in `src/cpp/core/temporal_csi_sampler.h` and `.cpp`; runtime, replay support, benchmarks, utilities, and integration tests reuse it directly. Detector-only unit tests may supply already-admitted samples, but runtime-equivalence and performance gates must use timestamped production admission. Identical timestamp sequences must produce identical Python and C++ decisions and counters.

Streamer firmware continues to transport raw timestamped CSI under collector-owned pacing. The collector's `--pps` value supplies the target for its live detector and derived sensing view without changing raw capture or pacing-credit semantics.

## Alternatives Considered

### Reconstruct the detector from measured packet rate

Rejected. It makes detector ownership and calibration depend on short-term network jitter and repeatedly discards valid state.

### Keep a packet-count ring after dropping closely spaced packets

Rejected. Missing slots would be compacted, so fixed packet offsets would silently represent longer physical lags and the configured window could span too much time.

### Interpolate missing CSI slots

Rejected. Synthesized RF observations would enter training and inference and could create motion evidence that was never measured.

### Feed admitted PPS into adaptive traffic control

Rejected. Admission caps its own output at the target, so using it as supply feedback can drive a positive oversupply loop when raw traffic is bursty.

## Consequences

Benefits:

- detector lifetime and calibration are stable under delivery jitter;
- a network burst cannot manufacture a full sensing window;
- feature offsets keep their physical-time interpretation;
- device, collector, replay, training, and validation share one contract; and
- raw research captures remain reproducible.

Trade-offs:

- detectors must track missing-slot validity in addition to feature values;
- datasets without trustworthy timestamps cannot claim exact runtime parity;
- target-rate changes require an explicit sampler and detector reset; and
- the C++ and MicroPython implementations require a deterministic parity gate.

## Related

- [`2026-08-10-configure-detector-windows-in-milliseconds.md`](2026-08-10-configure-detector-windows-in-milliseconds.md)
- [`../review/2026-08-15-temporal-csi-sampling-review.md`](../review/2026-08-15-temporal-csi-sampling-review.md)
