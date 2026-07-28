# ADR: derive detector timing from the measured packet rate

- Status: Accepted
- Date: 2026-07-25

## Current Implementation Status

The decision remains the target contract, but deployment is partial as of
2026-07-28. Host replay and validation paths use the full derived timing.
Deployed C++ and MicroPython runtimes use arrival timing for evaluation cadence,
gap detection, and state reset, while detector windows and feature lags remain
the packet-count values selected at construction time. The remaining C++ wiring
is tracked as `B-3` in the
[C++ review](../review/cpp-review-2026-07-28.md).

## Context

Detector timing was expressed in packets everywhere: windows, both feature lags,
the firmware evaluation cadence, and the host replay that validates them. Packet
counts only mean what they are supposed to mean at exactly `100 pps`.

Real streams do not run at exactly `100 pps`. The recorded corpus spans `90` to
`120`, with one deliberate `1000 pps` diagnostic pair, and deployments drift the
same way: ESP32 tops out around `70-80 pps` and delivers it in bursts with
holes. On such a stream a `100`-packet window no longer spans a second and a
`10`-packet lag no longer spans `100 ms`, so the features are computed over
different physical intervals than the ones their coefficients were fitted over.

Two classes of real data broke on this:

1. **Degraded streams with long holes or burst loss.** A capture can arrive with
   valid-looking CSI payloads but long pauses between packets. In packet-count
   space those pauses are invisible, so a detector evaluates a temporally stale
   window as if it were continuous.
2. **High-rate captures.** At `~1000 pps`, evaluating every `25` packets means
   evaluating every `25 ms` instead of `~250 ms`, and a `100`-packet window
   spans `~100 ms` instead of `~1 s`. That changes the detector's semantics
   rather than giving the same semantics more samples.

The damage was measured rather than assumed. Decimating a healthy `100 pps`
capture to simulate slower links moved Classic false positives from `0.0%` to
`10.1%` on the same recording, with recall swinging down to `83.2%` on another,
and the movement was not monotone, so no single correction factor could undo it.

## Decision

Express the detector's contract in microseconds and resolve it into packet
counts from the measured cadence. `derive_detector_timing()` is the single
definition, ported once per language:
[runtime_policy.py](../../src/python/micro_espectre/runtime_policy.py) and
[detector_timing.h](../../src/cpp/core/detector_timing.h). The host replay
imports the Python one rather than holding a copy.

**1. The lags track physical time; the window tracks sample count.**

The lags describe how far the channel has moved over an interval, so they have
to follow that interval. The window is different: its features are estimator
averages, and what matters is how many samples they average, not the time those
samples span.

This asymmetry is what measurement supports, and it is the opposite of what
symmetry suggests. Holding a one-second span at low rates leaves too few
samples, the estimates get noisier, startup calibration answers the wider quiet
distribution by lifting the threshold, and the entire cost lands on recall while
false positives stay low.

**2. Rate estimation needs two statistics, not one.**

The median inter-packet interval answers "is this gap a hole" and drives
contamination. The mean of non-contaminated intervals answers "how many packets
per second" and sizes the window. Real captures are bursty rather than evenly
paced, so the two differ materially and each is wrong for the other's job.

**3. Loss is measured against the stream's own sequence step.**

A stream that natively runs slower advances its packet counter by more than one
per delivered packet. Judging that against a hardcoded step of one reads the
whole stream as loss.

**4. A `+/-25%` dead band around the nominal cadence.**

Inside `80-133 pps` the timing snaps to nominal and nothing adapts.

**5. Cadence comes from the packet arrival timestamp, never from the loop
clock.**

Both runtimes read the Wi-Fi RX timestamp: `rx_ctrl.timestamp` in C++, `frame[4]`
in MicroPython, the same field the streamer already records as `wifi_rx_ts_us`.
The loop clock measures how fast packets are *processed*, which matches arrival
on hardware but not on replay, and it would make the cadence depend on host
scheduling. Wall-clock time is reserved for staleness watchdog duty, which
arrival time cannot do because a dead stream delivers no timestamps.

Sources that report no arrival timestamp, and the estimator's own warmup, fall
back to the packet counter.

**6. Bounds have stated reasons.**

`DETECTOR_MIN_WINDOW_SIZE` is `100` because that is where the measurement puts
the floor. `DETECTOR_MAX_WINDOW_SIZE` is `200` because a window that spans
several seconds smears short movements into the background and slows the
response, not because of memory: the detector working buffers were moved off
the CSI callback stack onto the heap, sized to the real window, so the original
stack rationale no longer applies. `L1_DELTA_LAG_MAX` is `32` because the
firmware profile ring is statically sized; it covers the `100 ms` contract to
about `320 pps`, past what any supported chip sustains.

**7. A gap contaminates the window, and contamination resets state.**

Packet timing is read in priority order: `device_ticks_us`, then
`wifi_rx_ts_us`, then the cadence estimate. Sequence gaps remain a
contamination signal, and so do large elapsed-time holes even when packet order
is monotonic. On contamination the detector state, the cadence, and the warm-up
accounting all reset, so a stale temporal window cannot leak into the metrics.

**8. Startup calibration counts only clean coverage.**

A contaminated prefix restarts the startup session rather than contributing a
toxic baseline to the threshold.

**9. Both detectors and every replay path share the contract.**

ML is not exempt. Paired replays, empty-room replays, and long-recording
replays all advance on the same elapsed-time cadence and take the same
contamination resets as the Classic path, in both languages.

**10. The `1000 pps` pair stays diagnostic-only.**

Non-`100 pps` captures are valuable for validation and debugging, but the first
`1000 pps` pair remains `dataset_role: exclude` until there is a deliberate
promotion policy for high-rate data and enough breadth to judge it fairly. It
is useful because it is hard, not because it is a passing baseline waiting to
be re-labeled.

## Validation

**The lags are decisive at high rate.** On the `1000 pps` diagnostic capture,
with the window held at `1000` packets:

| autocorrelation lag | recall | false positives | effective alarms |
| --- | --- | --- | --- |
| 1 packet (776 us) | 72.5% | 32.7% | 28 |
| 13 packets (10 ms) | 71.3% | 0.0% | 0 |

`calc_autocorrelation` defaults to lag `1`, so production autocorrelation spans
a single packet: `10 ms` at `100 pps` but under a millisecond at `1000 pps`,
where consecutive packets are almost perfectly correlated and the feature leaves
the range its coefficients were fitted over. Under the full contract the capture
scores `75.1%` recall at `0.3%` false positives. On packet-count cadence the
same pair scored `50.6%` recall at `8.35%` false positives for Classic and
`71.8% / 18.0%` for ML, against `90.8% / 0.58%` and `95.3% / 0.54%` on a nearby
`100 pps` C3 reference pair. Most of what looked pathological about the capture
was the timing, not the capture.

**The window is the opposite.** Ablation at fixed cadence on one C5 pair:

| cadence | window | lag | recall |
| --- | --- | --- | --- |
| 50 pps | 50 | 5 | 87.3% |
| 50 pps | 100 | 5 | 99.1% |
| 50 pps | 50 | 10 | 86.7% |
| 50 pps | 100 | 10 | 99.4% |
| 25 pps | 25 | 2 | 60.2% |
| 25 pps | 100 | 2 | 98.7% |

The window explains the whole effect and the lag explains none of it at these
rates. Recall collapses while false positives stay low, which is the signature
of startup calibration absorbing the extra variance by raising the threshold.

**Throughput, not spacing, sizes a window.** One C6 capture delivers a quarter
of its packets about `71 us` apart with `65-70 ms` pauses between bursts. Its
median interval claims `215 pps`; its throughput is the declared `97.9`. Using
the mean of non-contaminated intervals, the estimate is within `2.6 pps` of the
declared rate on every capture in the corpus.

**Cadence-relative loss detection unblocks slow streams.** A stream decimated
four to one was previously `100%` contaminated, which made startup calibration
impossible: it never completed and the replay returned nothing. It is now `0%`,
and gapped and contiguous sequence numbering produce identical results at every
rate.

**The dead band pays for itself.** Adapting inside `+/-25%` of nominal flips
packet counts between neighbouring values across streams that all run at
essentially the nominal rate, so one coefficient set has to cover slightly
different feature definitions. Measured on the training corpus, adapting there
costs `3.8` points of fitted recall at the same false-positive ceiling.

**The cadence is reproducible.** `test_csi_pipeline_evaluates_on_elapsed_packet_time`
replays `30 s` of packets and counts evaluation ticks: `120` at `100 pps` and
`120-150` at `500 pps`, where a packet-count cadence would give `600`. Because
arrival time is an input, the test is deterministic.

**Nothing moves at the nominal rate.** At `100 pps` the contract resolves to
window `100`, lag `10`, autocorrelation lag `1`, evaluation every `25` packets,
which is exactly the previous behaviour.

## Alternatives Considered

### Express the window in time as well

Rejected on measurement. It is the symmetric choice and it is worse: see the
ablation above, where a time-held window at `25 pps` scores `60.2%` recall
against `98.7%` for a sample-held one.

### Resample every stream to a canonical 100 pps

Rejected. Decimation preserves the statistics the coefficients were fitted on,
so it was the safe option, but it discards data and cannot help below the
canonical rate, where no resampling can invent packets. Making the features
rate-invariant subsumes it.

### Average packets in blocks to recompact fast streams for ML

Rejected for now, and worth revisiting with measurement rather than argument.
Averaging is a low-pass filter that lowers `l1_delta` and raises
`turb_autocorr` by an amount that depends on the block size, so it reintroduces
the rate dependence it was meant to remove, and it needs a non-integer factor
for rates like `120 pps`. Its one real advantage is noise reduction that scales
as the square root of the block, which is directly relevant to the weak-link
false alarms, so it deserves a head-to-head against decimation on the
`1000 pps` pair. Promoting it would require retraining ML and refitting Classic
on averaged data, because the averaging becomes part of the feature definition.

### Drive the cadence from the loop clock

Rejected. It measures processing speed rather than arrival, makes replay
non-reproducible, and lets host scheduling leak into a detector decision.

### Keep packet-count cadence and widen the window on high-rate captures

Rejected. It still ignores real-time holes in degraded streams, and it needs
per-dataset tuning instead of one deploy-time semantic.

### Normalize only the reporting, not the evaluation cadence

Rejected. Reporting on elapsed time while evaluating on packet count leaves the
detector itself mis-timed on both degraded and high-rate captures.

### Promote the `1000 pps` pair into the normal validation corpus

Rejected. It is too singular and too pathological to promote today, even after
the contract made it usable.

## Consequences

Benefits:

- a window spans the same physical time and holds a comparable number of
  samples across the cadences the hardware actually produces
- the `1000 pps` diagnostic capture is usable rather than pathological, and most
  of what looked pathological about it was ours
- streams with steady partial loss can calibrate at all
- startup calibration no longer locks onto a contaminated prefix
- every bound now records why it exists, so the next person can tell a memory
  constraint from a product decision

Trade-offs and open work:

- **The Classic coefficients are unverified off-nominal.** They remain valid at
  the nominal cadence, where the contract resolves to exactly the window and
  lags they were fitted under, but no fit has measured them elsewhere. A refit
  was attempted and rejected: with the fit tool corrected, the resulting
  coefficients lost to the incumbent at matched false positives on three chips
  and tied on a fourth. See `tools/fit_classic_detector.py` and task notes.
- Rates below about `40 pps` stay weak, with one S3 pair at `73-77%` recall.
  Accepted, because supported hardware does not go there.
- The firmware pays a static `PacketRateEstimator` per pipeline and a slightly
  larger L1 profile ring.
- ML replay and long-recording paths now measure their own cadence, so their
  reported numbers are not directly comparable to reports generated before this
  change.
- Validation and parity tooling depend on the saved packet timing metadata when
  it is present, which makes them more complex than a packet counter.

## Related

- [2026-07-25-select-the-classic-band-from-channel-coherence.md](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md](2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md)
