# Temporal CSI sampling review

- Date: 2026-08-15
- Scope: live sensing cadence, detector window ownership, startup calibration, CSI target-rate configuration, Micro-ESPectre parity, and validation
- Status: Open

## Summary

Native logs from both detector profiles show that the live packet-rate controller repeatedly resolves a different detector sample count from short-term CSI arrival jitter. The C++ runtime then reconstructs the detector, clears its history and motion filter, and restarts the profile's startup path. High Accuracy repeatedly loses readiness, while Lightweight repeatedly restarts its approximately ten-second calibration and may never reach steady-state detection.

The accepted millisecond-window ADR correctly requires a physical-time analysis contract, but the deployed runtime implements that contract indirectly as `measured rate -> packet count -> reconstructed detector`. Its validation covers stable rate-transformed replays, not a bursty live stream whose long-term rate is stable. The review direction is to preserve the configured temporal contract without letting network delivery jitter redefine feature geometry: admit at most one representative CSI packet per configured temporal slot, retain missing slots as missing, keep one detector instance, and derive every count from `csi_target_pps`, `segmentation_window_size_ms`, and `evaluation_interval_ms`.

The proposed configuration separates the CSI cadence target from traffic-source ownership. `csi_target_pps` is always positive and defines the temporal sampling grid. `csi_traffic_mode` remains the single source of truth for whether traffic is internal, external, paced, or unmanaged; zero must no longer act as a hidden disable sentinel.

## Findings

### TS-001 — P1 — Short-term rate jitter reconstructs the live detector

The rate estimator stores 64 inter-packet deltas and refreshes its cached estimate every 16 accepted intervals in `src/cpp/core/detector_timing.h`. At approximately 100 pps this covers about 0.64 seconds and can change about every 0.16 seconds. `CsiPipeline::process_normalized_packet_()` compares the resulting sample count with the current detector window on every accepted packet and posts a resize at a difference of approximately 5%. `EspIdfRuntime::on_detector_window_changed_()` allocates a replacement detector and installs it as a cold detector.

Observed High Accuracy evidence: 14 reconstructions in 4.4 seconds, with resolved windows oscillating through `122, 105, 93, 104, 94, 105, 118, 105, 120, 104, 109, 119, 107, 120` while one-second status remained near `103-107 pps`.

Precise locations:

- `src/cpp/core/detector_timing.h:23-32`
- `src/cpp/core/detector_timing.h:100-227`
- `src/cpp/runtime/esp_idf/csi_pipeline.cpp:228-238`
- `src/cpp/runtime/esp_idf/esp_idf_runtime.cpp:359-380`

Required resolution: remove measured-rate-driven detector reconstruction from the steady-state packet path. A short-term rate estimate may remain diagnostic, but it must not control detector ownership or feature-window capacity.

### TS-002 — P1 — Detector replacement destroys readiness and motion state

Installing a replacement calls `CsiPipeline::set_detector()`, which cold-clears the detector, resets accumulated cadence coverage, resets the effective motion filter to idle, and may emit an idle transition. High Accuracy requires a full feature window before inference and therefore loses useful detection coverage after every replacement.

Precise locations:

- `src/cpp/runtime/esp_idf/csi_pipeline.cpp:61-82`
- `src/cpp/core/base_detector.cpp:207-220`
- `src/cpp/core/base_detector.h:106-110`
- `src/cpp/core/high_accuracy_detector.cpp:235-246`

Required resolution: keep the detector instance stable across ordinary CSI timing variation. Reset it only for an explicit lifecycle boundary or contamination condition whose contract requires a cold restart.

### TS-003 — P1 — Lightweight calibration can be starved indefinitely

Every C++ window replacement cancels the active startup state and calls `start_calibration_()`. Lightweight calibration targets ten configured window durations. The attached Lightweight log shows 14 reconstruction and calibration starts in about 6.9 seconds; no attempt can finish its approximately ten-second clean-coverage budget. Micro-ESPectre performs the same replacement and synchronous recalibration after a rate-derived timing update.

Precise locations:

- `src/cpp/core/detector_limits.h:34-36`
- `src/cpp/runtime/esp_idf/esp_idf_runtime.cpp:371-380`
- `src/cpp/runtime/esp_idf/esp_idf_runtime.cpp:494-552`
- `src/python/micro_espectre/main.py:925-954`

Required resolution: natural slot occupancy changes must not restart calibration. Lightweight calibration must advance on clean admitted temporal coverage and restart only for a documented contamination or lifecycle event.

### TS-004 — P1 — Packet-count admission lets bursts masquerade as temporal evidence

The detector currently processes every accepted CSI callback. A network burst can therefore fill a nominal one-second sample window in a few milliseconds even though those closely spaced frames carry little independent motion evidence. Fixed packet-offset features also lose their intended temporal interpretation when adjacent accepted packets are delivered much closer together than the target cadence.

Precise locations:

- `src/cpp/runtime/evaluation_cadence.h:43-70`
- `src/cpp/runtime/esp_idf/csi_pipeline.cpp:241-267`
- `src/python/micro_espectre/runtime_policy.py:449-508`
- `src/python/micro_espectre/main.py:900-961`

Required resolution: introduce a temporal CSI admission layer before detector feature processing. Derive slot identity without cumulative rounding drift, for example from a wrap-safe unwrapped timestamp and `slot_index = timestamp_us * csi_target_pps / 1_000_000`. Admit at most one representative packet per slot, never interpolate a missing slot, and do not let a burst backfill elapsed coverage.

### TS-005 — P1 — Missing-slot semantics must preserve feature timing

Simply discarding close packets and then keeping the last N accepted packets is insufficient: at 80% occupancy, N accepted packets span longer than the configured window, and packet-relative lags cease to represent the configured slot offsets. The temporal sampler and detector storage must therefore retain slot identity or timestamps through feature extraction.

Precise locations:

- `src/cpp/core/base_detector.cpp:36-68`
- `src/cpp/core/high_accuracy_detector.cpp:228-246`
- `src/cpp/core/l1_delta_tracker.h:112-120`
- `src/cpp/core/lightweight_detector.cpp:84-114`
- `src/cpp/core/detector_limits.h:38-46`

Required resolution: define a fixed-duration slot window in which missing slots remain absent. Window statistics operate only on valid values in the configured interval. Lagged features consume only valid slot pairs at their configured offset, rather than treating the next accepted packet as temporally adjacent. Readiness is based on clean elapsed coverage and a configured occupancy ratio.

### TS-006 — P2 — The configuration conflates CSI target cadence with traffic enablement

`RuntimeConfig::traffic_generator_rate` currently means both the requested rate and, through zero, that internal traffic generation is disabled. The detector also falls back to a hidden default when the rate is zero. The existing `CsiTrafficMode` already owns whether traffic is internal, external, paced, or unmanaged, so the zero sentinel duplicates lifecycle state and leaves an external detector without an explicit sampling target.

Precise locations:

- `src/cpp/runtime/runtime_interface.h:82-109`
- `src/cpp/runtime/runtime_sensing_schema.h:80-97`
- `src/cpp/runtime/csi_traffic_types.h:17-41`
- `src/cpp/runtime/esp_idf/csi_traffic_service.cpp:26-39`
- `src/cpp/runtime/esp_idf/esp_idf_runtime.cpp:322-332`

Required resolution: replace the public target field with positive `csi_target_pps`. Use it for detector admission, managed traffic generation, external pacing, and stream target cadence. Keep `csi_traffic_mode` as the only enable/source selector, remove the zero fallback, and distinguish configured target, raw observed CSI rate, and detector-admitted CSI rate in diagnostics.

### TS-007 — P2 — Stale, duplicate, and out-of-order packets are not rejected consistently

Cadence accounting treats a missing or non-advancing timestamp as zero elapsed coverage, but the packet can still continue into detector processing. A time-aware admission contract must make packet acceptance explicit, including 32-bit timestamp wrap, duplicates, backward timestamps, packets older than the active window, and processing-backlog staleness when a compatible wall clock is available.

Precise locations:

- `src/cpp/core/detector_timing.h:48-59`
- `src/cpp/runtime/evaluation_cadence.h:53-69`
- `src/cpp/runtime/esp_idf/csi_pipeline.cpp:241-250`
- `src/python/micro_espectre/runtime_policy.py:449-472`

Required resolution: reject duplicate and out-of-order packets before feature processing, expire stored samples by slot or timestamp, clear temporal history after a contaminating gap, and keep wall-clock staleness separate from arrival-time progression.

### TS-008 — P1 — Existing validation does not exercise live burst stability

The millisecond-window ADR validates detector quality at stable transformed rates, and existing unit tests cover duration-to-count conversion and isolated resize thresholds. They do not feed a target-rate stream with realistic batching and jitter and assert bounded admissions, preserved calibration, stable detector ownership, correct slot occupancy, and unchanged motion quality.

Precise locations:

- `docs/adr/2026-08-10-configure-detector-windows-in-milliseconds.md:39-59`
- `test/cpp/suites/integration/test_packet_rate_adaptation.cpp:267-333`
- `test/cpp/suites/runtime/test_csi_pipeline.cpp`
- `test/python/test_runtime_policy.py`
- `test/python/test_packet_rate_adaptation_regression.py`

Required resolution: extend the owning runtime-policy, CSI-pipeline, calibration, and packet-rate integration suites with deterministic uniform, jittered, bursty, sparse, duplicate, stale, out-of-order, wraparound, and sustained off-target cases. Add an invariant that ordinary timing variation never reconstructs the detector or restarts calibration.

### TS-009 — P2 — The accepted ADR records reconstruction but does not evaluate temporal admission

The accepted ADR correctly selects a millisecond public window and acknowledges reconstruction and calibration restart as trade-offs. Its alternatives do not compare detector reconstruction with a fixed temporal slot queue, and the earlier dead-band evidence for bursty streams is no longer represented in the active decision.

Precise locations:

- `docs/adr/2026-08-10-configure-detector-windows-in-milliseconds.md:9-37`
- `docs/adr/2026-08-10-configure-detector-windows-in-milliseconds.md:61-90`
- commit `8c2ac9b8d0f249bf1830d8b6693ca7f377d09363`
- commit `b40c0807cb32bb9dcb49093ef4de34475515d8ac`

Required resolution: supersede or amend the timing ADR with the temporal admission and fixed detector-lifecycle decision, its missing-slot semantics, memory bounds, overflow behavior, and measured validation.

### TS-010 — P1 — Traffic control feedback and Streamer pacing must remain outside detector admission

The sensing runtime currently feeds capture-accepted CSI totals into the internal adaptive traffic generator, while Streamer uses host UDP pacing both to elicit fresh CSI and to grant uplink streaming opportunities. If the adaptive controller is changed to observe detector-admitted samples, the new sampler caps its own feedback at `csi_target_pps` and may respond to burst concentration by increasing traffic, creating a self-reinforcing oversupply loop. Applying detector admission to Streamer would also discard raw research data and break the collector pacing-credit contract.

Precise locations:

- `src/cpp/runtime/esp_idf/esp_idf_runtime.cpp:132-170`
- `src/cpp/runtime/esp_idf/csi_traffic_service.cpp:59-121`
- `src/cpp/runtime/traffic_rate_controller.cpp:29-111`
- `src/cpp/runtime/esp_idf/stream_esp_idf_runtime.cpp:85-170`
- `src/cpp/runtime/esp_idf/csi_stream_transport.cpp:508-590`
- `src/cpp/frontend/streamer/espectre/streamer_frontend.cpp:29-41`

Required resolution: preserve separate raw capture, temporal admission, and stream delivery counters. Internal adaptive generation observes pre-admission capture supply and socket backpressure, while occupancy remains detector health telemetry rather than a direct pacing actuator. Sensing `PACING` may use an external source at `csi_target_pps`, but the temporal admission filter still protects feature geometry. Streamer firmware remains collector-paced and preserves raw timestamped CSI transport. The collector, replay, training, and validation paths reuse the single production Python temporal-admission implementation with the collector's requested `--pps` target, so sensing-aligned data is uniform without discarding the raw source or changing pacing-credit semantics.

## Target contract

- `csi_target_pps` is a positive configured target and the single cadence source for detector slots and managed traffic.
- `csi_traffic_mode` alone selects internal, external, pacing, or unmanaged traffic; no rate value disables traffic.
- `segmentation_window_size_ms` remains the configured physical analysis duration.
- `evaluation_interval_ms` remains the elapsed-time detector evaluation cadence and is never replaced by a hardcoded packet count.
- `window_slots = ceil(csi_target_pps * segmentation_window_size_ms / 1000)` with overflow-safe integer arithmetic.
- `minimum_valid_slots = ceil(window_slots * minimum_coverage_ratio)`, where the initial policy is `0.8` and the ratio has one named, shared source of truth rather than an absolute packet floor.
- The admission grid accepts at most one representative CSI packet per slot. Excess packets are counted and discarded before feature processing.
- Missing slots remain missing. They are not interpolated, compacted away, or backfilled from a burst.
- The detector is ready only after the configured temporal coverage exists and valid occupancy meets the derived minimum.
- Evaluation, calibration, and steady-state detection consume the same admitted slot stream.
- The single Python implementation is a small production `TemporalCsiSampler` component in `src/python/micro_espectre/temporal_csi_sampler.py`. It remains MicroPython-compatible and is imported directly by the Micro-ESPectre runtime, collector, replay, training, integration tests, and host validation rather than copied into a host-only reference. The corresponding single C++ production implementation lives in `src/cpp/core/temporal_csi_sampler.h` and `.cpp` and is reused directly by the runtime pipeline, C++ replay support, benchmarks, utilities, and integration tests rather than mirrored under `test/cpp/`. Detector-only unit tests in either language may continue to call `detector.process_packet()` with an already-admitted synthetic stream; every runtime-equivalence and performance gate must pass timestamped packets through the production sampler. Production C++ must match the shared Python implementation's slot admission, timestamp handling, missing-slot feature semantics, readiness, calibration, and state results on identical inputs.
- Internal traffic generation uses the configured fixed send cadence by default. Adaptive generation is opt-in and, when enabled, observes raw identity-accepted CSI before detector admission; detector occupancy and same-slot drops are diagnostics and must not form a positive feedback loop that drives additional oversupply.
- Sensing frontends default to `INTERNAL` for autonomous operation. `PACING` is optional and requires a live external pacing source; it does not replace temporal admission because Wi-Fi delivery can still be bursty.
- Streamer firmware continues to use collector-owned `PACING`, with host `--pps` as the session target, and bypasses on-device detector admission so raw CSI transport and pacing credits retain their current semantics.
- The collector already imports the production detector and `RuntimeMotionPolicy` from `src/python/micro_espectre/`; it must also import the production `TemporalCsiSampler`, not reimplement admission in `espectre_cli`. The collector needs only call-site plumbing to ask the sampler whether a packet is admitted before passing it to `detector.process_packet()`, using collector `--pps` in place of device `csi_target_pps`. Raw capture remains the reproducible source, and every Streamer replay, training, or validation consumer derives the sensing view through the same production API.
- Diagnostics distinguish configured target PPS, raw observed PPS, admitted detector PPS, missing slots, excess same-slot drops, stale drops, out-of-order drops, and temporal-window occupancy.

## Implementation validation evidence

- Native, Matter, and Streamer ESP32-C3 builds completed successfully. The final Native image was flashed without erasing the `nvs` partition at `0x9000-0xdfff`; stored Wi-Fi and MQTT configuration remained usable.
- On the attached C3 and local AP, internal ping at a configured 100 pps produced about 100-106 raw accepted CSI pps but only 33-41 detector-admitted pps, 34-43% occupancy, about 55-69 missing slots/s, about 58-72 same-slot excess drops/s, and no stale or out-of-order drops. Internal DNS produced the same pattern.
- Reducing the internal target to 30 pps produced about 30-34 raw pps but only 10-20 admitted pps and 37-67% occupancy. External fixed pacing at 100 pps produced about 98-107 raw pps, 43-47 admitted pps, and 45-48% occupancy. These measurements show AP delivery aggregation rather than sampler staleness or timestamp-clock mismatch.
- A final Native run with internal adaptive traffic at 100 pps observed about 165-203 raw CSI callbacks/s, admitted 41-50 samples/s, reported 41-51% occupancy, rejected 120-157 same-slot packets/s, and reported no stale or out-of-order drops. The raw-supply controller reduced the generated send rate independently, while Lightweight remained in calibration instead of completing from clustered packets. No detector reconstruction or rate-driven calibration restart appeared in the log.
- The attached C3 was also flashed with the ESPHome development profile, whose generated `sdkconfig` does not enable Bluetooth. With the previous ESPHome Wi-Fi buffer profile, internal ping at a 100 pps target reported 82.8 traffic pps, 97.6 raw accepted CSI pps, 73.0 admitted pps, and 72% occupancy. Matching the Streamer high-rate Wi-Fi and lwIP profile raised the observed internal-ping result to 90.0 traffic pps, 101.9 raw pps, 76.2 admitted pps, and 76% occupancy. This confirms that BLE coexistence was a major Native-specific loss source, while buffer pressure was only a smaller part of the remaining temporal loss.
- On the optimized ESPHome image, internal DNS was worse than ping in the sampled interval: 90.8 traffic pps, 96.8 raw pps, 66.9 admitted pps, and 67% occupancy. Fixed host UDP pacing at 100 pps was the best measured source, with 100.3 pacing pps, 86.4 admitted pps, and 87% occupancy. The on-demand ESPHome API diagnostic read added application traffic, so its 126.1 raw pps is not a pure pacing count; the separately reported pacing and temporal-admission counters remain the relevant comparison.
- A bounded 120 pps external pacing experiment did not recover missing target slots: it produced 143 raw pps including application traffic, only 78 admitted pps, 78% occupancy, and 65 same-slot excess drops/s. Oversupplying the radio therefore amplified batching instead of improving temporal evidence and is rejected as an automatic compensation strategy.
- Disabling runtime debug telemetry and reducing the ESPHome logger from `DEBUG` to `INFO` left internal-ping occupancy unchanged at 76% (90.5 traffic pps, 102.5 raw pps, and 76.6 admitted pps). Debug instrumentation is therefore not the remaining bottleneck, and the development profile was restored after the bounded experiment.
- Fixing internal ping at the configured 100 pps target improved five consecutive ESPHome samples to 76-82% occupancy, averaging about 79%, with 98.6-99.9 generated pps and 76.7-81.7 admitted pps. The adaptive run had fallen to about 73-76% because Home Assistant and other application frames increased raw CSI while failing to replace uniformly scheduled sensing frames. Raw CSI is therefore not a sufficiently isolated feedback signal for adaptive pacing on an integrated frontend.
- External ICMP from the same Mac used for UDP pacing isolated host ownership from packet type. A 30-second, 100 pps run delivered all 3,000 echo requests and replies with 0% loss and 5.51 ms average RTT, while five diagnostic samples reported 75-82 admitted CSI pps, about 80 pps on average, 75-82% occupancy, and 11-30 same-slot excess packets/s. This matches fixed internal ping rather than the 86 admitted pps external-UDP result, so an external host alone does not explain the improvement; packet shape, QoS, or AP treatment of ICMP versus the pacing datagram remains material.
- The runtime therefore defaults internal traffic to fixed cadence, keeps adaptive pacing as an explicit opt-in, refuses to treat clustered raw frames as temporal evidence, keeps Lightweight calibration active while the window is invalid, and exposes the reason through diagnostics. Selecting a lower target is an explicit deployment and validation choice, never an automatic detector resize.

## Validation disposition

- The C++ configure, build, and required `test_motion_detection` gate passed. The full C++ suite passed 28 of 29 tests after rerunning its two UDP socket tests outside the sandbox; `test_empty_rooms` remains incompatible with several temporally bursty legacy captures and reports the same calibration and single-alarm cases as Python.
- The required Python `TestPerformanceMetrics` gate passed 71 tests and failed 11. The full Python suite passed 1000 tests, skipped 2, and reported 21 failures in the sandbox; both UDP failures passed outside the sandbox, leaving 19 corpus-related failures. These cover legacy captures that no longer provide enough independent temporal slots, two weak-link recall regressions after burst removal, and one S3 empty-room alarm.
- The SDK surface invariant suite passed all 32 tests, Doxygen completed without warnings, and the direct C++/CPython identical-trace admission parity test passed.
- The production MicroPython module is exercised directly under CPython, but device-runtime validation remains open because the attached C3 currently runs the ESPHome development firmware used for the no-Bluetooth experiments. Reflashing a separate CSI-enabled MicroPython image, collecting a time-uniform corpus, retraining, and accepting revised performance evidence are deliberately deferred; the validation thresholds and sampler contract were not weakened to make legacy captures pass.

## Progress checklist

- [x] REV-A01 Reproduce and quantify High Accuracy detector reconstruction from the supplied Native log.
- [x] REV-A02 Reproduce and quantify Lightweight reconstruction and repeated startup calibration from the supplied Native log.
- [x] REV-A03 Trace the C++ rate estimator, resize event, detector replacement, cold clear, motion reset, and calibration lifecycle.
- [x] REV-A04 Trace the equivalent Micro-ESPectre timing update, detector replacement, and recalibration path.
- [x] REV-A05 Identify the introducing implementation commit and review the superseded and active detector-timing ADR decisions.
- [x] REV-A06 Record the agreed target direction: configured temporal slots, one representative packet per slot, missing-slot preservation, fixed detector ownership, derived occupancy, and `csi_target_pps` configuration semantics.
- [x] REV-A07 Update or supersede the detector-window ADR with temporal admission, missing-slot feature semantics, lifecycle rules, memory bounds, overflow handling, and accepted alternatives.
- [x] REV-A08 Define `csi_target_pps` and the shared minimum-coverage ratio in the canonical C++ and Micro-ESPectre configuration sources; remove zero-as-disabled semantics and provide an intentional compatibility or migration path for `traffic_generator_rate`.
- [x] REV-A09 Update the public SDK configuration, validation helpers, Kconfig bindings, diagnostics, Native/Matter/Streamer setup, ESPHome schema and examples, host tooling, and owning documentation for `csi_target_pps`.
- [x] REV-A10 Prototype and implement the canonical temporal slot admission as the MicroPython-compatible `TemporalCsiSampler` in `src/python/micro_espectre/temporal_csi_sampler.py`, using configured PPS, window duration, and evaluation interval; cover timestamp unwrap, duplicate rejection, out-of-order rejection, stale expiry, gap contamination, slot selection, occupancy, counters, turbulence statistics, aggregated turbulence, Lightweight autocorrelation, High Accuracy L1 displacement, readiness, and calibration.
- [x] REV-A11 Integrate that same production Python implementation into the Micro-ESPectre device runtime and import it directly from CPython tests and host validation; do not create a second host-only implementation, interpolate missing slots, or compact accepted packets across missing time.
- [x] REV-A12 Remove Micro-ESPectre measured-rate detector reconstruction and recalibration from the steady-state loop.
- [x] REV-A13 Extend `test/python/test_runtime_policy.py` and the existing Micro-ESPectre owning suites with uniform, jittered, bursty, sparse, duplicate, stale, out-of-order, wraparound, and sustained off-target streams.
- [ ] REV-A14 Validate the production `src/python/micro_espectre/` implementation directly under CPython and on the MicroPython device path using the maintained packet-rate, motion, weak-link, empty-room, and long-recording gates; record memory, admission, readiness, recall, false-positive, and alarm effects before the C++ port.
- [x] REV-A15 Port the validated temporal admission to one frontend-agnostic production component in `src/cpp/core/temporal_csi_sampler.h` and `.cpp`, integrate its time-bounded feature-window behavior through the shared runtime layer, and reuse that component directly from all C++ consumers without adding blocking or allocation work to CSI callbacks.
- [x] REV-A16 Remove the C++ detector-window resize event, replacement callback, and rate-jitter calibration restart path after the temporal admission path owns the contract.
- [x] REV-A17 Add raw/admitted/missing/excess/stale/out-of-order/occupancy diagnostics and ensure the adaptive traffic controller observes the metric appropriate to its target without feeding detector admission back into a runaway pacing loop.
- [x] REV-A18 Extend `test/cpp/suites/runtime/test_csi_pipeline.cpp`, the owning detector suites, calibration tests, traffic-service/controller tests, and packet-rate integration suite with temporal admission and stable-lifecycle invariants. C++ integration, replay, benchmark, and utility code must include and exercise the production core sampler or runtime pipeline directly; do not create a test-only admission implementation or duplicate its constants and formulas.
- [ ] REV-A19 Run parity validation between the single `src/python/micro_espectre/` implementation, exercised under both CPython and MicroPython, and the single `src/cpp/core/` implementation, exercised directly by C++ tests and replay tools, for temporal admission, timestamp rejection, missing-slot and slot-relative features, readiness, calibration, state changes, and diagnostics on identical timestamped sequences.
- [x] REV-A20 Run the required detector validation gates: `cmake -S test/cpp -B test/cpp/build`, `cmake --build test/cpp/build`, `ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure`, and `.venv/bin/pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v`. The C++ gate passed; the Python result and legacy-corpus failures are recorded above.
- [x] REV-A21 Run the full maintained C++ and Python suites, the SDK surface gate, affected firmware builds, and documentation checks; report any unavailable command with its exact blocker. Native, Matter, and Streamer C3 builds, SDK invariants, Doxygen, and all non-corpus tests passed; the remaining corpus failures are recorded above.
- [x] REV-A22 Update `docs/ALGORITHMS.md`, `docs/SETUP.md`, `docs/TUNING.md`, `docs/EMBEDDING.md`, frontend README files, examples, and the active changelog with the final validated behavior and migration guidance.
- [x] REV-A24 Define and document the mode/profile matrix for Native, Matter, ESPHome, Micro-ESPectre, and Streamer, including which component owns `csi_target_pps`, which source supplies traffic, whether detector admission applies, and which counter drives adaptive control.
- [ ] REV-A25 Verify `INTERNAL`, `EXTERNAL`, `PACING`, and `DISABLED` lifecycle and recovery tests across the affected sensing frontends; separately verify that Streamer collector pacing, fresh-sample FIFO behavior, batching, backpressure adaptation, and raw dataset delivery remain unchanged.
- [x] REV-A26 Import the production `TemporalCsiSampler` directly in the collector and route its existing Micro-ESPectre detector call through it before `detector.process_packet()` using collector `--pps`; do not add a collector-side sampler. Preserve raw timestamped capture, reuse the same component in replay, training, integration tests, and validation, record the target and admission counters in dataset provenance, and prove that the derived sensing view matches device admission on identical inputs.
- [x] REV-A27 Characterize the attached C3 without Bluetooth through the ESPHome frontend, compare the previous and Streamer-aligned buffer profiles, and measure internal ping, internal DNS, external fixed pacing at the configured target, and a bounded 20% oversupply experiment.
- [x] REV-A28 Make fixed internal send cadence the shared C++ and MicroPython default, retain adaptive device pacing as an explicit opt-in, update owning documentation and configuration examples, rebuild ESPHome, Native, Matter, and Streamer for ESP32-C3, and verify the ESPHome development image on hardware with `adaptive=off`.
- [x] REV-A29 Compare externally paced ICMP with externally paced UDP from the same host at 100 pps, record delivery and temporal-admission results, and determine whether host ownership alone explains the external-UDP improvement.
- [x] REV-A23 Inspect the final diff for unrelated changes, verify generated artifacts through their owning workflows, and close this review only after every accepted activity is complete or explicitly deferred with rationale. The diff is scoped to temporal admission, configuration, affected consumers, tests, and owning documentation; `git diff --check`, SDK invariants, and Doxygen are clean. Device MicroPython parity, time-uniform corpus replacement, performance reacceptance, and Streamer end-to-end hardware validation remain open as recorded above.
