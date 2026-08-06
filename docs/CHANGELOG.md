# Changelog

All notable changes to this project will be documented in this file.

---

## [3.0.0] - Unreleased - Modular multi-frontend platform

### Highlights

- **Firmware architecture is now modular**: the codebase is split into `core`, `runtime`, and `frontend` layers so ESPHome, native, Matter, and streamer can share the same sensing foundations.
- **Native firmware is now a first-class frontend**: the standalone BLE/MQTT path is no longer embedded in ESPHome and now ships as its own firmware target.
- **Matter is now part of the published firmware surface**: releases, snapshots, CI, and the web flasher now include Matter artifacts for the supported targets.
- **Streamer workflows were promoted and cleaned up**: the C++ streamer path is now the main live-streaming implementation, with collector-driven discovery and broader multi-chip CLI support.
- **ESPectre Protocol is now a shared platform service**: BLE, MQTT, provisioning, telemetry, and command handling now form a reusable baseline across ESP-IDF frontends.
- **Classic now uses a vote-free weighted fusion of L1-delta and turbulence autocorrelation**, with double Hampel filtering and session-adapted probability thresholds.
- **Threshold modes were removed**: Classic now adapts automatically at startup, ML uses its trained threshold, and both accept session-only runtime adjustments.
- **Production ML now ships the promoted compact phaseless model**: the exported feature set is now ten scale-invariant signals, combining the five turbulence/L1 invariants with five coherence and channel-shape dynamics that survived the current-catalog refresh. The runtime removes the whole phase tracker, exports the `10 -> 24 -> 12 -> 1` MLP, and keeps the paired and quiet gates at `14/14` and pass on the refreshed corpus while cutting the compact no-phase layout from 897 to 577 MLP parameters. See [FEATURES.md](FEATURES.md).
- **The roadmap now frames `v3` as the modular multi-frontend platform phase**.

### Added

- **Home Assistant MQTT Discovery adapters for Native and Micro-ESPectre**:
  both frontends publish retained discovery payloads, republish them after the
  Home Assistant birth message, and keep the canonical ESPectre MQTT topics
  unchanged. Discovery is enabled by default in published Native firmware and
  remains opt-in for Micro-ESPectre.
- **ESPHome Intensity sensor**: the frontend now publishes a derived `intensity_sensor` (`min(200, movement / threshold × 100)`) so Home Assistant gauges stay meaningful when Classic adapts the threshold at boot, without a template helper.
- **Host-side ML candidate framework and a `--features` trainer flag**: production and host-only feature combinations can be evaluated without changing runtime extractors. Candidate-only runs require `--no-export`; export remains restricted to the current shared production feature set. The feature inventory and accumulated evidence live in [FEATURES.md](FEATURES.md).
- **Dual licensing**: ESPectre is now offered under GPLv3 or a separate commercial license for proprietary integrations (`LICENSING.md`), and the CLA check returns alongside the DCO check so contributions can be distributed under both tracks.
- **Embedding guide** (`docs/EMBEDDING.md`): how to integrate the shared `core` and `runtime` layers into third-party ESP32 firmware, with the published frontends as reference integrations.
- **Embeddable SDK surface aligned with standard `C++` SDK conventions**: a single `espectre_sdk.h` facade now reaches the whole supported surface and documents the integration model, threading contract, and stability tiers; `espectre_sdk_version.h` adds compile-time `ESPECTRE_SDK_VERSION_*` macros and the `ESPECTRE_SDK_VERSION_AT_LEAST()` guard, kept in step with `library.json` and `idf_component.yml` by the SDK packaging build. The public headers carry Doxygen documentation, `docs/Doxyfile` generates a browsable reference for the supported surface, and the SDK root is on the include path so layer-prefixed includes such as `runtime/runtime_interface.h` avoid collisions with generic basenames.
- **Automated gates for the published SDK surface**: `test/python/test_sdk_surface_invariants.py` fails when a type reachable from `espectre_sdk.h` is left as an unresolved forward declaration, or when a facade header is missing from the API reference or the `docs/EMBEDDING.md` header map, and CI now generates the Doxygen reference with warnings as errors so a half-documented member cannot ship. `AGENTS.md` carries the matching rules for the judgment calls no check can see, including surface breaks on the runtime interfaces and accessors that degrade silently under a build-time option.
- **Optional capability groups for the vendored ESP-IDF component**: MQTT, BLE, provisioning, OTA, frontend support, and the stream runtime are now selectable under the "ESPectre SDK" menuconfig menu, so a component-style install reaches the same sources as the `espectre_sources.cmake` path instead of being limited to the sensing runtime.

- **Four task-oriented website guides** now cover browser flashing and Wi-Fi provisioning, detection fundamentals, custom firmware and embedded sensing integration, and product use cases.
- **Per-device Matter onboarding data** generated with the device RNG and persisted in a dedicated factory partition, with the same QR available from the web flasher, serial logs, and `./espectre matter qr`.
- **Unified browser tools under `espectre.dev`** for firmware flashing, BLE configuration, MQTT monitoring, the motion-controlled game, and the Wi-Fi Theremin, with the same MQTT, BLE, and Theremin pages available locally through `./espectre ui`.
- **Targeted firmware benchmark runs** can select one frontend and one detector, reducing iteration time while debugging a specific firmware path.
- **Persisted runtime detector selection** for ESPHome and Native, including Home Assistant, BLE, and MQTT controls, automatic Classic calibration, and a fixed Classic Matter default without a writable Matter surface.
- **New source layout under `src/cpp/`** with shared `core`, `runtime`, and `frontend` layers.
- **Shared runtime/frontend infrastructure** with explicit runtime contracts, common frontend orchestration, and reusable ESP-IDF protocol services.
- **Shared ESP-IDF runtime debug telemetry** for periodic heap, configured CPU frequency, runtime-loop load, loop timing, and detector evaluation timing across frontends.
- **On-demand ESPHome CSI diagnostics**: production firmware always exposes a `Refresh Diagnostics` button and diagnostic entities for actual traffic TX, raw CSI callback, accepted CSI, and filtered CSI rates, plus the current Wi-Fi channel and RSSI. The cache reuses the existing sensing-update window and Home Assistant states change only on request, so diagnostics add neither a timer nor periodic publication.
- **On-demand Native MQTT CSI diagnostics**: Native shares the sensing-update rate calculation used by ESPHome and returns its cached traffic, CSI callback, accepted, filtered, Wi-Fi channel, and RSSI values only when a client requests `stats`. The diagnostics remain available independently of runtime debug logging.
- **Pipelined firmware hardware benchmark** for ESPHome and Native Classic/ML variants, with ML builds overlapping Classic monitoring and generated per-chip performance reports.
- **Matter frontend and release surface**, including published artifacts for releases, snapshots, and the web flasher.
- **Shared HTTPS OTA service for ESP-IDF frontends** under `runtime/esp_idf`.
- **Weighted two-feature Classic runtime** aligned across Python and C++, using gain-invariant L1-delta and turbulence autocorrelation.
- **Cross-language report parity gating** for `tools/generate_performance_report.py`: the published performance report now rebuilds and runs the host-side C++ integration suites, exports structured paired and long-recording metrics, and aborts if the Python and C++ aggregates drift.
- **Parallel multi-detector live collect** through `./espectre collect --detector classic,ml`.
- **Timed dataset collection now uses the selected production Classic or ML detector for its ready gate**, replacing the retired host-only moving-variance adapter.
- **BLE-assisted Wi-Fi provisioning for the streamer firmware** via the unified Configure page
- **Uplink CSI record batching in the streamer transport**: up to 7 records per UDP datagram via `ESPECTRE_STREAM_TX_BATCH_RECORDS` (default 4), cutting uplink packet rate and airtime pressure.
- **Per-record streamer PHY metadata**: stream V7 identifies the received PHY mode, exported LTF type, and normalized channel width; collected `.npz` files preserve these fields, and historical ML datasets are explicitly treated as HT20.
- **ESP32-specific streamer `sdkconfig` profile** with shallower Wi-Fi TX/RX buffer caps and lwIP IPv6 disabled to fit the original ESP32 resource budget.
- **Updated architecture documentation** in `docs/ARCHITECTURE.md`
- **CSI amplitude heatmap plotter** (`tools/plot_heatmap.py`) for paper-style time × subcarrier views of dataset samples, with a publication export used in the detection guide and algorithm docs

Historical decision context for the Classic and ML promotions now lives in:

- [`docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md) (active Classic decision, including weighted fusion and threshold-mode removal)
- [`docs/adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`](adr/2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md) (superseded L1 gated-calibration record)
- [`docs/adr/2026-07-28-drop-the-absolute-l1-features.md`](adr/2026-07-28-drop-the-absolute-l1-features.md) (five-feature ML transition record in the supersession chain)
- [FEATURES.md](FEATURES.md) (current, historical, rejected, and research feature inventory)

### Fixed

- **Wi-Fi channel changes no longer leave sensing or streaming sessions in an unstable CSI state**: the shared ESP-IDF capture service detects and rejects the first accepted packet on a new channel, defers recovery out of the Wi-Fi callback, and invalidates the session independently of the selected detector. Sensing frontends cold-reset Classic or ML, rearm CSI capture and traffic generation, and apply the detector's startup flow, while Streamer resets its transport session and rearms capture without rebooting the device.
- **Classic's first feature was a level statistic, so it inherited the very noise floor it was meant to see through**: the L1 profile is normalized by its own per-packet mean, which removes gain but not the floor, so the mean displacement rose whenever the link weakened whether or not anything moved. Across the corpus it separated motion at `1.0000` AUC on normal links but `0.8705` on weak ones, inverting sign on two of eight, while every shape statistic held. Two captures calibrated their threshold to `0.969` and `0.986` and Classic reached `69%` recall on them where ML reached `100%`. The feature is now `mean(displacement at lag) / mean(displacement at lag 1)`: noise saturates the displacement immediately so its ratio sits near `1.0`, real channel evolution keeps growing with the lag, and the shared unit divides the floor out. The worst pair in the corpus goes from `69.21%` to `90.83%` recall with mean false positives down from `2.12%` to `1.23%` and the empty-room gate still silent, and the fit's worst-session recall goes from `55.1%` to `82.0%`. Startup calibration is still needed, only easier: the idle level now varies `1.82x` across quiet captures instead of `14.29x`. See [2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md](adr/2026-07-26-replace-the-classic-l1-mean-with-a-lag-ratio.md).
- **The Classic fit reported only pooled rates, and a pooled rate hides the one recording that fails**: `choose_base_threshold` now also reports `worst session recall`, warns when it trails the pooled figure by more than 10 points, and accepts `--min-session-recall` to make it binding. The shipped fit shows `97.1%` pooled against `55.1%` worst session, and a single-feature experiment that scored `98.0%` pooled turned out to leave one capture at `62%`.
- **The dataset validator still anchored corpus review on one detector family, so detector limits could read as data-quality defects**: pair and idle review now run directly on the shared scale-invariant feature set instead of on `ClassicDetector` thresholds, probabilities, or self-calibrated logits. The generated dataset-quality report keeps integrity, continuity, overlap, and ML-readiness admission checks, but its score tables are now detector-agnostic: pairs are judged by feature-space `Cover`, `Sep`, `Tail`, and `Score`, while idle captures are judged by `Exc`, `Burst`, `Tail`, `Drift`, and `Score` on their own feature-evidence baseline. This keeps low-RSSI and gain-shifted captures comparable, removes threshold-relative bias from the report entirely, and leaves detector-specific promotion evidence in `docs/performance/README.md`.
- **ML decided differently in C++ than in Python because the compiler was free to fuse its multiply-adds**: `MLDetector::predict` accumulates in `float`, and contracting each `val += input * weight` into an FMA skips the intermediate rounding. That is invisible until the output meets a threshold, where it flipped whole evaluations on recordings whose probabilities sit near `0.5`, moving ten of the twenty-eight paired replays and the worst by `3.2` points of recall. It also made the result depend on the compiler and on surrounding code rather than on the model, so the firmware could have decided differently from every number the project had measured. Contraction is now disabled at the source, so ESP-IDF, PlatformIO, and the host tests share one arithmetic; the report parity gate passes on ML, and the published ML figures come down slightly because contraction had been inflating them. See [2026-07-26-run-ml-inference-without-float-contraction.md](adr/2026-07-26-run-ml-inference-without-float-contraction.md).
- **Classic recovers a startup threshold set from an unrepresentative prefix**: calibration reads the opening of a session and never revisited it, so a noisy opening left the threshold wrong for the whole run. On one ESP32 capture the prefix is `4.14x` noisier than the rest, leaving the threshold at `3.82x` the highest level the session ever reaches and `4.7` points of recall unreachable, while the features themselves separate at `0.9999` AUC. The detector now tracks the median of per-block metric maxima over a `60 s` dwell and lowers the threshold to it, one-sided and held up by any real motion in the window. Worst per-chip recall goes from `94.2%` to `97.7%` with per-chip false positives, the weak-link slice, and the empty-room gate all unchanged. See [2026-07-26-recover-the-startup-threshold-once-a-session-settles.md](adr/2026-07-26-recover-the-startup-threshold-once-a-session-settles.md).
- **Python/C++ performance-report parity**: four divergences between the two runtimes are gone. After a timing reset the C++ tracker inferred the missing interval from the mean where Python used the median, which shifted every later evaluation boundary by about a millisecond per contamination; `typical_interval_us()` guarded on the wrong cache and never refreshed, freezing the hole-detection median; the replay cadence lacked the packet-count fallback that Python and the production pipeline both keep; and the startup calibrator replayed weighted samples one packet at a time instead of folding them in bulk, splitting chunk boundaries differently. The C++ replay now also seeds its timing helpers from the measured interval, snapped for calibration and raw for replay, the way the Python paths do.
- **Classic no longer carries the obsolete low-RSSI L1 noise-blend safeguard**: after the lag-ratio feature replaced the plain mean, the startup L1 floor never again changed a detector decision on the current corpus, so the extra excursion branch and state were removed. The fitter's default false-positive ceiling also moves to `--fp-target 3.0`, matching the operating point accepted by the empty-room gate.
- **Original ESP32 streamer stability under sustained collection**: the streamer now stays on the shared HT20 sensing contract, while the pacing-health path reports sustained callback deficits as telemetry instead of trying to recover them by cycling CSI capture.
- **Embedded traffic-generator pacing no longer over-cuts under CSI surplus**: shared C++ and Micro-ESPectre adaptive pacing now react to local socket backpressure, keep a `70%` floor of the CSI target, settle between reductions, and avoid the previous proportional slash that could drop send rate from `100` toward below `50` pps.
- **Original ESP32 CSI stall handling is now passive telemetry across sensing runtimes**: streamer and Micro-ESPectre log sustained callback deficits for diagnostics, instead of trying to recover them by rearming CSI capture.
- **Shared CSI Wi-Fi protocol policy**: all published ESP-IDF targets now configure the supported 802.11n protocol ceiling directly instead of first attempting the unsupported `WIFI_PROTOCOL_11N`-only combination.
- **Native Wi-Fi association after CSI STA_START policy**: standalone station connect now applies the CSI radio policy before `esp_wifi_connect()` when it does not own the lifecycle handlers, and clears the connect latch on `WIFI_EVENT_STA_STOP` so BLE coexistence or protocol renegotiation can reassociate instead of leaving the radio idle.

### Changed

- **Sensing keeps HT20 while making the Wi-Fi band explicit**: 2.4 GHz remains
  the validated default on every target, while an ESP32-C5 integrator can choose
  5 GHz or automatic dual-band selection through the shared Kconfig,
  `RuntimeConfig`, or ESPHome's native `wifi.band_mode` setting. ESPHome C5
  configurations follow ESPHome's `AUTO` default when the setting is omitted,
  and the ESPectre component mirrors the effective choice into the shared
  runtime instead of defining a second Wi-Fi option.
  The lifecycle pins `11b/g/n` on
  2.4 GHz and `11a/n` on 5 GHz, with HT20 on every selected band; unsupported
  choices fail instead of falling back silently. Optional channel hints must
  match the chosen band. Detection quality on 5 GHz is not yet characterized:
  every ESPectre dataset was collected on 2.4 GHz. See
  [`2026-08-05-pin-ht20-on-every-band-instead-of-forcing-2-4-ghz.md`](adr/2026-08-05-pin-ht20-on-every-band-instead-of-forcing-2-4-ghz.md).
- **Delay-compensated coherence now evaluates its derotation with Horner
  instead of one sine and cosine pair per bin**: the aligned sum is
  `sum_k cross[k] * w**bin[k]` for `w = exp(-i * ramp)`, and the live bins of a
  contiguous band are consecutive, so the sum is a polynomial in `w`. Each band
  now costs one sine and cosine pair instead of one per bin, and because only
  the magnitude is used, the shared `w**start_bin` factor cancels and a band's
  absolute position drops out of the computation entirely. The MicroPython
  coherence tracker falls from 223.54 to 139.98 ms over 3000 packets. The
  roadmap had proposed an incremental power recurrence for this; that form was
  measured and rejected, since it drifts further than Horner without being
  faster. On the host `C++` build Horner is not faster than the trigonometric
  form it replaces, and its device-side cost was not measured. Accuracy against
  a double-precision reference moves from 3.6e-7 to 1.1e-6 on coherent
  profiles, which is roughly four thousand times finer than the `int8` CSI
  payload itself resolves; the detector performance and parity validations are
  unchanged, chip for chip.

- **The runtime policy checks a packet's shape once per packet rather than once
  per field**: `Mapping.__instancecheck__` walks the ABC registry while the
  lookup that follows is a plain dict hit, and every packet reads three timing
  fields. The normalized amplitude profile also accepts a caller buffer, as its
  `C++` counterpart already did, which removes two list allocations per packet
  from the MicroPython path. Abstract base class checks over a paired replay
  drop from 242k to 127k and profiled `observe_packet` from 0.814 to 0.701
  seconds, below wall-clock noise on the replay itself.

- **The coherence trackers now size their lag rings to the configured lag and
  share one cross-product array per reference**: the profile rings were static
  arrays dimensioned to `L1_DELTA_LAG_MAX`, while the 100 ms contract resolves
  to 10 packets at the nominal rate, so two thirds of the largest buffers in
  the detector went unused; they are now allocated to the real lag at
  configuration time, and a tracker configured with no capacity holds no ring
  at all. The full-band and subband coherences of one reference also read a
  single cross-product array, because the four subbands tile the live band
  exactly instead of each rebuilding all 56 products. Measured on a 100-packet
  window at lag 10, `ClassicDetector` falls from 33,160 to 28,256 bytes and
  `MLDetector` from 52,624 to 37,888 bytes, and the MicroPython coherence
  tracker gains roughly 8 percent with the host subband path gaining 10.
  Feature values are unchanged.

- **Host replay now hands the detectors plain integer CSI payloads**: the packet
  view stores `int8` NumPy arrays, and the MicroPython runtime indexes a payload
  dozens of times per packet, so every element read was building a NumPy scalar.
  The replay harness converts once per packet, which is worth roughly a factor
  of two in the profile extraction helpers and cuts the raw uncached Classic
  replay from 2.800 to 2.452 seconds over 26,564 packets. `int8` is already
  signed, so every value is preserved exactly and the replay feature checksum
  is unchanged.

- **Frequency coherence now walks the two live-band halves instead of scanning
  every subcarrier pair**: the DC null splits the HT20 live band into two
  contiguous 28-bin runs, so a pair at offset `d` is always `left + d` inside
  one run. `C++` drops the `56x56` scan it ran once per offset, and both `C++`
  and MicroPython cache the 56 squared magnitudes once per packet and share
  them across offsets `2`, `4`, and `12` through a combined entry point;
  MicroPython also reuses preallocated per-packet buffers, and the host NumPy
  path reads the halves as array views rather than materializing pair arrays.
  The pair sets (`52`, `48`, and `32`), the pair order, and the resulting
  features are unchanged, and `C++` `test_motion_detection` falls from roughly
  13.4 to 3.1 seconds. `test/python/test_frequency_coherence.py` and the new
  `core` helper tests pin all three implementations to the formula written out
  pair by pair from the bin table.

- **Warm-cache Python validation now avoids replay oversubscription and repeated
  feature extraction**: pytest auto-parallelism is capped at four workers in
  local and CI runs, packet-rate variants persist time-aware Classic and ML rows
  keyed by duration and target cadence, and long-recording raw-versus-row parity
  uses one deterministic representative while every curated recording retains
  its cached Classic and ML performance gates.

- **Host-side C++ test launchers now use all detected logical CPUs**: the normal
  runner, coverage workflow, and performance-report parity gate execute
  independent suites concurrently; `CTEST_PARALLEL_LEVEL` limits concurrency
  when required. Non-coverage test builds and the parity gate use
  `RelWithDebInfo`, reducing the full-corpus motion replay from roughly 85 to 14
  seconds without changing its Python parity or disabling test assertions;
  coverage remains a `Debug` build. The C++ and Python packet-rate regressions
  now focus 60-second prefixes on the supported 120, 100, and 80 pps boundary
  cases, and the Python gate runs as part of the normal test suite.

- **Classic host validation now reuses canonical time-aware feature rows**:
  paired, empty-room, long-recording, and performance-report replays recompute
  calibration and detector decisions from cached evaluation rows while
  preserving exact row-versus-packet results across native, high-rate, and
  decimated streams.

- **Host ML workflows now share one canonical time-aware persisted row cache**:
  training, dataset-quality validation, performance reporting, and ML replay
  tests reuse the same reset- and evaluation-aware feature artifacts. Validation
  derives idle summaries from those rows instead of persisting another cache;
  read misses no longer create directories, source-digest memoization also
  tracks change time, and `tools/prune_npz_cache.py` provides explicit cache
  maintenance.
- **Unclassified datasets now default to `exclude`**: training, validation, and
  performance-report workflows require an explicit `dataset_role` before a
  recording can enter training or reserved replay gates.
- **The maintained automated test surface now follows production boundaries**:
  C++, Micro-ESPectre, the host CLI and its direct dependencies, and detector
  performance/parity gates remain in the default suite. Standalone research
  tools are validated directly through their experiment workflows, and their
  dedicated unit tests were removed alongside tests for report generators,
  build scripts, and static configurations. Tombstone tests for removed
  commands, configuration, implementation details, and log text were also
  removed or rewritten around current positive contracts. Python and C++
  coverage gates remain active.
- **The production runtime ML feature surface is exactly the five exported features**: obsolete runtime extractors, ids, helpers, and tests were removed, while host-only research candidates live separately under `tools/` and cannot be exported. The current inventory, rejected predecessors, and candidate evidence are listed in [FEATURES.md](FEATURES.md).
- **The website now publishes a dedicated product roadmap**: the `#roadmap` SPA route and canonical `/roadmap/` page share one CI-generated content source, expose honest Available, Limited validation, Planned, and Research states, and distinguish Tuya Matter validation from a possible TuyaOpen/TuyaOS OEM integration. Strategic home-page references now connect embedded products, optional local/self-hosted/managed web orchestration, breathing research, and brief gesture research to that roadmap.
- **The website home page now pairs a cinematic ESP32-C3 setup hero with a scroll-driven sensing story**: the lightweight USB-C loop leads into seven full-screen, sticky scenes spanning lighting, security, climate, privacy-first industrial activity analytics, RF interference diagnostics, Matter integrations, and an animated multi-room orchestration map. Its visible router anchors direct and reflected paths, a moving person generates a localized constructive/destructive multipath field, and wall-mounted ESPectre nodes intensify in proportion to smoothed CSI deltas across representative subcarriers instead of acting as signal sources. The story also includes focused event UI, responsive controls, a reduced-motion fallback, and a matching social preview.
- **Classic false positives are now gated on empty-room recordings**: the long-standing "weak-link false alarm" gap did not survive measurement. The alarms occur on the strongest links as readily as on the weakest (`10` alarms at `-42.0 dBm` against none at `-84.3 dBm`), the L1 term contributes `0.04`-`0.24` logit units where turbulence autocorrelation contributes `4.20`-`7.93`, and the `12` empty-room recordings raise no alarm at all against `54` across the `29` static-presence baselines. They are the stationary occupant's own micro-motion, not detector error. `effective_alarms == 0` moved to the empty-room recordings for both detectors through a shared `replay_idle_stream`, static-presence baselines now carry a `12%` sanity bound instead of a `5%` false-positive gate, and the C++ harness gained empty-room discovery so both languages are gated on the same ground truth. See [2026-07-25-gate-classic-false-positives-on-empty-rooms.md](adr/2026-07-25-gate-classic-false-positives-on-empty-rooms.md).
- **Retired stale Jupyter notebooks and the offline moving-variance tooling cluster**: removed the outdated `notebooks/` walkthroughs, the unused `calculate_moving_variance` utility, Classic-recovery/MV naming leftovers, and the historical variance-baseline research scripts (`variance_baseline_core`, `analyze_filter_*`, `optimize_filter_params`). `SegmentationContext` now keeps only the shared turbulence buffer used by Classic/ML.
- **Sensing runtimes classify CSI formats before consuming them**: shared `CsiPipeline` (ESPHome, native, Matter), Micro-ESPectre, and the streamer enforce one classifier-first HT20 + HT-LTF + 64-subcarrier sensing contract, with per-reason drop telemetry and detector resets when the format stream changes. Host NPZ loaders, dataset-quality checks, the ML trainer, and the C++ test NPZ loader use the same sensing view and fail explicitly when format filtering removes every packet; historical captures without any per-record PHY metadata stay compatible only when the stored layout already matches the contract (`keep_all_phy=True` inspects mixed-PHY captures). The host cnpy helper now sizes NumPy unicode (`U*`) arrays correctly so PHY string fields load with the right stride.
- **ML trainer defaults to the exported model seed when `--seed` is omitted**: training and diagnostics such as `--cross-environment` / `--cross-chip` reuse the seed embedded in `ml_weights.py` / `ml_weights.h`; `--seed-search-until-improvement` still samples fresh seeds each trial.
- **ML trainer `--augment` applies the retained robustness recipe**: train-time feature jitter (`0.10`) plus moderate packet gain/noise/loss, with clean inference and paired validation.
- **Firmware CI now builds only the publishable chip matrix**: ESPHome, Native, Matter, and Streamer keep one five-chip build matrix per frontend, `develop` publishes a separate `snapshot-dev` release, and all QEMU smoke-test paths, examples, and helpers were removed because they added runtime and maintenance cost without validating Wi-Fi, BLE, or full chip coverage.
- **Website guides are now hands-on tutorials**: setup walks through firmware choice, browser flashing, per-frontend provisioning, calibration, and troubleshooting step by step; detection, hardware, and custom-firmware focus on practical decisions; pages now include real screenshots and capture visuals, and legacy images from the retired baselines were removed.
- **Paired real-data validation no longer hard-fails on per-feature Fisher separation**: legacy univariate feature gates were removed; model quality stays covered by Classic/ML paired metrics, long-quiet checks, and dataset quality admission.
- **Default `motion_on_hits` is now 4**: IDLE→MOTION now requires four consecutive evaluation hits across Python, ESP-IDF, and ESPHome defaults.
- **Tuning docs now explain evaluation cadence and hit-filter latency**: `TUNING.md` and `SETUP.md` document that default `IDLE -> MOTION` needs about `1 s` of sustained raw motion (`25` packets × `4` hits at `100` pps).
- **Performance report Effective Alarms now cover paired and long-quiet tables**: False Motion Evals were dropped from the published metrics; paired Classic/ML summaries sum filtered false MOTION transitions on static-presence segments.
- **Host paired validation, dataset-quality Classic replay, and C++/Python motion integration tests now use the production evaluation cadence**: performance report paired Classic/ML metrics, trainer paired gates, `validate_dataset_quality` Classic scores, and `test_motion_detection` sample detector state every `evaluation_interval` packets (default 25), matching long-quiet replay and deploy-time runtime policy instead of scoring every packet.
- **Matter now defaults back to Classic and is documented as available with limited validation**: published Matter firmware uses the conservative `classic` detector instead of `ml`, and the shared docs distinguish firmware availability from the still-limited controller coverage while ESPHome remains the more mature production integration surface.
- **ML trainer no longer exposes detector-guided sample weighting**: `--sample-weight-mode` and the L1-guided / hard-negative weighting paths were removed; training always starts from uniform sample weights, and the weight-matrix cache was dropped.
- **Dataset quality validation defaults to writing the markdown report and refreshing pair metadata**: `validate_dataset_quality` regenerates explicit static_presence/motion pairs on every run, updates `dataset_info.json` / `updated_at` only when content changes, and writes `DATASET_QUALITY_CHECK.md` unless `--no-report` is set; `--refresh-metadata`, `--strict`, and `--report` were removed. The report leads with the summary/domains and puts Validation rule last.
- **Dataset pair Ratio uses robust p95 separation**: Motion Scores report `Ratio = p95(motion)/threshold` instead of `max(motion)/threshold`.
- **ML trainer promotion no longer gates on long recordings**: quiet long-recording checks stay in the performance report and `test_validation_long_recordings.py`.
- **ML training exports by default again**: `--promote` was removed; use `--no-export` to leave runtime artifacts unchanged.
- **Home Assistant dashboards and browser tools now use the shared 0.0–1.0 probability scale**: gauge segments, Configure/Monitor/Game threshold ceilings, Theremin movement mapping, and the BLE `SET_THRESHOLD` example no longer assume the retired Classic 0–10 amplitude scale; dashboards also expose `select.espectre_detector`.
- **ML Hampel preprocessing now covers every production feature stream**: training, host replay, and Python/C++ runtimes apply the shared Hampel configuration to both turbulence and per-packet L1 profile deltas, with feature-cache invalidation requiring a clean retrain.
- **ML promotion now separates training data from reserved evaluation replays**: dataset entries carry `dataset_role` (`train`, `selection`, `holdout`, `exclude`), grouped CV splits by provenance lineage so synthetic derivatives share their real source's fold, paired plus quiet `empty` replays gate candidates at production cadence with runtime hit filtering, robust worst/tail grouped-CV metrics lead ranking (real sessions lead, synthetic sessions only guard against regressions), the sealed `holdout` role is evaluated once on the selected winner, and `exclude` keeps experimental captures cataloged while removing them from the current promotion flow. Static-presence replays allow one runtime-filtered alarm (sustained micro-motion of the present person is genuine motion), quiet `empty` replays require zero, and per-recording non-regression forbids exceeding the baseline's alarms anywhere. `--force-promote --seed` supports a deliberate, loudly reported baseline reset, and a gated multi-seed FP-weight campaign remains available. See [`docs/adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md`](adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md).
- **Real weak-link captures are stress diagnostics, not standard promotion material**: `low_rssi` pairs gate the ML detector at relaxed stress targets (recall above `90%`, FP below `10%`), Classic stays report-only on them while gaining strict per-pair gates on normal-link sessions, and the performance report separates normal-link tables (with ML split into reserved out-of-sample and in-sample training recordings) from a dedicated Low-RSSI stress section. In per-recording non-regression, weak replays ratchet only their alarm count; recall and FP move freely within the stress targets.
- **ML feature diagnostics now use deterministic grouped out-of-fold SHAP** with class-, chip-, and session-balanced training backgrounds and blocked held-out explanations, replacing in-sample random attribution.
- **Host `collect` pacing is adaptive by default**: `./espectre collect` now applies one chip-independent AIMD policy from sustained firmware TX backpressure, ignores isolated pressure and CSI callback deficits, and recovers toward the requested target unless `--fixed` is set.
- **Host `collect` inspects without saving unless `--label` is set**: live mode no longer needs `--no-save`; omitting `--label` runs inspection-only, and `--no-save` was removed.
- **Serial monitor reset is now opt-in**: `espectre monitor` attaches without resetting by default, while benchmark workflows pass `--reset` explicitly when they need boot-time markers or a clean restart.
- **Browser tools share a vertical movement bar**: The Game and Configure reuse `docs/web/movement-bar.js` and `movement-bar.css` for live movement and draggable threshold. Configure removes the old state/motion/threshold metric cards and uses a flatter settings layout with a slim detector/telemetry toolbar.
- **The website guide surface is now fully static**: direct HTML pages, a versioned sitemap, and focused CI checks replace Markdown-to-HTML generation and Pagefind indexing.
- **Internal CSI traffic generation is adaptive by default** across ESPHome, native, and Matter: one shared pacing task now regulates DNS or raw ICMP traffic from valid local CSI feedback, replacing the fixed-rate `esp_ping` session and keeping protocol-specific code limited to packet encoding and socket setup.
- **CSI Wi-Fi startup is now shared and association-safe** across frontends: the runtime applies protocol and HT20 policy at `WIFI_EVENT_STA_START`, initializes CSI after `IP_EVENT_STA_GOT_IP`, and passes the event gateway directly to the traffic generator. This avoids the ESPHome first-connect drop and removes duplicate frontend policy and netif lookups.
- **ESPHome, native, and Matter now share the same runtime foundations**: frontend setup, diagnostics, status reporting, and standalone Wi-Fi policy were consolidated to reduce duplication and keep behavior aligned.
- **The C++ source tree was normalized around explicit naming and layer placement**: `runtime/esp_idf/protocol/` became `frontend_support/`, `csi_manager` and `standalone_wifi_manager` became `csi_pipeline` and `standalone_wifi_service`, the streamer adapter is now `streamer_frontend`, HTTPS OTA follows the `ota_service_https` variant pattern, CSI layout constants moved from `utils.h` into `csi_format.h`, threshold validation moved into `threshold.h`, and shared `core/` and `runtime/` headers no longer include ESP-IDF-only headers.
- **The shared feature unit is now `csi_features`**: `core/features.h` and `micro_espectre/features.py` were renamed to `csi_features.h` and `csi_features.py` because the old basename shadowed the C library's `<features.h>` on host builds, and the portable `runtime_time` helper moved from `runtime/esp_idf/` into `runtime/`.
- **ESPectre Protocol was extracted from the native frontend into shared runtime code** so multiple ESP-IDF frontends can reuse the same telemetry, command, BLE, and provisioning helpers.
- **Native firmware was simplified into a dedicated standalone frontend**: BLE telemetry, MQTT diagnostics, device identity, and subscription behavior were cleaned up around the shared protocol contract.
- **Streamer workflows were modernized**: multi-chip CLI support was expanded, collection is now collector-driven, the C++ streamer protocol became the primary live-streaming path, and ESP32-C3 transport defaults were tuned for high-rate capture.
- **Dataset and sensing defaults were normalized across the project**: room-state labels were simplified, empty-room validation became part of the standard workflow, and the active runtime path now uses one fixed shared subcarrier set.
- **Classic startup calibration now adapts the learned probability boundary in logit space** using `75%` of the session's quiet `q95` displacement, restoring the aggregate C6 recall and S3 false-positive targets on the expanded real low-RSSI corpus.
- **Classic detection now uses direct weighted fusion without a recovery vote**, reducing runtime branches and calibration state.
- **Default runtime subcarriers were moved away from the DC bin**: the shared fixed 12-subcarrier set is now `[4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]`, improving current Classic real-data validation while keeping one cross-chip default band.
- **Hardware gain lock was removed completely**: ESPectre now keeps AGC active on all chips and uses one shared CV-normalized turbulence path (`std/mean`) across runtime, collection, datasets, and tooling. This avoids the forced-gain instability and Wi-Fi RX/TX problems that may lead to packet loss.
- **Matter build and CI flows were hardened**: published targets use the standard ESP-IDF path, commissioning behavior is stricter, and releases/snapshots now stay aligned with the standard firmware build path.
- **Firmware build optimization is consistent across frontends**: native, Matter, and streamer now default to ESP-IDF size optimization, matching ESPHome's release-oriented `-Os` profile for comparable firmware size and detector timing.
- **Shared detector timing is now continuously aggregated**: every runtime state-evaluation tick contributes to thread-safe duration, sample-count, minimum, and maximum statistics, while the firmware benchmark uses an exact sample-weighted average and excludes empty telemetry windows.
- **Repository tooling and docs were aligned with the new platform direction**: `./me` became `./espectre`, host-side tools now live at the top level (`collect`, `ui`, `mqtt`, `monitor`), `micro` is limited to MicroPython device commands, ESP-IDF frontend namespaces focus on build/flash, serial logs use the frontend-agnostic `monitor` command, the MQTT monitor was renamed from `espectre-monitor.html` to `espectre-mqtt.html`, ESPHome packaging no longer relies on symlinks, and the main docs were rewritten around the modular multi-frontend architecture.
- **Published firmware now separates installation and update formats**: releases and snapshots publish 15 full-flash images for ESPHome, Matter, and native, plus 5 application-only OTA payloads for native. GitHub Pages stages only the full-flash images, ESPHome Device Builder compiles updates from the adopted configuration, and native firmware resolves its per-chip OTA manifest from GitHub Releases. Streamer remains source-built because its Wi-Fi credentials are supplied at build time.
- **ESP32-S2 support was removed** because the target had no recorded hardware validation; supported firmware targets are now ESP32, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6.
- **Streamer transport and host collect were hardened for long live sessions**: the streamer now prefers PSRAM-backed staging where available, exposes more precise retry/duplicate telemetry (`wifi_dup`, `stim_dup`, `retry`), suspends BLE during sustained active streaming to reduce coexistence pressure, and uses chip-specific Wi-Fi defaults for `ESP32` and `ESP32-S3` transport tuning.
- **Streamer wire and dataset metadata now use a clean-break contract**: the old dedicated streamer metadata was removed from the streamer-collector exchange, CSI stream header, host parser, and checked-in capture datasets, with repository `.npz` samples migrated to dataset format `1.2`.
- **Host collection and serial monitoring are more resilient**: `collect` now reports requested and effective `SO_RCVBUF`, rolling status output was simplified around live session state, and `monitor` now uses the project `pyserial` path with auto-reconnect, port reuse, and raw-byte mode.
- **Host-side analysis tooling was modularized internally**: shared helpers were split into `tools/lib/` so end-user entrypoints stay at the top level, with dataset metadata resolution, CSI I/O, plotting helpers, path helpers, and paired variance-baseline sweep logic now living in focused internal modules instead of the old monolithic helper files.
- **The tooling support detector moved from the historical moving-variance baseline to the Classic/L1-delta path**: the legacy `optimal_threshold_gridsearch` metadata was retired, and dataset pair validation plus the detection-methods comparison now replay the production Classic startup calibration directly on the selected quiet capture.
- **ML seed search now evaluates every requested seed** and keeps the strongest robust improvement, instead of stopping at the first candidate that beats the baseline.
- **Micro-ESPectre was reorganized under src/python/micro_espectre/**: the runtime/device sources now live in a dedicated subdirectory.
- **ESPHome baseline `2026.6.0`**; examples now require `min_version: 2026.6.0`.
- **The Python baseline was raised from `3.12` to `3.14`** across the main workflow and the ML training environment.
- **The ML training stack was migrated from TensorFlow/Keras to PyTorch**: the trainer now runs on the PyTorch MLP path, exports the same runtime weights, and no longer produces the unused TFLite/scaler artifacts.
- **The ML runtime now exposes direct probabilities on a `0.0-1.0` scale**: Python, C++, and training-side reference inference use the raw sigmoid output, so the published movement metric is now a probability and the default binary decision threshold is `0.5`.
- **ML documentation was split by concern**: dataset collection stays in `docs/ML_DATA_COLLECTION.md`, while training, export, and validation guidance now live in `docs/ML_TRAINING.md`.
- **The roadmap was realigned around the platform split**: `v3` now defines the reusable local platform phase, while `v4` is positioned as an optional privacy-first orchestration layer across multiple ESPectre nodes.

---

## [2.8.0] - 2026-05-21 - Detection hardening, ML cross-chip reliability, and runtime motion policy

- Detection and calibration were hardened across stacks: tighter NBVI defaults, Hampel enabled by default, a 100-packet default window, and a clearer edge-driven motion policy.
- ML reliability improved across chips with shared CV-normalized turbulence, a refreshed 9-feature model, and stricter training/data quality controls.
- `ping` became the default CSI traffic source, `./me detect` was added for live ML inference, and notebooks plus CI/test coverage were expanded.

---

## [2.7.0] - 2026-03-17 - ESPectre configuration over BLE and subcarrier normalization

- BLE runtime control became a first-class standalone integration surface, including live threshold updates and a Web Bluetooth example client.
- CSI normalization was extended to `256->128`, `228->114`, and `114->128` payload remaps, with aligned behavior and tests across C++ and Micro-ESPectre.

---

## [2.6.0] - 2026-03-08 - ESP32-C5 Support, Context-Aware Calibration, and Stricter Validation Targets

- ESP32-C5 support was added and runtime handling on newer chips (`C5`/`C6`) was hardened.
- Calibration, thresholds, dataset metadata, and ML feature selection were aligned more strictly across C++ and Micro-ESPectre.
- Validation targets were tightened to `Recall >95%` and `FP <5%`, with related tooling and deploy diagnostics improved.

---

## [2.5.1] - 2026-02-23 - HT STBC Multi-Antenna Router Fix

- Fixed HT STBC CSI handling on ESP32-C5/C6 with multi-antenna routers by accepting 256-byte packets and using the first HT20 estimate.
- Fixed Micro-ESPectre NBVI calibration memory issues on ESP32-C3, improved calibration speed, and refreshed performance/snapshot documentation.

---

## [2.5.0] - 2026-02-15 - ML Detector, Training Pipeline & Pre-built Firmware

- Added the first experimental ML detector in both ESPHome/C++ and Micro-ESPectre/Python, with a training and weight-export pipeline.
- Added pre-built firmware releases for all supported ESP32 variants.
- Removed the PCA detector and the older P95 calibrator, leaving MVS plus NBVI as the main non-ML path at the time.

---

## [2.4.0] - 2026-01-24 - Live Recalibration, Adaptive Threshold & PCA

- Added live recalibration, adaptive thresholds by default, and a choice between MVS and experimental PCA detection.
- Standardized the runtime around HT20 CSI, improved calibration/subcarrier handling, and expanded tooling, tests, and Micro-ESPectre support.

---

## [2.3.0] - 2025-12-31 - End of Year Edition

- Added `ESPectre - The Game`, a browser-based motion-controlled tuning and demo client.
- Added sensor customization, external traffic mode, `ping` traffic generation, and configurable gain-lock behavior.
- Improved channel-change handling, NBVI calibration, and board support, including tested ESP32-C3 and original ESP32 paths.

---

## [2.2.0] - 2025-12-19 - Gain Lock, Low-Pass Filter & ML Data Collection

- Added gain-lock stabilization, low-pass filtering, and baseline variance normalization to make calibration more stable.
- Tightened NBVI behavior, moved variance evaluation to publish time, and auto-configured the required ESP-IDF options in the ESPHome path.
- Added the first labeled ML data-collection infrastructure (`me collect`, `.npz`, and `csi_utils.py`) plus broader testing/documentation.

---

## [2.1.0] - 2025-12-10 - Made for ESPHome Compliance

- All example configs were brought in line with "Made for ESPHome" requirements, including provisioning, dashboard import, and project metadata.
- Variance and Hampel behavior were unified and optimized across C++ and MicroPython.
- The test suite and coverage pipeline were expanded substantially.

---

## [2.0.0] - 2025-12-06 - ESPHome Native Integration

- Major platform migration from standalone ESP-IDF firmware to an ESPHome native integration for Home Assistant.
- Established the dual-platform model: ESPHome/C++ for production, and Micro-ESPectre/MicroPython for R&D and rapid experimentation.
- Migrated tests and CI toward the ESPHome-oriented workflow with host-side CMake/CTest coverage.

---

## [1.5.0] - 2025-12-03 - Automatic Subcarrier Selection

### Automatic Subcarrier Selection
- Zero-configuration subcarrier selection using NBVI (Normalized Baseline Variability Index) algorithm. 
- Auto-calibration at boot, re-calibration after factory_reset.
- Formula: `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`. 
- Achieves F1=97.6% (Recall 95.3%, Precision 100%, FP 0%). 

---

## [1.4.0] - 2025-11-28 - Major Refactoring & Technical Debt Reduction

### Major Refactoring
- **Feature extraction module**: Extracted to `csi_features.c/h`, reduced `csi_processor.c` by 50%
- **Configuration centralization**: All defaults in `espectre.h`, validation in `validation.h/c`
- **Two-pass variance**: Numerically stable calculation
- **Traffic generator**: Max rate 1000 pps (was 50), default 100 pps
- **CLI migration**: Bash → Python (cross-platform)
- **Wi-Fi Theremin**: `tools/web/espectre-theremin.html` for CSI sonification
- **Removed**: Redundant segmentation parameters (min_length, max_length, k_factor)

---

## [1.3.0] - 2025-11-22 - ESP32-C6 Platform Support

### ESP32-C6 Platform Support
- **WiFi 6 (802.11ax)** support with proper CSI configuration
- **Runtime-configurable parameters**: threshold, window_size via MQTT
- **Web Monitor**: `tools/web/espectre-monitor.html` with real-time visualization
- **System monitoring**: CPU/RAM usage in stats command
- **MQTT optimization**: Simplified message format, removed segment tracking

---

## [1.2.1] - 2025-11-17

### Wi-Fi Optimization
ESP-IDF best practices: disabled power save (`WIFI_PS_NONE`), configurable country code, HT20 bandwidth.

---

## [1.2.0] - 2025-11-16 - Simplified Architecture & MVS Segmentation

### Simplified Architecture
- **MVS algorithm**: Moving Variance Segmentation with adaptive threshold
- **Amplitude-based features**: +151% separation improvement for skewness/kurtosis
- **Traffic generator**: ICMP ping-based (was UDP broadcast)
- **64 subcarriers**: All available (was 52 filtered)
- **10 features**: Added temporal_delta_mean, temporal_delta_variance

---

## [1.1.0] - 2025-11-08

### Auto-Calibration System
- **Fisher's criterion**: Automatic feature selection (4-6 from 8)
- **Butterworth filter**: Order 4, cutoff 8Hz
- **Wavelet filter**: Daubechies db4 for high-noise environments
- **NVS persistence**: Configuration survives reboots
- **Modular architecture**: Split into 10 specialized modules

---

## [1.0.0] - 2025-11-01

### Initial Release
CSI-based movement detection for ESP32-S3. Hampel + Savitzky-Golay filters, 15 features, 4-state detection (IDLE/MICRO/DETECTED/INTENSE), MQTT publishing, CLI tool. 10-100 pps, <50ms latency, 3-8m range.
