# C++ Review — Layering, Duplication, Performance, Frontend Alignment

Date: 2026-07-28
Branch: `v3.0` (at `7df6361`)
Scope: all first-party C++ under `src/cpp/` (~18k lines). Excluded:
`src/cpp/frontend/matter/app/managed_components/`, `build*/`, and the vendored
connectedhomeip tree.

Baseline at review time: `ctest --test-dir test/cpp/build` → 27/27 passing.
Every finding below is on a green tree; none is a currently failing test.

Review status: **Complete.** All findings are resolved; rejected directions and
deferred feature work are recorded explicitly rather than left as open fixes.

---

## 1. How To Read This Document

Findings carry a stable id so they can be referenced from commits and issues:

| Prefix | Theme |
| --- | --- |
| `L-n` | Layer boundary violations |
| `D-n` | Dead or unreachable code |
| `R-n` | Repetition to centralize |
| `B-n` | Behaviour divergence and latent bugs |
| `P-n` | Performance |
| `M-n` | Minor correctness, style, and robustness |

Severity uses three levels:

- **High**: observable wrong behaviour, or a resource cost worth acting on now.
- **Medium**: correct today, but the design invites the bug or blocks reuse.
- **Low**: cosmetic, or cleanup with no behavioural consequence.

Section 2 is the single source of truth for progress. Detailed findings do not
repeat their status.

---

## 2. Progress Tracker

### Batch 1 — Mechanical cleanup, no behaviour change

- [x] **L-1** (Medium) — Move the log shim below its consumers → [§3](#l-1)
- [x] **D-3** (Low) — Remove the constant-`false` `is_extending()` alias → [§4](#d-3)
- [x] **D-5** (Low) — Delete or adopt the six test-only `core/` helpers → [§4](#d-5)
- [x] **R-4** (Low) — Share the identical frontend listener bodies → [§5](#r-4)
- [x] **M-1** (Low) — Drop the redundant null check → [§8](#m-1)
- [x] **M-2** (Low) — Include `<cstdio>` for `std::snprintf` → [§8](#m-2)
- [x] **M-3** (Low) — Use the schema constant for `motion_off_hits_` → [§8](#m-3)
- [x] **M-4** (Low) — Fix `interval_us_raw_()` visibility or naming → [§8](#m-4)

### Batch 2 — The stale ML metric

- [x] **B-1** (High) — Clear the ML metric on buffer reset → [§6](#b-1)

### Batch 3 — Runtime deduplication

- [x] **R-1** (Medium) — Share the traffic config conversion → [§5](#r-1)
- [x] **R-2** (Medium) — Extract a shared ESP-IDF runtime base → [§5](#r-2)
- [x] **L-2** (Medium) — Stop the shared status logger querying the radio → [§3](#l-2)
- [x] **B-7** (Low) — Declare capabilities instead of defaulting them → [§6](#b-7)
- [x] **D-4** (Low) — Use or delete the unused validators → [§4](#d-4)
- [x] **M-5** (Low) — Range-validate integer Kconfig values → [§8](#m-5)
- [x] **M-6** (Medium) — Recover when the STA-start policy was missed → [§8](#m-6)
- [x] **M-7** (Low) — Split the hidden side effect in `set_detection_algorithm()` → [§8](#m-7)

### Batch 4 — Shared numerics

- [x] **P-1** (Medium) — Reorder the rings without per-element modulo → [§7](#p-1)
- [x] **R-3** (Medium) — Share the two-pass mean and variance → [§5](#r-3)
- [x] **P-4** (Low) — Document or split the shared feature scratch → [§7](#p-4)
- [x] **P-2** (Low) — Gate the DEBUG-only timer reads → [§7](#p-2)

### Batch 5 — The dead startup floor

Resolved: the maintainer confirmed the path is abandoned, so it was deleted
rather than repaired. `sizeof(StartupThresholdCalibrator)` went from 4636 to
328 bytes, a measured saving of **4308 bytes**.

- [x] **D-1** (High) — Drop the unused startup floor path → [§4](#d-1)
- [x] **D-2** (Medium) — Moot: `floor_snapshot()` was removed with D-1 → [§4](#d-2)

### Batch 6 — Cadence and rate adaptation

The time-relative design was already chosen and partly shipped: the evaluation
cadence runs on packet timestamps on both device runtimes. Only the detector
sizing had never left the host tooling. Batch 6 finished that port and then
measured it, and the measurement rejected half of it.

- [x] **B-2** (High) — Align calibration and detection cadence → [§6](#b-2)
- [x] **B-3** (High) — Revert the lag derivation; measured as a regression → [§6](#b-3)
- [x] **B-4** (Medium) — Gate ML readiness on the L1 ring → [§6](#b-4)
- [x] **B-5** (Medium) — Clamp `ClassicDetector::lag_` → [§6](#b-5)
- [x] **P-3** (Low) — Correct the `profile_ring_` cost comment → [§7](#p-3)

### Batch 7 — Make the L1 lag ratio genuinely time-relative

Follows directly from the B-3 measurement: the L1 feature cannot be made
time-relative by scaling one lag, because it is a ratio of two. Doing it
properly is a feature change that needs a Classic refit and an ML retrain, so it
is its own batch rather than a fix inside batch 6.

- [x] **B-8** (High) — Deferred after measurement; keep the fitted `10:1` offsets → [§6](#b-8)

### Unbatched — needs a product decision

- [x] **B-6** part 2 — Native honours `ready_to_publish` → [§6](#b-6)
- [x] **B-6** part 3 — Matter null-check made consistent → [§6](#b-6)
- [x] **B-6** part 1 (Medium) — Publish filtered MQTT motion edges immediately and retain periodic telemetry → [§6](#b-6)

---

## 3. Layer Boundary Violations

The intended dependency direction is `Frontend -> Runtime contract -> Runtime
implementation -> Core`. A lower layer must never include from a higher one.

### L-1 — `core/` included `runtime/espectre_log.h` {#l-1}

**Severity**: Medium

**Where**

- [base_detector.cpp:14](../../src/cpp/core/base_detector.cpp)
- [classic_detector.cpp:9](../../src/cpp/core/classic_detector.cpp)
- [ml_detector.cpp:14](../../src/cpp/core/ml_detector.cpp)
- [filters.cpp:14](../../src/cpp/core/filters.cpp)

**What**

All four included `"espectre_log.h"` from the higher `runtime` layer. `core` is
the bottom layer, so this was an upward include in the strictest sense.

**Why it matters**

`AGENTS.md` grants a placement exception letting portable shims live in
`runtime/` rather than `runtime/esp_idf/`. That exception explains why the file
may sit outside the ESP-IDF directory; it does not authorise it to sit *above*
its mandatory consumer. Any future attempt to build `core` standalone — for a
host-side parity harness, a fuzzer, or a separately licensed distribution — pulls
in a `runtime/` include path for nothing.

**Resolution**

The shim now lives at
[espectre_log.h](../../src/cpp/core/espectre_log.h), below all its consumers.
The placement rule now permits a portable shim in `core/` when `core` itself
depends on it.

**Risk**: none. Header relocation only, verified by the host test build.

---

### L-2 — Shared status logger queries the radio {#l-2}

**Severity**: Medium

**Where**: [periodic_sensing_status_logger.cpp:11-12, :50-56](../../src/cpp/runtime/periodic_sensing_status_logger.cpp)

**What**

A file in the platform-agnostic `runtime/` layer includes `esp_wifi.h` behind
`__has_include` and calls `esp_wifi_sta_get_ap_info()` to obtain RSSI and
channel for its log line.

**Why it matters**

Two problems, one structural and one architectural:

- Placement: `AGENTS.md` states that anything including ESP-IDF belongs in
  `runtime/esp_idf/`. The `__has_include` guard makes it compile on the host,
  but the file is not a shim — it is a status formatter that happens to also do
  I/O.
- Responsibility inversion: a *logger* independently sources data the runtime
  already has. `CsiPipeline` tracks `current_channel_` and receives RSSI per
  packet. Two sources of truth for the same two numbers, one of which is sampled
  at a different instant from the snapshot it is printed next to.

**How to fix**

Add the two fields to the snapshot and make the logger pure:

```cpp
// runtime/runtime_snapshot.h
struct RuntimeSnapshot {
  // ... existing fields
  int8_t link_rssi_dbm{INT8_MIN};
  uint8_t link_channel{0};
};
```

`CsiPipeline` already sees both values on the packet path; surface them through
an accessor and have `EspIdfRuntime` copy them into `snapshot_` where it already
copies `movement_metric` and `threshold`. Then delete the `esp_wifi.h` include
and the `ESPECTRE_HAVE_ESP_WIFI` block from the logger.

Side benefit: the logged RSSI becomes the RSSI of the packets that produced the
metric, instead of a fresh AP query taken at print time.

**Risk**: changes which RSSI value appears in logs (from AP-record RSSI to
per-packet RSSI). Both are correct; the new one is more meaningful. No detection
behaviour is affected.

---

## 4. Dead Code

### D-1 — The startup-floor path is dead on both runtimes {#d-1}

**Severity**: High — the highest-yield item in the review.

**Where**

- Contract: [base_detector.h:180](../../src/cpp/core/base_detector.h)
- Storage and logic: [threshold.h:80-83, :246-267, :325-357, :359-440, :518-531, :612-635](../../src/cpp/core/threshold.h)
- Plumbing: [esp_idf_runtime.cpp:455-460](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp)
- Python mirror: [threshold.py:117, :149-160, :202-214, :335-345, :429-441](../../src/python/micro_espectre/threshold.py)
- Only consumer: [csi_io.py:1069-1071](../../tools/lib/csi_io.py), behind a `hasattr` probe

**What**

`BaseDetector::get_startup_floor_metric()` is declared virtual with a `0.0f`
default and is never overridden by `ClassicDetector` or `MLDetector`. On the
Python side, `csi_io.py` looks for `detector.apply_startup_floor` with `hasattr`
and no detector implements it. The feature is inert end to end.

What is still paid for on every calibration:

| Element | Cost |
| --- | --- |
| `floor_ring_[STARTUP_FLOOR_SIZE]` — 1000 floats | **4000 bytes** |
| `bootstrap_floor_[2][25]` | 200 bytes |
| `chunk_floor_[25]` | 100 bytes |
| `bootstrap_floor_counts_[2]`, `chunk_floor_count_`, `floor_idx_`, `floor_count_` | ~8 bytes |
| `std::fill_n` per motion chunk | per-packet work |
| `record_floor_samples_()` ring writes | per-packet work |
| `std::copy` of bootstrap floor buffers | per-chunk work |

Every value written is the constant `0.0f`.

**Why it matters**

`StartupThresholdCalibrator` is heap-allocated for the duration of calibration
only, so this is transient rather than permanent. It is still ~4.3 KB claimed at
the worst possible moment: boot, on a part that is simultaneously bringing up
Wi-Fi, BLE or Matter, and MQTT. The per-packet work runs inside the CSI callback
across the full 1000-packet calibration budget.

**How to fix**

Delete the feature symmetrically on both runtimes. This is a `Hard Constraint`
change (shared calibration algorithm), so C++ and Python must land in the same
commit.

C++ removals in `threshold.h`:

- Constants `STARTUP_FLOOR_SIZE`, `STARTUP_FLOOR_MIN`, `STARTUP_FLOOR_DISPERSION_CUT`
- Members `floor_ring_`, `floor_idx_`, `floor_count_`, `chunk_floor_`,
  `chunk_floor_count_`, `bootstrap_floor_`, `bootstrap_floor_counts_`
- Methods `floor_snapshot()`, `record_floor_samples_()`, `clear_floor_ring_()`
- The `floor_metric` parameter of `observe()` and `observe_motion_chunk_()`
- The `floor_samples` / `floor_sample_count` parameters of
  `consume_closed_motion_chunk_()`

`core/base_detector.h`: remove `get_startup_floor_metric()`.

`runtime/esp_idf/esp_idf_runtime.cpp`: drop the argument from the `observe()` call.

Python mirror: the same removals in `micro_espectre/threshold.py`, plus the
`hasattr`-guarded block in `tools/lib/csi_io.py`.

Tests to update: `test/cpp/suites/core/test_core_helpers.cpp:320-328`,
`test/python/test_threshold.py:23-33, :105-119, :224`. Also
`test/cpp/support/csi_replay_metrics.h:211` and
`test/cpp/suites/integration/test_low_rssi.cpp:42`, which pass the argument.

**Outcome**

Deleted on both runtimes in one change, per the maintainer's confirmation that
the path is abandoned rather than parked.

- `sizeof(StartupThresholdCalibrator)`: 4636 -> 328 bytes, **4308 saved**,
  measured by compiling the header before and after rather than summing the
  member sizes.
- The per-packet `std::fill_n` and ring writes are gone from the CSI callback.
- `test_motion_first_preserves_validated_quiet_floor_samples` covered both the
  floor snapshot *and* a long-quiet-prefix calibration scenario the other tests
  do not reproduce. It was rewritten as
  `test_motion_first_accepts_after_a_long_quiet_prefix`, asserting the
  calibration outcome, on both runtimes.
- Not touched: `_motion_floor` / `motion_floor_()`, which is the motion-level
  floor feeding `threshold_metric()` and is unrelated to the variance-floor
  snapshot despite the name.

**Verification**: the 356 replay metric lines are bit-for-bit identical to the
pre-change baseline, confirming the removed values never influenced any output.
C++ 27/27, Python 1051 passed.

**Decision record**: the abandoned direction and the measured memory saving are
preserved in
[2026-07-28-drop-the-unused-startup-variance-floor.md](../adr/2026-07-28-drop-the-unused-startup-variance-floor.md).

---

### D-2 — `floor_snapshot()` permutes the ring it snapshots {#d-2}

**Severity**: Medium (latent; becomes real if [D-1](#d-1) is kept instead of deleted)

**Where**: [threshold.h:246-267](../../src/cpp/core/threshold.h)

**What**

The method is non-`const` and calls `std::nth_element` twice directly on
`floor_ring_`, reordering it in place, while `floor_idx_` and `floor_count_`
continue to treat that array as a circular buffer.

**Why it matters**

Calling it before calibration completes silently corrupts the ring: subsequent
`record_floor_samples_()` writes land at `floor_idx_`, which no longer
corresponds to the oldest sample. Today only a test calls it, after completion,
so nothing observes the corruption — but nothing in the signature or the
documentation communicates the "call once, at the end" contract either.

**How to fix**

If [D-1](#d-1) is applied, this disappears with it. If the feature is kept:

- Make the method `const` and copy into a scratch before `nth_element`.
  `STARTUP_FLOOR_SIZE` is 1000 floats, too large for the CSI callback stack — so
  either shrink the ring, accept a second heap buffer, or compute median and p99
  with a streaming estimator.
- Failing that, document the contract on the declaration and add a
  `snapshot_taken_` latch that makes a second call, or any later
  `record_floor_samples_()`, an explicit error rather than silent corruption.

---

### D-3 — `is_extending()` is a constant-`false` alias {#d-3}

**Severity**: Low

**Where**: [threshold.h:173-174](../../src/cpp/core/threshold.h), mirrored at
[threshold.py:361](../../src/python/micro_espectre/threshold.py)

**What**

```cpp
/// Backward-compatible alias: startup no longer extends past the nominal target.
bool is_extending() const { return false; }
```

The only caller is `test/cpp/suites/core/test_core_helpers.cpp:203`, which
asserts it returns false.

**How to fix**

Delete it on both runtimes and delete the assertion. The comment already records
that startup no longer extends; if that fact is worth preserving, it belongs in
the class docstring, not in a function that exists to return a literal.

---

### D-4 — Unused validators and a vacuous factory {#d-4}

**Severity**: Low

**Where**: [runtime_config_utils.cpp:29-35, :76](../../src/cpp/runtime/runtime_config_utils.cpp)

**What**

- `validate_runtime_uint32()` and `validate_runtime_uint8()` have no callers
  anywhere in `src/` or `test/`.
- `make_runtime_sensing_config()` returns `RuntimeConfig{}` and has exactly one
  caller, which could default-construct directly.

**How to fix**

The two validators are the natural fix for [M-5](#m-5) — wire them up there
rather than deleting them. If [M-5](#m-5) is declined, delete both.

`make_runtime_sensing_config()` can go either way: delete it, or keep it as the
documented seam where sensing defaults would diverge from `RuntimeConfig{}`. If
kept, say that in a comment, because the current body reads as an oversight.

---

### D-5 — Test-only `core/` helpers shadowed by inline reimplementations {#d-5}

**Severity**: Low individually, Medium as a pattern

**What**

Six helpers ship in `core/` but are called only from `test/`, while the
production path open-codes the identical maths:

| Helper | Declared at | Production equivalent |
| --- | --- | --- |
| `hampel_filter()` | [filters.cpp:101](../../src/cpp/core/filters.cpp) | `hampel_filter_turbulence()` repeats median + MAD |
| `calculate_magnitude()` | [utils.h:159](../../src/cpp/core/utils.h) | `extract_subcarrier_amplitudes()` recomputes `sqrt(I*I+Q*Q)` |
| `normalize_amplitude_profile()` | [utils.h:176](../../src/cpp/core/utils.h) | `L1DeltaTracker::process()` normalizes inline, on two branches |
| `calculate_median_u8/i8()` | [utils.h:62, :78](../../src/cpp/core/utils.h) | none — no integer median is taken in production |
| `compare_float/int8/float_abs()` | [utils.h:203-233](../../src/cpp/core/utils.h) | none — `qsort` is no longer used anywhere |
| `calculate_spatial_turbulence()` / `..._from_csi()` | [csi_format.h:149, :238](../../src/cpp/core/csi_format.h) | `..._from_amplitudes()` |

**Why it matters**

This is the worst of both worlds: the firmware image carries the helper, the
tests assert against the helper, and the shipping code path executes a *second*
implementation the tests never touch. A divergence between the two is invisible
to the suite.

**How to fix**

Decide per helper, do not batch blindly:

- **Delete**: `calculate_median_u8`, `calculate_median_i8`, `compare_float`,
  `compare_int8`, `compare_float_abs`. No production role, and no plausible
  future one now that `std::sort` replaced `qsort`.
- **Adopt**: `calculate_magnitude()` and `normalize_amplitude_profile()`. Call
  them from `extract_subcarrier_amplitudes()` and `L1DeltaTracker::process()`
  respectively. This is the DRY-correct direction and immediately puts the tests
  on the real path. Note `L1DeltaTracker::process()` fuses normalization with the
  two displacement sums in one pass, so adopting the helper there costs an extra
  pass over 12 floats — measure before committing, or adopt only in the fallback
  branch.
- **Keep, with a comment**: `calculate_spatial_turbulence()` and
  `..._from_csi()`. They are the documented Python-parity entry points and the
  replay harness uses `..._from_csi()`. Mark them as parity/test entry points so
  the next reader does not assume the firmware runs them.
- **Delete**: `hampel_filter()`. `hampel_filter_turbulence()` is the only
  stateful form used, and the free function's tests should be retargeted onto it.

While in `csi_format.h`, note that `calculate_spatial_turbulence()` hardcodes
`float valid_mags[12]` and the literal `12` twice instead of
`HT20_SELECTED_BAND_SIZE`.

---

## 5. Repetition To Centralize

### R-1 — Traffic config conversion duplicated across the two runtimes {#r-1}

**Severity**: Medium

**Where**: [esp_idf_runtime.cpp:34-51](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp)
and [stream_esp_idf_runtime.cpp:45-62](../../src/cpp/runtime/esp_idf/stream_esp_idf_runtime.cpp)

**What**

`to_traffic_mode()` is byte-identical. `to_csi_traffic_config()` differs on
exactly one line — the fallback when `csi_traffic_mode == INTERNAL` and
`traffic_generator_rate == 0`:

| Runtime | Fallback |
| --- | --- |
| `EspIdfRuntime` | `CsiTrafficMode::EXTERNAL` |
| `StreamEspIdfRuntime` | `CsiTrafficMode::PACING` |

**How to fix**

One shared helper in `runtime/`, with the fallback as a parameter. It has no
ESP-IDF dependency (`CsiTrafficServiceConfig` is a plain struct), so it belongs
in the shared layer next to `csi_traffic_types.h`:

```cpp
CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config,
                                              CsiTrafficMode idle_fallback);
```

Both runtimes then call it with their own fallback and the anonymous namespaces
shrink to nothing.

---

### R-2 — `notify_fault_` and runtime state duplicated across the two runtimes {#r-2}

**Severity**: Medium

**Where**: [esp_idf_runtime.cpp:528-534](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp)
and [stream_esp_idf_runtime.cpp:312-318](../../src/cpp/runtime/esp_idf/stream_esp_idf_runtime.cpp)

**What**

The two `notify_fault_` bodies are identical except for the default message
string (`"Unknown runtime fault"` vs `"Unknown stream runtime fault"`). Along
with them, both classes independently declare `config_`, `snapshot_`,
`capabilities_`, `listener_`, `last_fault_`, `services_armed_`,
`setup_complete_`, and `debug_telemetry_`.

**How to fix**

Two options, in increasing order of ambition:

1. **Minimal** — a `RuntimeFaultReporter` helper in `runtime/` holding
   `listener_` and `last_fault_`, with the tag and default message passed at
   construction. Removes the duplicated function; leaves the members duplicated.
2. **Recommended** — an `EspIdfRuntimeBase : public IEspectreRuntime` in
   `runtime/esp_idf/` owning the eight shared members plus `notify_fault_`,
   `set_listener()`, `get_snapshot()`, and `get_capabilities()`. Both runtimes
   already implement those four identically. `StreamEspIdfRuntime` additionally
   returns constant `false` from four control methods; those can become the
   base's defaults so the stream runtime only overrides what it genuinely does.

Option 2 also gives [R-1](#r-1) a natural home.

---

### R-3 — Mean and variance hand-rolled three times in `core/` {#r-3}

**Severity**: Medium

**Where**

| Site | Code |
| --- | --- |
| [utils.h:30-37, :129-150](../../src/cpp/core/utils.h) | `calculate_mean()`, `calculate_variance_two_pass()` — the canonical pair |
| [classic_detector.cpp:83-95](../../src/cpp/core/classic_detector.cpp) | `calculate_turb_autocorr_()` unrolls both loops by hand |
| [csi_features.h:220-232](../../src/cpp/core/csi_features.h) | `compute_ml_series_stats()` unrolls both loops by hand |

**Why it matters**

Three implementations of the same two loops, inside the same layer, all of which
must produce bit-identical results because the outputs feed a threshold. The
comment block in `ml_detector.cpp` documents in detail how a rounding difference
in this arithmetic flipped whole detection decisions and broke the Python parity
gate. That is precisely the risk three copies create.

**How to fix**

Have both call sites use `calculate_variance_two_pass()`. The helper is already
two-pass `float` with the same accumulation order, so this should be a
byte-identical substitution — verify with the parity gate rather than by
inspection.

`compute_ml_series_stats()` needs the mean afterwards for `mean_denom`, so factor
a small struct rather than computing the mean twice:

```cpp
// core/utils.h
struct MeanVariance { float mean; float variance; };
inline MeanVariance calculate_mean_variance_two_pass(const float* values, size_t n);
```

`calculate_variance_two_pass()` then becomes a one-line wrapper, and both
detectors share one accumulation order by construction.

**Risk**: touches the arithmetic the ML/Classic parity gate protects. Run the
full C++ suite and `TestPerformanceMetrics` before and after, comparing reported
metrics rather than just pass/fail.

---

### R-4 — Identical listener bodies across three frontends {#r-4}

**Severity**: Low

**Where**

| Hook | ESPHome | Native | Matter |
| --- | --- | --- | --- |
| `on_threshold_changed` | [espectre.cpp:100-102](../../src/cpp/frontend/esphome/espectre/espectre.cpp) | [native_frontend.cpp:178-180](../../src/cpp/frontend/native/espectre/native_frontend.cpp) | [matter_frontend.cpp:75-78](../../src/cpp/frontend/matter/espectre/matter_frontend.cpp) |
| `on_detector_changed` | [espectre.cpp:109-111](../../src/cpp/frontend/esphome/espectre/espectre.cpp) | [native_frontend.cpp:185-187](../../src/cpp/frontend/native/espectre/native_frontend.cpp) | — |

All open with the same two statements: `runtime_.record_snapshot(snapshot)`
followed by writing the snapshot value back into `runtime_.config()`.

**How to fix**

[runtime_listener_utils.h](../../src/cpp/runtime/esp_idf/runtime_listener_utils.h)
already exists for exactly this pattern and holds `finalize_frontend_calibration`.
Add the two missing siblings:

```cpp
inline void apply_threshold_snapshot(RuntimeFrontendController &runtime,
                                     const RuntimeSnapshot &snapshot);
inline void apply_detector_snapshot(RuntimeFrontendController &runtime,
                                    const RuntimeSnapshot &snapshot);
```

Each frontend then keeps only its ecosystem-specific tail.

---

## 6. Behaviour Divergence And Latent Bugs

### B-1 — `MLDetector` publishes a stale probability after a buffer clear {#b-1}

**Severity**: High — the only finding with a user-visible effect today.

**Where**

| Behaviour | `ClassicDetector` | `MLDetector` |
| --- | --- | --- |
| `update_state()` not-ready branch | zeroes metric **and** forces `IDLE` ([classic_detector.cpp:113-117](../../src/cpp/core/classic_detector.cpp)) | zeroes metric only ([ml_detector.cpp:120-123](../../src/cpp/core/ml_detector.cpp)) |
| `clear_buffer()` | zeroes `current_probability_` ([classic_detector.cpp:248-256](../../src/cpp/core/classic_detector.cpp)) | does not ([ml_detector.cpp:208-211](../../src/cpp/core/ml_detector.cpp)) |
| `reset()` | overridden, clears metric ([classic_detector.cpp:238-246](../../src/cpp/core/classic_detector.cpp)) | not overridden |

**Failure scenario**

1. ML is the active detector and `current_probability_` sits at, say, `0.91`
   (motion).
2. A Wi-Fi channel change fires — or calibration finishes — and
   `clear_detector_buffer_deferred_()` runs.
3. `MLDetector::clear_buffer()` clears the turbulence ring and the L1 rings but
   leaves `current_probability_ == 0.91`.
4. The next periodic publish reads
   `snapshot_.movement_metric = detector_->get_motion_metric()` and ships `0.91`
   to Home Assistant, MQTT, or Matter.
5. The value persists until the ring refills — up to `window_size` packets, about
   one second at the nominal cadence, longer on a slow link.

With `ClassicDetector` the same path publishes `0.0`. The two detectors report
differently through the same code path, and the ESPHome intensity sensor shows
it directly.

**How to fix**

*Minimal* — three additions in `ml_detector.cpp`: force `IDLE` in the not-ready
branch, zero the metric in `clear_buffer()`, and add a `reset()` override.

*Recommended* — remove the possibility of drift. Both detectors keep the
identical `current_probability_` member, the identical `get_motion_metric()`
body, and the identical clearing obligation. Hoist it:

```cpp
// core/base_detector.h
float get_motion_metric() const { return current_metric_; }   // no longer virtual
// ...
protected:
  float current_metric_{0.0f};
```

`BaseDetector::clear_buffer()` and `BaseDetector::reset()` then zero
`current_metric_` and set `state_ = MotionState::IDLE` in one place. Each
detector assigns `current_metric_` at the end of its `update_state()` and keeps
its override only for its *extra* state — Classic's `current_logit_`,
`current_lag_ratio_`, and `current_turb_autocorr_`.

**Verification**: add a regression test asserting `get_motion_metric() == 0.0f`
immediately after `clear_buffer()` for **both** detectors, parameterised over the
detector type so the next one added inherits the check.

---

### B-2 — Calibration and detection evaluate on different cadences {#b-2}

**Severity**: High

**Where**

- Steady state: [csi_pipeline.cpp:205-213](../../src/cpp/runtime/esp_idf/csi_pipeline.cpp)
- Calibration: [esp_idf_runtime.cpp:449-457](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp)
- Interceptor return: [csi_pipeline.cpp:170-173](../../src/cpp/runtime/esp_idf/csi_pipeline.cpp)

**What**

In steady state the pipeline advances evaluation on **elapsed packet arrival
time** as soon as the rate estimator is trusted, falling back to a packet count
only during warmup:

```cpp
const bool cadence_due =
    packet_rate_.ready()
        ? elapsed_since_evaluation_us_ >= EVALUATION_INTERVAL_US
        : packets_since_evaluation_ >= evaluation_interval_;
```

During calibration the interceptor consumes the packet and returns `true`
*before* that block is ever reached. Calibration therefore always uses the
**packet counter** `config_.evaluation_interval`, and as a side effect
`packet_rate_` is never fed and `last_packet_us_` never advances across the whole
~1000-packet calibration budget.

**Why it matters**

The calibrator's chunking (`STARTUP_MOTION_CHUNK_SIZE`, the gate ring, the
quiet/motion classification) is defined in units of *evaluations*. If calibration
produces evaluations at one rate and steady-state detection produces them at
another, the threshold is fitted against a different feature resolution than the
one it will be applied to. On a stream running well off the nominal 100 pps the
two diverge by exactly the ratio of measured to nominal cadence.

This compounds with [B-3](#b-3): `detector_timing.h` documents that getting
cadence-derived quantities wrong at 1000 pps moved false positives from 0.0 % to
17.8–32.7 %.

**How to fix**

Move the cadence decision out of both call sites into one place:

1. Extract the `cadence_due` decision into an `EvaluationCadence` helper in
   `runtime/` owning `packet_rate_`, `last_packet_us_`,
   `elapsed_since_evaluation_us_`, and `packets_since_evaluation_`, exposing
   `observe(arrival_us) -> bool due`.
2. Feed the arrival timestamp to that helper **before** the interceptor check in
   `process_normalized_packet_`, so calibration and detection see the same clock.
3. Have `handle_threshold_calibration_packet_` ask the same helper whether an
   evaluation is due, instead of counting packets itself.

The `packet_weight` argument to `observe()` already carries "how many packets
this evaluation represents", so the calibrator needs no change beyond receiving a
correct weight.

**Risk**: changes calibration outcomes on off-nominal streams. It is a
`Hard Constraint` area — the Python calibrator must adopt the same cadence rule.
Validate with the replay suite (`test_packet_rate_adaptation`,
`test_long_recordings`, `test_empty_rooms`) and compare per-session recall and
false-positive rates before and after, not just pass/fail. Gate on the **worst**
session, not the pooled mean.

---

### B-3 — `derive_detector_timing()` is exercised only by tests {#b-3}

**Severity**: High

**Where**

- Definition: [detector_timing.h:88-126](../../src/cpp/core/detector_timing.h)
- Only callers: `test/cpp/support/csi_replay_metrics.h:174`,
  `test/cpp/suites/integration/test_packet_rate_adaptation.cpp:310, :442`
- Production construction: [esp_idf_runtime.cpp:297-310](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp)

**What**

The firmware always builds `ClassicDetector(window_size, threshold)` with the
default arguments, so `lag = L1_DELTA_LAG = 10` and `autocorr_lag = 1`,
regardless of the measured cadence. `derive_detector_timing()`, the
`L1_DELTA_LAG_US` / `TURB_AUTOCORR_LAG_US` contract, and the
`RATE_ADAPTATION_DEAD_BAND` logic are never reached on device.

The `ClassicDetector` constructor documentation explicitly describes a caller
that does not exist:

> Callers that know the measured cadence pass the counts spanning L1_DELTA_LAG_US
> and TURB_AUTOCORR_LAG_US instead […]

**Why it matters**

Two distinct problems:

- **Test/production mismatch.** The replay suite validates a rate-adaptive
  detector; the firmware runs a fixed-lag one. Metrics reported by
  `test_packet_rate_adaptation` do not describe shipped behaviour on any stream
  outside the dead band.
- **Measured risk.** `detector_timing.h` records that restoring the
  autocorrelation lag to its 10 ms scale at 1000 pps takes false positives from
  17.8–32.7 % down to 0.0 %. Firmware on a fast stream is running the
  configuration that scores 17.8–32.7 %.

**How to fix**

Pick one and make the code say so.

*Option A — wire it up (preferred if fast streams are supported).*

`CsiPipeline` already owns a `PacketRateEstimator`. Add a reconfiguration path:

1. Once `packet_rate_.ready()` and the derived timing differs from the active
   one, post a deferred event to the runtime loop (never reconfigure from the CSI
   callback).
2. `EspIdfRuntime` rebuilds the detector with `DetectorTiming{window_packets,
   lag, autocorr_lag}` and restarts calibration, since the previous threshold was
   fitted at the old resolution.
3. Debounce it: the dead band already suppresses churn near nominal, but a
   hysteresis latch is still needed so a jittery link does not rebuild the
   detector repeatedly.

This requires a `ClassicDetector` constructor call passing `timing.lag` and
`timing.autocorr_lag`, and it makes [B-5](#b-5) live rather than latent — fix
that one in the same change.

*Option B — declare the scope.*

If the supported envelope is genuinely 80-133 pps, state it in
`detector_timing.h` and in `classic_detector.h`, remove the misleading
constructor comment, and mark `derive_detector_timing()` as a replay-analysis
helper. The replay suite should then also pin the nominal timing so it measures
what ships.

**Outcome: Option A was implemented, then measured, and the lag derivation is
rejected.**

The standard replay suite could not settle it: the corpus sits at 90-120 pps,
inside the 25% dead band, so every metric is identical with and without the
change. Two purpose-built A/B replays were run instead, both holding the window
pinned at 100 and varying only the detector lags.

*Low-rate arm* — the 22 normal-link paired recordings, each decimated to 75, 65,
and 55 pps, 66 comparable cells:

| | value |
| --- | --- |
| cells where derived lags are worse | 13 |
| cells where derived lags are better | 3 |
| recall delta, mean | `-0.76` |
| recall delta, worst cell | `-11.0` |
| false-positive delta, mean | `-0.14` |

The sign is systematically negative and grows with the deviation from nominal:
`-7.8`, `-7.7`, `-4.4`, `-3.1`, and `-2.3` recall points on individual sessions.
There is no false-positive benefit to trade against it.

*High-rate arm* — the two `>= 500 pps` source pairs, decimated across 500 to 50
pps. At 500 pps the derived lags do cut false positives, from `9.2%` to `0.0%`
on the pair that has margin, but at a cost of `16.5` recall points. That regime
is replay-only: no supported part delivers 500 pps.

**Why it fails.** `delta_lag_ratio` is not a lag, it is a *ratio of two* lags:

```
ratio = mean(|profile[t] - profile[t-lag]|) / mean(|profile[t] - profile[t-1]|)
```

Deriving from elapsed time rescales only the numerator, while the denominator
stays pinned at "the previous packet". The coefficients were fitted on a `10:1`
relation; at 70 pps the derived lag makes it `7:1`, which is a different feature.
A third arm confirms it directly: deriving *only* the turbulence-autocorrelation
lag and leaving the L1 lag at its nominal value makes the low-rate regressions
disappear entirely, reproducing the fixed-lag numbers exactly at 70, 60, and 55
pps.

So the "lags are durations" argument holds for the autocorrelation lag and fails
for the L1 lag, because there the physical quantity is a ratio rather than an
interval.

**Resolution**

The lag derivation was reverted on both runtimes. Calibration starts on connect,
and production constructs both detectors with the fitted nominal offsets.
[B-2](#b-2), [B-4](#b-4), [B-5](#b-5), [P-3](#p-3), and the replay-only
`window_override` remain because they are independently correct. The rejected
direction and its measurement are preserved in
[2026-07-28-keep-production-feature-lags-at-nominal-offsets.md](../adr/2026-07-28-keep-production-feature-lags-at-nominal-offsets.md).

**Reproducing the measurement.** Both arms monkeypatch
`dataset_metadata.derive_detector_timing` to pin the window at
`SEG_WINDOW_SIZE` and, for the control arm, the lags at `L1_DELTA_LAG` and `1`,
then call `compute_classic_packet_result` on packets decimated with
`_decimate_packets` from `test_packet_rate_adaptation_regression.py`. The
throwaway scripts are not in the repository; promote them into `tools/` if
[B-8](#b-8) is pursued, because the same harness is what validates it.

---

### B-4 — ML L1 capacity and readiness depend on Classic's arithmetic {#b-4}

**Severity**: Medium (latent)

**Where**

| | `ClassicDetector` | `MLDetector` |
| --- | --- | --- |
| L1 capacity | uses configurable `lag_` ([classic_detector.cpp:43-45](../../src/cpp/core/classic_detector.cpp)) | hardcodes `L1_DELTA_LAG` ([ml_detector.cpp:178-184](../../src/cpp/core/ml_detector.cpp)) |
| `is_ready()` | overridden to require a full L1 ring ([classic_detector.cpp:71-74](../../src/cpp/core/classic_detector.cpp)) | not overridden — base checks only the turbulence ring |

**What**

`MLDetector` maintains the L1 tracker whenever the exported model needs it, and
`ML_FEAT_L1_DELTA_LAG_RATIO` is in the production feature set. But its readiness
gate ignores the L1 ring entirely.

It happens to be safe today: with `window_size = 100` and `lag = 10`, the
turbulence ring fills at packet 100 and the delta ring (capacity 90) also fills
at packet 100, because deltas start at packet 11. The two coincide *by
arithmetic accident*, and only because ML pins the lag to the constant.

**Why it matters**

The moment [B-3](#b-3) is wired up, or the model's feature set changes the L1
capacity relation, ML starts inferring on a partially filled delta ring with no
gate to stop it, and `delta_lag_ratio()` returns its no-motion sentinel `1.0`
rather than signalling "not ready".

**How to fix**

Give `MLDetector` the same gate Classic has, and pull the shared part up into
`BaseDetector`. Also route `MLDetector::l1_delta_capacity_()` through the
tracker's configured lag rather than the constant, so both detectors compute
capacity the same way.

---

### B-5 — `ClassicDetector` does not clamp `lag_` {#b-5}

**Severity**: Medium (latent; becomes live with [B-3](#b-3) Option A)

**Where**: [classic_detector.cpp:30](../../src/cpp/core/classic_detector.cpp) vs
[l1_delta_tracker.h:113-117](../../src/cpp/core/l1_delta_tracker.h)

**What**

```cpp
lag_(lag > 0U ? lag : 1U),          // ClassicDetector: no upper bound
```

```cpp
lag_ = std::min<uint16_t>(lag > 0U ? lag : 1U, L1_DELTA_LAG_MAX);   // tracker: clamped
```

With `lag > L1_DELTA_LAG_MAX` (32), the detector computes
`l1_delta_capacity_() = window_size_ - lag_` and gates `is_ready()` on the
*unclamped* value, while the tracker measures displacement at the *clamped* one.
The reported lag ratio then describes a different physical interval than the
detector believes.

Unreachable today because the only source of non-default lags is
`derive_detector_timing()`, which already caps at `L1_DELTA_LAG_MAX`, and that
function has no production caller.

**How to fix**

Clamp in the constructor with the same bound. Better still: have
`L1DeltaTracker::configure()` return the lag it actually adopted and have the
detector store *that*, so a single clamp exists.

---

### B-6 — Frontends disagree on when motion state is published {#b-6}

**Severity**: Medium

**Where**

| Frontend | `on_motion_state_changed` | Honours `ready_to_publish`? |
| --- | --- | --- |
| ESPHome | publishes binary sensor immediately ([espectre.cpp:65-74](../../src/cpp/frontend/esphome/espectre/espectre.cpp)) | yes |
| Matter | publishes occupancy immediately ([matter_frontend.cpp:57-64](../../src/cpp/frontend/matter/espectre/matter_frontend.cpp)) | yes |
| Native | records the snapshot only ([native_frontend.cpp:167-169](../../src/cpp/frontend/native/espectre/native_frontend.cpp)) | **no** |
| Streamer | records the snapshot only ([streamer_frontend.cpp:74](../../src/cpp/frontend/streamer/espectre/streamer_frontend.cpp)) | n/a (detector-free) |

**Three distinct issues**

1. **Latency asymmetry.** A native/MQTT client sees a motion transition only at
   the next `on_periodic_update`, up to `publish_interval` packets later — about
   one second at the nominal rate. ESPHome and Matter see it immediately. The
   native BLE path *does* push state per evaluation tick, so within the native
   frontend itself BLE and MQTT disagree.
2. **`ready_to_publish` ignored.** Native's `on_motion_state_changed` and
   `on_periodic_update` never test the flag, so MQTT telemetry goes out during
   the not-ready window the other two frontends suppress.
3. **Inconsistent null checks.** Matter dereferences `bindings_` without a check
   in `on_motion_state_changed` while checking it in `on_runtime_fault`.
   `setup()` rejects a null `bindings_`, so it is safe — but the reader cannot
   tell which of the two styles is the intended invariant.

**How to fix**

1. **Done.** Native MQTT now publishes filtered motion-state transitions
   immediately, matching ESPHome's edge-driven state behavior, while retaining
   periodic telemetry as a heartbeat and current-metrics snapshot. The protocol
   documents the hybrid cadence.
2. **Done.** Both Native MQTT publication paths return before publishing when
   `ready_to_publish` is false, matching ESPHome and Matter. The snapshot is
   still recorded either way.
3. **Done.** The redundant `bindings_` null check in
   `MatterFrontend::on_runtime_fault` is gone. `setup()` refuses a null
   `bindings_` and the runtime only calls back after a successful `setup()`, so
   the pointer is an invariant, which is what `on_motion_state_changed` already
   assumed.

---

### B-7 — `supports_ble_telemetry` defaults to `true` {#b-7}

**Severity**: Low

**Where**: [runtime_capabilities.h:13-20](../../src/cpp/runtime/runtime_capabilities.h),
set only by `StreamEspIdfRuntime`

**What**

`EspIdfRuntime` sets only `supports_runtime_detector_selection`. Every other flag
keeps its struct default, so Matter and ESPHome both advertise
`supports_ble_telemetry == true` despite having no BLE telemetry surface.

**Why it matters**

`ESPECTRE_PROTOCOL.md` tells clients not to infer command support from other
fields, precisely because the capability block is the contract. A capability that
is `true` by default rather than by declaration undermines that.

**How to fix**

Flip the struct defaults to the conservative value (`false`) and have each
runtime, or better each frontend, declare what it actually offers.
`NativeFrontend` is the only one implementing `on_live_telemetry` with a real
payload, so it is the only one that should set the flag. Note
`RuntimeFrontendController` caches capabilities at setup, so the frontend would
need to declare before `setup()` or the controller would need a post-setup
amendment hook.

---

### B-8 — Scale both lags of the L1 ratio together {#b-8}

**Severity**: High — this is the only route that makes the L1 feature actually
rate-independent, and [B-3](#b-3) showed the half-measure is worse than doing
nothing.

**Where**

- Feature: [l1_delta_tracker.h](../../src/cpp/core/l1_delta_tracker.h), the
  `lagged_` and `adjacent_` window pair and `delta_lag_ratio()`
- Python mirror: [csi_features.py](../../src/python/micro_espectre/csi_features.py)
- Coefficients: `CLASSIC_L1_CENTER`, `CLASSIC_L1_SCALE`, `CLASSIC_L1_WEIGHT` in
  [classic_detector.h](../../src/cpp/core/classic_detector.h) and their Python
  counterparts
- Exported model: `ML_FEAT_L1_DELTA_LAG_RATIO` and `ML_FEAT_L1_DELTA_AUTOCORR`
  in [csi_features.h](../../src/cpp/core/csi_features.h)

**What**

The tracker measures displacement at two distances and divides them. The lagged
distance is configurable; the adjacent one is hardcoded to the previous packet,
which is a *packet* offset and therefore a different physical interval at every
rate. The ratio is only meaningful when both ends scale together.

At the nominal rate the pair is `100 ms / 10 ms`. Deriving only the numerator at
70 pps gives `100 ms / 14.3 ms`; keeping both nominal gives `143 ms / 14.3 ms`,
which preserves the `10:1` relation and is why the fixed lags measure better
today. Neither is the contract.

**How to fix**

1. Make the reference distance a parameter. `L1DeltaWindow adjacent_` currently
   reads `profile_ring_[profile_index_ - 1]`; it needs its own configurable
   offset, and the profile ring has to be sized for the larger of the two rather
   than for `lag_` alone.
2. Have `derive_detector_timing()` return both, resolved from one duration pair
   (`L1_DELTA_LAG_US` and a new `L1_DELTA_REFERENCE_LAG_US = 10000`), so the
   ratio holds its physical meaning at any cadence and collapses to `10:1` at
   100 pps.
3. **Refit Classic.** The feature definition changes, so `CLASSIC_L1_CENTER`,
   `CLASSIC_L1_SCALE`, and `CLASSIC_L1_WEIGHT` no longer describe it. The
   fitting path is `tools/fit_classic_detector.py`, which already resolves the
   timing per recording.
4. **Retrain ML.** Both L1 features feed the exported model, so `ml_weights.h`
   has to be regenerated. This is the expensive half.
5. Validate with the same two-arm replay that rejected [B-3](#b-3): the low-rate
   regressions must disappear, and the high-rate false-positive benefit must
   arrive without the recall cost. Gate on the worst session per rate, not on the
   pooled mean.

**Why it is worth doing anyway**

`profile_ring_` is already sized to `L1_DELTA_LAG_MAX`, so the memory is spent
whether or not the second lag is configurable, and the autocorrelation lag —
which is a plain interval — already benefits from derivation with no downside at
low rates. The L1 ratio is the last quantity in the detector whose meaning still
depends on the delivered packet rate.

**Why it is not a batch 6 fix**

It changes what the feature *is*, so it cannot land without new coefficients and
a new model. Shipping the plumbing without the refit would reproduce exactly the
regression [B-3](#b-3) measured. Treat the refit and the retrain as the work,
and the code change as the small part.

**Resolution**

Deferred after measurement. The deployed C++ and MicroPython runtimes keep the
fitted `10:1` L1 offsets and lag-1 turbulence autocorrelation, and v3 declares
`80-133 pps` as the supported detector envelope. The consistent two-lag design
remains a future feature change, not an open correctness fix: it must include
the Classic refit, the ML retrain, and per-session non-regression evidence.
The decision and the rejected partial derivation are preserved in
[2026-07-28-keep-production-feature-lags-at-nominal-offsets.md](../adr/2026-07-28-keep-production-feature-lags-at-nominal-offsets.md).

---

## 7. Performance

### P-1 — Ring reordering costs a modulo per element {#p-1}

**Severity**: Medium

**Where**: [base_detector.cpp:132-136](../../src/cpp/core/base_detector.cpp) and
[l1_delta_tracker.h:231-235](../../src/cpp/core/l1_delta_tracker.h)

**What**

```cpp
for (uint16_t i = 0; i < buffer_count_; i++) {
    ordered_turbulence_[i] = turbulence_buffer_[(buffer_index_ + i) % window_size_];
}
```

At `window_size = 100` that is 100 integer divisions plus 100 element copies per
call. `ordered_turbulence()` runs once per evaluation for Classic and once for
ML, and `build_series()` runs once per ML evaluation. Integer division is not
single-cycle on the Xtensa and RISC-V cores in the supported parts.

**How to fix**

The wrapped branch only runs when `buffer_count_ == window_size_`, so the split
point is known and the copy is two contiguous runs:

```cpp
const uint16_t tail = static_cast<uint16_t>(window_size_ - buffer_index_);
std::memcpy(ordered_turbulence_, turbulence_buffer_ + buffer_index_,
            tail * sizeof(float));
std::memcpy(ordered_turbulence_ + tail, turbulence_buffer_,
            buffer_index_ * sizeof(float));
```

`L1DeltaTracker::build_series()` takes the same treatment, with the extra
short-circuit that when `lagged_.count < capacity_` the data is already
contiguous from index 0 and needs a single `memcpy`.

**Verification**: bit-identical output, so the existing parity tests suffice. The
detector-timing line in the `[telemetry]` debug log gives a direct before/after
measurement on device.

---

### P-2 — Unconditional timer reads for DEBUG-only telemetry {#p-2}

**Severity**: Low

**Where**: [csi_pipeline.cpp:216, :225](../../src/cpp/runtime/esp_idf/csi_pipeline.cpp)

**What**

Two `esp_timer_get_time()` calls bracket every evaluation solely to feed
`detection_timing_`, which exists to produce one `[telemetry]` line every ten
seconds at `DEBUG` level. They run even when debug telemetry is compiled out —
`RuntimeDebugTelemetry` already has a no-op variant.

**How to fix**

Guard both reads on the same compile-time condition that selects the no-op
telemetry class, so the release build pays nothing. Keep the runtime path
unconditional when telemetry is compiled in — a runtime branch would cost about
as much as the read.

---

### P-3 — `profile_ring_` sized for a lag production never uses {#p-3}

**Severity**: Low

**Where**: [l1_delta_tracker.h:263](../../src/cpp/core/l1_delta_tracker.h), rationale
at [detector_limits.h:42-48](../../src/cpp/core/detector_limits.h)

**What**

`float profile_ring_[L1_DELTA_LAG_MAX][HT20_SELECTED_BAND_SIZE]` is
32 × 12 × 4 = 1536 bytes, statically sized, while production always runs
`lag = 10` (480 bytes used). The header documents the trade as deliberate and
quotes a cost of "4.5 KB of static memory", which does not match the 1.5 KB the
declaration actually reserves — worth reconciling either way.

**How to fix**

Conditional on [B-3](#b-3):

- If B-3 Option A lands, keep the current size — it is exactly what the ceiling
  exists for — and correct the 4.5 KB figure in the comment.
- If B-3 Option B lands, size the ring to `L1_DELTA_LAG` and reclaim about 1 KB,
  noting in the comment that the bound follows the declared rate envelope.

---

### P-4 — Two series share one scratch with no documented contract {#p-4}

**Severity**: Low

**Where**: [ml_detector.cpp:102-107](../../src/cpp/core/ml_detector.cpp) and
[csi_features.h:277-285](../../src/cpp/core/csi_features.h)

**What**

`extract_ml_features_by_id()` passes the same `MLSeriesScratch` to
`compute_ml_series_stats()` for the turbulence series and then for the L1-delta
series. It is safe because `MLSeriesStats` holds only scalars, so the first
call's scratch contents are dead before the second call overwrites them.

**Why it matters**

Nothing in the signature communicates that. A future feature keeping a pointer
into the sorted view — a percentile, a trimmed mean, a histogram — would read the
wrong series with no compiler diagnostic and no test failure, because both series
produce plausible values.

**How to fix**

Cheapest: a comment on the `MLSeriesScratch` parameter stating that it is reused
across calls and that `MLSeriesStats` must therefore stay pointer-free. Stronger:
carve two non-overlapping slices from the existing `feature_scratch_` block,
which is already sized for it, at the cost of `window_size` extra floats.

---

## 8. Minor Findings

### M-1 — Redundant null check {#m-1}

[csi_pipeline.cpp:169](../../src/cpp/runtime/esp_idf/csi_pipeline.cpp) writes
`data != nullptr ? data->rx_ctrl.rssi : INT8_MIN` after `data` was already
rejected at `:150`. Dead branch; drop the conditional.

### M-2 — `std::snprintf` without `<cstdio>` {#m-2}

[esp_idf_runtime.cpp:361](../../src/cpp/runtime/esp_idf/esp_idf_runtime.cpp) uses
`std::snprintf`; the translation unit includes `<cstring>` but not `<cstdio>`.
Compiles today via a transitive include. Add the direct include.

### M-3 — Magic literal beside a schema constant {#m-3}

[csi_pipeline.h:218-219](../../src/cpp/runtime/esp_idf/csi_pipeline.h):

```cpp
uint8_t motion_on_hits_{RUNTIME_MOTION_ON_HITS_DEFAULT};
uint8_t motion_off_hits_{3};
```

Use `RUNTIME_MOTION_OFF_HITS_DEFAULT`, which is already `3`. Both are overwritten
by `EspIdfRuntime::setup()`, so this is about the default drifting silently, not
about current behaviour.

### M-4 — Public method with private naming {#m-4}

[detector_timing.h:224](../../src/cpp/core/detector_timing.h):
`interval_us_raw_()` sits in the public section but carries the trailing
underscore this file uses for private members and helpers. Its only callers are
`ready()` and `interval_us()`, both internal — move it below `private:`.

### M-5 — Integer config bypasses the range validation floats get {#m-5}

[runtime_sensing_kconfig.cpp:103-114](../../src/cpp/runtime/esp_idf/runtime_sensing_kconfig.cpp)
casts raw Kconfig integers for `segmentation_window_size`, `publish_interval`,
`evaluation_interval`, `motion_on_hits`, and `motion_off_hits` with no bounds
check, while floats go through `parse_float_or_default_()` with min/max and a
warning. The `RUNTIME_*_MIN` / `_MAX` constants exist for all of them, and so do
the validators — see [D-4](#d-4).

Downstream clamping partly covers it (`BaseDetector` clamps the window,
`set_evaluation_interval` and `set_motion_*_hits` force a floor of 1), but a bad
`sdkconfig` silently produces a different runtime than requested with no log line.

Fix: add an integer counterpart to `parse_float_or_default_()` using
`validate_runtime_uint32` / `validate_runtime_uint8`, which turns [D-4](#d-4)
from dead code into the fix for this.

### M-6 — A missed `STA_START` drops the connect event permanently {#m-6}

**Severity**: Medium

[wifi_lifecycle.cpp:379-386](../../src/cpp/runtime/esp_idf/wifi_lifecycle.cpp)
applies the CSI radio policy on `WIFI_EVENT_STA_START`. If that event fired
before `register_handlers()` ran, `started_policy_err_` stays at the
`ESP_ERR_INVALID_STATE` seeded at registration. Then:

1. `IP_EVENT_STA_GOT_IP` arrives and is posted to `connected_event_`.
2. `process_pending_events()` **consumes** the event with `take()`, calls
   `init()`, which returns the stored error, and returns early.
3. `connected_callback_` is never invoked, so CSI never starts.
4. The event is gone. Nothing retries until a genuine disconnect/reconnect cycle.

Ordering saves this today — the ESPHome component sets
`get_setup_priority() == 275.0f`, above ESPHome's `WIFI` priority of 250, and
`StandaloneWifiService` registers before calling `start()`. But the failure mode
is silent and unrecoverable.

Fix: in `init()`, if `started_policy_applied_` is false, apply the policy there
rather than failing. At `GOT_IP` the station is up, so the call is valid, and it
is strictly better than dropping the connection. Keep the existing `ESP_LOGE` so
the ordering anomaly stays visible.

### M-7 — `set_detection_algorithm()` has a hidden side effect {#m-7}

[espectre.h:57-60](../../src/cpp/frontend/esphome/espectre/espectre.h):

```cpp
void set_detection_algorithm(const std::string &algo) {
  this->runtime_.config().detection_algorithm = parse_detection_algorithm(algo.c_str());
  this->runtime_.config().runtime_detector_selection_enabled = true;
}
```

Choosing *which* algorithm also enables NVS-backed runtime selection. It works
because the codegen always emits the call (`__init__.py` gives
`detection_algorithm` a default), but the coupling is invisible from the YAML
surface and from the setter name.

Fix: either a separate `set_runtime_detector_selection(bool)` driven by its own
YAML key, or set the flag unconditionally in the constructor with a comment that
ESPHome always exposes the select entity. Compare with
`NativeFrontend::set_runtime_config()`, which sets it explicitly and readably.

---

## 9. Commit Guidance

Suggested commit subjects per batch, matching the tracker in [§2](#2-progress-tracker):

| Batch | Commits |
| --- | --- |
| 1 | `refactor(core): move the log shim below its consumers`<br>`refactor(core): drop unused statistical helpers`<br>`refactor(frontend): share the snapshot-application listener bodies` |
| 2 | `fix(core): clear the ML metric on buffer reset` |
| 3 | `refactor(runtime): extract the shared ESP-IDF runtime base`<br>`fix(runtime): declare capabilities instead of defaulting them`<br>`fix(runtime): recover when the STA-start policy was missed` |
| 4 | `perf(core): reorder the detector rings without per-element modulo`<br>`refactor(core): share the two-pass mean and variance` |
| 5 | `refactor(core): drop the unused startup floor path`, plus an ADR if the direction is formally abandoned |
| 6 | Split per finding; do not fold into an earlier batch, so a regression stays attributable |
| 7 | `feat(core): scale both lags of the L1 ratio with the cadence`, plus the Classic refit and the ML retrain as their own commits, and an ADR |

---

## 10. Verification

Host-side only. Firmware builds are run separately by the maintainer; nothing
here requires `idf.py`.

C++ baseline (27/27 at review time):

```bash
cmake -S test/cpp -B test/cpp/build && cmake --build test/cpp/build && ctest --test-dir test/cpp/build --output-on-failure
```

Detector-focused subset, for batches touching `core/`:

```bash
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
```

Python baseline:

```bash
pytest test/python -v
```

Performance and parity gate, required for batches 4, 5, and 6:

```bash
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v
```

For batches 4 and 6, capture the metric values before and after rather than
relying on the pass/fail verdict, and compare the **worst** session rather than
the pooled average — a pooled figure has hidden a per-pair regression on this
project before. `docs/performance/README.md` holds the current targets.

---

## 11. Maintainer Decisions {#open-questions}

- **[D-1](#d-1)** — the startup-floor path was confirmed abandoned and deleted.
- **[B-3](#b-3)** — off-nominal cadence is a supported envelope and the
  time-relative design was already chosen, so the port was finished rather than
  scoped out. Measurement rejected runtime lag derivation, which was reverted.
- **[B-8](#b-8)** — the two-lag refit and retrain are deferred. Production keeps
  the fitted `10:1` offsets inside the declared `80-133 pps` envelope.
- **[B-6](#b-6)** — MQTT publishes filtered motion edges immediately and keeps
  periodic telemetry as a heartbeat; BLE remains the opt-in per-evaluation
  low-latency surface.
