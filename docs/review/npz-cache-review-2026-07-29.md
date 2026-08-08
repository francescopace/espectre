# NPZ Cache Review — Correctness, Centralization, and Reuse

Date: 2026-07-29
Branch: `v3.0`
Scope: the shared NPZ cache introduced by `667e4dc` (`refactor(tools): centralize fine-grained npz cache`) and its consumers: `tools/lib/npz_cache.py`, `tools/lib/csi_io.py`, `tools/lib/performance_report.py`, `tools/train_ml_model.py`, `tools/validate_dataset_quality.py`, and the maintained Python test suite.

The review asks three questions: is the cache correct, is it centralized, and do all the intended consumers actually use it. It also records the remaining execution-speed opportunities that the cache design exposes.

---

## 1. How To Read This Document

Findings carry a stable id so they can be referenced from commits and issues:

| Prefix | Theme |
| --- | --- |
| `B-n` | Broken behavior shipped by the refactor |
| `K-n` | Cache identity and key design |
| `R-n` | Runtime cache safety and resource use |
| `C-n` | Consumer coverage, bypass surface, and documentation |
| `P-n` | Execution-speed opportunities beyond the defects |

Severity uses three levels:

- **High**: the cache can return or persist a wrong result, permanently fail to hit, or break a maintained check.
- **Medium**: the cache stays usable, but it duplicates work, grows without bound, or exposes an unsafe contract.
- **Low**: local cleanup, discoverability, or workflow tuning.

Section 2 is the single source of truth for progress. Detailed findings do not repeat their status.

Review completion and finding resolution are separate states. The review itself is complete as of the date above; the checklist tracks resolution.

---

## 2. Progress Tracker

### Batch 1 — Restore the maintained suite and correct cache identity

- [x] **B-1** (High) — Repair the dataset-quality tests left on removed signatures → [§3](#b-1)
- [x] **K-1** (High) — Stop declaring a filter configuration the validator does not use → [§4](#k-1)
- [x] **K-4** (Medium) — Normalize artifact parameters to JSON-safe values → [§4](#k-4)
- [x] **R-3** (Medium) — Give each writer a private temporary artifact path → [§5](#r-3)

### Batch 2 — Make the key model reusable

- [x] **K-2** (High) — Key feature artifacts per feature, not per requested set → [§4](#k-2)
- [x] **K-3** (High) — Identify sources by content, not by modification time → [§4](#k-3)
- [x] **R-2** (Medium) — Prune artifacts whose source identity no longer resolves → [§5](#r-2)

### Batch 3 — Runtime cache safety and consumer coverage

- [x] **R-1** (Medium) — Stop handing out shared mutable arrays and mappings → [§5](#r-1)
- [x] **R-4** (Medium) — Bound what the in-process cache retains per source → [§5](#r-4)
- [x] **C-1** (Medium) — Correct the documented reach of the persisted cache → [§6](#c-1)
- [x] **C-2** (Low) — Give dataset-quality validation a cache bypass → [§6](#c-2)
- [x] **C-3** (Medium) — Protect the cross-tool key contract with a test → [§6](#c-3)
- [x] **C-4** (Medium) — Stop the cache unit tests from destroying the workspace cache → [§6](#c-4)

### Batch 4 — Execution speed

- [x] **P-1** (Medium) — Persist detector replay results across processes → [§7](#p-1)
- [x] **P-2** (Low) — Restore the shared cache in CI → [§7](#p-2)
- [x] **P-3** (Low) — Distribute tests so workers keep dataset locality → [§7](#p-3)

---

## 3. Broken Behavior

<a id="b-1"></a>

### B-1 — Dataset-quality tests still call removed signatures

**Where**

- [test_dataset_quality_validation.py:93](../../test/python/test_dataset_quality_validation.py)
- [test_dataset_quality_validation.py:100](../../test/python/test_dataset_quality_validation.py)
- [test_dataset_quality_validation.py:890](../../test/python/test_dataset_quality_validation.py)
- [test_dataset_quality_validation.py:1207](../../test/python/test_dataset_quality_validation.py)
- [test_dataset_quality_validation.py:1220](../../test/python/test_dataset_quality_validation.py)

**Finding**

`667e4dc` removed the `npz_cache` and `calibration_cache` parameters from `validate_empty_sanity`, `_compute_idle_evidence_for_entry`, and `_evaluate_pair_capture`, but left the tests on the old signatures. Three tests fail on `HEAD`:

```
TypeError: validate_empty_sanity() got an unexpected keyword argument 'npz_cache'
TypeError: <lambda>() missing 1 required positional argument: 'npz_cache'
```

Measured: `3 failed, 35 passed` for the module, `5 failed, 895 passed` for the whole suite. The refactor shipped without running the narrowest relevant check.

The two remaining failures in `test_low_rssi_classic.py` are unrelated; see [§8](#not-defects).

**Resolution**

Update the tests to the current signatures. Keep them asserting the same behavior contract, since the removed parameters were per-run memos, not part of the product surface.

---

## 4. Cache Identity And Key Design

<a id="k-1"></a>

### K-1 — The validation feature key declares filters the validator never applies

**Where**

- [validate_dataset_quality.py:659](../../tools/validate_dataset_quality.py)
- [validate_dataset_quality.py:649](../../tools/validate_dataset_quality.py)
- [train_ml_model.py:1160](../../tools/train_ml_model.py)
- [train_ml_model.py:1460](../../tools/train_ml_model.py)

**Finding**

`_validation_feature_cache_parameters` writes `enable_hampel: false`, `hampel_window: 0`, `hampel_threshold: 0.0`, and `lowpass_cutoff: 0.0` into the cache key. `_feature_matrix_packets` then calls `extract_features()` without passing any of them, so the computation inherits the `config.py` defaults `ENABLE_HAMPEL_FILTER = True`, `HAMPEL_WINDOW = 7`, `HAMPEL_THRESHOLD = 5.0`. The persisted key therefore describes a configuration that was never used.

Measured against the working cache: 42 source/feature-set combinations are stored under both `enable_hampel: true` (trainer) and `enable_hampel: false` (validator), and every pair is **bitwise identical**.

Two consequences:

- The trainer and the validator never share an entry for identical work. Every capture is extracted twice and stored twice.
- A future producer that genuinely disables Hampel filtering would collide with these mislabeled entries and silently read filtered features.

**Resolution**

Derive both keys from one shared helper, and make the validator pass the same filter configuration its key names. Normalize disabled filters so their inactive parameters cannot fragment the key.

<a id="k-2"></a>

### K-2 — Keying on the full feature list prevents subset reuse

**Where**

- [npz_cache.py:256](../../tools/lib/npz_cache.py)
- [train_ml_model.py:1197](../../tools/train_ml_model.py)
- [validate_dataset_quality.py:675](../../tools/validate_dataset_quality.py)

**Finding**

The artifact key contains the entire requested `feature_names` list, so a 12-feature matrix, a 5-feature matrix, and a 1-feature matrix for the same capture are three unrelated entries.

Feature columns do not depend on the requested set. Measured against the working cache: 84 of 84 subset-versus-superset column comparisons are bitwise identical. Every subset request therefore recomputes columns the cache already holds.

Current cost: 199 entries for 73 distinct sources, 60 MB. `extract_features` costs 1.22 s per capture for 5 features and 0.48 s for 1 feature, against about 0.01 s to load a cached matrix. Feature-selection and seed-search loops, which request many overlapping subsets, pay this repeatedly.

**Resolution**

Make the persisted unit one capture plus one feature, or one capture plus a feature superset that requests slice by column. Assemble the requested matrix from cached columns and compute only the missing ones.

<a id="k-3"></a>

### K-3 — Source identity uses modification time, which git does not preserve

**Where**: [npz_cache.py:76](../../tools/lib/npz_cache.py)

**Finding**

`source_manifest` identifies a capture by `size` and `mtime_ns`. `git checkout` rewrites modification times, so:

- A `.npz_cache` restored in CI would miss on every entry, making [P-2](#p-2) worthless as written.
- Local branch switches invalidate the cache and orphan its contents. The working cache already holds 7 stale entries from this.

The design is also sensitive to timestamp granularity, which is why [test_npz_cache.py:40](../../test/python/test_npz_cache.py) has to `sleep` between rewrites of the same fixture.

**Resolution**

Identify sources by content digest. The corpus is 163 MB across 100 captures, so a full SHA-256 costs well under a second in aggregate. Keep `size` and `mtime_ns` as a fast path that skips rehashing an unchanged file.

<a id="k-4"></a>

### K-4 — Artifact parameters are not normalized to JSON-safe values

**Where**

- [npz_cache.py:88](../../tools/lib/npz_cache.py)
- [npz_cache.py:203](../../tools/lib/npz_cache.py)

**Finding**

`artifact_manifest` copies caller parameters verbatim, and both `manifest_digest` and `save_npz_artifact` serialize them with `json.dumps`. The module already has `_json_safe`, but applies it only to the idle-baseline payload. Two failure modes, both reproduced:

- A NumPy scalar in `parameters` raises `TypeError: Object of type float32 is not JSON serializable` at digest time.
- A tuple in `parameters` digests successfully, but `load_npz_artifact` compares the freshly built manifest against the JSON round-trip of the stored one. The tuple returns as a list, the comparison fails, and the entry never hits. The cache silently rewrites the same artifact on every run.

Today's callers happen to pass JSON-safe values, so this is latent rather than active, but nothing enforces it and the tuple case fails silently.

**Resolution**

Apply `_json_safe` to parameters inside `artifact_manifest` so the in-memory manifest and its serialized form are equal by construction.

---

## 5. Runtime Cache Safety And Resource Use

<a id="r-1"></a>

### R-1 — Cached arrays and mappings are shared and mutable

**Where**

- [csi_io.py:207](../../tools/lib/csi_io.py)
- [csi_io.py:217](../../tools/lib/csi_io.py)
- [csi_io.py:2355](../../tools/lib/csi_io.py)

**Finding**

`load_npz_arrays` returns the same `dict` object on every call, holding arrays with `flags.writeable = True`. Reproduced: inserting a key into the returned mapping is visible to the next caller. `load_npz_sensing_arrays` shares the same CSI array object, so an in-place edit reaches both views and every packet view built from them.

`load_npz_as_packets` shallow-copies the packet dictionaries, which protects against per-packet key edits but not against CSI array mutation.

No current caller mutates them, so this is an unsafe contract rather than an active bug — but it is exactly the kind of contract that breaks quietly once a new analysis script edits an array in place.

**Resolution**

Mark cached arrays read-only and hand out a mapping the caller cannot mutate. Keep `load_npz_as_packets` returning writable copies for its existing callers.

<a id="r-2"></a>

### R-2 — Persisted artifacts are never pruned

**Where**: [npz_cache.py:190](../../tools/lib/npz_cache.py)

**Finding**

`clear_persisted_artifacts` removes whole artifact trees, which is all-or-nothing. Nothing removes entries whose source no longer resolves or no longer matches. The working cache is 129 MB and already contains 7 dead entries. Combined with [K-3](#k-3), a branch switch orphans an entire generation of artifacts.

**Resolution**

Add a prune operation that drops entries whose source manifest no longer resolves, and expose it on the tooling surface.

<a id="r-3"></a>

### R-3 — Concurrent writers share one temporary artifact path

**Where**: [npz_cache.py:250](../../tools/lib/npz_cache.py)

**Finding**

The staging path is `<digest>.npz.tmp.npz`, derived only from the digest. Two processes writing the same artifact — `pytest -n auto`, or a training run alongside dataset-quality validation — interleave writes into one file before `os.replace`. The published artifact can be a truncated archive.

`load_npz_artifact` swallows the resulting exception and reports a miss, so the cache self-heals on the next write, but the corruption window wastes the work it was meant to save.

**Resolution**

Make the temporary name unique per writer.

<a id="r-4"></a>

### R-4 — The in-process cache retains every derived view without bound

**Where**

- [npz_cache.py:153](../../tools/lib/npz_cache.py)
- [csi_io.py:207](../../tools/lib/csi_io.py)

**Finding**

`get_runtime_artifact` has no eviction, and nothing in the tools calls `clear_runtime_artifacts`. Loading a packet view pins the raw array mapping for the lifetime of the process, because the packet view is built from it and the builder result is memoized separately.

Measured on one 1.9 MB capture: the packet view alone retains about 15 MB of per-packet dictionaries, on top of the raw and sensing mappings that stay pinned. Extrapolated across the 163 MB corpus, a process that touches every capture retains well over 1 GB. The previous code memoized packets too, but did not retain the raw arrays behind them.

**Resolution**

Release the raw mapping once a derived artifact is materialized, or bound the runtime cache so the heaviest views can be evicted.

---

## 6. Consumer Coverage

<a id="c-1"></a>

### C-1 — Documentation overstates the reach of the persisted cache

**Where**

- [ML_TRAINING.md](../ML_TRAINING.md)
- [performance_report.py:435](../../tools/lib/performance_report.py)

**Finding**

`ML_TRAINING.md` says the `.npz_cache/` artifacts are "shared with dataset-quality validation and performance reporting". Performance reporting and the maintained pytest gates use only the in-process layer through `load_npz_packet_view` and `load_npz_sensing_arrays`; they persist nothing. The statement is true of the runtime cache and false of `.npz_cache/`.

**Resolution**

Describe the two layers separately and name which tools persist artifacts.

<a id="c-2"></a>

### C-2 — Dataset-quality validation has no cache bypass

**Where**: [validate_dataset_quality.py:675](../../tools/validate_dataset_quality.py)

**Finding**

`train_ml_model.py` exposes `--no-cache`, which suppresses both the read and the write. `validate_dataset_quality.py` persists feature matrices and idle baselines with no equivalent switch, so the only way to bypass a suspect artifact is to delete the whole cache tree. That is a poor position to be in while diagnosing exactly the class of defect recorded as [K-1](#k-1).

**Resolution**

Add a matching bypass flag that suppresses both the read and the write.

<a id="c-3"></a>

### C-3 — No test protects the cross-tool key contract

**Where**: [test_npz_cache.py](../../test/python/test_npz_cache.py)

**Finding**

The existing tests cover round-trip, source invalidation, and tree clearing. They do not cover the contract that actually broke: that the trainer and the validator agree on a key for identical work. They also do not cover parameter serialization ([K-4](#k-4)) or the corrupt-artifact fallback.

The gap matters because agreement between two tools is the whole point of centralizing the cache, and nothing currently fails when they diverge.

**Resolution**

Add a test asserting that the trainer and the validator produce the same artifact identity for the same capture and feature request, plus coverage for parameter normalization.

<a id="c-4"></a>

### C-4 — The cache unit tests destroy the workspace cache

**Where**

- [test_npz_cache.py:57](../../test/python/test_npz_cache.py)
- [test_npz_cache.py:91](../../test/python/test_npz_cache.py)
- [npz_cache.py:33](../../tools/lib/npz_cache.py)

**Finding**

Found while verifying [R-2](#r-2). `npz_cache_dir()` always resolves to the repo-local `.npz_cache`, and the tests take their source captures from `tmp_path` but write artifacts into that shared directory, then call `clear_persisted_artifacts("feature_matrix")`, which is an `rmtree` of the real tree.

Reproduced: running the maintained suite deletes the developer's working cache. This review's own measurements lost 239 entries to it. The next command that needs the cache silently pays a full cold rebuild, which is exactly the cost the cache exists to avoid.

**Resolution**

Let the cache root be redirected, and point the tests at an isolated directory. The same redirection is useful for placing the cache outside the workspace.

---

## 7. Execution Speed

<a id="p-1"></a>

### P-1 — Detector replay results are recomputed in every process

**Where**

- [performance_report.py:596](../../tools/lib/performance_report.py)
- [performance_report.py:760](../../tools/lib/performance_report.py)
- [performance_report.py:845](../../tools/lib/performance_report.py)
- [performance_report.py:865](../../tools/lib/performance_report.py)

**Finding**

`compute_classic_dataset_result`, `compute_ml_dataset_result`, and the empty-room false-positive helpers are memoized with `lru_cache`, so they survive only within one process. Every `generate_performance_report` run and every pytest worker repeats the full replay. Their keys are already explicit — `(paths, band, window, threshold)` — and their results are small dictionaries of floats, which makes them the best remaining persistence candidates.

Current cost: the maintained suite is 74.6 s wall but 226 s CPU.

**Resolution**

Persist these results through the shared cache, keyed on the source captures and the detector parameters.

<a id="p-2"></a>

### P-2 — CI never restores the shared cache

**Where**: [ci.yml](../../.github/workflows/ci.yml)

**Finding**

The workflow caches pip, PlatformIO, and the ESP-IDF toolchains, but not `.npz_cache`. The Python test job replays the real corpus from scratch on every run.

This is only worth doing after [K-3](#k-3): with modification-time identity, a restored cache misses on every entry because `actions/checkout` rewrites timestamps.

**Resolution**

Restore and save `.npz_cache` in the Python test job once source identity is content-based, keyed on the dataset catalog revision and the tool sources rather than on a hash of the capture files themselves.

<a id="p-3"></a>

### P-3 — Work stealing defeats per-worker dataset locality

**Where**

- [ci.yml](../../.github/workflows/ci.yml)
- [pytest.ini](../../pytest.ini)

**Finding**

CI runs `pytest test/python -n auto --dist worksteal`. The in-process cache is per worker, so tests over the same capture scatter across workers and each worker reloads and recomputes independently. A distribution mode that keeps a module's tests on one worker preserves the locality the runtime cache depends on.

**Resolution**

Evaluate `--dist loadfile` against the current wall-clock time and adopt it if it wins.

---

<a id="not-defects"></a>

## 8. Reviewed And Not Defects

- **Long-recording view rebuild.** `667e4dc` removed the `lru_cache` from `_load_long_test_packets_cached`. Measured cost of a repeat `load_long_test_dataset` call is about 0 ms: the arrays come from the runtime cache and the view construction is a no-op on already contiguous data. No regression.
- **`test_low_rssi_classic` failures.** Two failures on `HEAD` (`recall 66.3%` against an 85% floor) come from the weak-link bedroom captures added in `66459c6`, not from the cache work. That path uses only packet views and no persisted artifact. Tracked as detector work, out of scope here.
- **Uncompressed artifact storage.** `np.savez` without compression keeps the feature cache at 60 MB. Compression would trade CPU for disk on the exact path the cache exists to make fast. Left as is.

---

## 9. Current Branch Status

Verified against the current working tree on 2026-07-29.

Resolved on this branch: **B-1, K-1, K-2, K-3, K-4, R-1, R-2, R-3, R-4, C-1, C-2, C-3, C-4, P-1, P-2, and P-3**.

No open review tasks remain.

`P-3` was closed by measurement rather than by changing the scheduler: with `-n 4`, `pytest` completed in about `74.7 s` under `--dist worksteal` and about `84.4 s` under `--dist loadfile`, so the existing CI distribution stays in place.

Verification run for this review update:

- `.venv/bin/pytest test/python/test_npz_cache.py -v`
- `.venv/bin/pytest test/python/test_dataset_quality_validation.py -v`
- `.venv/bin/pytest test/python/test_npz_cache.py test/python/test_validation_real_data.py -v`
- `.venv/bin/pytest test/python/test_validation_real_data.py -v`
- `.venv/bin/pytest test/python -n 4 --dist worksteal -q --durations=10`
- `.venv/bin/pytest test/python -n 4 --dist loadfile -q --durations=10`
