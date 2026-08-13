# Test Suite Scope Review

Date: 2026-08-13
Review status: Complete
Remediation status: Complete

## 1. Objective

Review every current C++ and Python test, identify tests that do not protect a current contract, and define a smaller maintained scope centered on production code, the ESPectre CLI, ML training, GitHub CI and release behavior, and the NPZ cache.

The current source inventory contains 29 C++ suites with 322 registered `RUN_TEST` cases and 37 Python test modules with 788 test functions. Pytest expands the Python suite to 1,110 collected cases through parametrization. The three JavaScript web test files are outside this review.

This review distinguishes finding resolution from review completion. The review itself is complete; no finding has been implemented yet.

## 2. Retention Policy

The maintained suite should protect current contracts, safety or correctness invariants, quantified detector performance, C++/Python parity, supported compatibility boundaries, and observable side effects. It should not preserve removed options, retired implementations, historical negative facts, or branches solely to increase coverage.

Retain these surfaces by default:

- First-party C++ production code under `src/cpp/`, including core, runtime, ESP-IDF boundaries, SDK, and frontends.
- MicroPython production code under `src/python/micro_espectre/`.
- The supported host CLI under `src/python/espectre_cli/`, including CSI collection and its post-collection integrity and continuity checks.
- The primary ML training workflow in `tools/train_ml_model.py` and its direct libraries.
- GitHub CI, release, supply-chain, license-compliance, and generated-artifact contracts.
- NPZ cache identity, invalidation, concurrency, pruning, source provenance, and reuse contracts.
- Quantified real-data detector gates and C++/Python parity gates.

Research benchmarks and secondary analysis tools should normally use their owning end-to-end workflow. A unit test is justified only when the code is also a direct dependency of a retained production, CLI, training, CI, or cache path.

## 3. Findings

### TEST-001 — High — C++ coverage-only tests pass unconditionally

Locations: [test_high_accuracy_detector.cpp](../../test/cpp/suites/core/test_high_accuracy_detector.cpp) `test_ml_inference_performance` at line 431; [test_frontend_controls.cpp](../../test/cpp/suites/frontend/test_frontend_controls.cpp) `test_runtime_fault_callback_handles_null_and_message_paths` at line 339 and `test_dump_config_covers_configuration_branches` at line 349; [test_sensor_publisher.cpp](../../test/cpp/suites/frontend/test_sensor_publisher.cpp) `test_sensor_publisher_log_status_handles_runtime_snapshot` at line 64 and `test_sensor_publisher_log_status_ignores_null_tag` at line 79.

These tests end with `TEST_PASS()` or `TEST_ASSERT_TRUE(true)` after exercising code. They cannot detect an incorrect result. The host inference loop also does not measure ESP32 or MicroPython performance.

Resolution: remove the inference loop test. Remove the logging tests unless the ESP logging mock is extended to expose an observable message, rate-limit state, or other documented side effect. If log output is made observable, assert stable semantics rather than incidental formatting.

### TEST-002 — High — Several C++ real-data assertions are tautological or materially weaker than existing gates

Locations: [test_utils.cpp](../../test/cpp/suites/core/test_utils.cpp) `test_magnitude_from_real_csi` at line 154, `test_variance_static_presence_vs_motion` at line 176, `test_turbulence_static_presence_vs_motion` at line 204, and `test_turbulence_from_csi_different_csi_lengths` at line 245; [test_hampel_filter.cpp](../../test/cpp/suites/core/test_hampel_filter.cpp) `test_hampel_with_real_motion_turbulence` at line 319 and `test_hampel_preserves_motion_signal` at line 373.

The utility tests mostly assert that magnitudes, variances, or turbulence are non-negative. The CSI-length test exercises only the normal 128-byte input. The two Hampel tests only require positive output and do not prove that motion information is preserved.

Resolution: remove the four weak utility replays and the two weak Hampel replays. Retain exact synthetic numerical tests, the injected-outlier real-data test, variance-separation tests, detector performance gates, and packet-format boundary tests.

### TEST-003 — Medium — Removed CLI options are preserved as tombstones

Locations: [test_espectre_cli_collect.py](../../test/python/test_espectre_cli_collect.py) `test_collect_parser_rejects_removed_samples_option` at line 255; [test_npz_cache.py](../../test/python/test_npz_cache.py) `test_prune_tool_rejects_removed_retention_options` at line 640.

Both tests verify that `argparse` rejects a historical option that is no longer registered. This behavior is already generic to every unknown option and does not define a supported compatibility boundary.

Resolution: remove both cases. Preserve every other NPZ cache test, especially artifact-version pruning and removal of currently recognized retired artifact classes, because those exercise the current pruning command rather than the absence of a former option.

### TEST-004 — High — Threshold regression tests test another test module

Location: [test_threshold_path_regressions.py](../../test/python/test_threshold_path_regressions.py), both tests at lines 30 and 76.

The module dynamically imports `test_validation_real_data.py`, replaces its helpers with fakes, and verifies the internal implementation of test-only calibration helpers. It does not protect production, CLI, training, or shared tool code.

Resolution: remove the module. If the calibration orchestration is a reusable contract, move it to the narrowest production or `tools/lib/` owner and add tests against that owner.

### TEST-005 — High — Retired host candidate behavior is kept executable by tombstone tests

Locations: [test_host_candidate_restoration.py](../../test/python/test_host_candidate_restoration.py) `test_restored_candidates_stay_out_of_the_runtime_surface` at line 54, `test_retired_full_band_spread_remains_evaluable_host_side` at line 117, `test_retired_l1_autocorrelation_remains_host_evaluable` at line 137, and `test_linear_aggregated_drift_is_removed` at line 152.

The suite maintains an explicit historical `RESTORED_CANDIDATES` list, requires retired features to remain evaluable, and requires one still-registered candidate to return a constant zero. A constant-zero feature should not remain in a training candidate registry. Historical evaluation evidence belongs in `docs/FEATURES.md`, not in the maintained executable contract.

Resolution: remove retired feature implementations and their tests after confirming that their evidence is recorded in `docs/FEATURES.md`. Replace the first case with generic current invariants: host-only candidates are disjoint from runtime features, unsupported candidates cannot be exported to C++, and production features resolve to valid C++ extractor IDs. Retain current candidate formula tests only while the corresponding `train_ml_model.py` mode remains supported.

### TEST-006 — Medium — ML inference coverage contains a host benchmark, duplicated checks, and a historical negative assertion

Locations: [test_ml_inference.py](../../test/python/test_ml_inference.py) the `WEIGHTS` absence assertion at line 45, `test_inference_matches_reference` at line 84, `test_inference_speed` at line 151, and the three `TestMLDetectorIntegration` cases beginning at line 183.

The CPython speed test is not a device performance gate. The first-100-sample reference test is a subset of `test_all_samples_match`. Import, initialization, and threshold-bound checks duplicate `test_high_accuracy_detector.py`. Requiring the former `WEIGHTS` symbol to be absent preserves an implementation migration rather than the current export contract.

Resolution: retain the positive `WEIGHTS_T` layout assertion, array saturation contract, all-sample exported-reference comparison, and output-range check. Remove the speed test, the first-100-sample duplicate, the three integration duplicates, and the negative `WEIGHTS` assertion. Device performance should be measured by the owning firmware benchmark workflow.

### TEST-007 — Medium — Filter and segmentation suites contain weak assertions and duplicate helper tests

Locations: [test_filters.py](../../test/python/test_filters.py) `test_outlier_replacement` at line 55 and `TestHampelFilterInsertionSort` at line 194; [test_segmentation_additional.py](../../test/python/test_segmentation_additional.py) initialization cases at lines 25 and 32, single-subcarrier case at line 109, and partial-buffer case at line 120.

The outlier assertion accepts an unchanged outlier. The insertion-sort class duplicates coverage in `test_utils.py`. The listed segmentation cases duplicate `test_segmentation.py`.

Resolution: rewrite the outlier case with a non-zero MAD and an exact expected replacement. Remove the duplicate insertion-sort and segmentation cases. Retain filter-reset coverage and exception-fallback coverage, but assert the exact raw fallback value and an observable error indication.

### TEST-008 — Medium — Some real-data tests can pass without proving detector or filter quality

Locations: [test_validation_real_data.py](../../test/python/test_validation_real_data.py) `test_hampel_reduces_spikes` at line 402; [test_validation_long_recordings.py](../../test/python/test_validation_long_recordings.py) `test_long_recording_replays` at line 122.

The Hampel test conditionally skips its only assertion when the selected data has no sufficiently large spike, and its `filtered_max <= raw_max` condition accepts an identity filter. The long-recording replay performs expensive work but primarily validates that computed percentages fall between zero and one hundred.

Resolution: remove the conditional Hampel test and retain the meaningful variance-separation test. Preserve long-recording replay and cache-versus-packet parity, but either add explicit documented long-recording gates or make the replay an explicit end-to-end report workflow instead of presenting range checks as performance validation.

### TEST-009 — Medium — Dataset-quality unit coverage exceeds the retained CLI surface

Location: [test_dataset_quality_validation.py](../../test/python/test_dataset_quality_validation.py), 49 test functions across the module.

The CLI directly uses `validate_file_integrity`, `validate_signal_quality`, and `validate_capture_continuity` after collection. Tests for safe NPZ loading, CSI shape, HT20 selection, packet rate, sequence gaps, inter-packet gaps, and low-RSSI continuity therefore protect a retained CLI path. Much of the remaining module tests standalone report wording, empirical review profiles, metadata refresh, ML readiness reporting, and other research-tool internals.

Resolution: retain focused unit tests for the three CLI-used validation functions. Replace standalone report and metadata orchestration unit coverage with a small owning end-to-end workflow, including `tools/validate_dataset_quality.py --check-current`. Do not remove the post-collection quality behavior from the CLI.

### TEST-010 — Low — Two secondary host-tool suites fall outside the selected maintained scope

Locations: [test_benchmark_classic_candidate_pairs.py](../../test/python/test_benchmark_classic_candidate_pairs.py), 22 test functions; [test_espectre_traffic_generator_tool.py](../../test/python/test_espectre_traffic_generator_tool.py), two test functions.

The first suite protects a research benchmark rather than the primary training workflow. The second protects a secondary host process wrapper and is distinct from `test_traffic_generator.py`, which tests the MicroPython production runtime.

Resolution: remove these unit suites if the selected scope remains production, CLI, primary training, CI, and cache. Validate the tools through their documented end-to-end commands while they remain in the repository.

### TEST-011 — Low — Interactive dataset-selection tests belong to secondary analysis tools

Location: [test_dataset_metadata_resolution.py](../../test/python/test_dataset_metadata_resolution.py), interactive selection and plot-window cases beginning at lines 256, 339, and 388.

Dataset-role normalization, report revision identity, pair resolution, and cache-relevant metadata should remain tested because they feed training, CSI I/O, generated-report checks, and cache invalidation. The interactive menu and Matplotlib cancellation paths are used by secondary analysis tools.

Resolution: retain metadata and pair-resolution contracts, and remove the three interactive UI tests under the selected scope.

### TEST-012 — Low — One CI assertion appears tied to a retired implementation name

Location: [test_ci_pipeline.py](../../test/python/test_ci_pipeline.py) `test_workflows_keep_publication_and_supply_chain_guardrails` at line 325.

The suite is in scope and should remain. SHA-pinned actions, explicit timeouts, fixed runner versions, ancestry validation, safe snapshot updates, reproducible archives, release validation, and the prohibition on destructive release deletion protect current CI and supply-chain contracts. The isolated assertion that `detect-push-origin` is absent appears to preserve the name of a former implementation rather than a semantic invariant.

Resolution: remove only that assertion unless a current failure mode is documented. Keep the remaining CI, release, website-publication, and license-compliance tests.

### TEST-013 — Low — C++ performance suites are metric producers, not independent target gates

Locations: [test_motion_detection.cpp](../../test/cpp/suites/integration/test_motion_detection.cpp) metric validation and target output around lines 180 and 340; [test_long_recordings.cpp](../../test/cpp/suites/integration/test_long_recordings.cpp) metric validation around line 197.

These suites mostly validate metric structure and ranges, while their structured output is consumed by `tools/generate_performance_report.py` for C++/Python parity. The actual numerical promotion gates live in `test_validation_real_data.py`. Removing the C++ suites would break parity and generated-report validation even though their local assertions look weak.

Resolution: preserve both suites. Clarify in names or documentation that they are parity metric producers, and keep the Python aggregate tests as the numerical detector gates.

### TEST-014 — High — Duplicated ownership gives small production changes an excessive test-change blast radius

Locations: suite-wide, with representative overlap across [test_csi_features.py](../../test/python/test_csi_features.py), [test_high_accuracy_detector.py](../../test/python/test_high_accuracy_detector.py), [test_ml_inference.py](../../test/python/test_ml_inference.py), [test_train_ml_model_augmentation.py](../../test/python/test_train_ml_model_augmentation.py), [test_validation_real_data.py](../../test/python/test_validation_real_data.py), [test_core_helpers.cpp](../../test/cpp/suites/core/test_core_helpers.cpp), and [test_high_accuracy_detector.cpp](../../test/cpp/suites/core/test_high_accuracy_detector.cpp).

The main cause of broad test diffs is not the number of test files by itself. Multiple suites frequently restate the same feature registry, defaults, layout, implementation detail, or historical migration fact. A correct production change then requires synchronized edits across unit, integration, training, and validation files even when only one public contract changed. Additional `*_additional.py`, `*_regressions.py`, and one-bug test modules also create new owners instead of extending the existing owner.

Reducing the suite to a few large files would hide this coupling and increase merge conflicts. The target structure is one coherent owner per contract, with distinct files retained where unit, integration, parity, and frontend responsibilities are genuinely different.

Resolution:

- Define a source-to-test ownership map in `test/cpp/README.md` for C++ and in a new concise testing section of the existing repository documentation owner for Python. Each production module or coherent subsystem should name its primary unit-test owner and any separate integration or parity gate.
- Extend the existing owner when adding a regression. Do not create a new regression test module when an owning suite exists.
- Consolidate `test_segmentation_additional.py` into `test_segmentation.py`, remove `test_threshold_path_regressions.py`, and fold useful ML inference cases into their actual runtime or training owner where TEST-006 identifies duplication.
- Keep canonical feature registries, schemas, defaults, and performance targets in production or shared configuration. One designated invariant test may assert the exact public schema; other tests should derive their cases from the canonical source instead of copying the full list.
- Share test inputs, packet builders, fixtures, and comparison helpers, but keep behavioral assertions in the owning suite. Do not move entire behavioral tests into generic fixtures or `conftest.py`.
- Prefer public state, return values, events, persistence, payloads, and side effects over private fields. Retain private-layout assertions only for explicit allocation, memory, timing, reset, safety, or compatibility invariants.
- Keep unit, integration, and quantified gate roles separate. A detector implementation change may legitimately update Python production, its Python owner, C++ production, and its C++ owner; it should normally run parity and performance gates without editing their expectations.
- Add an agent rule requiring an explicit explanation before modifying more than three test files for one logical production change. The explanation must identify the distinct contracts that require each file; shared implementation churn is not sufficient justification.
- Treat coverage as diagnostic evidence rather than a reason to add execution-only tests. New or changed tests must fail for a plausible contract regression.

Suggested `AGENTS.md` policy text:

> Extend the existing test owner for the changed contract. Do not create a new regression test module when an owning suite exists. Do not duplicate production constants, feature registries, schemas, or performance targets in tests. Prefer parametrizing the owner suite. A production change should not require modifying integration or performance gate code unless the public contract or gate itself deliberately changes. Before editing more than three test files for one logical production change, explain which distinct contracts require those edits.

> Test failures caused only by an internal refactor should be resolved by testing the public result, unless the internal property is an explicit memory, timing, reset, safety, or compatibility invariant.

## 4. Progress Checklist

- [x] `TEST-001` Remove or make the five C++ coverage-only cases observable.
- [x] `TEST-002` Remove the six tautological or weak C++ real-data cases.
- [x] `TEST-003` Remove the two removed-option tombstone tests while preserving current NPZ pruning coverage.
- [x] `TEST-004` Remove `test_threshold_path_regressions.py`, or move its reusable behavior to a real owner first.
- [x] `TEST-005` Remove retired host candidates and replace historical inventory assertions with generic current training/runtime boundaries.
- [x] `TEST-006` Consolidate `test_ml_inference.py` around export layout, saturation, reference parity, and output range.
- [x] `TEST-007` Remove filter and segmentation duplicates, and strengthen the exception and outlier assertions.
- [x] `TEST-008` Remove the conditional Hampel replay and decide whether long-recording validation is a quantified gate or an end-to-end report workflow.
- [x] `TEST-009` Split dataset-quality coverage between retained CLI primitives and the validator's end-to-end `--check-current` workflow.
- [x] `TEST-010` Remove unit suites for the two secondary host tools if the maintained scope remains unchanged.
- [x] `TEST-011` Remove interactive dataset-selection and plot-window tests while retaining metadata, pair-resolution, report-revision, and cache contracts.
- [x] `TEST-012` Remove the historical CI implementation-name assertion after confirming that it has no current threat model.
- [x] `TEST-013` Preserve and clarify the C++ parity metric-producer suites.
- [x] `TEST-014` Document one primary test owner and any separate integration or parity gates for each maintained production, CLI, training, CI, and cache subsystem.
- [x] `TEST-014` Consolidate supplemental and regression-only modules into their existing owners without creating monolithic cross-subsystem test files.
- [x] `TEST-014` Reduce copied feature registries, defaults, schemas, and performance targets to one canonical source plus one designated exact-schema invariant where required.
- [x] `TEST-014` Add the test-ownership and three-test-file explanation rules to `AGENTS.md`.
- [x] Inspect the final diff and confirm that GitHub CI, license compliance, and NPZ cache behavior were not weakened.
- [x] Run `.venv/bin/pytest test/python/test_ci_pipeline.py test/python/test_license_compliance.py test/python/test_npz_cache.py -v`.
- [x] Run the narrow Python suites affected by each implementation batch.
- [x] Run `.venv/bin/pytest test/python -v`.
- [x] Run `cmake -S test/cpp -B test/cpp/build` and `cmake --build test/cpp/build`.
- [x] Run `ctest --test-dir test/cpp/build --output-on-failure`.
- [x] Run `ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure`.
- [x] Run `.venv/bin/pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v`.

## 5. Review Validation

The initial inventory was validated with `.venv/bin/pytest test/python --collect-only -q -n 0`, which collected 1,110 Python cases. After remediation, the same command collects 991 cases.

The focused CI, license, and NPZ cache command passed 52 tests. The numerical Python gate passed 82 tests. The full Python run passed 986 tests and skipped two; its two UDP tests were rerun outside the network sandbox and passed. Its license-header case was rerun against a temporary index representing the pending file deletions and passed, because `git ls-files` intentionally continues to list tracked deletions in an unstaged working tree.

The C++ suite configured and built successfully. Twenty-seven tests passed in the sandbox; the two UDP suites failed only because local socket binding was denied, then both passed outside the network sandbox. The explicit `test_motion_detection` parity producer also passed.

The dataset-quality end-to-end workflow completed with 1,252 passes, 18 warnings, and no failures, regenerated `DATASET_QUALITY_CHECK.md`, and passed `tools/validate_dataset_quality.py --check-current`.
