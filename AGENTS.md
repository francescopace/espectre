# ESPectre Agent Rules

## Scope And Workflow

- Prototype sensing and detector changes in Python, validate them, then port
production behavior to the relevant shared `C++` layers and frontends.
- Treat `src/cpp/` as production firmware, `src/python/micro_espectre/` as the
MicroPython device path, and `src/python/espectre_cli/` plus `tools/` as
host-side code.
- Write code, comments, docs, filenames, and commit messages in English.
- Use the Oxford comma in English lists across project documentation and user-facing text.
- Keep changes surgical, prefer direct implementations, and match neighboring style.
- Update the existing topic owner by default. Create a new Markdown document
only when the task requests a persistent record or repository policy requires
a new ADR or review record.



## Source Of Truth

- Start from the narrowest source that directly owns the topic instead of
scanning every project document.
- Use `README.md` for project overview, quick start, and the documentation map.
- Use `docs/CLI.md`, `src/python/micro_espectre/README.md`, the relevant
frontend `README.md`, and `./espectre --help` for current CLI syntax and
operator workflows.
- Use `docs/SETUP.md`, `docs/ESPECTRE_PROTOCOL.md`, and `docs/ARCHITECTURE.md`
for shared configuration, protocol, and runtime architecture.
- Use `docs/ALGORITHMS.md`, `docs/FEATURES.md`, `docs/ML_DATA_COLLECTION.md`,
and `docs/ML_TRAINING.md` for detector behavior, feature inventory, dataset
collection, and training workflows.
- Use `docs/performance/README.md`, `docs/LITERATURE.md`, and `data/auto_generated/DATASET_QUALITY_CHECK.md`
for benchmark status, external research context, collection backlog, and
dataset quality snapshots.
- Use `docs/ROADMAP.md` for product direction, `docs/adr/*.md` for durable
architectural decisions, and `docs/review/*.md` only for dated review
context rather than current behavior.



## Protocol Rules

- Preserve **one message model, multiple transports**: MQTT, Direct HTTP, and
future transports carry the same canonical JSON contract and application
version; transport framing and delivery policy stay outside that model.
- Treat `docs/ESPECTRE_PROTOCOL.md` and the canonical protocol registry or
schema as the owners of messages, operations, discovery metadata, and version
semantics. Do not create transport-specific envelopes, aliases, or constants.
- Require cross-transport parity for serialized messages, validation, and
capability schemas; engine-level semantic parity alone is insufficient.



## Environment And Commands

- Use the repository virtual environment for direct Python commands when available.
- The `./espectre` wrapper auto-runs through `.venv` when it exists.
- ESP-IDF frontend builds require the ESP-IDF environment that provides `idf.py`.
- Prefer the repository `./espectre` wrapper for local workflows.
- Do not mutate GitHub state, including commenting, closing, merging, labeling,
or pushing, unless the user explicitly requests it.



## Definition Of Done

- Inspect the final diff for unrelated changes.
- Run the narrowest relevant checks first, followed by every required parity,
integration, or generated-artifact gate for the changed surface.
- Do not claim that a check passed unless it ran successfully.
- Report checks not run, including the exact command and blocker.
- Update the owning documentation when public behavior, configuration,
protocol, or operator workflow changes.



## Testing And Validation

- When tests fail, investigate the root cause and fix the implementation.
Never skip, disable, or weaken tests to make them pass; ask before changing a
supported behavior expectation.
- Maintained automated tests cover first-party `C++` production code, the
Micro-ESPectre runtime, the host CLI, directly imported runtime or CLI
modules, and quantified detector performance and `C++`/Python parity gates.
- Validate standalone research tools, one-off scripts, generated reports, build
configuration, example configuration, and CI plumbing through their owning
end-to-end workflows rather than new unit tests.
- Every maintained test must protect a current contract, a safety or correctness
invariant, or a quantified performance or parity gate.
- Do not add tombstone tests for removed behavior. Negative tests are appropriate
only at supported runtime, protocol, persistence, security, or compatibility
boundaries.
- Prefer assertions on state, return values, events, and side effects. Assert
output text only when it is a documented user-facing or machine-consumed
interface, and test stable semantics rather than incidental wording.
- Do not assert the presence, absence, or exact value of marketing copy,
headlines, captions, button labels, placeholders, helper text, or other
reader-facing website wording. This applies equally to snapshots, substring
checks, regular expressions, and raw HTML or JavaScript source scans, including
negative assertions. Website and HTML tests must assert stable structure or
behavior through selectors, attributes, routes, IDs, protocol values, and
documented machine-consumed strings such as emails, CLI commands, and option
values. Copy may change without a product regression.
- Keep the Python and `C++` coverage uploads and gates active.
- Run tests that bind local UDP sockets outside the network sandbox. Treat
`PermissionError` or `EPERM` during socket setup as a sandbox restriction,
not evidence that the test should use a different address.

- Extend the existing test owner for the changed contract. Do not create a new regression test module when an owning suite exists.
- Do not duplicate production constants, feature registries, schemas, or performance targets in tests. Prefer parametrizing the owner suite from the canonical source.
- A production change should not require editing integration or performance gate code unless the public contract or gate deliberately changes.
- Before editing more than three test files for one logical production change, explain which distinct contracts require those edits. Shared implementation churn is not sufficient justification.
- Test public results after internal refactors unless the internal property is an explicit memory, timing, reset, safety, or compatibility invariant.

After changing detection or calibration logic, run both parity validations:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
.venv/bin/pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v
```

Keep `C++` and Python algorithm trends aligned. If either validation cannot run,
report the exact command and blocker; see `docs/performance/README.md`.

## Python Rules

- Distinguish device/runtime code from host-side tooling.
- Device-oriented modules under `src/python/micro_espectre/` should stay MicroPython-friendly.
- Host-side code under `src/python/espectre_cli/`, `tools/`, and `test/python/` may use established CPython-only libraries.
- Use `src/python/micro_espectre/config.py` as the source of truth for shared
MicroPython runtime constants.
- Heavy libraries such as `numpy` and `pandas` belong in host-side analysis, training, and validation tools only.
- Use type hints where supported and useful.
- Do not use `asyncio` in MicroPython device code.
- Do not hardcode WiFi/MQTT credentials; use `config_local.py`.

Python test baseline:

```bash
.venv/bin/pytest test/python -v
```



## `C++` Architecture And Placement

- Use ESP-IDF for firmware code, not Arduino.
- `C++`17 features are available.
- Keep shared `core` and `runtime` code frontend-agnostic.
- Enforce the dependency direction `Frontend -> Runtime -> Core`; lower layers
must not include, query, or call higher layers.
- Follow ESPHome component conventions only inside `src/cpp/frontend/esphome/`.
- Use `ESP_LOGD`, `ESP_LOGI`, `ESP_LOGW`, `ESP_LOGE` in ESP-IDF firmware code.
- Do not edit `ml_weights.h` manually; it is generated by the training script.
- Do not add blocking code in firmware `loop()` paths or callbacks.
- Do not assume ESPHome-specific patterns apply to BLE, Matter, or shared runtime code.
- Use `snake_case` basenames; header and implementation of the same unit share the basename.
- Name the file after its primary class (`CsiCaptureService` → `csi_capture_service.h`).
- Boundary interfaces live in the shared layer as `<name>.h`; implementations as `<name>_<variant>` in the owning layer (`mqtt_transport_esp_idf`, `ble_bindings_nimble`).
- Suffixes: `_service` (start/stop lifecycle), `_transport` (data transport), `_bindings` (mockable boundary to an external stack), `_frontend` (runtime listener adapter), `_helpers` (free functions, domain-prefixed). Do not introduce new `_manager` files.
- Placement: algorithms and CSI format in `core/`; platform-agnostic contracts in `runtime/`; anything including ESP-IDF/FreeRTOS/lwIP in `runtime/esp_idf/`; single-frontend code in `frontend/<name>/`.
- Placement exception: portable shims that guard SDK includes behind `ESP_PLATFORM` or `__has_include` and degrade cleanly on host builds may live in `runtime/` (`pending_event`, `runtime_time`), or in `core/` when `core` itself depends on them (`espectre_log`).
- Headers in `core/` and `runtime/` must not include headers from `runtime/esp_idf/`.
- Generic basenames (`utils`, `helpers`, `common`) require a domain prefix or genuinely cross-cutting, homogeneous content.
- Core files with a Python counterpart keep the same basename (`threshold`, `csi_features`, `ml_weights`).



## Embeddable SDK Surface

- Treat everything reachable from `src/cpp/espectre_sdk.h` as the published SDK
surface. Adding a public type there means updating the facade include, the
`src/cpp/Doxyfile` INPUT list, and the header map in `docs/EMBEDDING.md` in the
same change. `test/python/test_sdk_surface_invariants.py` enforces this.
- Forward declarations are fine inside the surface, but the definition must
still arrive through the facade. A type an integrator can name in a signature
and cannot construct is a broken surface, not a decoupling win.
- Adding or changing a member of `IEspectreRuntime`, `IRuntimeListener`, or a
boundary interface breaks every external implementer. Give new members a
default implementation, or take the break deliberately and record it in the
active changelog section.
- Document accessors that degrade silently. A getter that returns empty or
zeroed data because a build-time option is off, such as a Kconfig choice, must
say so in its own comment; an integrator cannot see the `#if` from the header.
Prefer removing the gate over documenting it when the data is cheap to collect.
- Document a member fully or with a brief alone. `WARN_NO_PARAMDOC` is an error
in `src/cpp/Doxyfile`, so a half-filled `@param` list fails CI: partial
documentation reads as complete and is worse than none.

After changing the published surface, run both gates:

```bash
.venv/bin/pytest test/python/test_sdk_surface_invariants.py -v
python3 .github/scripts/generate_sdk_api.py
```



## `C++` Review Rules

- Review first-party `C++` for correctness, ownership, performance, duplication,
and frontend consistency; exclude generated, build, and vendored trees unless
requested.
- Put homogeneous shared behavior in the lowest layer that owns it, keep
orchestration out of `core`, and model intentional frontend differences as
explicit capabilities.
- Compare ESPHome, Native, and Matter defaults, validation, lifecycle,
events, reset behavior, error handling, and capabilities where applicable.
- Inspect CSI callbacks, runtime loops, and inference paths for blocking work,
allocation, copying, per-element division or modulo, repeated I/O, excessive
stack use, oversized buffers, and debug-only work active in release paths.
- Check for stale state, incomplete reset, incorrect readiness gates, missing
bounds or clamps, null-handling divergence, lifecycle ordering, and recovery gaps.
- Test shared behavior at the shared layer. Keep frontend-specific tests focused
on adapter behavior and declared capability differences.
- When the user requests a persistent review record, store it under
`docs/review/` with stable finding IDs, severity, precise locations, and one
progress checklist. Treat review completion and finding resolution as
separate states.



## Documentation Rules

- Use clear, concise, technical English, and prefer bullets or tables when they
improve readability. Use a neutral tone except in product-facing entry points.
- Do not hard-wrap prose in Markdown files; keep each paragraph and list item on a single source line unless Markdown syntax requires a line break.
- Keep frontend-specific workflows, protocols, and firmware surfaces in the local frontend README files.
- Verify current-state docs against implementation, runtime schema, and
generated artifacts; distinguish deployed, partial, and target behavior.
- Make public compatibility, controller-support, privacy, and security claims
only from repository evidence; use a validation matrix for incomplete coverage.
- Keep one source of truth per topic. Secondary documents should summarize and
link to it rather than repeat mutable formulas, metrics, commands, or corpus data.
- Keep `docs/ROADMAP.md` at the level of outcomes, gates, and sequencing. Put
experiment and collection detail in its owning document.
- Use ADRs for durable architectural or project-level decisions, including
important rejected directions. Keep one coherent decision per ADR, preserve
rationale, and update changed decision metadata consistently:
`Status`, `Supersedes`, and `Superseded by`. Do not leave superseded
decisions marked `Accepted`.
- Treat `docs/FEATURES.md` as the feature experiment ledger. Record every
seriously evaluated production, research, historical, planned, or rejected
feature before removal or moving on. For measured features retain the
definition, physical interpretation, scale invariance, implementation scope,
corpus/split/seed, primary and worst-group metrics, redundancy evidence,
verdict, and reason; mark unavailable evidence rather than reconstructing it.
- Keep host-only feature candidates under `tools/`, evaluate them with
`--no-export`, and do not add runtime extractors until a promotion decision
justifies Python/`C++` parity.
- Treat `docs/LITERATURE.md` as the external research ledger. Record the source
URL, release date, hardware and signal assumptions, methods, results, and
ESPectre transfer limits; exclude internal ESPectre research.
- Keep the active unreleased changelog focused on the final cumulative release
state. Put superseded experiments in `docs/FEATURES.md` or ADRs, and update
only the latest active section unless correcting an explicitly requested fact.
- Do not edit generated performance or dataset-quality reports manually.
Regenerate them from the current corpus, and run each generator's
`--check-current` mode before describing the reports as current.
- Use simple descriptive titles, filename-only text for internal file links,
and rare, purposeful emoji. Established entry points may retain branding.
- Update `docs/web/sitemap.xml` when public routes change. Do not edit generated
pages under `docs/web/guides/`, `docs/web/docs/`, `docs/web/media/`, or
`docs/web/roadmap/`; edit the shared fragments under `docs/web/content/`,
then run `.github/scripts/build_static_pages.py` to preview.



## GitHub And CI Rules

- Keep workflow changes minimal and explicit. Use `develop` as the default PR
target; `main` is release-only.
- Do not bypass branch protections, push directly to `main`, force push, or
merge with failing required checks.
- Prefer pinned major versions already used in the workflow or elsewhere in the repo.
- Before changing action versions, check current usage with `rg "uses: .*@" .github`.
- Keep action updates intentional, minimal, and grouped by purpose.
- Reply directly to GitHub issues/PRs only when explicitly requested.



## Commit Message Style

- Use Conventional Commits: `type(scope): subject` or `type: subject`.
- Preferred types are `feat`, `fix`, `refactor`, `docs`, `test`, `chore`,
`deps`, and `perf`.
- Use an imperative, concrete, lower-case subject with no trailing period;
target 72 characters or fewer.
- Use a scope only when it adds clarity. Prefer `deps(ci): ...` for workflow
dependency updates.
- Keep issue IDs out of the subject. Use `Fixes #93`, `Closes #93`, or
`Refs #93` in the footer when needed.
- Every commit intended for contribution must include a valid `Signed-off-by`
trailer; prefer `git commit -s`.



## License And Dependencies

- Dual-licensed: GPLv3 (`LICENSE`) plus separately offered commercial licenses (`LICENSING.md`).
- Keep `core`, `runtime`, and the Native and Matter frontends free of GPL-only dependencies; use permissively licensed components (Apache-2.0, MIT, BSD) so the code stays distributable under both tracks.
- Only `src/cpp/frontend/esphome/` may depend on GPLv3 ESPHome components.
- Do not add dependencies with licenses incompatible with GPLv3.
- Contributions require a one-time CLA signature; do not remove the CLA workflow.
- Add Python dependencies only when they are declared in the appropriate requirements file:
  - `requirements.txt` for the base workflow
  - `requirements-ml.txt` for ML/training extras



## Hard Constraints

- Do not modify CSI data format without updating both `C++` and Python implementations.
- Do not modify shared detection/calibration algorithms without keeping `C++` and Python aligned.
