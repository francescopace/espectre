# ESPectre Agent Rules

## Project Workflow

- Prototype sensing and detector changes in Python, validate them, then port
  production behavior to the relevant shared C++ layers and frontends.
- Treat `src/cpp/` as production firmware, `src/python/micro_espectre/` as the
  MicroPython device path, and `src/python/espectre_cli/` plus `tools/` as
  host-side code.

## Communication And Style

- Write code, comments, docs, filenames, and commit messages in English.
- Use the Oxford comma in English lists across project documentation and user-facing text.
- Keep changes surgical and aligned with existing repo patterns.
- Prefer simple, direct implementations over speculative abstractions.
- Match neighboring file style, including headers, naming, and formatting.
- Use `rg` for searching.
- Do not create new `.md` files unless explicitly requested.

## Source Of Truth

- `README.md`: project overview, quick start, documentation map, and public-facing context
- `docs/SETUP.md`: shared configuration parameters, defaults, ranges, and frontend chooser
- `docs/CLI.md`: current repository CLI command map and host-tool entry points
- `docs/TUNING.md`: tuning advice and operational guidance
- `docs/ALGORITHMS.md`: algorithm theory and detector explanations
- `docs/FEATURES.md`: ML feature inventory, retained metrics, verdicts, and research backlog
- `docs/LITERATURE.md`: reviewed sensing papers, methods, reported results, hardware limits, and ESPectre research relevance
- `docs/performance/README.md`: benchmark targets and current metrics
- `docs/ARCHITECTURE.md`: internal architecture, runtime/frontend split, and orchestration direction
- `docs/ESPECTRE_PROTOCOL.md`: shared device protocol, payloads, topics, and transport semantics
- `docs/adr/*.md`: architecture decision records, including durable decisions, historically important rejected directions, and the project-level rationale behind superseded baselines
- `src/python/micro_espectre/README.md`: Micro-ESPectre workflow, CLI, MQTT, and R&D positioning
- `docs/ML_DATA_COLLECTION.md`: dataset collection and labeling workflow
- `docs/ML_TRAINING.md`: ML training, export, and validation workflow
- `data/COLLECTION_PLAN.md`: mutable dataset collection and replacement backlog
- `data/auto_generated/DATASET_QUALITY_CHECK.md`: generated dataset admission and quality snapshot
- `docs/ROADMAP.md`: product direction and sequencing
- `docs/review/*.md`: dated review findings and their progress trackers, not
  current-state product documentation
- `src/cpp/frontend/*/README.md`: frontend-specific workflows, protocol surfaces, and firmware notes

For current CLI syntax, use `docs/CLI.md`,
`src/python/micro_espectre/README.md`, the relevant frontend README, and
`./espectre --help`. Avoid duplicating command examples in agent rules because
the wrapper evolves often.

## Environment And Commands

- Use the repository virtual environment for direct Python commands when available.
- The `./espectre` wrapper auto-runs through `.venv` when it exists.
- ESP-IDF frontend builds require the ESP-IDF environment that provides `idf.py`.
- Prefer the repository `./espectre` wrapper for local workflows.
- `./me` is legacy-only; do not add new references to it.
- Run `gh` commands only on explicit user request and with the required permissions.

## Testing And Validation

When tests fail:
1. Investigate the root cause first.
2. Prefer fixing implementation over weakening tests.
3. Ask the user before changing behavior expectations.

Never skip, disable, or weaken tests just to make them pass.

Run tests that bind local UDP sockets outside the filesystem/network sandbox. The
sandbox can reject `bind()` for every local address, including `127.0.0.1` and
`0.0.0.0`; treat `PermissionError` or `EPERM` from the test socket setup as a
sandbox restriction, not as evidence that the test should use a different IP.

After changing detection or calibration logic, run the relevant C++
motion-detection test and Python performance validation when feasible:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
.venv/bin/pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v
```

Keep C++ and Python algorithm trends aligned; see `docs/performance/README.md`.

## Python Rules

- Distinguish device/runtime code from host-side tooling.
- Device-oriented modules under `src/python/micro_espectre/` should stay MicroPython-friendly.
- Host-side code under `src/python/espectre_cli/`, `tools/`, and `test/python/` may use established CPython-only libraries.
- Use `config.py` as the source of truth for shared device/runtime constants.
- Heavy libraries such as `numpy` and `pandas` belong in host-side analysis, training, and validation tools only.
- Use type hints where supported and useful.
- Do not use `asyncio` in MicroPython device code.
- Do not hardcode WiFi/MQTT credentials; use `config_local.py`.

Python test baseline:

```bash
.venv/bin/pytest test/python -v
```

## C++ Rules

- Use ESP-IDF for firmware code, not Arduino.
- C++17 features are available.
- Keep shared `core` and `runtime` code frontend-agnostic.
- Follow ESPHome component conventions only inside `src/cpp/frontend/esphome/`.
- Use `ESP_LOGD`, `ESP_LOGI`, `ESP_LOGW`, `ESP_LOGE` in ESP-IDF firmware code.
- Do not edit `ml_weights.h` manually; it is generated by the training script.
- Do not add blocking code in firmware `loop()` paths or callbacks.
- Do not assume ESPHome-specific patterns apply to BLE, Matter, streamer, or shared runtime code.

## C++ File Naming And Placement

- Use `snake_case` basenames; header and implementation of the same unit share the basename.
- Name the file after its primary class (`CsiCaptureService` → `csi_capture_service.h`).
- Boundary interfaces live in the shared layer as `<name>.h`; implementations as `<name>_<variant>` in the owning layer (`mqtt_transport_esp_idf`, `ble_bindings_nimble`).
- Suffixes: `_service` (start/stop lifecycle), `_transport` (data transport), `_bindings` (mockable boundary to an external stack), `_frontend` (runtime listener adapter), `_helpers` (free functions, domain-prefixed). Do not introduce new `_manager` files.
- Placement: algorithms and CSI format in `core/`; platform-agnostic contracts in `runtime/`; anything including ESP-IDF/FreeRTOS/lwIP in `runtime/esp_idf/`; single-frontend code in `frontend/<name>/`.
- Placement exception: portable shims that guard SDK includes behind `ESP_PLATFORM` or `__has_include` and degrade cleanly on host builds may live in `runtime/` (`pending_event`, `runtime_time`), or in `core/` when `core` itself depends on them (`espectre_log`).
- Headers in `core/` and `runtime/` must not include headers from `runtime/esp_idf/`.
- Generic basenames (`utils`, `helpers`, `common`) require a domain prefix or genuinely cross-cutting, homogeneous content.
- Core files with a Python counterpart keep the same basename (`threshold`, `csi_features`, `ml_weights`).

## C++ Review Rules

- Review first-party C++ for correctness, duplication, responsibility,
  performance, and consistent frontend behavior; exclude generated, build, and
  vendored trees unless the task explicitly includes them.
- Enforce the dependency direction `Frontend -> Runtime -> Core`. A lower layer
  must not include, query, or call a higher layer.
- Place shared behavior in the lowest layer that correctly owns the
  responsibility and has the required dependencies. Do not move orchestration
  into `core` merely to centralize it.
- Consolidate repeated behavior only when its contract is genuinely
  homogeneous. Preserve intentional frontend differences through explicit
  capabilities rather than hidden branches or copied implementations.
- Compare ESPHome, Native, Matter, and Streamer where applicable for defaults,
  validation, lifecycle, events, reset behavior, error handling, and capability
  semantics. A difference is intentional only when the contract or a declared
  capability explains it.
- Inspect CSI callbacks, runtime loops, and inference paths for blocking work,
  allocation, copying, per-element division or modulo, repeated I/O, excessive
  stack use, oversized buffers, and debug-only work that remains active in
  release paths.
- Look explicitly for stale state, incomplete reset, incorrect readiness gates,
  missing bounds or clamps, null-handling divergence, lifecycle ordering bugs,
  and failure-recovery gaps.
- Test shared behavior at the shared layer. Keep frontend-specific tests focused
  on adapter behavior and declared capability differences.
- When the user requests a persistent review record, store it under
  `docs/review/` with stable finding ids, severity, precise locations, and one
  progress checklist. Treat review completion and finding resolution as
  separate states.

## Documentation Rules

- Clear, concise, technical style.
- Prefer bullets and tables over decorative formatting.
- Use a neutral informative tone for technical docs.
- Allow a more approachable product-facing tone in `README.md` and public entry-point docs.
- Keep frontend-specific workflows, protocols, and firmware surfaces in the local frontend README files.
- Verify current-state documentation against the implementation, runtime
  schema, and generated artifacts. Distinguish deployed behavior, partial
  implementation, and target direction explicitly.
- Make public compatibility, controller-support, privacy, and security claims
  only when repository evidence supports them. Use a validation matrix when
  coverage is incomplete.
- Keep one source of truth per topic. Secondary documents should summarize and
  link to the owner instead of repeating formulas, ranges, metrics, commands,
  or mutable corpus details.
- Keep `docs/ROADMAP.md` at the level of outcomes, gates, and sequencing. Put
  replay names, experiment metrics, and collection details in their owning
  feature, performance, or dataset documents.
- Use ADRs for durable architectural or project-level decisions, including
  historically important rejected directions. Keep one decision or coherent
  decision thread per ADR, preserve historical rationale, and prefer links to
  related ADRs or versioned snapshots over mutable narrative docs.
- When a decision changes, update the affected ADR metadata consistently:
  `Status`, `Supersedes`, and `Superseded by`. Do not leave superseded
  decisions marked `Accepted`.
- Treat `docs/FEATURES.md` as the feature experiment ledger. Record every
  seriously evaluated production, research, historical, planned, or rejected
  feature before removing its implementation or moving to the next candidate.
- For each measured feature, retain its physical interpretation and definition,
  scale-invariance status, implementation scope, corpus/split/seed context,
  primary and worst-group metrics, redundancy evidence when available, verdict,
  and the reason for that verdict. An unavailable metric must be marked as
  unavailable, not reconstructed from memory.
- Keep temporary experiment detail out of ADRs, the roadmap, and the changelog.
  Use `docs/FEATURES.md` for accumulated feature evidence, and create or update
  an ADR only when the evidence establishes, rejects, or supersedes a durable
  production direction.
- Keep host-only feature candidates under `tools/`, evaluate them with
  `--no-export`, and do not add runtime extractors until a promotion decision
  justifies Python/C++ parity.
- Treat `docs/LITERATURE.md` as the external research ledger. Record the source
  URL, release date, hardware and signal assumptions, methods, reported
  results, and ESPectre transfer limits; do not present internal ESPectre
  research as literature.
- Keep the active unreleased changelog focused on the final cumulative release
  state. Put superseded intermediate experiments in `docs/FEATURES.md` or ADRs.
- Do not edit generated performance or dataset-quality reports manually.
  Regenerate them from the current corpus, and run each generator's
  `--check-current` mode before describing the reports as current.
- Entry-point docs may include product/subproject branding in the title when established style supports it.
- Other docs should use simple descriptive titles.
- For internal Markdown links that point to files, use only the filename as the link text rather than the full relative path.
- Emoji should be rare and purposeful, not ornamental.
- Update `docs/CHANGELOG.md` only in the latest active section; do not rewrite
  historical release entries except to fix factual errors the user explicitly
  asked to correct.
- Update `docs/web/sitemap.xml` when public routes change. Do not edit the generated guide pages under `docs/web/guides/` (`hardware/`, `setup/`, `detection/`, `custom-firmware/`) or the generated roadmap page under `docs/web/roadmap/`; they are gitignored output of `.github/scripts/build_static_pages.py`, and direct edits evaporate at the next generation. Edit the fragments in `docs/web/guides/content/` or `docs/web/roadmap/content.html` instead, then re-run the script to preview.

## GitHub And CI Rules

- Keep workflow changes minimal and explicit.
- `develop` is the default PR target.
- `main` is release-only.
- Do not bypass branch protections.
- Do not push directly to `main`.
- Do not use force push.
- Do not merge with failing required checks.
- Prefer pinned major versions already used in the workflow or elsewhere in the repo.
- Before changing action versions, check current usage with `rg "uses: .*@" .github`.
- Keep action updates intentional, minimal, and grouped by purpose.
- Reply directly to GitHub issues/PRs only when explicitly requested.

## Commit Message Style

Use Conventional Commits with optional scope:
- `type(scope): subject`
- `type: subject`

Preferred types:
- `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `deps`, `perf`

Subject style:
- imperative mood
- concise and concrete
- lower-case after the colon
- no trailing period
- target 72 characters or less when practical

Use scopes when they add clarity, for example `ci` or `micro-espectre`. For dependency updates in workflows, prefer `deps(ci): ...`.

Do not add issue IDs in the subject line. Put references in the commit footer only when needed:
- `Fixes #93` or `Closes #93` when the commit closes the issue
- `Refs #93` when related but not closing

PR commits must include a valid `Signed-off-by` trailer. Prefer `git commit -s`.

## License And Dependencies

- Dual-licensed: GPLv3 (`LICENSE`) plus separately offered commercial licenses (`LICENSING.md`).
- Keep `core`, `runtime`, and the native/Matter/streamer frontends free of GPL-only dependencies; use permissively licensed components (Apache-2.0, MIT, BSD) so the code stays distributable under both tracks.
- Only `src/cpp/frontend/esphome/` may depend on GPLv3 ESPHome components.
- Do not add dependencies with licenses incompatible with GPLv3.
- Contributions require the DCO trailer and a one-time CLA signature; do not remove the CLA workflow.
- Add Python dependencies only when they are declared in the appropriate requirements file:
  - `requirements.txt` for the base workflow
  - `requirements-ml.txt` for ML/training extras

## Hard Constraints

- Do not modify CSI data format without updating both C++ and Python implementations.
- Do not modify shared detection/calibration algorithms without keeping C++ and Python aligned.
