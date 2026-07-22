# ESPectre Agent Rules

## Project Snapshot

ESPectre is a Wi-Fi CSI sensing platform with Home Assistant and Matter integration.

Main code areas:
- `src/cpp/`: production C++ firmware platform, including core, runtime, and frontends
- `src/python/`, `tools/`, `test/python/`: Python/MicroPython R&D, CLI, tooling, and tests
- `src/cpp/frontend/esphome/`: ESPHome frontend
- `src/cpp/frontend/native/`: Standalone native frontend
- `src/cpp/frontend/matter/`: Matter frontend
- `src/cpp/frontend/streamer/`: CSI streamer frontend

Innovation flow: prototype in Python, validate, then port to the relevant shared C++ layers and frontend(s).

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
- `docs/SETUP.md`: shared configuration parameters, defaults, ranges, frontend chooser, and current CLI entry points
- `docs/TUNING.md`: tuning advice and operational guidance
- `docs/ALGORITHMS.md`: algorithm theory and detector explanations
- `docs/performance/README.md`: benchmark targets and current metrics
- `docs/ARCHITECTURE.md`: internal architecture, runtime/frontend split, and orchestration direction
- `docs/adr/*.md`: architecture decision records for stable technical choices, including context, decision, alternatives, and consequences
- `docs/ESPECTRE_PROTOCOL.md`: shared device protocol, payloads, topics, and transport semantics
- `docs/adr/*.md`: architecture decision records, including durable decisions, historically important rejected directions, and the project-level rationale behind superseded baselines
- `src/python/micro_espectre/README.md`: Micro-ESPectre workflow, CLI, MQTT, and R&D positioning
- `docs/ML_DATA_COLLECTION.md`: dataset collection and labeling workflow
- `docs/ML_TRAINING.md`: ML training, export, and validation workflow
- `docs/ROADMAP.md`: product direction and sequencing
- `src/cpp/frontend/*/README.md`: frontend-specific workflows, protocol surfaces, and firmware notes

For current CLI syntax, use `docs/SETUP.md`, `src/python/micro_espectre/README.md`, the relevant frontend README, and `./espectre --help`. Avoid duplicating command examples in agent rules because the wrapper evolves often.

## Environment And Commands

- Use the repository virtual environment for direct Python commands when available.
- The `./espectre` wrapper auto-runs through `.venv` when it exists.
- ESP-IDF frontend builds require the ESP-IDF environment that provides `idf.py`.
- Prefer the repository `./espectre` wrapper for local workflows.
- `./me` is legacy-only; do not add new references to it.
- Run `gh` commands only on explicit user request and with the required permissions.

## Testing And Validation

Tests catch bugs; they are not a checkbox.

When tests fail:
1. Investigate the root cause first.
2. Prefer fixing implementation over weakening tests.
3. Ask the user before changing behavior expectations.

Never skip, disable, or weaken tests just to make them pass.

Run tests that bind local UDP sockets outside the filesystem/network sandbox. The
sandbox can reject `bind()` for every local address, including `127.0.0.1` and
`0.0.0.0`; treat `PermissionError` or `EPERM` from the test socket setup as a
sandbox restriction, not as evidence that the test should use a different IP.

After changing detection/calibration logic, run the relevant C++ motion-detection test and Python performance validation when feasible:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v
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
pytest test/python -v
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

C++ test baseline:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
```

## C++ File Naming And Placement

- Use `snake_case` basenames; header and implementation of the same unit share the basename.
- Name the file after its primary class (`CsiCaptureService` → `csi_capture_service.h`).
- Boundary interfaces live in the shared layer as `<name>.h`; implementations as `<name>_<variant>` in the owning layer (`mqtt_transport_esp_idf`, `ble_bindings_nimble`).
- Suffixes: `_service` (start/stop lifecycle), `_transport` (data transport), `_bindings` (mockable boundary to an external stack), `_frontend` (runtime listener adapter), `_helpers` (free functions, domain-prefixed). Do not introduce new `_manager` files.
- Placement: algorithms and CSI format in `core/`; platform-agnostic contracts in `runtime/`; anything including ESP-IDF/FreeRTOS/lwIP in `runtime/esp_idf/`; single-frontend code in `frontend/<name>/`.
- Placement exception: portable shims that guard SDK includes behind `ESP_PLATFORM` or `__has_include` and degrade cleanly on host builds may live in `runtime/` (`espectre_log`, `pending_event`, `runtime_time`).
- Headers in `core/` and `runtime/` must not include headers from `runtime/esp_idf/`.
- Generic basenames (`utils`, `helpers`, `common`) require a domain prefix or genuinely cross-cutting, homogeneous content.
- Core files with a Python counterpart keep the same basename (`threshold`, `csi_features`, `ml_weights`).

## Documentation Rules

- Clear, concise, technical style.
- Prefer bullets and tables over decorative formatting.
- Use a neutral informative tone for technical docs.
- Allow a more approachable product-facing tone in `README.md` and public entry-point docs.
- Keep frontend-specific workflows, protocols, and firmware surfaces in the local frontend README files.
- Document meaningful experiments in `docs/adr/` when they establish, reject, or materially clarify a durable project direction, so the project keeps a useful historical record and avoids repeating past mistakes.
- Use ADRs under `docs/adr/` for durable architectural or project-level decisions, including historically important rejected directions. Keep ADRs concise, one decision or coherent decision thread per file, and prefer links to related ADRs or changelog snapshots over mutable narrative docs.
- Entry-point docs may include product/subproject branding in the title when established style supports it.
- Other docs should use simple descriptive titles.
- For internal Markdown links that point to files, use only the filename as the link text rather than the full relative path.
- Emoji should be rare and purposeful, not ornamental.
- Update `docs/CHANGELOG.md` only in the latest active section; do not rewrite historical release entries except to fix factual errors the user explicitly asked to correct.

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
