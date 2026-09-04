# ESPectre C++ Agent Rules

## Architecture And Placement

- Use ESP-IDF for firmware code, not Arduino. `C++`17 features are available.
- Keep shared `core` and `runtime` code frontend-agnostic, and enforce `Frontend -> Runtime -> Core`.
- Follow ESPHome component conventions only inside `frontend/esphome/`. Do not assume ESPHome-specific patterns apply to Matter, Native, or shared runtime code.
- Use `ESPECTRE_LOGD`, `ESPECTRE_LOGI`, `ESPECTRE_LOGW`, and `ESPECTRE_LOGE` in shared `core` and `runtime` code so frontends own the logging backend. Frontend-specific ESP-IDF code may use `ESP_LOG*` directly. Do not add blocking work in firmware `loop()` paths or callbacks.
- Place algorithms and the CSI format in `core/`, platform-agnostic contracts in `runtime/`, ESP-IDF, FreeRTOS, or lwIP code in `runtime/esp_idf/`, and single-frontend code in `frontend/<name>/`.
- Portable shims may live in `runtime/`, or in `core/` when `core` depends on them, only when they guard SDK includes behind `ESP_PLATFORM` or `__has_include` and degrade cleanly on host builds.
- Headers in `core/` and `runtime/` must not include headers from `runtime/esp_idf/`.
- Use `snake_case` basenames, and give a header and implementation of the same unit the same basename. Name the file after its primary class (`CsiCaptureService` -> `csi_capture_service.h`).
- Boundary interfaces live in the shared layer as `<name>.h`; implementations live in the owning layer as `<name>_<variant>`.
- Use `_service` for start/stop lifecycle, `_transport` for data transport, `_bindings` for a mockable external-stack boundary, `_frontend` for a runtime listener adapter, and `_helpers` for domain-prefixed free functions. Do not introduce new `_manager` files.
- Generic basenames such as `utils`, `helpers`, and `common` require a domain prefix or genuinely cross-cutting, homogeneous content.
- Core files with a Python counterpart keep the same basename, including `threshold`, `csi_features`, and `ml_weights`.
- Do not edit `ml_weights.h` manually; regenerate it through the training workflow.

## Published SDK Surface

- Treat everything reachable from `espectre_sdk.h` as the published SDK surface.
- Adding a public type requires updating the facade include, the `Doxyfile` INPUT list, and the header map in `docs/SDK.md` in the same change. `test/python/contracts/test_sdk_surface_invariants.py` enforces this.
- Forward declarations are acceptable, but every public definition must still arrive through the facade. A type an integrator can name in a signature but cannot construct is a broken surface.
- Adding or changing a member of `IEspectreRuntime`, `IRuntimeListener`, or a boundary interface breaks external implementers. Give new members a default implementation, or take the break deliberately and record it in the active changelog section.
- Document accessors that silently degrade when a build-time option is disabled. Prefer removing a cheap build-time gate over hiding available data.
- Document a member fully or with a brief alone. `WARN_NO_PARAMDOC` is an error, so do not leave partial `@param` documentation.

After changing the published surface, run:

```bash
.venv/bin/pytest test/python/contracts/test_sdk_surface_invariants.py -q --tb=short
python3 .github/scripts/generate_sdk_api.py
```

`generate_sdk_api.py` requires Doxygen 1.17.0, the same version CI installs.

## Detection And Calibration Parity

- Keep `C++` and Python algorithm trends aligned. Do not change the CSI format or shared detection or calibration behavior on only one path.
- After changing detection or calibration logic, run both parity validations:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
.venv/bin/pytest test/python/performance/test_validation_real_data.py::TestPerformanceMetrics -q --tb=short
```

- If either validation cannot run, report the exact command and blocker; use `docs/performance/README.md` for the current performance workflow.

## Review Rules

- Review first-party `C++` for correctness, ownership, performance, duplication, and frontend consistency. Exclude generated, build, and vendored trees unless requested.
- Put homogeneous shared behavior in the lowest layer that owns it, keep orchestration out of `core`, and model intentional frontend differences as explicit capabilities.
- Compare ESPHome, Native, and Matter defaults, validation, lifecycle, events, reset behavior, error handling, and capabilities only where the reviewed behavior crosses those frontends.
- Inspect CSI callbacks, runtime loops, and inference paths for blocking work, allocation, copying, per-element division or modulo, repeated I/O, excessive stack use, oversized buffers, and debug-only work active in release paths.
- Check stale state, incomplete reset, readiness gates, bounds and clamps, null handling, lifecycle ordering, and recovery paths.
- Test shared behavior at the shared layer. Keep frontend-specific tests focused on adapter behavior and declared capability differences.
- When the user requests a persistent review record, store it under `docs/review/` with stable finding IDs, severity, precise locations, and one progress checklist. Keep review completion separate from finding resolution.

## Licensing

- Keep `core`, `runtime`, and the Native and Matter frontends free of GPL-only dependencies. Use permissively licensed components such as Apache-2.0, MIT, or BSD so those layers remain distributable under both licensing tracks.
- Only `frontend/esphome/` may depend on GPLv3 ESPHome components.
