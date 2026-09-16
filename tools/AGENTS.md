# ESPectre Tools Agent Rules

## Host-Side Scope

- Code under `tools/` is host-side CPython and may use dependencies declared in `requirements.txt` or `requirements-ml.txt`.
- Heavy libraries such as `numpy` and `pandas` belong here rather than in MicroPython device code.
- Keep host-only feature candidates under `tools/`, and evaluate them through the owning tool's evaluation-only workflow; use `--no-export` with `train_ml_model.py`. Do not add runtime extractors until a promotion decision justifies Python and `C++` parity.
- Regenerate `src/cpp/core/ml_weights.h` through training and export instead of editing it manually. Follow [AGENTS.md](../docs/AGENTS.md#generated-material) for performance and dataset-quality reports.
- Preserve the canonical CSI format and feature registry. Do not introduce tool-local aliases or duplicate production constants.

## Validation

- Use the owning end-to-end workflow in [README.md](README.md) to validate research tools and generated artifacts. Follow the test-ownership rules in [AGENTS.md](../test/AGENTS.md#test-ownership) when deciding whether to add maintained tests.
- Follow `docs/AGENTS.md` before updating feature, literature, ML, performance, or dataset documentation, and follow `test/AGENTS.md` before modifying maintained tests.

## Firmware Benchmark Contract

- Read [README.md](README.md#firmware-benchmark-contract) before changing `benchmark_firmware.py` or its split owners. It owns the canonical build, erase, provisioning, Direct sampling, and scoring contract.
- Keep the firmware benchmark a dumb client of `./espectre` for build, flash, erase, reset, provisioning, onboarding, and serial monitoring. Direct sampling and scoring remain benchmark responsibilities.
- Require an explicit benchmark serial port and pass it unchanged to delegated CLI commands. Do not perform benchmark-local serial discovery or track USB identities across re-enumeration.
- Do not import or invoke esptool, pyserial reset controls, USB power controls, or other hardware lifecycle mechanisms from the benchmark.
- Treat delegated CLI exit status as final. Do not parse human-readable flash output or add retries, fallback resets, power cycles, or recovery paths that turn a failed flash into success.
- Consume final machine-readable CLI records when a delegated workflow exposes them. Do not scrape equivalent human-readable output.
- Update the behavioral contract tests whenever the build, flash, provisioning, Direct evidence, BSSID evidence, or serial-error policy changes.
