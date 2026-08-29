# ESPectre Tools Agent Rules

## Host-Side Scope

- Code under `tools/` is host-side CPython and may use dependencies declared in `requirements.txt` or `requirements-ml.txt`.
- Heavy libraries such as `numpy` and `pandas` belong here rather than in MicroPython device code.
- Keep host-only feature candidates under `tools/`, evaluate them with `--no-export`, and do not add runtime extractors until a promotion decision justifies Python and `C++` parity.
- Do not edit `src/cpp/core/ml_weights.h`, performance reports, or dataset-quality reports manually; regenerate them through their owning tool.
- Preserve the canonical CSI format and feature registry. Do not introduce tool-local aliases or duplicate production constants.

## Validation And Context

- Large scripts such as training, benchmark, dataset-quality, and replay tools must be read by symbol or bounded range, not in full.
- Validate research tools, generated reports, build plumbing, and one-off scripts through their owning end-to-end workflow rather than adding unit tests without a maintained contract.
- Run generators in `--check-current` mode before describing committed generated artifacts as current.
- Keep full training, benchmark, and hardware logs out of the model context when the command succeeds. Report the summary and expose only bounded diagnostic tails when a failure requires investigation.
- Follow `docs/AGENTS.md` before updating feature, literature, ML, performance, or dataset documentation, and follow `test/AGENTS.md` before modifying maintained tests.

## Firmware Benchmark Contract

- Treat [`README.md`](README.md#firmware-benchmark-contract) as the normative owner of the firmware benchmark contract. Keep `benchmark_firmware.py` and its split owners aligned with that section.
- Do not add benchmark-specific firmware configuration, dedicated build directories, forced clean builds, serial-derived runtime metrics, or frontend-specific provisioning shortcuts that bypass the production workflow.
- Preserve provisioning boundaries explicitly: Matter must commission through a revision-compatible CHIP Tool controller, and Micro-ESPectre may inject connectivity settings because it does not support Improv Serial.
- Update the behavioral contract tests whenever the build, flash, provisioning, Direct evidence, BSSID evidence, or serial-error policy changes.
