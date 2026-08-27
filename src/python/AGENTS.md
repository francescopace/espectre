# ESPectre Python Agent Rules

## Runtime Boundaries

- Distinguish device runtime code from host-side tooling.
- Keep `micro_espectre/` MicroPython-friendly. Do not use `asyncio`, CPython-only APIs, or heavy libraries there.
- Host-side code under `espectre_cli/`, repository `tools/`, and `test/python/` may use established CPython-only libraries.
- Use `micro_espectre/config.py` as the source of truth for shared MicroPython runtime constants.
- Keep heavy libraries such as `numpy` and `pandas` in host-side analysis, training, and validation tools only.
- Use type hints where the target runtime supports them and they improve the contract.
- Do not hardcode WiFi or MQTT credentials; use `config_local.py` or the documented local environment.
- Keep CSI formats and shared detection or calibration behavior aligned with the `C++` implementation.

## Validation

- Run the narrow owning test first with `-q --tb=short`; rerun only failures with `-vv` when more detail is required.
- Run the full Python baseline only when the changed surface spans multiple Python owners or when contribution-level validation is requested:

```bash
.venv/bin/pytest test/python -q --tb=short
```

- Use the repository virtual environment for direct Python commands when available, and prefer `./espectre` for supported operator workflows.
- Follow `test/AGENTS.md` before modifying tests. Follow `tools/AGENTS.md` for host-side research, ML, dataset, benchmark, or report workflows.
