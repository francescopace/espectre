# ESPectre Python Agent Rules

## Runtime Boundaries

- Keep `micro_espectre/` MicroPython-friendly. Do not use `asyncio`, CPython-only APIs, or heavy libraries there.
- Host-side code under `espectre_cli/`, repository `tools/`, and `test/python/` may use established CPython-only libraries. Keep heavy libraries such as `numpy` and `pandas` in host-side analysis, training, and validation tools.
- Use `micro_espectre/config.py` as the source of truth for shared MicroPython runtime constants.
- Use type hints where the target runtime supports them and they improve the contract.
- Use `config_local.py` or the documented local environment for device-local connectivity settings.
- For shared detection or calibration changes, follow the regression, performance target, and C++/Python comparison requirements in [AGENTS.md](../cpp/AGENTS.md#detection-and-calibration-parity).

## Validation

- Run the full Python baseline only when the changed surface spans multiple Python owners or when contribution-level validation is requested:

```bash
.venv/bin/pytest test/python -q --tb=short
```

- Follow `test/AGENTS.md` before modifying tests. Follow `tools/AGENTS.md` for host-side research, ML, dataset, benchmark, or report workflows.
