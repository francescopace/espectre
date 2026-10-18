# ADR: prototype in Python before porting to production firmware

- Status: Accepted
- Date: 2025-11-28
- Recorded: 2026-07-09 (retrospective)

## Context

Micro-ESPectre made tuning parameters and trying configurations much faster than the C firmware. The 1.4.0 refactor of the C firmware was built on what we learned there.

## Decision

New algorithms start in Python, either in Micro-ESPectre or in host tools. Once they hold up on datasets and tests, we port them to the shared C++ core or runtime. Python is where we explore; C++ is where production behavior lives.

## Alternatives considered

- **Iterate directly in the firmware.** Slower, and experiments are harder to compare and tune.
- **Let Python and C++ evolve separately.** The two would drift, and Python results would stop saying anything about production.

## Consequences

C++ changes arrive with Python evidence behind them, and parity tests tie the two implementations together. The cost is that parity must be kept on purpose, and Python code is not production behavior until it is ported and revalidated.

## Related

- `1.4.0:CHANGELOG.md`
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- Commits `c971f874` and `14e3ceb0`
