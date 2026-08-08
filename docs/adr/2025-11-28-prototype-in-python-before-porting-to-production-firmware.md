# ADR: prototype in python before porting to production firmware

- Status: Accepted
- Date: 2025-11-28
- Recorded: 2026-07-09 (retrospective)

## Context

The `1.4.0` versioned changelog explicitly states that the refactoring of the C firmware was driven by lessons learned from the MicroPython implementation. It also states that Micro-ESPectre enabled faster parameter tuning and testing of optimal configurations, with successful patterns then ported back into the C firmware.

That workflow remains visible today in the project rules and structure: host-side and Python-side experimentation are used to validate ideas before they are promoted into shared C++ production paths.

## Decision

Use Python-first experimentation as the preferred innovation workflow:

- prototype and tune algorithms quickly in Micro-ESPectre or host-side Python
- validate promising results with tooling, datasets, and tests
- port the validated behavior into the shared C++ runtime or core layers

Treat the Python path as the fast exploration environment, and treat the shared C++ codebase as the place where validated production behavior is consolidated.

## Alternatives Considered

### Evolve the production firmware first and prototype directly there

Rejected. Firmware-only iteration is slower and makes experimental work harder to compare, tune, and revisit.

### Maintain completely separate algorithm evolution in Python and C++

Rejected. The project benefits when the Python path informs the production path instead of drifting into a parallel but disconnected implementation.

## Consequences

Benefits:

- algorithm exploration is faster and easier to validate
- production C++ changes can be grounded in prior Python evidence
- the project keeps a clear bridge between research work and deployed behavior

Trade-offs:

- parity between Python and C++ must be maintained intentionally
- experimental Python code cannot be treated as production behavior until it is ported and revalidated

## Related

- versioned changelog snapshot: `1.4.0:CHANGELOG.md`
- `docs/adr/2025-12-06-adopt-a-dual-platform-development-model.md`
- git commits: `c971f874`, `14e3ceb0`
