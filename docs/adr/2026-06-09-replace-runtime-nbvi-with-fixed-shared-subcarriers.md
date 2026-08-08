# ADR: replace runtime nbvi with fixed shared subcarriers

- Status: Accepted
- Date: 2026-06-09
- Recorded: 2026-07-09 (retrospective)
- Supersedes: 2025-12-03-adopt-nbvi-for-runtime-subcarrier-selection.md

## Context

NBVI had been a major part of the project's runtime calibration story for multiple releases. Over time, however, that path accumulated operational and architectural costs: calibration complexity, memory pressure, persistence concerns, and tighter coupling between runtime startup and subcarrier search.

By the time of the `v3` platform refactor, the project had also moved toward shared runtime behavior across frontends. The current changelog and architecture docs show that the active runtime path no longer centers on runtime NBVI. Instead, it uses one fixed shared subcarrier set and a shared startup bootstrap strategy.

## Decision

Remove runtime NBVI from the active production path and replace it with:

- one fixed shared subcarrier set across the project
- shared startup threshold bootstrap logic in the runtime path
- simpler cross-stack defaults that do not depend on runtime band search or SPIFFS-backed calibration state

Keep historical NBVI research and validation context in the changelog and experiments, but do not keep runtime NBVI as the active architecture.

## Alternatives Considered

### Continue hardening the runtime NBVI path

Rejected. The project had already invested heavily there, but the broader platform direction favored simpler, more shareable runtime behavior.

### Keep both runtime NBVI and fixed-band paths as first-class production modes

Rejected. That would preserve optionality, but it would also keep the runtime surface and validation story more complex than necessary.

## Consequences

Benefits:

- runtime startup behavior became simpler and easier to share across frontends
- the project reduced dependence on calibration-time band search and storage
- production behavior aligned better with the later Classic and ML paths

Trade-offs:

- earlier NBVI-centric releases became historical rather than current guidance
- some historical docs and mental models needed reinterpretation through the new fixed-band runtime baseline

## Related

- `docs/adr/2025-12-03-adopt-nbvi-for-runtime-subcarrier-selection.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- git commit: `d10a10e2`
