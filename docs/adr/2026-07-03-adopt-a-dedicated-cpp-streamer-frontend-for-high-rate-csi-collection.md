# ADR: adopt a dedicated c++ streamer frontend for high-rate csi collection

- Status: Accepted
- Date: 2026-07-03
- Recorded: 2026-07-09 (retrospective)

## Context

The older live-streaming workflow had grown around Python-first tooling, which
was valuable for exploration but no longer matched the project's needs once the
firmware architecture became explicitly modular and multi-frontend.

By the `v3` platform refactor, the repo needed a raw CSI transport path that
could do all of the following at the same time:

- sustain materially higher packet rates and larger dataset collection runs
- reuse the new `core / runtime / frontend` structure instead of keeping a
  side-channel streaming path outside the main firmware architecture
- support collector-driven target traffic rather than a firmware-owned traffic
  generator
- propagate stimulus and reference metadata into the saved CSI stream so the
  host can reason about temporal alignment, packet grouping, and later
  phase/coherence analysis
- let host-side tooling consume the same stream live for backend analysis, data
  capture, and side-by-side detector inspection

The resulting shift was not only a transport optimization. It also retired the
older Python-side streamer workflow as the primary path and moved the active
streaming architecture into dedicated ESP-IDF firmware plus the host-side
`collect` workflow.

The changelog records the same convergence: the C++ streamer path became the
main live-streaming implementation, collection became collector-driven, and the
CLI grew broader multi-chip support plus live parallel detector display through
`./espectre collect --detector classic,ml`.

## Decision

Adopt a dedicated C++ streamer frontend as the primary high-rate CSI collection
path.

Concretely:

- implement the streamer as its own ESP-IDF frontend under
  `src/cpp/frontend/streamer/`
- make the C++ streamer protocol the main live-streaming path for host-side
  collection and inspection
- remove the older Python-side streamer workflow from the active architecture
- use collector-driven external UDP stimulus instead of a firmware-owned traffic
  generator
- propagate `stimulus_id` and reference-frame markers into the CSI stream so
  host-side tooling can perform real-time analysis and save richer datasets
- treat `./espectre collect` as the host-side entrypoint for live collection,
  backend-side detector comparison, and dataset capture on top of the streamer
  transport

## Alternatives Considered

### Keep the Python-side streamer workflow as the main live-streaming path

Rejected. It remained useful historically for exploration, but it did not align
with the modular firmware architecture or the throughput and data-volume goals
that the dedicated firmware streamer could support.

### Fold raw CSI streaming into an existing runtime frontend

Rejected. Raw high-rate CSI transport has a different goal and state machine
than the ecosystem-facing frontends. A dedicated streamer frontend keeps that
path compact without re-coupling it to motion-telemetry surfaces.

## Consequences

Benefits:

- the project can collect substantially more data and sustain higher streaming
  rates on the supported firmware targets
- the active streaming path now fits the same modular frontend architecture as
  the rest of the firmware platform
- collector-driven stimulus and reference markers make backend-side temporal
  grouping, reference-assisted analysis, and later phase/coherence work more
  practical
- `collect` can inspect the same live stream with multiple detectors in
  parallel, which improves backend-side validation and A/B comparison during
  collection

Trade-offs:

- the streamer frontend is intentionally special-case and does not use the full
  `IEspectreRuntime` facade
- host-side collection now depends more explicitly on the coordinated
  firmware-plus-collector workflow
- the raw streaming protocol and the collector must evolve together when packet
  metadata changes

## Related

- versioned changelog snapshot: `3.0.0:CHANGELOG.md`
- `src/cpp/frontend/streamer/README.md`
- `docs/adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-07-04-preserve-multi-device-metadata-as-a-research-compatible-dataset-contract.md`
