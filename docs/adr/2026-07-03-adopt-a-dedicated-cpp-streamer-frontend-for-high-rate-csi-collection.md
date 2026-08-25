# ADR: adopt a dedicated c++ streamer frontend for high-rate csi collection

- Status: Superseded
- Date: 2026-07-03
- Recorded: 2026-07-09 (retrospective)
- Superseded by: `2026-08-25-unify-raw-csi-collection-over-http.md`

## Context

The older live-streaming workflow grew around Python-first tooling. Once the firmware became modular and multi-frontend, that separate path no longer matched the production architecture.

By the `v3` platform refactor, the repository needed a raw CSI transport path that could:

- sustain higher packet rates and larger dataset collection runs
- reuse the new `core / runtime / frontend` structure instead of keeping a side-channel streaming path outside the main firmware architecture
- support collector-driven target traffic rather than a firmware-owned traffic generator
- let the collector react to firmware-side TX saturation through explicit backpressure feedback instead of assuming a fixed safe pacing rate
- let host-side tooling consume the same stream live for backend analysis, data capture, and side-by-side detector inspection

The resulting shift was not only a transport optimization. It also retired the older Python-side streamer workflow as the primary path and moved the active streaming architecture into dedicated ESP-IDF firmware plus the host-side `collect` workflow.

That firmware-plus-collector split also established a closed-loop pacing model: the collector drives the rate by sending UDP pacing traffic, and the firmware reports cumulative TX backpressure when its uplink path cannot keep up. The collector can then reduce pacing quickly and recover more conservatively instead of treating stream rate as an open-loop constant.

The changelog records the same convergence: the C++ Streamer became the main live-streaming implementation, collection became collector-driven, and the CLI added broader multi-chip support plus live parallel detector display through `./espectre collect --detector lightweight,high_accuracy`.

## Decision

Adopt a dedicated C++ streamer frontend as the primary high-rate CSI collection path.

Concretely:

- implement the streamer as its own ESP-IDF frontend under `src/cpp/frontend/streamer/`
- make the C++ streamer protocol the main live-streaming path for host-side collection and inspection
- remove the older Python-side streamer workflow from the active architecture
- use collector-paced UDP traffic instead of a firmware-owned traffic generator
- expose firmware-side TX backpressure to the collector as an explicit pacing feedback signal
- treat `./espectre collect` as the host-side entrypoint for live collection, backend-side detector comparison, and dataset capture on top of the streamer transport

In this model, "backpressure" means the streamer firmware reached temporary TX capacity limits while trying to emit CSI datagrams for collector pacing slots. Rather than hiding those events as generic packet loss, the stream protocol surfaces cumulative backpressure telemetry (`tx_backpressure_total`) so the collector can adapt its pacing rate to current link conditions.

## Alternatives Considered

### Keep the Python-side streamer workflow as the main live-streaming path

Rejected. It remained useful historically for exploration, but it did not align with the modular firmware architecture or the throughput and data-volume goals that the dedicated firmware streamer could support.

### Fold raw CSI streaming into an existing runtime frontend

Rejected. Raw high-rate CSI transport has a different goal and state machine than the ecosystem-facing frontends. A dedicated streamer frontend keeps that path compact without re-coupling it to motion-telemetry surfaces.

## Consequences

Benefits:

- the project can collect at higher configured streaming rates on the supported firmware targets
- the active streaming path now fits the same modular frontend architecture as the rest of the firmware platform
- the collector can use firmware-reported backpressure as a control signal, reducing pacing quickly when the TX path saturates and recovering more slowly when the stream stabilizes
- `collect` can inspect the same live stream with multiple detectors in parallel, which improves backend-side validation and A/B comparison during collection

Trade-offs:

- host-side collection now depends more explicitly on the coordinated firmware-plus-collector workflow
- pacing behavior is no longer a static transport knob; it depends on the collector correctly interpreting firmware backpressure telemetry
- the raw streaming protocol and the collector must evolve together when packet metadata changes

## Related

- versioned changelog snapshot: `3.0.0:CHANGELOG.md`
- `src/cpp/frontend/streamer/README.md`
- `docs/adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
