# ADR: retain provenance-filtered CSI admission

- Status: Accepted
- Date: 2026-08-28
- Updated: 2026-08-29

## Context

ESPectre uses managed traffic to supply CSI samples to a fixed `100 pps` temporal grid. Direct HTTP control requests, an SSE client, and unrelated local traffic can also produce valid CSI callbacks. An experiment on an ESP32-S3 tested whether those incidental packets could supplement the external UDP generator.

The experimental firmware kept the existing RF, BSSID, channel, frame-format, and local-destination validation, but changed the external UDP marker from an admission gate into a provenance classifier. All otherwise valid local IPv4 traffic could therefore reach the detector. The detector target, temporal grid, occupancy denominator, worker configuration, and priorities remained unchanged.

The benchmark used Native Lightweight, BSSID `E6:FA:C4:20:19:DE`, one active SSE client, and one new Direct `diagnostics` connection per second. Each selected run measured 60 seconds after settling.

## Evidence

At an external generator rate of `100 pps`, opening the filter produced only a small change relative to the marker-filtered run:

| Admission policy | Generator | Callback | Admitted | Occupancy | Missing slots | Excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| marker-filtered | 100 pps | 117.69 pps | 91.27 pps | 91.35% | 8.71/s | 8.74 pps |
| open local IPv4 | 100 pps | 117.58 pps | 91.85 pps | 91.95% | 8.15/s | 25.61 pps |

The open filter gained `0.58 admitted pps` and `0.60` occupancy percentage points, while excess increased by `16.87 pps`. The selected run completed all 60 Direct requests without host timeouts or control/SSE send errors. Two earlier open-filter attempts ended in an intermittent `espectre_native` stack overflow and reboot; a later complete run did not reproduce the failure, so this is a risk signal rather than proof of causality.

Reducing the external generator to `90 pps` with the filter open failed the experiment's acceptance threshold:

| Generator | Callback | Admitted | Occupancy mean | Occupancy range | Missing slots | Excess |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 90 pps | 109.89 pps | 70.73 pps | 70.95% | 36–90% | 29.25/s | 29.33 pps |

All 60 Direct requests succeeded in that run, with a `187.10 ms` 95th-percentile complete-response latency and no reboot. However, both admitted rate and occupancy were below the required `80` threshold. The planned `85 pps` step was not run because `90 pps` had already failed.

These results show that incidental Direct, SSE, and ambient packets do not reliably replace a corresponding share of the managed stream. Their timing is bursty relative to the fixed grid, so most of the additional callbacks become same-slot excess rather than new temporal coverage.

## Decision

Keep configured traffic provenance as a live detector admission gate:

- in external mode, admit only a canonical UDP marker or a unicast ICMP Echo Request matching the device destination identity;
- in internal mode, admit only traffic matching the configured managed ping or DNS source;
- reject otherwise valid incidental Direct, SSE, and ambient traffic before temporal detector admission;
- keep `csi_target_pps=100`, the `100 pps` detector grid, and a fixed `100 pps` external generator for the current ESP32-S3 setup; and
- do not implement an occupancy-driven or excess-driven traffic controller from this experiment.

The experiment-specific benchmark changes are removed after recording the evidence. Existing general benchmark facilities remain outside this decision and are unchanged.

Local raw samples, manifests, analyses, and summaries remain under `data/untracked/firmware_benchmarks/`. The measurements above are recorded here because those laboratory artifacts are intentionally not versioned.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-08-28 | Admit every valid local IPv4 CSI callback after RF and destination validation | Rejected; the 100 pps run mostly increased same-slot excess and exposed an intermittent stability risk |
| 2026-08-28 | Reduce the external generator to 90 pps, then 85 pps, while incidental traffic fills the grid | Rejected; 90 pps reached only 70.73 admitted pps and 70.95% occupancy, so 85 pps was not run |
| 2026-08-28 | Retain provenance-filtered detector admission and the fixed 100 pps external source | Accepted |
| 2026-08-29 | Add bounded ICMP Echo Requests to the explicit external provenance set | Accepted; unlike the rejected open filter, this admits one exact diagnostic packet shape rather than incidental local traffic |

## Alternatives Considered

### Admit all valid local traffic

Rejected. Validity and relevance do not guarantee useful temporal placement. In the measured workload, extra callbacks primarily increased same-slot excess and did not provide enough new occupied slots to reduce managed traffic.

### Adapt the generator from occupancy and excess

Rejected for the current runtime. The experiment did not establish that incidental traffic can provide stable next-second coverage, and occupancy remains a capped observation of the fixed grid rather than a direct traffic-demand signal. Any future controller requires a separate design and validation effort.

### Keep 90 pps as an S3 operating point

Rejected. It missed both acceptance criteria by roughly 9 pps and 9 percentage points and showed a wide occupancy range.

## Consequences

The live detector receives an explicitly managed traffic source with bounded, deterministic provenance. Direct and SSE activity remains observable through transport diagnostics but cannot silently change sensing coverage or detector load. An operator can supply UDP markers or unicast Echo Requests in `external`; mixing them remains the external sender's responsibility. The external generator continues to cost `100 pps` on this setup, and future reductions require a managed replacement source or new evidence rather than relying on ambient traffic.

## Related

- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- [`2026-08-23-standardize-managed-csi-traffic-sources.md`](2026-08-23-standardize-managed-csi-traffic-sources.md)
