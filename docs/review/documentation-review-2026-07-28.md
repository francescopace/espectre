# Documentation Review — Ownership, Consistency, and Current State

Date: 2026-07-28
Branch: `v3.0`
Scope: first-party project documentation, documentation source fragments, ADRs,
dataset metadata references, and generated documentation inputs.

The review compares documentation with the current implementation, trainer
surface, dataset metadata, and project direction. It does not treat mutable
experiment results as architectural decisions.

---

## 1. How To Read This Document

Findings carry a stable id so they can be referenced from commits and issues:

| Prefix | Theme |
| --- | --- |
| `S-n` | Documented state does not match the implementation |
| `W-n` | Stale or misleading workflow |
| `H-n` | Historical record and ADR consistency |
| `D-n` | Dataset and generated-report consistency |
| `O-n` | Source-of-truth ownership and duplication |
| `M-n` | Minor links, presentation, and coverage |

Severity uses three levels:

- **High**: the documentation can cause an incorrect implementation, experiment,
  privacy assumption, or product claim.
- **Medium**: the project remains usable, but ownership or status is ambiguous
  enough to cause drift.
- **Low**: local cleanup, discoverability, or presentation work.

Section 2 is the single source of truth for progress. Detailed findings do not
repeat their status.

---

## 2. Progress Tracker

### Batch 1 — Current implementation and public contracts

- [x] **S-1** (High) — Scope adaptive detector timing to what runtimes implement
  today → [§3](#s-1)
- [x] **S-2** (High) — Correct the device-identity and privacy contract
  → [§3](#s-2)
- [x] **S-3** (Medium) — Align documented configuration ranges with the schema
  → [§3](#s-3)
- [x] **S-4** (Medium) — Qualify Matter controller support and publish the
  validation state → [§3](#s-4)

### Batch 2 — ML workflows and active history

- [x] **W-1** (High) — Remove obsolete synthetic-data modes and stale feature
  commands → [§4](#w-1)
- [x] **H-1** (High) — Make the active 3.0 changelog describe the final feature
  set and defaults → [§5](#h-1)
- [x] **H-2** (Medium) — Complete the ADR supersession chain and normalize
  statuses → [§5](#h-2)

### Batch 3 — Dataset and generated evidence

Completed against dataset revision
`sha256:ac37984e04d5e95e8249e7272a3ab261b4f63912d8439979c3760133ddc7d4e2`.
Both generators now embed the revision and expose a lightweight
`--check-current` staleness gate.

- [x] **D-1** (Medium) — Align the collection plan and roadmap with active
  dataset roles → [§6](#d-1)
- [x] **D-2** (Medium) — Regenerate performance and quality reports from a
  stable corpus, with a traceable dataset revision → [§6](#d-2)

### Batch 4 — Source-of-truth boundaries

- [x] **O-1** (Medium) — Keep roadmap entries at outcome level and move mutable
  evidence to its owning documents → [§7](#o-1)
- [x] **O-2** (Medium) — Make the feature catalog own feature definitions,
  status, and experiment evidence → [§7](#o-2)
- [x] **O-3** (Medium) — Complete the documentation ownership map and remove
  duplicate ownership declarations → [§7](#o-3)

### Batch 5 — Secondary documentation cleanup

- [x] **M-1** (Low) — Repair project-owned Markdown links and orphaned image
  captions → [§8](#m-1)
- [x] **M-2** (Low) — Bring the C++ test catalog and dataset-family list up to
  date → [§8](#m-2)
- [x] **M-3** (Low) — Use physically accurate wording for the generated ML
  performance summary → [§8](#m-3)

---

## 3. Current Implementation and Public Contracts

### S-1 — Adaptive timing is documented more broadly than it is deployed {#s-1}

**Where**

- [ALGORITHMS.md](../ALGORITHMS.md)
- [TUNING.md](../TUNING.md)
- [2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md](../adr/2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md)

**Finding**

The documentation says that deployed runtimes derive the detector window, lags,
and evaluation cadence from the measured packet rate. Host replay tooling does
derive the full timing configuration. The deployed C++ and MicroPython runtimes
currently adapt evaluation cadence and gap handling, but detector window and lag
values remain construction-time values. The unused runtime wiring is also
tracked as `B-3` in the C++ review.

**Resolution**

Document the current partial implementation explicitly. Preserve the ADR as the
accepted direction, but distinguish the target contract from deployed behavior.

### S-2 — The protocol privacy statement contradicts device-id derivation {#s-2}

**Where**: [ESPECTRE_PROTOCOL.md](../ESPECTRE_PROTOCOL.md)

**Finding**

The protocol overview says device identifiers are not MAC addresses, while the
wire-format section and current firmware derive the identifier from the station
MAC. The field is logically an opaque protocol identifier, but its current value
is a persistent hardware-derived identifier and must be treated accordingly.

**Resolution**

State the current derivation and privacy implications. Require managed or
privacy-sensitive deployments to pseudonymize or replace it before external
exposure.

### S-3 — Canonical setup ranges differ from the runtime schema {#s-3}

**Where**: [SETUP.md](../SETUP.md)

**Finding**

The documented segmentation-window and traffic-rate ranges do not match
`runtime_sensing_schema.h`. The schema accepts a 100–200 packet window and a
0–100000 packets-per-second arithmetic range. The latter is a validation bound,
not a claim that ESP32 radios can sustain that rate.

**Resolution**

Use schema ranges in the configuration table and separately explain practical
operating rates.

### S-4 — Public Matter claims exceed the verified controller matrix {#s-4}

**Where**

- [README.md](../../README.md)
- [setup.html](../web/guides/content/setup.html)
- [Matter frontend README](../../src/cpp/frontend/matter/README.md)

**Finding**

Public entry points imply verified operation across multiple controller
ecosystems. The Matter frontend documentation records limited controller
validation and no complete cross-controller matrix.

**Resolution**

Describe Matter as a standards-based compatibility path, publish the validation
state, and avoid presenting unverified controllers as tested.

---

## 4. ML Workflows

### W-1 — Removed synthetic modes and feature names remain in active guides {#w-1}

**Where**

- [ML_TRAINING.md](../ML_TRAINING.md)
- [ML_DATA_COLLECTION.md](../ML_DATA_COLLECTION.md)
- [tools/README.md](../../tools/README.md)

**Finding**

The training guide still describes `reference_match` and `shared_session`
synthetic-generation modes even though the generator was removed. Collection
documentation presents legacy Core-6 synthetic metadata as a current production
workflow. A tools example requests the removed `l1_delta` feature.

**Resolution**

Remove orphaned modes, label retained synthetic metadata as backward-compatible
legacy input, and use a valid host-only candidate example with `--no-export`.

---

## 5. Historical Record

### H-1 — The active 3.0 changelog contains superseded intermediate states {#h-1}

**Where**: [CHANGELOG.md](../CHANGELOG.md)

**Finding**

The unreleased section mixes the final five-feature model and Classic Matter
default with intermediate Core-6, Coherence-6/7, old subcarrier indices, ML-only
Matter, and a deleted synthetic generator. Because the section describes one
unreleased version, these entries read as simultaneous current behavior.

**Resolution**

Describe the final state in the active release section. Keep durable rejected or
superseded directions in ADRs and the feature catalog instead of preserving
every intermediate step as an active release claim.

### H-2 — Superseded ADRs still appear accepted {#h-2}

**Where**: [ADR index](../adr/README.md)

**Finding**

The Core-6, Coherence-6, lag-ratio, seven-feature-surface, and Matter-default
decisions form a supersession chain, but several older records still have
`Status: Accepted`. A few status fields also do not follow the metadata
convention documented by the ADR index.

**Resolution**

Mark superseded decisions explicitly, link successor and predecessor records,
and normalize status metadata without rewriting historical rationale.

---

## 6. Dataset and Generated Evidence

### D-1 — Collection-plan roles do not match current metadata {#d-1}

**Where**

- [COLLECTION_PLAN.md](../../data/COLLECTION_PLAN.md)
- [ROADMAP.md](../ROADMAP.md)
- `data/dataset_info.json`

**Finding**

The plan says both a weak S3 bedroom holdout pair and a quiet S3 selection replay
were excluded. Current metadata keeps the weak S3 pair in `holdout`; only the
quiet replay is excluded. The roadmap also labels an active C6 holdout pair as
C5.

**Resolution**

Correct the manual plan and roadmap. Keep the weak S3 pair active until a
replacement exists, while retaining its replacement priority.

### D-2 — Generated reports are not tied visibly to the corpus revision {#d-2}

**Where**

- [performance/README.md](../performance/README.md)
- `data/dataset_quality.md`
- `data/dataset_info.json`

**Finding**

The dataset metadata has changed since the latest generated reports. The reports
show generation time but not a dataset revision or content hash, so readers
cannot distinguish a current report from one generated against an earlier
corpus.

**Resolution**

Regenerate only when the intended corpus is stable. Then make report generators
embed the dataset `updated_at` value or a deterministic revision hash, and add a
staleness check suitable for CI.

---

## 7. Source-of-Truth Boundaries

### O-1 — The roadmap owns mutable experiment detail {#o-1}

**Where**: [ROADMAP.md](../ROADMAP.md)

**Finding**

The roadmap includes replay names, false-positive counts, corpus surgery, and
feature experiment metrics. These facts age faster than product sequencing and
already overlap the feature catalog, collection plan, performance report, and
ADRs.

**Resolution**

Keep outcomes, gates, and sequencing in the roadmap. Link to the owning evidence
documents for feature and corpus detail.

### O-2 — Exact feature definitions are duplicated outside the catalog {#o-2}

**Where**

- [FEATURES.md](../FEATURES.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [ML_TRAINING.md](../ML_TRAINING.md)

**Finding**

The feature catalog is intended to own formulas, implementation status,
experiment evidence, and research candidates, but exact production-feature
definitions are repeated in algorithm and workflow guides.

**Resolution**

Keep detector composition and theory in the algorithm guide, workflow in the
training guide, and exact feature definitions, status, and evidence in the
feature catalog.

### O-3 — The documentation ownership map is incomplete and duplicated {#o-3}

**Where**

- `AGENTS.md`
- [README.md](../../README.md)

**Finding**

The source-of-truth list duplicates the ADR entry and omits the CLI guide. It
does not identify the collection plan and generated dataset-quality report as
the owners of their narrower topics.

**Resolution**

Give each topic one declared owner and retain links from secondary documents
instead of repeating mutable content.

---

## 8. Secondary Documentation

### M-1 — Two project-owned links are broken and README captions are orphaned {#m-1}

**Where**

- [Native frontend README](../../src/cpp/frontend/native/README.md)
- [C++ review](cpp-review-2026-07-28.md)
- [README.md](../../README.md)

**Finding**

The Native guide refers to a removed local game README. The C++ review still
points to the former log-shim location after its move. The root README contains
two image captions without their images.

**Resolution**

Use the public game URL, update the resolved review finding, and restore the
existing project images.

### M-2 — The C++ test catalog and dataset-family list are incomplete {#m-2}

**Where**: [test/cpp/README.md](../../test/cpp/README.md)

**Finding**

The guide lists only part of the 27 registered CTest targets and omits C5 from
the dataset-family table.

**Resolution**

List the current test targets and all five capture families.

### M-3 — Generated ML summary uses an inaccurate feature description {#m-3}

**Where**

- [performance/README.md](../performance/README.md)
- `tools/lib/performance_report.py`

**Finding**

The generated comparison calls the ML detector “turbulence and spectral
features,” while the production model uses scale-invariant turbulence and
normalized channel-profile dynamics.

**Resolution**

Correct the generator wording and refresh the generated report with the stable
corpus as part of D-2.

---

## 9. Verified Consistent Areas

- `./espectre --help` and [CLI.md](../CLI.md) expose the same top-level command
  map.
- The current ML artifact, C++ feature extractor, MicroPython feature extractor,
  algorithm guide, and feature catalog agree on the five production features and
  the `5 -> 32 -> 16 -> 1` topology.
- C++ and Python agree on the selected subcarrier indices
  `[4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]`.
- The architecture guide matches the current `core`, `runtime`, and frontend
  dependency direction.
- Protocol version, topics, and BLE UUIDs are aligned across code and current
  documentation.
- No broken first-party HTML links were found. Repeated ESP-IDF prerequisite
  text across frontend guides is intentional shared setup, not conflicting
  ownership.
