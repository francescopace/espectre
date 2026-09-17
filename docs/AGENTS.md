# ESPectre Documentation Agent Rules

## Style And Ownership

- Use clear, concise, technical English and a neutral tone except in product-facing entry points. Prefer bullets or tables only when they improve readability.
- Do not hard-wrap prose. Keep each paragraph and list item on one source line unless Markdown syntax requires a line break.
- Use simple descriptive titles, filename-only text for internal links, and rare, purposeful emoji. Established entry points may retain branding.
- Keep one source of truth per topic. Secondary documents should summarize and link to the owner instead of repeating mutable formulas, metrics, commands, or corpus data.
- Keep frontend-specific setup and firmware workflows in the local frontend README files; link to `API.md` and `DISCOVERY.md` for shared contracts.
- Make public compatibility, controller-support, privacy, and security claims only from repository evidence. Use a validation matrix when coverage is incomplete.

## Topic Owners

- Use `CLI.md`, `src/python/micro_espectre/README.md`, the relevant frontend README, and `./espectre --help` for CLI syntax and operator workflows.
- Use `SETUP.md` for installation, initial configuration, sensor placement, the first detection check, and the USB versus OTA workflow for signed published firmware; use `TROUBLESHOOTING.md` for connectivity, sensing diagnostics, and practical tuning.
- Use `SDK.md` for shared configuration and integration contracts, `CSI.md` for traffic sources and CSI acquisition, and `ARCHITECTURE.md` for code layout, layer responsibilities, and execution flow. Keep public messages, operations, external traffic markers, and stream formats in `API.md`, and discovery contracts in `DISCOVERY.md`.
- Use `ALGORITHMS.md`, `FEATURES.md`, `ML_DATA_COLLECTION.md`, and `ML_TRAINING.md` for detector behavior, feature inventory, collection, and training workflows.
- Use `performance/README.md`, `LITERATURE.md`, and `data/auto_generated/DATASET_QUALITY_CHECK.md` for benchmark status, external research, collection backlog, and dataset quality.
- Use `RELEASING.md` for maintainer procedures covering firmware signing, SDK packaging, and publication.
- Use `ROADMAP.md` for product outcomes, gates, and sequencing; `adr/*.md` for durable decisions; and `review/*.md` only for dated review context.

## Durable Records

- Keep `ROADMAP.md` at the outcome, gate, and sequencing level. Put experiment and collection details in their owning documents.
- Use ADRs for durable architectural or project-level decisions, including important rejected directions. Keep one decision per ADR, preserve its rationale, and update `Status`, `Supersedes`, and `Superseded by` consistently.
- Treat `FEATURES.md` as the feature experiment ledger. Record every seriously evaluated production, research, historical, planned, or rejected feature before removal or moving on.
- For measured features, retain the definition, physical interpretation, scale invariance, implementation scope, corpus, split, seed, primary and worst-group metrics, redundancy evidence, verdict, and reason. Mark unavailable evidence instead of reconstructing it.
- Treat `LITERATURE.md` as the external research ledger. Record the source URL, release date, hardware and signal assumptions, methods, results, and ESPectre transfer limits; exclude internal ESPectre research.
- Keep the active unreleased changelog focused on the final cumulative release state. Put superseded experiments in `FEATURES.md` or ADRs, and update only the latest active section unless correcting an explicitly requested fact.

## Generated Material

- Regenerate performance and dataset-quality reports through their owning tools instead of editing them manually. Use `--check-current` with `tools/generate_performance_report.py` for the aggregate performance report and `tools/validate_dataset_quality.py` for dataset-quality reports, following the input-scope requirements in [README.md](../tools/README.md).
- Per-chip firmware reports come from measured `tools/benchmark_firmware.py` runs. That tool has no `--check-current` mode; assess the report against its recorded run evidence and scope.
- Follow `docs/web/AGENTS.md` before changing the public website.
