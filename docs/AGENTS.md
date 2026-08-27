# ESPectre Documentation Agent Rules

## Style And Ownership

- Use clear, concise, technical English and a neutral tone except in product-facing entry points. Prefer bullets or tables only when they improve readability.
- Do not hard-wrap prose. Keep each paragraph and list item on one source line unless Markdown syntax requires a line break.
- Use the Oxford comma, simple descriptive titles, filename-only text for internal links, and rare, purposeful emoji. Established entry points may retain branding.
- Keep one source of truth per topic. Secondary documents should summarize and link to the owner instead of repeating mutable formulas, metrics, commands, or corpus data.
- Keep frontend-specific workflows, protocols, and firmware surfaces in the local frontend README files.
- Verify current-state documentation against implementation, runtime schemas, and generated artifacts, and distinguish deployed, partial, and target behavior.
- Make public compatibility, controller-support, privacy, and security claims only from repository evidence. Use a validation matrix when coverage is incomplete.

## Topic Owners

- Use `CLI.md`, `src/python/micro_espectre/README.md`, the relevant frontend README, and `./espectre --help` for CLI syntax and operator workflows.
- Use `SETUP.md`, `ESPECTRE_PROTOCOL.md`, and `ARCHITECTURE.md` for shared configuration, protocol, and runtime architecture.
- Use `ALGORITHMS.md`, `FEATURES.md`, `ML_DATA_COLLECTION.md`, and `ML_TRAINING.md` for detector behavior, feature inventory, collection, and training workflows.
- Use `performance/README.md`, `LITERATURE.md`, and `data/auto_generated/DATASET_QUALITY_CHECK.md` for benchmark status, external research, collection backlog, and dataset quality.
- Use `ROADMAP.md` for product outcomes, gates, and sequencing; `adr/*.md` for durable decisions; and `review/*.md` only for dated review context.
- Large owner documents are targeted references. Search for the relevant heading or identifier, and do not read them in full unless the task truly spans the whole ledger.

## Durable Records

- Keep `ROADMAP.md` at the outcome, gate, and sequencing level. Put experiment and collection details in their owning documents.
- Use ADRs for durable architectural or project-level decisions, including important rejected directions. Keep one decision per ADR, preserve its rationale, and update `Status`, `Supersedes`, and `Superseded by` consistently.
- Treat `FEATURES.md` as the feature experiment ledger. Record every seriously evaluated production, research, historical, planned, or rejected feature before removal or moving on.
- For measured features, retain the definition, physical interpretation, scale invariance, implementation scope, corpus, split, seed, primary and worst-group metrics, redundancy evidence, verdict, and reason. Mark unavailable evidence instead of reconstructing it.
- Treat `LITERATURE.md` as the external research ledger. Record the source URL, release date, hardware and signal assumptions, methods, results, and ESPectre transfer limits; exclude internal ESPectre research.
- Keep the active unreleased changelog focused on the final cumulative release state. Put superseded experiments in `FEATURES.md` or ADRs, and update only the latest active section unless correcting an explicitly requested fact.

## Generated Material

- Do not edit generated performance or dataset-quality reports manually. Regenerate them from the current corpus, and run the generator's `--check-current` mode before calling them current.
- Follow `docs/web/AGENTS.md` before changing the public website.
