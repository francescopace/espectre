# ESPectre Web Documentation Agent Rules

## Source And Generation

- Edit shared fragments under `content/`; do not directly edit generated `index.html` pages in public route directories.
- Run `.venv/bin/python .github/scripts/build_static_pages.py` after changing shared fragments, and inspect the generated diff only for the affected routes.
- Update `routes.json` when public routes change; `sitemap.xml` is a generated deployment artifact.

## Testing And Review

- Follow [AGENTS.md](../../test/AGENTS.md#assertions-and-fixtures) for test assertions, including the distinction between reader-facing prose and documented text contracts.
- Do not hardcode route or page counts, duplicate a manifest inventory in a test, or freeze presentation order unless the count or order is itself a documented behavior. Derive completeness checks from the canonical registry or schema.
- Use targeted searches and bounded HTML ranges. Do not load generated pages or the full site index when a shared fragment or selector owns the change.
- When visual behavior changes, build the affected pages and perform a proportional visual check; do not broaden a copy-only task into a site-wide redesign.
