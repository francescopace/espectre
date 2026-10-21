# ESPectre Web Documentation Agent Rules

## Source And Generation

- Edit shared fragments under `content/`; do not directly edit generated `index.html` pages in public route directories.
- Run `.venv/bin/python .github/scripts/build_static_pages.py` after changing shared fragments, and inspect the generated diff only for the affected routes.
- Update `routes.json` when public routes change; `sitemap.xml` is a generated deployment artifact.

## Testing And Review

- Run `./test/web/run_tests.sh` after changing site scripts, fragments, or routes; pass test file paths to run a subset.
- Follow [AGENTS.md](../../test/AGENTS.md#assertions-and-fixtures) for test assertions, including the distinction between reader-facing prose and documented text contracts.
- Do not hardcode route or page counts, duplicate a manifest inventory in a test, or freeze presentation order unless the count or order is itself a documented behavior. Derive completeness checks from the canonical registry or schema.
- Use targeted searches and bounded HTML ranges. Do not load generated pages or the full site index when a shared fragment or selector owns the change.
- When visual behavior changes, build the affected pages and perform a proportional visual check; do not broaden a copy-only task into a site-wide redesign.
- Build visual changes from the palette, fonts, and components in `assets/css/styles.css`, the single source of truth for site styles. Do not introduce generic template styles the site does not use, such as cream or off-white page backgrounds, italic accent words in headlines, or numbered "01/02/03" section labels.
