# ESPectre Web Documentation Agent Rules

## Source And Generation

- Edit shared fragments under `content/`; do not directly edit generated `index.html` pages in public route directories.
- Run `.github/scripts/build_static_pages.py` after changing shared fragments, and inspect the generated diff only for the affected routes.
- Update `.github/scripts/sitemap.template.xml` when public routes change; `sitemap.xml` is a generated deployment artifact.
- Keep public compatibility, privacy, security, and controller-support claims grounded in repository evidence.

## Testing And Review

- Do not add tests that freeze marketing copy, headlines, captions, button labels, placeholders, helper text, or other reader-facing wording.
- Assert stable structure and behavior through selectors, attributes, routes, IDs, protocol values, and documented machine-consumed strings.
- Use targeted searches and bounded HTML ranges. Do not load generated pages or the full site index when a shared fragment or selector owns the change.
- When visual behavior changes, build the affected pages and perform a proportional visual check; do not broaden a copy-only task into a site-wide redesign.
