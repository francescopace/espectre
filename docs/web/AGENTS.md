# ESPectre Web Documentation Agent Rules

## Source And Generation

- Edit shared fragments under `content/`; do not directly edit generated `index.html` pages in public route directories.
- Run `.github/scripts/build_static_pages.py` after changing shared fragments, and inspect the generated diff only for the affected routes.
- Update `routes.json` when public routes change; `sitemap.xml` is a generated deployment artifact.
- Keep public compatibility, privacy, security, and controller-support claims grounded in repository evidence.

## Testing And Review

- Never assert reader-facing wording, either positively or negatively. This includes exact strings and regular expressions for headings, paragraphs, captions, card text, button labels, placeholders, helper text, marketing copy, and the absence of phrases. If a wording-only edit can fail a test, the test is invalid.
- Do not use the absence of an obsolete string, class name, function name, or source snippet as proof that a refactor is complete. A negative assertion is valid only when the forbidden value is itself a contract or security boundary; otherwise, test the resulting behavior or stable structure.
- Do not hardcode route or page counts, duplicate a manifest inventory in a test, or freeze presentation order unless the count or order is itself a documented behavior. Derive completeness checks from the canonical registry or schema.
- Assert the contract behind the interface through semantic structure, selectors, attributes, routes, IDs, accessibility relationships, protocol values, and documented machine-consumed strings. For example, test that a link targets `/sdk/http-api/`, not that it says `HTTP API`.
- Before adding a content-related assertion, identify the user-visible failure it detects. Omit the assertion when the only failure is different prose with unchanged structure and behavior.
- Use targeted searches and bounded HTML ranges. Do not load generated pages or the full site index when a shared fragment or selector owns the change.
- When visual behavior changes, build the affected pages and perform a proportional visual check; do not broaden a copy-only task into a site-wide redesign.
