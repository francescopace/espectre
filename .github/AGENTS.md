# ESPectre GitHub And CI Agent Rules

## Workflows And Contributions

- Keep workflow changes minimal, explicit, and grouped by purpose.
- Before changing an action version, inspect current usage with `rg "uses: .*@" .github`, and prefer pinned major versions already used by the repository.
- Keep the Python and `C++` coverage uploads and gates active.
- Keep linear-history enforcement on `main` and `develop`, and reject merge commits in pull requests before applying any bot DCO exemption.
- Contributions require a one-time CLA signature. Do not remove or weaken the CLA workflow.
