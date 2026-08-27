# ESPectre GitHub And CI Agent Rules

## Workflows And Contributions

- Keep workflow changes minimal, explicit, and grouped by purpose. Use `develop` as the default PR target; `main` is release-only.
- Before changing an action version, inspect current usage with `rg "uses: .*@" .github`, and prefer pinned major versions already used by the repository.
- Keep the Python and `C++` coverage uploads and gates active.
- Do not bypass branch protection, push directly to `main`, force push, or merge with failing required checks.
- Contributions require a one-time CLA signature. Do not remove or weaken the CLA workflow.
- Do not comment, close, merge, label, push, or otherwise mutate GitHub state unless the user explicitly requests that action.
