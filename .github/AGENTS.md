# ESPectre GitHub And CI Agent Rules

## Workflows And Contributions

- Before changing an action version, inspect current usage with `rg "uses: .*@" .github`. Pin external actions to a full commit SHA followed by a `# vN` comment, as every existing workflow does.
- Keep the Python and `C++` coverage uploads and gates active.
- Keep linear-history enforcement on `main` and `develop`, and reject merge commits in pull requests before applying any bot DCO exemption.
- Contributions require a one-time CLA signature. Do not remove or weaken the CLA workflow.
- In `.github/cla-signatures.json`, `cla.sha` and the older `claSha` entries identify the Git blob of `CLA.md` recorded at signing. Preserve these historical values when the document changes. New signers record their login and signing date as described in [How to sign](../CLA.md#how-to-sign). CI checks the pull request author’s login.
