# ESPectre Test Agent Rules

## Test Ownership

- Extend the existing test owner for the changed contract. Do not create a new regression module when an owning suite exists.
- Every maintained test must protect a current contract, a safety or correctness invariant, or a quantified performance or parity gate.
- Do not add tombstone tests for removed behavior.
- Validate standalone research tools, one-off scripts, generated reports, build configuration, example configuration, and CI plumbing through their owning end-to-end workflows instead of adding unit tests without a maintained runtime contract.
- A production change should not require editing integration or performance gate code unless the public contract or gate deliberately changes.
- Before editing more than three test files for one logical production change, explain which distinct contracts require those edits. Shared implementation churn is not sufficient justification.
- Keep the Python and `C++` coverage uploads and gates active.

## Assertions And Fixtures

- Prefer assertions on state, return values, events, and side effects. Assert output text only when it is a documented user-facing or machine-consumed interface, and test stable semantics rather than incidental wording.
- Do not freeze reader-facing website prose, including marketing copy, headlines, captions, labels, placeholders, and helper text, through snapshots, substrings, regular expressions, or source scans. Text assertions are appropriate only when the value is a documented behavioral, protocol, security, or machine-consumed contract, such as an email address, CLI command, or option value.
- Website and HTML tests should assert semantic structure or behavior through selectors, attributes, routes, IDs, accessibility relationships, and documented text contracts. Test link destinations rather than link labels; a wording-only change outside a documented contract must not fail a test.
- Do not duplicate production constants, feature registries, schemas, or performance targets in tests. Parameterize the owner suite from the canonical source.
- Test public results after internal refactors unless an internal property is an explicit memory, timing, reset, safety, or compatibility invariant.
- Do not use the absence of an obsolete string, class name, function name, or source snippet as proof that a refactor is complete. Assert the resulting behavior or stable structure; reserve negative assertions for supported runtime, protocol, persistence, security, or compatibility boundaries.
