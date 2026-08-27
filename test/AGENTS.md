# ESPectre Test Agent Rules

## Test Ownership

- Extend the existing test owner for the changed contract. Do not create a new regression module when an owning suite exists.
- Every maintained test must protect a current contract, a safety or correctness invariant, or a quantified performance or parity gate.
- Do not add tombstone tests for removed behavior. Negative tests belong only at supported runtime, protocol, persistence, security, or compatibility boundaries.
- Validate standalone research tools, one-off scripts, generated reports, build configuration, example configuration, and CI plumbing through their owning end-to-end workflows instead of adding unit tests without a maintained runtime contract.
- A production change should not require editing integration or performance gate code unless the public contract or gate deliberately changes.
- Before editing more than three test files for one logical production change, explain which distinct contracts require those edits. Shared implementation churn is not sufficient justification.

## Assertions And Fixtures

- Prefer assertions on state, return values, events, and side effects. Assert output text only when it is a documented user-facing or machine-consumed interface, and test stable semantics rather than incidental wording.
- Do not assert marketing copy, headlines, captions, button labels, placeholders, helper text, or other reader-facing website wording through snapshots, substrings, regular expressions, or raw source scans.
- Website and HTML tests must assert stable structure or behavior through selectors, attributes, routes, IDs, protocol values, and documented machine-consumed strings such as emails, CLI commands, and option values.
- Do not duplicate production constants, feature registries, schemas, or performance targets in tests. Parameterize the owner suite from the canonical source.
- Test public results after internal refactors unless an internal property is an explicit memory, timing, reset, safety, or compatibility invariant.

## Execution

- Run the narrowest owner test first with concise output. Use `-q --tb=short` for Python where appropriate, and rerun only failures verbosely.
- Keep the Python and `C++` coverage uploads and gates active.
- Tests that bind local UDP sockets must run outside the network sandbox. Treat `PermissionError` or `EPERM` during socket setup as a sandbox restriction, not evidence that the test should use another address.
- On failure, investigate the implementation root cause. Never skip, disable, or weaken a test to produce a passing result, and ask before changing a supported behavior expectation.
- Report every required check that did not run with its exact command and blocker.
