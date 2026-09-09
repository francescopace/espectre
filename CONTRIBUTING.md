# Contributing

Submit contributions through pull requests against `develop`. This guide covers review expectations, validation, and contribution requirements. Installation and technical workflows live in the documents linked below.

## Getting started

1. Fork and clone the repository, then create a branch from `develop`.
2. Set up the repository environment using [CLI.md](docs/CLI.md#local-build-prerequisites). Install ML extras only when needed, following [ML_TRAINING.md](docs/ML_TRAINING.md#prerequisites).
3. Read the document and local README that own your change. [ARCHITECTURE.md](docs/ARCHITECTURE.md) maps the code and dependency boundaries; [SETUP.md](docs/SETUP.md) covers device installation and configuration.

[ROADMAP.md](docs/ROADMAP.md) describes current priorities. Discuss scope and implementation questions in the issue associated with the work or in [GitHub Discussions](https://github.com/francescopace/espectre/discussions).

## Making changes

Keep each pull request focused on one feature or fix. Match the surrounding code, write code and comments in English, and add tests for new behavior. Preserve existing license notices, including third-party notices. New first-party source files should follow neighboring headers, including `SPDX-License-Identifier: GPL-3.0-only` and the commercial-license notice. [LICENSING.md](LICENSING.md) describes the licensing tracks.

Use [SDK.md](docs/SDK.md) for public integration contracts, [CSI.md](docs/CSI.md) for acquisition and traffic, and [API.md](docs/API.md) for shared messages and operations. Prototype detector changes in Python before porting them to C++; [ALGORITHMS.md](docs/ALGORITHMS.md) and [ML_TRAINING.md](docs/ML_TRAINING.md#required-validation) describe the detector behavior and validation gates.

Update the owning documentation whenever behavior, configuration, or operator workflows change.

## Validation

Run the narrowest checks for the changed behavior first, then the required integration and parity gates. Include the commands and results in the pull request, and explain any check you could not run.

- C++ host tests and coverage: follow [README.md](test/cpp/README.md). Run the coverage workflow for C++ changes and investigate unexplained regressions in the affected layer. Coverage helpers use Bash; Windows contributors can use CMake/CTest for host tests or WSL/Git Bash for coverage.
- Python runtime, CLI, tools, and validation tests: use the repository virtual environment. The full suite and coverage commands are below.
- Website tests, generated pages, and local preview: follow [README.md](docs/web/README.md#tests).
- Detector and model changes: run the required checks in [ML_TRAINING.md](docs/ML_TRAINING.md#required-validation). [README.md](docs/performance/README.md) records measured results.

```bash
.venv/bin/pytest test/python -q --tb=short
./test/python/run_coverage.sh
```

Python test auto-parallelism is capped at four workers because replay-heavy tests slow down at higher process counts. Set `PYTEST_XDIST_AUTO_NUM_WORKERS` to a positive integer to override the cap. Firmware builds and device checks follow the selected frontend's README.

## Pull requests

Target `develop`; `main` is reserved for releases. Explain the problem, the resulting behavior, and how you tested it. Include the relevant documentation changes. All required CI checks must pass, and the pull request needs at least one review approval.

### Commits

Use Conventional Commits with an imperative, lower-case subject of at most 72 characters. Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, and `chore`. Use a signed-off commit, for example:

```bash
git commit -s -m "fix: correct calibration for edge cases"
```

### DCO and CLA

Every human-authored commit must include a valid `Signed-off-by` trailer under the Developer Certificate of Origin (DCO), enforced by CI. Dependabot-authored commits are exempt from sign-off; all commits must satisfy the linear-history check.

The project also requires a one-time [CLA.md](CLA.md) signature for distribution under both licensing tracks in [LICENSING.md](LICENSING.md). Add your GitHub login and signing date to `.github/cla-signatures.json` in your first pull request, following the CLA instructions. You retain ownership of your contribution.

If your latest commit lacks a sign-off, use `git commit --amend -s` before updating your pull request branch.

### Updating and merging

Keep pull requests free of merge commits, including Dependabot updates. Update a pull request with **Update with rebase** in GitHub, or run these commands from its source branch:

```bash
git fetch origin
git rebase origin/develop
git push --force-with-lease
```

Rewrite only your pull request branch. Never force-push `main` or `develop`. Integrate pull requests with **Rebase and merge**; merge commits and squash merges are disabled.

## Data contributions

Follow [ML_DATA_COLLECTION.md](docs/ML_DATA_COLLECTION.md#contributing-data) for priority labels, recording requirements, metadata, and quality checks. Add recordings under the documented dataset layout and describe the setup in your pull request. Collection and training have separate workflows; [ML_TRAINING.md](docs/ML_TRAINING.md) explains dataset roles and model promotion.

Read the privacy requirements in [ML_DATA_COLLECTION.md](docs/ML_DATA_COLLECTION.md#data-privacy) before recording or submitting data. DCO and CLA requirements also apply to data contributions.

## Documentation

Update the existing topic owner instead of copying details into another guide. [AGENTS.md](docs/AGENTS.md#topic-owners) defines document ownership; its style rules require concise technical English, the Oxford comma, and clear distinctions between deployed, experimental, and planned behavior. Check commands, file paths, and link targets in your changes.

For website changes, edit authored fragments and follow the generation and verification workflow in [README.md](docs/web/README.md). Generated reports must be regenerated through their owning tools.

## Issues and questions

Search [GitHub Issues](https://github.com/francescopace/espectre/issues) before opening a report. Check [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and the frontend README, and verify whether the problem persists on the latest `develop` build.

A bug report should include reproduction steps, expected and observed behavior, the ESPectre version, chip, and relevant configuration. Include the Home Assistant version when applicable. Remove credentials and unrelated personal information from configuration and logs. For feature requests, explain the use case, proposed behavior, and alternatives considered.

Use [GitHub Discussions](https://github.com/francescopace/espectre/discussions) for help and design questions. Participation follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); report unacceptable behavior to contact@espectre.dev.

Contributors are acknowledged in pull requests, significant contributions in release notes, and dataset contributions in the dataset documentation.
