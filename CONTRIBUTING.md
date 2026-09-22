# Contributing

Contributions come in as pull requests against `develop`. This guide explains what we expect, how to test, and the sign-off requirements.

## Getting started

1. Fork and clone the repository, then create a branch from `develop`.
2. Set up the environment with the [local build prerequisites](docs/CLI.md#local-build-prerequisites). Add the [ML extras](docs/ML_TRAINING.md#prerequisites) only if you need them.
3. Read the documentation for the part you are changing. The [architecture overview](docs/ARCHITECTURE.md) shows how the code is organized; the [setup guide](docs/SETUP.md) shows how to install a device.

The [roadmap](docs/ROADMAP.md) shows current priorities. Discuss scope and design in the related issue or in [GitHub Discussions](https://github.com/francescopace/espectre/discussions).

## Making changes

- Keep each pull request to one feature or fix.
- Match the surrounding code, write code and comments in English, and add tests for new behavior.
- Keep existing license notices. New source files copy the header of their neighbors, including `SPDX-License-Identifier: GPL-3.0-only` and the commercial-license notice (see the [licensing terms](LICENSING.md)).
- Try detector changes in Python first, then port them to C++. See the [algorithms reference](docs/ALGORITHMS.md) and [required validation](docs/ML_TRAINING.md#required-validation).
- When behavior, settings, or workflows change, update the document that covers them.

Useful references: the [SDK guide](docs/SDK.md) for the public API, the [CSI guide](docs/CSI.md) for capture and traffic, and the [API reference](docs/API.md) for the device protocol.

## Validation

Run the checks closest to your change first, then the integration and parity checks it requires. In the pull request, list the commands and results, and say which checks you could not run and why.

- **C++:** follow the [C++ test guide](test/cpp/README.md), including coverage, and explain any coverage drop. The coverage scripts need Bash; on Windows, use CMake/CTest for tests and WSL or Git Bash for coverage.
- **Python** (runtime, CLI, tools): use the repository virtual environment and the commands below.
- **Website:** follow the [website tests](docs/web/README.md#tests).
- **Detector and model:** run the [required validation](docs/ML_TRAINING.md#required-validation).

```bash
.venv/bin/pytest test/python -q --tb=short
./test/python/run_coverage.sh
```

Python tests run on at most four workers, because the replay tests get slower with more. Set `PYTEST_XDIST_AUTO_NUM_WORKERS` to change that. For firmware builds and device checks, follow the frontend's guide.

### SDK validation

- `test/cpp/suites/runtime/test_sdk_surface.cpp` checks that everything documented is reachable through `espectre_sdk.h` and that the defaults match.
- `test/python/contracts/test_sdk_surface_invariants.py` checks the generated ESPHome schema, blocks frontends from using private SDK headers, and checks that every public header is documented in the API reference and in [public headers](docs/SDK.md#public-headers).
- After changing the sensing schema, run `.venv/bin/python .github/scripts/generate_esphome_schema.py` (`--check` only verifies).
- To check the packaged component and example, follow [package and validate](docs/RELEASING.md#package-and-validate).

## Pull requests

Open pull requests against `develop`; `main` is only for releases. Explain the problem, what changes, and how you tested it, and include the documentation updates. All required CI checks must pass, and at least one reviewer must approve.

### Commits

Use Conventional Commits with an imperative, lower-case subject of at most 72 characters. Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, and `chore`. Use a signed-off commit, for example:

```bash
git commit -s -m "fix: correct calibration for edge cases"
```

### DCO and CLA

- Every commit you write needs a `Signed-off-by` line (Developer Certificate of Origin, or DCO); CI checks it. Forgot it on the last commit? Run `git commit --amend -s`. Dependabot commits are exempt, but every commit must keep a linear history.
- Sign the [CLA](CLA.md) once, so your work can be distributed under both licenses (see the [licensing terms](LICENSING.md)): add your GitHub login and the date to `.github/cla-signatures.json` in your first pull request. You keep ownership of your contribution.

### Updating and merging

Keep pull requests free of merge commits, including Dependabot updates. Update a pull request with **Update with rebase** in GitHub, or run these commands from its source branch:

```bash
git fetch origin
git rebase origin/develop
git push --force-with-lease
```

Only rewrite your own pull request branch; never force-push `main` or `develop`. Pull requests are merged with **Rebase and merge** (merge commits and squash merges are disabled).

## Data contributions

Follow [contributing data](docs/ML_DATA_COLLECTION.md#contributing-data) for what to record, catalog fields, and checks, and describe your setup in the pull request. Read [data privacy](docs/ML_DATA_COLLECTION.md#data-privacy) before recording. The DCO and CLA apply to data too.

## Documentation

Each topic has one document; update it instead of copying details elsewhere. The [documentation rules](docs/AGENTS.md#topic-owners) list which document covers what, and ask for short, plain sentences, the Oxford comma, and a clear line between what ships, what is experimental, and what is planned. Check commands, paths, and links in your changes.

For the website, edit the source fragments and follow the [website guide](docs/web/README.md). Regenerate reports with their tools; never edit them by hand.

## Issues and questions

Before opening an issue, search [GitHub Issues](https://github.com/francescopace/espectre/issues), check the [troubleshooting guide](docs/TROUBLESHOOTING.md) and the frontend guide, and try the latest `develop` build.

- **Bug reports:** steps to reproduce, what you expected and what happened, ESPectre version, chip, configuration, and Home Assistant version if relevant. Remove passwords and personal data from configuration and logs.
- **Feature requests:** the use case, the behavior you propose, and the alternatives you considered.

Ask for help and discuss designs in [GitHub Discussions](https://github.com/francescopace/espectre/discussions). Everyone follows the [Code of Conduct](CODE_OF_CONDUCT.md); report problems to contact@espectre.dev.

Contributors are credited in pull requests, major contributions in release notes, and data contributions in the dataset documentation.

## Publishing releases

Firmware signing, key management, SDK packaging, and publication are in the [release guide](docs/RELEASING.md).
