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

## Firmware signing for maintainers

These procedures apply to maintainers who publish official firmware. Local development builds do not need release keys. [SECURITY.md](SECURITY.md#firmware-signing) summarizes what signature verification covers.

### Initial key setup

Publishing requires both GitHub Actions signing secrets to match enrolled public keys. Use this procedure to establish a new signing identity. Never use the temporary keys generated by pull-request CI for a release.

Generate the keys on a trusted workstation, in a protected directory outside the checkout. The following commands assume the directory already exists and is accessible only to its owner:

```bash
umask 077
SIGNING_DIR=/absolute/path/outside/the/repository
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$SIGNING_DIR/firmware-rsa.pem"
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:prime256v1 -out "$SIGNING_DIR/firmware-esp32.pem"
openssl pkey -in "$SIGNING_DIR/firmware-rsa.pem" -pubout -out "$SIGNING_DIR/firmware-rsa-public.pem"
openssl pkey -in "$SIGNING_DIR/firmware-esp32.pem" -pubout -out "$SIGNING_DIR/firmware-esp32-public.pem"
.venv/bin/python .github/scripts/firmware_signing.py --enroll "$SIGNING_DIR/firmware-rsa-public.pem"
.venv/bin/python .github/scripts/firmware_signing.py --enroll "$SIGNING_DIR/firmware-esp32-public.pem"
gh secret set FIRMWARE_SIGNING_KEY_RSA < "$SIGNING_DIR/firmware-rsa.pem"
gh secret set FIRMWARE_SIGNING_KEY_ESP32 < "$SIGNING_DIR/firmware-esp32.pem"
```

Commit only [firmware-signing-keys.json](docs/web/assets/firmware-signing-keys.json), after independently checking the printed public-key fingerprints. The RSA-3072 key signs C3, C5, C6, S2, and S3 OTA images and the domain-tagged release catalog. The ECDSA P-256 key signs classic ESP32 applications using the v1 format to support older chip revisions. Keep an encrypted backup of both private keys under maintainer control, separate from GitHub, and record the custodians and recovery location in the project's private operational records. Never place private keys in issues, build logs, caches, firmware artifacts, or the repository.

Protect `main`, `develop`, release tags, and changes to workflows, signing scripts, and the public-key registry. Only maintainers authorized to issue firmware should be able to change code that runs with the signing secrets. CI exposes release keys only on publication-eligible pushes and tagged releases; other builds use disposable test keys. Temporary key files are removed when the build wrapper exits and are outside toolchain cache and upload paths. A compromised signing job can authorize malicious firmware; GitHub Secrets is part of the release trust boundary.

### Rotation and recovery

Catalog verification accepts every explicitly enrolled RSA public key. Add a replacement public key before switching the secret, and retain old public keys while their signed channels remain available. Removing an old public key makes those catalogs unverifiable.

OTA key rotation requires USB installation. The v2 software verifier trusts the running application's first signature block; adding another signature is not a supported rotation procedure. The classic ESP32 verifier embeds its verification key in the app. After a loss or compromise, provision replacement secrets, publish newly signed full images, and install them via USB. Keep the old signing key only for explicitly authorized migration work; continued acceptance of compromised keys does not restore trust.

### Release validation

Publication CI checks the effective signing configuration, verifies each application signature, and checks that the full USB image contains the same signed application. Before release, validate signed upgrades, wrong-key and corrupt-image rejection, interruptions, startup recovery, and USB migration on each supported target. Native has two OTA slots and a manifest-version check, but no bootloader rollback or startup health-confirmation policy. ESPHome uses its upstream rollback behavior. See [SETUP.md](docs/SETUP.md#official-images-and-personal-builds) for establishing the initial trust chain through USB.

Hardware Secure Boot, flash encryption, and hardware anti-rollback require a separate manufacturing policy covering target-specific key custody, eFuse sequencing, and recovery. General-purpose builds leave these features disabled; the decision is recorded in [2026-09-16-sign-published-firmware-and-verify-browser-downloads.md](docs/adr/2026-09-16-sign-published-firmware-and-verify-browser-downloads.md).
