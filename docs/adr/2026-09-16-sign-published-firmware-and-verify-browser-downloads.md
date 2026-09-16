# ADR: sign published firmware and verify browser downloads

- Status: Accepted
- Date: 2026-09-16
- Supersedes: None
- Superseded by: None

## Context

HTTPS authenticates the download endpoint but does not independently establish who authorized a firmware image. Native and ESPHome support OTA, while Matter currently supports only full-image USB installation. Local development and ESPHome Device Builder must remain usable without access to ESPectre release secrets.

## Decision

Official Native and ESPHome builds require software-only ESP-IDF application signatures during OTA. Publication CI enables this policy and signs every official channel. Local builds retain unsigned defaults. Non-publishing CI validates the signing configuration with ephemeral keys. The compiled configuration and resulting OTA and factory images are checked before artifacts become publication inputs.

Use the legacy ECDSA application-signature scheme on classic ESP32 to retain older chip revisions, and RSA-3072 on the other supported targets. Keep hardware Secure Boot, flash encryption, and hardware anti-rollback outside general-purpose builds. Those irreversible provisioning choices belong to an explicitly designed production manufacturing process.

Sign an immutable release inventory covering every frontend's artifacts, including Matter factory images. The browser checks the signed inventory and the downloaded bytes before erasing or writing flash. Production verification failures block installation. Localhost and `test.espectre.dev` run the same checks but allow a failed verification to be overridden through explicit confirmation for the current attempt, supporting unsigned development builds without a query parameter. Website staging may relocate artifacts and filter the displayed inventory, but it cannot authorize new bytes or re-sign downloads. The CLI has no new verification behavior.

Use GitHub Actions secrets for private release keys and an enrolled public-key registry in the repository. Key custody, provisioning, and rotation are owned by [CONTRIBUTING.md](../../CONTRIBUTING.md#firmware-signing-for-maintainers). The operator USB versus OTA workflow is owned by [SETUP.md](../SETUP.md#official-images-and-personal-builds). Browser verification is isolated and documented as an executable example in [README.md](../web/README.md#firmware-signature-verification).

## Consequences

- Official firmware accepts only OTA images signed by its trusted key. Switching to a personal build requires a USB installation, including the first personalized ESPHome Device Builder image after adoption.
- Initial OTA key rotation uses USB recovery. Catalog keys may overlap during a transition, but that does not add multi-key OTA support to the running firmware.
- Release custodians enroll production public keys in the checked-in registry. Missing or mismatched secrets and trust records block publication; unsigned legacy website channels are omitted until rebuilt.
- Browser verification assumes the website code and public-key registry remain trusted. It adds an artifact-authentication check but cannot protect against replacement of the verifier itself.
- Signing does not establish startup health or prevent replay of previously signed artifacts. Native rollback remains separate work; existing ESPHome rollback behavior must still be validated with signed images.

## Alternatives Considered

- Require signatures in all local builds: rejected because it would impose release-key infrastructure on ordinary development and personal ESPHome configurations.
- Enable hardware Secure Boot in public images: rejected because flashing a general-purpose image must not silently introduce irreversible eFuse policy or remove ordinary USB recovery.
- Rely only on browser verification: rejected for OTA-capable frontends because a compromised website can replace its own verifier. Device-side verification provides a separate trust boundary.
- Omit browser verification: rejected because authenticated USB downloads are useful across all frontends and provide a practical example of catalog-signature and firmware-hash verification.

## Validation

Host tests exercise real RSA and ECDSA signatures, altered catalogs, wrong keys, corrupt images, staging preservation, and rejection before erase or write. CI builds all supported Native and ESPHome targets and checks their effective configuration and signed artifacts. Production key provisioning and per-target hardware upgrade, interruption, recovery, and rollback evidence remain required before release, as documented in [CONTRIBUTING.md](../../CONTRIBUTING.md#firmware-signing-for-maintainers).
