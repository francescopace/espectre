# ADR: sign published firmware and verify browser downloads

- Status: Accepted
- Date: 2026-09-16

## Context

HTTPS proves where a download came from, not who approved the firmware. Native and ESPHome support OTA; Matter only installs full images over USB. Local development and ESPHome Device Builder must keep working without access to release secrets.

## Decision

**Device side.** Official Native and ESPHome builds accept OTA images only with a valid ESP-IDF application signature, checked in software. Publication CI turns this on and signs every official channel. Local builds stay unsigned by default, and non-publishing CI checks the signing setup with throwaway keys. The compiled configuration and the resulting OTA and factory images are checked before they can be published. Classic ESP32 uses the legacy ECDSA scheme so older chip revisions keep working; the other targets use RSA-3072. Hardware Secure Boot, flash encryption, and hardware anti-rollback stay out of general-purpose builds, because those irreversible choices belong to a designed manufacturing process.

**Browser side.** Each release has a signed, immutable inventory covering every frontend's artifacts, Matter factory images included. The web flasher checks the inventory signature and the downloaded bytes before it erases or writes flash, and a failure blocks installation in production. On localhost and `test.espectre.dev` the same checks run, but the user can explicitly override a failure for that attempt, so unsigned development builds still work. Website staging can move artifacts and filter what is shown, but it cannot authorize new bytes or re-sign anything. The CLI does not change.

Private keys live in GitHub Actions secrets, and public keys in a registry checked into the repository. Key custody and rotation are in [RELEASING.md](../RELEASING.md#firmware-signing), the USB versus OTA workflow in [SETUP.md](../SETUP.md#official-images-and-personal-builds), and the browser verifier in the [website README](../web/README.md#firmware-signature-verification).

## Alternatives considered

- **Require signatures in every local build.** Puts release-key infrastructure on ordinary development and personal ESPHome configs.
- **Enable hardware Secure Boot in public images.** Flashing a general-purpose image must not silently burn eFuses or remove normal USB recovery.
- **Browser verification only.** A compromised website could replace its own verifier; checking on the device adds a separate trust boundary.
- **No browser verification.** Authenticated USB downloads help on every frontend and show catalog-signature and hash checking in practice.

## Consequences

Official firmware only accepts OTA images signed with its trusted key. Moving to a personal build, including the first personalized ESPHome Device Builder image after adoption, needs a USB install. The first OTA key rotation also goes through USB; catalog keys can overlap during a transition, but the firmware itself trusts one key.

Release custodians enroll production public keys in the registry. Missing or mismatched secrets or trust records block publication, and unsigned old website channels are hidden until rebuilt.

Browser verification trusts the website code and the key registry; it cannot protect against someone replacing the verifier. Signing also says nothing about startup health and does not stop replay of an older signed image. Native rollback is separate work, and ESPHome rollback still has to be validated with signed images.

## Validation

Host tests cover real RSA and ECDSA signatures, altered catalogs, wrong keys, corrupt images, staging, and rejection before erase or write. CI builds every Native and ESPHome target and checks their configuration and signed artifacts. Production key setup and per-target hardware evidence for upgrade, interruption, recovery, and rollback are still required before release, as listed in [RELEASING.md](../RELEASING.md#firmware-signing).

## Related

- Commit `31e807c1`
