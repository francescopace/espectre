# Security Policy

For responsible deployment guidance or to report suspected abuse of ESPectre or related Wi-Fi sensing, see [Security and responsible use](https://espectre.dev/security/). The process below is for product and project vulnerabilities.

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Open the [Security tab](https://github.com/francescopace/espectre/security) and choose "Report a vulnerability" to start a private report. If you cannot use GitHub Security Advisories, email <security@espectre.dev>.

Include a description, steps to reproduce, and potential impact. You can also suggest a fix.

Maintainers will acknowledge your report within 48 hours and provide an initial assessment within 7 days. Resolution depends on severity, typically taking 30-90 days. When a CVE is appropriate, maintainers can request one through the advisory or provide an existing identifier.

### Scope

Security issues relevant to ESPectre include:

- Wi-Fi/CSI data exposure
- MQTT authentication bypass
- ESPHome/Home Assistant integration vulnerabilities
- Firmware vulnerabilities on ESP32
- Vulnerable dependency versions or configurations distributed by ESPectre

### Out of Scope

- Vulnerabilities that exist only in an upstream dependency and do not affect any version or configuration distributed by ESPectre; report those to the upstream project
- Issues requiring physical access to the device
- Social engineering attacks

## Responsible Disclosure

When reporting a vulnerability:

- Give us reasonable time to fix the issue before public disclosure
- Avoid accessing or modifying other users' data
- Act in good faith to avoid privacy violations

## Firmware Signing

Official Native and ESPHome firmware accepts only OTA updates signed by its trusted key. Local builds are unsigned by default. The website verifies the signed catalog and each downloaded image before flashing Native, ESPHome, or Matter firmware. Verification failures block production installations; designated development hosts allow an explicit override for each attempt, as documented in [README.md](docs/web/README.md#firmware-signature-verification).

Matter has no OTA implementation or on-device signature enforcement. Signatures verify authorship and integrity; they do not guarantee successful startup or prevent replay of older signed firmware. Hardware Secure Boot, flash encryption, and hardware anti-rollback are disabled in these general-purpose builds.

See [SETUP.md](docs/SETUP.md#official-images-and-personal-builds) for USB versus OTA installation and switching to personal builds. Maintainer procedures for signing keys, rotation, recovery, and release validation are in [CONTRIBUTING.md](CONTRIBUTING.md#firmware-signing-for-maintainers).
