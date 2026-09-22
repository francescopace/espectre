# Security policy

This page is for reporting security vulnerabilities in ESPectre. To report misuse of ESPectre or Wi-Fi sensing, or for guidance on responsible use, see [Security and responsible use](https://espectre.dev/security/).

## Reporting a vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Open the [Security tab](https://github.com/francescopace/espectre/security) and choose "Report a vulnerability" for a private report. If you cannot use it, email <security@espectre.dev>.

Describe the problem, how to reproduce it, and its possible impact. A suggested fix is welcome.

We confirm receipt within 48 hours and send a first assessment within 7 days. A fix usually takes 30–90 days, depending on severity. When appropriate, we request a CVE through the advisory or use an existing one.

### Scope

Security issues relevant to ESPectre include:

- Wi-Fi/CSI data exposure
- MQTT authentication bypass
- ESPHome/Home Assistant integration vulnerabilities
- Firmware vulnerabilities on ESP32
- Vulnerable dependency versions or configurations distributed by ESPectre

### Out of scope

- Vulnerabilities only in an upstream dependency that do not affect anything ESPectre ships; report those upstream
- Issues requiring physical access to the device
- Social engineering attacks

## Responsible disclosure

When reporting a vulnerability:

- Give us reasonable time to fix the issue before public disclosure
- Avoid accessing or modifying other users' data
- Act in good faith to avoid privacy violations

## Firmware signing

- Official Native and ESPHome firmware accepts only OTA updates signed with the project key. Local builds are unsigned.
- The website checks the signed catalog and every download before flashing Native, ESPHome, or Matter. A failed check blocks installation; only development hosts allow a one-time override (see [firmware signature verification](docs/web/README.md#firmware-signature-verification)).
- Matter has no OTA and does not check signatures on the device.
- Signatures prove who built the firmware and that it was not altered. They do not guarantee it starts, and do not prevent reinstalling an older signed version.
- Hardware Secure Boot, flash encryption, and anti-rollback are off in these general-purpose builds.

For switching between official and personal builds, see [official images and personal builds](docs/SETUP.md#official-images-and-personal-builds). Key management is in [firmware signing](docs/RELEASING.md#firmware-signing).
