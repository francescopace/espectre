# ADR: persist per-device Matter onboarding data

- Status: Accepted
- Date: 2026-07-15

## Context

The Matter frontend previously depended on shared development commissioning values. That was sufficient for firmware development, but it did not provide a device-specific QR code or a consistent onboarding experience across the web flasher and repository CLI.

ESPectre also targets generic development boards flashed by end users. These boards do not pass through a controlled manufacturing line, and the public web flasher must not require a server-side secret or perform irreversible eFuse provisioning without an explicit manufacturing workflow.

## Decision

Generate Matter commissioning data locally on the device during the first boot and persist it in a dedicated `matter_factory` data partition.

Concretely:

- generate a valid random setup passcode, discriminator, and SPAKE2+ salt with the ESP32 hardware random source
- derive the SPAKE2+ verifier in the custom commissionable data provider
- preserve `matter_factory` during normal browser and CLI firmware flashes
- emit the QR payload and manual pairing code on serial at every boot
- make the web flasher and `./espectre matter qr` read the device-provided values instead of generating independent onboarding identities
- treat a complete flash erase as a new device initialization that generates a new onboarding identity

This is the development and community-firmware workflow. A production ESPectre device still requires per-device manufacturing data, including unique Device Attestation Credentials, and an associated QR-code distribution process.

## Alternatives Considered

### Keep shared Matter test commissioning values

Rejected. A shared passcode provides no per-device proof of possession and makes every ESPectre device expose the same onboarding code.

### Derive commissioning data from the device MAC address

Rejected. The MAC address is stable and unique enough for public identifiers, but it is not secret. An open derivation algorithm would make the setup passcode predictable for the entire device population.

### Provision an HMAC key in eFuse automatically

Rejected for the public browser workflow. It would preserve the same identity after a complete flash erase, but eFuse programming is irreversible, consumes a hardware key slot, and can conflict with secure boot, flash encryption, or other uses on generic user-owned boards.

### Require a manufacturing partition before flashing

Deferred to production hardware. It follows the standard Matter manufacturing model, but it is not available to users flashing arbitrary development boards from a static public website.

## Consequences

Benefits:

- each device gets an independent Matter onboarding identity
- browser and CLI workflows expose the same QR code
- rebuilds and normal firmware updates preserve onboarding data
- the public web flasher remains static and does not need to hold secrets
- generic boards are not modified through irreversible eFuse operations

Trade-offs:

- a complete flash erase generates a different QR code
- the setup passcode is retained in the local factory record so the device can reproduce its QR payload
- development firmware still uses development VID/PID and example attestation credentials until a production manufacturing pipeline is introduced
- users must retrieve the QR from the browser, CLI, or serial output because it is not printed on generic hardware

## Related

- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-one-message-model-and-command-engine-across-transports.md`](2026-07-02-use-one-message-model-and-command-engine-across-transports.md)
