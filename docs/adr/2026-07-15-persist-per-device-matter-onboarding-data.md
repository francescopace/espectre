# ADR: persist per-device Matter onboarding data

- Status: Accepted
- Date: 2026-07-15

## Context

The Matter frontend used the shared development commissioning values. Fine for development, but every device had the same QR code, and the web flasher and CLI had no consistent onboarding flow.

ESPectre runs on generic boards that users flash themselves. There is no manufacturing line, and the public web flasher cannot hold a server-side secret or burn eFuses without an explicit manufacturing step.

## Decision

On first boot the device generates its own Matter commissioning data and stores it in a dedicated `matter_factory` partition:

- setup passcode, discriminator, and SPAKE2+ salt come from the hardware random source;
- the SPAKE2+ verifier is derived in our commissionable data provider;
- normal browser and CLI flashes leave `matter_factory` untouched;
- the QR payload and manual pairing code are printed on serial at every boot;
- the web flasher and `./espectre matter qr` read these values from the device instead of generating their own;
- a full flash erase counts as a new device and produces a new identity.

This covers development and community firmware. A real ESPectre product still needs per-device manufacturing data, including unique Device Attestation Credentials, and a way to ship the QR code.

## Alternatives considered

- **Keep the shared test values.** Every device would expose the same code, with no per-device proof of possession.
- **Derive the data from the MAC.** The MAC is not secret, and an open algorithm would make every passcode predictable.
- **Burn an HMAC key into eFuse.** It would survive a full erase, but eFuse writes are irreversible, use up a key slot, and can clash with secure boot or flash encryption on a user's board. Rejected for the browser flow.
- **Require a manufacturing partition before flashing.** That is the standard Matter model, but users flashing a dev board from a static website cannot produce one. Deferred to production hardware.

## Consequences

Each device has its own onboarding identity, the browser and CLI show the same QR code, and updates keep it. The web flasher stays static and holds no secrets, and no board gets irreversible eFuse changes.

On the other side: a full erase changes the QR code, the passcode is kept in the local factory record so the device can rebuild its QR payload, development firmware still uses development VID/PID and example attestation credentials, and users have to get the QR from the browser, CLI, or serial because nothing is printed on the board.

## Related

- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md)
- Commit `4b0105e0`
