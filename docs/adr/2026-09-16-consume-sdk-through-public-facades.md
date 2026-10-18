# ADR: consume the SDK through public facades

- Status: Accepted
- Date: 2026-09-16
- Updated: 2026-09-23

## Context

Our own frontends used the public runtime controller for sensing, but also reached into undocumented runtime services and SDK sources in the same checkout. ESPHome even read the C++ sensing schema while its Python module loaded. That made it hard to check that the SDK worked outside the repository.

## Decision

Frontends use only public SDK headers:

- `espectre_sdk.h` for sensing and `espectre_core_sdk.h` for core-only use, as before;
- `espectre_services_sdk.h`, optional, for the ESP-IDF integration services, with documented methods and config types. Ownership and allocation do not change, and detectors stay out of the sensing and services facades;
- `espectre_mqtt_sdk.h` for MQTT, so firmware that uses other services does not pull in MQTT headers;
- `espectre_protocol_sdk.h` for messages, JSON, diagnostics, and transports.

Frontend source lists are kept apart from the SDK's own lists, and `ESPECTRE_SDK_ROOT` can point at the `src/cpp` directory of an extracted SDK bundle. Frontends can still compile the SDK from source without a binary build.

Only `ImprovSerialService` moves to shared frontend code, and Native and Matter own the Improv dependency. The SDK keeps the credential store and the provisioning service, which apply configuration whatever onboarding protocol is used.

ESPHome's Python validation constants are generated from the public SDK headers. The contract checks validate the generated file, and CMake compares its fingerprint with the selected SDK at configure time.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-09-16 | Public services and MQTT facades; frontends limited to public headers | Accepted |
| 2026-09-23 | Protocol surface in its own facade | Accepted |

## Alternatives considered

- **Grow the default sensing facade.** Minimal integrations would have to include optional platform headers.
- **Replace concrete services with factories.** Changes allocation and lifetimes, and is not needed to expose a public API.
- **Copy transport and provisioning code into each frontend.** Duplicates shared behavior and weakens protocol parity.

## Consequences

The optional service interfaces become maintained integration contracts; their private members and internals are not extension points. Reference generation checks that the public headers render, source-compatibility checks cover every facade, and the registry README introduces them and links to the reference for the packaged version.

The dependency check flags frontend use of private SDK headers, including those shipped in the bundle, and building against an extracted SDK catches any reliance on the repository layout. The workflow is in the [CLI guide](../CLI.md#building-against-an-sdk-bundle).

## Related

- [SDK guide](../SDK.md)
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- Commits `77dbd20b` (services and MQTT facades), `7757e54a` (extracted SDK bundles), `ba1b3d34` (protocol facade)
