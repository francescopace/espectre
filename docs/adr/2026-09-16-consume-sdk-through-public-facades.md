# ADR: consume the SDK through public facades

- Status: Accepted
- Date: 2026-09-16

## Context

The first-party frontends used the public runtime controller for sensing, but also depended on undocumented runtime services and SDK sources in the same checkout. ESPHome read the C++ sensing schema during Python module import. These dependencies made it difficult to validate the SDK outside the repository.

## Decision

Keep the existing sensing and core-only SDK facades. Expose the existing ESP-IDF integration services through the optional `espectre_services_sdk.h`, with documented public methods and configuration types. Service ownership and allocation remain unchanged, and detector implementations stay outside the sensing and services facades. Put the MQTT implementation in a separate `espectre_mqtt_sdk.h` so firmware using other services does not need MQTT headers.

Require first-party frontends to use public SDK headers. Keep frontend source lists separate from the distributed SDK source lists, and support `ESPECTRE_SDK_ROOT` to select the `src/cpp` directory of an extracted SDK bundle. Frontends can continue to compile the SDK through these source lists without a binary SDK.

Move only `ImprovSerialService` to shared frontend support. Native and Matter own the Improv dependency. The SDK retains the credential store and provisioning service, which apply configuration independently of the onboarding protocol.

Generate ESPHome's Python validation constants from the canonical public SDK headers. Validate the generated artifact in the contract checks and compare its fingerprint with the selected SDK during CMake configuration.

## Alternatives

- Expanding the default sensing facade would impose optional platform headers on minimal integrations.
- Replacing all concrete services with factories would change allocation and lifetime behavior. Exposing the public API does not require factories.
- Copying transport and provisioning implementations into each frontend would duplicate the shared behavior and weaken protocol parity.

## Consequences

The optional service interfaces become maintained integration contracts. Their private members and implementation dependencies are not independent extension points. SDK reference generation, header maps, and source-compatibility checks cover the additional facade.

The dependency check detects frontend use of private SDK headers, including those shipped in the bundle. Frontend builds against an extracted SDK detect dependencies on the repository layout. See [SDK.md](../SDK.md#first-party-sdk-consumers) for the validation workflow.
