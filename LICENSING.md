# Licensing

ESPectre first-party code is available under GPLv3, with paid commercial licenses offered separately for eligible proprietary integrations.

## GPLv3 (open source)

Unless a file carries a different SPDX license identifier or third-party notice, ESPectre first-party source code is released under the [GNU General Public License v3.0](LICENSE), identified as `GPL-3.0-only`. You may use, study, modify, and redistribute it under that license.

If you distribute firmware or an application that incorporates or links to ESPectre as a single combined program, you must license the combined work as a whole under GPLv3 and make its complete Corresponding Source, including your integration code, available as required by the license. This also applies to firmware distributed inside a device.

GPLv3 permits commercial use and sales. Private use and internal modifications that are not conveyed to others do not require publishing source code or purchasing a commercial license. When you distribute covered software, you must provide source through one of the methods permitted by GPLv3; a public repository is not always required.

Separate, independent applications do not become covered by GPLv3 merely because they exchange messages with ESPectre over MQTT or HTTP. The relevant distinction is whether they are separate works or parts of a single combined program.

## Commercial license

To distribute an eligible integration under proprietary terms without GPLv3's copyleft obligations, you must purchase a separate commercial license before distribution. The signed agreement defines the covered material, permitted uses, and fees.

For commercial licensing inquiries, contact our team at <contact@espectre.dev>.

## Integration services

Optional architecture review, firmware integration, validation, and tuning services may also be available under a separately scoped services agreement. These services are not included in a commercial license unless they are expressly included in the signed agreement.

For integration-service inquiries, contact our team at <contact@espectre.dev>.

## Contributions

Contributions are accepted so they can be distributed under both licensing tracks:

- Every commit must carry a DCO `Signed-off-by` trailer (`git commit -s`), certifying the origin of the change.
- Contributors sign the [CLA](CLA.md) once. The CLA grants the maintainer the rights needed to distribute contributions under both GPLv3 and the commercial license, while contributors retain ownership of their work.

## Third-party components

A commercial license may cover eligible ESPectre first-party material, including the shared `core` and `runtime` layers and the `Native` or `Matter` frontends. It does not replace third-party license terms, grant rights to third-party trademarks or media, or cover the GPL-only `ESPHome` frontend.

See the [third-party notices](THIRD_PARTY_NOTICES.md) for the complete attribution and dependency record. Published firmware builds include an SPDX SBOM, a notice summary, and the license files for their exact components, grouped in `firmware-compliance-<channel-or-version>.zip` on GitHub Releases and alongside the corresponding image on the ESPectre website.
