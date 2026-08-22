# Licensing

ESPectre first-party code is available under the open-source GPLv3 license, and eligible portions may also be licensed under a separate commercial agreement when GPLv3 obligations do not fit your product.

## GPLv3 (open source)

Unless a file carries a different SPDX license identifier or third-party notice, ESPectre first-party source code is released under the [GNU General Public License v3.0](LICENSE). You are free to use, study, modify, and redistribute that code, provided that firmware and applications that include it comply with GPLv3, including making the corresponding source available.

## Commercial license

Manufacturers and firmware teams that want to embed the ESPectre sensing engine (the shared `core` and `runtime` layers and the Native, Matter, or Streamer reference integrations) into proprietary firmware, without the source-disclosure obligations of GPLv3, can obtain a commercial license. The commercial license covers only material for which the maintainer can grant those rights; it does not replace third-party license terms, grant rights to third-party trademarks or media, or cover the GPL-only ESPHome C++ frontend.

For commercial licensing inquiries, contact our team at <contact@espectre.dev>.

## Integration services

Optional architecture review, firmware integration, validation, and tuning services may also be available under a separately scoped services agreement. These services are not included in a commercial license unless they are expressly included in the signed agreement.

For integration-service inquiries, contact our team at <contact@espectre.dev>.

## Contributions

Contributions are accepted so they can be distributed under both licensing tracks:

- Every commit must carry a DCO `Signed-off-by` trailer (`git commit -s`), certifying the origin of the change.
- Contributors sign the [CLA.md](CLA.md) once. The CLA grants the maintainer the rights needed to distribute contributions under both GPLv3 and the commercial license, while contributors retain ownership of their work.

## Third-party components

- The production `core`, `runtime`, Native, Matter, and Streamer firmware paths use components available under permissive terms, including Apache-2.0, MIT, BSD, ISC, zlib, CC0, and Unlicense terms. Major dependencies include ESP-IDF, esp-matter, Matter SDK, NimBLE, lwIP, mbedTLS, and FreeRTOS.
- The ESPHome frontend combines with the GPL-3.0-only ESPHome C++ runtime at build time, so that firmware frontend is available under GPLv3 only. The ESPHome Python build tooling is MIT-licensed.
- Host-side build, flashing, CLI, test, and research environments include separately installed GPL, LGPL, MPL, and EPL/EDL packages. They are not linked into the commercial firmware SDK. LGPL and MPL components retain their own redistribution obligations if a downstream product packages them, and `paho-mqtt` is consumed under its permissive Eclipse Distribution License option where GPL compatibility is required.
- `test/cpp/support/cnpy.*` is third-party MIT-licensed test support; its complete notice is stored in `test/cpp/support/LICENSE.cnpy`.
- Browser dependencies are staged with their upstream license files by `.github/scripts/stage_web_vendor.py`.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for repository-level attribution. Every published firmware build also ships an SPDX SBOM, a notice summary, and an archive containing the license files collected from the exact components used by that build. GitHub Releases group those per-build files into `firmware-compliance-<channel-or-version>.zip`, while the ESPectre website keeps them available next to the corresponding firmware image.
