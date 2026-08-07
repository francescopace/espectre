# Licensing

ESPectre is dual-licensed. You can use it under the open-source GPLv3 license,
or under a separate commercial license when GPLv3 obligations do not fit your
product.

## GPLv3 (open source)

Except for the Apache-2.0 files listed below, all source code in this
repository is released under the
[GNU General Public License v3.0](LICENSE). You are free to use, study,
modify, and redistribute ESPectre, provided that firmware and applications
distributed with ESPectre inside comply with GPLv3, including making the
corresponding source available.

## Apache-2.0 files

The web integration client is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) so any web
application, including proprietary ones, can embed it to talk to ESPectre
devices over the documented BLE protocol:

- `docs/web/assets/js/espectre-ble.js` — the client
- `test/web/test_espectre_ble.mjs` — its unit tests

Each file carries its own license header. Apache-2.0 is GPLv3-compatible, so
the GPLv3 website in this repository consumes the client without restriction.

## Commercial license

Manufacturers and firmware teams that want to embed the ESPectre sensing
engine (the shared `core` and `runtime` layers and their reference
integrations) into proprietary firmware, without the source-disclosure
obligations of GPLv3, can obtain a commercial license from the maintainer.

For commercial licensing inquiries, contact
Francesco Pace <francesco.pace@gmail.com>.

For the technical shape of an embedded integration, see
[EMBEDDING.md](docs/EMBEDDING.md).

## Contributions

Contributions are accepted so they can be distributed under both licensing
tracks:

- Every commit must carry a DCO `Signed-off-by` trailer (`git commit -s`),
  certifying the origin of the change.
- Contributors sign the [CLA.md](CLA.md) once. The CLA grants the maintainer
  the rights needed to distribute contributions under both GPLv3 and the
  commercial license, while contributors retain ownership of their work.

## Third-party components

- `core`, `runtime`, and the native, Matter, and streamer frontends depend
  only on permissively licensed components (Apache-2.0, MIT, or BSD), such as
  ESP-IDF, esp-matter, NimBLE, and ArduinoJson.
- The ESPHome frontend combines with GPLv3 ESPHome components at build time,
  so that frontend is available under GPLv3 only.
