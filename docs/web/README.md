# Website

The website `espectre.dev` is published through GitHub Pages. Each public page is also generated as a static page so search engines can index it; a single-page app (SPA) shell runs the browser tools and in-app navigation.

## Local preview

From the repository root:

```bash
python -m http.server 8090 --directory docs/web
```

Then open `http://localhost:8090`.

- **Firmware verification** always runs. On `localhost`, `127.0.0.1`, `[::1]`, and `test.espectre.dev`, a failed check asks you to confirm before installing; in production it blocks installation.
- **Local device access:** Native firmware accepts a local preview only when built with `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y`. Published firmware never does.
- **Browsers:** Flash, Improv Serial, and the Matter QR reader need a Chromium-based browser. The hosted Direct workflow is tested with Chrome 151 or later on macOS; Windows and Linux are not tested on hardware yet, and discovery depends on the system's mDNS support. A local preview does not prove that the hosted site works.

Before a local preview can flash a board, stage the browser dependencies (see [Browser dependencies](#browser-dependencies)).

## Sources and generated pages

Edit only the sources:

| Path | Contents |
|------|----------|
| `content/` | Page fragments |
| `assets/css/`, `assets/images/`, `assets/js/` | Styles, images, and scripts |
| `routes.json` | Public pages (`routes`), their Analytics groups (`contentGroups`), and the SDK channel pages (`sdkChannels`) |

Never edit the generated `index.html` files. Generate them before testing page URLs directly:

```bash
python3 .github/scripts/build_static_pages.py
```

The generator adds each page's title, description, canonical URL, and social metadata from `routes.json`. The SPA reads the same manifest (through `assets/js/route-registry.js`) and fragments, and updates the metadata when you navigate. All public routes go into the sitemap; SDK channel pages only when their manifest and page are staged.

Committed pages load first-party CSS, JavaScript, and brand assets with a `?v=` hash (first 12 characters of SHA-256). After changing one of these assets, update the hashes; the tests reject stale ones:

```bash
python3 .github/scripts/web_asset_versions.py
```

Generated pages compute their hashes at build time. The route manifest and fragments are not hashed; the browser revalidates them instead.

## Browser dependencies

The installer uses a local bundle built from pinned `esptool-js` 0.6.1 and `improv-wifi-serial-sdk` 2.8.1, extended with ESPectre's `GET_MATTER_ONBOARDING` (`0x80`) RPC. QRCode.js 1.0.0 draws the Matter setup code and ansi_up 6.0.6 colors the serial log. `package-lock.json` pins the versions.

```bash
npm --prefix docs/web ci --ignore-scripts
npm --prefix docs/web run stage:vendor
```

`stage:vendor` builds the Web Serial bundle and copies it, the QR and ANSI libraries, and their licenses. CI does the same; `build/`, `vendor/`, and `node_modules/` are not committed. There is no remote fallback, so a local preview needs both commands before it can connect to a board.

## USB installer

### What the user sees

- While connecting, identifying, or restarting a board, the Connect button shows a spinner, the current step, and the elapsed time.
- After flashing Native or ESPHome, the installer uses standard Improv Serial to set up Wi-Fi if needed, and passes the device's address to Device settings when the firmware reports one (the `target` parameter of the Improv URL).
- Matter only answers read-only Improv requests (firmware identity and the stored pairing codes). Its Wi-Fi is set up by the Matter controller.

### How the board is identified

The installer tries, in order:

1. **Improv Serial.** It retries the state request once per second while the firmware boots, and cancels pending requests before releasing the port.
2. **Boot logs**, after a reset. ESPHome logs report the `francescopace.espectre` project version, even on renamed devices.
3. **App descriptors**, read through the bootloader, when identity or version is still missing. An ESPHome descriptor named `espectre` is recognized, but its version is ESPHome's, so it is not shown. ESPHome builds with a custom app name need Improv or logs.

Micro-ESPectre is recognized from its `micro-espectre` descriptor or its startup log. Its descriptor carries the firmware build version, not later deployments; older logs identify it without a version. Plain MicroPython is shown as MicroPython (marked "inferred" when guessed from a banner or from the standard 4 MiB three-partition layout), cannot be updated in place to Native, and offers no Wi-Fi setup. Pairing codes are read only when you open the Matter QR action.

### Transfer limits

- The bootloader connection asks for a 64 KiB serial buffer. Metadata is read in 1024-byte blocks, one at a time, and each block's exact length is checked before it is acknowledged, so a truncated block fails at once.
- Each partition-table or descriptor read has a 10-second deadline. On failure, identification stops with an error, pending reads and writes are cancelled, and no late acknowledgement is sent.
- The error is shown before cleanup, which waits at most 3 seconds. If the browser keeps the port locked, the message asks you to unplug and reconnect the board.
- Failures keep their step and I/O stage in the error message; no per-block debug logs are written. These limits do not affect erase or write timeouts.

### Firmware signature verification

Before erasing or writing flash, the installer checks every download. [firmware-auth.mjs](assets/js/firmware-auth.mjs), used by [flash-tool.js](assets/js/flash-tool.js), is the reference implementation:

- `verifyCatalog()` checks the catalog signature and contents; `verifyArtifact()` checks each image's size and SHA-256; `authenticateDownload()` returns exactly the bytes that will be flashed.
- In production, a missing signature, unknown key, altered data, unavailable keys, or wrong digest blocks installation. A failed download always blocks it.
- On the four local and test hosts listed above, you can override a failed check once, after all images are downloaded; the same bytes are then flashed. Cancelling blocks installation, and the choice is never remembered. Query parameters change nothing.

Catalog format:

- `schema_version: 1`. Each artifact has `size` and `sha256`, and the catalog has an `authentication` object with `key_id`, `payload`, and `signature` (standard Base64).
- The signature is RSA-PSS with SHA-256, MGF1-SHA-256, and a 32-byte salt, verified by Web Crypto on the decoded payload bytes before any JSON parsing.
- The signed payload holds `format: "espectre-firmware-v1"`, channel, version, release tag, source commit, and the list of artifacts (frontend, chip, chip family, build type, filename, size, and SHA-256). URLs, timestamps, and display data are not signed, so staging can rewrite URLs and keep only factory images; every kept image is still checked against the signed list.
- Trusted public keys are in [firmware-signing-keys.json](assets/firmware-signing-keys.json), as Base64 DER SubjectPublicKeyInfo (`spki`). A key ID is the SHA-256 of that DER.
- Unsigned old channels are left out until rebuilt with signing; they are never re-signed on the fly.

Factory images are verified in full, bootloader and partition table included. An update that keeps device data still writes the whole application image, including its signature and padding. NVS and Matter pairing data are kept as before.

Limits: the checks trust the website code and key list, so whoever can replace them can bypass the checks. Device-side OTA verification is separate; Matter has only this browser-side check. The format does not prevent replaying an older signed catalog. Keys are managed as described in [firmware signing](../RELEASING.md#firmware-signing); the user workflow is in [official images and personal builds](../SETUP.md#official-images-and-personal-builds).

## Firmware and SDK artifacts

To try local firmware builds in the preview, stage their factory images:

```bash
./test/web/generate_firmware_manifest.sh
./test/web/generate_firmware_manifest.sh --dry-run
./test/web/generate_firmware_manifest.sh --replace
```

The helper writes the release catalog to `artifacts/firmware/release/` and keeps images already staged unless you pass `--replace`.

Channels:

- The site serves ESPectre 3 and later, prereleases included.
- **Release** is the most recently published numeric GitHub release (release candidates included; drafts and rolling tags excluded). It is a tested tagged release, not necessarily a stable one. Until one exists, Release is unavailable.
- Rolling versions come from the SDK manifest. Builds need a numeric 3.x (or later) tag in their history; builds still named `2.8.0-<commits>-g<sha>` cannot be deployed.

All downloads live under the ignored `artifacts/` folder: firmware in `artifacts/firmware/<channel>/`, SDK archives in `artifacts/sdk/<channel>/`, and the API reference in `artifacts/sdk/api/`.

**SDK pages.** The SDK page recommends the ESP-IDF Component Manager and offers source archives as an alternative. Tagged releases (prereleases included) go to the production registry; Preview (`main`) and Develop (`develop`) snapshots go to staging, with a branch suffix in their version. Older prereleases remain on staging. See [ESP Component Registry](../SDK.md#esp-component-registry).

**API reference.** Generate it with `python3 .github/scripts/generate_sdk_api.py` (needs Doxygen 1.17.0 and a pinned m.css revision; `--mcss-root` reuses an existing checkout). Each run replaces the whole reference, and every public header must produce a page. The browser loads it from `artifacts/sdk/api/api-index.json`; the `api` and `member` query parameters select a page and a symbol. Detailed contracts come from `src/cpp/sdk_integration.dox`, and package documentation links point to the matching source revision on GitHub.

**Version badges.** The home and roadmap badges show the Release firmware version (`release_tag`, else `version`) and link to the installer. They never fall back to Preview; without a Release, the home badge is hidden and the roadmap shows "unavailable". The API reference shows its own build version.

## Publication

- **Every commit:** CI runs the web tests, builds the pages and API reference, and verifies the site, without downloading published channels.
- **Snapshot and Release workflows** stage the firmware and SDK for their channel, recover the other published channels, and verify all of them. They then start `pages.yml` on `main` and wait for it.
- **`pages.yml`** checks the source run, downloads its verified Pages archive, and deploys it unchanged. After deployment, the live firmware catalog, SDK catalog, and signing keys must match the archive byte for byte (with a bounded wait for propagation) before IndexNow is notified.
- **Retrying only the website:** run `pages.yml` on `main` with `pages_run_id` and `pages_run_attempt` from the successful run, while its artifacts still exist. See [website deployment](../RELEASING.md#website-deployment).

The shared `build-pages` action stages dependencies, runs the tests, builds pages and the API reference, and verifies the result. `build_sitemap.py` writes the ignored `sitemap.xml` from `routes.json` and the staged SDK channels; its `lastmod` dates come from Git history, so Pages builds need the full history. IndexNow receives exactly that sitemap.

## Routing and analytics

**Routing.** The SPA uses canonical paths with the History API. Old `#` links still work and are replaced with the canonical path; static pages can use them to open a tool in the SPA. Device settings and Monitor load with the shared device session; CSI visualizer, Game, and Theremin load their scripts on first use (`data-script-src`). Keep `app.js` last among the core `defer` scripts, since it binds their initializers.

**404 suggestions.** `404-suggestions.json` maps old paths to a suggested page. On an exact match (with or without a trailing slash or `index.html`), the 404 page shows that one link; it never redirects or changes the 404 status. Destinations must be public routes or the project's GitHub repository.

**Demo mode.** Moving the mouse simulates motion. On touch or pen devices, drag on the Monitor chart or the Theremin pitch display; the Game keeps its press-and-hold control.

**Analytics.** `assets/js/analytics.js` enables GA4 only on production and allowlisted debug hosts, and only after consent.

- The router sends `page_view` events itself, with the canonical path, title, and content group. GA4's automatic history-based page views must stay disabled to avoid duplicates. The 404 page records the requested path without query parameters, and a click on a suggestion sends `select_404_suggestion`.
- Every custom event goes through `trackEvent()`, which rejects unknown events, drops unknown parameters, checks values and ranges, and normalizes errors and firmware versions (rolling versions become `<major>.<minor>.<patch>-dev`; unknown values become `unknown` or are dropped).
- Parameters must stay low-cardinality and never include device IDs, network names or addresses, credentials, pairing codes, payloads, raw CSI, or error messages. Enhanced Measurement is configured in GA4 and bypasses this gate.
- The tests cover every custom event. The public policy is [privacy.html](content/privacy.html). No Cloudflare or deployment setup is needed.

## Direct HTTP

`assets/js/espectre-direct.js` handles Direct requests, SSE parsing, cancellation, and reconnects; the protocol is in the [API reference](../API.md). Device settings and the live tools share one connection picker: Local, Demo, and a planned Remote option (the relay does not exist yet). The Local panel states the minimum firmware, ESPectre 3.0.0-rc1; USB installation and Demo work without it.

- Starting Demo or leaving a page cancels a pending discovery, and late results are ignored.
- The SSE connection stays open across pages and while the tab is hidden, so the device indicator stays live.
- Monitor requests only its eight diagnostic fields (including the device's hardware-error total), once per second, and only while its diagnostics panel is open and the tab is visible.
- Wi-Fi scan results are polled after 1 second, then 2, then every 3; polling stops when you leave Device settings or hide the tab.
- Refreshes share requests in flight and reuse data kept current by SSE. Wi-Fi, MQTT, and settings diagnostics load only when Device settings opens, and are dropped when you leave. A reconnect discards the old session's data and cancels its requests.
- After saving, only the changed resource is read back. The raw CSI parser handles split or merged HTTP chunks with a bounded buffer.

`assets/js/browser-support.js` holds the browser support list and the Local Network Access permission checks. The connection picker explains how to fix permission, origin, discovery, timeout, protocol, and SSE capacity errors. Direct never scans the network or relaxes security headers.

## Device settings

Device settings has MQTT presets for Home Assistant with the Mosquitto add-on, a local broker, EMQX Cloud, HiveMQ Cloud, Flespi, and a custom broker. Credentials are never prefilled.

- Cloud presets fill the TLS port and an editable endpoint template (`.emqxsl.com`, `.hivemq.cloud`). Their ports, and Flespi's fixed hostname, are read-only; endpoints, credentials, and topic prefixes stay editable.
- Secure presets save with `mqtts://` automatically.
- Monitor uses Direct HTTP, not MQTT over WebSockets.

To name a device, click its ID in the banner (or the current name to change it). The name is saved when the field loses focus or on Enter; Escape cancels.

## Tests

Run the Direct HTTP, Analytics, and structural tests, which need no hardware, from the repository root:

```bash
node --test 'test/web/*.mjs'
```

`test/web/generate_firmware_manifest.sh` stages local firmware and is not part of the test run.
