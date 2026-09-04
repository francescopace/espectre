# Website

The website is published at `espectre.dev` through GitHub Pages. Generated canonical pages provide indexable content, while a persistent SPA shell handles browser tools and in-app navigation.

## Local preview

From the repository root, run:

```bash
python -m http.server 8090 --directory docs/web
```

Open `http://localhost:8090`. Native development firmware accepts loopback Origins only when `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y`; published firmware keeps this exception disabled. Flash, Improv Serial, and the Matter QR reader require a Chromium-based browser.

The hosted Direct workflow is validated with Chrome 151 or later on macOS. Physical coverage on Windows and native Linux is still pending, and local discovery depends on the operating system's mDNS support. Other browsers are not guaranteed to work. A local HTTP preview does not prove hosted compatibility.

## Sources and generated pages

Edit shared page fragments under `content/`, styles under `assets/css/`, images under `assets/images/`, and first-party scripts under `assets/js/`. Do not edit generated route `index.html` files. In `routes.json`, `routes` owns public pages, metadata, canonical paths, navigation groups, and Analytics names, `contentGroups` owns their Analytics grouping, and `sdkChannels` owns the generated Release, Preview, and Develop SDK artifact pages. Public routes always contribute to the sitemap; SDK channels contribute only when both their manifest and generated page are staged. `assets/js/route-registry.js` loads the manifest directly in hosted and local previews.

Generate the standalone pages before testing direct route URLs:

```bash
python3 .github/scripts/build_static_pages.py
```

The generator adds route-specific titles, descriptions, canonical URLs, Open Graph metadata, and Twitter metadata from `routes.json`. The SPA loads the same manifest and fragments from `/content/`, and updates its runtime metadata when the active route changes.

First-party CSS, JavaScript, and brand assets referenced by committed entry pages use a 12-character SHA-256 prefix in `?v=`. The route manifest and SPA content fragments use HTTP revalidation so they remain directly loadable in local previews without generating assets. Restamp committed entry pages after changing hashed assets:

```bash
python3 .github/scripts/web_asset_versions.py
```

Website tests reject stale hashes. Generated static and SDK pages compute their asset hashes at build time.

## Browser dependencies

The browser installer uses a same-origin ESM bundle built from pinned `esptool-js` 0.6.1 and `improv-wifi-serial-sdk` 2.8.0 dependencies. The bundle adds ESPectre's `GET_MATTER_ONBOARDING` (`0x80`) RPC, QRCode.js 1.0.0 renders the returned Matter setup code, and ansi_up 6.0.6 renders ANSI styling in serial logs. Install and stage the dependencies locally with:

```bash
npm --prefix docs/web ci --ignore-scripts
npm --prefix docs/web run stage:vendor
```

`package-lock.json` owns the versions. `stage:vendor` builds the headless Web Serial bundle and copies it with the QR and ANSI renderers and upstream licenses. CI stages the same files, while `build/`, `vendor/`, and `node_modules/` remain ignored. There is no remote fallback: a local preview must run both commands before the installer can connect to a board.

## Firmware and artifacts

Use locally built firmware in the browser preview by restaging the available Native, Matter, and ESPHome factory images:

```bash
./test/web/generate_firmware_manifest.sh
./test/web/generate_firmware_manifest.sh --dry-run
./test/web/generate_firmware_manifest.sh --replace
```

The helper writes the release catalog under `artifacts/firmware/release/`. It preserves previously staged factory images unless `--replace` is present. Official deployments stage published GitHub Release assets through CI.

All downloads live under the ignored `artifacts/` tree. Firmware uses `artifacts/firmware/<channel>/`; SDK archives use `artifacts/sdk/<channel>/`; and the generated API reference uses `artifacts/sdk/api/`. Generate the API reference with `python3 .github/scripts/generate_sdk_api.py`. It requires Doxygen 1.17.0 and a pinned m.css revision; `--mcss-root` reuses an existing checkout.

The shared `build-pages` action stages dependencies and artifacts, runs the web tests, builds static routes and the API reference, and verifies the output before upload. `build_sitemap.py` generates the ignored `sitemap.xml` from `routes.json` and the SDK channels present in the staged Pages tree. Its `lastmod` dates come from the owning Git commits and staged SDK manifests, so Pages builds require full Git history. After deployment, IndexNow receives this exact generated sitemap inventory.

## Routing and Analytics

The SPA uses canonical paths with the History API. Legacy root hash links remain valid entry points and are replaced with their registered canonical path. Static tool calls to action may use this legacy handoff so the browser opens the persistent shell without losing the selected tool.

Device settings and Monitor load with the shared device session. CSI visualizer, Game, and Theremin load their scripts on first use through `data-script-src`. Keep `app.js` last among the core `defer` scripts because it binds their initializers.

`assets/js/analytics.js` enables GA4 on production and allowlisted debug hosts only after explicit consent. The router sends manual `page_view` events with canonical `page_location`, `page_path`, `page_title`, and `content_group` values. GA4 page changes based on browser history events must remain disabled to avoid duplicate page views.

All website custom events pass through `trackEvent()`. It rejects unregistered events, strips parameters outside the event contract, validates categorical values and numeric bounds, and normalizes error types and public firmware versions before calling `gtag`. Rolling Git versions are reported as `<major>.<minor>.<patch>-dev`; other unrecognized values become `unknown` or are omitted.

Keep Analytics parameters low-cardinality. They must not include device IDs, network names or addresses, credentials, pairing codes, payloads, raw CSI, or exception messages. Enhanced Measurement is configured in GA4 and does not pass through this custom-event gate. The Analytics tests verify every custom event emitted by the browser tools. The public policy is in [privacy.html](content/privacy.html).

## Direct HTTP

`assets/js/espectre-direct.js` owns resource-oriented Direct HTTP, incremental SSE parsing, abort, and reconnect behavior. Device settings and the live tools share one connection picker with Local connection, Demo, and the planned Remote connection. Relay support is not implemented. The wire contract and capability boundaries are in [API.md](../API.md).

`assets/js/browser-support.js` owns the browser matrix and Local Network Access permission checks. The active connection picker reports recovery guidance for permission, Origin, discovery, timeout, protocol, and SSE capacity failures. Direct support does not scan the LAN or relax a global security header.

## Tests

Run the hardware-independent Direct HTTP, Analytics, and structural tests from the repository root:

```bash
node --test 'test/web/*.mjs'
```

`test/web/generate_firmware_manifest.sh` stages local firmware and is not part of the Node test runner.
