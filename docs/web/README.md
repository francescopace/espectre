# Website

[espectre.dev](https://espectre.dev) is published through GitHub Pages. Public routes have generated static pages for direct access and indexing; a single-page app (SPA) handles navigation and browser tools.

## Local preview

From the repository root, install and stage the browser dependencies, generate the pages, and start a local server:

```bash
npm --prefix docs/web ci --ignore-scripts
npm --prefix docs/web run stage:vendor
.venv/bin/python .github/scripts/build_static_pages.py
.venv/bin/python -m http.server 8090 --directory docs/web
```

Open `http://localhost:8090`. Dependencies are pinned in `package-lock.json` and served locally, with no remote fallback. The generated `build/`, `vendor/`, and `node_modules/` directories are not committed.

USB tools require a desktop Chromium browser with Web Serial. Hosted Direct access has been tested with Chrome on macOS; check [browser-support.js](assets/js/browser-support.js) for the current capability policy. Windows and Linux have not been tested on hardware, and a local preview does not validate hosted access.

To connect from localhost, build firmware with `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y`. Published firmware keeps local development origins disabled.

## Sources and generated pages

| Source | Purpose |
|---|---|
| `content/` | Shared page fragments |
| `../CHANGELOG.md` | Release notes rendered into `/changelog/` during the static-page build |
| `assets/` | Styles, images, and browser code |
| `routes.json` | Public routes, metadata, content groups, SDK channels, and legacy redirects |
| `index.html`, `404.html` | SPA entry point and fallback page |
| [web_page_shell.py](../../.github/scripts/web_page_shell.py) | Shared layout for generated pages |

Edit fragments and shared sources, not the generated `index.html` files in route directories. The SPA and static-page builder use the same route manifest. Keep `app.js` last among the core deferred scripts so it can bind their initializers.

The changelog build uses Python Markdown from `requirements.txt` and writes both `content/changelog.html` and `/changelog/index.html`. Edit `docs/CHANGELOG.md`; both generated files stay untracked. Release headings have stable `release-<version>` anchors, with dots replaced by hyphens, and work in progress is labelled Unreleased. Relative documentation links resolve against their release tag, or `develop` for unreleased notes. The roadmap links to this page. The authored `/guides/ml-data-collection/` guide covers recording and contributing datasets; `docs/ML_DATA_COLLECTION.md` remains the detailed reference.

After changing first-party assets, refresh their cache hashes and rebuild the pages:

```bash
.venv/bin/python .github/scripts/web_asset_versions.py
.venv/bin/python .github/scripts/build_static_pages.py
```

Generated pages include section indexes and related-page navigation in their initial HTML. JavaScript enhances these controls without inserting navigation after first paint. Keep section IDs stable so inbound links and generated indexes remain valid.

Internal navigation keeps the current page visible while the destination fragment loads, then switches to the populated page. While it waits, the previous page is inert and a fixed progress bar shows that navigation has started. At startup a fixed loading status appears while the destination loads, and the footer stays hidden until the populated page is revealed. This also applies when refreshing a page reached through the SPA. Only the latest navigation may reveal a destination or move focus. Keep loading states and asynchronously updated release badges from shifting visible content.

Routes use canonical paths; legacy hash links remain supported. The deployment build generates the sitemap from the route manifest and staged SDK channels.

`legacyRedirects` in `routes.json` maps retired paths and legacy `doc` identifiers to current routes on the same site. Each entry has a root-relative `href`, a `title`, and a `path`, a `doc`, or both. Destinations must be canonical site routes; an optional fragment selects a section. External destinations are rejected by both the build and the browser registry, so redirects stay on the current host, including the test host. The static-page build generates an immediate meta refresh, a canonical link, and a visible destination link for each retired path. These pages remain outside the sitemap. The router handles known `doc` parameters on `/` and `/index.html` before initializing the app or measurement; incoming query parameters and fragments are not forwarded. Unknown or repeated `doc` values do not redirect. Unknown paths show the 404 page with links to the main site sections.

For a local Caddy preview, use `handle_errors 404` with `rewrite * /404.html` and `file_server` to serve the custom error page while preserving HTTP 404. Keep `X-Robots-Tag: noindex` on the test host. Production GitHub Pages serves `404.html` for unknown paths; known retired paths have their own generated redirect pages.

`legacyNotFound` lists removed documents that have no replacement page. Each entry pairs a legacy `doc` identifier with its missing path. The build removes any old generated redirect at that path, and verification requires the path to remain absent. Legacy query links navigate to that missing path so the server returns HTTP 404; visiting `/404.html` directly would return 200. Contributing and Testing use this policy. Changelog and ML data collection instead redirect to their new pages on the site.

## Browser tools

The main implementation entry points are:

- [flash-tool.js](assets/js/flash-tool.js): USB identification, firmware installation, and Improv setup.
- [firmware-auth.mjs](assets/js/firmware-auth.mjs): firmware catalog signatures and artifact verification.
- [espectre-direct.js](assets/js/espectre-direct.js) and [device-session.js](assets/js/device-session.js): Direct HTTP, SSE, and the device session shared by the live tools.
- [app.js](assets/js/app.js): routing, tool initialization, and interface behavior.

In Demo mode, the Monitor applies the configured motion-on and motion-off reading counts to state transitions and chart highlighting. The movement score stays unchanged. These settings remain active while navigating between tools and reset when a new demo session starts.

The Monitor's Destination MAC menu offers Automatic (AP) and discovered ESPectre sensors, with no manual address entry. A configured MAC remains selected even when its sensor has not been discovered. The menu searches only when its refresh button is clicked. Every click starts a fresh discovery through the connected device, without caching or reusing Find devices results. Search progress appears only on the refresh button, as with the BSSID refresh in Device settings. A failed search, or one that finds no other sensor, shows a short notice. Leaving the Monitor or changing the connection discards the results while preserving the configured MAC, or `auto` when none is configured. The discovery contract and receiver requirements are described in the [discovery reference](../DISCOVERY.md#client-validation-and-fallback).

Live tools share a Local or Demo connection. Remote access is planned and has no relay implementation. The canonical device contract is in the [API reference](../API.md), with discovery rules in the [discovery reference](../DISCOVERY.md).

### Firmware signature verification

Before erasing or writing flash, the installer authenticates the catalog and checks each downloaded image. [firmware-auth.mjs](assets/js/firmware-auth.mjs) owns the browser verification:

- `verifyCatalog()` verifies the original signed payload bytes with Web Crypto before parsing the payload as JSON. It uses RSA-3072/PSS with SHA-256 and trusted public keys from [firmware-signing-keys.json](assets/firmware-signing-keys.json).
- The signed payload binds the channel, version, release tag, source commit, and artifact identities to their sizes and SHA-256 hashes. Download URLs are excluded so deployment can relocate files without signing them again.
- `verifyArtifact()` checks each image against that signed inventory. Factory images are checked in full, including the bootloader and partition table. `authenticateDownload()` returns the verified bytes to the installer without downloading them again.

In production, unsigned catalogs, unknown keys, unavailable verification keys, invalid signatures, and size or hash mismatches block installation. On `localhost`, `127.0.0.1`, `[::1]`, and `test.espectre.dev`, a failed check offers a one-time confirmation override after all downloads finish. The override is never remembered; download failures always block installation.

These checks trust the website code and public-key registry, and do not prevent replay of an older signed catalog. They are separate from device-side OTA verification; Matter relies on the browser check and has no OTA update path. See [firmware signing](../RELEASING.md#firmware-signing) for key management and [official images and personal builds](../SETUP.md#official-images-and-personal-builds) for installation rules.

## Measurement and consent

[analytics.js](assets/js/analytics.js) manages separate opt-in preferences for website statistics and campaign measurement. Both default to denied and can be changed through Cookie settings. Remarketing and advertising personalization remain disabled; debug hosts never send Ads conversions.

Keep tracking changes within this module's event and parameter allowlists. Never send credentials, device identifiers, network details, pairing codes, raw CSI, or free-form error messages. Analytics sends the initial page view after consent, before tool events; the router reports subsequent route changes and deduplicates its startup view. Legacy hash entries use their destination route. GA4's automatic history-based page views must remain disabled to avoid duplicates.

Interpret firmware installation events as completed operations, not unique people or devices. Separate installation, real-device readiness, SDK link selections, and commercial contact link selections when evaluating acquisition. Consent-based collection does not measure every visit. Investigate missing landing pages against session timing and consent before treating them as lost traffic; these client-side ordering guarantees do not reconstruct historical data or control GA4 session expiration.

Commercial contact and SDK download events measure link selections, not received leads or completed downloads. Ads measures these directly; importing the same GA4 events into Ads would duplicate the outcomes. The [privacy notice](content/privacy.html) describes the visitor-facing policy.

## Artifacts and publication

Firmware, SDK archives, and the generated API reference are staged under the ignored `artifacts/` directory. To preview local firmware builds:

```bash
./test/web/generate_firmware_manifest.sh
```

This preserves already staged images; use `--replace` to replace them. SDK distribution and channel details belong in the [SDK guide](../SDK.md#source-bundles).

CI tests and builds the website. Publication workflows stage the firmware and SDK channels, verify the complete site, and pass the resulting archive to GitHub Pages. Deployment and retry instructions are in [website deployment](../RELEASING.md#website-deployment).

## Tests

Run the web tests from the repository root; they need no hardware:

```bash
node --test --test-reporter=dot test/web/*.mjs
```

For visual changes, rebuild the affected pages and check them in a browser. Firmware staging is separate from the test run.
