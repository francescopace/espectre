# Website

Static single-page app published at `espectre.dev` through GitHub Pages.

## Run locally

```bash
python -m http.server 8090 --directory docs/web
```

Then open `http://localhost:8090`. A Native development build with `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y` accepts HTTP loopback Origins on any port when the host is exactly `localhost`, `127.0.0.1`, or `[::1]`; published firmware leaves this exception disabled. Flash, Improv Serial, and the Matter QR reader need a Chromium-based browser. Configure connects to Native through a local Direct WebSocket. Monitor, Game, and Theremin support both Direct WebSocket and MQTT over WebSockets. The hosted Direct workflow supports Chrome 147 or later on desktop through WebSocket Local Network Access; Chrome 151 on macOS passed the hosted HTTPS validation path against a physical ESP32-C3. Firefox and Safari block the hosted HTTPS-to-`ws://` path; Edge and mobile Chrome remain unclaimed until their physical browser runs are recorded. A local HTTP preview does not prove hosted-portal compatibility.

First-party CSS, JS, and SPA content fragments use a 12-character SHA-256 prefix as `?v=`. After editing those files, restamp `index.html` and `404.html`:

```bash
python3 .github/scripts/web_asset_versions.py
```

The website tests fail if a committed hash does not match the file contents. Generated static and SDK pages compute the same hashes at build time, so only the changed file is cache-busted.

## Static content pages

Guides, docs, media, the roadmap, privacy, terms, legal, security, licensing, and contact content use shared HTML fragments for both SPA hash routes and canonical, indexable paths. Generate the standalone pages before previewing their direct URLs:

```bash
python3 .github/scripts/build_static_pages.py
```

Edit shared fragments under `content/`, including `content/guides.html`, `content/guides/*.html`, `content/sdk.html`, `content/sdk/*.html`, `content/media.html`, `content/roadmap.html`, `content/privacy.html`, `content/terms.html`, `content/legal.html`, `content/security.html`, `content/licensing.html`, and `content/contact.html`. Keep stylesheets under `assets/css/`, public images under `assets/images/`, and first-party scripts under `assets/js/`. Do not edit generated route `index.html` pages.

## Browser dependencies

Security-sensitive browser tools use pinned, same-origin copies under the generated `vendor/` directory in production:

- ESP Web Tools 10.4.0 for serial firmware installation;
- MQTT.js 5.3.0 for the live MQTT-over-WebSocket session shared by Monitor, Game, and Theremin; and
- QRCode.js 1.0.0 for Matter pairing codes.

Install and stage the pinned packages locally with:

```bash
npm --prefix docs/web ci --ignore-scripts
npm --prefix docs/web run stage:vendor
```

`package-lock.json` is the source of truth for dependency versions. CI stages the same files before deployment, while `vendor/` and `node_modules/` remain ignored. When a dependency is absent during development on `localhost`, the site may fall back to the matching version on unpkg. Production never uses that fallback and treats missing same-origin dependencies as an error.

## Analytics and consent

`assets/js/analytics.js` enables GA4 on production and allowlisted debug hosts only after explicit consent. Debug traffic sets GA4 `debug_mode`, so its events remain identifiable as developer traffic and available in DebugView. The site stores the choice under `espectre.analytics.consent.v1`, disables advertising storage and Google Signals, and exposes Cookie settings in the SPA, generated static, SDK, and 404 footers. The public policy is owned by `content/privacy.html`.

Guide and SDK analytics are convention-based: same-origin `/guides/<slug>/` and `/sdk/<slug>/` links report their registered route name as `guide_name` and `document_name`, while otherwise unmapped `guide-<slug>` and `sdk-<slug>` SPA routes receive human-readable page titles automatically. Route-registry metadata preserves established titles, historical parameter values, the SDK root, and artifact names; `analytics.js` contains no path maps. Tool analytics remain explicit because each tool owns distinct capabilities, events, and funnels.

`content/tools.html` owns the shared Tools catalog rendered by both the `/tools/` static page and the `#tools` SPA route. Interactive entry points use `/tools/<name>/` publicly and `#tool-<name>` inside the persistent app shell; `build_static_pages.py` generates the entry pages that preserve query parameters before handing off to the corresponding SPA route.

`assets/js/route-registry.js` is the single source of truth for deployment host roles, SPA route membership, navigation groups, page titles, canonical static paths, analytics content groups, and content-event names. `app.js` and `analytics.js` consume the same production, validation, and loopback classification. Register a new SPA page there once; the registry is also loaded by generated static pages, and structural tests require it to match every `main[data-page]` and `data-static-url` entry in `index.html`.

The event contract is intentionally low-cardinality and excludes Wi-Fi SSIDs and passwords, broker addresses and credentials, device identifiers, local device endpoints, Matter pairing codes, raw CSI, and MQTT payloads.

| Journey | Events and required parameters | Intended use |
|---|---|---|
| Navigation | `page_view` (`page_path`, `page_title`, `content_group`), `select_tool`, `select_guide`, `select_documentation`, `sdk_download`, `click_contact`, `click_security` | Content, SDK downloads, contact actions, and entry-point performance |
| Browser support | `tool_capability` (`tool_name`, `capability`, `result`) | Separate unsupported browsers from product failures |
| Firmware | `firmware_catalog`, `firmware_selection`, `firmware_install_start`, `firmware_install_result` | Measure catalog availability and the browser install funnel |
| Device tools | `tool_connection`, `tool_ready`, `tool_disconnect`, `tool_demo_start`, `device_profile` | Separate transport connection from the first valid data, measure duration, and report supported platform adoption |
| Configuration | `configure_change`, `ota_update_result`, `matter_qr_read` | Distinguish an accepted Direct setup write from a verified device state and the final OTA state |
| Experiences | `theremin_configuration`, `game_start`, `game_over` (`score`, `orbs`, `distance`), `game_abandon` (`score`, `distance`, `reason`) | Optional tool engagement, completion, and abandonment |

Outcome events use `result` values such as `accepted`, `success`, `failure`, `unconfirmed`, `cancelled`, `unsupported`, or `validation_failure`. `configure_change=accepted` means the Direct request was accepted; `success` is emitted only after matching device state confirms it. `tool_connection=success` means the transport connected, while `tool_ready` is emitted once after the first valid state, telemetry, or diagnostic payload. OTA analytics success requires the device to report `reboot_scheduled`; a disconnect or status timeout is `unconfirmed`. The OTA dialog then stays open until Monitor sees retained `status` `online: true` or a post-reboot `ota_status` snapshot, updates the current firmware version from `info`, and closes. Failures use a normalized `error_type`; never add raw exception messages. `frontend`, `chip`, `channel`, `format`, `transport`, `entry_point`, `tool_name`, `readiness`, and `ota_state` are candidate event-scoped custom dimensions. `latency_ms`, `duration_ms`, `duration_seconds`, `score`, `orbs`, and `distance` are candidate custom metrics. Property-side configuration, retention, internal-traffic filters, key events, and funnel explorations must be verified in GA4 after deployment.

## Generated artifacts

The website stages all downloadable output under the generated `artifacts/` tree. SDK downloads live under `artifacts/sdk/release/`, `artifacts/sdk/preview/`, and `artifacts/sdk/develop/`, the Doxygen reference lives under `artifacts/sdk/api/` (also the default `src/cpp/Doxyfile` output in this repository), and firmware lives under `artifacts/firmware/<channel>/`. GitHub Releases publish each firmware image directly and consolidate its build-specific SBOMs, notices, and license archives into `firmware-compliance-<channel-or-version>.zip`; website staging expands that bundle so each compliance file remains available next to its firmware image. CI recreates the entire tree before deployment; none of its contents are tracked. Generate the API reference locally with `python3 .github/scripts/generate_sdk_api.py`, which stamps Doxygen from the same `git describe` identity as the SDK bundles.

CI, official releases, and rolling preview builds use the same local `build-pages` action. It stages pinned browser dependencies, runs the web tests, generates static routes and the Doxygen reference, and verifies the complete tree before it can be uploaded to Pages. Channel-aware verification also rejects incomplete firmware matrices, mismatched SDK manifests, and missing artifacts.

The committed `.github/scripts/sitemap.template.xml` is the canonical URL inventory and intentionally contains neither `changefreq` nor generated dates. During the Pages build, `build_sitemap.py` writes the ignored deployment artifact `sitemap.xml` with date-only `lastmod` values from the latest owning Git commit for editorial routes and the API reference, and from the staged SDK manifests for `release`, `preview`, and `develop`. Unknown dates are omitted rather than replaced with the deployment time. Pages-producing checkouts must retain full Git history so these dates remain source-accurate.

## Browser protocol clients

`assets/js/espectre-direct.js` implements Direct WebSocket envelopes, request correlation, and reconnect behavior for Configure and the live tools, while `assets/js/espectre-mqtt.js` implements the MQTT protocol layer shared by Monitor, Game, and Theremin. Both are dependency-free first-party components released under the same GPLv3 and commercial licensing policy as the rest of ESPectre.

The wire contract, supported commands, topic families, and capability boundaries are documented in `docs/ESPECTRE_PROTOCOL.md`. Keep protocol behavior in these clients instead of duplicating it in `app.js`; broker connection policy remains an application concern.

`assets/js/browser-support.js` owns the declared browser matrix and queries the Local Network Access permission state when the browser exposes it. Direct failures stay visible in Configure and Monitor and give separate recovery guidance for a denied or pending permission, a hosted mixed-content restriction, an unsupported page Origin, `.local` resolution, an address timeout, a protocol mismatch, and occupied Direct client slots. The browser WebSocket API deliberately hides handshake response details, so an otherwise unclassified failure presents a safe checklist instead of claiming a cause. Direct support does not add a third-party script, relax a global security header, scan the LAN, or imply browser-side mDNS enumeration.

### Tests

The hardware-independent Direct WebSocket and MQTT surfaces, website analytics, and structural contracts are covered by unit tests:

```bash
node --test 'test/web/*.mjs'
```
