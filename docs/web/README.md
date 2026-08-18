# Website

Static single-page app published at `espectre.dev` through GitHub Pages.

## Run locally

```bash
python -m http.server 8090 --directory docs/web
```

Then open `http://localhost:8090`. The Flash tool and the Matter QR reader need a Chromium-based browser. Configure additionally needs `localhost` or HTTPS. Monitor, Game, and Theremin use MQTT over WebSockets and do not need Bluetooth.

## Static content pages

Guides, docs, media, the roadmap, and the privacy notice use shared HTML fragments for both SPA hash routes and canonical, indexable paths. Generate the standalone pages before previewing their direct URLs:

```bash
python3 .github/scripts/build_static_pages.py
```

Edit shared fragments under `content/`, including `content/guides.html`, `content/guides/*.html`, `content/docs.html`, `content/docs/*.html`, `content/media.html`, `content/roadmap.html`, and `content/privacy.html`. Keep stylesheets under `assets/css/`, public images under `assets/images/`, and first-party scripts under `assets/js/`. Do not edit generated route `index.html` pages.

## Browser dependencies

Security-sensitive browser tools use pinned, same-origin copies under the generated `vendor/` directory in production:

- ESP Web Tools 10.4.0 for serial firmware installation;
- MQTT.js 5.3.0 for Monitor’s live MQTT-over-WebSocket session; and
- QRCode.js 1.0.0 for Matter pairing codes.

Install and stage the pinned packages locally with:

```bash
npm --prefix docs/web ci --ignore-scripts
npm --prefix docs/web run stage:vendor
```

`package-lock.json` is the source of truth for dependency versions. CI stages the same files before deployment, while `vendor/` and `node_modules/` remain ignored. When a dependency is absent during development on `localhost`, the site may fall back to the matching version on unpkg. Production never uses that fallback and treats missing same-origin dependencies as an error.

## Analytics and consent

`assets/js/analytics.js` enables GA4 on `espectre.dev` and on loopback hosts only after explicit consent. Local previews always set GA4 `debug_mode`, so their events remain identifiable as developer traffic and available in DebugView. The site stores the choice under `espectre.analytics.consent.v1`, disables advertising storage and Google Signals, and exposes Cookie settings in every generated footer. The public policy is owned by `content/privacy.html`.

Guide and documentation analytics are convention-based: same-origin `/guides/<slug>/` and `/docs/<slug>/` links report their registered route name as `guide_name` and `document_name`, while otherwise unmapped `guide-<slug>` and `docs-<slug>` SPA routes receive human-readable page titles automatically. Route-registry metadata preserves established titles, historical parameter values, the documentation root, and SDK artifact names; `analytics.js` contains no path maps. Tool analytics remain explicit because each tool owns distinct capabilities, events, and funnels.

`assets/js/route-registry.js` is the single source of truth for SPA route membership, navigation groups, page titles, canonical static paths, analytics content groups, and content-event names. Register a new SPA page there once; `app.js` uses it for routing and active navigation, while `analytics.js` uses the same metadata. The registry is also loaded by generated static pages, and structural tests require it to match every `main[data-page]` and `data-static-url` entry in `index.html`.

The event contract is intentionally low-cardinality and excludes Wi-Fi SSIDs and passwords, broker addresses and credentials, device identifiers, Bluetooth identifiers, Matter pairing codes, raw CSI, and MQTT payloads.

| Journey | Events and required parameters | Intended use |
|---|---|---|
| Navigation | `page_view` (`page_path`, `page_title`, `content_group`), `select_tool`, `select_guide`, `select_documentation`, `sdk_download` | Content, SDK downloads, and entry-point performance |
| Browser support | `tool_capability` (`tool_name`, `capability`, `result`) | Separate unsupported browsers from product failures |
| Firmware | `firmware_catalog`, `firmware_selection`, `firmware_install_start`, `firmware_install_result`, `firmware_download` | Measure catalog availability and the complete install funnel |
| Device tools | `tool_connection`, `tool_ready`, `tool_disconnect`, `tool_demo_start`, `device_profile` | Separate transport connection from the first valid data, measure duration, and report supported platform adoption |
| Configuration | `configure_change`, `ota_update_result`, `matter_qr_read` | Distinguish an accepted BLE write from a verified sysinfo value and the final OTA state |
| Experiences | `theremin_configuration`, `game_start`, `game_over`, `game_abandon` | Optional tool engagement and completion |

Outcome events use `result` values such as `accepted`, `success`, `failure`, `unconfirmed`, `cancelled`, `unsupported`, or `validation_failure`. `configure_change=accepted` means the BLE write completed; `success` is emitted only after a matching sysinfo snapshot. `tool_connection=success` means the transport connected, while `tool_ready` is emitted once after the first valid sysinfo, telemetry, or diagnostic payload. OTA success requires the device to report `reboot_scheduled`; a disconnect or status timeout is `unconfirmed`. Failures use a normalized `error_type`; never add raw exception messages. `frontend`, `chip`, `channel`, `format`, `transport`, `entry_point`, `tool_name`, `readiness`, and `ota_state` are candidate event-scoped custom dimensions. `latency_ms`, `duration_ms`, and `duration_seconds` are candidate custom metrics. Property-side configuration, retention, internal-traffic filters, key events, and funnel explorations must be verified in GA4 after deployment.

## Generated artifacts

The website stages all downloadable output under the generated `artifacts/` tree. SDK downloads live under `artifacts/sdk/stable/` and `artifacts/sdk/main/`, the Doxygen reference lives under `artifacts/sdk/api/` (also the default `src/cpp/Doxyfile` output in this repository), and firmware lives under `artifacts/firmware/<channel>/`. CI recreates the entire tree before deployment; none of its contents are tracked.

CI, stable releases, and snapshots use the same local `build-pages` action. It stages pinned browser dependencies, runs the web tests, generates static routes and the Doxygen reference, and verifies the complete tree before it can be uploaded to Pages. Channel-aware verification also rejects incomplete firmware matrices, mismatched SDK manifests, missing artifacts, and obsolete `/sdk/api/` links.

The committed `.github/scripts/sitemap.template.xml` is the canonical URL inventory and intentionally contains neither `changefreq` nor generated dates. During the Pages build, `build_sitemap.py` writes the ignored deployment artifact `sitemap.xml` with date-only `lastmod` values from the latest owning Git commit for editorial routes and the API reference, and from the staged SDK manifests for `stable` and `main`. Unknown dates are omitted rather than replaced with the deployment time. Pages-producing checkouts must retain full Git history so these dates remain source-accurate.

## BLE client API

`assets/js/espectre-ble.js` is a dependency-free client for the ESPectre BLE setup surface defined in `docs/ESPECTRE_PROTOCOL.md`. It exposes two globals: `ESPectreBleClient` and `ESPectreValidationError`. Web Bluetooth needs a Chromium-based browser and a secure context (HTTPS or `localhost`); check `ESPectreBleClient.supported` before connecting. Native uses this surface for Wi-Fi, MQTT, identity, and OTA. Monitor stays on MQTT.

Unlike the rest of the site, the client is **Apache-2.0** licensed (see [Apache-2.0.txt](assets/js/LICENSES/Apache-2.0.txt)), so any web application, including proprietary ones, can embed it.

```js
const client = new ESPectreBleClient();
client.on('sysinfo', (values) => console.log(values.chip, values.device_id));
client.on('disconnect', () => console.log('device dropped'));

await client.connect();          // opens the browser device chooser
await client.requestSysinfo();   // resolves into a `sysinfo` event
await client.setWifiConfig({ ssid: 'Lab Network', password: 'secret' });
await client.disconnect();
```

### Events

Subscribe with `on(event, handler)`, which returns an unsubscribe function; `off(event, handler)` also works. A throwing handler is logged and never breaks the client or other handlers.

| Event | Payload | When |
|---|---|---|
| `sysinfo` | `(values, entries)` — object plus ordered pairs | A snapshot completed (`END` received) |
| `sysinfo-line` | raw line | Every sysinfo line, including `END` |
| `disconnect` | — | Unexpected GATT drop; never fired by `disconnect()` |

### Connection

| Member | Notes |
|---|---|
| `connect({ sysinfo })` | `sysinfo` defaults to `true`; reentrant (returns the in-flight promise or the connected device) |
| `disconnect()` | Idempotent; stops notifications and closes GATT |
| `setSysinfoNotifications(bool)` | Toggle sysinfo without disconnecting |
| `connected`, `name`, `device` | Read-only state |

### Commands

Every `set*` method validates locally, throws `ESPectreValidationError` on a bad argument, and writes the command over the control characteristic. The matching static `build*Command` functions are pure and return the wire string, so arguments can be validated without a connected device.

| Method | Command | Validation |
|---|---|---|
| `setWifiConfig({ ssid, password, bssid, channel, bandPolicy })` | `SET_WIFI_CONFIG` | `ssid` required; credentials optional; `bandPolicy` is optional `2g`, `5g`, or `auto`; `channel` is 0 (auto) or a matching 20 MHz center; `bssid` is empty or a MAC |
| `clearWifiConfig()` | `CLEAR_WIFI` | — |
| `setMqttConfig({ host, port, username, password, topicPrefix })` | `SET_MQTT_CONFIG` | `host` required; `port` 1-65535; credentials optional |
| `clearMqttConfig()` | `CLEAR_MQTT_CONFIG` | — |
| `setDeviceLabel(label)` | `SET_DEVICE_CONFIG` | single-line string, may be empty |
| `clearDeviceConfig()` | `CLEAR_DEVICE_CONFIG` | — |
| `otaStatus()` | `OTA_STATUS` | — |
| `otaCheck()` | `OTA_CHECK` | — |
| `otaStart()` | `OTA_START` | — |
| `stopBle()` | `STOP_BLE` | — |
| `requestSysinfo()` | `REQ_SYSINFO` | — |
| `writeControl(command)` | any | Escape hatch for commands the library does not model |

The library validates protocol correctness only. The device remains authoritative for hardware capabilities: clients should offer `5g` and `auto` only when sysinfo reports `supports_wifi_5ghz=true`. A changed band policy is persisted immediately and takes effect after the device restarts.

### Errors

Command builders throw `ESPectreValidationError` (`error.name === 'ESPectreValidationError'`); everything else that rejects is a transport or browser error. `ESPectreBleClient.VERSION` identifies the library version, independent of the device `proto_version` reported in sysinfo.

### Tests

The hardware-independent BLE surface and website analytics and structural contracts are covered by unit tests:

```bash
node --test 'test/web/*.mjs'
```
