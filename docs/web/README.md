# Website

The website in this directory is a single-page application published at
`espectre.dev`. It bundles the product pages, the browser tools, the guides,
and the SDK documentation behind one app shell with a persistent device
connection.

## Structure

| Path | Purpose |
|------|---------|
| [`index.html`](index.html) | The whole app: every page lives here as a routed `<main>` section |
| [`app.js`](app.js) | Hash routing, theme, shared BLE/demo connection, and per-tool logic |
| [`styles.css`](styles.css) | Palette and all component styles |
| [`espectre-ble.js`](espectre-ble.js) | Shared Web Bluetooth client (also used by external pages) |
| [`analytics.js`](analytics.js) | Google Analytics bootstrap and `trackEvent` helper |
| [`qrcode.js`](qrcode.js) | Vendored QR renderer for Matter commissioning codes |
| `flash/firmware/` | Published firmware manifests and binaries |
| `guides/images/` | Guide illustrations |
| `404.html` | Catch-all Pages serves for any retired URL; sends visitors to the app |

## Routing

Pages are hash routes handled in `app.js`: `#home`, `#tools`, `#flash`,
`#configure`, `#monitor`, `#theremin`, `#game`, `#guides`,
`#guide-hardware`, `#guide-setup`, `#guide-detection`, `#guide-firmware`,
`#docs`, `#docs-api`, `#docs-examples`, and `#docs-architecture`. The old
Every URL retired by the redesign — `/game/` and the generated
`/documentation/**` pages published before it — falls through to `404.html`,
which forwards to the app. Pages still answers with a 404 status there, so
search engines drop those URLs while visitors still land somewhere useful.

## Shared device connection

The header widget owns one connection shared by every page:

- **BLE mode** uses `espectre-ble.js` (Web Bluetooth). Telemetry drives all
  movement bars, and a `REQ_SYSINFO` snapshot populates the Configure fields
  and diagnostics. Configure writes use the standard control commands
  (`SET_WIFI_CONFIG`, `SET_MQTT_CONFIG`, `SET_DEVICE_CONFIG`, and their
  `CLEAR_*` counterparts) documented in `docs/ESPECTRE_PROTOCOL.md`.
- **Demo mode** simulates telemetry for browsers without Web Bluetooth and
  for trying the site without hardware. Nothing is ever written in demo mode.

The movement bars anchor the motion threshold at 55% of the visible scale;
`app.js` scales the raw score accordingly.

## Analytics

`analytics.js` configures GA4 with `send_page_view: false`, because a SPA
would otherwise report a single page view per session. The router in `app.js`
calls `window.trackRouteView(route)` on every navigation, which reports the
route under its real URL (`/#flash`) and sets the standard `content_group`
dimension. Tool routes group under their own name, guides and SDK docs share
`documentation`.

Two consequences worth keeping in mind:

- In the GA4 data stream, leave Enhanced Measurement's *Page changes based on
  browser history events* **off**. Page views are sent manually, so that
  setting would double-count every navigation.
- `app.js` sends events through `window.trackEvent`, guarded so the app keeps
  working when analytics is blocked or absent. Demo mode never reports
  `device_profile` or `configure_change`: no real device is involved.

## Theme And Palette

[`styles.css`](styles.css) is the single source of truth for the color
palette. The `:root` block defines the light theme and
`html[data-theme="dark"]` overrides it; the toggle in the header switches the
attribute. Page styles must consume the custom properties with `var(...)`
instead of introducing new color literals.

Intentional exceptions are:

- the raster favicon and social image, maintained as image assets
- the fixed black and white Matter QR colors, which preserve scanner contrast
- the dark code-block colors, which stay identical in both themes
- colors inside third-party or generated assets, such as `qrcode.js`

## Local Check

```bash
python -m http.server 8090 --directory docs/web
```

Then open `http://localhost:8090` and test the relevant pages at desktop and
mobile widths, in both themes. The Flash install button and Matter QR reader
need a Chromium-based browser; the BLE connection additionally needs
`localhost` or HTTPS.
