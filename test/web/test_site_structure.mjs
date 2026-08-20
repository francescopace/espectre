/*
 * ESPectre - Website structural contract tests
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');
const index = read('docs/web/index.html');
const app = read('docs/web/assets/js/app.js');
const mqttProtocol = read('docs/web/assets/js/espectre-mqtt.js');
const browserSupportSource = read('docs/web/assets/js/browser-support.js');
const routeRegistry = read('docs/web/assets/js/route-registry.js');
const styles = read('docs/web/assets/css/styles.css');
const security = read('docs/web/security/index.html');
const GPL_HTML_HEADER = `<!--
  SPDX-License-Identifier: GPL-3.0-only
  Commercial licensing available under separate agreement; see LICENSING.md.
-->
`;

describe('website security and asset policy', () => {
    it('renders the brand face in white instead of exposing the background', () => {
        const logo = read('docs/web/assets/images/brand/espectre-logo.svg');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const sdkStager = read('.github/scripts/stage_web_sdk.py');
        assert.doesNotMatch(logo, /<mask\b|mask="url\(/);
        assert.match(logo, /<use href="#ghost" fill="#4b7bee" stroke="#b8c8ff" stroke-width="2\.8" stroke-linejoin="round"\/>/);
        assert.doesNotMatch(logo, /<use[^>]+stroke="#fff"/);
        assert.equal((logo.match(/<ellipse[^>]+fill="#fff"/g) || []).length, 2);
        assert.match(logo, /<path[^>]+stroke="#fff"/);
        assert.match(logo, /<circle[^>]+fill="#fff"/);
        assert.match(index, /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=[0-9a-f]{12}" alt="" width="30" height="30"/);
        assert.match(read('docs/web/404.html'), /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=[0-9a-f]{12}" alt="" width="30" height="30"/);
        assert.match(staticPageBuilder, /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=\{logo_version\}" alt="" width="30" height="30"/);
        assert.match(sdkStager, /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=\{logo_version\}" alt="" width="30" height="30"/);
    });

    it('does not execute third-party scripts before an explicit analytics choice', () => {
        const externalScripts = [...index.matchAll(/<script[^>]+src="(https?:[^\"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(externalScripts, []);
        assert.doesNotMatch(index, /unpkg\.com|jsdelivr\.net/);
        assert.match(index, /\/assets\/css\/styles\.css/);
        assert.match(index, /\/assets\/js\/app\.js/);
        assert.match(index, /\/assets\/js\/browser-support\.js/);
        assert.match(index, /\/assets\/js\/route-registry\.js/);
        const firstPartyScripts = [...index.matchAll(/<script\b([^>]*)>/g)]
            .map((match) => match[1])
            .filter((attrs) => /src="\/assets\/js\//.test(attrs));
        assert.deepEqual(
            firstPartyScripts.map((attrs) => attrs.match(/src="(\/assets\/js\/[^"?]+)/)[1]),
            [
                '/assets/js/espectre-ble.js',
                '/assets/js/espectre-mqtt.js',
                '/assets/js/browser-support.js',
                '/assets/js/route-registry.js',
                '/assets/js/navigation.js',
                '/assets/js/analytics.js',
                '/assets/js/app.js'
            ]
        );
        for (const attrs of firstPartyScripts) {
            assert.match(attrs, /\bdefer\b/, `expected defer on ${attrs.trim()}`);
        }
        assert.ok(index.indexOf('/assets/js/app.js') < index.indexOf('</head>'));
        assert.match(app, /\/vendor\/esp-web-tools-10\.4\.0\/install-button\.js/);
        assert.match(app, /\/vendor\/mqtt-5\.3\.0\/mqtt\.min\.js/);
        assert.match(app, /\/vendor\/qrcodejs-1\.0\.0\/qrcode\.min\.js/);
        assert.match(app, /LOCAL_DEVELOPMENT_HOSTS = new Set\(\['localhost', '127\.0\.0\.1', '\[::1\]'\]\)/);
    });

    it('keeps first-party cache-busting hashes in lockstep with file contents', () => {
        const stamper = read('.github/scripts/web_asset_versions.py');
        assert.match(stamper, /HASH_LENGTH = 12/);
        const hashLength = 12;
        const assetVersion = (relativePath) => createHash('sha256')
            .update(readFileSync(new URL(`../../docs/web/${relativePath}`, import.meta.url)))
            .digest('hex')
            .slice(0, hashLength);
        const assertStamped = (html, label) => {
            const refs = [...html.matchAll(
                /(?:href|src|data-content-url)="((?:\/assets\/(?:css|js)\/|\/assets\/images\/brand\/espectre-logo\.svg|content\/)[^"]*)"/g
            )];
            assert.ok(refs.length > 0, `${label} references first-party assets`);
            for (const [, url] of refs) {
                const [assetPath, query = ''] = url.split('?');
                const version = new URLSearchParams(query).get('v');
                const relativePath = assetPath.replace(/^\//, '');
                assert.equal(
                    version,
                    assetVersion(relativePath),
                    `${label} ${assetPath}`
                );
            }
        };
        assertStamped(index, 'index.html');
        assertStamped(read('docs/web/404.html'), '404.html');
        assert.match(stamper, /--check-current/);
    });
});

describe('website analytics contracts', () => {
    it('separates connection, readiness, verified outcomes, and disconnects', () => {
        assert.match(app, /track\('tool_ready'/);
        assert.match(app, /readiness,/);
        assert.match(app, /latency_ms:/);
        assert.match(app, /track\('configure_change', \{ action, result: 'accepted' \}\)/);
        assert.match(app, /finishConfigVerification\('success'\)/);
        assert.match(app, /track\('ota_update_result'/);
        assert.match(app, /state === 'reboot_scheduled'/);
        assert.match(app, /state === 'error'/);
        assert.match(app, /entry_point: monitor\.entryPoint/);
        assert.match(mqttProtocol, /#pending\.get\(data\.command_id\)/);
        assert.match(mqttProtocol, /parsed\.suffix === 'commands\/accepted'/);
        assert.match(mqttProtocol, /parsed\.suffix === 'commands\/rejected'/);
        assert.match(app, /\.\.\.connectionParams\(\)/);
    });

    it('tracks abandonment and only reports valid download targets', () => {
        assert.match(app, /track\('game_abandon'/);
        assert.match(app, /reportGameAbandon\('restart'\)/);
        assert.match(app, /reportGameAbandon\('route_change'\)/);
        assert.match(app, /reportGameAbandon\('page_exit'\)/);
        assert.match(app, /if \(!link \|\| !flash\.downloadReady\) return;/);
        const sdkBuilder = read('.github/scripts/stage_web_sdk.py');
        assert.match(sdkBuilder, /data-sdk-channel=/);
        assert.match(sdkBuilder, /data-sdk-format=/);
    });

    it('keeps sensitive browser-tool values out of analytics calls', () => {
        const analyticsCalls = [...app.matchAll(/track\([^;]+\);/gs)].map((match) => match[0]).join('\n');
        assert.doesNotMatch(
            analyticsCalls,
            /ssid|bssid|password|mqtt_host|mqtt_username|topic_prefix|device_id|payload/
        );
    });
});

describe('website accessibility and navigation', () => {
    it('keeps one route registry aligned with the SPA pages and static paths', () => {
        const registeredRoutes = [...routeRegistry.matchAll(/\{ name: '([^']+)'/g)]
            .map((match) => match[1])
            .sort();
        const pageRoutes = [...index.matchAll(/<main\b[^>]*\bdata-page="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredRoutes, pageRoutes);

        const registeredStaticPaths = [...routeRegistry.matchAll(/staticPath: '([^']+)'/g)]
            .map((match) => match[1])
            .sort();
        const pageStaticPaths = [...index.matchAll(/\bdata-static-url="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredStaticPaths, pageStaticPaths);
        assert.doesNotMatch(app, /const (?:NAV_GROUPS|ROUTES|STATIC_PAGE_ROUTES)\b/);
        assert.doesNotMatch(read('docs/web/assets/js/analytics.js'), /_OVERRIDES|_BY_PATH/);
        assert.match(read('.github/scripts/build_static_pages.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
    });

    it('has a responsive navigation control and a live status region', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
        assert.match(index, /class="toast toast-sticky js-demo-toast"[^>]+role="status"/);
        assert.match(app, /function syncDemoToast/);
        assert.match(app, /el\.hidden = !\(conn\.mode === 'demo' && conn\.status === 'connected'\)/);
        assert.match(styles, /\.toast\.js-toast:not\(\[hidden\]\) ~ \.toast-sticky:not\(\[hidden\]\)/);
    });

    it('provides skip navigation, stable tool headings, and route focus management', () => {
        assert.match(index, /<a class="skip-link" href="#main-content">Skip to content<\/a>/);
        assert.match(index, /data-page="home" id="main-content" tabindex="-1"/);
        for (const heading of ['Configure', 'Monitor', 'Motion theremin', 'Motion reaction game']) {
            assert.match(index, new RegExp(`<h1 class="page-title">${heading}<\\/h1>`));
        }
        assert.match(app, /link\.setAttribute\('aria-current', 'page'\)/);
        assert.match(app, /target\.focus\(\{ preventScroll: true \}\)/);
        assert.match(app, /page\.id = 'main-content'/);
        for (const path of [
            '.github/scripts/build_static_pages.py',
            '.github/scripts/stage_web_sdk.py',
            'docs/web/404.html',
        ]) {
            const source = read(path);
            assert.match(source, /class=\"skip-link\" href=\"#main-content\"/);
            assert.match(source, /id=\"main-content\" tabindex=\"-1\"/);
        }
    });

    it('associates every form label with a control', () => {
        const labels = [...index.matchAll(/<label\b([^>]*)>/g)];
        assert.ok(labels.length > 10);
        for (const [, attributes] of labels) assert.match(attributes, /\bfor="[^"]+"/);
    });
});

describe('website UX and content contracts', () => {
    it('formats security guidance with project-native components', () => {
        assert.match(security, /class="security-guidelines">\s*<ul>/);
        assert.match(security, /Canonical MQTT sensing messages omit raw CSI, Wi-Fi names, BSSIDs, local IP and MAC addresses/);
        assert.match(security, /Minimized does not mean anonymous/);
        assert.doesNotMatch(security, /Default services should not collect/);
        assert.match(styles, /\.security-guidelines ul \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
        assert.match(styles, /\.security-page \.note \{[^}]*background: var\(--accent-soft\);[^}]*border: 1px solid var\(--accent-line\);/);
        assert.match(styles, /\.security-page \.docs-start-copy p \+ p \{ margin-top: 12px; \}/);
        assert.match(styles, /\.security-page \.docs-path > :is\(\.btn-primary, \.btn-secondary\) \{[^}]*margin-top: auto;/);
        assert.match(security, /class="docs-paths security-reporting-paths">/);
        assert.doesNotMatch(security, /<span class="docs-path-label">POLICY<\/span>|Read the full scope|Read SECURITY\.md/);
        assert.match(styles, /\.security-reporting-paths \{ grid-template-columns: repeat\(2, minmax\(0, 1fr\)\); \}/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.security-guidelines ul \{ grid-template-columns: 1fr; \}/);
    });

    it('uses natural scrolling and progressively loads narrative images', () => {
        assert.match(index, /data-src-mobile="\/assets\/images\/home\/scene-smart-heating-mobile\.webp"/);
        assert.match(index, /data-src="\/assets\/images\/home\/scene-embedded-sdk\.jpg" data-src-mobile="\/assets\/images\/home\/scene-embedded-sdk-mobile\.webp"/);
        assert.match(app, /image\.dataset\.srcMobile/);
        const sceneIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-scene\b[^"]*" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        const captionIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-caption\b[^"]*" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        const markerIds = [...index.matchAll(/class="js-scrolly-marker" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        assert.deepEqual(sceneIds, Array.from({ length: 6 }, (_, index) => index));
        assert.deepEqual(captionIds, sceneIds);
        assert.deepEqual(markerIds, sceneIds.slice(1));
        assert.match(index, /<span>\/ 05<\/span>/);
        assert.match(index, /ESPectre detects movement in a room by sensing tiny changes in the Wi-Fi waves already traveling through it\./);
        assert.match(index, /<h2>Privacy first\.<br>By design\.<\/h2>/);
        assert.match(index, /schools, gyms, offices, public venues, and places of worship/);
        assert.match(index, /data-scene="1" aria-hidden="true" inert/);
        assert.match(app, /el\.toggleAttribute\('inert', !isActive\)/);
        assert.match(app, /el\.setAttribute\('aria-hidden', String\(!isActive\)\)/);
        assert.match(app, /event\.key !== 'ArrowDown' && event\.key !== 'ArrowUp'/);
        assert.match(app, /target\.closest\('a, button, input, select, textarea, \[contenteditable="true"\]'\)/);
        assert.match(app, /document\.addEventListener\('keydown', scrollyHandleKeydown\)/);
        assert.match(app, /sceneProgress = \(nextScene \+ 0\.5\) \/ sceneCount/);
        assert.doesNotMatch(index, /class="hero-skip"/);
        assert.match(index, /href="#get-started" class="scrolly-skip">Skip the story/);
        assert.match(index, /<section class="home-action-hub" id="get-started" aria-labelledby="home-action-title">/);
        const actionHub = index.match(/<section class="home-action-hub"[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(actionHub, /<header class="home-action-head">\s*<h2 class="page-title" id="home-action-title">Choose what you want to do\.<\/h2>\s*<p class="page-sub">/);
        assert.match(actionHub, /class="home-action-group">\s*<span class="home-kicker">START HERE<\/span>\s*<div class="home-tool-grid"/);
        assert.match(actionHub, /href="#flash" class="home-tool-card home-tool-card-primary"/);
        assert.match(actionHub, /href="#configure" class="home-tool-card"/);
        assert.match(actionHub, /href="#monitor" class="home-tool-card"/);
        assert.match(actionHub, /class="home-action-group">\s*<span class="home-kicker">COMMERCIAL LICENSING<\/span>\s*<aside class="home-license-cta"/);
        assert.match(actionHub, /class="home-action-group">\s*<span class="home-kicker">EXPLORE<\/span>\s*<div class="home-resource-strip"/);
        const licenseCard = actionHub.match(/<aside class="home-license-cta"[\s\S]*?<\/aside>/)?.[0] || '';
        assert.doesNotMatch(licenseCard, /home-kicker/);
        assert.match(actionHub, /class="home-license-cta"[\s\S]*?class="home-resource-strip"/);
        assert.match(actionHub, /class="home-resource-links">\s*<a href="#tools"><strong>Tools<\/strong>[\s\S]*?href="\/guides\/"[\s\S]*?<strong>Guides<\/strong>[\s\S]*?href="\/media\/"[\s\S]*?<strong>Media<\/strong>[\s\S]*?href="\/roadmap\/"[\s\S]*?<strong>Roadmap<\/strong>[\s\S]*?href="\/docs\/"[\s\S]*?<strong>Docs<\/strong><span>SDK integration, API, and examples →<\/span>[\s\S]*?href="https:\/\/github\.com\/francescopace\/espectre" target="_blank" rel="noopener"><strong>GitHub<\/strong>/);
        assert.match(styles, /\.home-resource-links \{ display: grid; grid-template-columns: repeat\(6, minmax\(0, 1fr\)\); \}/);
        assert.doesNotMatch(actionHub, /home-resource-intro/);
        assert.match(styles, /\.home-license-cta \{[\s\S]*?background: var\(--surface\);/);
        assert.match(actionHub, /href="\/licensing\/" class="btn-primary">Explore commercial licensing →/);
        assert.doesNotMatch(actionHub, /js-start-detection|js-demo|or try the demo/);
        assert.doesNotMatch(index, /home-(?:after-story|commercial|path|quick-links)/);
        assert.match(styles, /\.home-action-hub \{[^}]*min-height: 100svh/);
        assert.match(styles, /\.home-action-inner \{[^}]*min-height: 100svh;[^}]*padding: calc\(var\(--header-height\) \+ 64px\) 40px 40px;[^}]*justify-content: center;/);
        assert.doesNotMatch(index, /home-privacy-grid/);
    });

    it('labels research and preview concepts without presenting simulated evidence', () => {
        assert.doesNotMatch(index, /Roadmap · Breathing/);
        assert.doesNotMatch(index, /13\.2 <em>cycles\/min<\/em>/);
        assert.match(index, /Matter · Limited validation/);
        assert.match(index, /Controller-dependent/);
        assert.doesNotMatch(index, /ESP-IDF 5\.1\+/);
        assert.doesNotMatch(index, /Apple Home|Google Home/);
        const breathingCard = index.match(/<h2>Breathing research<\/h2>[\s\S]*?<\/a>/)?.[0] || '';
        assert.match(breathingCard, /ROADMAP/);
        assert.doesNotMatch(breathingCard, /js-ble-chip/);
    });

    it('enforces the supported desktop, Android, and iOS capability matrix', () => {
        const detect = (navigator) => {
            const context = { window: { navigator } };
            runInNewContext(browserSupportSource, context);
            return context.window.ESPectreBrowserSupport.current;
        };
        const bluetooth = { requestDevice() {} };
        const serial = { requestPort() {} };
        const desktop = detect({ userAgent: 'Chrome', platform: 'Linux x86_64', bluetooth, serial });
        assert.equal(desktop.bluetooth, true);
        assert.equal(desktop.flash, true);
        const android = detect({ userAgent: 'Chrome Android Mobile', platform: 'Linux armv8', bluetooth, serial });
        assert.equal(android.bluetooth, true);
        assert.equal(android.flash, false);
        const ios = detect({ userAgent: 'CriOS iPhone Mobile', platform: 'iPhone', bluetooth, serial });
        assert.equal(ios.bluetooth, false);
        assert.equal(ios.flash, false);
        assert.match(app, /installTrigger\.disabled = !browserSupport\.flash/);
        assert.match(app, /button\.disabled = !browserSupport\.bluetooth/);
        assert.match(app, /if \(browserSupport\.flash\) \{\s*loadBrowserDependency/);
        assert.match(index, /class="link-btn js-demo">or try the demo/);
        assert.match(styles, /\.link-btn \{[\s\S]*?color: var\(--accent\);[\s\S]*?text-decoration: none;/);
        assert.match(styles, /\.link-btn:visited \{ color: var\(--accent\); \}/);
        assert.match(styles, /\.link-btn:hover \{ color: var\(--accent\); text-decoration: underline; \}/);
        assert.match(styles, /min-height: 100dvh/);
        assert.match(styles, /touch-action: manipulation/);
        const mobileCss = styles.split('@media (max-width: 720px)')[1]
            .split('@media (max-width: 480px)')[0];
        assert.match(mobileCss, /\.field input:not\(\[type="checkbox"\]\):not\(\[type="range"\]\)/);
        assert.match(mobileCss, /min-height: 44px/);
        assert.match(mobileCss, /font-size: 16px/);
        assert.doesNotMatch(index, /js-mqtt-support|js-browser-broker-fields|js-scrolly-progress/);
    });

    it('keeps privacy discoverable and serves a real 404 page', () => {
        assert.match(index, /data-page="privacy"/);
        assert.match(index, /data-content-url="content\/privacy\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<div class="footer-links">\s*<a href="#privacy">Privacy<\/a>/);
        assert.match(routeRegistry, /name: 'privacy'.*staticPath: '\/privacy\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/privacy\/">Privacy<\/a>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/privacy\/">Privacy<\/a>/);
        const sitemap = read('.github/scripts/sitemap.template.xml');
        assert.match(sitemap, /https:\/\/espectre\.dev\/privacy\//);
        assert.doesNotMatch(sitemap, /<(?:changefreq|lastmod)>/);
        assert.match(read('docs/web/content/privacy.html'), /Never included:/);
        const notFound = read('docs/web/404.html');
        assert.doesNotMatch(notFound, /http-equiv="refresh"|location\.replace/);
        assert.match(notFound, /404 · PAGE NOT FOUND/);
        assert.match(notFound, /<footer class="site-footer">/);
        assert.match(styles, /body \{[\s\S]*?display: flex;[\s\S]*?flex-direction: column;[\s\S]*?min-height: 100dvh;/);
        assert.match(styles, /body > main \{[\s\S]*?width: 100%;[\s\S]*?box-sizing: border-box;[\s\S]*?flex: 1 0 auto;/);
        const sharedFooterBrand = /<div class="footer-brand">\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg(?:\?v=(?:[0-9a-f]{12}|\{logo_version\}))?" alt="" width="23" height="23" aria-hidden="true">\s*ESPectre © 2026 · Open source Wi-Fi sensing platform\s*<\/div>/;
        assert.match(index, sharedFooterBrand);
        assert.match(notFound, sharedFooterBrand);
        assert.match(read('.github/scripts/build_static_pages.py'), sharedFooterBrand);
        assert.match(read('.github/scripts/stage_web_sdk.py'), sharedFooterBrand);
        assert.match(styles, /\.footer-brand \{[^}]*color: var\(--text\);/);
        assert.doesNotMatch(read('.github/scripts/build_static_pages.py'), /footer-brand[\s\S]*?GPLv3 \+ commercial licensing/);
        assert.match(notFound, /data-static-page data-site-section="other"/);
        assert.match(notFound, /<a href="\/privacy\/#cookie-settings" class="js-cookie-settings">Cookie settings<\/a>/);
        assert.doesNotMatch(notFound, /footer-link-button/);
        assert.match(styles, /\.footer-links a,\s*\.footer-links a:visited \{[\s\S]*?display: inline-flex;[\s\S]*?color: var\(--text\);[\s\S]*?text-decoration: none;/);
        assert.match(styles, /\.footer-links a:hover,\s*\.footer-links a:focus-visible \{ color: var\(--accent\); text-decoration: none; \}/);
        assert.match(read('docs/web/content/privacy.html'), /<h2 id="cookie-settings">Your choice<\/h2>/);
        assert.match(read('docs/web/assets/js/analytics.js'), /document\.querySelectorAll\('\.js-cookie-settings'\)[\s\S]*?event\.preventDefault\(\);[\s\S]*?showConsentBanner\(\);/);
        assert.match(notFound, /class="consent-banner js-consent-banner"/);
        const notFoundScripts = [...notFound.matchAll(/<script\b([^>]*)>/g)]
            .map((match) => match[1]);
        assert.deepEqual(
            notFoundScripts.map((attrs) => attrs.match(/src="(\/assets\/js\/[^"?]+)/)[1]),
            [
                '/assets/js/route-registry.js',
                '/assets/js/navigation.js',
                '/assets/js/analytics.js'
            ]
        );
        for (const attrs of notFoundScripts) {
            assert.match(attrs, /\bdefer\b/, `expected defer on ${attrs.trim()}`);
        }
    });

    it('publishes a dedicated commercial licensing page', () => {
        const licensingContent = read('docs/web/content/licensing.html');
        assert.match(index, /data-page="licensing"/);
        assert.match(index, /data-content-url="content\/licensing\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<a href="\/licensing\/">Licensing<\/a>/);
        assert.match(index, /<a href="\/contact\/">Contact<\/a>/);
        assert.match(routeRegistry, /name: 'licensing'.*staticPath: '\/licensing\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/licensing\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/licensing\/">Licensing<\/a>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/licensing\/">Licensing<\/a>/);
        assert.match(read('docs/web/404.html'), /<a href="\/licensing\/">Licensing<\/a>/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/licensing\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/licensing\/": \(Path\("docs\/web\/content\/licensing\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(licensingContent, /<h1 class="page-title">Commercial licensing<\/h1>/);
        assert.match(licensingContent, /ESPHome remains GPLv3/);
        assert.match(licensingContent, /mailto:contact@espectre\.dev\?subject=Commercial%20licensing%20inquiry/);
    });

    it('publishes a dedicated contact page from every footer', () => {
        const contactContent = read('docs/web/content/contact.html');
        assert.match(index, /data-page="contact"/);
        assert.match(index, /data-content-url="content\/contact\.html\?v=[0-9a-f]{12}"/);
        assert.match(routeRegistry, /name: 'contact'.*staticPath: '\/contact\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/contact\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/contact\/">Contact<\/a>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/contact\/">Contact<\/a>/);
        assert.match(read('docs/web/404.html'), /<a href="\/contact\/">Contact<\/a>/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/contact\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/contact\/": \(Path\("docs\/web\/content\/contact\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(contactContent, /<h1 class="page-title">Contact ESPectre<\/h1>/);
        assert.match(contactContent, /mailto:contact@espectre\.dev/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/discussions/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/issues/);
        assert.doesNotMatch(contactContent, /mailto:security@espectre\.dev/);
    });

    it('publishes a dedicated security and responsible-use page', () => {
        const securityContent = read('docs/web/content/security.html');
        assert.match(index, /data-page="security"/);
        assert.match(index, /data-content-url="content\/security\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<a href="\/security\/">Security<\/a>/);
        assert.match(routeRegistry, /name: 'security'.*staticPath: '\/security\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/security\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/security\/">Security<\/a>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/security\/">Security<\/a>/);
        assert.match(read('docs/web/404.html'), /<a href="\/security\/">Security<\/a>/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/security\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/security\/": \(Path\("docs\/web\/content\/security\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(securityContent, /<h1 class="page-title">Security and responsible use<\/h1>/);
        assert.match(securityContent, /does not enable Wi-Fi promiscuous mode/);
        assert.match(securityContent, /association is a deliberate safeguard, not proof of consent or authority/);
        assert.match(securityContent, /Illegal or unethical use is not acceptable/);
        assert.match(securityContent, /competent law enforcement, regulatory, or data protection authority/);
        assert.match(securityContent, /mailto:contact@espectre\.dev\?subject=Responsible%20use%20or%20abuse%20report/);
        assert.match(securityContent, /mailto:security@espectre\.dev/);
        assert.match(securityContent, /github\.com\/francescopace\/espectre\/security/);
        assert.doesNotMatch(securityContent, /Read the full scope|github\.com\/francescopace\/espectre\/blob\/main\/SECURITY\.md/);
    });

    it('publishes website terms and current legal information', () => {
        const termsContent = read('docs/web/content/terms.html');
        const legalContent = read('docs/web/content/legal.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const sdkPageBuilder = read('.github/scripts/stage_web_sdk.py');
        const notFound = read('docs/web/404.html');
        const sitemap = read('.github/scripts/sitemap.template.xml');
        const sitemapBuilder = read('.github/scripts/build_sitemap.py');

        assert.match(index, /data-page="terms"/);
        assert.match(index, /data-content-url="content\/terms\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /data-page="legal"/);
        assert.match(index, /data-content-url="content\/legal\.html\?v=[0-9a-f]{12}"/);
        assert.match(routeRegistry, /name: 'terms'.*staticPath: '\/terms\/'/);
        assert.match(routeRegistry, /name: 'legal'.*staticPath: '\/legal\/'/);

        for (const source of [index, staticPageBuilder, sdkPageBuilder, notFound]) {
            assert.match(source, /<a href="\/terms\/">Terms<\/a>/);
            assert.match(source, /<a href="\/legal\/">Legal<\/a>/);
        }
        assert.match(staticPageBuilder, /"source": "content\/terms\.html"/);
        assert.match(staticPageBuilder, /"source": "content\/legal\.html"/);
        assert.match(sitemap, /https:\/\/espectre\.dev\/terms\//);
        assert.match(sitemap, /https:\/\/espectre\.dev\/legal\//);
        assert.match(sitemapBuilder, /"\/terms\/": \(Path\("docs\/web\/content\/terms\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(sitemapBuilder, /"\/legal\/": \(Path\("docs\/web\/content\/legal\.html"\), STATIC_PAGE_BUILDER\)/);

        assert.match(termsContent, /<h1 class="page-title">Terms of use<\/h1>/);
        assert.match(termsContent, /operated by Francesco Pace, a natural person in Italy/);
        assert.match(termsContent, /does not currently provide checkout, payment, ordering, or automatic contract formation/);
        assert.match(termsContent, /mandatory protections or jurisdiction rules/);
        assert.match(legalContent, /<h1 class="page-title">Legal information<\/h1>/);
        assert.match(legalContent, /<dt>Name<\/dt><dd>Francesco Pace<\/dd>/);
        assert.match(legalContent, /<dt>Legal form<\/dt><dd>Natural person<\/dd>/);
        assert.match(legalContent, /<dt>Primary contact<\/dt><dd><a href="mailto:contact@espectre\.dev">contact@espectre\.dev<\/a><\/dd>/);
        assert.match(legalContent, /ESPectre is available through official resellers/);
        assert.match(legalContent, /For current reseller information, product availability, commercial licensing, integration, or support options/);
        assert.doesNotMatch(legalContent, /Current project status|not operated through an incorporated company|francesco\.pace@espectre\.dev|security@espectre\.dev|href="\/security\/"/);
    });

    it('treats top-level docs, roadmap, privacy, terms, legal, security, licensing, and contact as pages, not articles', () => {
        const docsContent = read('docs/web/content/docs.html');
        const roadmapContent = read('docs/web/content/roadmap.html');
        const privacyContent = read('docs/web/content/privacy.html');
        const termsContent = read('docs/web/content/terms.html');
        const legalContent = read('docs/web/content/legal.html');
        const securityContent = read('docs/web/content/security.html');
        const licensingContent = read('docs/web/content/licensing.html');
        const contactContent = read('docs/web/content/contact.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.ok(docsContent.startsWith(`${GPL_HTML_HEADER}<div class="docs-quickstart">`));
        assert.ok(roadmapContent.startsWith(`${GPL_HTML_HEADER}<div class="roadmap-page">`));
        assert.ok(privacyContent.startsWith(`${GPL_HTML_HEADER}<div class="privacy-page">`));
        assert.ok(termsContent.startsWith(`${GPL_HTML_HEADER}<div class="terms-page">`));
        assert.ok(legalContent.startsWith(`${GPL_HTML_HEADER}<div class="legal-page">`));
        assert.ok(securityContent.startsWith(`${GPL_HTML_HEADER}<div class="security-page">`));
        assert.ok(licensingContent.startsWith(`${GPL_HTML_HEADER}<div class="licensing-page">`));
        assert.ok(contactContent.startsWith(`${GPL_HTML_HEADER}<div class="contact-page">`));
        assert.match(index, /<main class="js-page page-narrow" data-page="roadmap"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="privacy"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="terms"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="legal"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="security"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="licensing"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="contact"/);
        assert.doesNotMatch(index, /<main class="js-page page-narrow page-article" data-page="(?:docs|roadmap|privacy|terms|legal|security|licensing|contact)"/);
        assert.match(staticPageBuilder, /"source": "content\/docs\.html",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/roadmap\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/privacy\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/terms\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/legal\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/security\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/licensing\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/contact\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /<meta property="og:type" content="\{og_type\}">/);
    });

    it('uses the shared page heading styles on every top-level inner page', () => {
        for (const path of ['guides', 'docs', 'roadmap', 'privacy', 'terms', 'legal', 'security', 'licensing', 'contact']) {
            const content = read(`docs/web/content/${path}.html`);
            assert.match(content, /<h1 class="page-title">/);
            assert.match(content, /<p class="page-sub">/);
        }
        assert.match(styles, /\.page-title \{ font-size: 40px;/);
        const pageSubRule = styles.match(/\.page-sub \{([^}]*)\}/)?.[1] || '';
        assert.match(pageSubRule, /font-size: 18px;/);
        assert.match(pageSubRule, /line-height: 1\.55;/);
        assert.match(styles, /@media \(max-width: 720px\) \{\s*\.page-title \{ font-size: 36px; \}\s*\.page-sub \{ font-size: 17px; \}/);
    });

    it('uses the widest inner-page measure for every footer information page', () => {
        assert.match(styles, /\.page-narrow \{[\s\S]*?max-width: 1120px;/);
        assert.match(styles, /\.privacy-page > \.article,\s*\.terms-page > \.article,\s*\.legal-page > \.article \{\s*max-width: none;\s*margin: 0;/);
    });

    it('gives the docs landing page a clear start-to-reference hierarchy', () => {
        const docsContent = read('docs/web/content/docs.html');
        assert.match(docsContent, /<h1 class="page-title">Docs<\/h1>/);
        assert.match(docsContent, /<section class="docs-start" aria-labelledby="docs-start-title">/);
        assert.match(docsContent, /<section class="docs-section" aria-labelledby="docs-paths-title">/);
        assert.match(docsContent, /<section class="docs-section" aria-labelledby="docs-quick-start-title">/);
        assert.match(docsContent, /<section class="docs-next" aria-labelledby="docs-next-title">/);
        assert.ok(docsContent.indexOf('class="docs-start"') < docsContent.indexOf('class="docs-paths"'));
        assert.ok(docsContent.indexOf('class="docs-paths"') < docsContent.indexOf('class="docs-steps"'));
        assert.ok(docsContent.indexOf('class="docs-steps"') < docsContent.indexOf('class="docs-next"'));
        const pathCards = docsContent.match(/<div class="docs-path(?: docs-path-recommended)?">[\s\S]*?<\/div>/g) || [];
        assert.equal(pathCards.length, 3);
        for (const card of pathCards) {
            assert.match(card, /<h3>/);
            assert.doesNotMatch(card, /<h2>/);
        }
    });

    it('publishes the detection profile guide through SPA and static routes', () => {
        const guide = read('docs/web/content/guides/detectors.html');
        assert.match(guide, /<h1>Choose your detection profile<\/h1>/);
        assert.match(guide, /Lightweight Detection/);
        assert.match(guide, /High-Accuracy Detection/);
        assert.match(index, /data-page="guide-detectors"/);
        assert.match(routeRegistry, /name: 'guide-detectors'.*staticPath: '\/guides\/detectors\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/detectors\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/detectors\//);
    });

    it('publishes the MicroPython contribution and runtime guide', () => {
        const guide = read('docs/web/content/guides/micropython.html');
        assert.match(guide, /<h1>Run ESPectre on MicroPython<\/h1>/);
        assert.match(guide, /ESPectre brought ESP32 CSI to mainline MicroPython/);
        assert.match(guide, /micropython\/micropython\/pull\/18460/);
        assert.doesNotMatch(index, /OPEN-SOURCE INFRASTRUCTURE/);
        assert.match(read('docs/web/content/docs.html'), /ESPectre also runs sensing directly in MicroPython/);
        assert.match(read('docs/web/content/guides.html'), /micropython-csi-runtime-card\.avif/);
        assert.match(guide, /micropython-csi-runtime\.webp/);
        assert.match(index, /data-page="guide-micropython"/);
        assert.match(routeRegistry, /name: 'guide-micropython'.*staticPath: '\/guides\/micropython\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/micropython\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/micropython\//);
    });

    it('adds anchor navigation and intrinsic image sizes to long guides', () => {
        const longPages = [
            'docs/web/content/docs.html',
            'docs/web/content/guides/setup.html',
            'docs/web/content/guides/firmware.html',
            'docs/web/content/guides/hardware.html',
            'docs/web/content/guides/placement.html',
            'docs/web/content/guides/detection.html',
            'docs/web/content/guides/detectors.html',
            'docs/web/content/guides/micropython.html',
        ];
        for (const path of longPages) {
            const content = read(path);
            assert.match(content, /<details class="page-toc" open>/);
            assert.match(content, /<nav aria-label="On this page">/);
        }
        for (const path of longPages.filter((path) => path.includes('/guides/'))) {
            const images = [...read(path).matchAll(/<img\b[^>]*>/g)].map((match) => match[0]);
            for (const image of images) {
                assert.match(image, /\bwidth="\d+"/);
                assert.match(image, /\bheight="\d+"/);
            }
        }
        assert.match(styles, /\.page-toc \{/);
        assert.match(read('docs/web/content/guides/detection.html'), /Raw CSI leaves the device only when you deliberately use the separate Streamer research workflow/);
    });

    it('loads generated firmware and SDK output from the shared artifacts tree', () => {
        assert.match(app, /\/artifacts\/firmware\//);
        assert.doesNotMatch(app, /\/flash\/firmware\//);
        assert.match(index, /id="flash-channel"/);
        assert.doesNotMatch(index, /id="flash-chip"/);
        assert.match(index, /js-flash-requirement/);
        assert.match(index, /USB flashing requires desktop Chrome or Edge/);
        assert.match(index, /js-flash-chip-downloads/);
        assert.match(index, /js-flash-next/);
        assert.doesNotMatch(index, /js-flash-download/);
        assert.doesNotMatch(index, /js-matter-panel/);
        assert.match(index, /<option value="release">Latest Release<\/option>/);
        assert.match(index, /<option value="preview">Release Preview<\/option>/);
        assert.match(index, /<option value="develop">Development<\/option>/);
        assert.match(app, /updateReleaseBadge[\s\S]*flashLoadManifest\('release'\)/);
        assert.doesNotMatch(app, /flashLoadManifest\('stable'\)/);
        assert.match(app, /builds: artifacts\.map\(\(artifact\) => \(\{/);
        assert.match(app, /chipFamily: artifact\.chip_family/);
        assert.match(app, /FLASH_CHIP_UNSUPPORTED_RE/);
        assert.match(app, /report\('unsupported'\)/);
        assert.match(app, /Published firmware is available for/);
        assert.match(app, /function flashRenderDownloads/);
        assert.match(app, /function flashSetNextStep/);
        assert.match(app, /function flashNextActionLink[\s\S]*document\.createElement\('a'\)/);
        assert.match(app, /flashNextActionLink\('Read the onboarding QR over USB', 'js-matter-read'\)/);
        assert.doesNotMatch(app, /function flashNextAction\(/);
        assert.match(app, /action\.getAttribute\('aria-disabled'\) === 'true'/);
        assert.match(index, /class="modal-backdrop js-matter-modal" hidden>[\s\S]*class="modal-card matter-modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /id="matter-modal-title">Commission this device<\/h2>/);
        assert.match(index, /class="matter-result js-matter-result" hidden>/);
        assert.match(app, /function matterOpen\(returnFocus\)/);
        assert.match(app, /matterOpen\(trigger\)/);
        assert.match(app, /if \(!\$\('\.js-matter-modal'\)\.hidden\) matterClose\(\)/);
        assert.match(app, /track\('matter_qr_read'/);
        assert.match(app, /installButton\.toggleAttribute\('inert', !browserSupport\.flash\)/);
        const setupGuide = read('docs/web/content/guides/setup.html');
        assert.match(setupGuide, /The installer detects the chip over USB/);
        assert.match(setupGuide, /\.\/espectre matter qr/);
        assert.match(setupGuide, /leave Device ID empty to scan <code>info<\/code> and <code>status<\/code>/);
        assert.match(setupGuide, /\.\/espectre mqtt/);
        assert.doesNotMatch(setupGuide, /your chip family/);
        assert.doesNotMatch(setupGuide, /the flasher shows the device's Matter QR/);
        const docsContent = read('docs/web/content/docs.html');
        assert.match(docsContent, /href="\/artifacts\/sdk\/release\/"/);
        assert.match(docsContent, /<details class="sdk-download">[\s\S]*?<summary class="btn-primary">Download SDK<\/summary>/);
        assert.match(docsContent, /href="\/artifacts\/sdk\/preview\/"/);
        assert.match(docsContent, /href="\/artifacts\/sdk\/develop\/"/);
        assert.match(docsContent, /data-sdk-version="release"/);
        assert.match(docsContent, /data-sdk-version="preview"/);
        assert.match(docsContent, /data-sdk-version="develop"/);
        assert.match(read('docs/web/assets/js/navigation.js'), /sdk-manifest-\$\{channel\}\.json/);
        assert.match(read('docs/web/assets/js/navigation.js'), /label\.textContent = `Version \$\{version\}`/);
        assert.match(read('docs/web/assets/js/navigation.js'), /details\.sdk-download\[open\][\s\S]*?!menu\.contains\(event\.target\)[\s\S]*?menu\.open = false/);
        assert.match(app, /window\.initSdkDownloadVersions\(container\)/);
        assert.doesNotMatch(docsContent, /Rolling bundles:/);
        assert.match(docsContent, /href="\/artifacts\/sdk\/api\/"/);
        assert.doesNotMatch(docsContent, /href="\/sdk\//);
        assert.match(read('docs/web/.gitignore'), /^\/artifacts\/$/m);
        for (const path of [
            'docs/web/content/docs.html',
            'docs/web/content/docs/api.html',
            'docs/web/content/docs/architecture.html',
            'docs/web/content/docs/examples.html',
        ]) {
            assert.doesNotMatch(read(path), /href="\/sdk\/api(?:\/|")/);
        }
    });

    it('maps Bluetooth capabilities, runtime controls, and dual-band Wi-Fi safely', () => {
        const ble = index.match(/data-page="configure"[\s\S]*?<\/main>/)?.[0] || '';
        const mqtt = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const onboarding = ble.match(/class="js-configure-onboarding"[\s\S]*?<div class="js-configure-workspace"/)?.[0] || '';
        const bleBanner = ble.match(/class="device-banner-actions"[\s\S]*?<\/div>/)?.[0] || '';
        const mqttBanner = mqtt.match(/class="device-banner-actions"[\s\S]*?<\/div>/)?.[0] || '';
        assert.match(index, /data-capability="supports_wifi_provisioning"/);
        assert.match(index, /class="field-row wifi-credentials-row">\s*<div class="field"><label for="cfg-ssid"[\s\S]*?<label for="cfg-wifi-pass"/);
        assert.match(index, /class="field-row field-row-2-1">\s*<div class="field"><label for="cfg-wifi-band"[\s\S]*?<label for="cfg-channel"/);
        assert.match(index, /id="cfg-wifi-band" disabled/);
        assert.match(app, /snapshot\.supports_wifi_5ghz/);
        assert.match(app, /select\.disabled = select\.options\.length === 1/);
        assert.match(app, /wifiBandPolicyAvailable \? \{ bandPolicy \}/);
        assert.match(index, /class="conn-dropdown-meta"/);
        assert.match(index, /js-menu-chip[\s\S]*js-menu-device-id[\s\S]*js-menu-firmware/);
        assert.match(index, /class="mono-sub device-banner-identity"/);
        assert.match(index, /js-device-banner-sub[\s\S]*js-firmware-update-notice/);
        assert.equal([...index.matchAll(/class="device-firmware-update js-firmware-update-notice"/g)].length, 3);
        assert.match(index, /class="conn-dropdown-name js-device-name"/);
        assert.match(app, /function formatDeviceIdentityLine/);
        assert.match(app, /parts\.push\('Chip ' \+ chip\)/);
        assert.match(app, /parts\.push\('Device ID ' \+ deviceId\)/);
        assert.match(app, /parts\.push\('Firmware ' \+ firmware\)/);
        assert.match(app, /conn\.deviceBannerSub = deviceIdentity/);
        assert.match(app, /write\('\.js-menu-chip', conn\.chip\)/);
        assert.match(app, /write\('\.js-menu-device-id', conn\.deviceId\)/);
        assert.match(app, /write\('\.js-menu-firmware', conn\.firmwareVersion\)/);
        assert.match(index, /energy-title">CURRENT STATE/);
        assert.match(app, /conn\.motion \? 'MOTION' : 'IDLE'/);
        assert.match(styles, /\.conn-dropdown \{[^}]*color: var\(--text\);/);
        assert.match(styles, /\.conn-dropdown-name \{[^}]*color: var\(--text\);/);
        assert.match(app, /deviceName = 'Demo Device'/);
        assert.match(app, /toast\('Connecting to the broker…'\)/);
        assert.match(app, /toast\('Sensing is live\.'\)/);
        assert.match(app, /toast\('The broker is connected, but the device is offline\.'\)/);
        assert.match(mqtt, /data-device-view="live"/);
        assert.match(ble, /data-device-view="connectivity"/);
        assert.match(mqtt, /<h2>Sensing controls<\/h2>/);
        assert.match(mqtt, /stat-label">Movement score/);
        assert.match(mqtt, /live-calibration" data-mqtt-command="recalibrate"/);
        assert.match(app, /statusFn: toast/);
        assert.match(app, /mqttCommand === 'recalibrate' && detector !== 'lightweight'/);
        assert.match(app, /function beginCalibration/);
        assert.match(app, /button\.textContent = monitor\.calibrating \? 'Calibrating…' : 'Recalibrate'/);
        assert.match(app, /case 'ha\/calibrate\/state'/);
        assert.match(index, /js-sense-recalibrate">Recalibrate/);
        assert.match(mqtt, /<details class="device-live-diagnostics">/);
        assert.doesNotMatch(mqtt, /<details class="device-live-diagnostics" open/);
        assert.match(app, /const showLiveEnergy = live/);
        assert.match(app, /js-device-edit-connectivity'\)\.addEventListener\('click', monitorStartBle\)/);
        assert.match(app, /edit\.disabled = monitor\.closingBleForLive/);
        assert.match(app, /monitor\.closingBleForLive = false;[\s\S]*setStatus\('connected'\)/);
        assert.match(app, /await monitorConnect\(\)/);
        assert.match(app, /targetRoute = view === 'connectivity' \? 'configure' : 'monitor'/);
        assert.match(app, /location\.hash = '#' \+ targetRoute/);
        assert.match(app, /ble: 'configure'/);
        assert.match(app, /mqtt: 'monitor'/);
        assert.match(app, /device: 'configure'/);
        assert.match(app, /const bleSetup = connected && conn\.mode === 'ble'/);
        assert.match(app, /const bleConnecting = conn\.status === 'connecting'/);
        assert.match(app, /const mqttConnecting = conn\.status === 'connecting' && !bleConnecting/);
        assert.match(app, /bleConnecting \? 'Connecting…'/);
        assert.match(app, /function validateMonitorConnection\(\)/);
        assert.match(app, /input\.setAttribute\('aria-invalid', 'true'\)/);
        assert.match(app, /monitorStatus\(''\)/);
        assert.match(styles, /\.field input\.is-invalid/);
        assert.match(styles, /@keyframes espFieldErrorBlink/);
        assert.match(app, /bindMqttToConnection\(\)/);
        assert.match(app, /conn\.status !== 'connected'/);
        assert.match(app, /if \(!monitor\.handoffReady\) return/);
        assert.match(app, /monitor\.handoffReady = true/);
        assert.match(app, /function syncThresholdControl\(/);
        assert.match(app, /function bindThresholdControls/);
        assert.match(app, /function commitThreshold/);
        assert.match(app, /sense !== document\.activeElement/);
        assert.match(app, /conn\.threshold = threshold;[\s\S]*syncThresholdControl\(threshold\)/);
        assert.match(app, /function gameThreshold/);
        assert.match(app, /function snapshotGameThreshold/);
        assert.match(app, /gameThresholdOverride = conn\.threshold/);
        assert.match(app, /if \(gameThresholdOverride === null\) paintGameThresholdControl\(\)/);
        assert.match(app, /if \(target === 'game' && previousRoute !== 'game'\) snapshotGameThreshold\(\)/);
        assert.match(app, /gameThresholdOverride = threshold/);
        assert.doesNotMatch(app, /textContent = 'Restart';\s*snapshotGameThreshold\(\)/);
        assert.match(app, /getElementById\('game-threshold'\)/);
        assert.match(index, /id="game-threshold"/);
        assert.doesNotMatch(index, /data-page="game"[\s\S]*data-mqtt-command="set_threshold"/);
        assert.doesNotMatch(app, /gameSlider\.addEventListener\('change', \(\) => commitThreshold/);
        assert.match(app, /getElementById\('sense-detector'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-on'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-off'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-csi-mode'\)\.addEventListener\('change'/);
        assert.match(index, /<select id="sense-csi-mode"><option value="internal">Internal<\/option><option value="external">External<\/option><option value="disabled">Disabled<\/option><\/select>/);
        assert.match(app, /getElementById\('sense-generator-mode'\)\.addEventListener\('change'/);
        assert.match(index, /Hold the <strong>BOOT<\/strong> button for 3 seconds/);
        assert.match(index, /For first setup or recovery, connect over Bluetooth/);
        assert.match(index, /For an already configured device, connect over MQTT/);
        assert.match(index, /activate Bluetooth setup/);
        assert.doesNotMatch(index, /device-connect-kicker|activate BLE/);
        assert.doesNotMatch(styles, /\.device-connect-kicker/);
        assert.match(index, /class="js-configure-onboarding"/);
        assert.match(index, /class="js-monitor-onboarding"/);
        assert.match(index, /class="device-connect-card[^"]*" data-transport="ble"[\s\S]*Connect with Bluetooth/);
        assert.match(index, /class="device-connect-card[^"]*" data-transport="mqtt"[\s\S]*Connect with MQTT/);
        assert.ok(onboarding.indexOf('device-recovery-hint') < onboarding.indexOf('js-connect-ble'));
        assert.ok(onboarding.indexOf('js-connect-ble') < onboarding.indexOf('or connect over MQTT'));
        assert.match(onboarding, /href="#monitor"[^>]*>or connect over MQTT if already configured/);
        assert.match(onboarding, /Which of the two buttons is BOOT\?/);
        assert.match(onboarding, /EN \/ RST[\s\S]*BOOT \/ FLASH/);
        assert.match(onboarding, /Choose the button labelled <strong>BOOT<\/strong> or <strong>FLASH<\/strong>/);
        assert.match(bleBanner, /js-start-detection/);
        assert.match(bleBanner, />Start sensing</);
        assert.match(mqttBanner, /js-device-edit-connectivity/);
        assert.match(mqttBanner, />Edit connectivity</);
        assert.match(app, /if \(startBle\) startBle\.addEventListener\('click', monitorStartBle\)/);
        assert.match(app, /command: 'set_ble', ble: 'on'/);
        assert.match(app, /monitor\.bleRequested = true;[\s\S]*await connectBle\(\)/);
        assert.match(app, /command: 'set_ble', ble: 'off'/);
        assert.match(app, /function ensureBleOffForLive/);
        assert.match(index, /class="modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /class="btn-primary js-ota-start" disabled>Update device<\/button>/);
        assert.match(index, /id="cfg-ota-message"/);
        assert.match(app, /function applyOtaStatus/);
        assert.match(app, /function startSilentOtaCheck/);
        assert.match(app, /function currentOtaCheckTransport/);
        assert.match(app, /if \(conn\.mode === 'demo'\) return;/);
        assert.match(app, /if \(!manual && transport && otaCheckTransport === transport\) return;/);
        assert.match(app, /if \(conn\.mode === 'ble'\) startSilentOtaCheck\(\)/);
        assert.match(app, /transport === 'ble' && bleClient && typeof bleClient\.otaCheck === 'function'/);
        assert.match(app, /transport !== 'mqtt' \|\| !monitorIsMqttLive\(\)/);
        assert.match(index, /js-menu-firmware[\s\S]*js-firmware-update-notice[\s\S]*js-disconnect/);
        assert.match(index, /class="conn-firmware-row"/);
        assert.doesNotMatch(index, /device-firmware-update-icon/);
        assert.doesNotMatch(styles, /device-firmware-update-icon/);
        assert.match(app, /\$\$\('\.js-firmware-update-notice'\)\.forEach\(\(button\) => \{/);
        assert.match(app, /button\.addEventListener\('click', \(event\) => otaOpen\(event\.currentTarget\)\)/);
        assert.match(app, /copy = 'Checking for updates…'/);
        assert.match(app, /copy = 'Latest'/);
        assert.match(app, /status = 'error'/);
        assert.match(app, /message: 'Unable to check for updates'/);
        assert.match(app, /function otaOpen\(returnFocus\)/);
        assert.match(index, /id="ota-channel"/);
        assert.match(index, /id="ota-channel"[\s\S]*?<option value="release" selected>Latest Release<\/option>/);
        assert.doesNotMatch(index, /Firmware default/);
        assert.match(app, /function selectedOtaChannel/);
        assert.match(app, /return value \|\| 'release'/);
        assert.match(app, /return \{ command, channel: selectedOtaChannel\(\) \}/);
        assert.match(app, /return \{ channel: selectedOtaChannel\(\) \}/);
        assert.match(app, /if \(conn\.mode === 'demo'\) return;/);
        assert.match(index, /<h2 class="panel-title-status">Wi-Fi <span class="dot dot-idle js-wifi-status-dot"/);
        assert.match(index, /<h2 class="panel-title-status">MQTT <span class="dot dot-idle js-mqtt-status-dot"/);
        assert.match(app, /setConnectionDot\('\.js-wifi-status-dot', snapshot\.wifi_connected\)/);
        assert.match(app, /setConnectionDot\('\.js-mqtt-status-dot', snapshot\.mqtt_connected\)/);
        assert.match(styles, /\.dot-error \{ background: var\(--danger\); \}/);
    });

    it('keeps MQTT diagnostics collapsed below Live and hides demo when live', () => {
        const mqttPage = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const broker = mqttPage.match(/<section class="device-connect-card[^"]*" data-transport="mqtt">[\s\S]*?<\/section>/)?.[0] || '';
        const diagnostics = mqttPage.match(/<details class="device-live-diagnostics">[\s\S]*?<\/details>/)?.[0] || '';
        assert.match(broker, /<div class="fields">/);
        assert.match(broker, /js-mon-connect/);
        assert.match(broker, /class="tool-note js-mon-status" role="status" hidden><\/div>/);
        assert.match(diagnostics, /js-mon-diag-status/);
        assert.match(diagnostics, /js-mon-admitted/);
        assert.match(diagnostics, /js-mon-filtered[\s\S]*js-mon-admitted/);
        assert.match(diagnostics, /class="diag-grid mon-stats-tiles"/);
        assert.match(mqttPage, /CSI pipeline, Wi-Fi, and runtime health/);
        assert.match(app, /function syncDiagnosticsPolling/);
        assert.match(app, /setInterval\(monitorRequestStats, interval\)/);
        assert.match(app, /function stopDiagnosticsPolling/);
        assert.match(app, /data\.csi_admitted_pps/);
        assert.match(app, /data\.csi_filtered_pps/);
        assert.match(app, /function monitorIsMqttLive/);
        assert.match(app, /connect\.disabled = mqttLive \|\| monitor\.discoveryActive/);
        assert.match(app, /monitor\.protocol\.subscriptionTopic/);
        assert.match(app, /MONITOR_DISCOVERY_TIMEOUT_MS = 2000/);
        assert.match(mqttProtocol, /static parseDiscoveryMessage/);
        assert.match(app, /function recordDiscoveredMqttDevice/);
        assert.match(app, /function monitorStartDiscovery/);
        assert.match(app, /client\.subscribe\(monitor\.discoveryTopics/);
        assert.match(app, /function monitorDeviceChipLabel/);
        assert.match(app, /function monitorDeviceStatus/);
        assert.match(app, /dotClass: 'dot-ok'/);
        assert.match(app, /dotClass: 'dot-error'/);
        assert.match(app, /dotClass: 'dot-idle'/);
        assert.match(app, /dot\.className = `dot \$\{status\.dotClass\}`/);
        assert.doesNotMatch(app, /const frontend = device\.frontend \|\| 'unknown'/);
        assert.match(styles, /\.device-choice-option \{/);
        assert.match(app, /function monitorSelectDevice/);
        assert.match(mqttProtocol, /`\$\{prefix\}\/\+\/info`/);
        assert.match(mqttProtocol, /`\$\{prefix\}\/\+\/status`/);
        assert.match(app, /Selected device: /);
        assert.match(app, /No devices discovered\. Enter a device ID\./);
        assert.match(app, /\[deviceInput, deviceValid, 'Enter a device ID without \/ or wildcards\.'\]/);
        assert.doesNotMatch(app, /\[deviceInput, !!device, 'Enter a device ID\.'\]/);
        assert.match(app, /function ingestMqttMessage/);
        assert.match(mqttProtocol, /function mqttUtf8/);
        assert.match(app, /function applyMqttLiveTelemetry/);
        assert.match(app, /MONITOR_CHART_WINDOW_MS = 60 \* 1000/);
        assert.match(app, /function monitorHasFreshTelemetry/);
        assert.match(app, /function monitorResetChart/);
        assert.match(app, /function resetMonitorLiveView/);
        assert.match(app, /function adoptDeviceId/);
        assert.match(app, /monitorStopAll\('device_changed'\)/);
        assert.match(app, /if \(ctx\) ctx\.clearRect\(0, 0, canvas\.width, canvas\.height\)/);
        assert.match(app, /if \(monitor\.boundDeviceId && conn\.deviceId && monitor\.boundDeviceId !== conn\.deviceId\) return;/);
        assert.match(app, /case 'ha\/movement\/state':[\s\S]*?if \(monitorHasFreshTelemetry\(\)\) return;/);
        assert.match(app, /case 'ha\/motion\/state':[\s\S]*?if \(monitorHasFreshTelemetry\(\)\) return;/);
        assert.match(mqttPage, /last 60 seconds/);
        assert.match(app, /case 'commands\/catalog'/);
        assert.match(app, /monitor\.commandCatalogReady = true/);
        assert.match(app, /function monitorStartBle/);
        assert.match(app, /command: 'set_ble', ble: 'on'/);
        assert.match(app, /command: 'set_ble', ble: 'off'/);
    });

    it('prefills Home Assistant MQTT defaults and splits the monitor device id', () => {
        assert.match(index, /id="cfg-mqtt-host"[^>]*value="homeassistant.local"/);
        assert.match(index, /id="cfg-mqtt-port"[^>]*value="1883"/);
        assert.match(index, /id="cfg-mqtt-user"[^>]*value="mqtt"/);
        assert.match(index, /id="cfg-mqtt-pass"[^>]*value="mqtt"/);
        assert.match(index, /id="cfg-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(index, /id="mon-host"[^>]*value="homeassistant.local"/);
        assert.match(index, /id="mon-port"[^>]*value="9001"/);
        assert.match(index, /id="mon-user"[^>]*value="mqtt"/);
        assert.match(index, /id="mon-pass"[^>]*value="mqtt"/);
        assert.match(index, /id="mon-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(index, /id="mon-device"[^>]*placeholder="3cf79180d3a0aca4"/);
        assert.match(index, /for="mon-device">Device ID <span class="opt">\(optional\)<\/span>/);
        assert.match(index, /id="mon-device-choice" class="device-choice-list"/);
        assert.match(index, /class="field js-mon-device-picker"/);
        assert.match(index, /Leave Device ID empty to discover devices on the broker/);
        assert.match(index, /id="mon-path"[^>]*value="\/mqtt"/);
        assert.match(index, /id="mon-tls"/);
        const mqttPage = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const onboardingHtml = mqttPage.match(/class="js-monitor-onboarding"[\s\S]*?<div class="js-monitor-workspace"/)?.[0] || '';
        const broker = mqttPage.match(/<section class="device-connect-card[^"]*" data-transport="mqtt">[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(onboardingHtml, /<div class="fields">/);
        assert.match(broker, /id="mon-host"[\s\S]*id="mon-user"[\s\S]*id="mon-port"[\s\S]*id="mon-topic-prefix"[\s\S]*id="mon-device"[\s\S]*js-mon-connect/);
        assert.match(broker, /id="mon-path"/);
        assert.match(broker, /id="mon-tls"/);
        assert.match(mqttProtocol, /static baseTopic/);
        assert.match(app, /function applyConfigureMqttDefaults/);
        assert.match(app, /host: 'homeassistant.local'/);
        assert.match(app, /topicPrefix: 'espectre\/v1\/devices'/);
        assert.match(app, /function applyConfigureMqttToMonitor/);
        assert.match(app, /function startDetection/);
        assert.match(app, /bindMqttToConnection/);
        assert.match(app, /stopBleForDetection/);
        assert.match(app, /await ensureBleOffForLive/);
        assert.match(app, /if \(monitor\.bleRequested\) ensureBleOffForLive/);
        assert.match(index, /connect over Bluetooth to set Wi-Fi, MQTT, and the device name/);
        assert.match(index, /connect over MQTT to watch live motion, tune detection, and inspect diagnostics/);
        assert.match(index, /js-start-detection/);
        assert.match(index, /js-connect-ble/);
        assert.match(
            index,
            /class="conn-pill conn-disconnected js-header-connect"[\s\S]*?<span class="js-connect-label">Connect device<\/span>/
        );
        assert.match(
            app,
            /\$\('\.js-header-connect'\)\.addEventListener\('click', \(\) => \{[\s\S]*?location\.hash = '#configure';/
        );
        assert.match(index, /js-has-live/);
        assert.doesNotMatch(app, /getElementById\('mon-port'\)\.value = .*(?:cfg-mqtt-port|1883)/);
        assert.match(app, /input_mode: connectionInputMode\(\)/);
    });

    it('keeps Game and Theremin demo sessions on the current tool', () => {
        assert.match(index, /data-page="theremin"[\s\S]*?class="link-btn js-demo">or try the demo/);
        assert.match(index, /data-page="game"[\s\S]*?class="link-btn js-demo">or try the demo/);
        assert.match(
            app,
            /if \(route !== 'game' && route !== 'theremin'\) setDeviceView\('live'\)/
        );
    });

    it('calibrates Game and Theremin to the detector evaluation cadence', () => {
        assert.match(app, /const EVALUATION_INTERVAL_MS_DEFAULT = 250/);
        assert.match(app, /function applySensingCadence/);
        assert.match(app, /function evaluationIntervalMs/);
        assert.match(app, /snapshot\.evaluation_interval_ms/);
        assert.match(app, /snapshot\.publish_interval_ms/);
        assert.match(app, /snapshot\.csi_target_pps/);
        assert.match(app, /demoTimer = setInterval\(\(\) => \{[\s\S]*?\}, evaluationIntervalMs\(\)\)/);
        assert.match(app, /function gameSensingActive/);
        assert.match(app, /return conn\.movement >= gameThreshold\(\)/);
        assert.doesNotMatch(app, /game\.phase === 'hold' && conn\.motion/);
        assert.doesNotMatch(app, /game\.phase === 'strike' && conn\.motion/);
        assert.match(app, /const tau = evaluationIntervalMs\(\) \/ 2000/);
        assert.match(app, /function monitorChartMaxPoints/);
        assert.match(app, /function monitorTelemetryStaleMs/);
        assert.match(app, /setInterval\(monitorRequestStats, interval\)/);
        assert.match(styles, /transition: width \.25s linear/);
    });
});
