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
const security = read('docs/web/content/security.html');
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
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const sdkPageBuilder = read('.github/scripts/stage_web_sdk.py');
        const externalScripts = [...index.matchAll(/<script[^>]+src="(https?:[^\"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(externalScripts, []);
        assert.doesNotMatch(index, /unpkg\.com|jsdelivr\.net/);
        for (const source of [index, staticPageBuilder, sdkPageBuilder]) {
            assert.doesNotMatch(source, /fonts\.googleapis\.com|fonts\.gstatic\.com/);
        }
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
                '/assets/js/espectre-direct.js',
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
        assert.match(app, /sitePolicy\.isLoopbackHostname\(location\.hostname\)/);
        assert.doesNotMatch(app, /new Set\(\['localhost', '127\.0\.0\.1', '\[::1\]'\]\)/);
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
        assert.match(app, /CONFIG_VERIFICATION_RETRY_MS = 1500/);
        assert.match(app, /CONFIG_VERIFICATION_MAX_ATTEMPTS = 4/);
        assert.match(app, /MONITOR_CONNECTION_TIMEOUT_MS = 10000/);
        assert.match(app, /error_type: 'ConnectionTimeout'/);
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
        const gameAbandonEvent = app.match(/track\('game_abandon', \{[\s\S]*?\n        \}\);/)?.[0] || '';
        const gameOverEvent = app.match(/track\('game_over', \{[\s\S]*?\n        \}\);/)?.[0] || '';
        assert.match(gameAbandonEvent, /score: game\.score/);
        assert.match(gameAbandonEvent, /distance: Math\.floor\(game\.distance\)/);
        assert.match(gameAbandonEvent, /reason/);
        assert.match(gameOverEvent, /score: game\.score/);
        assert.match(gameOverEvent, /orbs: game\.orbs/);
        assert.match(gameOverEvent, /distance: Math\.floor\(game\.distance\)/);
        assert.doesNotMatch(gameOverEvent, /rounds|best_time/);
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

    it('uses canonical paths for static pages and hashes for SPA-only navigation', () => {
        const staticRouteNames = [...routeRegistry.matchAll(/\{ name: '([^']+)'[^\n]+staticPath:/g)]
            .map((match) => match[1]);
        for (const routeName of staticRouteNames) {
            assert.doesNotMatch(index, new RegExp(`href="(?:/)?#${routeName}"`));
        }
        assert.match(index, /href="\/guides\/" class="nav-link" data-route-link="guides"/);
        assert.match(index, /href="\/sdk\/" class="nav-link" data-route-link="sdk"/);
        assert.match(index, /href="#tools" class="nav-link" data-route-link="tools"/);
        for (const source of [
            index,
            read('docs/web/404.html'),
            read('.github/scripts/build_static_pages.py'),
            read('.github/scripts/stage_web_sdk.py'),
        ]) {
            assert.ok(source.indexOf('href="/sdk/" class="nav-link') < source.indexOf('href="/roadmap/" class="nav-link'));
        }
        assert.match(app, /const staticRoute = routeRegistry\.routeForPath\(href\);[\s\S]*?location\.hash = '#' \+ staticRoute;/);
    });

    it('normalizes explicit HTML entries and static-page clicks to root SPA hashes', () => {
        const navigation = read('docs/web/assets/js/navigation.js');
        const listeners = new Map();
        const replacements = [];
        const assignments = [];
        const location = {
            pathname: '/index.html',
            href: 'https://espectre.dev/index.html#contact',
            search: '',
            hash: '#contact',
            assign: (href) => assignments.push(href),
        };
        const history = {
            state: { source: 'test' },
            replaceState: (state, title, href) => replacements.push(String(href)),
        };
        const document = {
            documentElement: { hasAttribute: (name) => name === 'data-static-page' },
            addEventListener: (type, listener) => listeners.set(type, listener),
            querySelectorAll: () => [],
        };
        const window = {
            location,
            history,
            matchMedia: () => ({ matches: false, addEventListener: () => {} }),
            ESPectreRoutes: { routeForPath: (href) => href === '/contact/' ? 'contact' : '' },
        };
        runInNewContext(navigation, { document, URL, window });

        assert.deepEqual(replacements, ['https://espectre.dev/#contact']);
        let prevented = false;
        listeners.get('click')({
            defaultPrevented: false,
            button: 0,
            metaKey: false,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            target: {
                closest: () => ({
                    target: '',
                    getAttribute: () => '/contact/',
                }),
            },
            preventDefault: () => { prevented = true; },
        });
        assert.equal(prevented, true);
        assert.deepEqual(assignments, ['/#contact']);
    });

    it('has a responsive navigation control and a live status region', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.conn \{ margin-left: auto; min-width: 0; order: 2; \}/);
        assert.match(styles, /\.conn-connected \.js-device-name \{ min-width: 0; overflow: hidden; text-overflow: ellipsis; \}/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
        assert.match(index, /class="toast toast-sticky js-demo-toast"[^>]+role="status"/);
        assert.match(app, /function syncDemoToast/);
        assert.match(app, /el\.hidden = !\(conn\.mode === 'demo' && conn\.status === 'connected'\)/);
        assert.match(styles, /\.toast\.js-toast:not\(\[hidden\]\) ~ \.toast-sticky:not\(\[hidden\]\)/);
    });

    it('provides skip navigation, tool page titles, and route focus management', () => {
        assert.match(index, /<a class="skip-link" href="#main-content"/);
        assert.match(index, /data-page="home" id="main-content" tabindex="-1"/);
        for (const page of ['configure', 'monitor', 'theremin', 'game']) {
            assert.match(index, new RegExp(`data-page="${page}"[\\s\\S]*?<h1 class="page-title">`));
        }
        assert.match(app, /link\.setAttribute\('aria-current', 'page'\)/);
        assert.match(app, /target\.focus\(\{ preventScroll: true \}\)/);
        assert.match(app, /page\.id = 'main-content'/);
        assert.match(app, /const routeAtStart = route;/);
        assert.match(app, /if \(route === routeAtStart\) focusRouteContent\(routeAtStart\);/);
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

    it('isolates modal focus from the page behind the dialog', () => {
        assert.match(app, /const openModal = \$\$\('\.modal-backdrop'\)\.find\(\(modal\) => !modal\.hidden\);/);
        assert.match(app, /const shouldBeInert = Boolean\(openModal\) && child !== openModal;/);
        assert.match(app, /child\.inert = true;/);
        assert.match(app, /child\.inert = false;/);
    });

    it('associates every form label with a control', () => {
        const labels = [...index.matchAll(/<label\b([^>]*)>/g)];
        assert.ok(labels.length > 10);
        for (const [, attributes] of labels) assert.match(attributes, /\bfor="[^"]+"/);
    });
});

describe('website UX contracts', () => {
    it('formats security guidance with project-native components', () => {
        assert.match(security, /class="security-guidelines">\s*<ul>/);
        assert.match(styles, /\.security-guidelines ul \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
        assert.match(styles, /\.security-page \.note \{[^}]*background: var\(--accent-soft\);[^}]*border: 1px solid var\(--accent-line\);/);
        assert.match(styles, /\.security-page \.docs-start-copy p \+ p \{ margin-top: 12px; \}/);
        assert.match(styles, /\.security-page \.docs-path > :is\(\.btn-primary, \.btn-secondary\) \{[^}]*margin-top: auto;/);
        assert.match(security, /class="docs-paths security-reporting-paths">/);
        assert.doesNotMatch(security, /SECURITY\.md/);
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
        assert.match(index, /class="js-scrolly-current"/);
        assert.match(index, /data-scene="1" aria-hidden="true" inert/);
        assert.match(app, /el\.toggleAttribute\('inert', !isActive\)/);
        assert.match(app, /el\.setAttribute\('aria-hidden', String\(!isActive\)\)/);
        assert.match(app, /event\.key !== 'ArrowDown' && event\.key !== 'ArrowUp'/);
        assert.match(app, /target\.closest\('a, button, input, select, textarea, \[contenteditable="true"\]'\)/);
        assert.match(app, /document\.addEventListener\('keydown', scrollyHandleKeydown\)/);
        assert.match(app, /sceneProgress = \(nextScene \+ 0\.5\) \/ sceneCount/);
        assert.doesNotMatch(index, /class="hero-skip"/);
        assert.match(index, /href="#get-started" class="scrolly-skip"/);
        assert.match(index, /<section class="home-action-hub" id="get-started" aria-labelledby="home-action-title">/);
        const actionHub = index.match(/<section class="home-action-hub"[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(actionHub, /<header class="home-action-head">\s*<h2 class="page-title" id="home-action-title">/);
        assert.match(actionHub, /class="home-action-group">\s*<span class="home-kicker">[\s\S]*?<div class="home-tool-grid"/);
        assert.match(actionHub, /href="#flash" class="home-tool-card home-tool-card-primary"/);
        assert.match(actionHub, /href="#configure" class="home-tool-card"/);
        assert.match(actionHub, /href="#monitor" class="home-tool-card"/);
        assert.match(actionHub, /class="home-action-group">[\s\S]*?<aside class="home-license-cta"/);
        assert.match(actionHub, /class="home-action-group">[\s\S]*?<div class="home-resource-strip"/);
        const licenseCard = actionHub.match(/<aside class="home-license-cta"[\s\S]*?<\/aside>/)?.[0] || '';
        assert.doesNotMatch(licenseCard, /home-kicker/);
        assert.match(actionHub, /class="home-license-cta"[\s\S]*?class="home-resource-strip"/);
        const resourceHrefs = [...actionHub.matchAll(/class="home-resource-links">[\s\S]*?<\/div>/g)][0]?.[0]
            .match(/href="([^"]+)"/g)?.map((entry) => entry.slice(6, -1)) || [];
        assert.deepEqual(resourceHrefs, [
            '#tools',
            '/guides/',
            '/media/',
            '/sdk/',
            '/roadmap/',
            'https://github.com/francescopace/espectre',
        ]);
        assert.match(styles, /\.home-resource-links \{ display: grid; grid-template-columns: repeat\(6, minmax\(0, 1fr\)\); \}/);
        assert.doesNotMatch(actionHub, /home-resource-intro/);
        assert.match(styles, /\.home-license-cta \{[\s\S]*?background: var\(--surface\);/);
        assert.match(actionHub, /href="\/licensing\/" class="btn-primary"/);
        assert.doesNotMatch(actionHub, /js-start-detection|js-demo/);
        assert.doesNotMatch(index, /home-(?:after-story|commercial|path|quick-links)/);
        assert.match(styles, /\.home-action-hub \{[^}]*min-height: 100svh/);
        assert.match(styles, /\.home-action-inner \{[^}]*min-height: 100svh;[^}]*padding: calc\(var\(--header-height\) \+ 64px\) 40px 40px;[^}]*justify-content: center;/);
        assert.doesNotMatch(index, /home-privacy-grid/);
    });

    it('labels research and preview concepts without presenting simulated evidence', () => {
        assert.doesNotMatch(index, /13\.2 <em>cycles\/min<\/em>/);
        assert.doesNotMatch(index, /ESP-IDF 5\.1\+/);
        const roadmapCards = [...index.matchAll(/<a href="\/roadmap\/"[\s\S]*?<\/a>/g)]
            .map((match) => match[0])
            .filter((card) => card.includes('<h2>'));
        assert.ok(roadmapCards.length >= 3);
        for (const card of roadmapCards) {
            assert.match(card, /class="chip">ROADMAP/);
            assert.doesNotMatch(card, /js-direct-chip/);
        }
    });

    it('enforces the supported browser capability matrix', async () => {
        const detect = (navigator) => {
            const context = { window: { navigator } };
            runInNewContext(browserSupportSource, context);
            return context.window.ESPectreBrowserSupport.current;
        };
        const serial = { requestPort() {} };
        const desktop = detect({ userAgent: 'Chrome', platform: 'Linux x86_64', serial });
        assert.equal(desktop.flash, true);
        const android = detect({ userAgent: 'Chrome Android Mobile', platform: 'Linux armv8', serial });
        assert.equal(android.flash, false);
        const ios = detect({ userAgent: 'CriOS iPhone Mobile', platform: 'iPhone', serial });
        assert.equal(ios.flash, false);
        const chrome147 = detect({
            userAgent: 'Mozilla/5.0 Chrome/147.0.0.0 Safari/537.36', platform: 'Linux x86_64'
        });
        assert.equal(chrome147.hostedDirect, 'targeted');
        assert.equal(chrome147.browser, 'chrome');
        const firefox = detect({ userAgent: 'Mozilla/5.0 Firefox/148.0', platform: 'Linux x86_64' });
        assert.equal(firefox.hostedDirect, 'unsupported');
        const safari = detect({
            userAgent: 'Mozilla/5.0 Version/26.0 Safari/605.1.15', platform: 'MacIntel'
        });
        assert.equal(safari.hostedDirect, 'unsupported');
        const edge = detect({
            userAgent: 'Mozilla/5.0 Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0',
            platform: 'Win32'
        });
        assert.equal(edge.hostedDirect, 'unclaimed');
        const permissionContext = {
            window: {
                navigator: {
                    permissions: { query: async ({ name }) => ({ state: name === 'local-network-access' ? 'denied' : 'prompt' }) }
                }
            }
        };
        runInNewContext(browserSupportSource, permissionContext);
        assert.equal(
            await permissionContext.window.ESPectreBrowserSupport.localNetworkAccessState(
                permissionContext.window.navigator
            ),
            'denied'
        );
        assert.equal(
            await permissionContext.window.ESPectreBrowserSupport.localNetworkAccessState({}),
            'unavailable'
        );
        assert.match(app, /installTrigger\.disabled = !browserSupport\.flash/);
        assert.match(app, /button\.disabled = directConnecting/);
        assert.match(app, /if \(browserSupport\.flash\) \{\s*loadBrowserDependency/);
        assert.match(index, /class="link-btn js-demo"/);
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
        assert.match(index, /<div class="footer-links">\s*<a href="\/privacy\/"/);
        assert.match(routeRegistry, /name: 'privacy'.*staticPath: '\/privacy\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/privacy\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/privacy\/"/);
        const sitemap = read('.github/scripts/sitemap.template.xml');
        assert.match(sitemap, /https:\/\/espectre\.dev\/privacy\//);
        assert.doesNotMatch(sitemap, /<(?:changefreq|lastmod)>/);
        assert.match(read('docs/web/content/privacy.html'), /id="cookie-settings"/);
        const notFound = read('docs/web/404.html');
        assert.doesNotMatch(notFound, /http-equiv="refresh"|location\.replace/);
        assert.match(notFound, /<footer class="site-footer">/);
        assert.match(styles, /body \{[\s\S]*?display: flex;[\s\S]*?flex-direction: column;[\s\S]*?min-height: 100dvh;/);
        assert.match(styles, /body > main \{[\s\S]*?width: 100%;[\s\S]*?box-sizing: border-box;[\s\S]*?flex: 1 0 auto;/);
        const sharedFooterBrand = /<div class="footer-brand">\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg(?:\?v=(?:[0-9a-f]{12}|\{logo_version\}))?" alt="" width="23" height="23" aria-hidden="true">/;
        assert.match(index, sharedFooterBrand);
        assert.match(notFound, sharedFooterBrand);
        assert.match(read('.github/scripts/build_static_pages.py'), sharedFooterBrand);
        assert.match(read('.github/scripts/stage_web_sdk.py'), sharedFooterBrand);
        assert.match(styles, /\.footer-brand \{[^}]*color: var\(--text\);/);
        assert.doesNotMatch(read('.github/scripts/build_static_pages.py'), /footer-brand[\s\S]*?GPLv3 \+ commercial licensing/);
        assert.match(notFound, /data-static-page data-site-section="other"/);
        assert.match(notFound, /<a href="\/privacy\/#cookie-settings" class="js-cookie-settings"/);
        assert.doesNotMatch(notFound, /footer-link-button/);
        assert.match(styles, /\.footer-links a,\s*\.footer-links a:visited \{[\s\S]*?display: inline-flex;[\s\S]*?color: var\(--text\);[\s\S]*?text-decoration: none;/);
        assert.match(styles, /\.footer-links a:hover,\s*\.footer-links a:focus-visible \{ color: var\(--accent\); text-decoration: none; \}/);
        assert.match(read('docs/web/content/privacy.html'), /<h2 id="cookie-settings">/);
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
        assert.match(index, /<a href="\/licensing\/"/);
        assert.match(index, /<a href="\/contact\/"/);
        assert.match(routeRegistry, /name: 'licensing'.*staticPath: '\/licensing\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/licensing\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/licensing\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/licensing\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/licensing\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/licensing\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/licensing\/": \(Path\("docs\/web\/content\/licensing\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(licensingContent, /<h1 class="page-title">/);
        assert.match(licensingContent, /mailto:contact@espectre\.dev\?subject=Commercial%20licensing%20inquiry/);
    });

    it('publishes a dedicated contact page from every footer', () => {
        const contactContent = read('docs/web/content/contact.html');
        assert.match(index, /data-page="contact"/);
        assert.match(index, /data-content-url="content\/contact\.html\?v=[0-9a-f]{12}"/);
        assert.match(routeRegistry, /name: 'contact'.*staticPath: '\/contact\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/contact\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/contact\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/contact\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/contact\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/contact\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/contact\/": \(Path\("docs\/web\/content\/contact\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(contactContent, /<h1 class="page-title">/);
        assert.match(contactContent, /mailto:contact@espectre\.dev/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/discussions/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/issues/);
        assert.doesNotMatch(contactContent, /mailto:security@espectre\.dev/);
    });

    it('publishes a dedicated security and responsible-use page', () => {
        const securityContent = read('docs/web/content/security.html');
        assert.match(index, /data-page="security"/);
        assert.match(index, /data-content-url="content\/security\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<a href="\/security\/"/);
        assert.match(routeRegistry, /name: 'security'.*staticPath: '\/security\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/security\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/security\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/security\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/security\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/security\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/security\/": \(Path\("docs\/web\/content\/security\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(securityContent, /<h1 class="page-title">/);
        assert.match(securityContent, /mailto:contact@espectre\.dev\?subject=Responsible%20use%20or%20abuse%20report/);
        assert.match(securityContent, /mailto:security@espectre\.dev/);
        assert.match(securityContent, /github\.com\/francescopace\/espectre\/security/);
        assert.doesNotMatch(securityContent, /github\.com\/francescopace\/espectre\/blob\/main\/SECURITY\.md/);
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
            assert.match(source, /<a href="\/terms\/"/);
            assert.match(source, /<a href="\/legal\/"/);
        }
        assert.match(staticPageBuilder, /"source": "content\/terms\.html"/);
        assert.match(staticPageBuilder, /"source": "content\/legal\.html"/);
        assert.match(sitemap, /https:\/\/espectre\.dev\/terms\//);
        assert.match(sitemap, /https:\/\/espectre\.dev\/legal\//);
        assert.match(sitemapBuilder, /"\/terms\/": \(Path\("docs\/web\/content\/terms\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(sitemapBuilder, /"\/legal\/": \(Path\("docs\/web\/content\/legal\.html"\), STATIC_PAGE_BUILDER\)/);

        assert.match(termsContent, /<h1 class="page-title">/);
        assert.match(legalContent, /<h1 class="page-title">/);
        assert.match(legalContent, /<dt>Name<\/dt><dd>Francesco Pace<\/dd>/);
        assert.match(legalContent, /<dt>Legal form<\/dt><dd>Natural person<\/dd>/);
        assert.match(legalContent, /<dt>Primary contact<\/dt><dd><a href="mailto:contact@espectre\.dev">contact@espectre\.dev<\/a><\/dd>/);
        assert.doesNotMatch(legalContent, /available through official resellers/);
        assert.doesNotMatch(legalContent, /francesco\.pace@espectre\.dev|security@espectre\.dev|href="\/security\/"/);
    });

    it('treats top-level SDK, roadmap, privacy, terms, legal, security, licensing, and contact as pages, not articles', () => {
        const sdkContent = read('docs/web/content/sdk.html');
        const roadmapContent = read('docs/web/content/roadmap.html');
        const privacyContent = read('docs/web/content/privacy.html');
        const termsContent = read('docs/web/content/terms.html');
        const legalContent = read('docs/web/content/legal.html');
        const securityContent = read('docs/web/content/security.html');
        const licensingContent = read('docs/web/content/licensing.html');
        const contactContent = read('docs/web/content/contact.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.ok(sdkContent.startsWith(`${GPL_HTML_HEADER}<div class="docs-quickstart">`));
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
        assert.doesNotMatch(index, /<main class="js-page page-narrow page-article" data-page="(?:sdk|roadmap|privacy|terms|legal|security|licensing|contact)"/);
        assert.match(staticPageBuilder, /"source": "content\/sdk\.html",[\s\S]*?"og_type": "website"/);
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
        for (const path of ['guides', 'sdk', 'roadmap', 'privacy', 'terms', 'legal', 'security', 'licensing', 'contact']) {
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

    it('uses one shared measure for every inner page', () => {
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.match(styles, /\.page-narrow \{[\s\S]*?max-width: 1120px;/);
        assert.match(styles, /\.article \{ width: 100%; \}/);
        assert.doesNotMatch(styles, /\.article \{[^}]*max-width:/);
        const innerPages = [...index.matchAll(/<main class="([^"]*)" data-page="([^"]+)"/g)]
            .filter(([, , page]) => page !== 'home');
        for (const [, classes] of innerPages) {
            assert.match(classes, /(?:^|\s)page-narrow(?:\s|$)/);
        }
        assert.match(staticPageBuilder, /spec\.get\("main_class", "page-narrow page-article"\)/);
    });

    it('gives the SDK landing page a clear start-to-reference hierarchy', () => {
        const sdkContent = read('docs/web/content/sdk.html');
        assert.match(sdkContent, /<h1 class="page-title">/);
        assert.match(sdkContent, /<section class="docs-start" aria-labelledby="docs-start-title">/);
        assert.match(sdkContent, /<section class="docs-section" aria-labelledby="docs-paths-title">/);
        assert.match(sdkContent, /<section class="docs-section" aria-labelledby="docs-quick-start-title">/);
        assert.match(sdkContent, /<section class="docs-next" aria-labelledby="docs-next-title">/);
        assert.match(sdkContent, /class="docs-cover"[\s\S]*?sdk-firmware-pipeline\.avif/);
        assert.ok(sdkContent.indexOf('class="docs-start"') < sdkContent.indexOf('class="docs-paths"'));
        assert.ok(sdkContent.indexOf('class="docs-paths"') < sdkContent.indexOf('class="docs-steps"'));
        assert.ok(sdkContent.indexOf('class="docs-steps"') < sdkContent.indexOf('class="docs-next"'));
        const sdkIndexLinks = [...sdkContent.matchAll(/<a href="(\/sdk\/(?:api|examples|architecture|detectors)\/)" class="doc-link">/g)].map((match) => match[1]);
        assert.deepEqual(sdkIndexLinks, ['/sdk/architecture/', '/sdk/api/', '/sdk/detectors/', '/sdk/examples/']);
        const pathCards = sdkContent.match(/<div class="docs-path(?: docs-path-recommended)?">[\s\S]*?<\/div>/g) || [];
        assert.equal(pathCards.length, 3);
        for (const card of pathCards) {
            assert.match(card, /<h3>/);
            assert.doesNotMatch(card, /<h2>/);
        }
    });

    it('documents both public SDK facades and their compatibility boundary', () => {
        const apiContent = read('docs/web/content/sdk/api.html');
        assert.match(apiContent, /href="\/artifacts\/sdk\/api\/espectre__sdk_8h\.html"/);
        assert.match(apiContent, /href="\/artifacts\/sdk\/api\/espectre__core__sdk_8h\.html"/);
        assert.doesNotMatch(apiContent, /Supported means reachable from <code>espectre_sdk\.h<\/code>/);
    });

    it('links the SDK pages in one previous and next sequence', () => {
        const sdkPages = [
            { file: 'architecture', previous: null, next: '/sdk/api/' },
            { file: 'api', previous: '/sdk/architecture/', next: '/sdk/detectors/' },
            { file: 'detectors', previous: '/sdk/api/', next: '/sdk/examples/' },
            { file: 'examples', previous: '/sdk/detectors/', next: 'https://github.com/francescopace/espectre/blob/main/docs/EMBEDDING.md' },
        ];
        for (const page of sdkPages) {
            const content = read(`docs/web/content/sdk/${page.file}.html`);
            const articleNav = content.slice(content.lastIndexOf('<div class="article-nav">'));
            assert.match(articleNav, /^<div class="article-nav">/);
            const links = [...articleNav.matchAll(/<a href="([^"]+)" class="doc-link(?: doc-link-next)?"[^>]*>/g)].map((match) => match[1]);
            assert.deepEqual(links, [page.previous, page.next].filter(Boolean), `${page.file} follows the SDK page order`);
            if (page.next) {
                assert.match(articleNav, new RegExp(`<a href="${page.next}" class="doc-link doc-link-next"`), `${page.file} identifies its next page`);
            }
        }
        assert.doesNotMatch(read('docs/web/content/sdk/architecture.html'), /docs\/EMBEDDING\.md/);
        assert.match(read('docs/web/content/sdk/examples.html'), /docs\/EMBEDDING\.md/);
    });

    it('publishes the detector architecture through SDK SPA and static routes', () => {
        const detectors = read('docs/web/content/sdk/detectors.html');
        assert.match(detectors, /<h1>/);
        assert.match(detectors, /<h2 id="sdk-detectors-performance">/);
        assert.doesNotMatch(detectors, /<h3>Current Native device benchmark<\/h3>/);
        assert.doesNotMatch(detectors, /docs\/performance\/ESP32(?:-C[356]|-S3)?\.md/);
        assert.match(detectors, /DetectionAlgorithm::HIGH_ACCURACY/);
        const detectorGuide = read('docs/web/content/guides/detectors.html');
        assert.match(detectorGuide, /href="\/sdk\/detectors\/"/);
        assert.doesNotMatch(detectorGuide, /performance report/);
        assert.match(index, /data-page="sdk-detectors"[\s\S]*?data-static-url="\/sdk\/detectors\/"/);
        assert.match(routeRegistry, /name: 'sdk-detectors'.*staticPath: '\/sdk\/detectors\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/sdk\/detectors\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/sdk\/detectors\//);
    });

    it('publishes the detection profile guide through SPA and static routes', () => {
        const guide = read('docs/web/content/guides/detectors.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /<h2 id="detectors-two">/);
        assert.equal((guide.match(/class="profile-comparison-row"/g) || []).length, 5);
        assert.doesNotMatch(guide, /<div class="table-wrap">/);
        assert.match(guide, /role="tablist" aria-label="Detection profile interface"/);
        assert.equal((guide.match(/data-detector-interface=/g) || []).length, 4);
        assert.match(guide, /id="detectors-native-tab"[^>]*aria-selected="true"/);
        assert.match(guide, /id="detectors-cli"/);
        assert.doesNotMatch(guide, /<h2 id="detectors-cli">/);
        assert.match(styles, /\.profile-comparison-row \{[\s\S]*?grid-template-columns: minmax\(130px, \.55fr\) repeat\(2, minmax\(0, 1fr\)\);/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.profile-comparison-row \{ grid-template-columns: minmax\(0, 1fr\); gap: 12px; \}/);
        assert.match(index, /data-page="guide-detectors"/);
        assert.match(routeRegistry, /name: 'guide-detectors'.*staticPath: '\/guides\/detectors\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/detectors\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/detectors\//);
    });

    it('publishes the MicroPython contribution and runtime guide', () => {
        const guide = read('docs/web/content/guides/micropython.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /micropython\/micropython\/pull\/18460/);
        assert.match(read('docs/web/content/guides.html'), /micropython-csi-runtime-card\.avif/);
        assert.match(guide, /micropython-csi-runtime-card\.avif/);
        assert.match(index, /data-page="guide-micropython"/);
        assert.match(routeRegistry, /name: 'guide-micropython'.*staticPath: '\/guides\/micropython\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/micropython\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/micropython\//);
    });

    it('publishes the IEEE 802.11bf future sensing guide', () => {
        const guide = read('docs/web/content/guides/future-wifi-sensing.html');
        const guideIndex = read('docs/web/content/guides.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /<h2 id="future-origin">/);
        assert.match(guide, /standards\.ieee\.org\/ieee\/802\.11bf\/11574\//);
        assert.match(guide, /www\.ieee802\.org\/11\/Reports\/tgbf_update\.htm/);
        assert.match(guide, /future-wifi-sensing-card\.avif/);
        assert.match(guideIndex, /href="\/guides\/future-wifi-sensing\/"/);
        assert.doesNotMatch(guideIndex, /href="\/guides\/custom-firmware\/"/);
        assert.match(index, /data-page="guide-future-wifi-sensing"/);
        assert.match(routeRegistry, /name: 'guide-future-wifi-sensing'.*staticPath: '\/guides\/future-wifi-sensing\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/future-wifi-sensing\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/future-wifi-sensing\//);
    });

    it('publishes the Home Assistant dashboard guide with a distinct cover and an in-guide dashboard screenshot', () => {
        const guide = read('docs/web/content/guides/home-assistant.html');
        const guideIndex = read('docs/web/content/guides.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /home-assistant-dashboard\.yaml/);
        assert.match(guide, /home-assistant-dashboard\.png/);
        assert.match(guide, /home-assistant-dashboard-card\.avif/);
        assert.ok(guide.indexOf('home-assistant-dashboard-card.avif') < guide.indexOf('id="ha-before"'));
        assert.ok(guide.indexOf('id="ha-result"') < guide.indexOf('home-assistant-dashboard.png'));
        assert.match(guide, /sensor\.espectre_c3_f61093_movement_score/);
        assert.match(guide, /sensor\.espectre_c3_f61093_movement_score_2/);
        assert.match(guide, /id="ha-recreate-ids"/);
        assert.match(guide, /homeassistant\/#/);
        assert.doesNotMatch(guide, /sensor\.native_&lt;device-id&gt;_movement_score|sensor\.micro_&lt;client-id&gt;_movement_score/);
        assert.match(guideIndex, /href="\/guides\/home-assistant\/"/);
        assert.match(guideIndex, /home-assistant-dashboard-card\.avif/);
        assert.doesNotMatch(guide, /id="ha-cli"/);
        assert.ok(guideIndex.indexOf('href="/guides/detection/"') < guideIndex.indexOf('href="/guides/hardware/"'));
        assert.ok(guideIndex.indexOf('href="/guides/placement/"') < guideIndex.indexOf('href="/guides/home-assistant/"'));
        assert.ok(guideIndex.indexOf('href="/guides/home-assistant/"') < guideIndex.indexOf('href="/guides/detectors/"'));
        assert.match(index, /data-page="guide-home-assistant"/);
        assert.match(routeRegistry, /name: 'guide-home-assistant'.*staticPath: '\/guides\/home-assistant\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/home-assistant\.html"/);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/guides\/home-assistant\/": \(Path\("docs\/web\/content\/guides\/home-assistant\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/home-assistant\//);
    });

    it('uses one cover, practical CLI sections, and a previous/next sequence across the official guides', () => {
        const guides = [
            { file: 'detection', cover: 'csi-multipath-room.avif', cli: null, previous: null, next: '/guides/hardware/' },
            { file: 'hardware', cover: 'esp32-chip-family-card.avif', cli: null, previous: '/guides/detection/', next: '/guides/setup/' },
            { file: 'setup', cover: 'flash-connect-usb-card.avif', cli: 'setup-cli', previous: '/guides/hardware/', next: '/guides/placement/' },
            { file: 'placement', cover: 'sensor-placement-card.avif', cli: null, previous: '/guides/setup/', next: '/guides/home-assistant/' },
            { file: 'home-assistant', cover: 'home-assistant-dashboard-card.avif', cli: null, previous: '/guides/placement/', next: '/guides/detectors/' },
            { file: 'detectors', cover: 'detection-profiles-card.avif', cli: 'detectors-cli', previous: '/guides/home-assistant/', next: '/guides/micropython/' },
            { file: 'micropython', cover: 'micropython-csi-runtime-card.avif', cli: 'micropython-cli', previous: '/guides/detectors/', next: '/guides/future-wifi-sensing/' },
            { file: 'future-wifi-sensing', cover: 'future-wifi-sensing-card.avif', cli: null, previous: '/guides/micropython/', next: null },
        ];
        for (const guide of guides) {
            const path = `docs/web/content/guides/${guide.file}.html`;
            const content = read(path);
            assert.doesNotMatch(content, /class="page-toc"/, `${guide.file} has no in-page shortcut`);
            const firstImage = content.match(/<img\b[^>]*>/)?.[0] || '';
            assert.ok(firstImage.includes(guide.cover), `${guide.file} starts with its guide-card cover`);
            assert.ok(content.indexOf(guide.cover) < content.indexOf('<h2'), `${guide.file} cover precedes its sections`);
            if (guide.cli) {
                assert.match(content, new RegExp(`<h[23][^>]*\\bid="${guide.cli}"`), `${guide.file} documents its CLI equivalent`);
            }
            const images = [...content.matchAll(/<img\b[^>]*>/g)].map((match) => match[0]);
            for (const image of images) {
                assert.match(image, /\bwidth="\d+"/);
                assert.match(image, /\bheight="\d+"/);
            }
            const articleNav = content.slice(content.lastIndexOf('<div class="article-nav">'));
            assert.match(articleNav, /^<div class="article-nav">/);
            const links = [...articleNav.matchAll(/<a href="([^"]+)" class="doc-link(?: doc-link-next)?">/g)].map((match) => match[1]);
            assert.deepEqual(links, [guide.previous, guide.next].filter(Boolean), `${guide.file} follows the official guide order`);
            if (guide.next) {
                assert.match(articleNav, new RegExp(`<a href="${guide.next}" class="doc-link doc-link-next">`), `${guide.file} identifies its next guide`);
            }
        }
        for (const guide of guides) {
            assert.doesNotMatch(read(`docs/web/content/guides/${guide.file}.html`), /<h2[^>]*>Next steps<\/h2>/i);
        }
        assert.match(styles, /\.page-toc \{/);
        assert.match(styles, /\.article-nav \{[\s\S]*?display: flex;[\s\S]*?gap: 16px;/);
        assert.match(styles, /\.article-nav \.doc-link \{ flex: 1 1 0; min-width: 0; \}/);
        assert.match(styles, /\.article-nav \.doc-link-next \{ text-align: right; \}/);
        assert.match(styles, /\.article-nav \.doc-link:only-child \{[\s\S]*?flex: 0 0 calc\(50% - 8px\);/);
        assert.match(styles, /\.article-nav \.doc-link-next:only-child \{ margin-left: auto; \}/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.article-nav \{ flex-direction: column; gap: 12px; \}/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.article-nav \.doc-link:only-child \{ flex-basis: auto; \}/);
        const detectionGuide = read('docs/web/content/guides/detection.html');
        assert.match(detectionGuide, /csi-multipath-room\.avif/);
        assert.match(detectionGuide, /csi-iq-motion\.svg/);
        assert.doesNotMatch(detectionGuide, /csi-quiet-vs-movement|csi-amplitude-heatmap|wifi-frame-csi-path/);
        assert.doesNotMatch(detectionGuide, /id="detection-cli"/);
        assert.ok(detectionGuide.indexOf('csi-multipath-room.avif') < detectionGuide.indexOf('id="detection-room"'));
        assert.ok(detectionGuide.indexOf('id="detection-room"') < detectionGuide.indexOf('id="detection-csi"'));
        assert.ok(detectionGuide.indexOf('id="detection-csi"') < detectionGuide.indexOf('id="detection-motion"'));
        assert.ok(detectionGuide.indexOf('id="detection-motion"') < detectionGuide.indexOf('csi-iq-motion.svg'));
        const hardwareGuide = read('docs/web/content/guides/hardware.html');
        assert.doesNotMatch(hardwareGuide, /id="hardware-cli"|id="hardware-product"/);
        const placementGuide = read('docs/web/content/guides/placement.html');
        assert.doesNotMatch(placementGuide, /id="placement-cli"/);
        assert.doesNotMatch(placementGuide, /id="placement-wall"|id="placement-rooms"/);
    });

    it('loads generated firmware and SDK output from the shared artifacts tree', () => {
        assert.match(app, /\/artifacts\/firmware\//);
        assert.doesNotMatch(app, /\/flash\/firmware\//);
        assert.match(index, /id="flash-channel"/);
        assert.doesNotMatch(index, /id="flash-chip"/);
        assert.match(index, /js-flash-requirement/);
        assert.match(index, /js-flash-chip-downloads/);
        assert.match(index, /js-flash-next/);
        assert.doesNotMatch(index, /js-flash-download/);
        assert.doesNotMatch(index, /js-matter-panel/);
        assert.match(index, /<option value="release"/);
        assert.match(index, /<option value="preview"/);
        assert.match(index, /<option value="develop"/);
        assert.match(app, /updateReleaseBadge[\s\S]*flashLoadManifest\('release'\)/);
        assert.doesNotMatch(app, /flashLoadManifest\('stable'\)/);
        assert.match(app, /builds: artifacts\.map\(\(artifact\) => \(\{/);
        assert.match(app, /chipFamily: artifact\.chip_family/);
        assert.match(app, /const requestId = \+\+flash\.refreshRequest;/);
        assert.match(app, /const selectedChannel = channelSel\.value;/);
        assert.match(app, /const manifest = await flashLoadManifest\(selectedChannel\);/);
        assert.ok((app.match(/if \(requestId !== flash\.refreshRequest\) return;/g) || []).length >= 2);
        assert.match(app, /FLASH_CHIP_UNSUPPORTED_RE/);
        assert.match(app, /report\('unsupported'\)/);
        assert.match(app, /function flashRenderDownloads/);
        assert.match(app, /function flashSetNextStep/);
        assert.match(app, /function flashNextActionLink[\s\S]*document\.createElement\('a'\)/);
        assert.match(app, /flashNextActionLink\([^,]+, 'js-matter-read'\)/);
        assert.doesNotMatch(app, /function flashNextAction\(/);
        assert.match(app, /action\.getAttribute\('aria-disabled'\) === 'true'/);
        assert.match(index, /class="modal-backdrop js-matter-modal" hidden>[\s\S]*class="modal-card matter-modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /id="matter-modal-title"/);
        assert.match(index, /class="matter-result js-matter-result" hidden>/);
        assert.match(app, /function matterOpen\(returnFocus\)/);
        assert.match(app, /matterOpen\(trigger\)/);
        assert.match(app, /if \(!\$\('\.js-matter-modal'\)\.hidden\) matterClose\(\)/);
        assert.match(app, /track\('matter_qr_read'/);
        assert.match(app, /installButton\.toggleAttribute\('inert', !browserSupport\.flash\)/);
        const setupGuide = read('docs/web/content/guides/setup.html');
        assert.match(setupGuide, /flash-connect-usb-card\.avif/);
        assert.match(setupGuide, /href="\/guides\/hardware\/"/);
        assert.doesNotMatch(setupGuide, /flash-connect-usb\.webp/);
        assert.doesNotMatch(setupGuide, /home-assistant-dashboard\.png/);
        assert.match(setupGuide, /href="\/guides\/home-assistant\/"/);
        assert.match(setupGuide, /\.\/espectre matter qr/);
        assert.match(setupGuide, /\.\/espectre mqtt/);
        for (const frontend of ['esphome', 'native', 'matter', 'streamer']) {
            assert.match(setupGuide, new RegExp(`\\.\\/espectre ${frontend} build`));
            assert.match(setupGuide, new RegExp(`\\.\\/espectre ${frontend} flash`));
        }
        assert.match(setupGuide, /\.\/espectre esphome monitor/);
        assert.doesNotMatch(setupGuide, /\.\/espectre esphome config/);
        assert.match(setupGuide, /<code>--dev<\/code>/);
        assert.match(setupGuide, /<code>--config<\/code>/);
        assert.match(setupGuide, /\.\/espectre monitor --reset/);
        assert.match(setupGuide, /\.\/espectre devices --frontend native/);
        assert.match(setupGuide, /\.\/espectre devices --frontend streamer/);
        assert.match(setupGuide, /sdkconfig\.wifi/);
        assert.doesNotMatch(setupGuide, /CLI flashing \(fallback\)/);
        assert.equal((setupGuide.match(/class="code-tabs" data-code-tabs/g) || []).length, 2);
        assert.match(setupGuide, /role="tablist"/);
        const cliTabsSection = setupGuide.slice(setupGuide.indexOf('id="setup-cli"'), setupGuide.indexOf('id="setup-network"'));
        assert.equal((cliTabsSection.match(/role="tab"/g) || []).length, 4);
        assert.equal((cliTabsSection.match(/role="tabpanel"/g) || []).length, 4);
        assert.match(setupGuide, /id="setup-native-tab"[^>]*aria-selected="true"[\s\S]*?id="setup-esphome-tab"[^>]*aria-selected="false"/);
        assert.ok(setupGuide.indexOf('data-frontend="native"') < setupGuide.indexOf('data-frontend="esphome"'));
        for (const chip of ['c5', 'c6', 's3', 'esp32']) {
            assert.match(setupGuide, new RegExp(`<code>${chip}<\\/code>`));
        }
        for (const frontend of ['esphome', 'native', 'matter', 'streamer']) {
            const panel = setupGuide.match(new RegExp(`<div class="code-tab-panel"[^>]*data-frontend="${frontend}"[^>]*>([\\s\\S]*?)<\\/div>`));
            assert.ok(panel, `${frontend} CLI tab exists`);
            assert.match(panel[1], new RegExp(`\\.\\/espectre ${frontend} build --chip c3 --clean`));
            assert.match(panel[1], new RegExp(`\\.\\/espectre ${frontend} flash`));
            assert.match(panel[1], frontend === 'esphome'
                ? /\.\/espectre esphome monitor --chip c3/
                : /\.\/espectre monitor --reset/);
            assert.doesNotMatch(panel[1], /--device|--port/);
        }
        const esphomePanel = setupGuide.match(/<div class="code-tab-panel"[^>]*data-frontend="esphome"[^>]*>([\s\S]*?)<\/div>/)[1];
        assert.match(esphomePanel, /--config path\/to\/espectre\.yaml/);
        assert.match(esphomePanel, /<code>--dev<\/code>/);
        assert.doesNotMatch(esphomePanel, /--device/);
        const streamerPanel = setupGuide.match(/<div class="code-tab-panel"[^>]*data-frontend="streamer"[^>]*>([\s\S]*?)<\/div>/)[1];
        assert.match(streamerPanel, /sdkconfig\.wifi/);
        assert.match(streamerPanel, /\.\/espectre devices --frontend streamer/);
        assert.match(streamerPanel, /\.\/espectre collect<\/code>/);
        assert.ok(streamerPanel.indexOf('./espectre monitor --reset') < streamerPanel.indexOf('./espectre devices --frontend streamer'));
        assert.doesNotMatch(streamerPanel, /--target/);
        const frontendOperations = setupGuide.slice(setupGuide.indexOf('id="setup-network"'));
        assert.doesNotMatch(frontendOperations, /\.\/espectre (?:esphome|native|matter|streamer) build/);
        const networkTabsSection = setupGuide.slice(setupGuide.indexOf('id="setup-network"'), setupGuide.indexOf('id="setup-test"'));
        assert.match(networkTabsSection, /role="tablist"/);
        assert.equal((networkTabsSection.match(/role="tab"/g) || []).length, 4);
        assert.equal((networkTabsSection.match(/role="tabpanel"/g) || []).length, 4);
        assert.match(networkTabsSection, /id="setup-network-native-tab"[^>]*aria-selected="true"[\s\S]*?id="setup-network-esphome-tab"[^>]*aria-selected="false"/);
        for (const frontend of ['native', 'esphome', 'matter', 'streamer']) {
            assert.match(networkTabsSection, new RegExp(`data-network-frontend="${frontend}"`));
        }
        const nativeNetworkPanel = networkTabsSection.match(/<div class="code-tab-panel"[^>]*data-network-frontend="native"[^>]*>([\s\S]*?)<\/div>/)[1];
        assert.match(nativeNetworkPanel, /\.\/espectre mqtt --broker 192\.168\.1\.20/);
        assert.match(nativeNetworkPanel, /espectre&gt; help/);
        assert.match(nativeNetworkPanel, /espectre&gt; set_threshold 0\.35/);
        assert.match(nativeNetworkPanel, /espectre&gt; recalibrate/);
        assert.doesNotMatch(nativeNetworkPanel, /set_ble off/);
        assert.match(networkTabsSection, /data-network-frontend="esphome"[\s\S]*?<code>--dev<\/code>[\s\S]*?<code>secrets\.yaml<\/code>/);
        assert.match(styles, /\.code-tabs \{[\s\S]*?border: 1px solid var\(--border\);[\s\S]*?border-radius: 14px;[\s\S]*?background: var\(--surface\);/);
        assert.match(styles, /\.code-tabs-list \{[\s\S]*?display: flex;[\s\S]*?overflow-x: auto;/);
        assert.match(styles, /\.code-tab-panel \{ padding: 18px; \}/);
        assert.match(styles, /\.code-tab-subsection \{[\s\S]*?border-top: 1px solid var\(--border\);/);
        assert.match(styles, /\.code-tabs-list \[role="tab"\]\[aria-selected="true"\]/);
        const navigation = read('docs/web/assets/js/navigation.js');
        assert.match(navigation, /function initCodeTabs/);
        assert.match(navigation, /event\.key === 'ArrowRight'/);
        assert.match(navigation, /event\.key === 'ArrowLeft'/);
        assert.match(app, /window\.initCodeTabs\(container\)/);
        const sdkContent = read('docs/web/content/sdk.html');
        assert.match(sdkContent, /href="\/artifacts\/sdk\/release\/"/);
        assert.match(sdkContent, /<details class="sdk-download">[\s\S]*?<summary class="btn-primary">/);
        assert.match(sdkContent, /href="\/artifacts\/sdk\/preview\/"/);
        assert.match(sdkContent, /href="\/artifacts\/sdk\/develop\/"/);
        assert.match(sdkContent, /data-sdk-version="release"/);
        assert.match(sdkContent, /data-sdk-version="preview"/);
        assert.match(sdkContent, /data-sdk-version="develop"/);
        assert.match(read('docs/web/assets/js/navigation.js'), /sdk-manifest-\$\{channel\}\.json/);
        assert.match(read('docs/web/assets/js/navigation.js'), /label\.textContent = `Version \$\{version\}`/);
        assert.match(read('docs/web/assets/js/navigation.js'), /details\.sdk-download\[open\][\s\S]*?!menu\.contains\(event\.target\)[\s\S]*?menu\.open = false/);
        assert.match(app, /window\.initSdkDownloadVersions\(container\)/);
        assert.match(sdkContent, /href="\/artifacts\/sdk\/api\/"/);
        assert.match(sdkContent, /href="\/sdk\/api\/" class="doc-link"/);
        assert.match(read('docs/web/.gitignore'), /^\/artifacts\/$/m);
    });

    it('maps Direct capabilities, runtime controls, and dual-band Wi-Fi safely', () => {
        const configure = index.match(/data-page="configure"[\s\S]*?<\/main>/)?.[0] || '';
        const mqtt = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const onboarding = configure.match(/class="js-configure-onboarding"[\s\S]*?<div class="js-configure-workspace"/)?.[0] || '';
        const configureBanner = configure.match(/class="device-banner-actions"[\s\S]*?<\/div>/)?.[0] || '';
        const mqttBanner = mqtt.match(/class="device-banner-actions"[\s\S]*?<\/div>/)?.[0] || '';
        assert.doesNotMatch(onboarding, /class="empty-alt"/);
        assert.match(index, /data-capability="supports_wifi_provisioning"/);
        assert.match(app, /if \(!Object\.prototype\.hasOwnProperty\.call\(snapshot, capability\)\) return;/);
        assert.match(app, /if \(!capabilities\.some\(\(key\) => Object\.prototype\.hasOwnProperty\.call\(snapshot, key\)\)\) return;/);
        assert.match(index, /class="field-row wifi-credentials-row">\s*<div class="field"><label for="cfg-ssid"[\s\S]*?<label for="cfg-wifi-pass"/);
        assert.match(index, /class="field-row field-row-2-1">\s*<div class="field"><label for="cfg-wifi-band"[\s\S]*?<label for="cfg-channel"/);
        assert.match(index, /id="cfg-wifi-band" disabled/);
        assert.match(app, /snapshot\.supports_wifi_5ghz/);
        assert.match(app, /select\.disabled = select\.options\.length === 1/);
        assert.match(app, /wifiBandPolicyAvailable \? \{ band_policy: bandPolicy \}/);
        assert.match(index, /class="conn-dropdown-meta"/);
        assert.match(index, /js-menu-chip[\s\S]*js-menu-device-id[\s\S]*js-menu-firmware/);
        assert.match(index, /class="mono-sub device-banner-identity"/);
        assert.match(index, /js-device-banner-sub[\s\S]*js-firmware-update-notice/);
        assert.match(index, /class="device-banner-name-editor js-configure-name-editor"[\s\S]*?js-configure-name-trigger[\s\S]*?js-configure-name-display[\s\S]*?class="device-banner-name-icon"[\s\S]*?aria-hidden="true"[\s\S]*?js-configure-name-input/);
        assert.match(index, /class="device-banner-name-editor js-monitor-name-editor"[\s\S]*?js-monitor-name-trigger[\s\S]*?js-monitor-name-display[\s\S]*?class="device-banner-name-icon"[\s\S]*?aria-hidden="true"[\s\S]*?js-monitor-name-input/);
        assert.doesNotMatch(index, /id="cfg-device-id"|id="cfg-device-name"|id="cfg-label"|js-dev-save/);
        assert.doesNotMatch(configure, /<h2>Device<\/h2>/);
        assert.match(configure, /class="config-grid connectivity-grid js-connectivity-setup"/);
        assert.match(styles, /\.connectivity-grid \{ grid-template-columns: minmax\(320px, 1fr\) minmax\(0, 2fr\); \}/);
        assert.equal([...index.matchAll(/class="device-firmware-update js-firmware-update-notice"/g)].length, 3);
        assert.match(index, /class="conn-dropdown-name js-device-name js-connection-device-name"/);
        assert.match(app, /function formatDeviceIdentityLine/);
        assert.match(app, /function renderConfigureDeviceNameEditor/);
        assert.match(app, /conn\.deviceLabel \|\| conn\.generatedName \|\| conn\.deviceId/);
        assert.match(app, /identity\.textContent = formatDeviceIdentityLine\(\s*conn\.chip,\s*conn\.deviceId,\s*conn\.firmwareVersion/);
        assert.match(app, /const chip = snapshot\.chip \? String\(snapshot\.chip\)\.toUpperCase\(\) : conn\.chip/);
        assert.match(app, /const firmware = snapshot\.firmware_version \|\| snapshot\.version \|\| conn\.firmwareVersion/);
        assert.match(app, /const mqttCanEdit = conn\.mode === 'mqtt'[\s\S]*?monitor\.commands\.has\('set_device_label'\)/);
        assert.match(app, /function startConfigureDeviceNameEdit/);
        assert.match(app, /function saveConfigureDeviceNameOnBlur/);
        assert.match(app, /nameInput\.addEventListener\('blur',[\s\S]*?saveConfigureDeviceNameOnBlur/);
        assert.match(app, /function renderMonitorDeviceNameEditor/);
        assert.match(app, /function startMonitorDeviceNameEdit/);
        assert.match(app, /function saveMonitorDeviceNameOnBlur/);
        assert.match(app, /js-monitor-name-trigger'\)\.addEventListener\('click', startMonitorDeviceNameEdit\)/);
        assert.match(app, /nameInput\.addEventListener\('blur',[\s\S]*?saveMonitorDeviceNameOnBlur/);
        assert.match(app, /function cfgSaveDeviceLabel\(label\)[\s\S]*?'set_device_label'/);
        assert.match(app, /function cfgSaveDeviceLabel\(label\)[\s\S]*?conn\.mode === 'mqtt'[\s\S]*?monitorPublishCommand[\s\S]*?command: 'set_device_label', device_label: label/);
        assert.match(styles, /\.device-banner-name-trigger \{[\s\S]*?display: inline-flex;[\s\S]*?cursor: pointer;[\s\S]*?\.device-banner-name-icon[\s\S]*?\.device-banner-name-input/);
        assert.match(app, /parts\.push\('Chip ' \+ chip\)/);
        assert.match(app, /parts\.push\('Device ID ' \+ deviceId\)/);
        assert.match(app, /parts\.push\('Firmware ' \+ firmware\)/);
        assert.match(app, /conn\.deviceBannerSub = conn\.mode === 'ws'[\s\S]*?\? deviceIdentity/);
        assert.match(app, /write\('\.js-menu-chip', identity\.chip\)/);
        assert.match(app, /write\('\.js-menu-device-id', identity\.deviceId\)/);
        assert.match(app, /write\('\.js-menu-firmware', identity\.firmwareVersion\)/);
        assert.match(index, /js-transport-tag[^>]*hidden/);
        assert.match(index, /js-menu-device-id-label/);
        assert.match(index, /js-menu-firmware-label/);
        assert.match(index, /js-usb-port-note[^>]*hidden/);
        assert.match(app, /const transportLabels = \{ ws: 'WS', usb: 'USB', mqtt: 'MQTT' \}/);
        assert.match(app, /const displayedMode = usbConnected \? 'usb' : conn\.mode/);
        assert.match(index, /energy-title/);
        assert.match(app, /conn\.motion \? 'MOTION' : 'IDLE'/);
        assert.match(styles, /\.conn-dropdown \{[^}]*color: var\(--text\);/);
        assert.match(styles, /\.conn-dropdown-name \{[^}]*color: var\(--text\);/);
        assert.match(mqtt, /data-device-view="live"/);
        assert.match(configure, /data-device-view="connectivity"/);
        assert.match(mqtt, /live-calibration" data-mqtt-command="recalibrate"/);
        assert.match(app, /statusFn: toast/);
        assert.match(app, /mqttCommand === 'recalibrate' && detector !== 'lightweight'/);
        assert.match(app, /function beginCalibration/);
        assert.match(app, /case 'ha\/calibrate\/state'/);
        assert.match(index, /js-sense-recalibrate/);
        assert.match(mqtt, /<details class="device-live-diagnostics">/);
        assert.doesNotMatch(mqtt, /<details class="device-live-diagnostics" open/);
        assert.match(app, /const showLiveEnergy = live/);
        assert.match(app, /js-device-edit-connectivity'\)\.addEventListener\('click', monitorEditOrCancel\)/);
        assert.match(app, /result: 'cancelled'/);
        assert.match(app, /monitor\.switchingTransport = false;[\s\S]*setStatus\('connected'\)/);
        assert.match(app, /await monitorConnect\(\)/);
        assert.match(app, /targetRoute = view === 'connectivity' \? 'configure' : 'monitor'/);
        assert.match(app, /location\.hash = '#' \+ targetRoute/);
        assert.doesNotMatch(app, /ble: 'configure'/);
        assert.match(app, /mqtt: 'monitor'/);
        assert.match(app, /device: 'configure'/);
        assert.match(app, /const directSetup = connected && conn\.mode === 'ws'/);
        assert.match(app, /const directConnecting = conn\.status === 'connecting'/);
        assert.match(app, /const mqttConnectionPending = monitorConnectionPending\(\)/);
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
        assert.match(app, /const GAME_THRESHOLD_DEFAULT = 0\.5/);
        assert.match(app, /function gameThreshold/);
        assert.match(app, /function resetGameThreshold/);
        assert.match(app, /gameThresholdOverride = GAME_THRESHOLD_DEFAULT/);
        assert.doesNotMatch(app, /gameThresholdOverride = conn\.threshold/);
        assert.match(app, /if \(target === 'game' && previousRoute !== 'game'\) resetGameThreshold\(\)/);
        assert.match(app, /gameThresholdOverride = threshold/);
        assert.match(app, /js-game-fullscreen-threshold/);
        assert.match(app, /const applyFullscreenPointerThreshold = \(event\) =>/);
        assert.match(app, /\(bounds\.bottom - event\.clientY\) \/ bounds\.height/);
        assert.doesNotMatch(app, /textContent = 'Restart';\s*resetGameThreshold\(\)/);
        assert.match(app, /getElementById\('game-threshold'\)/);
        assert.match(index, /id="game-threshold"/);
        assert.doesNotMatch(index, /data-page="game"[\s\S]*data-mqtt-command="set_threshold"/);
        assert.doesNotMatch(app, /gameSlider\.addEventListener\('change', \(\) => commitThreshold/);
        assert.match(index, /<div class="game-screen">[\s\S]*<div class="game-status"[\s\S]*js-game-score[\s\S]*js-game-best[\s\S]*js-game-fullscreen/);
        assert.equal((index.match(/class="js-game-score"/g) || []).length, 1);
        assert.doesNotMatch(index, /class="game-stats"/);
        assert.match(app, /function gameToggleFullscreen/);
        assert.match(app, /requestFullscreen \|\| screen\.webkitRequestFullscreen/);
        assert.match(app, /document\.addEventListener\('fullscreenchange', gameOnFullscreenChange\)/);
        assert.match(styles, /\.game-screen:fullscreen/);
        assert.match(styles, /\.game-screen:fullscreen \.game-canvas/);
        assert.match(styles, /\.game-screen:fullscreen \.game-motion-gauge/);
        assert.match(index, /class="game-motion-threshold-input js-game-fullscreen-threshold"/);
        assert.match(styles, /\.game-motion-threshold-input \{[\s\S]*?cursor: ns-resize;/);
        assert.match(styles, /\.game-status \{[\s\S]*?right: 14px;/);
        assert.match(styles, /\.game-motion-gauge \{[\s\S]*?top: 0;[\s\S]*?bottom: 0;/);
        assert.match(styles, /\.game-screen\[data-phase="done"\] \.game-status \{[\s\S]*?top: 50%;[\s\S]*?left: 50%;[\s\S]*?transform: translate\(-50%, -50%\);/);
        assert.match(styles, /\.game-screen\[data-phase="done"\] \.game-score strong \{[\s\S]*?font-size: clamp\(52px, 10vw, 78px\);/);
        assert.match(app, /getElementById\('sense-detector'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-on'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-off'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-csi-mode'\)\.addEventListener\('change'/);
        assert.match(index, /<select id="sense-csi-mode">[\s\S]*?<option value="internal"[\s\S]*?<option value="external"[\s\S]*?<option value="disabled"/);
        assert.match(app, /getElementById\('sense-generator-mode'\)\.addEventListener\('change'/);
        assert.doesNotMatch(index, /device-connect-kicker/);
        assert.doesNotMatch(styles, /\.device-connect-kicker/);
        assert.match(index, /class="js-configure-onboarding"/);
        assert.match(index, /class="js-monitor-onboarding"/);
        assert.match(index, /data-transport="ws"/);
        assert.match(index, /data-monitor-transport-panel="mqtt" data-transport="mqtt"/);
        assert.ok(onboarding.indexOf('device-recovery-hint') < onboarding.indexOf('js-connect-direct'));
        assert.match(onboarding, /EN \/ RST[\s\S]*BOOT \/ FLASH/);
        assert.match(configureBanner, /js-start-detection/);
        assert.match(mqttBanner, /js-device-edit-connectivity/);
        assert.match(app, /function connectDirect/);
        assert.match(app, /function directConnectionErrorMessage\(error, endpoint, permissionState/);
        assert.match(app, /function directPageOriginKind\(\)/);
        assert.match(app, /function directBrowserGuidance\(\)/);
        assert.match(app, /browserSupport\.hostedDirect === 'unclaimed'/);
        assert.match(app, /function renderDirectBrowserGuidance\(\)/);
        assert.match(app, /renderDirectBrowserGuidance\(\);/);
        assert.match(app, /permissionState === 'denied'/);
        assert.match(app, /browserSupport\.hostedDirect === 'unsupported'/);
        assert.match(app, /Close other ESPectre Configure or Monitor tabs/);
        assert.match(app, /function setDirectConnectionHelp\(message = ''\)/);
        assert.match(app, /await localNetworkAccessState\(\)/);
        assert.equal((index.match(/class="direct-connection-help js-(?:cfg|mon)-direct-help" role="alert" hidden/g) || []).length, 2);
        assert.equal((index.match(/class="tool-note js-direct-browser-note" role="status"/g) || []).length, 2);
        assert.match(index, /Browser and local-network help/);
        assert.match(security, /portal does not scan the LAN or enumerate DNS-SD services/);
        assert.match(security, /two Direct client slots/);
        assert.match(app, /error\.code \|\| error\.name/);
        assert.match(app, /await client\.handshake\(\)/);
        assert.match(app, /DIRECT_RECONNECT_DELAYS_MS = Object\.freeze\(\[500, 1500, 3000\]\)/);
        assert.match(app, /function scheduleDirectReconnect\(client\)/);
        assert.match(app, /await client\.connect\(\{ timeoutMs: 5000 \}\)/);
        assert.match(app, /if \(pendingConfigVerification\) requestConfigVerification\(\)/);
        assert.match(index, /list="direct-remembered-endpoints"/);
        assert.match(index, /id="direct-remembered-endpoints"/);
        assert.match(index, /js-direct-share[\s\S]*js-direct-forget/);
        assert.match(index, /class="modal-backdrop js-direct-share-modal"/);
        assert.match(app, /DIRECT_ENDPOINT_STORAGE_KEY = 'espectre\.direct\.endpoints\.v1'/);
        assert.match(app, /function rememberDirectEndpoint\(endpoint\)/);
        assert.match(app, /function forgetDirectEndpoint\(\)/);
        assert.match(app, /function directShareUrl\(endpoint\)/);
        assert.match(app, /function consumeDirectHandoff\(\)/);
        assert.match(app, /params\.get\('transport'\) !== 'ws'/);
        assert.match(app, /DirectProtocolClient\.normalizeEndpoint\(params\.get\('endpoint'\)\)/);
        assert.match(app, /await client\.request\('start_sensing'\)/);
        assert.doesNotMatch(app, /set_ble|STOP_BLE/);
        assert.match(index, /class="modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /class="btn-primary js-ota-start" disabled/);
        assert.match(index, /id="cfg-ota-message"/);
        assert.match(app, /function applyOtaStatus/);
        assert.match(app, /function flashDialogText\(root\)/);
        assert.match(app, /function activateUsbConnection\(dialog\)/);
        assert.match(app, /dialog\.port && typeof dialog\.port\.getInfo === 'function'/);
        assert.match(app, /activateUsbConnection\(dialog\);/);
        assert.match(app, /dialog\.addEventListener\('closed', \(\) => scheduleUsbConnectionRelease\(dialog\)/);
        assert.match(app, /scheduleUsbConnectionRelease\(dialog\);[\s\S]*if \(started && !reported\) report\('cancelled'\)/);
        assert.match(app, /if \(element\.shadowRoot\) text \+= ' ' \+ flashDialogText\(element\.shadowRoot\)/);
        assert.match(app, /const observedRoots = new WeakSet\(\)/);
        assert.match(app, /observeRoot\(dialog\.shadowRoot\);[\s\S]*flashDialogText\(dialog\.shadowRoot\)/);
        assert.match(app, /headline\.textContent\.trim\(\) === 'Visit Device'[\s\S]*'Configure Device'/);
        const installDialog = read('docs/web/vendor/esp-web-tools-10.4.0/install-dialog-im156JnI.js');
        assert.equal(
            installDialog.match(/this\._manifest\.name==="ESPectre Native"\?"Configure Device":"Visit Device"/g)?.length,
            2
        );
        assert.match(app, /snapshot\.wifi_bssid\.toLowerCase\(\) === bssid\.toLowerCase\(\)/);
        assert.match(app, /directClient\.request\('clear_wifi_config', \{\}, \{ timeoutMs: 3000 \}\)/);
        assert.match(app, /error\?\.code !== 'timeout' && error\?\.code !== 'closed'/);
        assert.match(app, /Wi-Fi clear sent\. The device disconnected as expected; provision it again via Improv Serial\./);
        assert.match(app, /teardownConnection\('wifi_cleared'\)/);
        assert.match(app, /track\('firmware_installer_open', flashParams\(\)\)/);
        assert.match(app, /dialog\._installConfirmed === true[\s\S]*markStarted\(\)/);
        assert.match(app, /track\('firmware_install_start', flashParams\(\)\)/);
        assert.match(app, /installState === 'finished'[\s\S]*report\('success'\)/);
        assert.match(app, /inspectTimer = setInterval\(inspect, 250\)/);
        assert.match(app, /clearInterval\(inspectTimer\)/);
        assert.match(styles, /ewt-install-dialog \{[\s\S]*--md-dialog-container-shape: 16px/);
        assert.match(styles, /ewt-install-dialog \{[\s\S]*--md-dialog-headline-font: 'Space Grotesk'/);
        assert.match(styles, /ewt-install-dialog \{[\s\S]*--md-sys-color-primary: var\(--accent\)/);
        assert.match(styles, /ewt-install-dialog \{[\s\S]*--md-circular-progress-active-indicator-color: var\(--accent\)/);
        assert.match(app, /function startSilentOtaCheck/);
        assert.match(app, /function maybeStartSilentOtaCheck/);
        assert.match(app, /function currentOtaCheckTransport/);
        assert.match(app, /supportsOta \? directClient\.request\('ota_status'\) : Promise\.resolve\(null\)/);
        assert.match(app, /if \(otaStatus\) applyOtaStatus\(otaStatus\)/);
        assert.match(app, /if \(conn\.mode === 'demo'\) return;/);
        assert.match(app, /if \(conn\.mode === 'ws' && directClient\?\.connected\) return 'ws'/);
        assert.match(app, /if \(!transport\) return;/);
        assert.match(app, /if \(!manual && transport && otaCheckTransport === transport\) return;/);
        assert.match(app, /if \(!otaDefaultChannel \|\| otaBusy\) return;/);
        assert.match(app, /maybeStartSilentOtaCheck\(\);/);
        assert.doesNotMatch(app, /Bluetooth|Web Bluetooth/);
        assert.match(app, /otaSupported = monitor\.commands\.has\('ota_check'\) && monitor\.commands\.has\('ota_start'\)/);
        assert.match(app, /otaTransportReady = conn\.mode === 'ws'/);
        assert.match(app, /el\.hidden = Boolean\(flash\.usbDialog\) \|\| otaSupported === false/);
        assert.match(app, /if \(!currentOtaCheckTransport\(\)\) return;/);
        assert.match(app, /monitorPublishCommand\(otaCommandFields\('ota_start'\)/);
        assert.match(app, /OTA_TRACKING_TIMEOUT_MS/);
        assert.match(app, /otaBusy \|\| otaTracking \|\| otaState === 'reboot_scheduled'/);
        assert.match(app, /if \(otaTargetVersion && version !== otaTargetVersion\)/);
        assert.match(app, /finishOtaTracking\('success', null, 'reconnected'\)/);
        assert.match(app, /if \(otaAwaitingReconnect\) completeOtaReconnect\(\)/);
        assert.match(index, /js-menu-firmware[\s\S]*js-firmware-update-notice[\s\S]*js-disconnect/);
        assert.match(index, /class="conn-firmware-row"/);
        assert.doesNotMatch(index, /device-firmware-update-icon/);
        assert.doesNotMatch(styles, /device-firmware-update-icon/);
        assert.match(app, /\$\$\('\.js-firmware-update-notice'\)\.forEach\(\(button\) => \{/);
        assert.match(app, /button\.addEventListener\('click', \(event\) => otaOpen\(event\.currentTarget\)\)/);
        assert.match(app, /status = 'error'/);
        assert.match(app, /function otaOpen\(returnFocus\)/);
        assert.match(index, /id="ota-channel"/);
        assert.match(index, /<option value="release" selected/);
        assert.match(app, /function selectedOtaChannel/);
        assert.match(app, /function applyOtaDefaultChannel/);
        assert.match(app, /applyOtaDefaultChannel\(status\.default_channel/);
        assert.match(app, /otaChannelChanged = true/);
        assert.match(app, /return channel \? \{ command, channel \} : \{ command \}/);
        assert.match(app, /if \(conn\.mode === 'demo'\) return;/);
        assert.match(index, /<h2 class="panel-title-status">Wi-Fi <span class="dot dot-idle js-wifi-status-dot"/);
        assert.match(index, /<h2 class="panel-title-status">MQTT <span class="dot dot-idle js-mqtt-status-dot"/);
        assert.match(app, /setConnectionDot\('\.js-wifi-status-dot', snapshot\.wifi_connected\)/);
        assert.match(app, /setConnectionDot\('\.js-mqtt-status-dot', snapshot\.mqtt_connected\)/);
        assert.match(styles, /\.dot-error \{ background: var\(--danger\); \}/);
    });

    it('keeps MQTT diagnostics collapsed below Live and hides demo when live', () => {
        const mqttPage = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const broker = mqttPage.match(/<section class="device-connect-card[^"]*"[^>]*data-transport="mqtt"[^>]*>[\s\S]*?<\/section>/)?.[0] || '';
        const diagnostics = mqttPage.match(/<details class="device-live-diagnostics">[\s\S]*?<\/details>/)?.[0] || '';
        assert.match(broker, /<div class="fields">/);
        assert.match(broker, /js-mon-connect/);
        assert.match(broker, /class="tool-note js-mon-status" role="status" hidden><\/div>/);
        assert.match(diagnostics, /js-mon-diag-status/);
        assert.match(diagnostics, /js-mon-admitted/);
        assert.match(diagnostics, /js-mon-filtered[\s\S]*js-mon-admitted/);
        assert.match(diagnostics, /class="diag-grid mon-stats-tiles"/);
        assert.match(app, /function syncDiagnosticsPolling/);
        assert.match(app, /setInterval\(monitorRequestStats, interval\)/);
        assert.match(app, /function stopDiagnosticsPolling/);
        assert.match(app, /command: direct \? 'diagnostics' : 'stats'/);
        assert.match(app, /if \(direct && data && typeof data === 'object'\)[\s\S]*markMonitorReady\('diagnostics'\);[\s\S]*monitorStats\(data\)/);
        assert.match(app, /if \(conn\.mode === 'ws'\) return monitor\.diagRequestPending/);
        assert.match(app, /data\.csi_admitted_pps/);
        assert.match(app, /data\.csi_filtered_pps/);
        assert.match(app, /function monitorIsMqttLive/);
        assert.match(app, /function monitorConnectionPending/);
        assert.match(app, /const mqttSession = live \|\| \(monitor\.handoffReady && monitorIsMqttLive\(\)\)/);
        assert.match(app, /connect\.disabled = mqttConnected/);
        assert.match(app, /function monitorCancelConnection/);
        assert.match(app, /if \(monitorConnectionPending\(\)\) \{[\s\S]*monitorCancelConnection\(\)/);
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
        assert.match(app, /function monitorShowDeviceSelection/);
        assert.match(app, /monitorStatus\('Select a device, or enter a device ID\.'\);[\s\S]*monitorShowDeviceSelection\(\)/);
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
        assert.match(app, /case 'commands\/catalog'/);
        assert.match(app, /monitor\.commandCatalogReady = true/);
        assert.match(app, /function monitorOpenConnectivity/);
        assert.match(app, /if \(conn\.mode === 'ws' && directClient\?\.connected\)/);
    });

    it('offers MQTT broker presets and splits device TCP from browser WebSockets', () => {
        assert.match(index, /id="cfg-mqtt-preset"[\s\S]*?value="home_assistant" selected[\s\S]*?value="lan_broker"[\s\S]*?value="emqx_cloud"[\s\S]*?value="hivemq_cloud"[\s\S]*?value="flespi"[\s\S]*?value="cloud_broker"/);
        assert.match(index, /class="field-row mqtt-host-port-row">[\s\S]*?id="cfg-mqtt-host"[\s\S]*?id="cfg-mqtt-port"/);
        assert.doesNotMatch(index, /<optgroup label="Cloud brokers">/);
        assert.doesNotMatch(index, /value="local_broker"/);
        assert.match(index, /id="cfg-mqtt-host"[^>]*value="homeassistant.local"/);
        assert.match(index, /id="cfg-mqtt-port"[^>]*value="1883"/);
        assert.doesNotMatch(index, /id="cfg-mqtt-user"[^>]*value=/);
        assert.doesNotMatch(index, /id="cfg-mqtt-pass"[^>]*value=/);
        assert.match(index, /id="cfg-mqtt-user"[^>]*placeholder="MQTT username"/);
        assert.match(index, /id="cfg-mqtt-pass"[^>]*placeholder="MQTT password"/);
        assert.doesNotMatch(index, /js-cfg-mqtt-preset-note|Enter the MQTT credentials created for ESPectre\./);
        assert.match(index, /id="cfg-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(index, /id="mon-mqtt-preset"[\s\S]*?value="home_assistant" selected[\s\S]*?value="lan_broker"[\s\S]*?value="emqx_cloud"[\s\S]*?value="hivemq_cloud"[\s\S]*?value="flespi"[\s\S]*?value="cloud_broker"/);
        assert.match(index, /id="mon-host"[^>]*value="homeassistant.local"/);
        assert.match(index, /id="mon-port"[^>]*value="9001"/);
        assert.doesNotMatch(index, /id="mon-user"[^>]*value=/);
        assert.doesNotMatch(index, /id="mon-pass"[^>]*value=/);
        assert.match(index, /id="mon-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(index, /id="mon-device"[^>]*placeholder="3cf79180d3a0aca4"/);
        assert.match(index, /id="mon-device-choice" class="device-choice-list"/);
        assert.match(index, /class="field js-mon-device-picker"/);
        assert.match(index, /id="mon-path"[^>]*value="\/mqtt"/);
        assert.match(index, /id="mon-tls"/);
        const mqttPage = index.match(/data-page="monitor"[\s\S]*?<\/main>/)?.[0] || '';
        const onboardingHtml = mqttPage.match(/class="js-monitor-onboarding"[\s\S]*?<div class="js-monitor-workspace"/)?.[0] || '';
        const broker = mqttPage.match(/<section class="device-connect-card[^"]*"[^>]*data-transport="mqtt"[^>]*>[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(onboardingHtml, /<div class="fields">/);
        assert.match(broker, /id="mon-host"[\s\S]*id="mon-user"[\s\S]*id="mon-port"[\s\S]*id="mon-topic-prefix"[\s\S]*id="mon-device"[\s\S]*js-mon-connect/);
        assert.match(broker, /id="mon-path"/);
        assert.match(broker, /id="mon-tls"/);
        assert.match(mqttProtocol, /static baseTopic/);
        assert.match(app, /function applyConfigureMqttDefaults/);
        assert.match(app, /const MQTT_PRESETS = Object\.freeze/);
        assert.match(app, /home_assistant:[\s\S]*?host: 'homeassistant\.local', port: '1883'/);
        assert.doesNotMatch(app, /local_broker:/);
        assert.match(app, /lan_broker:[\s\S]*?configure:[\s\S]*?host: '', port: '1883'[\s\S]*?monitor:[\s\S]*?host: 'localhost', port: '9001'/);
        assert.match(app, /emqx_cloud:[\s\S]*?host: 'deployment-id\.ala\.region\.emqxsl\.com', port: '8883'[\s\S]*?host: 'deployment-id\.ala\.region\.emqxsl\.com', port: '8084', path: '\/mqtt', tls: true/);
        assert.match(app, /hivemq_cloud:[\s\S]*?host: 'cluster-id\.s1\.region\.hivemq\.cloud', port: '8883'[\s\S]*?host: 'cluster-id\.s1\.region\.hivemq\.cloud', port: '8884', path: '\/mqtt', tls: true/);
        assert.match(app, /flespi:[\s\S]*?host: 'mqtt\.flespi\.io', port: '8883'[\s\S]*?host: 'mqtt\.flespi\.io', port: '443', path: '\/mqtt', tls: true/);
        assert.match(app, /cloud_broker:[\s\S]*?configure:[\s\S]*?host: 'cluster\.example\.com', port: ''[\s\S]*?monitor:[\s\S]*?host: 'cluster\.example\.com', port: '', path: '\/mqtt', tls: true/);
        assert.match(app, /topicPrefix: 'espectre\/v1\/devices'/);
        assert.match(app, /function applyConfigureMqttPreset/);
        assert.match(app, /function applyMonitorMqttPreset/);
        assert.match(app, /function applyMqttPresetFieldLocks/);
        assert.match(app, /function applyConfigureMqttCredentialPolicy/);
        assert.match(app, /username\.placeholder = isFlespi \? 'Token' : 'MQTT username'/);
        assert.match(app, /password\.placeholder = isFlespi \? 'No password' : 'MQTT password'/);
        assert.match(app, /password\.disabled = isFlespi/);
        assert.match(app, /if \(isFlespi\) password\.value = ''/);
        assert.match(app, /locked: Object\.freeze\(\['port'\]\)/);
        assert.match(app, /locked: Object\.freeze\(\['port', 'path', 'tls'\]\)/);
        assert.match(app, /locked: Object\.freeze\(\['host', 'port'\]\)/);
        assert.match(app, /locked: Object\.freeze\(\['host', 'port', 'path', 'tls'\]\)/);
        assert.match(app, /input\.readOnly = isLocked/);
        assert.match(app, /input\.disabled = isLocked/);
        assert.match(styles, /\.field input\[data-preset-locked\][\s\S]*?cursor: not-allowed/);
        assert.match(styles, /\.broker-host-control \{[\s\S]*?grid-template-columns: 170px minmax\(0, 1fr\);[\s\S]*?\.broker-host-control select[\s\S]*?\.broker-host-control input/);
        assert.match(styles, /\.mqtt-host-port-row \{ grid-template-columns: minmax\(0, 1fr\) 96px; \}/);
        assert.match(app, /const SECURE_CLOUD_MQTT_PRESETS = new Set/);
        assert.match(app, /SECURE_CLOUD_MQTT_PRESETS\.has\(presetName\)[\s\S]*?'mqtts:\/\/' \+ enteredHost/);
        assert.match(app, /function applyConfigureMqttToMonitor/);
        assert.match(app, /if \(host && browserBrokerHost\(host\.value\)\)/);
        assert.match(app, /function startDetection/);
        assert.match(app, /bindMqttToConnection/);
        assert.match(app, /if \(directClient\) \{[\s\S]*?client\.close\(\)/);
        assert.match(index, /js-start-detection/);
        assert.match(index, /js-connect-direct/);
        assert.match(index, /class="conn-pill conn-disconnected js-header-connect"/);
        assert.match(
            app,
            /\$\('\.js-header-connect'\)\.addEventListener\('click', \(\) => \{[\s\S]*?location\.hash = '#configure';/
        );
        assert.match(index, /js-has-live/);
        assert.doesNotMatch(app, /getElementById\('mon-port'\)\.value = .*cfg-mqtt-port/);
        assert.match(app, /input_mode: connectionInputMode\(\)/);
    });

    it('keeps Game and Theremin demo sessions on the current tool', () => {
        assert.match(index, /data-page="theremin"[\s\S]*?class="link-btn js-demo"/);
        assert.match(index, /data-page="game"[\s\S]*?class="link-btn js-demo"/);
        assert.match(app, /function connectDemo\(\)[\s\S]*?rememberLiveDestination\(\)/);
        assert.match(app, /function connectDemo\(\)[\s\S]*?completeLiveConnectionNavigation\(\)/);
    });

    it('returns live connection flows to their requesting experience', () => {
        assert.match(app, /const LIVE_EXPERIENCE_ROUTES = new Set\(\['game', 'theremin'\]\)/);
        assert.match(app, /function rememberLiveDestination/);
        assert.match(app, /function completeLiveConnectionNavigation/);
        assert.match(app, /const destination = pendingLiveDestination;[\s\S]*?pendingLiveDestination = ''/);
        assert.match(app, /function startDetection\(\) \{\s*rememberLiveDestination\(\)/);
        assert.match(app, /function bindMqttToConnection\(\)[\s\S]*?completeLiveConnectionNavigation\(\)/);
        assert.match(app, /monitor\.entryPoint = connectionIntentRoute\(\)/);
        assert.match(app, /target !== 'monitor' && target !== 'configure'/);
        assert.match(app, /if \(pendingLiveDestination \|\| route === 'monitor' \|\| route === 'configure'\)/);
        assert.match(
            app,
            /\$\('\.js-header-connect'\)\.addEventListener\('click',[\s\S]*?rememberLiveDestination\(\);[\s\S]*?location\.hash = '#configure'/
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

    it('runs Game as a binary motion-flight endless runner', () => {
        const gameScreen = index.match(/<div class="game-screen">[\s\S]*?<\/div>\s*<div class="game-msg/)?.[0] || '';
        const gameFinish = app.slice(app.indexOf('function gameFinish'), app.indexOf('function gameSetFlight'));
        const gameUpdate = app.slice(app.indexOf('function gameUpdate(dt)'), app.indexOf('function gameSensingActive'));
        assert.match(index, /class="game-canvas js-game-canvas"/);
        assert.match(styles, /\.game-stage \{[\s\S]*?width: 100%;[\s\S]*?max-width: none;/);
        assert.match(styles, /\.game-canvas \{[\s\S]*?height: clamp\(250px, 40vw, 440px\);/);
        assert.match(gameScreen, /class="game-status"[\s\S]*js-game-score[\s\S]*js-game-orbs[\s\S]*js-game-distance[\s\S]*js-game-best/);
        assert.match(gameScreen, /class="game-play js-game-start"/);
        assert.match(gameScreen, /class="game-sound js-game-sound"[\s\S]*?aria-pressed="true"/);
        assert.equal((index.match(/js-game-start/g) || []).length, 1);
        assert.match(gameScreen, /js-game-motion-fill[\s\S]*js-game-motion-threshold/);
        assert.match(app, /function renderGameMotionGauge/);
        assert.match(app, /fill\.style\.height = Math\.round\(energyFraction\(\) \* 100\) \+ '%'/);
        assert.match(app, /play\.hidden = phase === 'ready' \|\| phase === 'running'/);
        assert.doesNotMatch(gameFinish, /gameExitFullscreen/);
        assert.match(app, /const GAME_ORB_POINTS = 100/);
        assert.match(app, /function gameFlightY/);
        assert.match(app, /function gameOrbY\(lane\)/);
        assert.match(app, /y: gameOrbY\(lane\),\s*lane,/);
        assert.match(app, /entity\.y = gameOrbY\(entity\.lane\)/);
        assert.doesNotMatch(app, /entity\.y = gameGroundY\(\) - \(oldGround - entity\.y\) \* scaleY/);
        assert.match(app, /Math\.min\(maxAir, Math\.max\(0, air \* scaleY\)\)/);
        assert.match(app, /function gameSpawnCourse/);
        assert.match(app, /gameAddOrb\([^;]+, 'high'\)/);
        assert.match(app, /gameAddOrb\([^;]+, 'low'\)/);
        assert.match(app, /gameAddObstacle\('aerial_spikes'/);
        assert.match(app, /function gameRectsOverlap/);
        assert.match(app, /function gameOrbTouchesPlayer/);
        assert.match(app, /game\.raf = requestAnimationFrame\(gameFrame\)/);
        assert.match(app, /const targetY = game\.flightActive \? gameFlightY\(\)/);
        assert.match(app, /const responseSeconds = game\.flightActive \? 0\.15 : 0\.18/);
        assert.match(app, /function gameUpdatePlayer\(dt\)/);
        assert.match(app, /function gamePreviewFrame\(now\)/);
        assert.match(app, /game\.phase === 'idle' \|\| game\.phase === 'ready'/);
        assert.match(app, /function gameStartPreview\(\)/);
        assert.match(app, /gameStartPreview\(\);/);
        assert.match(app, /const canFloat = game\.phase === 'idle' \|\| game\.phase === 'ready' \|\| game\.phase === 'running'/);
        assert.match(app, /const bobAmplitude = game\.flightActive \? 2\.6/);
        assert.match(app, /gameSetFlight\(gameSensingActive\(\)\)/);
        assert.match(app, /function gameDemoFlight/);
        assert.match(app, /\['idle', 'ready', 'running'\]\.includes\(game\.phase\)/);
        assert.match(gameUpdate, /const player = game\.player/);
        assert.match(app, /function gameScore\(\)/);
        assert.match(app, /const GAME_MUSIC_NOTES = \[/);
        assert.match(app, /function gameStartMusic\(\)/);
        assert.match(app, /function gameStopMusic\(\)/);
        assert.match(app, /function gameStartMotionSound\(\)/);
        assert.match(app, /function gameUpdateMotionSound\(dt\)/);
        assert.match(app, /const frequency = 96 \* Math\.pow\(2, gameAudio\.motionSmoothed \* 1\.8\);/);
        assert.match(gameUpdate, /gameUpdateMotionSound\(dt\)/);
        assert.match(app, /function gamePlaySound\(kind\)/);
        assert.match(app, /gamePlaySound\('start'\)/);
        assert.match(app, /gamePlaySound\('orb'\)/);
        assert.match(gameFinish, /gamePlaySound\('hit'\)/);
        assert.match(app, /function gameToggleSound\(\)/);
        assert.match(app, /js-game-sound'\)\.addEventListener\('click', gameToggleSound\)/);
        assert.match(styles, /\.game-sound\[aria-pressed="false"\]/);
        assert.match(app, /Math\.floor\(game\.distance\) \+ game\.orbs \* GAME_ORB_POINTS/);
        assert.match(app, /game\.score = gameScore\(\)/);
        assert.match(app, /game\.scrollX \+= travel/);
        assert.match(app, /entity\.x -= travel/);
        assert.match(app, /game\.scrollX \* 0\.18/);
        assert.match(app, /- game\.scrollX\) % width/);
        assert.match(app, /gameFactoryImage\.src = '\/assets\/images\/game\/hardware-factory\.png'/);
        assert.match(app, /function gameDrawFactoryBackdrop/);
        assert.match(app, /function gameDrawFactoryParallax/);
        assert.match(app, /game\.scrollX \* 0\.12/);
        assert.match(app, /game\.scrollX \* 0\.24/);
        assert.match(app, /function gameDrawChip/);
        assert.match(app, /gameFactoryImage\.addEventListener\('load', gameDraw\)/);
        assert.ok(readFileSync(new URL('../../docs/web/assets/images/game/hardware-factory.png', import.meta.url)).length > 100000);
        assert.doesNotMatch(app, /game\.distance \* 12/);
        assert.match(app, /if \(obstacleHit\) gameFinish\(\)/);
        assert.doesNotMatch(app, /TOTAL_ROUNDS|game\.phase === 'strike'|game\.inputArmed|function gameJump/);
    });
});
