/*
 * ESPectre - Website structural contract tests
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');
const index = read('docs/web/index.html');
const app = read('docs/web/assets/js/app.js');
const browserSupportSource = read('docs/web/assets/js/browser-support.js');
const routeRegistry = read('docs/web/assets/js/route-registry.js');
const styles = read('docs/web/assets/css/styles.css');
const GPL_HTML_HEADER = `<!--
  SPDX-License-Identifier: GPL-3.0-only
  Commercial licensing available under separate agreement; see LICENSING.md.
-->
`;

describe('website security and asset policy', () => {
    it('does not execute third-party scripts before an explicit analytics choice', () => {
        const externalScripts = [...index.matchAll(/<script[^>]+src="(https?:[^\"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(externalScripts, []);
        assert.doesNotMatch(index, /unpkg\.com|jsdelivr\.net/);
        assert.match(index, /\/assets\/css\/styles\.css/);
        assert.match(index, /\/assets\/js\/app\.js/);
        assert.match(index, /\/assets\/js\/browser-support\.js/);
        assert.match(index, /\/assets\/js\/route-registry\.js/);
        assert.ok(index.indexOf('/assets/js/route-registry.js') < index.indexOf('/assets/js/analytics.js'));
        assert.ok(index.indexOf('/assets/js/route-registry.js') < index.indexOf('/assets/js/app.js'));
        assert.ok(index.indexOf('/assets/js/browser-support.js') < index.indexOf('/assets/js/app.js'));
        assert.match(app, /\/vendor\/esp-web-tools-10\.4\.0\/install-button\.js/);
        assert.match(app, /\/vendor\/mqtt-5\.3\.0\/mqtt\.min\.js/);
        assert.match(app, /\/vendor\/qrcodejs-1\.0\.0\/qrcode\.min\.js/);
        assert.match(app, /LOCAL_DEVELOPMENT_HOSTS = new Set\(\['localhost', '127\.0\.0\.1', '\[::1\]'\]\)/);
    });

    it('keeps cache-busting versions in lockstep', () => {
        const versions = new Set([...index.matchAll(/[?&]v=([0-9.]+)/g)].map((match) => match[1]));
        assert.equal(versions.size, 1);
        const notFoundVersions = new Set(
            [...read('docs/web/404.html').matchAll(/[?&]v=([0-9.]+)/g)]
                .map((match) => match[1])
        );
        assert.deepEqual([...notFoundVersions], [...versions]);
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
        assert.match(app, /monitor\.pendingCommands\.get\(data\.command_id\)/);
        assert.match(app, /suffix === 'commands\/accepted'/);
        assert.match(app, /suffix === 'commands\/rejected'/);
        assert.match(app, /\.\.\.connectionParams\(\)/);
    });

    it('tracks abandonment and only reports valid download targets', () => {
        assert.match(app, /track\('game_abandon'/);
        assert.match(app, /reportGameAbandon\('restart'\)/);
        assert.match(app, /reportGameAbandon\('route_change'\)/);
        assert.match(app, /reportGameAbandon\('page_exit'\)/);
        assert.match(app, /if \(flash\.downloadReady\)/);
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
        assert.match(read('.github/scripts/build_static_pages.py'), /route-registry\.js/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /route-registry\.js/);
    });

    it('has a responsive navigation control and a live status region', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
    });

    it('provides skip navigation, stable tool headings, and route focus management', () => {
        assert.match(index, /<a class="skip-link" href="#main-content">Skip to content<\/a>/);
        assert.match(index, /data-page="home" id="main-content" tabindex="-1"/);
        for (const heading of ['Device console', 'Motion theremin', 'Motion reaction game']) {
            assert.match(index, new RegExp(`<h1 class="page-title">${heading}<\\/h1>`));
        }
        assert.doesNotMatch(index, /<h1>No device connected<\/h1>/);
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
    it('uses natural scrolling and progressively loads narrative images', () => {
        assert.doesNotMatch(app, /function scrollyWheel|function scrollyTouch|function stepScrolly/);
        assert.match(index, /data-src-mobile="\/assets\/images\/home\/scene-motion-lights-mobile\.webp"/);
        assert.match(index, /data-src="\/assets\/images\/home\/scene-standards-backend\.jpg" data-src-mobile="\/assets\/images\/home\/scene-standards-backend-mobile\.webp"/);
        assert.match(app, /image\.dataset\.srcMobile/);
        const sceneIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-scene\b[^"]*" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        const captionIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-caption\b[^"]*" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        const markerIds = [...index.matchAll(/class="js-scrolly-marker" data-scene="(\d+)"/g)].map((match) => Number(match[1]));
        assert.deepEqual(sceneIds, Array.from({ length: 13 }, (_, index) => index));
        assert.deepEqual(captionIds, sceneIds);
        assert.deepEqual(markerIds, sceneIds.slice(1));
        assert.match(index, /<span>\/ 12<\/span>/);
        assert.match(index, /href="#get-started" class="hero-skip">Skip the story/);
        assert.match(index, /<section class="home-section" id="get-started">/);
    });

    it('labels research and preview concepts without presenting simulated evidence', () => {
        assert.match(index, /Roadmap · Breathing/);
        assert.match(index, /Experimental <em>R&amp;D only<\/em>/);
        assert.doesNotMatch(index, /13\.2 <em>cycles\/min<\/em>/);
        assert.match(index, /Matter · Limited validation/);
        assert.match(index, /Controller-dependent/);
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
        assert.doesNotMatch(app, /starting demo mode|connectDemo\(\);\s*return;[\s\S]{0,120}result: 'unsupported'/);
        assert.match(app, /installTrigger\.disabled = !browserSupport\.flash/);
        assert.match(app, /button\.disabled = !browserSupport\.bluetooth/);
        assert.match(app, /if \(browserSupport\.flash\) \{\s*loadBrowserDependency/);
        assert.match(index, /class="empty-alt js-demo-disconnected"><button class="link-btn js-demo">or try the demo/);
    });

    it('keeps privacy discoverable and serves a real 404 page', () => {
        assert.match(index, /data-page="privacy"/);
        assert.match(index, /data-content-url="content\/privacy\.html(?:\?v=[0-9.]+)?"/);
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
    });

    it('treats top-level docs, roadmap, and privacy as pages, not articles', () => {
        const docsContent = read('docs/web/content/docs.html');
        const roadmapContent = read('docs/web/content/roadmap.html');
        const privacyContent = read('docs/web/content/privacy.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.ok(docsContent.startsWith(`${GPL_HTML_HEADER}<div class="docs-quickstart">`));
        assert.ok(roadmapContent.startsWith(`${GPL_HTML_HEADER}<div class="roadmap-page">`));
        assert.ok(privacyContent.startsWith(`${GPL_HTML_HEADER}<div class="privacy-page">`));
        assert.doesNotMatch(docsContent, /^<article\b/);
        assert.doesNotMatch(roadmapContent, /^<article\b/);
        assert.doesNotMatch(privacyContent, /^<article\b/);
        assert.match(index, /<main class="js-page page-narrow" data-page="roadmap"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="privacy"/);
        assert.doesNotMatch(index, /<main class="js-page page-narrow page-article" data-page="(?:docs|roadmap|privacy)"/);
        assert.match(staticPageBuilder, /"source": "content\/docs\.html",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/roadmap\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/privacy\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /<meta property="og:type" content="\{og_type\}">/);
    });

    it('uses the shared page heading styles on every top-level inner page', () => {
        for (const path of ['guides', 'docs', 'roadmap', 'privacy']) {
            const content = read(`docs/web/content/${path}.html`);
            assert.match(content, /<h1 class="page-title">/);
            assert.match(content, /<p class="page-sub">/);
        }
        assert.match(styles, /\.page-title \{ font-size: 40px;/);
        const pageSubRule = styles.match(/\.page-sub \{([^}]*)\}/)?.[1] || '';
        assert.match(pageSubRule, /font-size: 18px;/);
        assert.match(pageSubRule, /line-height: 1\.55;/);
        assert.match(styles, /@media \(max-width: 720px\) \{\s*\.page-title \{ font-size: 36px; \}\s*\.page-sub \{ font-size: 17px; \}/);
        assert.doesNotMatch(styles, /\.docs-hero \.page-title|\.docs-intro|\.roadmap-hero h1|\.roadmap-hero p/);
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

    it('adds anchor navigation and intrinsic image sizes to long guides', () => {
        const longPages = [
            'docs/web/content/docs.html',
            'docs/web/content/guides/setup.html',
            'docs/web/content/guides/firmware.html',
            'docs/web/content/guides/hardware.html',
            'docs/web/content/guides/placement.html',
            'docs/web/content/guides/detection.html',
            'docs/web/content/guides/detectors.html',
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
        const docsContent = read('docs/web/content/docs.html');
        assert.match(docsContent, /href="\/artifacts\/sdk\/stable\/"/);
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

    it('maps BLE capabilities, runtime controls, and dual-band Wi-Fi safely', () => {
        const device = index.match(/data-page="device"[\s\S]*?<\/main>/)?.[0] || '';
        const devicePanel = device.match(/<section class="panel" data-capability="supports_device_config">[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(index, /data-capability="supports_wifi_provisioning"/);
        assert.doesNotMatch(index, /js-threshold-slider|id="cfg-threshold"|js-threshold-save/);
        assert.doesNotMatch(index, /data-capability="supports_runtime_detector"|id="cfg-detector"/);
        assert.doesNotMatch(index, /js-detector-save|js-motion-save|js-runtime-recalibrate/);
        assert.match(index, /class="field-row wifi-credentials-row">\s*<div class="field"><label for="cfg-ssid"[\s\S]*?<label for="cfg-wifi-pass"/);
        assert.match(index, /class="field-row field-row-2-1">\s*<div class="field"><label for="cfg-wifi-band"[\s\S]*?<label for="cfg-channel"/);
        assert.match(index, /class="conn-dropdown-meta"/);
        assert.match(index, /js-menu-chip[\s\S]*js-menu-device-id[\s\S]*js-menu-firmware/);
        assert.match(index, /class="mono-sub js-device-banner-sub"/);
        assert.match(index, /class="conn-dropdown-name js-device-name"/);
        assert.match(index, /energy-title">CURRENT STATE/);
        assert.match(app, /conn\.motion \? 'MOTION' : 'IDLE'/);
        assert.doesNotMatch(app, /conn\.motion \? 'MOTION' : 'quiet'/);
        assert.match(styles, /\.conn-dropdown \{[^}]*color: var\(--text\);/);
        assert.match(styles, /\.conn-dropdown-name \{[^}]*color: var\(--text\);/);
        assert.match(device, /js-device-state-pill/);
        assert.match(app, /kind === 'live' \|\| kind === 'paused' \|\| kind === 'connecting'/);
        assert.match(app, /toast\('Connecting to the broker…'\)/);
        assert.match(app, /toast\('Sensing is live\.'\)/);
        assert.match(device, /data-device-view="live"/);
        assert.match(device, /data-device-view="connectivity"/);
        assert.doesNotMatch(device, /device-tabs|js-device-tab|role="tablist"|role="tabpanel"/);
        assert.doesNotMatch(device, /data-device-view="sensing"|data-device-view="diagnostics"/);
        assert.doesNotMatch(device, /Ready to sense|connectivity-finish|js-start-monitor/);
        const livePanel = device.match(/<section class="device-view js-device-view" data-device-view="live"[\s\S]*?<section class="device-view js-device-view" data-device-view="connectivity"/)?.[0] || '';
        assert.match(livePanel, /<h2>Sensing controls<\/h2>/);
        assert.match(livePanel, /stat-label">Movement score/);
        assert.doesNotMatch(livePanel, /Live status|mon-tiles|js-mon-thr|js-live-detector|js-mon-dev|js-live-session/);
        assert.match(livePanel, /<details class="device-live-diagnostics">/);
        assert.doesNotMatch(livePanel, /<details class="device-live-diagnostics" open/);
        assert.doesNotMatch(livePanel, /Apply threshold|Apply profile|Apply stability|Apply traffic settings/);
        const wifiPanel = index.match(/data-capability="supports_wifi_provisioning"[\s\S]*?<\/section>/)?.[0] || '';
        assert.doesNotMatch(wifiPanel, /js-ble-stop|js-start-detection|js-start-monitor/);
        const mqttPanel = index.match(/data-capability="supports_mqtt_config"[\s\S]*?<\/section>/)?.[0] || '';
        assert.doesNotMatch(mqttPanel, /js-start-detection|js-ble-stop|js-start-monitor/);
        assert.match(mqttPanel, /Native firmware uses this TCP broker/);
        assert.match(mqttPanel, /connects over WebSockets from Connect with MQTT/);
        assert.doesNotMatch(index, /Advanced browser connection|device-advanced-connectivity|js-browser-settings-slot/);
        assert.doesNotMatch(index, /class="stat-grid"|js-uptime|class="stat-value js-detector"/);
        assert.match(app, /function formatDeviceIdentityLine/);
        assert.match(app, /parts\.push\('Chip ' \+ chip\)/);
        assert.match(app, /parts\.push\('Device ID ' \+ deviceId\)/);
        assert.match(app, /parts\.push\('Firmware ' \+ firmware\)/);
        assert.match(app, /conn\.deviceBannerSub = deviceIdentity/);
        assert.match(app, /write\('\.js-menu-chip', conn\.chip\)/);
        assert.match(app, /write\('\.js-menu-device-id', conn\.deviceId\)/);
        assert.match(app, /write\('\.js-menu-firmware', conn\.firmwareVersion\)/);
        assert.doesNotMatch(app, /deviceBannerSub = \[chip, frontend\]/);
        assert.doesNotMatch(app, /deviceMenuSub/);
        assert.match(index, /id="cfg-wifi-band" disabled/);
        assert.doesNotMatch(index, /Band changes take effect after restarting the device|js-wifi-band-help/);
        assert.match(app, /snapshot\.supports_wifi_5ghz/);
        assert.match(app, /select\.disabled = select\.options\.length === 1/);
        assert.doesNotMatch(app, /buildThresholdCommand|buildDetectorCommand|cfgSaveDetector|cfgSaveMotionHits/);
        assert.doesNotMatch(styles, /threshold-slider|device-banner-meter|meter-threshold/);
        assert.match(app, /const showLiveEnergy = live/);
        assert.doesNotMatch(app, /js-setup-actions|js-start-monitor/);
        assert.doesNotMatch(app, /showDeviceHandoff|hideDeviceHandoff|js-device-handoff/);
        assert.doesNotMatch(index, /device-handoff-modal|Switching to live sensing/);
        assert.match(app, /function deviceBannerAction\(\)/);
        assert.match(app, /edit\.textContent = bleSetup \? 'Start sensing' : 'Edit connectivity'/);
        assert.match(app, /edit\.disabled = monitor\.closingBleForLive/);
        assert.match(app, /monitor\.closingBleForLive = false;[\s\S]*setStatus\('connected'\)/);
        assert.doesNotMatch(app, /Back to live/);
        assert.match(app, /await monitorConnect\(\)/);
        assert.match(app, /if \(route !== 'device'\) location\.hash = '#device'/);
        assert.doesNotMatch(app, /function cfgStopBle|js-ble-stop/);
        assert.match(app, /const bleSetup = connected && conn\.mode === 'ble'/);
        assert.match(app, /const bleConnecting = conn\.status === 'connecting'/);
        assert.match(app, /\(conn\.status === 'connecting' && !bleConnecting\)/);
        assert.match(app, /bleConnecting \? 'Connecting…'/);
        assert.match(app, /function validateMonitorConnection\(\)/);
        assert.match(app, /input\.setAttribute\('aria-invalid', 'true'\)/);
        assert.match(app, /monitorStatus\(''\)/);
        assert.doesNotMatch(app, /Set the broker host, topic prefix, and device id first\./);
        assert.match(styles, /\.field input\.is-invalid/);
        assert.match(styles, /@keyframes espFieldErrorBlink/);
        assert.match(app, /setDeviceSessionState\('paused'/);
        assert.doesNotMatch(app, /Sensing active|Setup mode/);
        assert.match(app, /bindMqttToConnection\(\)/);
        assert.match(app, /conn\.status !== 'connected'/);
        assert.match(app, /if \(!monitor\.handoffReady\) return/);
        assert.match(app, /monitor\.handoffReady = true/);
        assert.match(app, /getElementById\('sense-threshold'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-detector'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-on'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-motion-off'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-csi-mode'\)\.addEventListener\('change'/);
        assert.match(app, /getElementById\('sense-generator-mode'\)\.addEventListener\('change'/);
        assert.doesNotMatch(index, /js-cfg-start-ble/);
        assert.match(index, /Hold the <strong>BOOT<\/strong> button for 3 seconds/);
        const onboarding = index.match(/class="js-device-onboarding"[\s\S]*?<div class="js-device-workspace"/)?.[0] || '';
        assert.match(index, /class="js-device-onboarding"/);
        assert.match(index, /class="device-connect-card" data-transport="ble"[\s\S]*Connect with Bluetooth/);
        assert.match(index, /class="device-connect-card" data-transport="mqtt"[\s\S]*Connect with MQTT/);
        assert.ok(onboarding.indexOf('device-recovery-hint') < onboarding.indexOf('js-connect-ble'));
        assert.match(onboarding, /Which of the two buttons is BOOT\?/);
        assert.match(onboarding, /EN \/ RST[\s\S]*BOOT \/ FLASH/);
        assert.match(onboarding, /Choose the button labelled <strong>BOOT<\/strong> or <strong>FLASH<\/strong>/);
        const recoveryStart = index.indexOf('class="config-grid connectivity-recovery"');
        const recovery = recoveryStart >= 0 ? index.slice(recoveryStart, index.indexOf('</main>', recoveryStart)) : '';
        assert.match(recovery, /device-recovery-hint/);
        assert.match(recovery, /Which of the two buttons is BOOT\?/);
        assert.match(recovery, /EN \/ RST[\s\S]*BOOT \/ FLASH/);
        assert.doesNotMatch(recovery, /<details>|<summary>/);
        assert.match(app, /if \(startBle\) startBle\.addEventListener\('click', monitorStartBle\)/);
        assert.match(app, /command: 'set_ble', ble: 'on'/);
        assert.match(app, /monitor\.bleRequested = true;[\s\S]*await connectBle\(\)/);
        assert.doesNotMatch(app, /conn\.deviceName = .*bleClient\.name/);
        assert.match(app, /command: 'set_ble', ble: 'off'/);
        assert.match(app, /function ensureBleOffForLive/);
        assert.doesNotMatch(devicePanel, /js-dev-clear|Clear device/);
        assert.doesNotMatch(app, /cfgClearDevice|CLEAR_DEVICE_CONFIG/);
        assert.match(index, /class="modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /class="btn-primary js-ota-start" disabled>Update device<\/button>/);
        assert.match(index, /id="cfg-ota-message"/);
        assert.match(app, /function applyOtaStatus/);
        assert.match(app, /function startSilentOtaCheck/);
        assert.doesNotMatch(app, /function setFirmwareSummary|js-device-firmware/);
        assert.doesNotMatch(index, /id="diag-ota"/);
        assert.doesNotMatch(app, /set\('diag-ota'/);
        assert.doesNotMatch(index, /js-ota-status|Refresh OTA|Start OTA|js-device-ota-check|Check for update/);
        assert.match(index, /class="conn-firmware-update"[\s\S]*js-firmware-update-notice[\s\S]*js-disconnect/);
        const bannerActions = index.match(/class="device-banner-actions"[\s\S]*?<\/div>/)?.[0] || '';
        assert.match(bannerActions, /js-device-edit-connectivity/);
        assert.doesNotMatch(bannerActions, /js-firmware-update-notice/);
        assert.match(app, /copy = 'Checking for updates…'/);
        assert.match(app, /copy = 'Latest'/);
        assert.match(app, /status = 'error'/);
        assert.match(app, /message: 'Unable to check for updates'/);
        assert.doesNotMatch(index, /class="readonly-value js-device-firmware"|field-label">Firmware/);
        assert.match(app, /function otaOpen\(returnFocus\)/);
        assert.match(app, /button\.disabled = otaActionPending \|\| otaBusy \|\| !otaUpdateAvailable/);
        assert.doesNotMatch(styles, /\.panel-diagnostics/);
        assert.doesNotMatch(index, /diag-startup-threshold|diag-subcarriers/);
        assert.doesNotMatch(app, /snapshot\.startup_threshold|snapshot\.subcarriers/);
        assert.doesNotMatch(index, /diag-wifi-password|Wi-Fi password/);
        assert.doesNotMatch(app, /wifi_password_set/);
        assert.doesNotMatch(index, /id="diag-traffic-mode"|id="diag-traffic-rate"|id="diag-traffic"/);
        assert.doesNotMatch(app, /set\('diag-traffic-mode'|set\('diag-traffic-rate'|set\('diag-protocol'|set\('diag-firmware'/);
        assert.doesNotMatch(app, /snapshot\.publish_interval\b|every 100 pkts/);
        assert.doesNotMatch(app, /snapshot\.evaluation_interval\b|every 25 pkts/);
        assert.doesNotMatch(index, /runtime-hits-caption/);
        assert.match(index, /<h2 class="panel-title-status">Wi-Fi <span class="dot dot-idle js-wifi-status-dot"/);
        assert.match(index, /<h2 class="panel-title-status">MQTT <span class="dot dot-idle js-mqtt-status-dot"/);
        assert.doesNotMatch(index, /js-diag-(?:wifi|mqtt)-dot/);
        assert.match(app, /setConnectionDot\('\.js-wifi-status-dot', snapshot\.wifi_connected\)/);
        assert.match(app, /setConnectionDot\('\.js-mqtt-status-dot', snapshot\.mqtt_connected\)/);
        assert.match(styles, /\.dot-error \{ background: var\(--danger\); \}/);
        assert.match(app, /wifiBandPolicyAvailable \? \{ bandPolicy \}/);
        assert.doesNotMatch(app, /Wi-Fi needs both SSID and password/);
        assert.doesNotMatch(app, /MQTT needs host, port, username, and password/);
    });

    it('keeps MQTT diagnostics collapsed below Live and hides demo when live', () => {
        const device = index.match(/data-page="device"[\s\S]*?<\/main>/)?.[0] || '';
        const broker = device.match(/<section class="device-connect-card" data-transport="mqtt">[\s\S]*?<\/section>/)?.[0] || '';
        const diagnostics = device.match(/<details class="device-live-diagnostics">[\s\S]*?<\/details>/)?.[0] || '';
        assert.match(broker, /js-browser-broker-fields/);
        assert.match(broker, /js-mon-connect/);
        assert.match(broker, /class="tool-note js-mon-status" role="status" hidden><\/div>/);
        assert.doesNotMatch(broker, /Not connected\./);
        assert.doesNotMatch(broker, /js-mon-stats/);
        assert.doesNotMatch(diagnostics, /js-mon-stats|Refresh diagnostics/);
        assert.match(diagnostics, /js-mon-diag-status/);
        assert.match(diagnostics, /js-mon-admitted/);
        assert.match(diagnostics, /js-mon-filtered[\s\S]*js-mon-admitted/);
        assert.doesNotMatch(diagnostics, /js-mon-occupancy|js-mon-accepted|CSI occupancy|CSI accepted/);
        assert.doesNotMatch(diagnostics, /js-device-firmware|js-firmware-update|Check for update/);
        assert.doesNotMatch(diagnostics, /device-live-diagnostics-body|device-diagnostics-panel|On-demand diagnostics/);
        assert.match(diagnostics, /class="diag-grid mon-stats-tiles"/);
        assert.doesNotMatch(device, /Firmware, CSI pipeline/);
        assert.match(device, /CSI pipeline, Wi-Fi, and runtime health/);
        assert.doesNotMatch(broker, /js-mon-diag-status/);
        assert.match(app, /function syncDiagnosticsPolling/);
        assert.match(app, /setInterval\(monitorRequestStats, 1000\)/);
        assert.match(app, /function stopDiagnosticsPolling/);
        assert.match(app, /data\.csi_admitted_pps/);
        assert.match(app, /data\.csi_filtered_pps/);
        assert.doesNotMatch(app, /Number\(data\.csi_occupancy\) \* 100/);
        assert.match(app, /function monitorIsMqttLive/);
        assert.match(app, /connect\.disabled = mqttLive/);
        assert.match(app, /base \+ '\/#'/);
        assert.match(app, /function ingestMqttPayload/);
        assert.match(app, /function mqttUtf8/);
        assert.match(app, /function applyMqttLiveTelemetry/);
        assert.match(app, /suffix === 'commands\/catalog'/);
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
        assert.match(index, /id="mon-device"[^>]*placeholder="0x0000acebe64ae708"/);
        assert.match(index, /id="mon-path"[^>]*value="\/mqtt"/);
        assert.match(index, /id="mon-tls"/);
        const devicePage = index.match(/data-page="device"[\s\S]*?<\/main>/)?.[0] || '';
        const onboardingHtml = devicePage.match(/class="js-device-onboarding"[\s\S]*?<div class="js-device-workspace"/)?.[0] || '';
        const broker = devicePage.match(/<section class="device-connect-card" data-transport="mqtt">[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(onboardingHtml, /js-browser-broker-fields/);
        assert.match(broker, /id="mon-host"[\s\S]*id="mon-user"[\s\S]*id="mon-port"[\s\S]*id="mon-topic-prefix"[\s\S]*id="mon-device"[\s\S]*js-mon-connect/);
        assert.match(broker, /id="mon-path"/);
        assert.match(broker, /id="mon-tls"/);
        assert.doesNotMatch(index, /device-browser-connect|js-onboarding-browser-settings-slot|js-browser-ws-fields|js-browser-identity-fields/);
        assert.match(app, /function monitorBaseTopic/);
        assert.match(app, /function applyConfigureMqttDefaults/);
        assert.match(app, /host: 'homeassistant.local'/);
        assert.match(app, /topicPrefix: 'espectre\/v1\/devices'/);
        assert.match(app, /function applyConfigureMqttToMonitor/);
        assert.doesNotMatch(app, /browserConnectionUsesDeviceMqtt|js-browser-settings-slot|js-copy-browser-settings/);
        assert.match(app, /function startDetection/);
        assert.match(app, /bindMqttToConnection/);
        assert.match(app, /stopBleForDetection/);
        assert.match(app, /await ensureBleOffForLive/);
        assert.match(app, /await ensureBleOffForLive[\s\S]*setDeviceView\('live'\)/);
        assert.doesNotMatch(app, /setDeviceView\('live'\);\s*await monitorConnect/);
        assert.match(index, /Set up connectivity, watch live sensing, tune detection, and inspect diagnostics/);
        assert.match(index, /js-start-detection/);
        assert.doesNotMatch(index, /js-start-monitor/);
        assert.match(index, /js-connect-ble/);
        assert.match(index, /js-header-connect/);
        assert.match(index, /js-has-live/);
        assert.doesNotMatch(app, /getElementById\('mon-port'\)\.value = .*(?:cfg-mqtt-port|1883)/);
        assert.match(app, /input_mode: connectionInputMode\(\)/);
    });
});
