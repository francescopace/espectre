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
        assert.match(app, /monitorStopAll\('route_change'\)/);
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
        for (const heading of ['Configure device', 'Motion theremin', 'Motion reaction game']) {
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
        assert.match(index, /data-content-url="content\/privacy\.html"/);
        assert.match(index, /<div class="footer-links">\s*<a href="#privacy">Privacy<\/a>/);
        assert.match(routeRegistry, /name: 'privacy'.*staticPath: '\/privacy\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/privacy\/">Privacy<\/a>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/privacy\/">Privacy<\/a>/);
        const sitemap = read('docs/web/sitemap.xml');
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
        assert.match(read('docs/web/sitemap.xml'), /https:\/\/espectre\.dev\/guides\/detectors\//);
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
            assert.match(read(path), /<nav class="page-toc" aria-label="On this page">/);
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
        assert.match(index, /data-capability="supports_wifi_provisioning"/);
        assert.match(index, /class="threshold-slider js-threshold-slider"[^>]+type="range"/);
        assert.doesNotMatch(index, /id="cfg-threshold"|js-threshold-save/);
        assert.match(index, /data-capability="supports_runtime_detector"/);
        assert.doesNotMatch(index, /js-detector-save|js-motion-save/);
        assert.match(index, /class="field-row wifi-credentials-row">\s*<div class="field"><label for="cfg-ssid"[\s\S]*?<label for="cfg-wifi-pass"/);
        assert.match(index, /class="field-row field-row-2-1">\s*<div class="field"><label for="cfg-wifi-band"[\s\S]*?<label for="cfg-channel"/);
        assert.match(index, /class="mono-sub js-device-menu-sub"/);
        assert.match(index, /class="mono-sub js-device-banner-sub"/);
        const deviceBanner = index.match(/<div class="device-banner">[\s\S]*?<div class="device-banner-meter">/)?.[0] || '';
        assert.doesNotMatch(deviceBanner, /class="dot /);
        assert.doesNotMatch(index, /class="stat-grid"|js-uptime|class="stat-value js-detector"/);
        assert.match(app, /deviceBannerSub = \[chip, frontend\]/);
        assert.match(app, /deviceMenuSub = \[chip, frontend, firmware\]/);
        assert.match(index, /id="cfg-wifi-band" disabled/);
        assert.doesNotMatch(index, /Band changes take effect after restarting the device|js-wifi-band-help/);
        assert.match(app, /snapshot\.supports_wifi_5ghz/);
        assert.match(app, /select\.disabled = select\.options\.length === 1/);
        assert.match(app, /buildThresholdCommand/);
        assert.match(app, /thresholdSlider\.addEventListener\('input'/);
        assert.match(app, /thresholdSlider\.addEventListener\('change'/);
        assert.match(app, /getElementById\('cfg-detector'\)\.addEventListener\('change', cfgSaveDetector\)/);
        assert.match(app, /getElementById\(id\)\.addEventListener\('change', cfgSaveMotionHits\)/);
        assert.match(app, /buildDetectorCommand/);
        const runtimePanel = index.match(/<section class="panel">\s*<h2>Runtime<\/h2>[\s\S]*?<\/section>/)?.[0] || '';
        const devicePanel = index.match(/<section class="panel" data-capability-any="supports_device_config supports_ota">[\s\S]*?<\/section>/)?.[0] || '';
        assert.doesNotMatch(runtimePanel, /supports_ota|cfg-ota|js-ota/);
        assert.match(devicePanel, /class="btn-ghost btn-sm js-ota-check"[^>]*>Check update<\/button>/);
        assert.doesNotMatch(devicePanel, /js-dev-clear|Clear device/);
        assert.doesNotMatch(app, /cfgClearDevice|CLEAR_DEVICE_CONFIG/);
        assert.match(index, /class="modal-card" role="dialog" aria-modal="true"/);
        assert.match(index, /class="btn-primary js-ota-start" disabled>Update device<\/button>/);
        assert.match(index, /id="cfg-ota-message"/);
        assert.match(app, /set\('cfg-ota-message', snapshot\.ota_message/);
        assert.doesNotMatch(index, /id="diag-ota"/);
        assert.doesNotMatch(app, /set\('diag-ota'/);
        assert.doesNotMatch(index, /js-ota-status|Refresh OTA|Start OTA/);
        assert.match(app, /function otaOpen\(returnFocus\)/);
        assert.match(app, /button\.disabled = otaActionPending \|\| otaBusy \|\| !otaUpdateAvailable/);
        assert.match(styles, /\.panel-diagnostics \{ grid-column: span 2; \}/);
        assert.doesNotMatch(index, /diag-startup-threshold|diag-subcarriers/);
        assert.doesNotMatch(app, /snapshot\.startup_threshold|snapshot\.subcarriers/);
        assert.doesNotMatch(index, /diag-wifi-password|Wi-Fi password/);
        assert.doesNotMatch(app, /wifi_password_set/);
        assert.match(index, /id="diag-traffic-mode"/);
        assert.match(index, /id="diag-traffic-rate"/);
        assert.doesNotMatch(index, /id="diag-traffic"/);
        assert.match(app, /set\('diag-traffic-mode', snapshot\.traffic_mode\)/);
        assert.match(app, /set\('diag-traffic-rate'/);
        assert.match(app, /snapshot\.publish_interval_ms && 'every ' \+ snapshot\.publish_interval_ms \+ ' ms'/);
        assert.doesNotMatch(app, /snapshot\.publish_interval\b|every 100 pkts/);
        assert.match(app, /snapshot\.evaluation_interval_ms && 'every ' \+ snapshot\.evaluation_interval_ms \+ ' ms'/);
        assert.doesNotMatch(app, /snapshot\.evaluation_interval\b|every 25 pkts/);
        assert.match(index, /class="tool-note runtime-hits-caption"[^>]*>Consecutive evaluations above or below the threshold required to enter or leave the motion state\.<\/p>/);
        assert.match(index, /<h2 class="panel-title-status">Wi-Fi <span class="dot dot-idle js-wifi-status-dot"/);
        assert.match(index, /<h2 class="panel-title-status">MQTT <span class="dot dot-idle js-mqtt-status-dot"/);
        assert.doesNotMatch(index, /js-diag-(?:wifi|mqtt)-dot/);
        assert.match(app, /setConnectionDiagnostic\('diag-wifi', '\.js-wifi-status-dot', snapshot\.wifi_connected\)/);
        assert.match(app, /setConnectionDiagnostic\('diag-mqtt', '\.js-mqtt-status-dot', snapshot\.mqtt_connected\)/);
        assert.match(styles, /\.dot-error \{ background: var\(--danger\); \}/);
        assert.match(app, /wifiBandPolicyAvailable \? \{ bandPolicy \}/);
        assert.doesNotMatch(app, /Wi-Fi needs both SSID and password/);
        assert.doesNotMatch(app, /MQTT needs host, port, username, and password/);
    });
});
