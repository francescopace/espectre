/*
 * ESPectre - Website accessibility contracts
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
import { app, browserSupportSource, directProtocol, GPL_HTML_HEADER, index, read, roadmapContent, routeRegistry, security, styles, toolContent, toolFragments, toolsContent } from './fixtures/site_test_helpers.mjs';

describe('website accessibility and UX contracts', () => {
    it('has a responsive navigation control and a live status region', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(styles, /@media \(max-width: 840px\) \{[\s\S]*?\.conn \{ margin-left: auto; min-width: 0; order: 2; \}/);
        assert.match(styles, /\.conn-connected \.js-device-name \{ min-width: 0; overflow: hidden; text-overflow: ellipsis; \}/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
        assert.doesNotMatch(index, /js-demo-toast|toast-sticky/);
        assert.match(index, /class="connection-callout js-connection-callout"[^>]+role="status"[^>]+hidden/);
        assert.match(app, /const DIRECT_CALLOUT_DURATION_MS = 4000;/);
        assert.match(app, /function setStatus\(status\) \{[\s\S]*?enteringDirectConnection[\s\S]*?showDirectConnectionCallout\(\)/);
        assert.match(app, /function showDirectConnectionCallout\(\) \{[\s\S]*?setTimeout\([\s\S]*?directCalloutVisible = false;[\s\S]*?syncConnectionCallout\(\);[\s\S]*?DIRECT_CALLOUT_DURATION_MS/);
        assert.match(app, /function syncConnectionCallout\(\) \{[\s\S]*?const demo =[\s\S]*?const direct =[\s\S]*?dropdownOpen/);
        assert.match(styles, /\.connection-callout \{[\s\S]*?position: absolute;[\s\S]*?pointer-events: none;/);
        assert.match(styles, /\.connection-callout::before \{[\s\S]*?transform: rotate\(45deg\);/);
    });

    it('provides skip navigation, tool page titles, and route focus management', () => {
        assert.match(index, /<a class="skip-link" href="#main-content"/);
        assert.match(index, /data-page="home" id="main-content" tabindex="-1"/);
        for (const tool of ['flash', 'configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.match(index, new RegExp(`data-page="tool-${tool}"[\\s\\S]*?content/tools/${tool}\\.html`));
            assert.match(toolContent[tool], /<h1 class="page-title">/);
        }
        assert.match(app, /link\.setAttribute\('aria-current', 'page'\)/);
        assert.match(app, /target\.focus\(\{ preventScroll: true \}\)/);
        assert.match(app, /page\.id = 'main-content'/);
        assert.match(app, /const routeAtStart = route;/);
        assert.match(app, /if \(!ready \|\| route !== routeAtStart\) return;[\s\S]*?focusRouteContent\(routeAtStart\);/);
        assert.match(styles, /h1\[tabindex="-1"\]:focus \{ outline: none; \}/);
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
        const labels = [...`${index}\n${toolFragments}`.matchAll(/<label\b([^>]*)>/g)];
        assert.ok(labels.length > 10);
        for (const [, attributes] of labels) assert.match(attributes, /\bfor="[^"]+"/);
    });
    it('keeps shared button typography and action hierarchy consistent', () => {
        assert.match(
            styles,
            /:is\(\.btn-primary, \.btn-secondary, \.btn-ghost, \.btn-danger, \.btn-danger-ghost\) \{[\s\S]*?font: inherit;[\s\S]*?font-weight: 600;/
        );
        assert.match(styles, /\.btn-secondary \{ border-color: var\(--border\); background: var\(--surface\); color: var\(--dim\); \}/);
        assert.match(styles, /\.btn-ghost \{ border-color: transparent; background: transparent; color: var\(--dim\); \}/);
        assert.match(styles, /\.btn-secondary:hover,\s*\.btn-ghost:hover \{ color: var\(--text\); \}/);
        assert.match(styles, /\.btn-ghost:hover \{ background: var\(--surface2\); \}/);
        assert.match(styles, /\.btn-primary:hover,\s*\.hero-cta-primary:hover \{ filter: brightness\(1\.08\); \}/);
        assert.match(toolContent.monitor, /class="btn-primary btn-sm js-device-edit-connectivity"/);
        assert.match(toolContent.monitor, /class="btn-secondary btn-sm js-sense-recalibrate"/);
        assert.match(toolContent.configure, /class="wifi-bssid-control">[\s\S]*?<select id="cfg-bssid"[\s\S]*?<button type="button" class="wifi-refresh-button js-wifi-scan" aria-label="[^"]+"[^>]*><svg[^>]*aria-hidden="true"/);
        assert.doesNotMatch(toolContent.configure, /class="panel-actions wifi-panel-actions">[\s\S]*?js-wifi-scan/);
        assert.match(styles, /\.wifi-bssid-control \{[\s\S]*?display: flex;[\s\S]*?gap: 8px;[\s\S]*?\.wifi-refresh-button \{[\s\S]*?flex: 0 0 30px;/);
        assert.match(styles, /\.wifi-refresh-button\.is-scanning svg \{[\s\S]*?animation: espSpin \.7s linear infinite;/);
        assert.match(app, /scanButton\.classList\.toggle\('is-scanning', scanning\);\s*scanButton\.setAttribute\('aria-busy', String\(scanning\)\);/);
        assert.match(styles, /\.wifi-panel-actions \{ flex-wrap: nowrap; \}/);
        const contactContent = read('docs/web/content/contact.html');
        assert.equal((contactContent.match(/class="btn-primary"/g) || []).length, 1);
        assert.equal((contactContent.match(/class="btn-secondary"/g) || []).length, 2);

        for (const path of [
            'docs/web/index.html',
            'docs/web/404.html',
            '.github/scripts/build_static_pages.py',
            '.github/scripts/stage_web_sdk.py',
        ]) {
            const source = read(path);
            assert.match(source, /class="btn-secondary btn-sm js-consent-reject"/);
            assert.match(source, /class="btn-secondary btn-sm js-consent-accept"/);
        }
    });

    it('formats security guidance with project-native components', () => {
        assert.match(security, /class="security-guidelines">\s*<ul>/);
        assert.match(styles, /\.security-guidelines ul \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
        assert.match(styles, /\.security-page \.note \{[^}]*background: var\(--accent-soft\);[^}]*border: 1px solid var\(--accent-line\);/);
        assert.match(styles, /\.security-page \.docs-start-copy p \+ p \{ margin-top: 12px; \}/);
        assert.match(styles, /\.security-page \.docs-path > :is\(\.btn-primary, \.btn-secondary\) \{[^}]*margin-top: auto;/);
        assert.match(security, /class="docs-paths security-reporting-paths">/);
        assert.match(styles, /\.security-reporting-paths \{ grid-template-columns: repeat\(2, minmax\(0, 1fr\)\); \}/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.security-guidelines ul \{ grid-template-columns: 1fr; \}/);
    });

    it('uses natural scrolling and progressively loads narrative images', () => {
        assert.match(index, /data-src-mobile="\/assets\/images\/home\/scene-smart-heating-mobile\.webp"/);
        assert.match(index, /data-src="\/assets\/images\/home\/scene-embedded-sdk\.webp" data-src-mobile="\/assets\/images\/home\/scene-embedded-sdk-mobile\.webp"/);
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
        assert.match(index, /href="#get-started" class="scrolly-skip"/);
        assert.match(index, /<section class="home-action-hub" id="get-started" aria-labelledby="home-action-title">/);
        const actionHub = index.match(/<section class="home-action-hub"[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(actionHub, /<header class="home-action-head">\s*<h2 class="page-title" id="home-action-title">/);
        assert.match(actionHub, /class="home-action-group">\s*<span class="home-kicker">[\s\S]*?<div class="home-tool-grid"/);
        assert.match(actionHub, /href="\/tools\/flash\/" class="home-tool-card home-tool-card-primary"/);
        assert.match(actionHub, /href="\/tools\/configure\/" class="home-tool-card"/);
        assert.match(actionHub, /href="\/tools\/monitor\/" class="home-tool-card"/);
        assert.match(actionHub, /class="home-action-group">[\s\S]*?<aside class="home-license-cta"/);
        assert.match(actionHub, /class="home-action-group">[\s\S]*?<div class="home-resource-strip"/);
        const licenseCard = actionHub.match(/<aside class="home-license-cta"[\s\S]*?<\/aside>/)?.[0] || '';
        assert.match(actionHub, /class="home-license-cta"[\s\S]*?class="home-resource-strip"/);
        const resourceHrefs = [...actionHub.matchAll(/class="home-resource-links">[\s\S]*?<\/div>/g)][0]?.[0]
            .match(/href="([^"]+)"/g)?.map((entry) => entry.slice(6, -1)) || [];
        assert.deepEqual(resourceHrefs, [
            '/tools/',
            '/guides/',
            '/media/',
            '/sdk/',
            '/roadmap/',
            'https://github.com/francescopace/espectre',
        ]);
        assert.match(styles, /\.home-resource-links \{ display: grid; grid-template-columns: repeat\(6, minmax\(0, 1fr\)\); \}/);
        assert.match(styles, /\.home-license-cta \{[\s\S]*?background: var\(--surface\);/);
        assert.match(actionHub, /href="\/licensing\/" class="btn-primary"/);
        assert.match(styles, /\.home-action-hub \{[^}]*min-height: 100svh/);
        assert.match(styles, /\.home-action-inner \{[^}]*min-height: 100svh;[^}]*padding: calc\(var\(--header-height\) \+ 64px\) 40px 40px;[^}]*justify-content: center;/);
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
        const chrome147Linux = detect({
            userAgent: 'Mozilla/5.0 Chrome/147.0.0.0 Safari/537.36', platform: 'Linux x86_64'
        });
        assert.equal(chrome147Linux.hostedDirect, 'unclaimed');
        assert.equal(chrome147Linux.browser, 'chrome');
        assert.equal(chrome147Linux.operatingSystem, 'linux');
        const chrome151Mac = detect({
            userAgent: 'Mozilla/5.0 Chrome/151.0.0.0 Safari/537.36', platform: 'MacIntel'
        });
        assert.equal(chrome151Mac.hostedDirect, 'targeted');
        assert.equal(chrome151Mac.operatingSystem, 'macos');
        const chrome151Windows = detect({
            userAgent: 'Mozilla/5.0 Chrome/151.0.0.0 Safari/537.36', platform: 'Win32'
        });
        assert.equal(chrome151Windows.hostedDirect, 'targeted');
        assert.equal(chrome151Windows.operatingSystem, 'windows');
        const chrome151Linux = detect({
            userAgent: 'Mozilla/5.0 Chrome/151.0.0.0 Safari/537.36', platform: 'Linux x86_64'
        });
        assert.equal(chrome151Linux.hostedDirect, 'targeted');
        assert.equal(chrome151Linux.operatingSystem, 'linux');
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
                    permissions: { query: async ({ name }) => ({ state: name === 'local-network' ? 'denied' : 'prompt' }) }
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
        const legacyPermissionContext = {
            window: {
                navigator: {
                    permissions: {
                        query: async ({ name }) => {
                            if (name === 'local-network') throw new TypeError('unsupported permission');
                            return { state: 'granted' };
                        }
                    }
                }
            }
        };
        runInNewContext(browserSupportSource, legacyPermissionContext);
        assert.equal(
            await legacyPermissionContext.window.ESPectreBrowserSupport.localNetworkAccessState(
                legacyPermissionContext.window.navigator
            ),
            'granted'
        );
        assert.equal(
            await permissionContext.window.ESPectreBrowserSupport.localNetworkAccessState({}),
            'unavailable'
        );
        assert.match(app, /installTrigger\.disabled = !browserSupport\.flash/);
        assert.match(app, /button\.disabled = directConnecting/);
        assert.match(app, /if \(browserSupport\.flash\) \{\s*loadBrowserDependency/);
        assert.match(styles, /min-height: 100dvh/);
        assert.match(styles, /touch-action: manipulation/);
        const mobileCss = styles.split('@media (max-width: 720px)')[1]
            .split('@media (max-width: 480px)')[0];
        assert.match(mobileCss, /\.field input:not\(\[type="checkbox"\]\):not\(\[type="range"\]\)/);
        assert.match(mobileCss, /min-height: 44px/);
        assert.match(mobileCss, /font-size: 16px/);
    });

});
