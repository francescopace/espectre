/*
 * ESPectre - Website security contracts
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

describe('website security, asset, and analytics contracts', () => {
    it('embeds the generated SDK reference inside the shared website shell', () => {
        const doxyfile = read('src/cpp/Doxyfile');
        const apiContent = read('docs/web/content/sdk/api.html');
        assert.match(doxyfile, /^GENERATE_HTML\s*=\s*NO$/m);
        assert.match(doxyfile, /^GENERATE_XML\s*=\s*YES$/m);
        assert.match(apiContent, /data-api-reference-browser/);
        assert.match(apiContent, /data-api-index="\/artifacts\/sdk\/api\/api-index\.json"/);
        assert.match(apiContent, /data-api-reference-content/);
        assert.match(apiContent, /data-api-reference-picker/);
        assert.match(apiContent, /data-api-reference-filter/);
        assert.match(apiContent, /data-api-reference-results/);
        assert.doesNotMatch(apiContent, /data-api-reference-toggle/);
        assert.match(apiContent, /data-page-toc/);
        assert.match(apiContent, /data-page-path="sdk"/);
        assert.doesNotMatch(apiContent, /<iframe/);
        assert.doesNotMatch(apiContent, /api-reference-index|api-reference-layout|api-reference-browser-head/);
        assert.match(styles, /\.api-reference-controls \{[\s\S]*?display: flex;[\s\S]*?justify-content: flex-end;/);
        assert.match(styles, /\.api-reference-picker \{ position: relative; flex: 0 1 440px;/);
        assert.match(styles, /\.api-reference-picker-field input \{/);
        assert.doesNotMatch(styles, /\.api-reference-search/);
        assert.match(styles, /\.api-map \{[\s\S]*?display: grid;/);
        assert.doesNotMatch(styles, /nav\.m-block/);
        assert.equal((styles.match(/\{/g) || []).length, (styles.match(/\}/g) || []).length);
        const navigation = read('docs/web/assets/js/navigation.js');
        assert.match(navigation, /function initApiReferenceBrowsers/);
        assert.match(navigation, /function renderApiReferencePicker/);
        assert.match(navigation, /function setApiReferencePickerOpen/);
        assert.match(navigation, /candidate\.discoverable === false/);
        assert.match(navigation, /function refreshApiReferenceToc/);
        assert.match(navigation, /\/artifacts\/sdk\/api\/\$\{entry\.fragment\}/);
        assert.match(navigation, /browser\.apiReferenceOverview = content\.innerHTML/);
        assert.match(app, /window\.initApiReferenceBrowsers\(container\)/);
    });

    it('renders the brand face in white instead of exposing the background', () => {
        const logo = read('docs/web/assets/images/brand/espectre-logo.svg');
        const generatedShell = read('.github/scripts/web_page_shell.py');
        assert.doesNotMatch(logo, /<mask\b|mask="url\(/);
        assert.match(logo, /<use href="#ghost" fill="#4b7bee" stroke="#b8c8ff" stroke-width="2\.8" stroke-linejoin="round"\/>/);
        assert.doesNotMatch(logo, /<use[^>]+stroke="#fff"/);
        assert.equal((logo.match(/<ellipse[^>]+fill="#fff"/g) || []).length, 2);
        assert.match(logo, /<path[^>]+stroke="#fff"/);
        assert.match(logo, /<circle[^>]+fill="#fff"/);
        assert.match(index, /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=[0-9a-f]{12}" alt="" width="30" height="30"/);
        assert.match(read('docs/web/404.html'), /class="brand"[^>]*>\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg\?v=[0-9a-f]{12}" alt="" width="30" height="30"/);
        assert.match(generatedShell, /class=\"brand\"[^>]*>\s*<img src=\"\/assets\/images\/brand\/espectre-logo\.svg\?v=\{logo_version\}\" alt=\"\" width=\"30\" height=\"30\"/);
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
                '/assets/js/route-bootstrap.js',
                '/assets/js/espectre-direct.js',
                '/assets/js/browser-support.js',
                '/assets/js/route-registry.js',
                '/assets/js/navigation.js',
                '/assets/js/analytics.js',
                '/assets/js/device-session.js',
                '/assets/js/direct-discovery.js',
                '/assets/js/configure-tool.js',
                '/assets/js/monitor-tool.js',
                '/assets/js/app.js'
            ]
        );
        const deferredToolScripts = [...index.matchAll(/data-script-src="(\/assets\/js\/[^"?]+)(?:\?v=[0-9a-f]{12})?"/g)]
            .map((match) => match[1]);
        assert.deepEqual(new Set(deferredToolScripts), new Set([
            '/assets/js/csi-tool.js',
            '/assets/js/game-tool.js',
            '/assets/js/theremin-tool.js',
        ]));
        assert.doesNotMatch(firstPartyScripts[0], /\bdefer\b/);
        for (const attrs of firstPartyScripts.slice(1)) {
            assert.match(attrs, /\bdefer\b/, `expected defer on ${attrs.trim()}`);
        }
        assert.ok(index.indexOf('/assets/js/app.js') < index.indexOf('</head>'));
        assert.match(app, /\/vendor\/esp-web-tools-10\.4\.0\/install-button\.js/);
        assert.match(app, /\/vendor\/qrcodejs-1\.0\.0\/qrcode\.min\.js/);
        assert.match(app, /sitePolicy\.isLoopbackHostname\(location\.hostname\)/);
    });

    it('keeps first-party cache-busting hashes in lockstep with file contents', () => {
        const stamper = read('.github/scripts/web_asset_versions.py');
        const stampCommand = 'python3 .github/scripts/web_asset_versions.py';
        assert.match(stamper, /HASH_LENGTH = 12/);
        assert.match(stamper, /python3 \.github\/scripts\/web_asset_versions\.py/);
        const hashLength = 12;
        const assetVersion = (relativePath) => createHash('sha256')
            .update(readFileSync(new URL(`../../docs/web/${relativePath}`, import.meta.url)))
            .digest('hex')
            .slice(0, hashLength);
        const mismatches = [];
        const collectStamped = (html, label) => {
            const refs = [...html.matchAll(
                /(?:href|src|data-script-src)="((?:\/assets\/(?:css|js)\/|\/assets\/images\/brand\/espectre-logo\.svg)[^"]*)"/g
            )];
            assert.ok(refs.length > 0, `${label} references first-party assets`);
            for (const [, url] of refs) {
                const [assetPath, query = ''] = url.split('?');
                const version = new URLSearchParams(query).get('v');
                const relativePath = assetPath.replace(/^\//, '');
                if (version !== assetVersion(relativePath)) {
                    mismatches.push(`${label} ${assetPath}`);
                }
            }
        };
        collectStamped(index, 'index.html');
        collectStamped(read('docs/web/404.html'), '404.html');
        assert.ok(
            mismatches.length === 0,
            `${mismatches.join(', ')}; run ${stampCommand}`
        );
        assert.match(stamper, /--check-current/);
    });
    it('separates connection, readiness, verified outcomes, and disconnects', () => {
        assert.match(app, /track\('tool_ready'/);
        assert.match(app, /readiness,/);
        assert.match(app, /latency_ms:/);
        assert.match(app, /if \(!conn\.mode \|\| conn\.status !== 'connected'\) return;/);
        assert.match(app, /if \(conn\.toolName === 'configure'\) markToolReady\('info'\);/);
        assert.match(app, /markToolReady\('telemetry'\)/);
        assert.match(app, /markToolReady\('raw_stream'\)/);
        assert.match(app, /track\('configure_change', \{ action, result: 'accepted' \}\)/);
        assert.match(app, /finishConfigVerification\('success'\)/);
        assert.match(app, /CONFIG_VERIFICATION_RETRY_MS = 1500/);
        assert.match(app, /CONFIG_VERIFICATION_MAX_ATTEMPTS = 4/);
        assert.match(app, /track\('ota_update_result'/);
        assert.match(app, /track\('ota_update_attempt'/);
        assert.match(app, /track\('sensing_change'/);
        assert.match(app, /track\('raw_csi_stream'/);
        assert.match(app, /state === 'reboot_scheduled'/);
        assert.match(app, /state === 'error'/);
        assert.match(app, /entry_point: monitor\.entryPoint/);
        assert.match(app, /\.\.\.connectionParams\(\)/);
    });

    it('tracks abandonment and keeps SDK download metadata explicit', () => {
        const gameAbandonEvent = app.match(/track\('game_abandon', \{[\s\S]*?\n        \}\);/)?.[0] || '';
        const gameOverEvent = app.match(/track\('game_over', \{[\s\S]*?\n        \}\);/)?.[0] || '';
        assert.match(gameAbandonEvent, /score: game\.score/);
        assert.match(gameAbandonEvent, /distance: Math\.floor\(game\.distance\)/);
        assert.match(gameAbandonEvent, /reason/);
        assert.match(gameOverEvent, /score: game\.score/);
        assert.match(gameOverEvent, /orbs: game\.orbs/);
        assert.match(gameOverEvent, /distance: Math\.floor\(game\.distance\)/);
        assert.match(app, /reportGameAbandon\('restart'\)/);
        assert.match(app, /reportGameAbandon\('route_change'\)/);
        assert.match(app, /reportGameAbandon\('page_exit'\)/);
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

    it('keeps peer discovery failures distinct, accessible, cancellable, and privacy bounded', () => {
        const discoveryFailureHandler = app.match(
            /function directDiscoveryFailureMessage[\s\S]*?async function queryLocalPeers/
        )?.[0] || '';
        for (const code of [
            'local_network_denied',
            'unsupported_crypto',
            'unsupported_capability',
            'timeout',
            'invalid_envelope',
            'unsupported_version',
            'invalid_peer_result',
            'frame_too_large',
            'connection_failed'
        ]) {
            assert.match(discoveryFailureHandler, new RegExp(`['"]${code}['"]`));
        }
        assert.match(index, /class="direct-discovery js-direct-discovery" role="status" aria-live="polite"/);
        assert.match(app, /function cancelDirectDiscovery[\s\S]*?directDiscoveryClient = null;[\s\S]*?client\?\.close\(\)/);
        assert.match(app, /function discoveredPeerChipLabel[\s\S]*?replace\(\/\^ESP32/);
        assert.match(app, /heading\.className = 'direct-discovery-device-heading'/);
        assert.match(app, /metadata\.className = 'direct-discovery-device-meta'/);
        assert.match(app, /displayDeviceId: shortId/);
        assert.match(styles, /\.direct-discovery-device-meta \{[\s\S]*?grid-template-columns: repeat\(3, minmax\(0, 1fr\)\)/);
        const discoveryEvents = [...app.matchAll(/track\('local_discovery',[\s\S]*?\}\);/g)]
            .map((match) => match[0]).join('\n');
        assert.match(discoveryEvents, /device_count/);
        assert.doesNotMatch(
            discoveryEvents,
            /device_id|address|hostname|endpoint|firmware|capabilities|payload/
        );
    });

    it('reports every registered SPA route, including Raw CSI', () => {
        assert.match(app, /if \(window\.trackRouteView\) window\.trackRouteView\(route\);/);
        assert.match(app, /window\.trackRouteView\(route, \{ sendPageView: false \}\)/);
    });
});
