/*
 * ESPectre - Website accessibility contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { browserSupportSource, index, routeManifest, toolContent, toolFragments } from './fixtures/site_test_helpers.mjs';

describe('website accessibility and UX contracts', () => {
    it('exposes navigation controls and live status regions', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
        assert.match(index, /class="connection-callout js-connection-callout"[^>]+role="status"[^>]+hidden/);
    });

    it('provides skip navigation and one heading for every tool route', () => {
        assert.match(index, /<a class="skip-link" href="#main-content"/);
        assert.match(index, /data-page="home" id="main-content" tabindex="-1"/);
        for (const tool of ['flash', 'configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.ok(routeManifest.routes.some((route) => route.name === `tool-${tool}`));
            assert.match(index, new RegExp(`data-page="tool-${tool}"`));
            assert.match(toolContent[tool], /<h1 class="page-title">/);
        }
    });

    it('associates every form label with a control', () => {
        const labels = [...`${index}\n${toolFragments}`.matchAll(/<label\b([^>]*)>/g)];
        assert.ok(labels.length > 0);
        for (const [, attributes] of labels) assert.match(attributes, /\bfor="[^"]+"/);
    });

    it('keeps narrative scene relationships consistent', () => {
        const sceneIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-scene\b[^"]*" data-scene="(\d+)"/g)]
            .map((match) => Number(match[1]));
        const captionIds = [...index.matchAll(/class="[^"]*\bjs-scrolly-caption\b[^"]*" data-scene="(\d+)"/g)]
            .map((match) => Number(match[1]));
        const markerIds = [...index.matchAll(/class="js-scrolly-marker" data-scene="(\d+)"/g)]
            .map((match) => Number(match[1]));
        assert.ok(sceneIds.length > 0);
        assert.deepEqual(sceneIds, Array.from({ length: sceneIds.length }, (_, index) => index));
        assert.deepEqual(captionIds, sceneIds);
        assert.deepEqual(markerIds, sceneIds.slice(1));
        assert.match(index, /class="js-scrolly-current"/);
        assert.match(index, /data-scene="1" aria-hidden="true" inert/);
    });

    it('enforces the supported browser capability matrix', async () => {
        const detect = (navigator) => {
            const context = { window: { navigator } };
            runInNewContext(browserSupportSource, context);
            return context.window.ESPectreBrowserSupport.current;
        };
        const serial = { requestPort() {} };
        assert.equal(detect({ userAgent: 'Chrome', platform: 'Linux x86_64', serial }).flash, true);
        assert.equal(detect({ userAgent: 'Chrome Android Mobile', platform: 'Linux armv8', serial }).flash, false);
        assert.equal(detect({ userAgent: 'CriOS iPhone Mobile', platform: 'iPhone', serial }).flash, false);
        assert.equal(detect({
            userAgent: 'Mozilla/5.0 Chrome/147.0.0.0 Safari/537.36',
            platform: 'Linux x86_64',
        }).hostedDirect, 'unclaimed');
        assert.equal(detect({
            userAgent: 'Mozilla/5.0 Chrome/151.0.0.0 Safari/537.36',
            platform: 'MacIntel',
        }).hostedDirect, 'targeted');
        assert.equal(detect({
            userAgent: 'Mozilla/5.0 Firefox/148.0',
            platform: 'Linux x86_64',
        }).hostedDirect, 'unsupported');

        const permissions = {
            query: async ({ name }) => ({ state: name === 'local-network' ? 'denied' : 'prompt' }),
        };
        const context = { window: { navigator: { permissions } } };
        runInNewContext(browserSupportSource, context);
        assert.equal(
            await context.window.ESPectreBrowserSupport.localNetworkAccessState(context.window.navigator),
            'denied'
        );
        assert.equal(
            await context.window.ESPectreBrowserSupport.localNetworkAccessState({}),
            'unavailable'
        );
    });
});
