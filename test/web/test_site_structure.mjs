/*
 * ESPectre - Website structural contract tests
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (path) => readFileSync(new URL(`../../${path}`, import.meta.url), 'utf8');
const index = read('docs/web/index.html');
const app = read('docs/web/assets/js/app.js');

describe('website security and asset policy', () => {
    it('does not execute third-party scripts before an explicit analytics choice', () => {
        const externalScripts = [...index.matchAll(/<script[^>]+src="(https?:[^\"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(externalScripts, []);
        assert.doesNotMatch(index, /unpkg\.com|jsdelivr\.net/);
        assert.match(index, /\/assets\/css\/styles\.css/);
        assert.match(index, /\/assets\/js\/app\.js/);
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

describe('website accessibility and navigation', () => {
    it('has a responsive navigation control and a live status region', () => {
        assert.match(index, /class="nav-toggle"[^>]+aria-controls="main-navigation"/);
        assert.match(index, /id="main-navigation"/);
        assert.match(index, /class="toast js-toast"[^>]+role="status"[^>]+aria-live="polite"/);
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
        assert.match(app, /image\.dataset\.srcMobile/);
    });

    it('keeps privacy discoverable and serves a real 404 page', () => {
        assert.match(index, /href="\/privacy\/"/);
        assert.match(read('docs/web/sitemap.xml'), /https:\/\/espectre\.dev\/privacy\//);
        assert.match(read('docs/web/content/privacy.html'), /Never included:/);
        const notFound = read('docs/web/404.html');
        assert.doesNotMatch(notFound, /http-equiv="refresh"|location\.replace/);
        assert.match(notFound, /404 · PAGE NOT FOUND/);
    });

    it('loads generated firmware and SDK output from the shared artifacts tree', () => {
        assert.match(app, /\/artifacts\/firmware\//);
        assert.doesNotMatch(app, /\/flash\/firmware\//);
        const docsContent = read('docs/web/content/docs.html');
        assert.match(docsContent, /href="\/artifacts\/sdk\/stable\/"/);
        assert.match(docsContent, /href="\/artifacts\/sdk\/api\/"/);
        assert.doesNotMatch(docsContent, /href="\/sdk\//);
        assert.match(read('docs/web/.gitignore'), /^\/artifacts\/$/m);
    });

    it('maps BLE capabilities, runtime controls, and dual-band Wi-Fi safely', () => {
        assert.match(index, /data-capability="supports_wifi_provisioning"/);
        assert.match(index, /data-capability="supports_runtime_threshold"/);
        assert.match(index, /data-capability="supports_runtime_detector"/);
        assert.match(index, /id="cfg-wifi-band"/);
        assert.match(app, /snapshot\.supports_wifi_5ghz/);
        assert.match(app, /buildThresholdCommand/);
        assert.match(app, /buildDetectorCommand/);
        assert.match(app, /wifiBandPolicyAvailable \? \{ bandPolicy \}/);
        assert.doesNotMatch(app, /Wi-Fi needs both SSID and password/);
        assert.doesNotMatch(app, /MQTT needs host, port, username, and password/);
    });
});
