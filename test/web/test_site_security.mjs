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
import { app, index, read } from './fixtures/site_test_helpers.mjs';

describe('website security and asset contracts', () => {
    it('embeds the generated SDK reference without active HTML', () => {
        const doxyfile = read('src/cpp/Doxyfile');
        const api = read('docs/web/content/sdk/api.html');
        assert.match(doxyfile, /^GENERATE_HTML\s*=\s*NO$/m);
        assert.match(doxyfile, /^GENERATE_XML\s*=\s*YES$/m);
        assert.match(api, /data-api-reference-browser/);
        assert.match(api, /data-api-index="\/artifacts\/sdk\/api\/api-index\.json"/);
        assert.match(api, /data-api-reference-content/);
        assert.doesNotMatch(api, /<iframe/);

        const navigation = read('docs/web/assets/js/navigation.js');
        assert.match(navigation, /content\.replaceChildren\(parsePassiveApiReferenceFragment\(fragment\)\)/);
        assert.doesNotMatch(navigation, /content\.innerHTML = fragment/);
    });

    it('does not load third-party scripts before analytics consent', () => {
        const externalScripts = [...index.matchAll(/<script[^>]+src="(https?:[^"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(externalScripts, []);
        assert.doesNotMatch(index, /unpkg\.com|jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com/);
        const firstPartyScripts = [...index.matchAll(/<script\b([^>]*)>/g)]
            .map((match) => match[1])
            .filter((attributes) => /src="\/assets\/js\//.test(attributes));
        assert.ok(firstPartyScripts.length > 0);
        for (const attributes of firstPartyScripts) {
            assert.match(attributes, /src="\/assets\/js\//);
        }
    });

    it('keeps first-party asset hashes aligned with file contents', () => {
        const mismatches = [];
        const collect = (html, label) => {
            const references = [...html.matchAll(
                /(?:href|src|data-script-src)="((?:\/assets\/(?:css|js)\/|\/assets\/images\/brand\/espectre-logo\.svg)[^"]*)"/g
            )];
            assert.ok(references.length > 0, `${label} references first-party assets`);
            for (const [, url] of references) {
                const [assetPath, query = ''] = url.split('?');
                const version = new URLSearchParams(query).get('v');
                if (!version) {
                    mismatches.push(`${label} ${assetPath}`);
                    continue;
                }
                const digest = createHash('sha256')
                    .update(readFileSync(new URL(`../../docs/web/${assetPath.slice(1)}`, import.meta.url)))
                    .digest('hex')
                    .slice(0, version.length);
                if (version !== digest) mismatches.push(`${label} ${assetPath}`);
            }
        };
        collect(index, 'index.html');
        collect(read('docs/web/404.html'), '404.html');
        assert.deepEqual(mismatches, []);
    });

    it('keeps sensitive browser-tool values out of analytics calls', () => {
        const analyticsCalls = [...app.matchAll(/track\([^;]+\);/gs)].map((match) => match[0]).join('\n');
        assert.doesNotMatch(
            analyticsCalls,
            /ssid|bssid|password|mqtt_host|mqtt_username|topic_prefix|device_id|payload/
        );
    });

    it('keeps peer discovery reporting accessible and privacy bounded', () => {
        assert.match(index, /class="direct-discovery js-direct-discovery" role="status" aria-live="polite"/);
        const discoveryEvents = [...app.matchAll(/track\('local_discovery',[\s\S]*?\}\);/g)]
            .map((match) => match[0])
            .join('\n');
        assert.match(discoveryEvents, /device_count/);
        assert.doesNotMatch(
            discoveryEvents,
            /device_id|address|hostname|endpoint|firmware|capabilities|payload/
        );
    });
});
