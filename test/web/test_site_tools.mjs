/*
 * ESPectre - Website tool contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { index, read, routeManifest, toolContent } from './fixtures/site_test_helpers.mjs';

describe('website tool contracts', () => {
    it('publishes the firmware and SDK artifact channels', () => {
        const sdk = read('docs/web/content/sdk.html');
        for (const { sdkChannel: channel, path } of routeManifest.sdkChannels) {
            assert.match(toolContent.flash, new RegExp(`<option value="${channel}"`));
            assert.match(
                sdk,
                new RegExp(`href="${path}"[\\s\\S]*?data-sdk-version="${channel}"`)
            );
        }
        assert.match(
            read('docs/web/content/sdk/api.html'),
            /data-api-index="\/artifacts\/sdk\/api\/api-index\.json"/
        );
    });

    it('offers the shared connection picker from every connected browser tool', () => {
        for (const tool of ['configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.match(
                toolContent[tool],
                new RegExp(`<espectre-connection-picker[^>]*data-surface="${tool}"`)
            );
        }
    });

    it('keeps MQTT configuration values in Device settings', () => {
        const configure = toolContent.configure;
        assert.match(configure, /id="cfg-mqtt-scheme"/);
        assert.match(configure, /id="cfg-mqtt-host"/);
        assert.match(configure, /id="cfg-mqtt-port"/);
        assert.match(configure, /id="cfg-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(configure, /id="cfg-mqtt-credentials-clear"/);
        assert.doesNotMatch(configure, /id="cfg-mqtt-user"[^>]*value=/);
        assert.doesNotMatch(configure, /id="cfg-mqtt-pass"[^>]*value=/);
    });

    it('publishes one Raw CSI visualization selector with stable option values', () => {
        const rawCsi = toolContent['raw-csi'];
        const values = [...rawCsi.matchAll(/<option value="([^"]+)"/g)].map((match) => match[1]);
        assert.deepEqual(values, [
            'subcarrier-amplitudes',
            'csi-amplitude-surface',
            'channel-profile-deviation',
            'iq-constellation',
            'relative-phase-trails',
        ]);
        assert.equal((rawCsi.match(/class="js-raw-visualization"/g) || []).length, 1);
    });

    it('keeps unavailable Relay controls inert', () => {
        const template = index.match(/<template id="connection-picker-template">[\s\S]*?<\/template>/)?.[0] || '';
        const relay = template.match(/data-connection-panel="relay"[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(relay, /class="btn-primary" disabled/);
        assert.doesNotMatch(relay, /<input|<select|js-connect/);
    });
});
