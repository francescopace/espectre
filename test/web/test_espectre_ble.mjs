/*
 * ESPectre - Web Bluetooth client unit tests
 *
 * Covers the hardware-independent surface of docs/web/assets/js/espectre-ble.js: the
 * pure command builders and their validation, and the event API. The GATT paths
 * need a physical device and are exercised through the website's Configure tool
 * instead.
 *
 * Run with: node --test 'test/web/*.mjs'
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// The library is a classic script that exports onto `window`.
globalThis.window = globalThis.window ?? {};
createRequire(import.meta.url)('../../docs/web/assets/js/espectre-ble.js');

const Client = window.ESPectreBleClient;
const ValidationError = window.ESPectreValidationError;

function assertValidationError(fn, messagePart) {
    assert.throws(fn, (error) => {
        assert.equal(error.name, 'ESPectreValidationError');
        assert.ok(error instanceof ValidationError);
        if (messagePart) assert.match(error.message, messagePart);
        return true;
    });
}

describe('command builders: wire format', () => {
    it('encodes a full Wi-Fi config', () => {
        assert.equal(
            Client.buildWifiConfigCommand({
                ssid: 'Lab Network',
                password: 'secret-password',
                bssid: 'aa:bb:cc:dd:ee:ff',
                channel: 6,
                bandPolicy: '2g'
            }),
            'SET_WIFI_CONFIG:ssid=Lab%20Network&password=secret-password'
            + '&bssid=aa%3Abb%3Acc%3Add%3Aee%3Aff&channel=6&band_policy=2g'
        );
    });

    it('encodes Wi-Fi defaults with every key present', () => {
        assert.equal(
            Client.buildWifiConfigCommand({ ssid: 'Net', password: 'p' }),
            'SET_WIFI_CONFIG:ssid=Net&password=p&bssid=&channel=0'
        );
    });

    it('encodes a full MQTT config with the default topic prefix', () => {
        assert.equal(
            Client.buildMqttConfigCommand({
                host: '192.168.1.20',
                port: 1883,
                username: 'mqtt',
                password: 'secret-password'
            }),
            'SET_MQTT_CONFIG:host=192.168.1.20&port=1883&username=mqtt'
            + '&password=secret-password&topic_prefix=espectre%2Fv1%2Fdevices'
        );
    });

    it('allows anonymous MQTT brokers', () => {
        assert.equal(
            Client.buildMqttConfigCommand({ host: 'h', port: 1 }),
            'SET_MQTT_CONFIG:host=h&port=1&username=&password='
            + '&topic_prefix=espectre%2Fv1%2Fdevices'
        );
    });

    it('leaves the device label unencoded, matching the firmware parser', () => {
        assert.equal(
            Client.buildDeviceLabelCommand('Living Room'),
            'SET_DEVICE_CONFIG:device_label=Living Room'
        );
    });

    it('accepts an empty device label', () => {
        assert.equal(
            Client.buildDeviceLabelCommand(''),
            'SET_DEVICE_CONFIG:device_label='
        );
    });

    it('builds the stop-BLE command', () => {
        assert.equal(Client.buildStopBleCommand(), 'STOP_BLE');
    });
});

describe('command builders: validation', () => {
    it('rejects a missing SSID', () => {
        assertValidationError(() => Client.buildWifiConfigCommand({ password: 'x' }), /ssid/);
    });

    it('rejects out-of-range and non-integer channels', () => {
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: 15 }), /channel/);
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: -1 }), /channel/);
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: NaN }), /channel/);
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: 1.5 }), /channel/);
        // Between the 20 MHz centers of the 5 GHz band plan.
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: 37 }), /channel/);
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: 150 }), /channel/);
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', channel: 181 }), /channel/);
    });

    it('accepts 5 GHz channel locks and leaves radio capability to the device', () => {
        assert.equal(
            Client.buildWifiConfigCommand({ ssid: 'Net', channel: 36 }),
            'SET_WIFI_CONFIG:ssid=Net&password=&bssid=&channel=36');
        assert.equal(
            Client.buildWifiConfigCommand({ ssid: 'Net', channel: 149 }),
            'SET_WIFI_CONFIG:ssid=Net&password=&bssid=&channel=149');
    });

    it('encodes and validates an explicit Wi-Fi band policy', () => {
        assert.equal(
            Client.buildWifiConfigCommand({ ssid: 'Net', channel: 36, bandPolicy: '5g' }),
            'SET_WIFI_CONFIG:ssid=Net&password=&bssid=&channel=36&band_policy=5g');
        assert.equal(
            Client.buildWifiConfigCommand({ ssid: 'Net', channel: 0, bandPolicy: 'auto' }),
            'SET_WIFI_CONFIG:ssid=Net&password=&bssid=&channel=0&band_policy=auto');
        assertValidationError(
            () => Client.buildWifiConfigCommand({ ssid: 'Net', channel: 36, bandPolicy: '2g' }),
            /does not match/);
        assertValidationError(
            () => Client.buildWifiConfigCommand({ ssid: 'Net', channel: 6, bandPolicy: '5g' }),
            /does not match/);
        assertValidationError(
            () => Client.buildWifiConfigCommand({ ssid: 'Net', bandPolicy: '6g' }),
            /bandPolicy/);
    });

    it('rejects a malformed BSSID', () => {
        assertValidationError(() => Client.buildWifiConfigCommand({ ssid: 'a', bssid: 'nope' }), /bssid/);
    });

    it('rejects a missing host and invalid ports', () => {
        assertValidationError(() => Client.buildMqttConfigCommand({ port: 1883 }), /host/);
        assertValidationError(() => Client.buildMqttConfigCommand({ host: 'h', port: 0 }), /port/);
        assertValidationError(() => Client.buildMqttConfigCommand({ host: 'h', port: 65536 }), /port/);
        assertValidationError(() => Client.buildMqttConfigCommand({ host: 'h', port: 18.83 }), /port/);
    });

    it('rejects an empty topic prefix', () => {
        assertValidationError(
            () => Client.buildMqttConfigCommand({ host: 'h', port: 1, topicPrefix: '' }),
            /topicPrefix/
        );
    });

    it('rejects multi-line device labels', () => {
        assertValidationError(() => Client.buildDeviceLabelCommand('a\nb'), /label/);
    });
});

describe('event API', () => {
    it('subscribes, unsubscribes, and ignores unknown pairs', () => {
        const client = new Client();
        const seen = [];
        const handler = (line) => seen.push(line);
        const unsubscribe = client.on('sysinfo-line', handler);
        assert.equal(typeof unsubscribe, 'function');
        unsubscribe();
        client.off('sysinfo-line', handler); // Already removed: must not throw.
    });

    it('rejects unknown events and non-function handlers', () => {
        const client = new Client();
        assertValidationError(() => client.on('nope', () => {}), /unknown event/);
        assertValidationError(() => client.on('sysinfo', 42), /handler/);
    });

    it('exposes frozen constants and a version', () => {
        assert.ok(Object.isFrozen(Client.EVENTS));
        assert.ok(Object.isFrozen(Client.UUIDS));
        assert.match(Client.VERSION, /^\d+\.\d+\.\d+$/);
        assert.equal(Client.UUIDS.service, 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0');
        assert.ok(Client.EVENTS.includes('sysinfo'));
        assert.ok(!Client.EVENTS.includes('telemetry'));
        assert.equal(typeof Client.buildThresholdCommand, 'undefined');
        assert.equal(typeof Client.parseTelemetry, 'undefined');
    });

    it('starts disconnected with empty read-only state', () => {
        const client = new Client();
        assert.equal(client.connected, false);
        assert.equal(client.name, '');
        assert.equal(client.device, null);
    });
});
