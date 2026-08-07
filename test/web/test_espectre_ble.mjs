/*
 * ESPectre - Web Bluetooth client unit tests
 *
 * Covers the hardware-independent surface of docs/web/assets/js/espectre-ble.js: the
 * pure command builders and their validation, the telemetry parser, and the
 * event API. The GATT paths need a physical device and are exercised through
 * the website's Configure tool instead.
 *
 * Run with: node --test 'test/web/*.mjs'
 *
 * Apache-2.0 like the library itself, so client and tests can travel
 * together into other projects.
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
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

    it('formats the threshold with six decimals', () => {
        assert.equal(Client.buildThresholdCommand(0.35), 'SET_THRESHOLD:0.350000');
        assert.equal(Client.buildThresholdCommand(0), 'SET_THRESHOLD:0.000000');
        assert.equal(Client.buildThresholdCommand(1), 'SET_THRESHOLD:1.000000');
    });

    it('builds detector commands', () => {
        assert.equal(Client.buildDetectorCommand('classic'), 'SET_DETECTOR:classic');
        assert.equal(Client.buildDetectorCommand('ml'), 'SET_DETECTOR:ml');
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

    it('rejects thresholds outside 0..1', () => {
        assertValidationError(() => Client.buildThresholdCommand(1.2), /threshold/);
        assertValidationError(() => Client.buildThresholdCommand(-0.1), /threshold/);
        assertValidationError(() => Client.buildThresholdCommand(NaN), /threshold/);
    });

    it('rejects unknown detectors', () => {
        assertValidationError(() => Client.buildDetectorCommand('quantum'), /detector/);
    });

    it('rejects multi-line device labels', () => {
        assertValidationError(() => Client.buildDeviceLabelCommand('a\nb'), /label/);
    });
});

describe('parseTelemetry', () => {
    const view = (bytes) => new DataView(new Uint8Array(bytes).buffer);
    const f32 = (value) => {
        const buffer = new ArrayBuffer(4);
        new DataView(buffer).setFloat32(0, value, true);
        return [...new Uint8Array(buffer)];
    };

    it('parses movement, threshold, and motion state', () => {
        const telemetry = Client.parseTelemetry(view([...f32(0.25), ...f32(0.5), 1]));
        assert.deepEqual(telemetry, { movement: 0.25, threshold: 0.5, motionState: 1 });
    });

    it('reports a null motion state on 8-byte payloads', () => {
        const telemetry = Client.parseTelemetry(view([...f32(0.1), ...f32(0.2)]));
        assert.equal(telemetry.motionState, null);
    });

    it('returns null for short payloads', () => {
        assert.equal(Client.parseTelemetry(view([1, 2, 3])), null);
        assert.equal(Client.parseTelemetry(null), null);
    });

    it('returns null for non-finite values', () => {
        const nan = [0, 0, 192, 127]; // little-endian float32 NaN
        assert.equal(Client.parseTelemetry(view([...nan, ...f32(0.5)])), null);
        assert.equal(Client.parseTelemetry(view([...f32(0.5), ...nan])), null);
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
        assertValidationError(() => client.on('telemetry', 42), /handler/);
    });

    it('exposes frozen constants and a version', () => {
        assert.ok(Object.isFrozen(Client.EVENTS));
        assert.ok(Object.isFrozen(Client.UUIDS));
        assert.match(Client.VERSION, /^\d+\.\d+\.\d+$/);
        assert.equal(Client.UUIDS.service, 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0');
    });

    it('starts disconnected with empty read-only state', () => {
        const client = new Client();
        assert.equal(client.connected, false);
        assert.equal(client.name, '');
        assert.equal(client.device, null);
    });
});
