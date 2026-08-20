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

    it('builds the stop-Bluetooth command', () => {
        assert.equal(Client.buildStopBleCommand(), 'STOP_BLE');
    });

    it('builds OTA commands with an optional channel', () => {
        assert.equal(Client.buildOtaStatusCommand(), 'OTA_STATUS');
        assert.equal(Client.buildOtaCheckCommand(), 'OTA_CHECK');
        assert.equal(Client.buildOtaStartCommand(), 'OTA_START');
        assert.equal(Client.buildOtaCheckCommand({ channel: 'preview' }), 'OTA_CHECK:channel=preview');
        assert.equal(Client.buildOtaStartCommand({ channel: 'develop' }), 'OTA_START:channel=develop');
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

    it('rejects an unknown OTA channel', () => {
        assertValidationError(() => Client.buildOtaCheckCommand({ channel: 'latest' }), /channel must be one of/);
        assertValidationError(() => Client.buildOtaStartCommand({ channel: 'stable' }), /channel must be one of/);
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
        assertValidationError(() => Client.buildDeviceLabelCommand('a\rb'), /label/);
    });

    it('matches firmware byte limits and the 512-byte control queue', () => {
        assertValidationError(
            () => Client.buildWifiConfigCommand({ ssid: 'é'.repeat(17) }),
            /1\.\.32 UTF-8 bytes/
        );
        assertValidationError(
            () => Client.buildWifiConfigCommand({ ssid: 'ok', password: 'é'.repeat(32) }),
            /0\.\.63 UTF-8 bytes/
        );
        assertValidationError(
            () => Client.buildMqttConfigCommand({
                host: 'broker',
                port: 1883,
                password: 'x'.repeat(480)
            }),
            /512 UTF-8 bytes/
        );
        assertValidationError(
            () => Client.buildMqttConfigCommand({ host: 'broker\0hidden', port: 1883 }),
            /NUL/
        );
    });
});

describe('GATT lifecycle', () => {
    function deferred() {
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        return { promise, resolve };
    }

    function fixture({ firstWrite } = {}) {
        const deviceListeners = new Map();
        const sysinfoListeners = new Map();
        const writes = [];
        const sysinfo = {
            async startNotifications() {},
            async stopNotifications() {},
            addEventListener(name, handler) { sysinfoListeners.set(name, handler); },
            removeEventListener(name) { sysinfoListeners.delete(name); },
            emitLine(line) {
                const bytes = new TextEncoder().encode(line);
                const value = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
                sysinfoListeners.get('characteristicvaluechanged')?.({ target: { value } });
            }
        };
        const control = {
            async writeValueWithResponse(value) {
                writes.push(new TextDecoder().decode(value));
                if (writes.length === 1 && firstWrite) await firstWrite.promise;
            }
        };
        const service = {
            async getCharacteristic(uuid) {
                return uuid === Client.UUIDS.sysinfo ? sysinfo : control;
            }
        };
        const server = {
            connected: false,
            async getPrimaryService() { return service; },
            disconnect() { this.connected = false; }
        };
        const device = {
            id: 'device-a',
            name: 'ESPectre BLE',
            gatt: {
                async connect() {
                    server.connected = true;
                    return server;
                }
            },
            addEventListener(name, handler) { deviceListeners.set(name, handler); },
            removeEventListener(name) { deviceListeners.delete(name); }
        };
        return {
            control,
            device,
            server,
            sysinfo,
            writes,
            emitDisconnect() {
                server.connected = false;
                deviceListeners.get('gattserverdisconnected')?.();
            }
        };
    }

    function useBluetooth(requestDevice) {
        Object.defineProperty(globalThis, 'navigator', {
            configurable: true,
            value: { bluetooth: { requestDevice } }
        });
    }

    it('serializes control writes so commands cannot overtake each other', async () => {
        const firstWrite = deferred();
        const gatt = fixture({ firstWrite });
        useBluetooth(async () => gatt.device);
        const client = new Client();
        await client.connect({ sysinfo: false });
        const first = client.writeControl('OTA_STATUS');
        const second = client.writeControl('REQ_SYSINFO');
        await new Promise((resolve) => setImmediate(resolve));
        assert.deepEqual(gatt.writes, ['OTA_STATUS']);
        firstWrite.resolve();
        await Promise.all([first, second]);
        assert.deepEqual(gatt.writes, ['OTA_STATUS', 'REQ_SYSINFO']);
        await client.disconnect();
    });

    it('ignores incomplete sysinfo frames and emits a complete ordered snapshot', async () => {
        const gatt = fixture();
        useBluetooth(async () => gatt.device);
        const client = new Client();
        const snapshots = [];
        client.on('sysinfo', (values, entries) => snapshots.push({ values, entries }));
        await client.connect({ sysinfo: false });
        gatt.sysinfo.emitLine('orphan=value');
        gatt.sysinfo.emitLine('END');
        gatt.sysinfo.emitLine('proto_version=1');
        gatt.sysinfo.emitLine('device_id=0x1234');
        gatt.sysinfo.emitLine('END');
        assert.equal(snapshots.length, 1);
        assert.equal(snapshots[0].values.device_id, '0x1234');
        assert.deepEqual(snapshots[0].entries, [
            ['proto_version', '1'],
            ['device_id', '0x1234']
        ]);
        await client.disconnect();
    });

    it('invalidates queued writes and emits once after an unexpected disconnect', async () => {
        const firstWrite = deferred();
        const gatt = fixture({ firstWrite });
        useBluetooth(async () => gatt.device);
        const client = new Client();
        let disconnects = 0;
        client.on('disconnect', () => { disconnects += 1; });
        await client.connect({ sysinfo: false });
        const first = client.writeControl('OTA_STATUS');
        const queued = client.writeControl('REQ_SYSINFO');
        await new Promise((resolve) => setImmediate(resolve));
        gatt.emitDisconnect();
        firstWrite.resolve();
        await first;
        await assert.rejects(queued, (error) => error.name === 'AbortError');
        assert.deepEqual(gatt.writes, ['OTA_STATUS']);
        assert.equal(client.connected, false);
        assert.equal(disconnects, 1);
    });

    it('cancels a device chooser result after an explicit disconnect', async () => {
        const chooser = deferred();
        const gatt = fixture();
        let connectCalls = 0;
        gatt.device.gatt.connect = async () => {
            connectCalls += 1;
            return gatt.server;
        };
        useBluetooth(() => chooser.promise);
        const client = new Client();
        const connecting = client.connect({ sysinfo: false });
        await client.disconnect();
        chooser.resolve(gatt.device);
        await assert.rejects(connecting, (error) => error.name === 'AbortError');
        assert.equal(connectCalls, 0);
        assert.equal(client.connected, false);
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
        assert.deepEqual([...Client.EVENTS], ['sysinfo', 'sysinfo-line', 'disconnect']);
    });

    it('starts disconnected with empty read-only state', () => {
        const client = new Client();
        assert.equal(client.connected, false);
        assert.equal(client.name, '');
        assert.equal(client.device, null);
    });
});
