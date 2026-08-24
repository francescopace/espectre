/*
 * ESPectre - Browser MQTT protocol client unit tests
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

globalThis.window = globalThis.window ?? {};
createRequire(import.meta.url)('../../docs/web/assets/js/espectre-mqtt.js');

const Client = window.ESPectreMqttClient;
const MqttError = window.ESPectreMqttError;

function fakeTransport() {
    return {
        publishes: [],
        publish(topic, payload, options, callback) {
            this.publishes.push({ topic, payload: JSON.parse(payload), options });
            callback?.(null);
        }
    };
}

describe('MQTT topic and discovery contract', () => {
    it('constructs canonical device and discovery topics', () => {
        assert.equal(
            Client.baseTopic('espectre/v1/devices/', '0x1234'),
            'espectre/v1/devices/0x1234'
        );
        assert.deepEqual(Client.discoveryTopics('espectre/v1/devices'), [
            'espectre/v1/devices/+/info',
            'espectre/v1/devices/+/status'
        ]);
        assert.throws(() => Client.baseTopic('devices/#', 'abc'), MqttError);
        assert.throws(() => Client.baseTopic('devices', 'a/b'), MqttError);
    });

    it('parses only info and status discovery messages below the requested prefix', () => {
        const info = Client.parseDiscoveryMessage(
            'espectre/v1/devices',
            'espectre/v1/devices/0x1234/info',
            Buffer.from('{"protocol_version":"1.0","device_id":"0x1234"}')
        );
        assert.equal(info.deviceId, '0x1234');
        assert.equal(info.suffix, 'info');
        assert.equal(info.data.device_id, '0x1234');
        assert.equal(Client.parseDiscoveryMessage(
            'espectre/v1/devices',
            'espectre/v1/devices/0x1234/telemetry',
            '{}'
        ), null);
    });
});

describe('MQTT command lifecycle', () => {
    it('publishes protocol 1.0 commands and correlates accepted responses', async () => {
        const transport = fakeTransport();
        const client = new Client(transport, {
            topicPrefix: 'espectre/v1/devices',
            deviceId: '0x1234'
        });
        const resultPromise = client.publishCommand(
            { command: 'set_threshold', threshold: 0.42 },
            { commandId: 'web-test' }
        );
        assert.deepEqual(transport.publishes[0], {
            topic: 'espectre/v1/devices/0x1234/commands/request',
            payload: {
                protocol_version: '1.0',
                command_id: 'web-test',
                command: 'set_threshold',
                threshold: 0.42
            },
            options: { qos: 0, retain: false }
        });
        assert.equal(client.hasPendingCommand('set_threshold'), true);
        assert.equal(client.ingest(
            'espectre/v1/devices/0x1234/commands/result',
            '{"protocol_version":"1.0","device_id":"0x1234","command_id":"web-test","command":"set_threshold","accepted":true,"code":"ok","message":"threshold updated"}'
        ), true);
        assert.equal((await resultPromise).accepted, true);
        assert.equal(client.hasPendingCommand('set_threshold'), false);
    });

    it('rejects device rejections and all pending work when closed', async () => {
        const transport = fakeTransport();
        const client = new Client(transport, { deviceId: 'device-a' });
        const rejected = client.publishCommand(
            { command: 'set_detector', detector: 'unknown' },
            { commandId: 'reject-me' }
        );
        client.ingest(
            'espectre/v1/devices/device-a/commands/result',
            '{"protocol_version":"1.0","device_id":"device-a","command_id":"reject-me","command":"set_detector","accepted":false,"code":"unsupported","message":"unsupported detector"}'
        );
        await assert.rejects(rejected, /unsupported detector/);

        const closed = client.publishCommand({ command: 'diagnostics' }, { commandId: 'close-me' });
        client.close();
        await assert.rejects(closed, /Broker connection closed/);
    });

    it('cleans up pending state when the transport throws synchronously', async () => {
        const transport = {
            publish() { throw new Error('transport failed'); }
        };
        const client = new Client(transport, { deviceId: 'device-a' });
        await assert.rejects(
            client.publishCommand({ command: 'diagnostics' }, { commandId: 'sync-failure' }),
            /transport failed/
        );
        assert.equal(client.hasPendingCommand('diagnostics'), false);
    });
});

describe('MQTT payload validation and events', () => {
    it('delivers canonical JSON and additive HA scalar messages through one API', () => {
        const client = new Client(fakeTransport(), { deviceId: 'device-a' });
        const messages = [];
        client.on('message', (message) => messages.push(message));
        assert.equal(client.ingest(
            'espectre/v1/devices/device-a/telemetry',
            '{"protocol_version":"1.0","device_id":"device-a","movement_score":0.2,"threshold":0.4}'
        ), true);
        assert.equal(client.ingest(
            'espectre/v1/devices/device-a/ha/movement/state',
            Buffer.from('0.25')
        ), true);
        assert.equal(messages[0].data.movement_score, 0.2);
        assert.equal(messages[1].data, null);
        assert.equal(messages[1].text, '0.25');
    });

    it('rejects malformed JSON, wrong protocol versions, and other devices', () => {
        const client = new Client(fakeTransport(), { deviceId: 'device-a' });
        const errors = [];
        client.on('protocol-error', (error) => errors.push(error));
        assert.equal(client.ingest('espectre/v1/devices/device-a/info', '{nope'), false);
        assert.equal(client.ingest(
            'espectre/v1/devices/device-a/status',
            '{"protocol_version":"2.0","online":true}'
        ), false);
        assert.equal(client.ingest(
            'espectre/v1/devices/device-a/status',
            '{"protocol_version":"1.0","device_id":"device-b","online":true}'
        ), false);
        assert.equal(client.ingest(
            'espectre/v1/devices/device-b/status',
            '{"protocol_version":"1.0","device_id":"device-b","online":true}'
        ), false);
        assert.equal(errors.length, 3);
    });

    it('isolates throwing event listeners', () => {
        const client = new Client(fakeTransport(), { deviceId: 'device-a' });
        let delivered = false;
        const originalError = console.error;
        console.error = () => {};
        try {
            client.on('message', () => { throw new Error('listener failed'); });
            client.on('message', () => { delivered = true; });
            client.ingest(
                'espectre/v1/devices/device-a/status',
                '{"protocol_version":"1.0","device_id":"device-a","online":true}'
            );
        } finally {
            console.error = originalError;
        }
        assert.equal(delivered, true);
    });
});
