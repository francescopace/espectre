/*
 * ESPectre - Browser Direct WebSocket client unit tests
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

globalThis.window = globalThis.window ?? {};
createRequire(import.meta.url)('../../docs/web/assets/js/espectre-direct.js');

const Client = window.ESPectreDirectClient;
const DirectError = window.ESPectreDirectError;

class FakeWebSocket {
    static instances = [];
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url, protocols) {
        this.url = url;
        this.protocols = protocols;
        this.protocol = '';
        this.readyState = FakeWebSocket.CONNECTING;
        this.listeners = new Map();
        this.sent = [];
        FakeWebSocket.instances.push(this);
    }

    addEventListener(name, handler) {
        if (!this.listeners.has(name)) this.listeners.set(name, []);
        this.listeners.get(name).push(handler);
    }

    emit(name, fields = {}) {
        for (const handler of this.listeners.get(name) || []) handler({ target: this, ...fields });
    }

    open(protocol = Client.SUBPROTOCOL) {
        this.protocol = protocol;
        this.readyState = FakeWebSocket.OPEN;
        this.emit('open');
    }

    send(payload) { this.sent.push(JSON.parse(payload)); }

    receive(payload) {
        this.emit('message', { data: typeof payload === 'string' ? payload : JSON.stringify(payload) });
    }

    close(code = 1000, reason = '') {
        this.readyState = FakeWebSocket.CLOSED;
        this.emit('close', { code, reason });
    }
}

afterEach(() => {
    delete globalThis.WebSocket;
    FakeWebSocket.instances.length = 0;
});

describe('Direct endpoint policy', () => {
    it('normalizes private IP, local hostname, and HTTP input', () => {
        assert.equal(Client.normalizeEndpoint('192.168.1.42'), 'ws://192.168.1.42/espectre/v1/ws');
        assert.equal(Client.normalizeEndpoint('espectre-a1.local'), 'ws://espectre-a1.local/espectre/v1/ws');
        assert.equal(
            Client.normalizeEndpoint('https://espectre-a1.local/espectre/v1/ws'),
            'wss://espectre-a1.local/espectre/v1/ws'
        );
    });

    it('rejects public addresses, credentials, queries, and unrelated paths', () => {
        assert.throws(() => Client.normalizeEndpoint('8.8.8.8'), DirectError);
        assert.throws(() => Client.normalizeEndpoint('ws://user:pass@192.168.1.2'), DirectError);
        assert.throws(() => Client.normalizeEndpoint('ws://192.168.1.2/?token=x'), DirectError);
        assert.throws(() => Client.normalizeEndpoint('ws://192.168.1.2/admin'), DirectError);
    });
});

describe('Direct request lifecycle', () => {
    it('classifies connection timeout and protocol negotiation failures', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const timeoutClient = new Client('192.168.1.42');
        await assert.rejects(
            timeoutClient.connect({ timeoutMs: 1 }),
            (error) => error.code === 'timeout'
        );

        const protocolClient = new Client('192.168.1.43');
        const connected = protocolClient.connect();
        FakeWebSocket.instances.at(-1).open('wrong.protocol');
        await assert.rejects(
            connected,
            (error) => error.code === 'subprotocol_mismatch'
        );
    });

    it('classifies a browser connection failure without exposing an endpoint', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('device.local');
        const connected = client.connect();
        FakeWebSocket.instances[0].emit('error');
        await assert.rejects(
            connected,
            (error) => error.code === 'connection_failed'
                && !error.message.includes('device.local')
        );
    });

    it('negotiates espectre.v1, handshakes, and correlates command results', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('192.168.1.42');
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        assert.equal(socket.protocols, 'espectre.v1');
        socket.open();
        await connected;

        const handshake = client.handshake({ requestId: 'cap-1' });
        assert.deepEqual(socket.sent[0], {
            v: 1, type: 'request', id: 'cap-1', method: 'capabilities', params: {}
        });
        socket.receive({
            v: 1, type: 'response', id: 'cap-1', ok: true,
            result: { subprotocol: 'espectre.v1', methods: ['set_threshold'] }
        });
        assert.equal((await handshake).subprotocol, 'espectre.v1');
        assert.equal(client.compatible, true);

        const command = client.request('set_threshold', { threshold: 0.42 }, { requestId: 'write-1' });
        assert.deepEqual(socket.sent[1], {
            v: 1, type: 'request', id: 'write-1', method: 'set_threshold',
            params: { threshold: 0.42 }
        });
        socket.receive({
            v: 1, type: 'response', id: 'write-1', ok: true,
            result: { message: 'threshold updated' }
        });
        assert.equal((await command).message, 'threshold updated');
    });

    it('blocks mutations before compatibility is established', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('device.local');
        const connected = client.connect();
        FakeWebSocket.instances[0].open();
        await connected;
        await assert.rejects(
            client.request('set_mqtt_config', { host: 'broker.local' }),
            (error) => error.code === 'handshake_required'
        );
    });

    it('requires an explicit compatible protocol and method catalog', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('device.local');
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        socket.open();
        await connected;

        const missingProtocol = client.handshake({ requestId: 'cap-invalid-1' });
        socket.receive({
            v: 1, type: 'response', id: 'cap-invalid-1', ok: true,
            result: { methods: ['set_threshold'] }
        });
        await assert.rejects(missingProtocol, (error) => error.code === 'invalid_capabilities');
        assert.equal(client.compatible, false);

        const invalidMethods = client.handshake({ requestId: 'cap-invalid-2' });
        socket.receive({
            v: 1, type: 'response', id: 'cap-invalid-2', ok: true,
            result: { subprotocol: 'espectre.v1', methods: ['set threshold'] }
        });
        await assert.rejects(invalidMethods, (error) => error.code === 'invalid_capabilities');
        assert.equal(client.compatible, false);
    });

    it('delivers events and rejects malformed or uncorrelated envelopes', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('10.0.0.2');
        const events = [];
        const errors = [];
        client.on('event', (name, data) => events.push({ name, data }));
        client.on('protocol-error', (error) => errors.push(error.code));
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        socket.open();
        await connected;

        socket.receive({ v: 1, type: 'event', event: 'telemetry', data: { movement_score: 0.7 } });
        socket.receive('{');
        socket.receive({ v: 1, type: 'response', id: 'missing', ok: true, result: {} });
        assert.deepEqual(events, [{ name: 'telemetry', data: { movement_score: 0.7 } }]);
        assert.deepEqual(errors, ['invalid_json', 'unknown_request']);
    });

    it('surfaces device errors and rejects pending work on close', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('172.20.0.4');
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        socket.open();
        await connected;

        const rejected = client.request('info', {}, { requestId: 'read-1' });
        socket.receive({
            v: 1, type: 'response', id: 'read-1', ok: false,
            error: { code: 'rejected', message: 'not ready' }
        });
        await assert.rejects(rejected, (error) => error.code === 'rejected' && error.message === 'not ready');

        const pending = client.request('status', {}, { requestId: 'read-2' });
        socket.close(1006, 'network lost');
        await assert.rejects(pending, (error) => error.code === 'closed');
    });
});
