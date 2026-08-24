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
import { peerDiscoveryScenarios } from './fixtures/peer_discovery_fixture.mjs';

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

    it('constructs bracketed local IPv6 endpoints', () => {
        assert.equal(
            Client.normalizeEndpoint('ws://[fd12:3456:789a::42]:6054/espectre/v1/ws'),
            'ws://[fd12:3456:789a::42]:6054/espectre/v1/ws'
        );
    });

    it('creates a fresh 96-bit lowercase bootstrap endpoint from injected entropy', () => {
        let invocation = 0;
        const randomSource = {
            getRandomValues(bytes) {
                bytes.fill(invocation++);
                return bytes;
            }
        };
        assert.equal(
            Client.createDiscoveryEndpoint(randomSource),
            'ws://espectre-devices-000000000000000000000000.local/espectre/v1/ws'
        );
        assert.equal(
            Client.createDiscoveryEndpoint(randomSource),
            'ws://espectre-devices-010101010101010101010101.local/espectre/v1/ws'
        );
    });

    it('fails explicitly when Web Crypto is unavailable', () => {
        assert.throws(
            () => Client.createDiscoveryEndpoint(null),
            (error) => error instanceof DirectError && error.code === 'unsupported_crypto'
        );
    });
});

describe('Peer discovery schema', () => {
    it('validates bounded local peers and constructs unique endpoints', () => {
        const result = Client.validatePeerDiscoveryResult({
            schema_version: 1, elapsed_ms: 3000, status: 'complete', truncated: false,
            rejected_results: 2,
            devices: [{
                device_id: '0123456789abcdef', instance: 'Kitchen sensor',
                hostname: 'espectre-0123456789abcdef', name: 'Kitchen', frontend: 'native',
                txt_version: 1, protocol_version: 1, path: '/espectre/v1/ws',
                firmware: '3.0.0-rc1', chip: 'esp32c3', tls: false, port: 80,
                capabilities: ['config', 'monitor', 'peer_discovery'],
                addresses: ['192.168.1.42']
            }]
        });
        assert.deepEqual(result.devices[0].endpoints, [
            'ws://192.168.1.42/espectre/v1/ws'
        ]);
    });

    it('rejects off-LAN addresses, duplicate identities, and oversized results', () => {
        const peer = {
            device_id: '0123456789abcdef', instance: 'Kitchen sensor',
            hostname: 'espectre-0123456789abcdef', name: '', frontend: 'native',
            txt_version: 1, protocol_version: 1, path: '/espectre/v1/ws',
            firmware: '3.0.0', chip: 'esp32c3', tls: false, port: 80,
            capabilities: ['monitor'], addresses: ['8.8.8.8']
        };
        const envelope = {
            schema_version: 1, elapsed_ms: 1, status: 'complete', truncated: false,
            rejected_results: 0, devices: [peer]
        };
        assert.throws(() => Client.validatePeerDiscoveryResult(envelope), DirectError);
        assert.throws(() => Client.validatePeerDiscoveryResult({
            ...envelope,
            devices: [
                { ...peer, addresses: ['192.168.1.2'] },
                { ...peer, addresses: ['192.168.1.3'] }
            ]
        }), DirectError);
        assert.throws(() => Client.validatePeerDiscoveryResult({
            ...envelope, devices: Array.from({ length: 9 }, () => peer)
        }), DirectError);
    });

    it('accepts the deterministic mixed-address, partial, truncated, and timeout fixtures', () => {
        const multiFrontend = Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.multiFrontend);
        assert.deepEqual(multiFrontend.devices.map((device) => [device.frontend, device.endpoints[0]]), [
            ['native', 'ws://192.168.1.42/espectre/v1/ws'],
            ['streamer', 'ws://192.168.1.43/espectre/v1/ws'],
            ['esphome', 'ws://192.168.1.44:6054/espectre/v1/ws'],
            ['matter', 'ws://192.168.1.45/espectre/v1/ws']
        ]);
        const mixed = Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.mixedAddresses);
        assert.deepEqual(mixed.devices.map((device) => device.endpoints[0]), [
            'ws://192.168.1.42/espectre/v1/ws',
            'ws://[fd12:3456:789a::42]:6054/espectre/v1/ws'
        ]);
        assert.equal(Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.partial).rejected_results, 3);
        assert.equal(Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.truncated).truncated, true);
        assert.equal(Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.timeout).status, 'timeout');
    });

    it('rejects every hostile deterministic peer fixture', () => {
        for (const name of ['duplicateIdentity', 'malformed', 'oversized', 'nonLocal', 'malformedAddress']) {
            assert.throws(
                () => Client.validatePeerDiscoveryResult(peerDiscoveryScenarios[name]),
                (error) => error.code === 'invalid_peer_result',
                name
            );
        }
    });
});

describe('Direct request lifecycle', () => {
    it('accepts responses larger than requests and rejects responses above 8192 bytes', async () => {
        globalThis.WebSocket = FakeWebSocket;
        assert.equal(Client.MAX_FRAME_BYTES, 4096);
        assert.equal(Client.MAX_REQUEST_FRAME_BYTES, 4096);
        assert.equal(Client.MAX_RESPONSE_FRAME_BYTES, 8192);

        const client = new Client('192.168.1.42');
        const errors = [];
        client.on('protocol-error', (error) => errors.push(error.code));
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        socket.open();
        await connected;

        const handshake = client.handshake({ requestId: 'large-capabilities' });
        socket.receive({
            v: 1, type: 'response', id: 'large-capabilities', ok: true,
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'discover_peers' }], padding: 'x'.repeat(4200)
            } }
        });
        assert.equal((await handshake).commands[0].name, 'discover_peers');

        socket.receive({
            v: 1, type: 'event', event: 'telemetry', data: { padding: 'x'.repeat(8200) }
        });
        assert.deepEqual(errors, ['frame_too_large']);
    });

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
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'set_threshold' }]
            } }
        });
        assert.equal((await handshake).commands[0].name, 'set_threshold');
        assert.equal(client.compatible, true);

        const command = client.request('set_threshold', { threshold: 0.42 }, { requestId: 'write-1' });
        assert.deepEqual(socket.sent[1], {
            v: 1, type: 'request', id: 'write-1', method: 'set_threshold',
            params: { threshold: 0.42 }
        });
        socket.receive({
            v: 1, type: 'response', id: 'write-1', ok: true,
            result: { command: 'set_threshold', code: 'ok', message: 'threshold updated' }
        });
        assert.equal((await command).message, 'threshold updated');
    });

    it('requests peer discovery only after capability negotiation', async () => {
        globalThis.WebSocket = FakeWebSocket;
        const client = new Client('device.local');
        const connected = client.connect();
        const socket = FakeWebSocket.instances[0];
        socket.open();
        await connected;
        await assert.rejects(client.discoverPeers(), (error) => error.code === 'unsupported_capability');

        const handshake = client.handshake({ requestId: 'cap-peers' });
        socket.receive({
            v: 1, type: 'response', id: 'cap-peers', ok: true,
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'discover_peers' }]
            } }
        });
        await handshake;
        const discovery = client.discoverPeers({ requestId: 'peers-1' });
        assert.equal(socket.sent.at(-1).method, 'discover_peers');
        socket.receive({
            v: 1, type: 'response', id: 'peers-1', ok: true,
            result: { command: 'discover_peers', code: 'ok', message: 'peer discovery completed', data: {
                schema_version: 1, elapsed_ms: 42, status: 'complete', truncated: false,
                rejected_results: 0, devices: []
            } }
        });
        assert.deepEqual((await discovery).devices, []);
    });

    it('handles delayed discovery, timeout, and responder disconnect deterministically', async () => {
        globalThis.WebSocket = FakeWebSocket;

        const delayedClient = new Client('device.local');
        const delayedConnection = delayedClient.connect();
        const delayedSocket = FakeWebSocket.instances.at(-1);
        delayedSocket.open();
        await delayedConnection;
        const delayedHandshake = delayedClient.handshake({ requestId: 'caps-delayed' });
        delayedSocket.receive({
            v: 1, type: 'response', id: 'caps-delayed', ok: true,
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'discover_peers' }]
            } }
        });
        await delayedHandshake;
        const delayed = delayedClient.discoverPeers({ requestId: 'peers-delayed', timeoutMs: 50 });
        setTimeout(() => delayedSocket.receive({
            v: 1, type: 'response', id: 'peers-delayed', ok: true,
            result: { command: 'discover_peers', code: 'ok', message: 'peer discovery completed', data: peerDiscoveryScenarios.partial }
        }), 2);
        assert.equal((await delayed).devices.length, 1);
        delayedClient.close();

        const timeoutClient = new Client('device.local');
        const timeoutConnection = timeoutClient.connect();
        const timeoutSocket = FakeWebSocket.instances.at(-1);
        timeoutSocket.open();
        await timeoutConnection;
        const timeoutHandshake = timeoutClient.handshake({ requestId: 'caps-timeout' });
        timeoutSocket.receive({
            v: 1, type: 'response', id: 'caps-timeout', ok: true,
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'discover_peers' }]
            } }
        });
        await timeoutHandshake;
        await assert.rejects(
            timeoutClient.discoverPeers({ requestId: 'peers-timeout', timeoutMs: 1 }),
            (error) => error.code === 'timeout'
        );
        timeoutClient.close();

        const disconnectedClient = new Client('device.local');
        const disconnectedConnection = disconnectedClient.connect();
        const disconnectedSocket = FakeWebSocket.instances.at(-1);
        disconnectedSocket.open();
        await disconnectedConnection;
        const disconnectedHandshake = disconnectedClient.handshake({ requestId: 'caps-disconnect' });
        disconnectedSocket.receive({
            v: 1, type: 'response', id: 'caps-disconnect', ok: true,
            result: { command: 'capabilities', code: 'ok', message: 'capabilities returned', data: {
                commands: [{ name: 'discover_peers' }]
            } }
        });
        await disconnectedHandshake;
        const pending = disconnectedClient.discoverPeers({ requestId: 'peers-disconnect' });
        disconnectedSocket.close(1006, 'responder lost');
        await assert.rejects(pending, (error) => error.code === 'closed');
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
