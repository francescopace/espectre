/*
 * ESPectre - Browser Direct HTTP client unit tests
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { DIRECT_PORT, peerDiscoveryScenarios } from './fixtures/peer_discovery_fixture.mjs';

globalThis.window = globalThis.window ?? {};
createRequire(import.meta.url)('../../docs/web/assets/js/espectre-direct.js');

const Client = window.ESPectreDirectClient;
const DirectError = window.ESPectreDirectError;
const RawParser = window.ESPectreRawCsiParser;

function rawFrame({
    sessionId = '00112233445566778899aabbccddeeff',
    sequence = 1n,
    fresh = 1n,
    dropped = 0n,
    backpressure = 0n,
    payload = new Uint8Array([1, 2, 3, 4])
} = {}) {
    const record = Buffer.alloc(64 + payload.length);
    record.writeUInt16LE(0x4353, 0);
    record.writeUInt8(8, 2);
    record.writeUInt8(64, 3);
    record.writeUInt8(4, 4);
    record.writeUInt8(1 << 3, 5);
    const saturatedSequence = sequence > 0xFFFFFFFFn ? 0xFFFFFFFF : Number(sequence);
    const saturatedFresh = fresh > 0xFFFFFFFFn ? 0xFFFFFFFF : Number(fresh);
    record.writeUInt32LE(saturatedSequence, 6);
    record.writeUInt16LE(payload.length / 2, 10);
    record.writeUInt16LE(payload.length, 12);
    record.writeBigUInt64LE(backpressure, 45);
    record.writeUInt32LE(saturatedFresh, 53);
    record.writeUInt32LE(saturatedSequence, 57);
    Buffer.from(payload).copy(record, 64);

    const prefix = Buffer.alloc(60);
    prefix.writeUInt32LE(0x52505345, 0);
    prefix.writeUInt8(1, 4);
    prefix.writeUInt8(8, 5);
    prefix.writeUInt16LE(60, 6);
    Buffer.from(sessionId, 'hex').copy(prefix, 8);
    prefix.writeBigUInt64LE(sequence, 24);
    prefix.writeUInt16LE(record.length, 32);
    prefix.writeUInt16LE(0, 34);
    prefix.writeBigUInt64LE(fresh, 36);
    prefix.writeBigUInt64LE(dropped, 44);
    prefix.writeBigUInt64LE(backpressure, 52);
    return new Uint8Array(Buffer.concat([prefix, record]));
}

function pendingBody(chunks = []) {
    let index = 0;
    return {
        getReader() {
            return {
                async read() {
                    if (index < chunks.length) return { value: new TextEncoder().encode(chunks[index++]), done: false };
                    return new Promise(() => {});
                },
                releaseLock() {}
            };
        }
    };
}

function responseEnvelope(request, result = {}) {
    return JSON.stringify({
        protocol_version: '1.0',
        device_id: '0123456789abcdef',
        command_id: request.command_id,
        command: request.command,
        accepted: true,
        code: 'ok',
        message: 'completed',
        data: result
    });
}

function installHttpFixture({ eventChunks = [], resultFor = () => ({}) } = {}) {
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        if (options.method === 'GET') return { ok: true, status: 200, body: pendingBody(eventChunks) };
        const request = JSON.parse(options.body);
        return { ok: true, status: 200, text: async () => responseEnvelope(request, resultFor(request)) };
    };
    return calls;
}

async function connectedClient(fixture = {}) {
    const calls = installHttpFixture(fixture);
    const client = new Client('192.168.1.42');
    await client.connect();
    return { client, calls };
}

afterEach(() => {
    delete globalThis.fetch;
    delete window.ESPectreBrowserSupport;
});

describe('Direct HTTP endpoint policy', () => {
    it('normalizes private IPv4, .local, HTTPS, and local IPv6 endpoints', () => {
        assert.equal(Client.DEFAULT_PORT, DIRECT_PORT);
        assert.equal(Client.normalizeEndpoint('192.168.1.42'), `http://192.168.1.42:${DIRECT_PORT}/espectre/v1/request`);
        assert.equal(Client.normalizeEndpoint('espectre-a1.local'), `http://espectre-a1.local:${DIRECT_PORT}/espectre/v1/request`);
        assert.equal(Client.normalizeEndpoint('https://espectre-a1.local/espectre/v1/request'), `https://espectre-a1.local:${DIRECT_PORT}/espectre/v1/request`);
        assert.equal(Client.normalizeEndpoint('http://[fd12:3456:789a::42]:61443/espectre/v1/request'), 'http://[fd12:3456:789a::42]:61443/espectre/v1/request');
    });

    it('rejects WebSocket, public, credentialed, queried, and unrelated endpoints', () => {
        for (const endpoint of [
            'ws://192.168.1.2/espectre/v1/ws', 'wss://espectre-a.local/espectre/v1/ws',
            '8.8.8.8', 'http://user:pass@192.168.1.2', 'http://192.168.1.2/?token=x',
            'http://192.168.1.2/admin'
        ]) assert.throws(() => Client.normalizeEndpoint(endpoint), DirectError);
    });

    it('creates a distinct lowercase 96-bit bootstrap hostname from injected entropy', () => {
        let invocation = 0;
        const randomSource = { getRandomValues(bytes) { bytes.fill(invocation++); return bytes; } };
        assert.equal(Client.createDiscoveryEndpoint(randomSource), `http://espectre-devices-000000000000000000000000.local:${DIRECT_PORT}/espectre/v1/request`);
        assert.equal(Client.createDiscoveryEndpoint(randomSource), `http://espectre-devices-010101010101010101010101.local:${DIRECT_PORT}/espectre/v1/request`);
        assert.throws(() => Client.createDiscoveryEndpoint(null), (error) => error.code === 'unsupported_crypto');
    });
});

describe('Raw CSI HTTP parser', () => {
    it('reconstructs split and aggregated frames and exposes counters', () => {
        const parser = new RawParser('00112233445566778899aabbccddeeff');
        const first = rawFrame();
        const second = rawFrame({ sequence: 3n, fresh: 2n, dropped: 1n });
        assert.deepEqual(parser.append(first.subarray(0, 17)), []);
        const records = parser.append(new Uint8Array([
            ...first.subarray(17),
            ...second
        ]));
        assert.equal(records.length, 2);
        assert.deepEqual(records.map((record) => record.streamSequence), [1n, 3n]);
        assert.equal(parser.freshRecordTotal, 2n);
        assert.equal(parser.rawDropTotal, 1n);
        assert.equal(parser.sendBackpressureTotal, 0n);
        assert.equal(parser.bufferedBytes, 0);
    });

    it('fails closed on session, flags, counter, and V8 sequence mismatches', () => {
        const wrongSession = rawFrame({ sessionId: '10112233445566778899aabbccddeeff' });
        assert.throws(
            () => new RawParser('00112233445566778899aabbccddeeff').append(wrongSession),
            (error) => error.code === 'invalid_raw_frame'
        );

        const wrongFlags = rawFrame();
        new DataView(wrongFlags.buffer, wrongFlags.byteOffset).setUint16(34, 1, true);
        assert.throws(() => new RawParser('00112233445566778899aabbccddeeff').append(wrongFlags));

        const skippedFresh = rawFrame({ fresh: 2n });
        assert.throws(() => new RawParser('00112233445566778899aabbccddeeff').append(skippedFresh));

        const wrongRecordSequence = rawFrame();
        new DataView(wrongRecordSequence.buffer, wrongRecordSequence.byteOffset)
            .setUint32(60 + 6, 2, true);
        assert.throws(
            () => new RawParser('00112233445566778899aabbccddeeff').append(wrongRecordSequence),
            (error) => error.code === 'invalid_raw_record'
        );
    });
});

describe('Peer discovery schema v2', () => {
    it('uses one bootstrap POST without opening SSE or querying capabilities', async () => {
        const calls = installHttpFixture({ resultFor: () => peerDiscoveryScenarios.multiFrontend });
        const client = new Client(Client.createDiscoveryEndpoint({
            getRandomValues(bytes) { bytes.fill(0xab); return bytes; }
        }));
        const result = await client.discoverPeersBootstrap();
        assert.equal(result.devices.length, 3);
        assert.equal(client.connected, false);
        assert.equal(calls.length, 1);
        assert.equal(calls[0].options.method, 'POST');
        assert.equal(JSON.parse(calls[0].options.body).command, 'discover_peers');
        client.close();
    });

    it('accepts HTTP peers and constructs request endpoints', () => {
        const result = Client.validatePeerDiscoveryResult(peerDiscoveryScenarios.multiFrontend);
        assert.equal(result.devices.length, 3);
        assert.equal(result.devices[0].endpoints[0], `http://192.168.1.42:${DIRECT_PORT}/espectre/v1/request`);
        const esphome = result.devices.find((device) => device.frontend === 'esphome');
        assert.equal(esphome.endpoints[0], `http://192.168.1.44:${DIRECT_PORT}/espectre/v1/request`);
    });

    it('accepts partial results, and rejects hostile or old-schema peers', () => {
        for (const name of ['partial', 'truncated', 'timeout', 'mixedAddresses']) {
            assert.doesNotThrow(() => Client.validatePeerDiscoveryResult(peerDiscoveryScenarios[name]));
        }
        for (const name of ['duplicateIdentity', 'malformed', 'oversized', 'nonLocal', 'malformedAddress', 'websocket']) {
            assert.throws(() => Client.validatePeerDiscoveryResult(peerDiscoveryScenarios[name]), DirectError);
        }
    });
});

describe('Direct HTTP request and SSE lifecycle', () => {
    it('does not issue a local request when browser permission is denied', async () => {
        let fetchCalls = 0;
        globalThis.fetch = async () => { fetchCalls += 1; };
        window.ESPectreBrowserSupport = {
            localNetworkAccessState: async () => 'denied'
        };
        const client = new Client('192.168.1.42');
        await assert.rejects(client.connect(), (error) => error.code === 'local_network_denied');
        assert.equal(fetchCalls, 0);
    });

    it('opens SSE with private-network options and posts correlated JSON without caching', async () => {
        const capabilities = { commands: [{ name: 'capabilities' }, { name: 'info' }], features: { raw_csi: false } };
        const { client, calls } = await connectedClient({
            resultFor: (request) => request.command === 'capabilities' ? capabilities : { firmware: '4.0.0' }
        });
        await client.handshake();
        assert.deepEqual(await client.request('info'), { firmware: '4.0.0' });
        assert.equal(calls[0].url, `http://192.168.1.42:${DIRECT_PORT}/espectre/v1/events`);
        assert.equal(calls[0].options.targetAddressSpace, 'local');
        assert.equal(calls[0].options.cache, 'no-store');
        assert.equal(calls[1].options.method, 'POST');
        assert.equal(calls[1].options.headers['Content-Type'], 'application/json');
        assert.equal(JSON.parse(calls[2].options.body).command, 'info');
        client.close();
    });

    it('parses SSE events split across fetch chunks', async () => {
        const envelope = JSON.stringify({ protocol_version: '1.0', device_id: '0123456789abcdef', movement_score: 0.42 });
        const midpoint = Math.floor(envelope.length / 2);
        installHttpFixture({
            eventChunks: [`event: telemetry\ndata: ${envelope.slice(0, midpoint)}`, `${envelope.slice(midpoint)}\n\n`]
        });
        const client = new Client('192.168.1.42');
        const eventPromise = new Promise((resolve) => client.on('event', (name, data) => resolve({ name, data })));
        await client.connect();
        const event = await eventPromise;
        assert.deepEqual(event, { name: 'telemetry', data: { protocol_version: '1.0', device_id: '0123456789abcdef', movement_score: 0.42 } });
        client.close();
    });

    it('parses an SSE delimiter split between CR and LF chunks', async () => {
        const envelope = JSON.stringify({ protocol_version: '1.0', device_id: '0123456789abcdef', sensing: true });
        installHttpFixture({ eventChunks: [`event: status\r\ndata: ${envelope}\r`, '\n\r\n'] });
        const client = new Client('192.168.1.42');
        const eventPromise = new Promise((resolve) => client.on('event', (name, data) => resolve({ name, data })));
        await client.connect();
        assert.deepEqual(await eventPromise, { name: 'status', data: { protocol_version: '1.0', device_id: '0123456789abcdef', sensing: true } });
        client.close();
    });

    it('blocks mutations before handshake and sends the raw bearer when stopping', async () => {
        const sessionId = '00112233445566778899aabbccddeeff';
        const { client, calls } = await connectedClient({
            resultFor: (request) => {
                if (request.command === 'capabilities') {
                    return { commands: [{ name: 'capabilities' }, { name: 'start_raw_stream' }, { name: 'stop_raw_stream' }], features: { raw_csi: true } };
                }
                if (request.command === 'start_raw_stream') return { session_id: sessionId };
                return {};
            }
        });
        await assert.rejects(client.request('start_raw_stream'), (error) => error.code === 'handshake_required');
        await client.handshake();
        await client.request('start_raw_stream');
        assert.equal(client.rawSessionId, sessionId);
        await client.request('stop_raw_stream');
        assert.equal(calls.at(-1).options.headers.Authorization, `Bearer ${sessionId}`);
        assert.equal(client.rawSessionId, '');
        client.close();
    });

    it('reports HTTP rate limits explicitly', async () => {
        globalThis.fetch = async (_url, options) => {
            if (options.method === 'GET') return { ok: true, status: 200, body: pendingBody() };
            return { ok: false, status: 429, text: async () => 'rate limited' };
        };
        const client = new Client('192.168.1.42');
        await client.connect();
        await assert.rejects(client.request('capabilities', {}, { allowBeforeHandshake: true }), (error) => error.code === 'http_429');
        client.close();
    });
});
