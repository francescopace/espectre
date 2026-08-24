/*
 * ESPectre Direct WebSocket Client
 *
 * Dependency-free client for the versioned local Native endpoint. It owns
 * endpoint validation, the WebSocket subprotocol, request correlation,
 * envelope validation, and device events; UI policy remains with callers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const ENVELOPE_VERSION = 1;
    const SUBPROTOCOL = 'espectre.v1';
    const ENDPOINT_PATH = '/espectre/v1/ws';
    const MAX_REQUEST_FRAME_BYTES = 4096;
    const MAX_RESPONSE_FRAME_BYTES = 8192;
    const DEFAULT_TIMEOUT_MS = 8000;
    const PEER_DISCOVERY_TIMEOUT_MS = 5000;
    const PEER_DISCOVERY_MAX_DEVICES = 8;
    const PEER_DISCOVERY_MAX_ADDRESSES = 2;
    const DISCOVERY_NONCE_BYTES = 12;
    const DISCOVERY_HOST_PREFIX = 'espectre-devices-';
    const EVENTS = Object.freeze(['open', 'close', 'event', 'protocol-error']);
    const MUTATING_METHODS = Object.freeze(new Set([
        'clear_mqtt_config', 'clear_wifi_config', 'ota_start', 'recalibrate',
        'set_csi_traffic_mode', 'set_detector', 'set_device_label',
        'set_motion_hits', 'set_mqtt_config', 'set_threshold',
        'set_traffic_generator_mode', 'set_wifi_config', 'set_sensing'
    ]));

    class ESPectreDirectError extends Error {
        constructor(message, code = 'client_error') {
            super(message);
            this.name = 'ESPectreDirectError';
            this.code = code;
        }
    }

    function isLocalHostname(hostname) {
        const host = hostname.toLowerCase().replace(/^\[|\]$/g, '');
        if (host === 'localhost' || host.endsWith('.local')) return true;
        if (/^10\./.test(host) || /^192\.168\./.test(host)) return true;
        const ipv4 = host.match(/^172\.(\d{1,3})\./);
        if (ipv4 && Number(ipv4[1]) >= 16 && Number(ipv4[1]) <= 31) return true;
        if (host === '::1' || /^fe[89ab][0-9a-f]:/i.test(host) || /^f[cd][0-9a-f]{2}:/i.test(host)) {
            return true;
        }
        return false;
    }

    function isLocalPeerAddress(address) {
        if (typeof address !== 'string' || !address || address.includes('%')) return false;
        if (address.includes('.')) {
            const octets = address.split('.');
            if (octets.length !== 4 || octets.some((octet) => !/^(0|[1-9][0-9]{0,2})$/.test(octet))) {
                return false;
            }
            const values = octets.map(Number);
            if (values.some((value) => value > 255)) return false;
            return values[0] === 10
                || (values[0] === 172 && values[1] >= 16 && values[1] <= 31)
                || (values[0] === 192 && values[1] === 168);
        }
        if (!address.includes(':') || address === '::1') return false;
        try {
            new URL(`http://[${address}]/`);
        } catch (_error) {
            return false;
        }
        return /^fe[89ab][0-9a-f]:/i.test(address) || /^f[cd][0-9a-f]{2}:/i.test(address);
    }

    function normalizeEndpoint(value) {
        if (typeof value !== 'string' || !value.trim()) {
            throw new ESPectreDirectError('Enter a device IP address, .local hostname, or WebSocket URL.', 'invalid_endpoint');
        }
        const input = value.trim();
        const explicitScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(input);
        let url;
        try {
            url = new URL(explicitScheme ? input : `ws://${input}`);
        } catch (_error) {
            throw new ESPectreDirectError('The device endpoint is not a valid URL.', 'invalid_endpoint');
        }
        if (url.protocol === 'http:') url.protocol = 'ws:';
        if (url.protocol === 'https:') url.protocol = 'wss:';
        if (!['ws:', 'wss:'].includes(url.protocol)) {
            throw new ESPectreDirectError('The device endpoint must use ws:// or wss://.', 'invalid_scheme');
        }
        if (url.username || url.password || url.search || url.hash) {
            throw new ESPectreDirectError('The device endpoint cannot contain credentials, a query, or a fragment.', 'invalid_endpoint');
        }
        if (!isLocalHostname(url.hostname)) {
            throw new ESPectreDirectError('Use a private IP address, localhost, or a .local device name.', 'non_local_endpoint');
        }
        if (url.pathname !== '/' && url.pathname !== ENDPOINT_PATH) {
            throw new ESPectreDirectError(`The Direct endpoint path must be ${ENDPOINT_PATH}.`, 'invalid_path');
        }
        url.pathname = ENDPOINT_PATH;
        return url.toString();
    }

    function createDiscoveryEndpoint(randomSource = globalThis.crypto) {
        if (!randomSource || typeof randomSource.getRandomValues !== 'function') {
            throw new ESPectreDirectError(
                'Web Crypto is required for local Auto-discovery.',
                'unsupported_crypto'
            );
        }
        const bytes = new Uint8Array(DISCOVERY_NONCE_BYTES);
        randomSource.getRandomValues(bytes);
        const nonce = [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
        return normalizeEndpoint(`${DISCOVERY_HOST_PREFIX}${nonce}.local`);
    }

    function parseObject(text) {
        let data;
        try {
            data = JSON.parse(text);
        } catch (_error) {
            throw new ESPectreDirectError('Direct frame must be valid JSON.', 'invalid_json');
        }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
            throw new ESPectreDirectError('Direct frame must be a JSON object.', 'invalid_envelope');
        }
        if (data.v !== ENVELOPE_VERSION) {
            throw new ESPectreDirectError(`Unsupported Direct envelope version ${String(data.v)}.`, 'unsupported_version');
        }
        return data;
    }

    function validText(value, maximum, { empty = false, token = false } = {}) {
        if (typeof value !== 'string' || (!empty && !value) || value.length > maximum) return false;
        if (![...value].every((character) => character >= ' ' && character <= '~')) return false;
        return !token || /^[A-Za-z0-9_-]+$/.test(value);
    }

    function validatePeerDiscoveryResult(result) {
        if (!result || typeof result !== 'object' || Array.isArray(result)
            || result.schema_version !== 1
            || !Number.isInteger(result.elapsed_ms) || result.elapsed_ms < 0 || result.elapsed_ms > 10000
            || !['complete', 'timeout'].includes(result.status)
            || typeof result.truncated !== 'boolean'
            || !Number.isInteger(result.rejected_results) || result.rejected_results < 0
            || !Array.isArray(result.devices) || result.devices.length > PEER_DISCOVERY_MAX_DEVICES) {
            throw new ESPectreDirectError('Device returned an invalid peer discovery result.', 'invalid_peer_result');
        }
        const identities = new Set();
        const devices = result.devices.map((peer) => {
            const capabilities = peer?.capabilities;
            const addresses = peer?.addresses;
            const valid = peer && typeof peer === 'object' && !Array.isArray(peer)
                && /^[0-9a-f]{16}$/.test(peer.device_id)
                && validText(peer.instance, 63)
                && validText(peer.hostname, 63, { token: true })
                && validText(peer.name, 63, { empty: true })
                && ['native', 'streamer', 'esphome', 'matter'].includes(peer.frontend)
                && peer.txt_version === 1 && peer.protocol_version === 1
                && peer.path === ENDPOINT_PATH
                && validText(peer.firmware, 48)
                && validText(peer.chip, 16, { token: true })
                && peer.tls === false
                && Number.isInteger(peer.port) && peer.port > 0 && peer.port <= 65535
                && Array.isArray(capabilities) && capabilities.length > 0 && capabilities.length <= 8
                && capabilities.every((capability) => validText(capability, 32, { token: true }))
                && new Set(capabilities).size === capabilities.length
                && Array.isArray(addresses) && addresses.length > 0
                && addresses.length <= PEER_DISCOVERY_MAX_ADDRESSES
                && addresses.every(isLocalPeerAddress);
            if (!valid || identities.has(peer?.device_id)) {
                throw new ESPectreDirectError('Device returned an invalid or duplicate peer.', 'invalid_peer_result');
            }
            identities.add(peer.device_id);
            const endpoints = addresses.map((address) => {
                const host = address.includes(':') ? `[${address}]` : address;
                return normalizeEndpoint(`${peer.tls ? 'wss' : 'ws'}://${host}:${peer.port}${peer.path}`);
            });
            return Object.freeze({ ...peer, capabilities: [...capabilities], addresses: [...addresses], endpoints });
        });
        return Object.freeze({ ...result, devices });
    }

    function frameText(value) {
        if (typeof value === 'string') return value;
        if (value instanceof ArrayBuffer) return new TextDecoder().decode(new Uint8Array(value));
        if (ArrayBuffer.isView(value)) {
            return new TextDecoder().decode(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
        }
        throw new ESPectreDirectError('Direct frames must be text.', 'invalid_frame_type');
    }

    class ESPectreDirectClient {
        static get VERSION() { return '1.0.0'; }
        static get ENVELOPE_VERSION() { return ENVELOPE_VERSION; }
        static get SUBPROTOCOL() { return SUBPROTOCOL; }
        static get ENDPOINT_PATH() { return ENDPOINT_PATH; }
        static get MAX_FRAME_BYTES() { return MAX_REQUEST_FRAME_BYTES; }
        static get MAX_REQUEST_FRAME_BYTES() { return MAX_REQUEST_FRAME_BYTES; }
        static get MAX_RESPONSE_FRAME_BYTES() { return MAX_RESPONSE_FRAME_BYTES; }
        static get EVENTS() { return EVENTS; }
        static normalizeEndpoint(value) { return normalizeEndpoint(value); }
        static createDiscoveryEndpoint(randomSource) { return createDiscoveryEndpoint(randomSource); }
        static validatePeerDiscoveryResult(value) { return validatePeerDiscoveryResult(value); }

        #endpoint;
        #socket = null;
        #listeners = new Map();
        #pending = new Map();
        #sequence = 0;
        #compatible = false;
        #capabilities = null;
        #closing = false;

        constructor(endpoint) {
            this.#endpoint = normalizeEndpoint(endpoint);
        }

        get endpoint() { return this.#endpoint; }
        get connected() { return this.#socket?.readyState === 1; }
        get compatible() { return this.#compatible; }
        get capabilities() { return this.#capabilities; }

        on(event, handler) {
            if (!EVENTS.includes(event)) throw new ESPectreDirectError(`Unknown Direct event ${String(event)}.`);
            if (typeof handler !== 'function') throw new ESPectreDirectError('Event handler must be a function.');
            if (!this.#listeners.has(event)) this.#listeners.set(event, new Set());
            this.#listeners.get(event).add(handler);
            return () => this.off(event, handler);
        }

        off(event, handler) { this.#listeners.get(event)?.delete(handler); }

        #emit(event, ...args) {
            for (const handler of [...(this.#listeners.get(event) || [])]) {
                try { handler(...args); } catch (error) { console.error(`Direct ${event} handler failed:`, error); }
            }
        }

        connect({ timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
            if (this.#socket) return Promise.reject(new ESPectreDirectError('Direct client is already active.'));
            const Socket = globalThis.WebSocket || window.WebSocket;
            if (typeof Socket !== 'function') {
                return Promise.reject(new ESPectreDirectError('WebSocket is not available in this browser.', 'unsupported'));
            }
            this.#closing = false;
            return new Promise((resolve, reject) => {
                const socket = new Socket(this.#endpoint, SUBPROTOCOL);
                this.#socket = socket;
                socket.binaryType = 'arraybuffer';
                const timeout = setTimeout(() => {
                    if (this.#socket !== socket || socket.readyState === 1) return;
                    this.#closing = true;
                    socket.close(1000, 'connection timeout');
                    this.#socket = null;
                    reject(new ESPectreDirectError('Timed out connecting to the local device.', 'timeout'));
                }, timeoutMs);
                socket.addEventListener('open', () => {
                    clearTimeout(timeout);
                    if (socket.protocol !== SUBPROTOCOL) {
                        this.#closing = true;
                        socket.close(1002, 'subprotocol required');
                        this.#socket = null;
                        reject(new ESPectreDirectError('Device did not negotiate espectre.v1.', 'subprotocol_mismatch'));
                        return;
                    }
                    this.#emit('open');
                    resolve();
                }, { once: true });
                socket.addEventListener('message', (event) => this.#ingest(event.data));
                socket.addEventListener('error', () => {
                    if (socket.readyState !== 1) {
                        clearTimeout(timeout);
                        reject(new ESPectreDirectError('Direct WebSocket connection failed.', 'connection_failed'));
                    }
                });
                socket.addEventListener('close', (event) => {
                    clearTimeout(timeout);
                    if (this.#socket === socket) this.#socket = null;
                    this.#compatible = false;
                    this.#capabilities = null;
                    this.#rejectAll('Direct WebSocket closed.', 'closed');
                    this.#emit('close', { code: event.code, reason: event.reason, expected: this.#closing });
                });
            });
        }

        async handshake(options = {}) {
            const result = await this.request('capabilities', {}, { ...options, allowBeforeHandshake: true });
            if (!result || typeof result !== 'object' || Array.isArray(result)
                || !Array.isArray(result.commands)
                || result.commands.some((item) => !item || typeof item.name !== 'string'
                    || !/^[A-Za-z0-9_.-]{1,64}$/.test(item.name))) {
                throw new ESPectreDirectError('Device returned invalid capabilities.', 'invalid_capabilities');
            }
            this.#compatible = true;
            this.#capabilities = result;
            return result;
        }

        request(method, params = {}, {
            timeoutMs = DEFAULT_TIMEOUT_MS,
            requestId,
            allowBeforeHandshake = false
        } = {}) {
            if (!this.connected) return Promise.reject(new ESPectreDirectError('Direct WebSocket is not connected.', 'not_connected'));
            if (typeof method !== 'string' || !/^[A-Za-z0-9_.-]{1,64}$/.test(method)) {
                return Promise.reject(new ESPectreDirectError('Direct method is invalid.', 'invalid_method'));
            }
            if (!params || typeof params !== 'object' || Array.isArray(params)) {
                return Promise.reject(new ESPectreDirectError('Direct params must be an object.', 'invalid_params'));
            }
            if (!allowBeforeHandshake && MUTATING_METHODS.has(method) && !this.#compatible) {
                return Promise.reject(new ESPectreDirectError('Complete the Direct capability handshake before changing the device.', 'handshake_required'));
            }
            const id = requestId || `web-${Date.now().toString(36)}-${(++this.#sequence).toString(36)}`;
            if (!/^[A-Za-z0-9_.:-]{1,64}$/.test(id) || this.#pending.has(id)) {
                return Promise.reject(new ESPectreDirectError('Direct request id is invalid or already pending.', 'invalid_request_id'));
            }
            const payload = JSON.stringify({ v: ENVELOPE_VERSION, type: 'request', id, method, params });
            if (new TextEncoder().encode(payload).byteLength > MAX_REQUEST_FRAME_BYTES) {
                return Promise.reject(new ESPectreDirectError('Direct request exceeds the 4096-byte frame limit.', 'frame_too_large'));
            }
            return new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    this.#pending.delete(id);
                    reject(new ESPectreDirectError(`Direct request ${method} timed out.`, 'timeout'));
                }, timeoutMs);
                this.#pending.set(id, { method, resolve, reject, timer });
                try {
                    this.#socket.send(payload);
                } catch (error) {
                    clearTimeout(timer);
                    this.#pending.delete(id);
                    reject(error);
                }
            });
        }

        async discoverPeers(options = {}) {
            if (!this.#compatible || !this.#capabilities?.commands?.some((item) => item.name === 'discover_peers')) {
                throw new ESPectreDirectError('This responder does not support peer discovery.', 'unsupported_capability');
            }
            const result = await this.request('discover_peers', {}, {
                timeoutMs: PEER_DISCOVERY_TIMEOUT_MS,
                ...options
            });
            return validatePeerDiscoveryResult(result);
        }

        #ingest(value) {
            let text;
            let envelope;
            try {
                text = frameText(value);
                if (new TextEncoder().encode(text).byteLength > MAX_RESPONSE_FRAME_BYTES) {
                    throw new ESPectreDirectError('Direct response exceeds the 8192-byte frame limit.', 'frame_too_large');
                }
                envelope = parseObject(text);
            } catch (error) {
                this.#emit('protocol-error', error);
                return;
            }
            if (envelope.type === 'event') {
                if (typeof envelope.event !== 'string' || !envelope.data || typeof envelope.data !== 'object' || Array.isArray(envelope.data)) {
                    this.#emit('protocol-error', new ESPectreDirectError('Direct event envelope is invalid.', 'invalid_envelope'));
                    return;
                }
                this.#emit('event', envelope.event, envelope.data);
                return;
            }
            if (envelope.type !== 'response' || typeof envelope.id !== 'string' || typeof envelope.ok !== 'boolean') {
                this.#emit('protocol-error', new ESPectreDirectError('Direct response envelope is invalid.', 'invalid_envelope'));
                return;
            }
            const pending = this.#pending.get(envelope.id);
            if (!pending) {
                this.#emit('protocol-error', new ESPectreDirectError('Direct response has no matching request.', 'unknown_request'));
                return;
            }
            clearTimeout(pending.timer);
            this.#pending.delete(envelope.id);
            if (envelope.ok) {
                if (!envelope.result || typeof envelope.result !== 'object' || Array.isArray(envelope.result)) {
                    pending.reject(new ESPectreDirectError('Direct success result must be an object.', 'invalid_envelope'));
                    return;
                }
                pending.resolve(envelope.result.data ?? envelope.result);
                return;
            }
            const code = typeof envelope.error?.code === 'string' ? envelope.error.code : 'device_error';
            const message = typeof envelope.error?.message === 'string' ? envelope.error.message : 'Device rejected the request.';
            pending.reject(new ESPectreDirectError(message, code));
        }

        #rejectAll(message, code) {
            for (const pending of this.#pending.values()) {
                clearTimeout(pending.timer);
                pending.reject(new ESPectreDirectError(message, code));
            }
            this.#pending.clear();
        }

        close(code = 1000, reason = 'client closed') {
            this.#closing = true;
            const socket = this.#socket;
            this.#socket = null;
            this.#compatible = false;
            this.#capabilities = null;
            this.#rejectAll('Direct client closed.', 'closed');
            if (socket && socket.readyState < 2) socket.close(code, reason);
        }
    }

    window.ESPectreDirectClient = ESPectreDirectClient;
    window.ESPectreDirectError = ESPectreDirectError;
})();
