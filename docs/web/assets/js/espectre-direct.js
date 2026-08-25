/*
 * ESPectre Direct HTTP Client
 *
 * Dependency-free client for the versioned local HTTP API. It owns endpoint
 * validation, correlated POST requests, incremental SSE parsing, and device
 * events; UI policy remains with callers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const ENVELOPE_VERSION = 1;
    const REQUEST_PATH = '/espectre/v1/request';
    const EVENTS_PATH = '/espectre/v1/events';
    const RAW_PATH = '/espectre/v1/csi';
    const MAX_REQUEST_BYTES = 4096;
    const MAX_RESPONSE_BYTES = 8192;
    const DEFAULT_TIMEOUT_MS = 8000;
    const PEER_DISCOVERY_TIMEOUT_MS = 10000;
    const PEER_DISCOVERY_MAX_DEVICES = 8;
    const PEER_DISCOVERY_MAX_ADDRESSES = 2;
    const DISCOVERY_NONCE_BYTES = 12;
    const DISCOVERY_HOST_PREFIX = 'espectre-devices-';
    const EVENTS = Object.freeze(['open', 'close', 'event', 'protocol-error']);
    const MUTATING_METHODS = Object.freeze(new Set([
        'clear_mqtt_config', 'clear_wifi_config', 'ota_start', 'recalibrate', 'scan_wifi_access_points',
        'set_csi_traffic_mode', 'set_detector', 'set_device_label',
        'set_motion_hits', 'set_mqtt_config', 'set_threshold',
        'set_traffic_generator_mode', 'set_wifi_bssid', 'set_sensing',
        'start_raw_stream', 'stop_raw_stream'
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
        return host === '::1' || /^fe[89ab][0-9a-f]:/i.test(host) || /^f[cd][0-9a-f]{2}:/i.test(host);
    }

    function isLocalPeerAddress(address) {
        if (typeof address !== 'string' || !address || address.includes('%')) return false;
        if (address.includes('.')) {
            const octets = address.split('.');
            if (octets.length !== 4 || octets.some((octet) => !/^(0|[1-9][0-9]{0,2})$/.test(octet))) return false;
            const values = octets.map(Number);
            if (values.some((value) => value > 255)) return false;
            return values[0] === 10
                || (values[0] === 172 && values[1] >= 16 && values[1] <= 31)
                || (values[0] === 192 && values[1] === 168);
        }
        if (!address.includes(':') || address === '::1') return false;
        try { new URL(`http://[${address}]/`); } catch (_error) { return false; }
        return /^fe[89ab][0-9a-f]:/i.test(address) || /^f[cd][0-9a-f]{2}:/i.test(address);
    }

    function normalizeEndpoint(value) {
        if (typeof value !== 'string' || !value.trim()) {
            throw new ESPectreDirectError('Enter a device IP address, .local hostname, or HTTP URL.', 'invalid_endpoint');
        }
        const input = value.trim();
        const explicitScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(input);
        let url;
        try { url = new URL(explicitScheme ? input : `http://${input}`); } catch (_error) {
            throw new ESPectreDirectError('The device endpoint is not a valid URL.', 'invalid_endpoint');
        }
        if (!['http:', 'https:'].includes(url.protocol)) {
            throw new ESPectreDirectError('The device endpoint must use http:// or https://.', 'invalid_scheme');
        }
        if (url.username || url.password || url.search || url.hash) {
            throw new ESPectreDirectError('The device endpoint cannot contain credentials, a query, or a fragment.', 'invalid_endpoint');
        }
        if (!isLocalHostname(url.hostname)) {
            throw new ESPectreDirectError('Use a private IP address, localhost, or a .local device name.', 'non_local_endpoint');
        }
        if (url.pathname !== '/' && url.pathname !== REQUEST_PATH) {
            throw new ESPectreDirectError(`The Direct endpoint path must be ${REQUEST_PATH}.`, 'invalid_path');
        }
        url.pathname = REQUEST_PATH;
        return url.toString();
    }

    function endpointWithPath(endpoint, path) {
        const url = new URL(endpoint);
        url.pathname = path;
        return url.toString();
    }

    function createDiscoveryEndpoint(randomSource = globalThis.crypto) {
        if (!randomSource || typeof randomSource.getRandomValues !== 'function') {
            throw new ESPectreDirectError('Web Crypto is required for local Auto-discovery.', 'unsupported_crypto');
        }
        const bytes = new Uint8Array(DISCOVERY_NONCE_BYTES);
        randomSource.getRandomValues(bytes);
        const nonce = [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
        return normalizeEndpoint(`${DISCOVERY_HOST_PREFIX}${nonce}.local`);
    }

    async function localNetworkAccessState() {
        const detectState = window.ESPectreBrowserSupport?.localNetworkAccessState;
        return typeof detectState === 'function'
            ? detectState(globalThis.navigator) : 'unavailable';
    }

    async function localFetch(url, options) {
        if (await localNetworkAccessState() === 'denied') {
            throw new ESPectreDirectError(
                'Local network access is blocked for this site.',
                'local_network_denied'
            );
        }
        try {
            return await globalThis.fetch(url, {
                ...options,
                cache: 'no-store',
                targetAddressSpace: 'local'
            });
        } catch (error) {
            if (!options.signal?.aborted && await localNetworkAccessState() === 'denied') {
                throw new ESPectreDirectError(
                    'Local network access is blocked for this site.',
                    'local_network_denied'
                );
            }
            throw error;
        }
    }

    function parseObject(text) {
        let data;
        try { data = JSON.parse(text); } catch (_error) {
            throw new ESPectreDirectError('Direct payload must be valid JSON.', 'invalid_json');
        }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
            throw new ESPectreDirectError('Direct payload must be a JSON object.', 'invalid_envelope');
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
            || result.schema_version !== 2
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
                && peer.txt_version === 2 && peer.protocol_version === 1
                && peer.transport === 'http'
                && peer.path === REQUEST_PATH && peer.events === EVENTS_PATH
                && validText(peer.firmware, 48)
                && validText(peer.chip, 16, { token: true })
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
                return normalizeEndpoint(`http://${host}:${peer.port}${peer.path}`);
            });
            return Object.freeze({ ...peer, capabilities: [...capabilities], addresses: [...addresses], endpoints });
        });
        return Object.freeze({ ...result, devices });
    }

    class ESPectreDirectClient {
        static get VERSION() { return '2.0.0'; }
        static get ENVELOPE_VERSION() { return ENVELOPE_VERSION; }
        static get ENDPOINT_PATH() { return REQUEST_PATH; }
        static get EVENTS_PATH() { return EVENTS_PATH; }
        static get RAW_PATH() { return RAW_PATH; }
        static get MAX_FRAME_BYTES() { return MAX_REQUEST_BYTES; }
        static get MAX_REQUEST_FRAME_BYTES() { return MAX_REQUEST_BYTES; }
        static get MAX_RESPONSE_FRAME_BYTES() { return MAX_RESPONSE_BYTES; }
        static get EVENTS() { return EVENTS; }
        static normalizeEndpoint(value) { return normalizeEndpoint(value); }
        static createDiscoveryEndpoint(randomSource) { return createDiscoveryEndpoint(randomSource); }
        static validatePeerDiscoveryResult(value) { return validatePeerDiscoveryResult(value); }

        #endpoint;
        #listeners = new Map();
        #sequence = 0;
        #compatible = false;
        #capabilities = null;
        #connected = false;
        #closing = false;
        #eventController = null;
        #requestControllers = new Set();
        #rawSessionId = '';

        constructor(endpoint) { this.#endpoint = normalizeEndpoint(endpoint); }

        get endpoint() { return this.#endpoint; }
        get eventsEndpoint() { return endpointWithPath(this.#endpoint, EVENTS_PATH); }
        get rawEndpoint() { return endpointWithPath(this.#endpoint, RAW_PATH); }
        get rawSessionId() { return this.#rawSessionId; }
        get connected() { return this.#connected; }
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

        async connect({ timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
            if (this.#eventController) throw new ESPectreDirectError('Direct client is already active.');
            if (typeof globalThis.fetch !== 'function') {
                throw new ESPectreDirectError('Streaming fetch is not available in this browser.', 'unsupported');
            }
            this.#closing = false;
            const controller = new AbortController();
            this.#eventController = controller;
            const timer = setTimeout(() => controller.abort('connection timeout'), timeoutMs);
            let response;
            try {
                response = await localFetch(this.eventsEndpoint, {
                    method: 'GET',
                    headers: { Accept: 'text/event-stream' },
                    signal: controller.signal
                });
            } catch (error) {
                if (this.#eventController === controller) this.#eventController = null;
                if (error instanceof ESPectreDirectError) throw error;
                throw new ESPectreDirectError(
                    controller.signal.aborted ? 'Timed out connecting to the local device.' : `Direct HTTP connection failed: ${error.message}`,
                    controller.signal.aborted ? 'timeout' : 'connection_failed'
                );
            } finally {
                clearTimeout(timer);
            }
            if (!response.ok || !response.body || typeof response.body.getReader !== 'function') {
                this.#eventController = null;
                controller.abort();
                throw new ESPectreDirectError(`Direct event stream returned HTTP ${response.status}.`, 'connection_failed');
            }
            this.#connected = true;
            this.#emit('open');
            this.#pumpEvents(response.body.getReader(), controller);
        }

        async #pumpEvents(reader, controller) {
            const decoder = new TextDecoder();
            let buffer = '';
            try {
                while (!controller.signal.aborted) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let boundary;
                    while ((boundary = buffer.match(/\r?\n\r?\n/))) {
                        const block = buffer.slice(0, boundary.index);
                        buffer = buffer.slice(boundary.index + boundary[0].length);
                        this.#ingestSseBlock(block);
                    }
                }
            } catch (error) {
                if (!controller.signal.aborted) this.#emit('protocol-error', error);
            } finally {
                try { reader.releaseLock(); } catch (_error) { /* already released */ }
                if (this.#eventController !== controller) return;
                this.#eventController = null;
                const expected = this.#closing || controller.signal.aborted;
                this.#connected = false;
                this.#compatible = false;
                this.#capabilities = null;
                this.#emit('close', { code: 0, reason: expected ? 'client closed' : 'event stream ended', expected });
            }
        }

        #ingestSseBlock(block) {
            if (!block || block.startsWith(':')) return;
            let eventName = '';
            const dataLines = [];
            for (const line of block.split(/\r?\n/)) {
                if (line.startsWith('event:')) eventName = line.slice(6).trim();
                else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
            }
            if (!eventName || !dataLines.length) return;
            try {
                const text = dataLines.join('\n');
                if (new TextEncoder().encode(text).byteLength > MAX_RESPONSE_BYTES) {
                    throw new ESPectreDirectError('Direct event exceeds the 8192-byte limit.', 'frame_too_large');
                }
                const envelope = parseObject(text);
                if (envelope.type !== 'event' || envelope.event !== eventName
                    || !envelope.data || typeof envelope.data !== 'object' || Array.isArray(envelope.data)) {
                    throw new ESPectreDirectError('Direct SSE envelope is invalid.', 'invalid_envelope');
                }
                this.#emit('event', eventName, envelope.data);
            } catch (error) {
                this.#emit('protocol-error', error);
            }
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

        async request(method, params = {}, options = {}) {
            if (!this.connected) throw new ESPectreDirectError('Direct HTTP is not connected.', 'not_connected');
            return this.#sendRequest(method, params, options);
        }

        async #sendRequest(method, params = {}, {
            timeoutMs = DEFAULT_TIMEOUT_MS,
            requestId,
            allowBeforeHandshake = false
        } = {}) {
            if (typeof method !== 'string' || !/^[A-Za-z0-9_.-]{1,64}$/.test(method)) {
                throw new ESPectreDirectError('Direct method is invalid.', 'invalid_method');
            }
            if (!params || typeof params !== 'object' || Array.isArray(params)) {
                throw new ESPectreDirectError('Direct params must be an object.', 'invalid_params');
            }
            if (!allowBeforeHandshake && MUTATING_METHODS.has(method) && !this.#compatible) {
                throw new ESPectreDirectError('Complete the Direct capability handshake before changing the device.', 'handshake_required');
            }
            const id = requestId || `web-${Date.now().toString(36)}-${(++this.#sequence).toString(36)}`;
            if (!/^[A-Za-z0-9_.:-]{1,64}$/.test(id)) {
                throw new ESPectreDirectError('Direct request id is invalid.', 'invalid_request_id');
            }
            const payload = JSON.stringify({ v: ENVELOPE_VERSION, type: 'request', id, method, params });
            if (new TextEncoder().encode(payload).byteLength > MAX_REQUEST_BYTES) {
                throw new ESPectreDirectError('Direct request exceeds the 4096-byte limit.', 'frame_too_large');
            }
            const controller = new AbortController();
            this.#requestControllers.add(controller);
            const timer = setTimeout(() => controller.abort('request timeout'), timeoutMs);
            const headers = { Accept: 'application/json', 'Content-Type': 'application/json' };
            if (method === 'stop_raw_stream' && this.#rawSessionId) {
                headers.Authorization = `Bearer ${this.#rawSessionId}`;
            }
            let response;
            try {
                response = await localFetch(this.#endpoint, {
                    method: 'POST',
                    headers,
                    body: payload,
                    signal: controller.signal
                });
            } catch (error) {
                if (error instanceof ESPectreDirectError) throw error;
                throw new ESPectreDirectError(
                    controller.signal.aborted ? `Direct request ${method} timed out.` : `Direct HTTP request failed: ${error.message}`,
                    controller.signal.aborted ? 'timeout' : 'connection_failed'
                );
            } finally {
                clearTimeout(timer);
                this.#requestControllers.delete(controller);
            }
            if (!response.ok) {
                const detail = (await response.text()).slice(0, 256).trim();
                throw new ESPectreDirectError(detail || `Direct HTTP returned ${response.status}.`, `http_${response.status}`);
            }
            const text = await response.text();
            if (new TextEncoder().encode(text).byteLength > MAX_RESPONSE_BYTES) {
                throw new ESPectreDirectError('Direct response exceeds the 8192-byte limit.', 'frame_too_large');
            }
            const envelope = parseObject(text);
            if (envelope.type !== 'response' || envelope.id !== id || typeof envelope.ok !== 'boolean') {
                throw new ESPectreDirectError('Direct response envelope is invalid.', 'invalid_envelope');
            }
            if (!envelope.ok) {
                const code = typeof envelope.error?.code === 'string' ? envelope.error.code : 'device_error';
                const message = typeof envelope.error?.message === 'string' ? envelope.error.message : 'Device rejected the request.';
                throw new ESPectreDirectError(message, code);
            }
            if (!envelope.result || typeof envelope.result !== 'object' || Array.isArray(envelope.result)) {
                throw new ESPectreDirectError('Direct success result must be an object.', 'invalid_envelope');
            }
            const result = envelope.result.data ?? envelope.result;
            if (!result || typeof result !== 'object' || Array.isArray(result)) {
                throw new ESPectreDirectError('Direct response data must be an object.', 'invalid_envelope');
            }
            if (method === 'start_raw_stream' && /^[0-9a-f]{32}$/.test(result.session_id)) {
                this.#rawSessionId = result.session_id;
            } else if (method === 'stop_raw_stream') {
                this.#rawSessionId = '';
            }
            return result;
        }

        async discoverPeersBootstrap(options = {}) {
            return validatePeerDiscoveryResult(await this.#sendRequest('discover_peers', {}, {
                timeoutMs: PEER_DISCOVERY_TIMEOUT_MS,
                allowBeforeHandshake: true,
                ...options
            }));
        }

        close(_code = 0, _reason = 'client closed') {
            this.#closing = true;
            this.#connected = false;
            this.#compatible = false;
            this.#capabilities = null;
            this.#rawSessionId = '';
            this.#eventController?.abort('client closed');
            this.#eventController = null;
            for (const controller of this.#requestControllers) controller.abort('client closed');
            this.#requestControllers.clear();
        }
    }

    window.ESPectreDirectClient = ESPectreDirectClient;
    window.ESPectreDirectError = ESPectreDirectError;
})();
