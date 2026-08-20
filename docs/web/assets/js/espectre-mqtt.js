/*
 * ESPectre MQTT Protocol Client
 *
 * Dependency-free protocol layer for an MQTT.js-compatible transport. It owns
 * ESPectre topic construction, payload validation, discovery parsing, and
 * command-response correlation; broker connection policy remains with callers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const PROTOCOL_VERSION = '1.0';
    const DEFAULT_TOPIC_PREFIX = 'espectre/v1/devices';
    const EVENTS = Object.freeze(['message', 'protocol-error']);
    const JSON_SUFFIXES = Object.freeze(new Set([
        'telemetry',
        'status',
        'info',
        'stats',
        'commands/catalog',
        'commands/accepted',
        'commands/rejected',
        'ota/state'
    ]));

    class ESPectreMqttError extends Error {
        constructor(message) {
            super(message);
            this.name = 'ESPectreMqttError';
        }
    }

    function mqttUtf8(value) {
        if (typeof value === 'string') return value;
        if (value instanceof ArrayBuffer) return new TextDecoder().decode(new Uint8Array(value));
        if (ArrayBuffer.isView(value)) {
            return new TextDecoder().decode(
                new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
        }
        return String(value ?? '');
    }

    function normalizedTopicPrefix(value) {
        if (typeof value !== 'string') throw new ESPectreMqttError('topicPrefix must be a string');
        const prefix = value.trim().replace(/^\/+|\/+$/g, '');
        if (!prefix || /[\u0000+#]/.test(prefix)) {
            throw new ESPectreMqttError('topicPrefix must be non-empty and contain no MQTT wildcards');
        }
        return prefix;
    }

    function normalizedDeviceId(value, { optional = false } = {}) {
        if (optional && (value === undefined || value === null || value === '')) return '';
        if (typeof value !== 'string') throw new ESPectreMqttError('deviceId must be a string');
        const deviceId = value.trim().replace(/^\/+|\/+$/g, '');
        if (!deviceId || /[\u0000/+#]/.test(deviceId)) {
            throw new ESPectreMqttError('deviceId must be one topic segment without MQTT wildcards');
        }
        return deviceId;
    }

    function parseJsonObject(text) {
        let data;
        try {
            data = JSON.parse(text);
        } catch (error) {
            throw new ESPectreMqttError('payload must be valid JSON');
        }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
            throw new ESPectreMqttError('payload must be a JSON object');
        }
        if (data.protocol_version !== PROTOCOL_VERSION) {
            throw new ESPectreMqttError(
                data.protocol_version === undefined
                    ? 'payload is missing protocol_version'
                    : `unsupported protocol_version ${String(data.protocol_version)}`);
        }
        return data;
    }

    function requireMatchingDeviceId(data, deviceId) {
        if (typeof data.device_id !== 'string' || data.device_id !== deviceId) {
            throw new ESPectreMqttError('payload device_id must match the MQTT topic');
        }
        return data;
    }

    function parseTopic(topicPrefix, topic) {
        const prefix = normalizedTopicPrefix(topicPrefix);
        const topicName = mqttUtf8(topic);
        const root = prefix + '/';
        if (!topicName.startsWith(root)) return null;
        const remainder = topicName.slice(root.length);
        const separator = remainder.indexOf('/');
        if (separator <= 0 || separator === remainder.length - 1) return null;
        const deviceId = remainder.slice(0, separator);
        if (/[\u0000/+#]/.test(deviceId)) return null;
        return { topic: topicName, deviceId, suffix: remainder.slice(separator + 1) };
    }

    class ESPectreMqttClient {
        static get VERSION() { return '1.0.0'; }
        static get PROTOCOL_VERSION() { return PROTOCOL_VERSION; }
        static get DEFAULT_TOPIC_PREFIX() { return DEFAULT_TOPIC_PREFIX; }
        static get EVENTS() { return EVENTS; }

        static normalizeTopicPrefix(value) {
            return normalizedTopicPrefix(value);
        }

        static normalizeDeviceId(value) {
            return normalizedDeviceId(value);
        }

        static baseTopic(topicPrefix, deviceId) {
            return `${normalizedTopicPrefix(topicPrefix)}/${normalizedDeviceId(deviceId)}`;
        }

        static discoveryTopics(topicPrefix) {
            const prefix = normalizedTopicPrefix(topicPrefix);
            return [`${prefix}/+/info`, `${prefix}/+/status`];
        }

        static parseTopic(topicPrefix, topic) {
            return parseTopic(topicPrefix, topic);
        }

        static parseDiscoveryMessage(topicPrefix, topic, payload) {
            const parsed = parseTopic(topicPrefix, topic);
            if (!parsed || !['info', 'status'].includes(parsed.suffix)) return null;
            const text = mqttUtf8(payload).trim();
            if (!text) return null;
            const data = requireMatchingDeviceId(parseJsonObject(text), parsed.deviceId);
            return { ...parsed, text, data };
        }

        #mqttClient;
        #topicPrefix;
        #deviceId = '';
        #listeners = new Map();
        #pending = new Map();
        #commandSequence = 0;
        #closed = false;

        constructor(mqttClient, {
            topicPrefix = DEFAULT_TOPIC_PREFIX,
            deviceId = ''
        } = {}) {
            if (!mqttClient || typeof mqttClient.publish !== 'function') {
                throw new ESPectreMqttError('mqttClient must expose publish()');
            }
            this.#mqttClient = mqttClient;
            this.#topicPrefix = normalizedTopicPrefix(topicPrefix);
            this.#deviceId = normalizedDeviceId(deviceId, { optional: true });
        }

        get topicPrefix() { return this.#topicPrefix; }
        get deviceId() { return this.#deviceId; }
        get baseTopic() {
            return this.#deviceId ? `${this.#topicPrefix}/${this.#deviceId}` : '';
        }
        get subscriptionTopic() {
            return this.baseTopic ? this.baseTopic + '/#' : '';
        }

        setTopicPrefix(topicPrefix) {
            const next = normalizedTopicPrefix(topicPrefix);
            if (next !== this.#topicPrefix) this.#rejectAll('MQTT topic prefix changed.');
            this.#topicPrefix = next;
            return this.#topicPrefix;
        }

        setDevice(deviceId) {
            const next = normalizedDeviceId(deviceId, { optional: true });
            if (next !== this.#deviceId) this.#rejectAll('MQTT device selection changed.');
            this.#deviceId = next;
            return this.#deviceId;
        }

        on(event, handler) {
            if (!EVENTS.includes(event)) {
                throw new ESPectreMqttError(
                    `unknown event "${event}"; expected one of: ${EVENTS.join(', ')}`);
            }
            if (typeof handler !== 'function') throw new ESPectreMqttError('handler must be a function');
            if (!this.#listeners.has(event)) this.#listeners.set(event, new Set());
            this.#listeners.get(event).add(handler);
            return () => this.off(event, handler);
        }

        off(event, handler) {
            this.#listeners.get(event)?.delete(handler);
        }

        #emit(event, ...args) {
            const handlers = this.#listeners.get(event);
            if (!handlers) return;
            for (const handler of [...handlers]) {
                try {
                    handler(...args);
                } catch (error) {
                    console.error(`ESPectreMqttClient "${event}" handler failed:`, error);
                }
            }
        }

        ingest(topic, payload) {
            if (this.#closed || !this.#deviceId) return false;
            const parsed = parseTopic(this.#topicPrefix, topic);
            if (!parsed || parsed.deviceId !== this.#deviceId) return false;
            const text = mqttUtf8(payload).trim();
            if (!text) return false;

            let data = null;
            if (JSON_SUFFIXES.has(parsed.suffix)) {
                try {
                    data = requireMatchingDeviceId(parseJsonObject(text), parsed.deviceId);
                } catch (error) {
                    this.#emit('protocol-error', error, { ...parsed, text });
                    return false;
                }
            }

            if (parsed.suffix === 'commands/accepted' || parsed.suffix === 'commands/rejected') {
                this.#settleCommand(parsed.suffix, data);
            }
            this.#emit('message', { ...parsed, text, data });
            return true;
        }

        publishCommand(fields, { timeoutMs = 8000, commandId } = {}) {
            if (this.#closed) return Promise.reject(new ESPectreMqttError('MQTT protocol client is closed'));
            if (!this.baseTopic) return Promise.reject(new ESPectreMqttError('Select a device first'));
            if (!fields || typeof fields !== 'object' || Array.isArray(fields)
                    || typeof fields.command !== 'string' || !fields.command) {
                return Promise.reject(new ESPectreMqttError('command must be a non-empty string'));
            }
            if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
                return Promise.reject(new ESPectreMqttError('timeoutMs must be positive'));
            }
            const id = commandId || this.#nextCommandId();
            if (typeof id !== 'string' || !id || this.#pending.has(id)) {
                return Promise.reject(new ESPectreMqttError('commandId must be unique and non-empty'));
            }
            const payload = JSON.stringify({
                ...fields,
                protocol_version: PROTOCOL_VERSION,
                command_id: id
            });

            return new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    this.#pending.delete(id);
                    reject(new ESPectreMqttError('The device did not confirm the command in time.'));
                }, timeoutMs);
                this.#pending.set(id, { resolve, reject, timer, command: fields.command });
                try {
                    this.#mqttClient.publish(
                        this.baseTopic + '/commands/request',
                        payload,
                        { qos: 0, retain: false },
                        (error) => {
                            if (!error) return;
                            this.#rejectCommand(id, error);
                        }
                    );
                } catch (error) {
                    this.#rejectCommand(id, error);
                }
            });
        }

        hasPendingCommand(command) {
            for (const pending of this.#pending.values()) {
                if (pending.command === command) return true;
            }
            return false;
        }

        close(message = 'Broker connection closed.') {
            if (this.#closed) return;
            this.#closed = true;
            this.#rejectAll(message);
        }

        #nextCommandId() {
            this.#commandSequence = (this.#commandSequence + 1) % Number.MAX_SAFE_INTEGER;
            return `web-${Date.now()}-${this.#commandSequence.toString(16)}`;
        }

        #rejectCommand(commandId, error) {
            const pending = this.#pending.get(commandId);
            if (!pending) return;
            clearTimeout(pending.timer);
            this.#pending.delete(commandId);
            pending.reject(error);
        }

        #rejectAll(message) {
            for (const id of [...this.#pending.keys()]) {
                this.#rejectCommand(id, new ESPectreMqttError(message));
            }
        }

        #settleCommand(suffix, data) {
            if (!data || typeof data.command_id !== 'string') return;
            const pending = this.#pending.get(data.command_id);
            if (!pending) return;
            clearTimeout(pending.timer);
            this.#pending.delete(data.command_id);
            if (suffix === 'commands/accepted' && data.accepted !== false) {
                pending.resolve(data);
            } else {
                pending.reject(new ESPectreMqttError(
                    data.message || 'The device rejected the command.'));
            }
        }
    }

    window.ESPectreMqttClient = ESPectreMqttClient;
    window.ESPectreMqttError = ESPectreMqttError;
}());
