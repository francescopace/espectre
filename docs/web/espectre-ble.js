/*
 * ESPectre - Web Bluetooth client
 *
 * Standalone client for the ESPectre BLE surface documented in
 * docs/ESPECTRE_PROTOCOL.md: binary telemetry notifications, the streamed
 * sysinfo snapshot, and the text control commands.
 *
 * No dependencies. Web Bluetooth requires a Chromium-based browser and a
 * secure context (HTTPS or localhost); check `ESPectreBleClient.supported`
 * before connecting. The full API is documented in this directory's
 * README.md.
 *
 * Unlike the rest of this repository, this file is Apache-2.0 licensed so it
 * can be embedded in any web application, including proprietary ones.
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

(function () {
    'use strict';

    const UUIDS = Object.freeze({
        service: 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0',
        telemetry: '119d5cac-48da-4bd9-bfc3-169805868258',
        sysinfo: 'c8c89ffa-c401-461f-9ffc-942fa04adfe3',
        control: '33ed9214-a8d7-40e8-82d1-c82747dcdc71'
    });

    /**
     * Events emitted by the client. Subscribe with `on(event, handler)`.
     *
     * - `telemetry`         ({ movement, threshold, motionState }) per notification
     * - `invalid-telemetry` (byteLength) when a notification fails to parse
     * - `sysinfo-line`      (line) for every raw sysinfo line, including `END`
     * - `sysinfo`           (values, entries) when a snapshot completes
     * - `disconnect`        () on an unexpected GATT drop, never on `disconnect()`
     */
    const EVENTS = Object.freeze([
        'telemetry', 'invalid-telemetry', 'sysinfo', 'sysinfo-line', 'disconnect'
    ]);

    const BSSID_PATTERN = /^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$/;
    const DETECTORS = Object.freeze(['classic', 'ml']);
    const DEFAULT_TOPIC_PREFIX = 'espectre/v1/devices';

    /**
     * Thrown by the command builders when an argument cannot produce a valid
     * protocol command. Distinguishable from transport errors via
     * `error.name === 'ESPectreValidationError'`.
     */
    class ESPectreValidationError extends Error {
        constructor(message) {
            super(message);
            this.name = 'ESPectreValidationError';
        }
    }

    /* ------------------------------------------------------------ helpers */

    function encodeFields(fields) {
        return Object.entries(fields)
            .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value ?? ''))}`)
            .join('&');
    }

    function requireNonEmptyString(value, label) {
        if (typeof value !== 'string' || value.length === 0) {
            throw new ESPectreValidationError(`${label} must be a non-empty string`);
        }
        return value;
    }

    function requireIntegerInRange(value, min, max, label) {
        if (!Number.isInteger(value) || value < min || value > max) {
            throw new ESPectreValidationError(`${label} must be an integer between ${min} and ${max}`);
        }
        return value;
    }

    /* ------------------------------------------------------------- client */

    class ESPectreBleClient {
        /** Library version; independent from the device protocol version. */
        static get VERSION() { return '1.0.0'; }

        /** GATT service and characteristic UUIDs of the ESPectre BLE surface. */
        static get UUIDS() { return UUIDS; }

        /** Event names accepted by `on()`/`off()`. */
        static get EVENTS() { return EVENTS; }

        /** Whether this browser exposes Web Bluetooth. */
        static get supported() { return 'bluetooth' in navigator; }

        /**
         * Parses a telemetry notification payload.
         *
         * Layout (little-endian): float32 movement, float32 threshold, and an
         * optional trailing uint8 motion state on firmware that publishes it.
         *
         * @param {DataView} view - Characteristic value.
         * @returns {?{movement: number, threshold: number, motionState: ?number}}
         *   Parsed telemetry, or null when the payload is malformed.
         */
        static parseTelemetry(view) {
            if (!view || view.byteLength < 8) return null;
            const movement = view.getFloat32(0, true);
            const threshold = view.getFloat32(4, true);
            if (!Number.isFinite(movement) || !Number.isFinite(threshold)) return null;
            return {
                movement,
                threshold,
                motionState: view.byteLength >= 9 ? view.getUint8(8) : null
            };
        }

        /* ---------------------------------------------- command builders */
        /*
         * Builders are pure and validate locally, so callers can check
         * arguments without a connected device, and the wire format lives in
         * exactly one place. Each instance `set*` method writes the built
         * command over the control characteristic.
         */

        /**
         * @param {number} value - Motion threshold on the 0.0-1.0 probability
         *   scale shared by both detectors.
         * @returns {string}
         */
        static buildThresholdCommand(value) {
            if (!Number.isFinite(value) || value < 0 || value > 1) {
                throw new ESPectreValidationError('threshold must be a number between 0 and 1');
            }
            return `SET_THRESHOLD:${value.toFixed(6)}`;
        }

        /**
         * @param {string} detector - `classic` or `ml`.
         * @returns {string}
         */
        static buildDetectorCommand(detector) {
            if (!DETECTORS.includes(detector)) {
                throw new ESPectreValidationError(`detector must be one of: ${DETECTORS.join(', ')}`);
            }
            return `SET_DETECTOR:${detector}`;
        }

        /**
         * @param {object} options
         * @param {number} options.motionOnHits
         * @param {number} options.motionOffHits
         * @returns {string}
         */
        static buildMotionHitsCommand({ motionOnHits, motionOffHits } = {}) {
            requireIntegerInRange(motionOnHits, 1, 20, 'motionOnHits');
            requireIntegerInRange(motionOffHits, 1, 20, 'motionOffHits');
            return `SET_MOTION_HITS:on=${motionOnHits}&off=${motionOffHits}`;
        }

        /** @returns {string} */
        static buildOtaStatusCommand() {
            return 'OTA_STATUS';
        }

        /** @returns {string} */
        static buildOtaCheckCommand() {
            return 'OTA_CHECK';
        }

        /** @returns {string} */
        static buildOtaStartCommand() {
            return 'OTA_START';
        }

        /**
         * Builds the command that replaces the persisted Wi-Fi station block.
         * The password may be empty for open networks; whether the firmware
         * accepts that is a firmware policy, not a client one.
         *
         * @param {object} options
         * @param {string} options.ssid
         * @param {string} [options.password='']
         * @param {string} [options.bssid=''] - Empty, or `aa:bb:cc:dd:ee:ff`.
         * @param {number} [options.channel=0] - 0 (auto) to 14.
         * @returns {string}
         */
        static buildWifiConfigCommand({ ssid, password = '', bssid = '', channel = 0 } = {}) {
            requireNonEmptyString(ssid, 'ssid');
            requireIntegerInRange(channel, 0, 14, 'channel');
            if (bssid !== '' && !BSSID_PATTERN.test(bssid)) {
                throw new ESPectreValidationError('bssid must be empty or match aa:bb:cc:dd:ee:ff');
            }
            return 'SET_WIFI_CONFIG:' + encodeFields({ ssid, password, bssid, channel });
        }

        /**
         * Builds the command that replaces the persisted MQTT broker block.
         * Username and password may be empty for anonymous brokers.
         *
         * @param {object} options
         * @param {string} options.host
         * @param {number} options.port - 1 to 65535.
         * @param {string} [options.username='']
         * @param {string} [options.password='']
         * @param {string} [options.topicPrefix='espectre/v1/devices']
         * @returns {string}
         */
        static buildMqttConfigCommand({
            host, port, username = '', password = '', topicPrefix = DEFAULT_TOPIC_PREFIX
        } = {}) {
            requireNonEmptyString(host, 'host');
            requireIntegerInRange(port, 1, 65535, 'port');
            requireNonEmptyString(topicPrefix, 'topicPrefix');
            return 'SET_MQTT_CONFIG:' + encodeFields({
                host, port, username, password, topic_prefix: topicPrefix
            });
        }

        /**
         * Builds the command that sets the user-facing device label. The
         * label travels unencoded, matching the firmware parser; it may be
         * empty to clear the label alone.
         *
         * @param {string} label
         * @returns {string}
         */
        static buildDeviceLabelCommand(label) {
            if (typeof label !== 'string' || label.includes('\n')) {
                throw new ESPectreValidationError('label must be a single-line string');
            }
            return `SET_DEVICE_CONFIG:device_label=${label}`;
        }

        /* -------------------------------------------------------- state */

        #device = null;
        #server = null;
        #characteristics = { telemetry: null, sysinfo: null, control: null };
        #notificationsActive = { telemetry: false, sysinfo: false };
        #listeners = new Map();
        #sysinfoEntries = [];
        #connectPromise = null;
        #disconnecting = false;

        // Bound once so add/removeEventListener see the same references.
        #onGattDisconnected = () => this.#handleGattDisconnected();
        #onTelemetryNotification = (event) => this.#handleTelemetryNotification(event);
        #onSysinfoNotification = (event) => this.#handleSysinfoNotification(event);

        /** Whether a GATT connection is currently established. */
        get connected() {
            return Boolean(this.#server && this.#server.connected);
        }

        /** Advertised device name, or its id when the name is empty. */
        get name() {
            return this.#device ? (this.#device.name || this.#device.id) : '';
        }

        /** The underlying BluetoothDevice, or null when disconnected. */
        get device() {
            return this.#device;
        }

        /* ------------------------------------------------------- events */

        /**
         * Subscribes a handler to one of `ESPectreBleClient.EVENTS`.
         *
         * @param {string} event
         * @param {Function} handler
         * @returns {Function} Unsubscribe function for this registration.
         */
        on(event, handler) {
            if (!EVENTS.includes(event)) {
                throw new ESPectreValidationError(`unknown event "${event}"; expected one of: ${EVENTS.join(', ')}`);
            }
            if (typeof handler !== 'function') {
                throw new ESPectreValidationError('handler must be a function');
            }
            if (!this.#listeners.has(event)) this.#listeners.set(event, new Set());
            this.#listeners.get(event).add(handler);
            return () => this.off(event, handler);
        }

        /**
         * Removes a previously subscribed handler. Unknown pairs are ignored.
         *
         * @param {string} event
         * @param {Function} handler
         */
        off(event, handler) {
            const handlers = this.#listeners.get(event);
            if (handlers) handlers.delete(handler);
        }

        // A throwing handler must not starve the others or break the client.
        #emit(event, ...args) {
            const handlers = this.#listeners.get(event);
            if (!handlers) return;
            for (const handler of [...handlers]) {
                try {
                    handler(...args);
                } catch (error) {
                    console.error(`ESPectreBleClient "${event}" handler failed:`, error);
                }
            }
        }

        /* --------------------------------------------------- connection */

        /**
         * Opens the browser device chooser and connects. Safe to call while
         * already connected (returns the current device) or while a connect
         * is in flight (returns the pending promise).
         *
         * @param {object} [options]
         * @param {boolean} [options.telemetry=true] - Start telemetry notifications.
         * @param {boolean} [options.sysinfo=true] - Start sysinfo notifications.
         * @returns {Promise<BluetoothDevice>}
         */
        async connect({ telemetry = true, sysinfo = true } = {}) {
            if (!ESPectreBleClient.supported) {
                throw new Error('Web Bluetooth is not available in this browser.');
            }
            if (this.connected) return this.#device;
            if (this.#connectPromise) return this.#connectPromise;

            this.#connectPromise = this.#establish(telemetry, sysinfo)
                .finally(() => { this.#connectPromise = null; });
            return this.#connectPromise;
        }

        async #establish(telemetry, sysinfo) {
            try {
                this.#device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [UUIDS.service] }]
                });
                this.#device.addEventListener('gattserverdisconnected', this.#onGattDisconnected);

                this.#server = await this.#device.gatt.connect();
                const service = await this.#server.getPrimaryService(UUIDS.service);
                this.#characteristics.telemetry = await service.getCharacteristic(UUIDS.telemetry);
                this.#characteristics.sysinfo = await service.getCharacteristic(UUIDS.sysinfo);
                this.#characteristics.control = await service.getCharacteristic(UUIDS.control);

                this.#characteristics.telemetry.addEventListener(
                    'characteristicvaluechanged', this.#onTelemetryNotification);
                this.#characteristics.sysinfo.addEventListener(
                    'characteristicvaluechanged', this.#onSysinfoNotification);

                await this.setTelemetryNotifications(telemetry);
                await this.setSysinfoNotifications(sysinfo);
                return this.#device;
            } catch (error) {
                await this.disconnect();
                throw error;
            }
        }

        /**
         * Stops notifications and closes the GATT connection. Idempotent;
         * does not emit `disconnect` (that event is for unexpected drops).
         */
        async disconnect() {
            if (this.#disconnecting) return;
            this.#disconnecting = true;
            try {
                for (const kind of ['telemetry', 'sysinfo']) {
                    try {
                        await this.#setNotifications(kind, false);
                    } catch (error) {
                        // The link may already be gone; cleanup continues.
                    }
                }
                if (this.connected) this.#server.disconnect();
            } finally {
                this.#clearConnectionState();
                this.#disconnecting = false;
            }
        }

        /**
         * Enables or disables telemetry notifications without disconnecting,
         * for callers that only need telemetry while their view is visible.
         *
         * @param {boolean} enabled
         */
        setTelemetryNotifications(enabled) {
            return this.#setNotifications('telemetry', enabled);
        }

        /**
         * Enables or disables sysinfo notifications without disconnecting.
         *
         * @param {boolean} enabled
         */
        setSysinfoNotifications(enabled) {
            return this.#setNotifications('sysinfo', enabled);
        }

        async #setNotifications(kind, enabled) {
            const characteristic = this.#characteristics[kind];
            if (!characteristic || enabled === this.#notificationsActive[kind]) return;
            if (enabled) {
                await characteristic.startNotifications();
            } else {
                await characteristic.stopNotifications();
            }
            this.#notificationsActive[kind] = enabled;
        }

        /* ----------------------------------------------------- commands */

        /**
         * Writes a raw control command. Prefer the typed `set*` methods; this
         * is the escape hatch for commands the library does not model yet.
         *
         * @param {string} command
         */
        async writeControl(command) {
            const control = this.#characteristics.control;
            if (!control) {
                throw new Error('ESPectre is not connected.');
            }
            if (command === 'REQ_SYSINFO') {
                this.#sysinfoEntries = [];
            }
            const payload = new TextEncoder().encode(command);
            if (typeof control.writeValueWithResponse === 'function') {
                await control.writeValueWithResponse(payload);
            } else if (typeof control.writeValueWithoutResponse === 'function') {
                await control.writeValueWithoutResponse(payload);
            } else {
                await control.writeValue(payload);
            }
        }

        /** Requests a full sysinfo snapshot; resolves in a `sysinfo` event. */
        requestSysinfo() {
            return this.writeControl('REQ_SYSINFO');
        }

        /** @see ESPectreBleClient.buildThresholdCommand */
        setThreshold(value) {
            return this.writeControl(ESPectreBleClient.buildThresholdCommand(value));
        }

        /** @see ESPectreBleClient.buildDetectorCommand */
        setDetector(detector) {
            return this.writeControl(ESPectreBleClient.buildDetectorCommand(detector));
        }

        /** @see ESPectreBleClient.buildMotionHitsCommand */
        setMotionHits(options) {
            return this.writeControl(ESPectreBleClient.buildMotionHitsCommand(options));
        }

        /** @see ESPectreBleClient.buildWifiConfigCommand */
        setWifiConfig(options) {
            return this.writeControl(ESPectreBleClient.buildWifiConfigCommand(options));
        }

        /** Clears the persisted Wi-Fi station settings. */
        clearWifiConfig() {
            return this.writeControl('CLEAR_WIFI');
        }

        /** @see ESPectreBleClient.buildMqttConfigCommand */
        setMqttConfig(options) {
            return this.writeControl(ESPectreBleClient.buildMqttConfigCommand(options));
        }

        /** Clears the persisted MQTT broker settings. */
        clearMqttConfig() {
            return this.writeControl('CLEAR_MQTT_CONFIG');
        }

        /** @see ESPectreBleClient.buildDeviceLabelCommand */
        setDeviceLabel(label) {
            return this.writeControl(ESPectreBleClient.buildDeviceLabelCommand(label));
        }

        /** Resets device naming and MQTT settings, keeping the device id. */
        clearDeviceConfig() {
            return this.writeControl('CLEAR_DEVICE_CONFIG');
        }

        /** Requests the current OTA status snapshot over the existing sysinfo surface. */
        otaStatus() {
            return this.writeControl(ESPectreBleClient.buildOtaStatusCommand());
        }

        /** Starts an OTA manifest check using the firmware-embedded release URL. */
        otaCheck() {
            return this.writeControl(ESPectreBleClient.buildOtaCheckCommand());
        }

        /** Starts OTA using the firmware-embedded manifest and release image. */
        otaStart() {
            return this.writeControl(ESPectreBleClient.buildOtaStartCommand());
        }

        /* ------------------------------------------------ notifications */

        #handleTelemetryNotification(event) {
            const telemetry = ESPectreBleClient.parseTelemetry(event.target.value);
            if (telemetry) {
                this.#emit('telemetry', telemetry);
            } else {
                const value = event.target.value;
                this.#emit('invalid-telemetry', value ? value.byteLength : 0);
            }
        }

        /*
         * Sysinfo arrives as text lines. A snapshot starts at
         * `proto_version=` (which also recovers from a half-received one),
         * accumulates `key=value` pairs, and closes on `END`.
         */
        #handleSysinfoNotification(event) {
            const value = event.target.value;
            if (!value) return;
            const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
            const line = new TextDecoder().decode(bytes).trim();
            if (!line) return;
            this.#emit('sysinfo-line', line);

            if (line === 'END') {
                const entries = this.#sysinfoEntries;
                this.#sysinfoEntries = [];
                this.#emit('sysinfo', Object.fromEntries(entries), entries);
                return;
            }

            const separator = line.indexOf('=');
            if (separator <= 0) return;
            if (line.startsWith('proto_version=')) this.#sysinfoEntries = [];
            this.#sysinfoEntries.push([
                line.slice(0, separator).trim(),
                line.slice(separator + 1).trim()
            ]);
        }

        #handleGattDisconnected() {
            if (this.#disconnecting) return;
            this.#clearConnectionState();
            this.#emit('disconnect');
        }

        #clearConnectionState() {
            if (this.#device) {
                this.#device.removeEventListener('gattserverdisconnected', this.#onGattDisconnected);
            }
            if (this.#characteristics.telemetry) {
                this.#characteristics.telemetry.removeEventListener(
                    'characteristicvaluechanged', this.#onTelemetryNotification);
            }
            if (this.#characteristics.sysinfo) {
                this.#characteristics.sysinfo.removeEventListener(
                    'characteristicvaluechanged', this.#onSysinfoNotification);
            }
            this.#device = null;
            this.#server = null;
            this.#characteristics = { telemetry: null, sysinfo: null, control: null };
            this.#notificationsActive = { telemetry: false, sysinfo: false };
            this.#sysinfoEntries = [];
        }
    }

    window.ESPectreBleClient = ESPectreBleClient;
    window.ESPectreValidationError = ESPectreValidationError;
}());
