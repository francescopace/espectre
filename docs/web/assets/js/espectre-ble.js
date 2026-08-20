/*
 * ESPectre - Web Bluetooth client
 *
 * Standalone client for the ESPectre BLE setup surface documented in
 * docs/ESPECTRE_PROTOCOL.md: the streamed sysinfo snapshot and the text
 * control commands for Wi-Fi, MQTT, identity, and OTA.
 *
 * No dependencies. Web Bluetooth requires a Chromium-based browser and a
 * secure context (HTTPS or localhost); check `ESPectreBleClient.supported`
 * before connecting.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
     * - `sysinfo-line`      (line) for every raw sysinfo line, including `END`
     * - `sysinfo`           (values, entries) when a snapshot completes
     * - `disconnect`        () on an unexpected GATT drop, never on `disconnect()`
     */
    const EVENTS = Object.freeze([
        'sysinfo', 'sysinfo-line', 'disconnect'
    ]);

    const BSSID_PATTERN = /^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$/;
    const WIFI_BAND_POLICIES = Object.freeze(['2g', '5g', 'auto']);
    const OTA_CHANNELS = Object.freeze(['release', 'preview', 'develop']);
    const DEFAULT_TOPIC_PREFIX = 'espectre/v1/devices';
    const MAX_CONTROL_BYTES = 512;
    const MAX_SSID_BYTES = 32;
    const MAX_WIFI_PASSWORD_BYTES = 63;

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
        if (value.includes('\0')) {
            throw new ESPectreValidationError(`${label} must not contain NUL`);
        }
        return value;
    }

    function utf8Length(value) {
        return new TextEncoder().encode(value).byteLength;
    }

    function requireUtf8Length(value, min, max, label) {
        if (typeof value !== 'string' || value.includes('\0')) {
            throw new ESPectreValidationError(`${label} must be a string without NUL`);
        }
        const length = utf8Length(value);
        if (length < min || length > max) {
            throw new ESPectreValidationError(`${label} must be ${min}..${max} UTF-8 bytes`);
        }
        return value;
    }

    function requireControlCommand(command) {
        if (typeof command !== 'string' || command.length === 0 || command.includes('\0')) {
            throw new ESPectreValidationError('command must be a non-empty string without NUL');
        }
        if (utf8Length(command) > MAX_CONTROL_BYTES) {
            throw new ESPectreValidationError(
                `command must not exceed ${MAX_CONTROL_BYTES} UTF-8 bytes`);
        }
        return command;
    }

    function requireIntegerInRange(value, min, max, label) {
        if (!Number.isInteger(value) || value < min || value > max) {
            throw new ESPectreValidationError(`${label} must be an integer between ${min} and ${max}`);
        }
        return value;
    }

    /**
     * Accepts 0 (auto), a 2.4 GHz channel, or a 5 GHz 20 MHz channel center.
     * Whether the device can actually tune a 5 GHz channel depends on its
     * radio, so that check stays with the firmware.
     */
    function requireWifiChannel(value, label) {
        const is2g = Number.isInteger(value) && value >= 0 && value <= 14;
        const is5g = Number.isInteger(value) && (
            (((value >= 36 && value <= 64) || (value >= 100 && value <= 144)) && value % 4 === 0) ||
            (value >= 149 && value <= 177 && value % 4 === 1));
        if (!is2g && !is5g) {
            throw new ESPectreValidationError(
                `${label} must be 0..14, or a 5 GHz channel (36..64, 100..144, 149..177)`);
        }
        return value;
    }

    function requireWifiBandPolicy(value) {
        if (!WIFI_BAND_POLICIES.includes(value)) {
            throw new ESPectreValidationError(
                `bandPolicy must be one of: ${WIFI_BAND_POLICIES.join(', ')}`);
        }
        return value;
    }

    function requireOtaChannel(value) {
        if (!OTA_CHANNELS.includes(value)) {
            throw new ESPectreValidationError(
                `channel must be one of: ${OTA_CHANNELS.join(', ')}`);
        }
        return value;
    }

    function buildOtaActionCommand(verb, { channel } = {}) {
        if (channel === undefined || channel === '') return verb;
        requireOtaChannel(channel);
        return `${verb}:channel=${encodeURIComponent(channel)}`;
    }

    function requireChannelMatchesBandPolicy(channel, bandPolicy) {
        if (channel === 0 || bandPolicy === 'auto') return;
        const channelIs2g = channel <= 14;
        if ((bandPolicy === '2g' && !channelIs2g) || (bandPolicy === '5g' && channelIs2g)) {
            throw new ESPectreValidationError(`channel does not match the ${bandPolicy} band policy`);
        }
    }

    /* ------------------------------------------------------------- client */

    class ESPectreBleClient {
        /** Library version; independent from the device protocol version. */
        static get VERSION() { return '1.3.0'; }

        /** GATT service and characteristic UUIDs of the ESPectre BLE surface. */
        static get UUIDS() { return UUIDS; }

        /** Event names accepted by `on()`/`off()`. */
        static get EVENTS() { return EVENTS; }

        /** Whether this browser exposes Web Bluetooth. */
        static get supported() {
            return typeof navigator !== 'undefined'
                && navigator.bluetooth
                && typeof navigator.bluetooth.requestDevice === 'function';
        }

        /* ---------------------------------------------- command builders */
        /*
         * Builders are pure and validate locally, so callers can check
         * arguments without a connected device, and the wire format lives in
         * exactly one place. Each instance `set*` method writes the built
         * command over the control characteristic.
         */

        /** @returns {string} */
        static buildOtaStatusCommand() {
            return 'OTA_STATUS';
        }

        /**
         * @param {object} [options]
         * @param {string} [options.channel] - Optional `release`, `preview`, or `develop`.
         * @returns {string}
         */
        static buildOtaCheckCommand({ channel } = {}) {
            return buildOtaActionCommand('OTA_CHECK', { channel });
        }

        /**
         * @param {object} [options]
         * @param {string} [options.channel] - Optional `release`, `preview`, or `develop`.
         * @returns {string}
         */
        static buildOtaStartCommand({ channel } = {}) {
            return buildOtaActionCommand('OTA_START', { channel });
        }

        /** Stops BLE after Wi-Fi is configured so sensing can resume. */
        static buildStopBleCommand() {
            return 'STOP_BLE';
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
         * @param {number} [options.channel=0] - 0 (auto), 1..14, or a 5 GHz channel.
         * @param {string} [options.bandPolicy] - Optional `2g`, `5g`, or `auto` policy.
         * @returns {string}
         */
        static buildWifiConfigCommand({
            ssid, password = '', bssid = '', channel = 0, bandPolicy
        } = {}) {
            requireUtf8Length(ssid, 1, MAX_SSID_BYTES, 'ssid');
            requireUtf8Length(password, 0, MAX_WIFI_PASSWORD_BYTES, 'password');
            requireWifiChannel(channel, 'channel');
            if (bandPolicy !== undefined) {
                requireWifiBandPolicy(bandPolicy);
                requireChannelMatchesBandPolicy(channel, bandPolicy);
            }
            if (bssid !== '' && !BSSID_PATTERN.test(bssid)) {
                throw new ESPectreValidationError('bssid must be empty or match aa:bb:cc:dd:ee:ff');
            }
            const fields = { ssid, password, bssid, channel };
            if (bandPolicy !== undefined) fields.band_policy = bandPolicy;
            return requireControlCommand('SET_WIFI_CONFIG:' + encodeFields(fields));
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
            [host, username, password, topicPrefix].forEach((value) => {
                if (typeof value !== 'string' || value.includes('\0')) {
                    throw new ESPectreValidationError('MQTT fields must be strings without NUL');
                }
            });
            return requireControlCommand('SET_MQTT_CONFIG:' + encodeFields({
                host, port, username, password, topic_prefix: topicPrefix
            }));
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
            if (typeof label !== 'string' || /[\r\n\0]/.test(label)) {
                throw new ESPectreValidationError('label must be a single-line string');
            }
            return requireControlCommand(`SET_DEVICE_CONFIG:device_label=${label}`);
        }

        /* -------------------------------------------------------- state */

        #device = null;
        #server = null;
        #characteristics = { sysinfo: null, control: null };
        #notificationsActive = { sysinfo: false };
        #listeners = new Map();
        #sysinfoEntries = [];
        #sysinfoActive = false;
        #connectPromise = null;
        #disconnecting = false;
        #connectionRevision = 0;
        #writeChain = Promise.resolve();

        // Bound once so add/removeEventListener see the same references.
        #onGattDisconnected = () => this.#handleGattDisconnected();
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
         * @param {boolean} [options.sysinfo=true] - Start sysinfo notifications.
         * @returns {Promise<BluetoothDevice>}
         */
        async connect({ sysinfo = true } = {}) {
            if (!ESPectreBleClient.supported) {
                throw new Error('Web Bluetooth is not available in this browser.');
            }
            if (this.connected) return this.#device;
            if (this.#connectPromise) return this.#connectPromise;

            const revision = ++this.#connectionRevision;
            this.#connectPromise = this.#establish(sysinfo, revision)
                .finally(() => { this.#connectPromise = null; });
            return this.#connectPromise;
        }

        async #establish(sysinfo, revision) {
            let device = null;
            let server = null;
            try {
                device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [UUIDS.service] }]
                });
                this.#assertConnectionRevision(revision);
                this.#device = device;
                device.addEventListener('gattserverdisconnected', this.#onGattDisconnected);

                server = await device.gatt.connect();
                this.#assertConnectionRevision(revision, server);
                this.#server = server;
                const service = await server.getPrimaryService(UUIDS.service);
                this.#assertConnectionRevision(revision, server);
                this.#characteristics.sysinfo = await service.getCharacteristic(UUIDS.sysinfo);
                this.#assertConnectionRevision(revision, server);
                this.#characteristics.control = await service.getCharacteristic(UUIDS.control);
                this.#assertConnectionRevision(revision, server);

                this.#characteristics.sysinfo.addEventListener(
                    'characteristicvaluechanged', this.#onSysinfoNotification);

                await this.setSysinfoNotifications(sysinfo);
                this.#assertConnectionRevision(revision, server);
                return this.#device;
            } catch (error) {
                if (server?.connected) server.disconnect();
                this.#clearConnectionState();
                throw error;
            }
        }

        #assertConnectionRevision(revision, server = null) {
            if (revision === this.#connectionRevision && (!server || server.connected)) return;
            const cancelled = revision !== this.#connectionRevision;
            if (cancelled && server?.connected) server.disconnect();
            const error = new Error(cancelled
                ? 'Bluetooth connection attempt was cancelled.'
                : 'Bluetooth disconnected while the connection was being established.');
            error.name = cancelled ? 'AbortError' : 'NetworkError';
            throw error;
        }

        /**
         * Stops notifications and closes the GATT connection. Idempotent;
         * does not emit `disconnect` (that event is for unexpected drops).
         */
        async disconnect() {
            if (this.#disconnecting) return;
            this.#disconnecting = true;
            this.#connectionRevision += 1;
            try {
                for (const kind of ['sysinfo']) {
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
         * Enables or disables sysinfo notifications without disconnecting.
         *
         * @param {boolean} enabled
         */
        setSysinfoNotifications(enabled) {
            if (typeof enabled !== 'boolean') {
                throw new ESPectreValidationError('enabled must be a boolean');
            }
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
        writeControl(command) {
            requireControlCommand(command);
            const revision = this.#connectionRevision;
            const operation = this.#writeChain.then(() => {
                if (revision !== this.#connectionRevision) {
                    const error = new Error('Bluetooth connection changed before the command was written.');
                    error.name = 'AbortError';
                    throw error;
                }
                return this.#writeControlNow(command);
            });
            this.#writeChain = operation.catch(() => {});
            return operation;
        }

        async #writeControlNow(command) {
            const control = this.#characteristics.control;
            if (!control) {
                throw new Error('ESPectre is not connected.');
            }
            if (command === 'REQ_SYSINFO') {
                this.#sysinfoEntries = [];
                this.#sysinfoActive = false;
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

        /** Starts an OTA manifest check. Omit `channel` to use the firmware default. */
        otaCheck({ channel } = {}) {
            return this.writeControl(ESPectreBleClient.buildOtaCheckCommand({ channel }));
        }

        /** Starts OTA using the selected or firmware-default channel. */
        otaStart({ channel } = {}) {
            return this.writeControl(ESPectreBleClient.buildOtaStartCommand({ channel }));
        }

        /** Stops BLE after Wi-Fi is configured so sensing can resume. */
        stopBle() {
            return this.writeControl(ESPectreBleClient.buildStopBleCommand());
        }

        /* ------------------------------------------------ notifications */

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
                if (!this.#sysinfoActive || this.#sysinfoEntries.length === 0) return;
                const entries = this.#sysinfoEntries;
                this.#sysinfoEntries = [];
                this.#sysinfoActive = false;
                this.#emit('sysinfo', Object.fromEntries(entries), entries);
                return;
            }

            const separator = line.indexOf('=');
            if (separator <= 0) return;
            if (line.startsWith('proto_version=')) {
                this.#sysinfoEntries = [];
                this.#sysinfoActive = true;
            }
            if (!this.#sysinfoActive) return;
            this.#sysinfoEntries.push([
                line.slice(0, separator).trim(),
                line.slice(separator + 1).trim()
            ]);
        }

        #handleGattDisconnected() {
            if (this.#disconnecting) return;
            this.#connectionRevision += 1;
            this.#clearConnectionState();
            this.#emit('disconnect');
        }

        #clearConnectionState() {
            if (this.#device) {
                this.#device.removeEventListener('gattserverdisconnected', this.#onGattDisconnected);
            }
            if (this.#characteristics.sysinfo) {
                this.#characteristics.sysinfo.removeEventListener(
                    'characteristicvaluechanged', this.#onSysinfoNotification);
            }
            this.#device = null;
            this.#server = null;
            this.#characteristics = { sysinfo: null, control: null };
            this.#notificationsActive = { sysinfo: false };
            this.#sysinfoEntries = [];
            this.#sysinfoActive = false;
        }
    }

    window.ESPectreBleClient = ESPectreBleClient;
    window.ESPectreValidationError = ESPectreValidationError;
}());
