/*
 * ESPectre - Shared Web Bluetooth client
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

(function () {
    'use strict';

    const UUIDS = Object.freeze({
        service: 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0',
        telemetry: '119d5cac-48da-4bd9-bfc3-169805868258',
        sysinfo: 'c8c89ffa-c401-461f-9ffc-942fa04adfe3',
        control: '33ed9214-a8d7-40e8-82d1-c82747dcdc71'
    });

    class ESPectreBleClient {
        constructor(options = {}) {
            this.onTelemetry = options.onTelemetry || null;
            this.onInvalidTelemetry = options.onInvalidTelemetry || null;
            this.onSysinfoLine = options.onSysinfoLine || null;
            this.onSysinfoSnapshot = options.onSysinfoSnapshot || null;
            this.onDisconnected = options.onDisconnected || null;

            this.device = null;
            this.server = null;
            this.telemetryCharacteristic = null;
            this.sysinfoCharacteristic = null;
            this.controlCharacteristic = null;
            this.telemetryNotificationsActive = false;
            this.sysinfoNotificationsActive = false;
            this.disconnecting = false;
            this.sysinfoEntries = [];

            this.handleGattDisconnected = this.handleGattDisconnected.bind(this);
            this.handleTelemetryNotification = this.handleTelemetryNotification.bind(this);
            this.handleSysinfoNotification = this.handleSysinfoNotification.bind(this);
        }

        static get UUIDS() {
            return UUIDS;
        }

        static get supported() {
            return 'bluetooth' in navigator;
        }

        static parseTelemetry(value) {
            if (!value || value.byteLength < 8) return null;
            const movement = value.getFloat32(0, true);
            const threshold = value.getFloat32(4, true);
            if (!Number.isFinite(movement) || !Number.isFinite(threshold)) return null;
            return {
                movement,
                threshold,
                motionState: value.byteLength >= 9 ? value.getUint8(8) : null
            };
        }

        get connected() {
            return Boolean(this.server && this.server.connected);
        }

        get name() {
            return this.device ? (this.device.name || this.device.id) : '';
        }

        async connect({ telemetry = true, sysinfo = true } = {}) {
            if (!ESPectreBleClient.supported) {
                throw new Error('Web Bluetooth is not available in this browser.');
            }
            if (this.connected) return this.device;

            try {
                this.device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [UUIDS.service] }]
                });
                this.device.addEventListener('gattserverdisconnected', this.handleGattDisconnected);

                this.server = await this.device.gatt.connect();
                const service = await this.server.getPrimaryService(UUIDS.service);
                this.telemetryCharacteristic = await service.getCharacteristic(UUIDS.telemetry);
                this.sysinfoCharacteristic = await service.getCharacteristic(UUIDS.sysinfo);
                this.controlCharacteristic = await service.getCharacteristic(UUIDS.control);

                this.telemetryCharacteristic.addEventListener(
                    'characteristicvaluechanged',
                    this.handleTelemetryNotification
                );
                this.sysinfoCharacteristic.addEventListener(
                    'characteristicvaluechanged',
                    this.handleSysinfoNotification
                );

                await this.setTelemetryNotifications(telemetry);
                await this.setSysinfoNotifications(sysinfo);
                return this.device;
            } catch (error) {
                await this.disconnect();
                throw error;
            }
        }

        async setTelemetryNotifications(enabled) {
            if (!this.telemetryCharacteristic || enabled === this.telemetryNotificationsActive) return;
            if (enabled) {
                await this.telemetryCharacteristic.startNotifications();
            } else {
                await this.telemetryCharacteristic.stopNotifications();
            }
            this.telemetryNotificationsActive = enabled;
        }

        async setSysinfoNotifications(enabled) {
            if (!this.sysinfoCharacteristic || enabled === this.sysinfoNotificationsActive) return;
            if (enabled) {
                await this.sysinfoCharacteristic.startNotifications();
            } else {
                await this.sysinfoCharacteristic.stopNotifications();
            }
            this.sysinfoNotificationsActive = enabled;
        }

        clearSysinfoBuffer() {
            this.sysinfoEntries = [];
        }

        async writeControl(command) {
            if (!this.controlCharacteristic) {
                throw new Error('ESPectre is not connected.');
            }
            if (command === 'REQ_SYSINFO') {
                this.clearSysinfoBuffer();
            }
            const payload = new TextEncoder().encode(command);
            if (typeof this.controlCharacteristic.writeValueWithResponse === 'function') {
                await this.controlCharacteristic.writeValueWithResponse(payload);
            } else if (typeof this.controlCharacteristic.writeValueWithoutResponse === 'function') {
                await this.controlCharacteristic.writeValueWithoutResponse(payload);
            } else {
                await this.controlCharacteristic.writeValue(payload);
            }
        }

        handleTelemetryNotification(event) {
            const telemetry = ESPectreBleClient.parseTelemetry(event.target.value);
            if (telemetry) {
                if (this.onTelemetry) this.onTelemetry(telemetry);
                return;
            }
            if (this.onInvalidTelemetry) {
                const value = event.target.value;
                this.onInvalidTelemetry(value ? value.byteLength : 0);
            }
        }

        handleSysinfoNotification(event) {
            const value = event.target.value;
            if (!value) return;
            const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
            const line = new TextDecoder().decode(bytes).trim();
            if (!line) return;
            if (this.onSysinfoLine) this.onSysinfoLine(line);

            if (line === 'END') {
                const entries = this.sysinfoEntries;
                this.sysinfoEntries = [];
                if (this.onSysinfoSnapshot) {
                    this.onSysinfoSnapshot(Object.fromEntries(entries), entries);
                }
                return;
            }

            const separator = line.indexOf('=');
            if (separator <= 0) return;
            if (line.startsWith('proto_version=')) this.sysinfoEntries = [];
            this.sysinfoEntries.push([
                line.slice(0, separator).trim(),
                line.slice(separator + 1).trim()
            ]);
        }

        handleGattDisconnected() {
            if (this.disconnecting) return;
            this.clearConnectionState();
            if (this.onDisconnected) {
                Promise.resolve(this.onDisconnected()).catch((error) => console.error(error));
            }
        }

        async disconnect() {
            if (this.disconnecting) return;
            this.disconnecting = true;

            try {
                await this.stopNotifications();
                if (this.server && this.server.connected) this.server.disconnect();
            } finally {
                this.clearConnectionState();
                this.disconnecting = false;
            }
        }

        async stopNotifications() {
            if (this.telemetryCharacteristic && this.telemetryNotificationsActive) {
                try {
                    await this.telemetryCharacteristic.stopNotifications();
                } catch (error) {}
            }
            if (this.sysinfoCharacteristic && this.sysinfoNotificationsActive) {
                try {
                    await this.sysinfoCharacteristic.stopNotifications();
                } catch (error) {}
            }
        }

        clearConnectionState() {
            if (this.device) {
                this.device.removeEventListener('gattserverdisconnected', this.handleGattDisconnected);
            }
            if (this.telemetryCharacteristic) {
                this.telemetryCharacteristic.removeEventListener(
                    'characteristicvaluechanged',
                    this.handleTelemetryNotification
                );
            }
            if (this.sysinfoCharacteristic) {
                this.sysinfoCharacteristic.removeEventListener(
                    'characteristicvaluechanged',
                    this.handleSysinfoNotification
                );
            }

            this.device = null;
            this.server = null;
            this.telemetryCharacteristic = null;
            this.sysinfoCharacteristic = null;
            this.controlCharacteristic = null;
            this.telemetryNotificationsActive = false;
            this.sysinfoNotificationsActive = false;
            this.sysinfoEntries = [];
        }
    }

    window.ESPectreBleClient = ESPectreBleClient;
}());
