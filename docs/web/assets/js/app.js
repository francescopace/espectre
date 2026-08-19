/*
 * ESPectre - Website app shell
 *
 * Hash routing and a persistent session shared by every page. Configure
 * uses Web Bluetooth; Monitor uses MQTT over WebSockets for live sensing,
 * runtime controls, diagnostics, and recovery.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const routeRegistry = window.ESPectreRoutes;
    if (!routeRegistry) throw new Error('ESPectre route registry is unavailable');
    const browserSupport = window.ESPectreBrowserSupport && window.ESPectreBrowserSupport.current;
    if (!browserSupport) throw new Error('ESPectre browser capability policy is unavailable');

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    function mqttUtf8(value) {
        if (typeof value === 'string') return value;
        if (value == null) return '';
        if (value instanceof ArrayBuffer) return new TextDecoder().decode(value);
        if (ArrayBuffer.isView(value)) return new TextDecoder().decode(value);
        return String(value);
    }

    // analytics.js is optional: the app must work with it blocked or absent.
    const track = (name, params) => window.trackEvent ? window.trackEvent(name, params) : false;
    const errorType = (error) => (error && error.name) || 'Error';
    const activeToolName = () => routeRegistry.groupOf(route) === 'tools' ? route : 'monitor';
    const LEGACY_TOOL_ROUTES = Object.freeze({
        ble: 'configure',
        mqtt: 'monitor',
        device: 'configure'
    });
    const LOCAL_DEVELOPMENT_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);
    const MQTT_FORM_DEFAULTS = {
        host: 'homeassistant.local',
        port: '1883',
        username: 'mqtt',
        password: 'mqtt',
        topicPrefix: 'espectre/v1/devices'
    };

    const dependencyPromises = new Map();
    const browserDependencyPromises = new Map();

    function loadScriptOnce(src, { module = false } = {}) {
        if (dependencyPromises.has(src)) return dependencyPromises.get(src);
        const promise = new Promise((resolve, reject) => {
            const existing = document.querySelector(`script[src="${src}"]`);
            if (existing && existing.dataset.loaded === 'true') {
                resolve();
                return;
            }
            const script = existing || document.createElement('script');
            if (module) script.type = 'module';
            script.src = src;
            script.addEventListener('load', () => {
                script.dataset.loaded = 'true';
                resolve();
            }, { once: true });
            script.addEventListener('error', () => {
                script.remove();
                reject(new Error(`Unable to load ${src}`));
            }, { once: true });
            if (!existing) document.head.appendChild(script);
        });
        dependencyPromises.set(src, promise);
        promise.catch(() => dependencyPromises.delete(src));
        return promise;
    }

    function loadBrowserDependency(localSrc, developmentCdnSrc, options = {}) {
        if (browserDependencyPromises.has(localSrc)) {
            return browserDependencyPromises.get(localSrc);
        }
        const promise = loadScriptOnce(localSrc, options).catch((error) => {
            if (!LOCAL_DEVELOPMENT_HOSTS.has(location.hostname)) throw error;
            console.warn(`Local dependency unavailable; using development CDN fallback: ${developmentCdnSrc}`);
            return loadScriptOnce(developmentCdnSrc, options);
        });
        browserDependencyPromises.set(localSrc, promise);
        promise.catch(() => browserDependencyPromises.delete(localSrc));
        return promise;
    }

    /* ==================================================== shared connection */

    const conn = {
        mode: null,             // 'ble' | 'mqtt' | 'demo'
        status: 'disconnected', // disconnected | connecting | connected
        movement: 0,
        threshold: 0.5,
        motion: false,
        deviceName: '',
        deviceId: '',
        generatedName: '',
        deviceLabel: '',
        chip: '',
        firmwareVersion: '',
        deviceBannerSub: '—',
        connectedAt: 0,
        startedAt: 0,
        toolName: '',
        entryPoint: '',
        readyState: '',
        readyAt: 0,
        readyTracked: false
    };

    let bleClient = null;
    let suppressBleDisconnectTeardown = false;
    let demoTimer = null;
    let demoInputEnergy = 0;
    const demoPointer = { x: null, y: null, t: 0 };
    let route = 'home';
    let lastTrackedProfile = null;
    let wifiBandPolicyAvailable = false;
    let currentWifiBandPolicy = '2g';
    let otaUpdateAvailable = false;
    let otaBusy = false;
    let otaState = '';
    let otaMessage = '';
    let otaSupported = null;
    let otaActionPending = false;
    let otaCheckTransport = '';
    let otaModalReturnFocus = null;
    let otaPollTimer = null;
    let otaTracking = null;
    let otaAwaitingReconnect = false;
    let pendingConfigVerification = null;
    let activeDeviceView = 'live';
    let latestDeviceInfo = null;

    const sysinfoBoolean = (value) => value === true || value === 'true' || value === '1';

    function applyConfigureCapabilities(snapshot) {
        $$('[data-capability]').forEach((element) => {
            element.hidden = !sysinfoBoolean(snapshot[element.dataset.capability]);
        });
        $$('[data-capability-any]').forEach((element) => {
            const capabilities = element.dataset.capabilityAny.split(/\s+/).filter(Boolean);
            element.hidden = !capabilities.some((key) => sysinfoBoolean(snapshot[key]));
        });
    }

    function applyWifiBandOptions(snapshot) {
        const select = document.getElementById('cfg-wifi-band');
        if (!select) return;
        const supports5ghz = sysinfoBoolean(snapshot.supports_wifi_5ghz);
        const selected = snapshot.wifi_band_policy || '2g';
        select.replaceChildren(new Option('2.4 GHz', '2g'));
        if (supports5ghz) {
            select.add(new Option('5 GHz', '5g'));
            select.add(new Option('Automatic (2.4/5 GHz)', 'auto'));
        }
        select.disabled = select.options.length === 1;
        currentWifiBandPolicy = [...select.options].some((option) => option.value === selected)
            ? selected
            : '2g';
        select.value = currentWifiBandPolicy;
        wifiBandPolicyAvailable = snapshot.wifi_band_policy !== undefined;
    }

    /*
     * Both detection profiles emit a probability on an absolute 0..1 scale, so
     * the display maps the value directly. Scaling against the threshold
     * would saturate well before 1 and hide how far past it a reading is.
     */
    function energyFraction() {
        return Math.min(1, Math.max(0, conn.movement));
    }

    function setStatus(status) {
        conn.status = status;
        renderConnection();
    }

    function rememberConnectionOrigin() {
        conn.toolName = activeToolName();
        conn.entryPoint = route;
        conn.startedAt = Date.now();
        conn.readyState = '';
        conn.readyAt = 0;
        conn.readyTracked = false;
    }

    function connectionParams() {
        return {
            tool_name: conn.toolName || activeToolName(),
            entry_point: conn.entryPoint || route
        };
    }

    function connectionTransport() {
        if (conn.mode === 'mqtt') return 'mqtt_websocket';
        if (conn.mode === 'demo') return 'simulation';
        return 'bluetooth';
    }

    function connectionInputMode() {
        if (conn.mode === 'demo') return 'demo';
        if (conn.mode === 'mqtt') return 'mqtt';
        return 'bluetooth';
    }

    function hasLiveDetection() {
        return conn.mode === 'demo' && conn.status === 'connected'
            || (conn.status === 'connected' && conn.mode === 'mqtt');
    }

    function setDeviceView(view, { focus = false } = {}) {
        const targetRoute = view === 'connectivity' ? 'configure' : 'monitor';
        activeDeviceView = targetRoute === 'configure' ? 'connectivity' : 'live';
        if (route !== targetRoute) {
            location.hash = '#' + targetRoute;
            return;
        }
        const targetPanel = document.querySelector(
            `[data-page="${targetRoute}"] .device-view[data-device-view="${activeDeviceView}"]`
        );
        if (focus && targetPanel) {
            if (!targetPanel.hasAttribute('tabindex')) targetPanel.setAttribute('tabindex', '-1');
            targetPanel.focus({ preventScroll: true });
        }
        if (targetRoute === 'monitor') {
            monitorResizeChart();
            if (monitor.bleRequested) ensureBleOffForLive();
        }
        syncDiagnosticsPolling();
    }

    function applyCsiTrafficModeSelect(value) {
        const select = document.getElementById('sense-csi-mode');
        if (!select || !value) {
            return;
        }
        const normalized = value === 'pacing' ? 'external' : value;
        if (Array.from(select.options).some((option) => option.value === normalized)) {
            select.value = normalized;
        }
    }

    function applySensingSnapshot(snapshot) {
        const detection = snapshot.detection || {};
        const detector = snapshot.detector || detection.algorithm;
        const threshold = Number(snapshot.threshold ?? detection.threshold);
        const motionHits = String(snapshot.motion_hits || '').split('/');
        if (Number.isFinite(threshold)) {
            conn.threshold = threshold;
            syncThresholdControl(threshold);
        }
        if (detector) {
            document.getElementById('sense-detector').value = detector;
        }
        if (motionHits.length === 2) {
            document.getElementById('sense-motion-on').value = motionHits[0];
            document.getElementById('sense-motion-off').value = motionHits[1];
        }
        if (snapshot.motion_on_hits !== undefined) {
            document.getElementById('sense-motion-on').value = snapshot.motion_on_hits;
        }
        if (snapshot.motion_off_hits !== undefined) {
            document.getElementById('sense-motion-off').value = snapshot.motion_off_hits;
        }
        if (snapshot.csi_traffic_mode) {
            applyCsiTrafficModeSelect(snapshot.csi_traffic_mode);
        }
        if (snapshot.traffic_mode || snapshot.traffic_generator_mode) {
            document.getElementById('sense-generator-mode').value =
                snapshot.traffic_mode || snapshot.traffic_generator_mode;
        }
    }

    function syncSensingControls() {
        const detector = document.getElementById('sense-detector')?.value;
        $$('[data-mqtt-command]').forEach((panel) => {
            const supported = conn.mode === 'demo'
                || !monitor.commandCatalogReady
                || monitor.commands.has(panel.dataset.mqttCommand);
            const lightweightOnly = panel.dataset.mqttCommand === 'recalibrate' && detector !== 'lightweight';
            panel.hidden = !supported || lightweightOnly;
            panel.querySelectorAll('button, input, select').forEach((control) => {
                const calibrating = panel.dataset.mqttCommand === 'recalibrate' && monitor.calibrating;
                control.disabled = !supported || lightweightOnly || calibrating
                    || (conn.mode !== 'demo' && !hasLiveDetection());
            });
        });
    }

    function setCalibrationBusy(busy) {
        monitor.calibrating = !!busy;
        if (!monitor.calibrating && monitor.calibrationTimer) {
            clearTimeout(monitor.calibrationTimer);
            monitor.calibrationTimer = null;
        }
        const button = $('.js-sense-recalibrate');
        if (button) {
            button.textContent = monitor.calibrating ? 'Calibrating…' : 'Recalibrate';
            button.setAttribute('aria-busy', monitor.calibrating ? 'true' : 'false');
        }
        syncSensingControls();
    }

    function scheduleCalibrationIdle(delayMs) {
        clearTimeout(monitor.calibrationTimer);
        monitor.calibrationTimer = setTimeout(() => setCalibrationBusy(false), delayMs);
    }

    function applyDeviceIdentity(data) {
        if (!data || typeof data !== 'object') return;
        if (data.device_id) adoptDeviceId(data.device_id);
        if (data.device_name) conn.generatedName = data.device_name;
        if (data.device_label !== undefined) conn.deviceLabel = data.device_label;
        if (data.device_label || data.device_name) {
            conn.deviceName = data.device_label || data.device_name;
        }
        if (data.chip) conn.chip = String(data.chip).toUpperCase();
        if (data.firmware_version || data.version) {
            conn.firmwareVersion = data.firmware_version || data.version;
        }
    }

    function renderDeviceIdentity() {
        const write = (selector, value) => {
            $$(selector).forEach((el) => { el.textContent = value || '—'; });
        };
        write('.js-menu-chip', conn.chip);
        write('.js-menu-device-id', conn.deviceId);
        write('.js-menu-firmware', conn.firmwareVersion);
    }

    function formatDeviceIdentityLine(chip, deviceId, firmware) {
        const parts = [];
        if (chip) parts.push('Chip ' + chip);
        if (deviceId) parts.push('Device ID ' + deviceId);
        if (firmware) parts.push('Firmware ' + firmware);
        return parts.join(' · ');
    }

    function applyDeviceInfo(data) {
        if (!data || typeof data !== 'object') return;
        latestDeviceInfo = data;
        applyDeviceIdentity(data);
        conn.deviceName = data.device_label || data.device_name || conn.deviceName || 'ESPectre';
        const line = formatDeviceIdentityLine(
            data.chip && String(data.chip).toUpperCase(),
            data.device_id || conn.deviceId,
            data.firmware_version
        );
        if (line) conn.deviceBannerSub = line;
        applySensingSnapshot(data);
        renderConnection();
    }

    function adoptDeviceId(deviceId) {
        const next = String(deviceId || '').trim();
        if (!next) return;
        const previous = String(conn.deviceId || '').trim();
        const previousBound = String(monitor.boundDeviceId || '').trim();
        conn.deviceId = next;
        const monitorDevice = document.getElementById('mon-device');
        if (monitorDevice && monitorDevice.value.trim() !== next) {
            monitorDevice.value = next;
        }
        const switched = (previous && previous !== next) || (previousBound && previousBound !== next);
        if (!switched) return;
        monitor.handoffReady = false;
        resetMonitorLiveView();
        otaCheckTransport = '';
        if (monitorIsMqttLive() && previousBound && previousBound !== next) {
            monitorStopAll('device_changed');
        }
    }

    function markToolReady(readiness) {
        if (!conn.mode) return;
        if (!conn.readyState) conn.readyAt = Date.now();
        conn.readyState = readiness;
        if (conn.readyTracked) return;
        conn.readyTracked = track('tool_ready', {
            ...connectionParams(),
            transport: connectionTransport(),
            input_mode: connectionInputMode(),
            readiness,
            latency_ms: Math.max(0, conn.readyAt - (conn.startedAt || conn.connectedAt))
        });
    }

    function syncThresholdControl(threshold) {
        const input = document.getElementById('sense-threshold');
        if (!input || input === document.activeElement) return;
        if (Number(input.value) === threshold) return;
        input.value = String(threshold);
    }

    function applyLiveTelemetry(movement, threshold, motionState) {
        markToolReady('telemetry');
        conn.movement = movement;
        if (Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) {
            conn.threshold = threshold;
            syncThresholdControl(threshold);
        }
        conn.motion = motionState !== null && motionState !== undefined
            ? motionState === 1 || motionState === 'motion'
            : movement >= conn.threshold;
        renderTelemetry();
        gameOnTelemetry();
    }

    /* ------------------------------------------------------------ BLE mode */

    function makeBleClient() {
        const client = new window.ESPectreBleClient();
        client.on('sysinfo', (snapshot) => applySysinfo(snapshot));
        client.on('disconnect', () => {
            bleClient = null;
            if (suppressBleDisconnectTeardown || conn.mode === 'mqtt') return;
            teardownConnection('unexpected');
            toast('Device disconnected.');
        });
        return client;
    }

    async function connectBle() {
        if (bleClient || (conn.status !== 'disconnected' && conn.mode !== 'mqtt')) return;
        if (!browserSupport.bluetooth
                || !window.ESPectreBleClient || !window.ESPectreBleClient.supported) {
            track('tool_connection', {
                tool_name: activeToolName(), entry_point: route,
                transport: 'bluetooth', result: 'unsupported'
            });
            toast(bleUnsupportedMessage());
            return;
        }
        const returningFromMqtt = monitorIsMqttLive();
        if (!returningFromMqtt) {
            rememberConnectionOrigin();
        }
        track('tool_connection', {
            ...connectionParams(),
            transport: 'bluetooth', result: 'attempt'
        });
        try {
            bleClient = makeBleClient();
            if (!returningFromMqtt) setStatus('connecting');
            await bleClient.connect();
            conn.mode = 'ble';
            conn.deviceBannerSub = 'reading device info…';
            conn.connectedAt = Date.now();
            monitor.bleRequested = false;
            setStatus('connected');
            setDeviceView('connectivity');
            track('tool_connection', {
                ...connectionParams(),
                transport: 'bluetooth', result: 'success'
            });
            try {
                await bleClient.requestSysinfo();
            } catch (error) {
                console.warn('Sysinfo request failed:', error);
                startSilentOtaCheck();
            }
        } catch (error) {
            bleClient = null;
            if (returningFromMqtt) {
                conn.mode = 'mqtt';
                setStatus('connected');
            } else {
                setStatus('disconnected');
            }
            // The chooser being dismissed is a user choice, not a failure.
            const cancelled = error && error.name === 'NotFoundError';
            track('tool_connection', {
                ...connectionParams(),
                transport: 'bluetooth',
                result: cancelled ? 'cancelled' : 'failure',
                error_type: errorType(error)
            });
            conn.startedAt = 0;
            conn.toolName = '';
            conn.entryPoint = '';
            if (cancelled) {
                if (returningFromMqtt) {
                    setDeviceView('live', { focus: true });
                    renderConnection();
                }
                return;
            }
            toast(error && error.message ? error.message : 'Bluetooth connection failed.');
        }
    }

    function applyConfigureMqttToMonitor() {
        const host = document.getElementById('cfg-mqtt-host');
        const user = document.getElementById('cfg-mqtt-user');
        const pass = document.getElementById('cfg-mqtt-pass');
        const prefix = document.getElementById('cfg-topic-prefix');
        const device = document.getElementById('cfg-device-id');
        const monHost = document.getElementById('mon-host');
        const monUser = document.getElementById('mon-user');
        const monPass = document.getElementById('mon-pass');
        const monPrefix = document.getElementById('mon-topic-prefix');
        const monDevice = document.getElementById('mon-device');
        if (host && host.value.trim()) monHost.value = host.value.trim();
        if (user && user.value.trim()) monUser.value = user.value.trim();
        if (pass && pass.value) monPass.value = pass.value;
        if (prefix && prefix.value.trim()) {
            monPrefix.value = prefix.value.trim().replace(/\/+$/, '');
        }
        const deviceId = device
            ? (device.value || device.textContent || '').trim()
            : '';
        if (deviceId && deviceId !== '—') monDevice.value = deviceId;
        // Device MQTT stores the broker TCP port (1883). The browser keeps its WebSocket port, path, and TLS.
    }

    async function stopBleForDetection() {
        if (!bleClient) return true;
        suppressBleDisconnectTeardown = true;
        try {
            await bleClient.writeControl(window.ESPectreBleClient.buildStopBleCommand());
        } catch (error) {
            suppressBleDisconnectTeardown = false;
            console.warn('STOP_BLE failed:', error);
            toast('Setup could not close. Sensing remains paused.');
            return false;
        }
        const client = bleClient;
        bleClient = null;
        try {
            await client.disconnect();
        } catch (error) {
            console.warn(error);
        }
        suppressBleDisconnectTeardown = false;
        monitor.bleRequested = false;
        return true;
    }

    async function ensureBleOffForLive({ statusFn = () => {} } = {}) {
        if (!monitorIsMqttLive()) return true;
        if (monitor.commandCatalogReady && !monitor.commands.has('set_ble')) return true;
        try {
            await monitorPublishCommand({ command: 'set_ble', ble: 'off' }, {
                pendingMessage: 'Closing nearby Bluetooth setup…',
                statusFn
            });
            monitor.bleRequested = false;
        } catch (error) {
            console.warn('MQTT set_ble off failed:', error);
        }
        return true;
    }

    function bindMqttToConnection() {
        if (conn.mode === 'demo') return;
        const device = document.getElementById('mon-device').value.trim();
        if (conn.status !== 'connected') {
            if (!conn.startedAt) rememberConnectionOrigin();
            conn.deviceName = conn.deviceName || device || 'ESPectre';
            if (!conn.deviceBannerSub || conn.deviceBannerSub === '—') {
                conn.deviceBannerSub = 'MQTT live';
            }
            conn.connectedAt = Date.now();
        }
        conn.mode = 'mqtt';
        monitor.closingBleForLive = false;
        setStatus('connected');
        setDeviceView('live', { focus: true });
        toast('Sensing is live.');
    }

    async function startDetection() {
        if (conn.mode === 'demo') {
            setDeviceView('live');
            return;
        }
        applyConfigureMqttToMonitor();
        const nextDevice = document.getElementById('mon-device').value.trim();
        if (conn.mode === 'mqtt' && monitorIsMqttLive()
                && (!nextDevice || nextDevice === monitor.boundDeviceId)) {
            setDeviceView('live');
            return;
        }
        const host = document.getElementById('mon-host').value.trim();
        if (!host) {
            toast('Save MQTT settings before starting sensing.');
            location.hash = '#monitor';
            return;
        }
        if (nextDevice) {
            if (conn.status === 'disconnected') {
                rememberConnectionOrigin();
                setStatus('connecting');
            }
            monitor.closingBleForLive = true;
            setDeviceView('live');
        } else if (route !== 'monitor') {
            location.hash = '#monitor';
        }
        await monitorConnect();
    }

    function applySysinfo(snapshot) {
        if (conn.mode === 'ble' && conn.toolName === 'configure'
                && (snapshot.frontend || snapshot.chip || snapshot.proto_version)) {
            markToolReady('sysinfo');
        }
        applyConfigureCapabilities(snapshot);
        applyWifiBandOptions(snapshot);
        const chip = (snapshot.chip || '').toUpperCase();
        const proto = snapshot.proto_version || snapshot.espectre_protocol_version || '';
        const firmware = snapshot.firmware_version || snapshot.version || '';
        const deviceIdentity = formatDeviceIdentityLine(chip, snapshot.device_id || conn.deviceId, firmware) || '—';
        conn.chip = chip;
        conn.firmwareVersion = firmware;
        conn.deviceBannerSub = deviceIdentity;

        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== '') {
                if (el.tagName === 'INPUT' || el.tagName === 'SELECT') el.value = value;
                else el.textContent = value;
            }
        };
        const setConnectionDot = (dotSelector, value) => {
            if (value === undefined) return;
            const connected = sysinfoBoolean(value);
            const dot = $(dotSelector);
            dot.classList.toggle('dot-idle', false);
            dot.classList.toggle('dot-ok', connected);
            dot.classList.toggle('dot-error', !connected);
            dot.title = connected ? 'Connected' : 'Disconnected';
        };
        set('cfg-ssid', snapshot.wifi_ssid);
        set('cfg-bssid', snapshot.wifi_bssid);
        set('cfg-channel', snapshot.wifi_channel);
        if (snapshot.mqtt_host) {
            set('cfg-mqtt-host', snapshot.mqtt_host);
            set('cfg-mqtt-port', snapshot.mqtt_port);
            const mqttUser = document.getElementById('cfg-mqtt-user');
            if (mqttUser) mqttUser.value = snapshot.mqtt_username || '';
            set('cfg-topic-prefix', snapshot.topic_prefix || MQTT_FORM_DEFAULTS.topicPrefix);
            const mqttPass = document.getElementById('cfg-mqtt-pass');
            if (mqttPass) mqttPass.value = '';
        }
        set('cfg-device-id', snapshot.device_id);
        set('cfg-device-name', snapshot.device_name);
        set('cfg-label', snapshot.device_label);
        applyDeviceIdentity(snapshot);
        if (snapshot.supports_ota !== undefined) otaSupported = sysinfoBoolean(snapshot.supports_ota);
        applySensingSnapshot(snapshot);
        applyOtaStatus({
            state: snapshot.ota_state,
            current_version: snapshot.ota_current_version,
            update_available: snapshot.ota_update_available,
            busy: snapshot.ota_busy,
            target_version: snapshot.ota_target_version,
            channel: snapshot.ota_channel,
            message: snapshot.ota_message
        });
        evaluateConfigVerification(snapshot);
        setConnectionDot('.js-wifi-status-dot', snapshot.wifi_connected);
        setConnectionDot('.js-mqtt-status-dot', snapshot.mqtt_connected);
        if (conn.mode === 'ble') startSilentOtaCheck();

        // Real hardware only: demo values would pollute the adoption report.
        if (conn.mode === 'ble' && snapshot.frontend && snapshot.chip) {
            const profile = snapshot.frontend + ':' + snapshot.chip;
            if (profile !== lastTrackedProfile) {
                const reported = track('device_profile', {
                    ...connectionParams(),
                    frontend: snapshot.frontend.toLowerCase(),
                    chip: snapshot.chip.toLowerCase(),
                    detector: String(snapshot.detector || 'unknown').toLowerCase(),
                    protocol_version: String(proto || 'unknown'),
                    firmware_version: String(snapshot.firmware_version || snapshot.version || 'unknown')
                });
                if (reported) lastTrackedProfile = profile;
            }
        }
        renderConnection();
    }

    /* ----------------------------------------------------------- demo mode */

    function connectDemo() {
        if (conn.status !== 'disconnected') return;
        rememberConnectionOrigin();
        track('tool_demo_start', connectionParams());
        setStatus('connecting');
        setTimeout(() => {
            conn.mode = 'demo';
            conn.deviceName = 'Demo Device';
            conn.deviceBannerSub = '—';
            conn.threshold = 0.5;
            conn.movement = 0.04;
            conn.connectedAt = Date.now();
            setStatus('connected');
            markToolReady('telemetry');
            monitor.commands = new Set([
                'set_threshold', 'set_motion_hits', 'set_detector', 'recalibrate',
                'set_csi_traffic_mode', 'set_traffic_generator_mode', 'stats',
                'ota_status', 'ota_check', 'set_ble'
            ]);
            monitor.commandCatalogReady = true;
            applySysinfo({
                chip: 'esp32-c5',
                frontend: 'native',
                proto_version: '1.0',
                firmware_version: '3.0.0-dev',
                supports_wifi_provisioning: 'true',
                supports_mqtt_config: 'true',
                supports_device_config: 'true',
                supports_extended_diagnostics: 'true',
                supports_ota: 'true',
                supports_wifi_5ghz: 'true',
                detector: 'lightweight',
                threshold: '0.500000',
                window: '100',
                lowpass: 'off',
                hampel: 'on',
                hampel_window: '5',
                hampel_threshold: '3.0',
                csi_traffic_mode: 'internal',
                traffic_mode: 'ping',
                csi_target_pps: '98',
                publish_interval_ms: '1000',
                evaluation_interval_ms: '250',
                wifi_connected: 'true',
                wifi_band_policy: '2g',
                mqtt_connected: 'true',
                wifi_ssid: 'HomeNet-5G',
                mqtt_host: 'homeassistant.local',
                mqtt_port: '1883',
                mqtt_username: 'mqtt',
                topic_prefix: 'espectre/v1/devices',
                device_id: '0x00007c2c6742bbac',
                device_name: 'Demo Device',
                device_label: 'Demo Device',
                motion_hits: '4/3',
                ota_state: 'up_to_date',
                ota_busy: 'false',
                ota_update_available: 'false',
                ota_current_version: '3.0.0-dev',
                ota_target_version: '',
                ota_message: ''
            });
            if (route !== 'game' && route !== 'theremin') setDeviceView('live');
            monitorResetChart();
            let t = 0;
            demoTimer = setInterval(() => {
                t += 0.16;
                const gameDemoActive = route === 'game' && game.phase !== 'idle' && game.phase !== 'done';
                const idle = 0.035 + Math.sin(t * 0.8) * 0.01 + Math.sin(t * 1.9) * 0.004;
                const target = Math.min(1, idle + demoInputEnergy * 0.95);
                const smoothing = gameDemoActive ? 0.42 : 0.28;
                conn.movement += (target - conn.movement) * smoothing;
                conn.movement = Math.max(0.01, Math.min(1, conn.movement));
                demoInputEnergy *= gameDemoActive ? 0.62 : 0.72;
                if (demoInputEnergy < 0.01) demoInputEnergy = 0;
                monitorFeed(
                    conn.movement,
                    conn.threshold,
                    conn.movement >= conn.threshold ? 'motion' : 'idle'
                );
                applyLiveTelemetry(conn.movement, conn.threshold, conn.movement >= conn.threshold ? 1 : 0);
            }, 160);
        }, 600);
    }

    function demoTrackMouse(event) {
        if (conn.mode !== 'demo') return;
        const now = performance.now();
        if (demoPointer.x !== null) {
            const dt = Math.max(16, now - demoPointer.t);
            const dx = event.clientX - demoPointer.x;
            const dy = event.clientY - demoPointer.y;
            const pxPerSecond = Math.hypot(dx, dy) * 1000 / dt;
            const normalized = Math.min(1, pxPerSecond / 1800);
            demoInputEnergy = Math.max(demoInputEnergy, normalized);
        }
        demoPointer.x = event.clientX;
        demoPointer.y = event.clientY;
        demoPointer.t = now;
    }

    function demoResetMotion() {
        if (conn.mode !== 'demo') return;
        demoInputEnergy = 0;
        conn.movement = 0.04;
        conn.motion = false;
        renderTelemetry();
    }

    /* ----------------------------------------------------- shared teardown */

    function disconnect() {
        suppressBleDisconnectTeardown = true;
        if (bleClient) {
            const client = bleClient;
            bleClient = null;
            client.disconnect().catch((error) => console.warn(error));
        }
        teardownConnection('user');
        suppressBleDisconnectTeardown = false;
    }

    function stopMqttTransport() {
        const client = monitor.client;
        monitor.client = null;
        monitor.closing = true;
        if (client) client.end(true);
        monitor.closing = false;
        monitor.baseTopic = null;
        monitor.pendingCommands.forEach((pending) => {
            clearTimeout(pending.timer);
            pending.reject(new Error('Broker connection closed.'));
        });
        monitor.pendingCommands.clear();
        monitor.commands.clear();
        monitor.commandCatalogReady = false;
        monitor.bleRequested = false;
        monitor.handoffReady = false;
        monitor.boundDeviceId = '';
        if (monitor.discoveryTimer) {
            clearTimeout(monitor.discoveryTimer);
            monitor.discoveryTimer = 0;
        }
        monitor.discoveryActive = false;
        monitor.discoveredDevices = {};
        monitor.discoveryPrefix = '';
        monitor.discoveryTopics = [];
        monitor.brokerUrl = '';
        resetMonitorDevicePicker();
        setCalibrationBusy(false);
        stopDiagnosticsPolling();
        resetMonitorLiveView();
        clearInterval(monitor.demoTimer);
        monitor.demoTimer = null;
        monitor.startedAt = 0;
        monitor.connectedAt = 0;
        monitor.entryPoint = '';
        monitor.inputMode = null;
        monitor.readyState = '';
        monitor.readyAt = 0;
        monitor.readyTracked = false;
        syncMonitorDemoButton();
    }

    function teardownConnection(reason = 'route_change') {
        monitor.closingBleForLive = false;
        if (otaTracking) finishOtaTracking('unconfirmed', 'ClientDisconnected');
        if (pendingConfigVerification) {
            finishConfigVerification('unconfirmed', 'ClientDisconnected');
        }
        reportGameAbandon('disconnect');
        const previousMode = conn.mode;
        const durationSeconds = conn.connectedAt
            ? Math.max(0, Math.round((Date.now() - conn.connectedAt) / 1000))
            : 0;
        if (previousMode) {
            track('tool_disconnect', {
                ...connectionParams(),
                transport: previousMode === 'mqtt' ? 'mqtt_websocket'
                    : previousMode === 'demo' ? 'simulation' : 'bluetooth',
                input_mode: previousMode === 'demo' ? 'demo'
                    : previousMode === 'mqtt' ? 'mqtt' : 'bluetooth',
                reason,
                duration_seconds: durationSeconds
            });
        }
        clearInterval(demoTimer);
        demoTimer = null;
        bleClient = null;
        demoInputEnergy = 0;
        demoPointer.x = null;
        demoPointer.y = null;
        demoPointer.t = 0;
        stopMqttTransport();
        conn.mode = null;
        conn.movement = 0;
        conn.motion = false;
        conn.deviceName = '';
        conn.deviceId = '';
        conn.generatedName = '';
        conn.deviceLabel = '';
        conn.chip = '';
        conn.firmwareVersion = '';
        conn.deviceBannerSub = '—';
        conn.connectedAt = 0;
        conn.startedAt = 0;
        conn.toolName = '';
        conn.entryPoint = '';
        conn.readyState = '';
        conn.readyAt = 0;
        conn.readyTracked = false;
        lastTrackedProfile = null;
        otaUpdateAvailable = false;
        otaBusy = false;
        otaState = '';
        otaMessage = '';
        otaSupported = null;
        otaCheckTransport = '';
        otaAwaitingReconnect = false;
        syncFirmwareUpdateNotice();
        gameReset();
        thereminStop();
        otaClose(false);
        setStatus('disconnected');
    }

    /* ----------------------------------------------------------- rendering */

    let dropdownOpen = false;

    function bleUnsupportedMessage() {
        if (browserSupport.ios) {
            return 'Bluetooth connection is unavailable on iPhone and iPad. Use the demo, desktop Chrome or Edge, or Chrome on Android.';
        }
        return 'Web Bluetooth is unavailable in this browser. Use the demo, desktop Chrome or Edge, or Chrome on Android.';
    }

    function flashUnsupportedMessage() {
        if (browserSupport.mobile) {
            return 'USB flashing is not available on mobile. Use desktop Chrome or Edge.';
        }
        return 'USB flashing is not available in this browser. Use desktop Chrome or Edge.';
    }

    function renderBrowserSupport() {
        const bleConnecting = conn.status === 'connecting'
            && !!bleClient
            && !bleClient.connected;
        $$('.js-connect-ble').forEach((button) => {
            button.disabled = !browserSupport.bluetooth || bleConnecting;
            button.setAttribute('aria-disabled', String(button.disabled));
            button.title = browserSupport.bluetooth ? '' : bleUnsupportedMessage();
            const label = button.querySelector('.js-connect-label');
            if (label) {
                if (!label.dataset.supportedLabel) label.dataset.supportedLabel = label.textContent;
                label.textContent = !browserSupport.bluetooth
                    ? 'Bluetooth unavailable'
                    : bleConnecting ? 'Connecting…' : label.dataset.supportedLabel;
            }
        });
        $$('.js-ble-chip').forEach((chip) => {
            chip.classList.toggle('unavailable', !browserSupport.bluetooth);
            if (!browserSupport.bluetooth) {
                chip.classList.remove('ready');
                chip.textContent = 'BLE · UNAVAILABLE';
            }
        });
        $$('.js-ble-support').forEach((notice) => {
            notice.hidden = browserSupport.bluetooth;
            notice.textContent = browserSupport.bluetooth ? '' : bleUnsupportedMessage();
        });

        $$('.js-flash-chip').forEach((chip) => {
            chip.classList.toggle('unavailable', !browserSupport.flash);
            chip.textContent = browserSupport.flash
                ? 'WEB SERIAL'
                : browserSupport.mobile ? 'DESKTOP ONLY' : 'UNAVAILABLE';
        });
        const flashRequirement = $('.js-flash-requirement');
        if (flashRequirement) {
            flashRequirement.classList.toggle('is-error', !browserSupport.flash);
            flashRequirement.textContent = browserSupport.flash
                ? 'USB flashing requires desktop Chrome or Edge.'
                : flashUnsupportedMessage();
        }
        const installTrigger = $('.js-flash-install [slot="activate"]');
        const installButton = $('.js-flash-install');
        if (installTrigger) {
            installTrigger.disabled = !browserSupport.flash;
            installTrigger.setAttribute('aria-disabled', String(!browserSupport.flash));
            installTrigger.title = browserSupport.flash ? '' : flashUnsupportedMessage();
        }
        if (installButton) {
            installButton.classList.toggle('is-disabled', !browserSupport.flash);
            installButton.toggleAttribute('inert', !browserSupport.flash);
        }
    }

    function renderConnection() {
        const connected = conn.status === 'connected';
        const live = hasLiveDetection();
        const bleConnecting = conn.status === 'connecting'
            && !!bleClient
            && !bleClient.connected;
        const mqttConnecting = conn.status === 'connecting' && !bleConnecting;
        const bleSetup = connected && conn.mode === 'ble';
        const mqttSession = live || mqttConnecting || monitor.closingBleForLive;

        $('.js-conn-disconnected').hidden = conn.status !== 'disconnected';
        $('.js-conn-connecting').hidden = conn.status !== 'connecting';
        $('.js-conn-connected').hidden = !connected;
        $('.js-dropdown').hidden = !(connected && dropdownOpen);
        $('.js-dropdown-toggle').setAttribute('aria-expanded', String(connected && dropdownOpen));
        $('.js-demo-tag').hidden = conn.mode !== 'demo';
        const setupTag = $('.js-setup-tag');
        if (setupTag) setupTag.hidden = conn.mode !== 'ble';

        $('.js-demo-connected').hidden = !live;
        $$('.js-demo-disconnected').forEach((el) => { el.hidden = live; });
        $$('.js-needs-conn').forEach((el) => { el.hidden = connected; });
        $$('.js-has-conn').forEach((el) => { el.hidden = !connected; });
        $$('.js-needs-live').forEach((el) => { el.hidden = live; });
        $$('.js-has-live').forEach((el) => { el.hidden = !live; });
        const showLiveEnergy = live;
        $$('.js-live-energy').forEach((el) => { el.hidden = !showLiveEnergy; });
        const paused = $('.js-sensing-paused');
        if (paused) paused.hidden = conn.mode !== 'ble';
        const configureOnboarding = $('.js-configure-onboarding');
        const configureWorkspace = $('.js-configure-workspace');
        const monitorOnboarding = $('.js-monitor-onboarding');
        const monitorWorkspace = $('.js-monitor-workspace');
        const connectivitySetup = $('.js-connectivity-setup');
        const setupNote = $('.js-setup-mode-note');
        const edit = $('.js-device-edit-connectivity');
        const startSensing = document.querySelector('[data-page="configure"] .js-start-detection');
        if (configureOnboarding) configureOnboarding.hidden = bleSetup || conn.mode === 'demo';
        if (configureWorkspace) configureWorkspace.hidden = !(bleSetup || conn.mode === 'demo');
        if (monitorOnboarding) monitorOnboarding.hidden = mqttSession;
        if (monitorWorkspace) monitorWorkspace.hidden = !mqttSession;
        if (connectivitySetup) connectivitySetup.hidden = !(bleSetup || conn.mode === 'demo');
        if (setupNote) setupNote.hidden = !bleSetup;
        if (startSensing) startSensing.disabled = monitor.closingBleForLive;
        if (edit) {
            edit.hidden = false;
            edit.disabled = monitor.closingBleForLive;
        }

        $$('.js-device-name').forEach((el) => { el.textContent = conn.deviceName || 'ESPectre'; });
        $$('.js-device-banner-sub').forEach((el) => { el.textContent = conn.deviceBannerSub; });
        renderDeviceIdentity();
        $$('.js-ble-chip').forEach((chip) => {
            chip.classList.toggle('ready', connected && conn.mode === 'ble' && browserSupport.bluetooth);
            chip.textContent = connected && conn.mode === 'ble' && browserSupport.bluetooth
                ? 'BLE · READY'
                : 'BLE';
        });

        syncMonitorDemoButton();
        syncSensingControls();
        syncDiagnosticsPolling();
        renderBrowserSupport();
        renderTelemetry();
        syncDemoToast();
    }

    function renderTelemetry() {
        const pct = Math.round(energyFraction() * 100) + '%';
        $$('.js-energy-fill').forEach((el) => { el.style.width = pct; });
        $$('.js-motion-label').forEach((el) => {
            el.textContent = conn.motion ? 'MOTION' : 'IDLE';
            el.classList.toggle('motion', conn.motion);
        });
    }

    /* ============================================================= routing */

    function focusRouteContent() {
        const page = $(`[data-page="${route}"]`);
        if (!page) return;
        const target = page.querySelector('h1') || page;
        if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
    }

    function applyRoute({ focus = true } = {}) {
        $$('.js-page').forEach((page) => {
            const current = page.dataset.page === route;
            page.hidden = !current;
            if (current) page.id = 'main-content';
            else page.removeAttribute('id');
        });
        $$('[data-route-link]').forEach((link) => {
            const target = link.dataset.routeLink;
            const active = target === route
                || routeRegistry.groupOf(route) === target;
            link.classList.toggle('active', active);
            if (active) link.setAttribute('aria-current', 'page');
            else link.removeAttribute('aria-current');
        });
        document.title = window.getRouteTitle
            ? window.getRouteTitle(route)
            : 'ESPectre — Wi-Fi motion sensing';
        window.scrollTo(0, 0);
        if (route !== 'theremin') thereminStop();
        if (route === 'monitor') monitorResizeChart();
        const contentPromise = $(`[data-page="${route}"] .js-static-content`)
            ? loadStaticContent(route)
            : Promise.resolve();
        if (route === 'home') updateReleaseBadge();
        if (route === 'flash') {
            if (browserSupport.flash) {
                loadBrowserDependency(
                    '/vendor/esp-web-tools-10.4.0/install-button.js',
                    'https://unpkg.com/esp-web-tools@10.4.0/dist/web/install-button.js?module',
                    { module: true }
                ).catch((error) => flashStatus(error.message, 'is-error'));
            }
            flashRefresh();
        }
        if (focus) contentPromise.finally(focusRouteContent);
        // The router owns navigation, so it reports it.
        if (window.trackRouteView) window.trackRouteView(route);
    }

    /**
     * Single entry point for navigation. `force` applies the current route on
     * startup; without it a repeated route is ignored so one navigation never
     * reports two page views.
     */
    function setRoute(next, { force = false, focus = true } = {}) {
        const remapped = LEGACY_TOOL_ROUTES[next] || next;
        const target = routeRegistry.has(remapped) ? remapped : 'home';
        if (!force && target === route) return;
        const previousRoute = route;
        if (previousRoute === 'game' && target !== 'game') reportGameAbandon('route_change');
        route = target;
        dropdownOpen = false;
        applyRoute({ focus });
        renderConnection();
    }

    function onHashChange() {
        setRoute((location.hash || '#home').slice(1));
    }

    /* ======================================================= static content */

    /*
     * Guides, docs, media, and the roadmap live in shared HTML fragments,
     * which also build their canonical static pages. The SPA fetches each
     * fragment on first visit so text is not duplicated and the device
     * connection survives.
     */
    const staticContentCache = new Map();

    async function loadStaticContent(route) {
        const container = $(`[data-page="${route}"] .js-static-content`);
        if (!container || container.dataset.loaded === 'true') return;
        const contentUrl = container.dataset.contentUrl;
        try {
            if (!staticContentCache.has(contentUrl)) {
                const response = await fetch(contentUrl);
                if (!response.ok) throw new Error('HTTP ' + response.status);
                staticContentCache.set(contentUrl, await response.text());
            }
            container.innerHTML = staticContentCache.get(contentUrl);
            container.dataset.loaded = 'true';
            if (window.initPageTocs) window.initPageTocs(container);
        } catch (error) {
            console.warn('Static content fetch failed:', error);
            container.innerHTML = '<p class="guide-loading">This page could not be loaded. '
                + '<a href="' + container.dataset.staticUrl + '">Open the standalone page</a>.</p>';
        }
    }

    /*
     * Fragments and content cards link to the canonical static URLs. Inside
     * the app those clicks become hash navigation, so the page never reloads
     * and an active connection survives; modified clicks (new tab) are left
     * to the browser. Root-anchored hash links are normalized for the same
     * reason: from /index.html they would otherwise reload onto /. Same-page
     * anchors scroll within the active SPA page without replacing its route.
     */
    function interceptCanonicalLinks(event) {
        if (event.defaultPrevented || event.button !== 0) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        const link = event.target.closest('a[href]');
        if (!link) return;
        const href = link.getAttribute('href');
        const staticRoute = routeRegistry.routeForPath(href);
        if (staticRoute) {
            event.preventDefault();
            location.hash = '#' + staticRoute;
        } else if (href.startsWith('/#')) {
            event.preventDefault();
            location.hash = href.slice(1);
        } else if (href.startsWith('#') && href.length > 1) {
            const page = $(`[data-page="${route}"]`);
            let targetId = '';
            try {
                targetId = decodeURIComponent(href.slice(1));
            } catch (error) {
                return;
            }
            const target = page && document.getElementById(targetId);
            if (!target || !page.contains(target)) return;
            event.preventDefault();
            target.scrollIntoView();
            if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
            target.focus({ preventScroll: true });
        }
    }

    /* =============================================================== toast */

    let toastTimer = null;

    function toast(message) {
        const el = $('.js-toast');
        el.textContent = message;
        el.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
    }

    function syncDemoToast() {
        const el = $('.js-demo-toast');
        if (!el) return;
        el.hidden = !(conn.mode === 'demo' && conn.status === 'connected');
    }

    /* ====================================================== scroll narrative */

    let activeScrollyScene = -1;
    let scrollyFrame = null;
    let heroFrameTimer = null;
    const HERO_FRAME_HOLD = 2000;

    function scrollySceneFromPosition(section, sceneCount) {
        const rect = section.getBoundingClientRect();
        const travel = Math.max(1, rect.height - window.innerHeight);
        const progress = Math.min(1, Math.max(0, -rect.top / travel));
        return Math.min(sceneCount - 1, Math.floor(progress * sceneCount));
    }

    function stopHeroFrameSequence() {
        clearTimeout(heroFrameTimer);
        heroFrameTimer = null;
    }

    function startHeroFrameSequence() {
        const media = $('.hero-media');
        stopHeroFrameSequence();
        media.classList.remove('is-connected');
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            media.classList.add('is-connected');
            return;
        }
        heroFrameTimer = setTimeout(() => {
            media.classList.add('is-connected');
            heroFrameTimer = null;
        }, HERO_FRAME_HOLD);
    }

    function setScrollyScene(scene) {
        if (scene === activeScrollyScene) return;
        activeScrollyScene = scene;

        const scenes = $$('.js-scrolly-scene');
        const useMobileAsset = window.matchMedia('(max-width: 720px)').matches;
        [scene, scene + 1].forEach((index) => {
            const image = scenes[index] && scenes[index].querySelector('img[data-src]');
            if (!image) return;
            image.src = useMobileAsset && image.dataset.srcMobile
                ? image.dataset.srcMobile
                : image.dataset.src;
            image.removeAttribute('data-src');
            image.removeAttribute('data-src-mobile');
        });

        $$('.js-scrolly-scene, .js-scrolly-caption, .js-scrolly-marker').forEach((el) => {
            el.classList.toggle('is-active', Number(el.dataset.scene) === scene);
        });
        $('.scrolly-stage').classList.toggle('is-intro', scene === 0);
        if (scene === 0) startHeroFrameSequence();
        else stopHeroFrameSequence();
        if (scene > 0) $('.js-scrolly-current').textContent = String(scene).padStart(2, '0');
    }

    function renderScrolly() {
        scrollyFrame = null;
        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null) return;

        const sceneCount = $$('.js-scrolly-scene').length;
        setScrollyScene(scrollySceneFromPosition(section, sceneCount));
    }

    function queueScrollyRender() {
        if (scrollyFrame !== null) return;
        scrollyFrame = requestAnimationFrame(renderScrolly);
    }

    function scrollyInit() {
        window.addEventListener('scroll', queueScrollyRender, { passive: true });
        window.addEventListener('resize', queueScrollyRender);
        renderScrolly();
    }

    /* =============================================================== flash */

    const flash = {
        manifests: {}, installUrl: null, badgeChecked: false,
        installerObserver: null, watchedDialogs: new WeakSet(), catalogReports: new Set(),
        downloadReady: false, detectedChip: '', supportedChipLabels: []
    };

    /*
     * Presentation order for the Flash selectors and published-chip list.
     * Anything not listed keeps its manifest order and lands after the listed
     * entries, so a new frontend or chip still shows up without touching this code.
     */
    const FRONTEND_ORDER = ['native', 'esphome', 'matter'];
    const CHIP_ORDER = ['esp32', 'esp32s3', 'esp32c3', 'esp32c5', 'esp32c6'];
    const FLASH_CHIP_FOUND_RE = /Initialized\. Found ([A-Z0-9-]+)/i;
    const FLASH_CHIP_UNSUPPORTED_RE = /Your ([A-Z0-9-]+) board is not supported/i;

    function flashManifestFrontends(manifest) {
        const frontends = manifest && manifest.frontends;
        if (!frontends || typeof frontends !== 'object' || Array.isArray(frontends)) {
            const error = new Error(
                'Firmware catalog format is invalid for browser flashing.'
            );
            error.name = 'FirmwareCatalogFormatError';
            throw error;
        }
        return frontends;
    }

    function byPreferredOrder(order, a, b) {
        const ia = order.indexOf(a);
        const ib = order.indexOf(b);
        if (ia === -1 && ib === -1) return 0;
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
    }

    function flashResolveUrl(url) {
        return url;
    }

    async function flashLoadManifest(channel) {
        if (flash.manifests[channel]) return flash.manifests[channel];
        const response = await fetch(
            '/artifacts/firmware/' + channel + '/firmware-manifest-' + channel + '.json',
            { cache: 'no-store' }
        );
        if (!response.ok) {
            const error = new Error('Unable to load the ' + channel + ' firmware manifest.');
            error.status = response.status;
            throw error;
        }
        const manifest = await response.json();
        flash.manifests[channel] = manifest;
        return manifest;
    }

    function flashStatus(message, kind) {
        const el = $('.js-flash-status');
        el.textContent = message;
        el.className = 'flash-status js-flash-status' + (kind ? ' ' + kind : '');
    }

    function formatEnglishList(items) {
        if (items.length === 0) return '';
        if (items.length === 1) return items[0];
        if (items.length === 2) return items[0] + ' and ' + items[1];
        return items.slice(0, -1).join(', ') + ', and ' + items[items.length - 1];
    }

    function flashUnsupportedBoardMessage(chipFamily) {
        const detected = chipFamily
            ? ('This ' + chipFamily + ' board is not supported.')
            : 'This board is not supported.';
        const list = formatEnglishList(flash.supportedChipLabels);
        if (!list) return detected;
        return detected + ' Published firmware is available for ' + list + '.';
    }

    function flashCreateDownloadLink(label, href, chip) {
        const link = document.createElement('a');
        link.className = 'btn-ghost btn-sm';
        link.href = href;
        link.textContent = label;
        if (chip) {
            link.dataset.chip = chip;
            return link;
        }
        link.target = '_blank';
        link.rel = 'noopener';
        return link;
    }

    function flashRenderDownloads(artifacts) {
        const container = $('.js-flash-chip-downloads');
        container.replaceChildren();
        if (!artifacts.length) {
            container.append(flashCreateDownloadLink(
                'Browse GitHub releases',
                'https://github.com/francescopace/espectre/releases'
            ));
            flash.downloadReady = false;
            return;
        }
        for (const artifact of artifacts) {
            const link = flashCreateDownloadLink(
                artifact.chip_label,
                flashResolveUrl(artifact.url),
                artifact.chip
            );
            if (artifact.filename) link.setAttribute('download', artifact.filename);
            container.append(link);
        }
        flash.downloadReady = true;
    }

    function flashNextLink(href, label) {
        const link = document.createElement('a');
        link.href = href;
        link.textContent = label;
        return link;
    }

    function flashNextAction(label, className) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'link-btn ' + className;
        button.textContent = label;
        return button;
    }

    function flashHideMatterQr() {
        const status = $('.js-matter-status');
        const result = $('.js-matter-result');
        if (status) {
            status.hidden = true;
            status.textContent = '';
        }
        if (result) result.hidden = true;
    }

    function flashSetNextStep(frontendKey) {
        const note = $('.js-flash-next');
        if (frontendKey !== 'matter') flashHideMatterQr();
        if (frontendKey === 'native') {
            note.replaceChildren(
                'After flashing Native, open ',
                flashNextLink('#configure', 'Configure'),
                ' to provision connectivity, then ',
                flashNextLink('#monitor', 'Monitor'),
                '.'
            );
            note.hidden = false;
            return;
        }
        if (frontendKey === 'esphome') {
            note.replaceChildren(
                'After flashing ESPHome, there are several ways to configure Wi-Fi. See the ',
                flashNextLink('/guides/setup/', 'setup guide'),
                '.'
            );
            note.hidden = false;
            return;
        }
        if (frontendKey === 'matter') {
            const readQr = flashNextAction('Read the onboarding QR over USB', 'js-matter-read');
            if (!browserSupport.flash) {
                readQr.disabled = true;
                readQr.title = flashUnsupportedMessage();
            }
            note.replaceChildren(
                'After flashing Matter, commission the device with a Matter controller. ',
                readQr,
                ' or see the ',
                flashNextLink('/guides/setup/', 'setup guide'),
                '.'
            );
            note.hidden = false;
            return;
        }
        note.replaceChildren();
        note.hidden = true;
    }

    async function flashRefresh() {
        const frontendSel = document.getElementById('flash-frontend');
        const channelSel = document.getElementById('flash-channel');
        const summary = $('.js-flash-summary');
        const installButton = $('.js-flash-install');
        flash.downloadReady = false;

        try {
            const manifest = await flashLoadManifest(channelSel.value);
            const frontendsMap = flashManifestFrontends(manifest);

            const frontends = Object.entries(frontendsMap)
                .sort(([a], [b]) => byPreferredOrder(FRONTEND_ORDER, a, b));
            const successKey = channelSel.value + ':success';
            if (!flash.catalogReports.has(successKey)) {
                const reported = track('firmware_catalog', {
                    channel: channelSel.value,
                    result: 'success',
                    frontend_count: frontends.length,
                    artifact_count: frontends.reduce(
                        (total, [, frontend]) => total + (frontend.artifacts || []).length, 0
                    )
                });
                if (reported) flash.catalogReports.add(successKey);
            }
            const previousFrontend = frontendSel.value;
            frontendSel.innerHTML = '';
            for (const [key, value] of frontends) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = value.label || key;
                frontendSel.appendChild(option);
            }
            if (frontends.some(([key]) => key === previousFrontend)) frontendSel.value = previousFrontend;

            const selectedFrontend = frontendsMap[frontendSel.value];
            const artifacts = ((selectedFrontend || {}).artifacts || [])
                .filter((a) => a.build_type === 'factory' && a.chip_family && a.url)
                .sort((a, b) => byPreferredOrder(CHIP_ORDER, a.chip, b.chip));
            flash.supportedChipLabels = artifacts.map((artifact) => artifact.chip_label);
            if (flash.installUrl) {
                URL.revokeObjectURL(flash.installUrl);
                flash.installUrl = null;
            }
            installButton.removeAttribute('manifest');
            if (!artifacts.length) {
                flashRenderDownloads([]);
                flashSetNextStep(frontendSel.value);
                summary.textContent = 'No matching firmware was found for the selected combination.';
                flashStatus('Change the selection or use the manual download.', 'is-error');
                return;
            }

            const frontendLabel = (selectedFrontend || {}).label || frontendSel.value;
            const installManifest = {
                name: 'ESPectre ' + frontendLabel,
                version: manifest.version,
                builds: artifacts.map((artifact) => ({
                    chipFamily: artifact.chip_family,
                    parts: [{ path: flashResolveUrl(artifact.url), offset: 0 }]
                }))
            };
            flash.installUrl = URL.createObjectURL(
                new Blob([JSON.stringify(installManifest)], { type: 'application/json' })
            );
            installButton.setAttribute('manifest', flash.installUrl);

            summary.replaceChildren();
            const title = document.createElement('strong');
            title.textContent = frontendLabel;
            const detail = document.createTextNode(manifest.release_tag + ' ');
            const channel = document.createElement('span');
            channel.className = 'mono-sub';
            channel.textContent = '(' + manifest.channel + ')';
            summary.append(title, document.createElement('br'), detail, channel);
            flashRenderDownloads(artifacts);
            flashSetNextStep(frontendSel.value);

            if (!browserSupport.flash) {
                flashStatus(flashUnsupportedMessage(), 'is-error');
            } else {
                flashStatus('Ready. Connect the board over USB, then install.', 'is-ready');
            }
        } catch (error) {
            flash.supportedChipLabels = [];
            if (flash.installUrl) {
                URL.revokeObjectURL(flash.installUrl);
                flash.installUrl = null;
            }
            installButton.removeAttribute('manifest');
            flashRenderDownloads([]);
            flashSetNextStep('');
            summary.textContent = 'Firmware metadata is currently unavailable.';
            flashStatus(error.message, 'is-error');
            const failureKey = channelSel.value + ':failure';
            if (!flash.catalogReports.has(failureKey)) {
                const reported = track('firmware_catalog', {
                    channel: channelSel.value,
                    result: 'failure',
                    error_type: errorType(error)
                });
                if (reported) flash.catalogReports.add(failureKey);
            }
        }
    }

    function matterDelay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function matterResetDevice(port) {
        await port.setSignals({ dataTerminalReady: false, requestToSend: true });
        await matterDelay(100);
        await port.setSignals({ dataTerminalReady: false, requestToSend: false });
    }

    async function matterReadCodes(port, timeoutMs = 20000) {
        const reader = port.readable.getReader();
        const decoder = new TextDecoder();
        const deadline = Date.now() + timeoutMs;
        let input = '';
        try {
            while (Date.now() < deadline) {
                const result = await Promise.race([
                    reader.read(),
                    matterDelay(deadline - Date.now()).then(() => ({ timedOut: true }))
                ]);
                if (result.timedOut || result.done) break;
                input += decoder.decode(result.value, { stream: true });
                if (input.length > 16384) input = input.slice(-8192);
                const qr = input.match(/MATTER_QR=(MT:[A-Z0-9.\-]+)/);
                const manual = input.match(/MATTER_MANUAL_CODE=([0-9]+)/);
                if (qr && manual) return { qr: qr[1], manual: manual[1] };
            }
        } finally {
            await reader.cancel().catch(() => {});
            reader.releaseLock();
        }
        throw new Error('Matter codes were not received. Press reset on the board, then try again.');
    }

    async function matterReadQr() {
        const status = $('.js-matter-status');
        const result = $('.js-matter-result');
        const trigger = $('.js-matter-read');
        if (!browserSupport.flash) {
            status.hidden = false;
            status.textContent = flashUnsupportedMessage();
            track('matter_qr_read', { result: 'unsupported' });
            return;
        }
        try {
            await loadBrowserDependency(
                '/vendor/qrcodejs-1.0.0/qrcode.min.js',
                'https://unpkg.com/qrcodejs@1.0.0/qrcode.min.js'
            );
        } catch (error) {
            status.hidden = false;
            status.textContent = 'The QR renderer could not be loaded.';
            track('matter_qr_read', { result: 'failure', error_type: 'QrRendererMissing' });
            return;
        }
        let port;
        if (trigger) trigger.disabled = true;
        result.hidden = true;
        status.hidden = false;
        status.textContent = 'Choose the ESPectre serial port, then wait for the device to restart.';
        try {
            port = await navigator.serial.requestPort();
            await port.open({ baudRate: 115200 });
            await matterResetDevice(port);
            const codes = await matterReadCodes(port);
            const canvas = $('.js-matter-canvas');
            canvas.innerHTML = '';
            new window.QRCode(canvas, {
                text: codes.qr,
                width: 220,
                height: 220,
                colorDark: '#000000',
                colorLight: '#ffffff',
                correctLevel: window.QRCode.CorrectLevel.M
            });
            $('.js-matter-payload').textContent = codes.qr;
            $('.js-matter-manual').textContent = codes.manual;
            result.hidden = false;
            status.textContent = 'This QR is stored on the device and remains the same after normal updates.';
            track('matter_qr_read', { result: 'success' });
        } catch (error) {
            status.textContent = error.message || 'Unable to read the Matter QR code.';
            track('matter_qr_read', { result: 'failure', error_type: errorType(error) });
        } finally {
            if (port && (port.readable || port.writable)) {
                await port.close().catch(() => {});
            }
            if (trigger) trigger.disabled = false;
        }
    }

    /**
     * Shows the latest published release in the hero badge. The stable
     * manifest is staged by CI from the GitHub release tag, so it is already
     * the newest version and needs no API call. The badge is decorative:
     * it stays hidden when the manifest is unavailable.
     */
    async function updateReleaseBadge() {
        if (flash.badgeChecked) return;
        flash.badgeChecked = true;
        try {
            const manifest = await flashLoadManifest('stable');
            const version = String(manifest.release_tag || manifest.version || '').replace(/^v/, '');
            if (!version) return;
            $('.js-release-text').textContent = 'v' + version + ' available';
            $('.js-release-badge').hidden = false;
        } catch (error) {
            if (error && error.status !== 404) {
                console.warn('Release badge unavailable:', error);
            }
        }
    }

    function flashParams() {
        return {
            frontend: document.getElementById('flash-frontend').value,
            channel: document.getElementById('flash-channel').value,
            chip: flash.detectedChip
        };
    }

    function watchFirmwareInstallDialog(dialog) {
        if (flash.watchedDialogs.has(dialog)) return;
        flash.watchedDialogs.add(dialog);
        let started = false;
        let reported = false;
        let shadowObserver = null;

        const report = (result) => {
            if (reported) return;
            reported = true;
            track('firmware_install_result', { ...flashParams(), result });
            if (shadowObserver) shadowObserver.disconnect();
        };
        const inspect = () => {
            const text = dialog.shadowRoot ? dialog.shadowRoot.textContent : '';
            if (/Installing|Preparing installation/i.test(text)) started = true;
            const found = text.match(FLASH_CHIP_FOUND_RE);
            if (found) flash.detectedChip = found[1];
            const unsupported = text.match(FLASH_CHIP_UNSUPPORTED_RE);
            if (unsupported) {
                flash.detectedChip = unsupported[1];
                flashStatus(flashUnsupportedBoardMessage(unsupported[1]), 'is-error');
                report('unsupported');
                return;
            }
            if (/Installation complete!/i.test(text)) report('success');
            else if (/Installation failed/i.test(text)) report('failure');
        };
        const attach = () => {
            if (!dialog.shadowRoot || shadowObserver) return false;
            shadowObserver = new MutationObserver(inspect);
            shadowObserver.observe(dialog.shadowRoot, {
                childList: true, subtree: true, characterData: true
            });
            inspect();
            return true;
        };
        if (!attach()) [0, 50, 200].forEach((delay) => setTimeout(attach, delay));

        const removalObserver = new MutationObserver(() => {
            if (dialog.isConnected) return;
            removalObserver.disconnect();
            if (started && !reported) report('cancelled');
            if (shadowObserver) shadowObserver.disconnect();
        });
        removalObserver.observe(document.body, { childList: true, subtree: true });
    }

    function observeFirmwareInstaller() {
        if (flash.installerObserver) return;
        const inspect = () => {
            document.querySelectorAll('ewt-install-dialog').forEach(watchFirmwareInstallDialog);
        };
        flash.installerObserver = new MutationObserver(inspect);
        flash.installerObserver.observe(document.body, { childList: true, subtree: true });
        inspect();
    }

    function flashInit() {
        const selectionType = {
            'flash-frontend': 'frontend', 'flash-channel': 'channel'
        };
        Object.keys(selectionType).forEach((id) => {
            document.getElementById(id).addEventListener('change', () => {
                track('firmware_selection', {
                    selection_type: selectionType[id],
                    frontend: document.getElementById('flash-frontend').value,
                    channel: document.getElementById('flash-channel').value
                });
                flashRefresh();
            });
        });
        $('.js-flash-install').addEventListener('click', (event) => {
            if (!browserSupport.flash) {
                event.preventDefault();
                flashStatus(flashUnsupportedMessage(), 'is-error');
                return;
            }
            flash.detectedChip = '';
            flashStatus('Select the serial port. The installer detects the chip and chooses the matching firmware.');
            track('firmware_install_start', flashParams());
        });
        $('.js-flash-chip-downloads').addEventListener('click', (event) => {
            const link = event.target.closest('a[data-chip]');
            if (!link || !flash.downloadReady) return;
            track('firmware_download', {
                ...flashParams(),
                chip: link.dataset.chip,
                result: 'started'
            });
        });
        $('.js-flash-next').addEventListener('click', (event) => {
            if (!event.target.closest('.js-matter-read')) return;
            event.preventDefault();
            matterReadQr();
        });
        if (browserSupport.flash) observeFirmwareInstaller();
    }

    /* ============================================================= monitor */

    const MONITOR_CHART_WINDOW_MS = 5 * 60 * 1000;
    const MONITOR_CHART_MAX_POINTS = 5 * 60 * 10;
    const MONITOR_CHART_COALESCE_MS = 100;
    const MONITOR_TELEMETRY_STALE_MS = 1500;
    const MONITOR_CALIBRATION_FALLBACK_MS = 45 * 1000;
    const MONITOR_CALIBRATION_SAFETY_MS = 90 * 1000;
    const MONITOR_DEMO_CALIBRATION_MS = 2500;
    const MONITOR_DISCOVERY_TIMEOUT_MS = 2000;

    const monitor = {
        client: null,
        baseTopic: null,
        demoTimer: null,
        demoT: 0,
        demoMove: 0.05,
        points: [],
        chartFrame: 0,
        lastTelemetryAt: 0,
        startedAt: 0,
        connectedAt: 0,
        entryPoint: '',
        inputMode: null,
        readyState: '',
        readyAt: 0,
        readyTracked: false,
        closing: false,
        pendingCommands: new Map(),
        commands: new Set(),
        commandCatalogReady: false,
        bleRequested: false,
        handoffReady: false,
        closingBleForLive: false,
        diagTimer: null,
        calibrating: false,
        calibrationTimer: null,
        boundDeviceId: '',
        discoveryActive: false,
        discoveredDevices: {},
        discoveryPrefix: '',
        discoveryTopics: [],
        discoveryTimer: 0,
        brokerUrl: ''
    };

    function monitorStatus(message) {
        const status = $('.js-mon-status');
        if (!status) return;
        status.hidden = !message;
        status.textContent = message || '';
    }

    function monitorDiagStatus(message) {
        const el = $('.js-mon-diag-status');
        if (!el) return;
        el.hidden = !message;
        el.textContent = message || '';
    }

    function monitorBaseTopic() {
        const prefix = document.getElementById('mon-topic-prefix').value.trim().replace(/\/+$/, '');
        const device = document.getElementById('mon-device').value.trim().replace(/^\/+|\/+$/g, '');
        if (!prefix || !device) return '';
        return prefix + '/' + device;
    }

    function clearMonitorFieldError(input) {
        input.classList.remove('is-invalid');
        input.removeAttribute('aria-invalid');
        input.setCustomValidity('');
    }

    function markMonitorFieldError(input, message) {
        clearMonitorFieldError(input);
        // Force a reflow so repeated submissions restart the brief error flash.
        void input.offsetWidth;
        input.classList.add('is-invalid');
        input.setAttribute('aria-invalid', 'true');
        input.setCustomValidity(message);
    }

    function validateMonitorConnection() {
        const hostInput = document.getElementById('mon-host');
        const portInput = document.getElementById('mon-port');
        const prefixInput = document.getElementById('mon-topic-prefix');
        const deviceInput = document.getElementById('mon-device');
        const pathInput = document.getElementById('mon-path');
        const host = hostInput.value.trim();
        const port = portInput.value.trim();
        const prefix = prefixInput.value.trim().replace(/\/+$/, '');
        const device = deviceInput.value.trim().replace(/^\/+|\/+$/g, '');
        const path = pathInput.value.trim();
        const portNumber = Number(port);
        const deviceValid = !device || (!device.includes('/') && !/[+#]/.test(device));
        const fields = [
            [hostInput, !!host && !/\s|:\/\/|\//.test(host), 'Enter a valid broker host.'],
            [portInput, !!port && Number.isInteger(portNumber)
                && portNumber >= 1 && portNumber <= 65535, 'Enter a port from 1 to 65535.'],
            [prefixInput, !!prefix, 'Enter a topic prefix.'],
            [deviceInput, deviceValid, 'Enter a device ID without / or wildcards.'],
            [pathInput, path.startsWith('/') && !/\s/.test(path), 'Enter a path starting with /.']
        ];
        const invalidFields = fields.filter(([, valid]) => !valid);
        fields.forEach(([input, valid, message]) => {
            if (valid) clearMonitorFieldError(input);
            else markMonitorFieldError(input, message);
        });
        if (invalidFields.length) {
            monitorStatus('');
            invalidFields[0][0].focus({ preventScroll: true });
            return null;
        }
        return {
            host,
            port,
            path,
            tls: document.getElementById('mon-tls').checked,
            prefix,
            device,
            base: device ? prefix + '/' + device : ''
        };
    }

    function monitorIsMqttLive() {
        return monitor.inputMode === 'mqtt' && !!monitor.client;
    }

    function applyMqttLiveTelemetry(movement, threshold, motionState) {
        if (!monitor.handoffReady) return;
        if (monitorIsMqttLive() && conn.mode !== 'demo'
                && (conn.mode !== 'mqtt' || conn.status !== 'connected')) {
            bindMqttToConnection();
        }
        markMonitorReady('telemetry');
        monitorFeed(movement, threshold, motionState);
        monitorResizeChart();
        applyLiveTelemetry(movement, threshold, motionState);
    }

    function ingestMqttPayload(base, topic, payload) {
        const topicName = mqttUtf8(topic);
        const text = mqttUtf8(payload).trim();
        if (!topicName || !text) return;
        if (monitor.boundDeviceId && conn.deviceId && monitor.boundDeviceId !== conn.deviceId) return;
        const suffix = topicName.startsWith(base + '/') ? topicName.slice(base.length + 1) : '';
        try {
            if (suffix === 'commands/accepted' || suffix === 'commands/rejected') {
                const data = JSON.parse(text);
                const pending = monitor.pendingCommands.get(data.command_id);
                if (!pending) return;
                clearTimeout(pending.timer);
                monitor.pendingCommands.delete(data.command_id);
                if (suffix === 'commands/accepted' && data.accepted !== false) {
                    pending.resolve(data);
                } else {
                    pending.reject(new Error(data.message || 'The device rejected the command.'));
                }
                return;
            }
            if (suffix === 'commands/catalog') {
                const data = JSON.parse(text);
                if (!data || !Array.isArray(data.commands)) return;
                monitor.commands = new Set(data.commands);
                monitor.commandCatalogReady = true;
                syncSensingControls();
                return;
            }
            if (suffix === 'info') {
                applyDeviceInfo(JSON.parse(text));
                return;
            }
            if (suffix === 'status') {
                const data = JSON.parse(text);
                const online = data.online === true;
                handleOtaDeviceAvailability(online);
                if (!online && !otaAwaitingReconnect && otaState !== 'reboot_scheduled') {
                    toast('The broker is connected, but the device is offline.');
                    monitorStatus('Device offline. Waiting for it to reconnect…');
                }
                return;
            }
            if (suffix === 'ota/state') {
                applyOtaStatus(JSON.parse(text));
                return;
            }
            if (suffix === 'stats') {
                const data = JSON.parse(text);
                if (!data || typeof data !== 'object'
                        || !['traffic_tx_pps', 'csi_callback_pps', 'free_memory_kb']
                            .some((key) => data[key] !== undefined)) return;
                markMonitorReady('diagnostics');
                monitorStats(data);
                if (data.traffic_tx_pps === undefined) {
                    monitorDiagStatus('Diagnostics received — this firmware does not expose the extended fields.');
                }
                return;
            }
            if (suffix === 'telemetry') {
                const data = JSON.parse(text);
                if (!data || typeof data !== 'object') return;
                const movement = Number(data.movement_score ?? data.movement);
                const threshold = Number(data.threshold);
                if (!Number.isFinite(movement) || !Number.isFinite(threshold)) return;
                monitor.lastTelemetryAt = Date.now();
                applyMqttLiveTelemetry(
                    movement,
                    threshold,
                    data.motion_state || data.state
                );
                return;
            }
            if (suffix === 'ha/movement/state') {
                if (monitorHasFreshTelemetry()) return;
                const movement = Number(text);
                if (!Number.isFinite(movement)) return;
                applyMqttLiveTelemetry(movement, conn.threshold, conn.motion ? 'motion' : 'idle');
                return;
            }
            if (suffix === 'ha/threshold/state') {
                const threshold = Number(text);
                if (!Number.isFinite(threshold)) return;
                conn.threshold = threshold;
                syncThresholdControl(threshold);
                renderTelemetry();
                return;
            }
            if (suffix === 'ha/motion/state') {
                if (monitorHasFreshTelemetry()) return;
                const motion = text === 'ON' || text === '1' || text === 'motion';
                applyMqttLiveTelemetry(conn.movement, conn.threshold, motion ? 'motion' : 'idle');
                return;
            }
            if (suffix === 'ha/detector/state') {
                document.getElementById('sense-detector').value = text;
                if (text !== 'lightweight') setCalibrationBusy(false);
                else syncSensingControls();
                return;
            }
            if (suffix === 'ha/calibrate/state') {
                const calibrating = text === 'ON' || text === '1';
                setCalibrationBusy(calibrating);
                if (calibrating) scheduleCalibrationIdle(MONITOR_CALIBRATION_SAFETY_MS);
                return;
            }
            if (suffix === 'ha/motion_on_hits/state') {
                document.getElementById('sense-motion-on').value = text;
                return;
            }
            if (suffix === 'ha/motion_off_hits/state') {
                document.getElementById('sense-motion-off').value = text;
                return;
            }
            if (suffix === 'ha/csi_traffic_mode/state') {
                applyCsiTrafficModeSelect(text);
                return;
            }
            if (suffix === 'ha/traffic_generator_mode/state') {
                document.getElementById('sense-generator-mode').value = text;
            }
        } catch (error) { /* ignore malformed payloads */ }
    }

    function syncMonitorDemoButton() {
        const demo = $('.js-mon-demo');
        const ble = $('.js-mon-ble');
        const connect = $('.js-mon-connect');
        const mqttLive = monitorIsMqttLive();
        if (demo) demo.hidden = mqttLive;
        if (ble) ble.hidden = !mqttLive;
        if (connect) {
            connect.disabled = mqttLive || monitor.discoveryActive;
            connect.textContent = mqttLive ? 'Connected'
                : monitor.discoveryActive ? 'Scanning…'
                : 'Connect broker';
        }
    }

    function monitorHasFreshTelemetry() {
        return monitor.lastTelemetryAt > 0
            && (Date.now() - monitor.lastTelemetryAt) < MONITOR_TELEMETRY_STALE_MS;
    }

    function monitorResetChart() {
        monitor.points = [];
        monitor.lastTelemetryAt = 0;
        if (monitor.chartFrame) {
            cancelAnimationFrame(monitor.chartFrame);
            monitor.chartFrame = 0;
        }
        const canvas = $('.js-mon-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function resetMonitorLiveView() {
        monitorResetChart();
        const stateEl = $('.js-mon-state');
        if (stateEl) {
            stateEl.textContent = '—';
            stateEl.classList.remove('motion');
        }
        const moveEl = $('.js-mon-move');
        if (moveEl) moveEl.textContent = '—';
        monitorStats({});
        monitorDiagStatus('');
        conn.movement = 0;
        conn.motion = false;
        renderTelemetry();
    }

    function monitorQueueChart() {
        if (monitor.chartFrame) return;
        monitor.chartFrame = requestAnimationFrame(() => {
            monitor.chartFrame = 0;
            monitorDrawChart();
        });
    }

    function monitorFeed(movement, threshold, state) {
        const now = Date.now();
        const motion = state !== null && state !== undefined
            ? state === 1 || state === 'motion'
            : movement >= threshold;
        const last = monitor.points[monitor.points.length - 1];
        if (last && now - last.at < MONITOR_CHART_COALESCE_MS) {
            last.m = movement;
            last.t = threshold;
            last.at = now;
            last.on = motion;
        } else {
            monitor.points.push({ m: movement, t: threshold, at: now, on: motion });
        }
        const oldest = now - MONITOR_CHART_WINDOW_MS;
        while (monitor.points.length
                && (monitor.points[0].at < oldest || monitor.points.length > MONITOR_CHART_MAX_POINTS)) {
            monitor.points.shift();
        }
        const stateEl = $('.js-mon-state');
        stateEl.textContent = motion ? 'MOTION' : 'IDLE';
        stateEl.classList.toggle('motion', motion);
        $('.js-mon-move').textContent = movement.toFixed(3);
        monitorQueueChart();
    }

    function monitorStat(value, digits, suffix) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
        return Number(value).toFixed(digits) + suffix;
    }

    function monitorStats(data) {
        $('.js-mon-traffic').textContent = monitorStat(data.traffic_tx_pps, 1, ' pps');
        $('.js-mon-callbacks').textContent = monitorStat(data.csi_callback_pps, 1, ' pps');
        $('.js-mon-filtered').textContent = monitorStat(data.csi_filtered_pps, 1, ' pps');
        $('.js-mon-admitted').textContent = monitorStat(data.csi_admitted_pps, 1, ' pps');
        $('.js-mon-channel').textContent = monitorStat(data.wifi_channel, 0, '');
        $('.js-mon-rssi').textContent = monitorStat(data.wifi_rssi_dbm, 0, ' dBm');
        $('.js-mon-heap').textContent = monitorStat(data.free_memory_kb, 1, ' KiB');
        $('.js-mon-loop').textContent = monitorStat(data.loop_time_ms, 2, ' ms');
    }

    function monitorDrawChart() {
        const canvas = $('.js-mon-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        if (width < 2 || height < 2 || monitor.points.length < 2) return;

        const styles = getComputedStyle(document.documentElement);
        const accent = styles.getPropertyValue('--accent').trim() || '#4f6bff';
        const accentSoft = styles.getPropertyValue('--accent-soft').trim() || 'rgba(79, 107, 255, 0.09)';
        const dim = styles.getPropertyValue('--dim').trim() || '#888';
        const border = styles.getPropertyValue('--border').trim() || '#e6e9ee';
        const labelH = 16;
        const plotH = Math.max(8, height - labelH);
        const now = monitor.points[monitor.points.length - 1].at;
        const t0 = now - MONITOR_CHART_WINDOW_MS;
        const x = (at) => ((at - t0) / MONITOR_CHART_WINDOW_MS) * width;
        const y = (v) => plotH - Math.min(1, Math.max(0, v)) * (plotH - 8) - 4;

        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.fillStyle = dim;
        ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
        ctx.textBaseline = 'top';
        const minuteMs = 60 * 1000;
        const labelEvery = width >= 420 ? 1 : 2;
        for (let age = MONITOR_CHART_WINDOW_MS; age >= 0; age -= minuteMs) {
            const px = Math.max(0.5, Math.min(width - 0.5, x(now - age)));
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, plotH);
            ctx.stroke();
            const minutes = age / minuteMs;
            if (minutes % labelEvery !== 0 && minutes !== 0) continue;
            const label = minutes === 0 ? 'now' : `−${minutes}m`;
            ctx.textAlign = minutes === 0 ? 'right' : (age === MONITOR_CHART_WINDOW_MS ? 'left' : 'center');
            ctx.fillText(label, px, plotH + 2);
        }

        ctx.fillStyle = accentSoft;
        let bandStart = null;
        monitor.points.forEach((p) => {
            if (p.on) {
                if (bandStart === null) bandStart = p.at;
                return;
            }
            if (bandStart !== null) {
                ctx.fillRect(x(bandStart), 0, Math.max(1, x(p.at) - x(bandStart)), plotH);
                bandStart = null;
            }
        });
        if (bandStart !== null) {
            ctx.fillRect(x(bandStart), 0, Math.max(1, x(now) - x(bandStart)), plotH);
        }

        ctx.strokeStyle = dim;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        monitor.points.forEach((p, i) => {
            const px = x(p.at);
            i === 0 ? ctx.moveTo(px, y(p.t)) : ctx.lineTo(px, y(p.t));
        });
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.lineWidth = 2;
        ctx.strokeStyle = accent;
        ctx.beginPath();
        monitor.points.forEach((p, i) => {
            const px = x(p.at);
            i === 0 ? ctx.moveTo(px, y(p.m)) : ctx.lineTo(px, y(p.m));
        });
        ctx.stroke();
    }

    function monitorResizeChart() {
        const canvas = $('.js-mon-chart');
        const rect = canvas.getBoundingClientRect();
        if (rect.width > 0 && canvas.width !== Math.round(rect.width)) {
            canvas.width = Math.round(rect.width);
            monitorDrawChart();
        }
    }

    function markMonitorReady(readiness) {
        if (!monitor.inputMode) return;
        if (!monitor.readyState) monitor.readyAt = Date.now();
        monitor.readyState = readiness;
        if (monitor.readyTracked) return;
        monitor.readyTracked = track('tool_ready', {
            tool_name: 'monitor',
            entry_point: monitor.entryPoint || 'monitor',
            transport: monitor.inputMode === 'mqtt' ? 'mqtt_websocket' : 'simulation',
            input_mode: monitor.inputMode,
            readiness,
            latency_ms: Math.max(0, monitor.readyAt - (monitor.startedAt || monitor.connectedAt))
        });
    }

    function monitorStopAll(reason = 'replaced') {
        if (monitor.inputMode && monitor.connectedAt) {
            track('tool_disconnect', {
                tool_name: 'monitor',
                entry_point: monitor.entryPoint || 'monitor',
                transport: monitor.inputMode === 'mqtt' ? 'mqtt_websocket' : 'simulation',
                input_mode: monitor.inputMode,
                reason,
                duration_seconds: Math.max(0, Math.round((Date.now() - monitor.connectedAt) / 1000))
            });
        }
        stopMqttTransport();
    }

    function resetMonitorDevicePicker() {
        const picker = $('.js-mon-device-picker');
        const choice = document.getElementById('mon-device-choice');
        if (picker) picker.hidden = true;
        if (!choice) return;
        clearMonitorFieldError(choice);
        choice.replaceChildren();
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select a device';
        choice.appendChild(placeholder);
        choice.value = '';
    }

    function monitorExtractDeviceIdFromTopic(topic, prefix) {
        const root = prefix + '/';
        if (!topic.startsWith(root)) return '';
        const parts = topic.slice(root.length).split('/');
        if (parts.length < 2 || !parts[0]) return '';
        if (parts[1] !== 'info' && parts[1] !== 'status') return '';
        return parts[0];
    }

    function recordDiscoveredMqttDevice(topic, payload) {
        const prefix = monitor.discoveryPrefix;
        if (!prefix) return;
        const topicName = mqttUtf8(topic);
        const topicId = monitorExtractDeviceIdFromTopic(topicName, prefix);
        if (!topicId) return;
        let data;
        try {
            data = JSON.parse(mqttUtf8(payload).trim());
        } catch (error) {
            return;
        }
        if (!data || typeof data !== 'object') return;
        const device = monitor.discoveredDevices[topicId] || {
            topic_id: topicId,
            device_id: topicId
        };
        if (data.device_id) device.device_id = String(data.device_id);
        if (topicName.endsWith('/info')) {
            ['device_name', 'device_label', 'frontend', 'chip'].forEach((key) => {
                if (data[key]) device[key] = data[key];
            });
        } else if (topicName.endsWith('/status') && 'online' in data) {
            device.online = data.online === true;
        }
        monitor.discoveredDevices[topicId] = device;
    }

    function monitorDeviceChoiceLabel(device) {
        const label = device.device_label || device.device_name || 'unnamed';
        const frontend = device.frontend || 'unknown';
        const online = device.online ? 'online' : 'offline/unknown';
        return device.device_id + ' · ' + label + ' · ' + frontend + ' · ' + online;
    }

    function populateMonitorDevicePicker(devices) {
        const picker = $('.js-mon-device-picker');
        const choice = document.getElementById('mon-device-choice');
        if (!choice) return;
        choice.replaceChildren();
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select a device';
        choice.appendChild(placeholder);
        devices.forEach((device) => {
            const option = document.createElement('option');
            option.value = device.topic_id || device.device_id;
            option.textContent = monitorDeviceChoiceLabel(device);
            choice.appendChild(option);
        });
        if (picker) picker.hidden = false;
        choice.focus({ preventScroll: true });
    }

    function monitorUnsubscribeDiscovery(client) {
        if (!client || typeof client.unsubscribe !== 'function' || !monitor.discoveryTopics.length) {
            monitor.discoveryTopics = [];
            return;
        }
        client.unsubscribe(monitor.discoveryTopics);
        monitor.discoveryTopics = [];
    }

    function monitorSelectDevice(deviceId) {
        const client = monitor.client;
        const prefix = document.getElementById('mon-topic-prefix').value.trim().replace(/\/+$/, '');
        const device = String(deviceId || '').trim().replace(/^\/+|\/+$/g, '');
        const deviceInput = document.getElementById('mon-device');
        if (!client || !prefix || !device) return;
        if (device.includes('/') || /[+#]/.test(device)) {
            if (deviceInput) markMonitorFieldError(deviceInput, 'Enter a device ID without / or wildcards.');
            return;
        }
        if (deviceInput) {
            deviceInput.value = device;
            clearMonitorFieldError(deviceInput);
        }
        resetMonitorDevicePicker();
        monitor.discoveryActive = false;
        if (monitor.discoveryTimer) {
            clearTimeout(monitor.discoveryTimer);
            monitor.discoveryTimer = 0;
        }
        monitorUnsubscribeDiscovery(client);
        syncMonitorDemoButton();
        if (conn.status === 'disconnected') {
            rememberConnectionOrigin();
            setStatus('connecting');
        }
        if (bleClient) monitor.closingBleForLive = true;
        monitorBindSelectedDevice(client, prefix, device);
    }

    function monitorFinishDiscovery(client) {
        monitor.discoveryActive = false;
        syncMonitorDemoButton();
        const devices = Object.values(monitor.discoveredDevices)
            .sort((a, b) => a.device_id.localeCompare(b.device_id));
        if (devices.length === 1) {
            monitorStatus('Selected device: ' + devices[0].device_id);
            monitorSelectDevice(devices[0].topic_id || devices[0].device_id);
            return;
        }
        if (devices.length > 1) {
            populateMonitorDevicePicker(devices);
            monitorStatus('Select a device, or enter a device ID.');
            return;
        }
        const deviceInput = document.getElementById('mon-device');
        monitorStatus('No devices discovered. Enter a device ID.');
        if (deviceInput) {
            markMonitorFieldError(deviceInput, 'Enter a device ID.');
            deviceInput.focus({ preventScroll: true });
        }
    }

    function monitorStartDiscovery(client, prefix) {
        resetMonitorDevicePicker();
        monitor.discoveryActive = true;
        monitor.discoveredDevices = {};
        monitor.discoveryPrefix = prefix;
        const infoTopic = prefix + '/+/info';
        const statusTopic = prefix + '/+/status';
        monitor.discoveryTopics = [infoTopic, statusTopic];
        monitorStatus('Scanning MQTT for devices…');
        toast('Scanning MQTT for devices…');
        syncMonitorDemoButton();
        client.subscribe([infoTopic, statusTopic], (error) => {
            if (monitor.client !== client) return;
            if (error) {
                monitor.discoveryActive = false;
                monitor.discoveryTopics = [];
                monitorStatus('Subscribe failed: ' + error.message);
                syncMonitorDemoButton();
                track('tool_connection', {
                    tool_name: 'monitor',
                    entry_point: monitor.entryPoint,
                    transport: 'mqtt_websocket',
                    result: 'subscription_failure',
                    error_type: errorType(error)
                });
                return;
            }
            if (monitor.discoveryTimer) clearTimeout(monitor.discoveryTimer);
            monitor.discoveryTimer = setTimeout(() => {
                monitor.discoveryTimer = 0;
                if (monitor.client !== client || !monitor.discoveryActive) return;
                monitorFinishDiscovery(client);
            }, MONITOR_DISCOVERY_TIMEOUT_MS);
        });
    }

    function monitorBindSelectedDevice(client, prefix, device) {
        const base = prefix + '/' + device;
        monitor.baseTopic = base;
        monitor.boundDeviceId = device;
        monitor.inputMode = 'mqtt';
        client.subscribe(base + '/#', async (error) => {
            if (monitor.client !== client) return;
            monitorStatus(error
                ? 'Subscribe failed: ' + error.message
                : 'Broker connected. Waiting for device telemetry…');
            track('tool_connection', {
                tool_name: 'monitor',
                entry_point: monitor.entryPoint,
                transport: 'mqtt_websocket',
                result: error ? 'subscription_failure' : 'success',
                ...(error ? { error_type: errorType(error) } : {})
            });
            if (error) {
                monitor.closingBleForLive = false;
                monitorStopAll('subscription_failure');
                if (conn.status === 'connecting' && conn.mode !== 'ble') setStatus('disconnected');
                return;
            }
            syncMonitorDemoButton();
            monitorResizeChart();
            const stopped = await stopBleForDetection();
            if (!stopped || monitor.client !== client) {
                monitor.closingBleForLive = false;
                if (conn.mode === 'ble') setDeviceView('connectivity');
                return;
            }
            await ensureBleOffForLive({
                statusFn: (message) => {
                    if (message) monitorStatus(message);
                }
            });
            if (monitor.client !== client) {
                monitor.closingBleForLive = false;
                return;
            }
            monitor.handoffReady = true;
            conn.mode = 'mqtt';
            monitor.closingBleForLive = false;
            setDeviceView('live');
            setStatus('connecting');
            monitorPublishCommand({ command: 'commands' }, {
                pendingMessage: 'Reading device capabilities…',
                statusFn: monitorStatus
            }).catch(() => {});
            monitorPublishCommand({ command: 'info' }, {
                pendingMessage: 'Reading device information…',
                statusFn: monitorStatus
            }).catch(() => {});
            monitorPublishCommand({ command: 'ota_status' }, {
                pendingMessage: 'Reading firmware status…',
                statusFn: () => {}
            }).catch(() => {});
            startSilentOtaCheck();
        });
    }

    async function monitorConnect() {
        const connection = validateMonitorConnection();
        if (!connection) {
            monitor.closingBleForLive = false;
            track('tool_connection', {
                tool_name: 'monitor', entry_point: route,
                transport: 'mqtt_websocket', result: 'validation_failure'
            });
            if (conn.status === 'connecting' && !bleClient) setStatus('disconnected');
            return;
        }
        try {
            await loadBrowserDependency(
                '/vendor/mqtt-5.3.0/mqtt.min.js',
                'https://unpkg.com/mqtt@5.3.0/dist/mqtt.min.js'
            );
        } catch (error) {
            monitorStatus('The local MQTT client could not be loaded.');
            monitor.closingBleForLive = false;
            track('tool_connection', {
                tool_name: 'monitor', entry_point: route,
                transport: 'mqtt_websocket', result: 'dependency_failure',
                error_type: errorType(error)
            });
            if (conn.status === 'connecting' && !bleClient) setStatus('disconnected');
            return;
        }
        const { host, port, path, tls, prefix, device } = connection;
        const url = (tls ? 'wss://' : 'ws://') + host + ':' + port + path;
        if (monitor.client && monitor.brokerUrl === url && !monitorIsMqttLive()) {
            if (device) {
                monitorSelectDevice(device);
                return;
            }
            if (!monitor.discoveryActive) monitorStartDiscovery(monitor.client, prefix);
            return;
        }
        if (device && conn.status === 'disconnected') {
            rememberConnectionOrigin();
            setStatus('connecting');
        }
        monitorStopAll('replaced');
        monitor.closing = false;
        resetMonitorLiveView();
        monitor.baseTopic = device ? connection.base : null;
        monitor.boundDeviceId = device || '';
        monitor.handoffReady = false;
        monitor.startedAt = Date.now();
        monitor.entryPoint = route;
        monitor.readyState = '';
        monitor.readyAt = 0;
        monitor.readyTracked = false;
        monitor.brokerUrl = url;
        monitorStatus('Connecting to ' + url + ' …');
        toast('Connecting to the broker…');
        // The URL is not tracked: it would carry the user's broker address.
        track('tool_connection', {
            tool_name: 'monitor', entry_point: route,
            transport: 'mqtt_websocket', result: 'attempt'
        });
        const client = window.mqtt.connect(url, {
            username: document.getElementById('mon-user').value || undefined,
            password: document.getElementById('mon-pass').value || undefined,
            clientId: 'espectre-mockup-' + Math.random().toString(16).slice(2, 8),
            connectTimeout: 8000,
            reconnectPeriod: 0,
            protocolVersion: 4
        });
        monitor.client = client;
        client.on('connect', () => {
            if (monitor.client !== client) return;
            monitor.connectedAt = Date.now();
            if (device) {
                monitorBindSelectedDevice(client, prefix, device);
                return;
            }
            monitorStartDiscovery(client, prefix);
        });
        client.on('message', (topic, payload) => {
            if (monitor.client !== client) return;
            if (monitor.discoveryActive) {
                recordDiscoveredMqttDevice(topic, payload);
                return;
            }
            if (!monitor.baseTopic) return;
            ingestMqttPayload(monitor.baseTopic, topic, payload);
        });
        client.on('error', (error) => {
            if (monitor.client !== client) return;
            monitorStatus('Connection failed: ' + error.message);
            monitor.closingBleForLive = false;
            track('tool_connection', {
                tool_name: 'monitor',
                entry_point: monitor.entryPoint,
                transport: 'mqtt_websocket',
                result: 'failure',
                error_type: errorType(error)
            });
            monitorStopAll('error');
            if (conn.mode === 'mqtt') {
                teardownConnection('error');
            } else if (conn.status === 'connecting' && !bleClient) {
                setStatus('disconnected');
            }
        });
        client.on('close', () => {
            if (monitor.client !== client || monitor.closing) return;
            if (conn.mode === 'ble') return;
            if (conn.mode !== 'mqtt') {
                monitorStatus('Disconnected from broker.');
                monitorStopAll('unexpected');
                return;
            }
            monitorStatus('Disconnected from broker.');
            teardownConnection('unexpected');
        });
    }

    function monitorPublishCommand(fields, {
        pendingMessage = 'Sending command…',
        statusFn = monitorStatus,
        timeoutMs = 8000
    } = {}) {
        if (!monitorIsMqttLive() || !monitor.baseTopic) {
            const error = new Error('Connect through the broker before changing the device.');
            statusFn(error.message);
            return Promise.reject(error);
        }
        const commandId = 'web-' + Date.now() + '-' + Math.random().toString(16).slice(2, 7);
        const command = JSON.stringify({
            protocol_version: '1.0',
            command_id: commandId,
            ...fields
        });
        statusFn(pendingMessage);
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                monitor.pendingCommands.delete(commandId);
                reject(new Error('The device did not confirm the command in time.'));
            }, timeoutMs);
            monitor.pendingCommands.set(commandId, { resolve, reject, timer, command: fields.command });
            monitor.client.publish(
                monitor.baseTopic + '/commands/request',
                command,
                { qos: 0, retain: false },
                (error) => {
                    if (!error) return;
                    clearTimeout(timer);
                    monitor.pendingCommands.delete(commandId);
                    reject(error);
                }
            );
        });
    }

    function diagnosticsRequestPending() {
        for (const pending of monitor.pendingCommands.values()) {
            if (pending.command === 'stats') return true;
        }
        return false;
    }

    function diagnosticsPanelOpen() {
        const panel = $('.device-live-diagnostics');
        const workspace = $('.js-monitor-workspace');
        return !!(panel && panel.open && route === 'monitor' && workspace && !workspace.hidden);
    }

    function stopDiagnosticsPolling() {
        if (!monitor.diagTimer) return;
        clearInterval(monitor.diagTimer);
        monitor.diagTimer = null;
    }

    function syncDiagnosticsPolling() {
        const canPoll = diagnosticsPanelOpen()
            && (conn.mode === 'demo' || monitorIsMqttLive());
        if (!canPoll) {
            stopDiagnosticsPolling();
            return;
        }
        if (monitor.diagTimer) return;
        monitorRequestStats();
        monitor.diagTimer = setInterval(monitorRequestStats, 1000);
    }

    async function monitorRequestStats() {
        if (!diagnosticsPanelOpen()) {
            stopDiagnosticsPolling();
            return;
        }
        if (conn.mode === 'demo') {
            monitorStats({
                traffic_tx_pps: 100,
                csi_callback_pps: 96,
                csi_filtered_pps: 6,
                csi_admitted_pps: 84,
                wifi_channel: 10,
                wifi_rssi_dbm: -55,
                free_memory_kb: 161.4,
                loop_time_ms: 0.31
            });
            return;
        }
        if (diagnosticsRequestPending()) return;
        try {
            await monitorPublishCommand({ command: 'stats' }, {
                pendingMessage: '',
                statusFn: () => {}
            });
        } catch (error) {
            if (diagnosticsPanelOpen()) monitorDiagStatus(error.message);
        }
    }

    async function monitorStartBle() {
        setDeviceView('connectivity');
        renderConnection();
        if (monitor.bleRequested) {
            await connectBle();
            return;
        }
        if (conn.mode === 'demo') {
            setDeviceView('connectivity');
            return;
        }
        try {
            await monitorPublishCommand({ command: 'set_ble', ble: 'on' }, {
                pendingMessage: 'Opening Configure. Sensing will pause…'
            });
            monitor.bleRequested = true;
            renderConnection();
            await connectBle();
        } catch (error) {
            monitorStatus(error.message);
            toast(error.message);
        }
    }

    async function beginCalibration() {
        if (monitor.calibrating) return;
        setCalibrationBusy(true);
        if (conn.mode === 'demo') {
            toast('Calibration started. (demo)');
            scheduleCalibrationIdle(MONITOR_DEMO_CALIBRATION_MS);
            return;
        }
        try {
            const result = await monitorPublishCommand({ command: 'recalibrate' }, {
                pendingMessage: 'Starting calibration…',
                statusFn: toast
            });
            toast(result.message || 'Calibration started.');
            scheduleCalibrationIdle(MONITOR_CALIBRATION_FALLBACK_MS);
        } catch (error) {
            toast(error.message);
            setCalibrationBusy(false);
        }
    }

    async function runSensingCommand(fields, pendingMessage, successMessage, demoUpdate) {
        if (conn.mode === 'demo') {
            if (demoUpdate) demoUpdate();
            toast(successMessage + ' (demo)');
            return;
        }
        try {
            const result = await monitorPublishCommand(fields, { pendingMessage, statusFn: toast });
            toast(result.message || successMessage);
        } catch (error) {
            toast(error.message);
        }
    }

    function monitorInit() {
        $('.js-mon-connect').addEventListener('click', monitorConnect);
        ['mon-host', 'mon-port', 'mon-topic-prefix', 'mon-device', 'mon-path'].forEach((id) => {
            const input = document.getElementById(id);
            input.addEventListener('input', () => clearMonitorFieldError(input));
        });
        const deviceChoice = document.getElementById('mon-device-choice');
        if (deviceChoice) {
            deviceChoice.addEventListener('change', () => {
                const selected = deviceChoice.value.trim();
                if (!selected) return;
                clearMonitorFieldError(deviceChoice);
                monitorSelectDevice(selected);
            });
        }
        const diagnostics = $('.device-live-diagnostics');
        if (diagnostics) {
            diagnostics.addEventListener('toggle', syncDiagnosticsPolling);
        }
        $('.js-device-edit-connectivity').addEventListener('click', monitorStartBle);
        $$('.js-firmware-update-notice').forEach((button) => {
            button.addEventListener('click', (event) => otaOpen(event.currentTarget));
        });
        document.getElementById('sense-threshold').addEventListener('change', () => {
            const threshold = Number(document.getElementById('sense-threshold').value);
            if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
                toast('Threshold must be between 0 and 1.');
                return;
            }
            runSensingCommand(
                { command: 'set_threshold', threshold },
                'Applying threshold…',
                'Threshold updated.',
                () => { conn.threshold = threshold; renderTelemetry(); }
            );
        });
        document.getElementById('sense-detector').addEventListener('change', () => {
            const detector = document.getElementById('sense-detector').value;
            syncSensingControls();
            runSensingCommand(
                { command: 'set_detector', detector },
                'Applying detection profile…',
                'Detection profile updated.'
            );
        });
        const applyMotionHits = () => {
            const motionOnHits = Number(document.getElementById('sense-motion-on').value);
            const motionOffHits = Number(document.getElementById('sense-motion-off').value);
            if (![motionOnHits, motionOffHits].every((value) => Number.isInteger(value) && value >= 1 && value <= 20)) {
                toast('Motion stability values must be whole numbers from 1 to 20.');
                return;
            }
            runSensingCommand(
                { command: 'set_motion_hits', motion_on_hits: motionOnHits, motion_off_hits: motionOffHits },
                'Applying motion stability…',
                'Motion stability updated.'
            );
        };
        document.getElementById('sense-motion-on').addEventListener('change', applyMotionHits);
        document.getElementById('sense-motion-off').addEventListener('change', applyMotionHits);
        $('.js-sense-recalibrate').addEventListener('click', beginCalibration);
        document.getElementById('sense-csi-mode').addEventListener('change', () => {
            const csiTrafficMode = document.getElementById('sense-csi-mode').value;
            runSensingCommand(
                { command: 'set_csi_traffic_mode', csi_traffic_mode: csiTrafficMode },
                'Applying traffic ownership…',
                'Traffic ownership updated.'
            );
        });
        document.getElementById('sense-generator-mode').addEventListener('change', () => {
            const trafficGeneratorMode = document.getElementById('sense-generator-mode').value;
            runSensingCommand(
                { command: 'set_traffic_generator_mode', traffic_generator_mode: trafficGeneratorMode },
                'Applying traffic generator…',
                'Traffic generator updated.'
            );
        });
        window.addEventListener('resize', monitorResizeChart);
    }

    /* ============================================================ theremin */

    const theremin = { ctx: null, osc: null, gain: null, raf: null, smoothed: 0 };

    function thereminStart() {
        if (theremin.ctx) return;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            toast('Web Audio is not available in this browser.');
            return;
        }
        theremin.ctx = new AudioCtx();
        theremin.osc = theremin.ctx.createOscillator();
        theremin.gain = theremin.ctx.createGain();
        theremin.osc.type = document.getElementById('th-wave').value;
        theremin.osc.frequency.value = 140;
        theremin.gain.gain.value = 0;
        theremin.osc.connect(theremin.gain).connect(theremin.ctx.destination);
        theremin.osc.start();
        $('.js-th-toggle').textContent = '⏹ Stop sound';
        const loop = () => {
            const f = energyFraction();
            theremin.smoothed += (f - theremin.smoothed) * 0.12;
            const freq = 140 * Math.pow(2, theremin.smoothed * 2.6);
            const now = theremin.ctx.currentTime;
            theremin.osc.frequency.setTargetAtTime(freq, now, 0.05);
            theremin.gain.gain.setTargetAtTime(0.02 + theremin.smoothed * 0.35, now, 0.08);
            $('.js-th-freq').textContent = Math.round(freq);
            theremin.raf = requestAnimationFrame(loop);
        };
        loop();
    }

    function thereminStop() {
        if (!theremin.ctx) return;
        cancelAnimationFrame(theremin.raf);
        theremin.osc.stop();
        theremin.ctx.close();
        theremin.ctx = null;
        theremin.osc = null;
        theremin.gain = null;
        theremin.smoothed = 0;
        const toggle = $('.js-th-toggle');
        if (toggle) toggle.textContent = '▶ Start sound';
        const freq = $('.js-th-freq');
        if (freq) freq.textContent = '—';
    }

    function thereminInit() {
        $('.js-th-toggle').addEventListener('click', () => {
            const starting = !theremin.ctx;
            starting ? thereminStart() : thereminStop();
            track('theremin_configuration', {
                control: 'playback', setting_value: starting ? 'start' : 'stop'
            });
        });
        document.getElementById('th-wave').addEventListener('change', (event) => {
            if (theremin.osc) theremin.osc.type = event.target.value;
            track('theremin_configuration', {
                control: 'waveform', setting_value: event.target.value
            });
        });
    }

    /* =========================================================== configure */

    function cfgValue(id) {
        return document.getElementById(id).value;
    }

    /**
     * Builds and writes a control command, reporting the outcome as
     * configure_change. `buildCommand` runs even in demo mode so the
     * library's validation gives the same feedback with or without a
     * device; nothing is written and nothing is tracked in demo, because
     * no device is involved.
     */
    function finishConfigVerification(result, errorType) {
        if (!pendingConfigVerification) return;
        clearTimeout(pendingConfigVerification.timer);
        const action = pendingConfigVerification.action;
        pendingConfigVerification = null;
        track('configure_change', {
            action,
            result,
            ...(errorType ? { error_type: errorType } : {})
        });
    }

    function requestConfigVerification() {
        const pending = pendingConfigVerification;
        if (!pending || !bleClient) {
            finishConfigVerification('unconfirmed', 'VerificationUnavailable');
            return;
        }
        pending.attempts += 1;
        bleClient.requestSysinfo().catch(() => {
            finishConfigVerification('unconfirmed', 'VerificationRequestFailed');
        });
        pending.timer = setTimeout(() => {
            if (pendingConfigVerification !== pending) return;
            if (pending.attempts >= 3) finishConfigVerification('unconfirmed', 'VerificationTimeout');
            else requestConfigVerification();
        }, 700);
    }

    function beginConfigVerification(action, verify) {
        if (pendingConfigVerification) finishConfigVerification('unconfirmed', 'Superseded');
        pendingConfigVerification = { action, verify, attempts: 0, timer: null };
        pendingConfigVerification.timer = setTimeout(requestConfigVerification, 180);
    }

    function evaluateConfigVerification(snapshot) {
        const pending = pendingConfigVerification;
        if (!pending) return;
        clearTimeout(pending.timer);
        if (pending.verify(snapshot)) {
            finishConfigVerification('success');
        } else if (pending.attempts >= 3) {
            finishConfigVerification('unconfirmed', 'VerificationMismatch');
        } else {
            pending.timer = setTimeout(requestConfigVerification, 250);
        }
    }

    async function cfgApply(action, successMessage, buildCommand, verify) {
        let command;
        try {
            command = buildCommand();
        } catch (error) {
            if (error && error.name === 'ESPectreValidationError') {
                cfgValidationFailed(action, error.message);
                return false;
            }
            throw error;
        }
        if (conn.mode === 'demo') {
            toast(successMessage + ' (demo — nothing written)');
            return true;
        }
        if (!bleClient) {
            toast('ESPectre is not connected.');
            track('configure_change', { action, result: 'failure', error_type: 'NotConnected' });
            return false;
        }
        try {
            await bleClient.writeControl(command);
            toast(successMessage);
            track('configure_change', { action, result: 'accepted' });
            if (verify) beginConfigVerification(action, verify);
            return true;
        } catch (error) {
            toast('Write failed: ' + (error.message || error));
            track('configure_change', { action, result: 'failure', error_type: errorType(error) });
            return false;
        }
    }

    function cfgValidationFailed(action, message) {
        toast(message);
        track('configure_change', { action, result: 'validation_failure' });
    }

    async function cfgRefreshSysinfo() {
        if (conn.mode === 'ble' && bleClient) {
            try {
                await bleClient.requestSysinfo();
            } catch (error) {
                console.warn('Sysinfo request failed:', error);
            }
        }
    }

    async function cfgSaveWifi() {
        const ssid = cfgValue('cfg-ssid').trim();
        const password = cfgValue('cfg-wifi-pass');
        if (!ssid) {
            cfgValidationFailed('set_wifi', 'Wi-Fi needs an SSID.');
            return;
        }
        const bandPolicy = cfgValue('cfg-wifi-band');
        const bandChanged = wifiBandPolicyAvailable && bandPolicy !== currentWifiBandPolicy;
        const ok = await cfgApply(
            'set_wifi',
            bandChanged ? 'Wi-Fi saved; restart required to apply the band.' : 'Wi-Fi saved; station reconnecting.',
            () => window.ESPectreBleClient.buildWifiConfigCommand({
                ssid,
                password,
                bssid: cfgValue('cfg-bssid').trim(),
                channel: Number(cfgValue('cfg-channel') || 0),
                ...(wifiBandPolicyAvailable ? { bandPolicy } : {})
            }),
            (snapshot) => snapshot.wifi_ssid === ssid
                && (!wifiBandPolicyAvailable || snapshot.wifi_band_policy === bandPolicy));
        if (ok) {
            document.getElementById('cfg-wifi-pass').value = '';
            if (bandChanged) {
                currentWifiBandPolicy = bandPolicy;
                $('.js-wifi-restart-note').hidden = false;
            }
        }
    }

    async function cfgClearWifi() {
        const ok = await cfgApply(
            'clear_wifi', 'Wi-Fi credentials cleared.', () => 'CLEAR_WIFI',
            (snapshot) => !snapshot.wifi_ssid);
        if (ok) {
            ['cfg-ssid', 'cfg-wifi-pass', 'cfg-bssid', 'cfg-channel'].forEach((id) => {
                document.getElementById(id).value = '';
            });
        }
    }

    async function cfgSaveMqtt() {
        const host = cfgValue('cfg-mqtt-host').trim();
        const username = cfgValue('cfg-mqtt-user').trim();
        const password = cfgValue('cfg-mqtt-pass');
        if (!host || !cfgValue('cfg-mqtt-port')) {
            cfgValidationFailed('set_mqtt', 'MQTT needs a host and port.');
            return;
        }
        const port = Number(cfgValue('cfg-mqtt-port'));
        const ok = await cfgApply('set_mqtt', 'MQTT settings saved.',
            () => window.ESPectreBleClient.buildMqttConfigCommand({
                host,
                port,
                username,
                password,
                topicPrefix: cfgValue('cfg-topic-prefix').trim().replace(/\/+$/, '') || undefined
            }),
            (snapshot) => snapshot.mqtt_host === host && Number(snapshot.mqtt_port) === port);
        if (ok) document.getElementById('cfg-mqtt-pass').value = '';
    }

    function applyConfigureMqttDefaults() {
        document.getElementById('cfg-mqtt-host').value = MQTT_FORM_DEFAULTS.host;
        document.getElementById('cfg-mqtt-port').value = MQTT_FORM_DEFAULTS.port;
        document.getElementById('cfg-mqtt-user').value = MQTT_FORM_DEFAULTS.username;
        document.getElementById('cfg-mqtt-pass').value = MQTT_FORM_DEFAULTS.password;
        document.getElementById('cfg-topic-prefix').value = MQTT_FORM_DEFAULTS.topicPrefix;
    }

    async function cfgClearMqtt() {
        const ok = await cfgApply(
            'clear_mqtt', 'MQTT settings cleared.', () => 'CLEAR_MQTT_CONFIG',
            (snapshot) => !snapshot.mqtt_host);
        if (ok) applyConfigureMqttDefaults();
    }

    async function cfgSaveDevice() {
        const label = cfgValue('cfg-label').trim();
        await cfgApply('set_device', 'Device label saved.',
            () => window.ESPectreBleClient.buildDeviceLabelCommand(label),
            (snapshot) => (snapshot.device_label || '') === label);
    }

    function finishOtaTracking(result, errorType, state) {
        if (!otaTracking) return;
        clearTimeout(otaPollTimer);
        const startedAt = otaTracking.startedAt;
        otaTracking = null;
        otaPollTimer = null;
        track('ota_update_result', {
            result,
            ota_state: state || 'unknown',
            duration_ms: Math.max(0, Date.now() - startedAt),
            channel: selectedOtaChannel(),
            ...(errorType ? { error_type: errorType } : {})
        });
    }

    function pollOtaStatus() {
        if (!otaTracking) return;
        if (!bleClient) {
            finishOtaTracking('unconfirmed', 'StatusUnavailable', otaTracking.lastState);
            return;
        }
        otaTracking.attempts += 1;
        if (otaTracking.attempts > 120) {
            finishOtaTracking('unconfirmed', 'StatusTimeout', otaTracking.lastState);
            return;
        }
        bleClient.requestSysinfo().catch(() => {
            finishOtaTracking('unconfirmed', 'StatusRequestFailed', otaTracking?.lastState);
        });
        otaPollTimer = setTimeout(pollOtaStatus, 1000);
    }

    function beginOtaTracking() {
        if (otaTracking) finishOtaTracking('unconfirmed', 'Superseded', otaTracking.lastState);
        otaTracking = { startedAt: Date.now(), attempts: 0, lastState: 'starting' };
        clearTimeout(otaPollTimer);
        otaPollTimer = setTimeout(pollOtaStatus, 250);
    }

    function evaluateOtaTracking(snapshot) {
        if (!otaTracking || !snapshot.ota_state) return;
        const state = String(snapshot.ota_state).toLowerCase();
        otaTracking.lastState = state;
        if (state === 'reboot_scheduled') {
            finishOtaTracking('success', null, state);
        } else if (state === 'error') {
            finishOtaTracking('failure', 'DeviceReportedError', state);
        }
    }

    function otaModalDescriptionElement() {
        const modal = $('.js-ota-modal');
        return modal ? modal.querySelector('.modal-description') : null;
    }

    function setOtaModalDescription(text) {
        const description = otaModalDescriptionElement();
        if (description && text) description.textContent = text;
    }

    function syncOtaModalDescription() {
        const modal = $('.js-ota-modal');
        if (!modal || modal.hidden) return;
        const state = String(otaState || '').toLowerCase();
        if (state === 'downloading') {
            setOtaModalDescription('Downloading firmware…');
        } else if (state === 'applying') {
            setOtaModalDescription('Applying firmware…');
        } else if (state === 'reboot_scheduled') {
            setOtaModalDescription('Update applied. Waiting for the device to come back online…');
        } else if (state === 'error') {
            setOtaModalDescription(otaMessage && otaMessage !== '—' ? otaMessage : 'Update failed.');
        }
    }

    function completeOtaReconnect() {
        if (!otaAwaitingReconnect) return;
        otaAwaitingReconnect = false;
        otaBusy = false;
        const version = conn.firmwareVersion || (latestDeviceInfo && latestDeviceInfo.firmware_version) || '';
        applyOtaStatus({
            state: 'idle',
            busy: false,
            current_version: version || undefined,
            target_version: '',
            update_available: false,
            message: version ? ('Back online · firmware ' + version) : 'Back online after update'
        });
        setOtaModalDescription(version
            ? 'Update applied. Device is back online on ' + version + '.'
            : 'Update applied. Device is back online.');
        otaClose();
        toast(version
            ? 'Device is back online on ' + version + '.'
            : 'Device is back online after the update.');
        otaCheckTransport = '';
        startSilentOtaCheck();
    }

    function handleOtaDeviceAvailability(online) {
        if (!online) {
            if (otaState === 'reboot_scheduled') {
                otaAwaitingReconnect = true;
                setOtaModalDescription('Update applied. Waiting for the device to come back online…');
                monitorStatus('Device rebooting after the update…');
            }
            return;
        }
        if (otaAwaitingReconnect || otaState === 'reboot_scheduled') {
            otaAwaitingReconnect = true;
            completeOtaReconnect();
        }
    }

    function selectedOtaChannel() {
        const el = document.getElementById('ota-channel');
        const value = (el && el.value ? String(el.value) : '').trim();
        return value || 'release';
    }

    function otaCommandFields(command) {
        return { command, channel: selectedOtaChannel() };
    }

    function otaBleOptions() {
        return { channel: selectedOtaChannel() };
    }

    function syncOtaUpdateButton() {
        const button = $('.js-ota-start');
        if (!button) return;
        button.disabled = conn.mode === 'demo' || otaActionPending || otaBusy || !otaUpdateAvailable;
        button.textContent = otaBusy ? 'Update in progress…' : 'Update device';
    }

    function syncFirmwareUpdateNotice() {
        const target = (document.getElementById('cfg-ota-target')?.textContent || '').trim();
        const state = String(otaState || '').toLowerCase();
        let status = 'idle';
        let copy = 'Checking for updates…';
        if (state === 'checking' || state === 'idle' || state === '') {
            status = 'idle';
            copy = 'Checking for updates…';
        } else if (otaBusy || state === 'downloading' || state === 'applying') {
            status = 'busy';
            copy = 'Updating…';
        } else if (state === 'reboot_scheduled') {
            status = 'busy';
            copy = 'Reboot scheduled';
        } else if (state === 'error') {
            status = 'error';
            copy = otaMessage && otaMessage !== '—' ? otaMessage : 'Unable to check for updates';
        } else if (otaUpdateAvailable || state === 'update_available') {
            status = 'update';
            copy = target && target !== '—'
                ? 'Update available · ' + target
                : 'Update available';
        } else if (state === 'up_to_date') {
            status = 'latest';
            copy = 'Latest';
        }
        $$('.js-firmware-update-copy').forEach((el) => { el.textContent = copy; });
        $$('.js-firmware-update-notice').forEach((el) => {
            el.dataset.otaStatus = status;
            el.hidden = otaSupported === false;
        });
    }

    function applyOtaStatus(status) {
        if (!status || typeof status !== 'object') return;
        const write = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== '') el.textContent = value;
        };
        const state = status.state;
        if (state) {
            write('cfg-ota-state', state);
            otaState = String(state).toLowerCase();
        }
        if (status.current_version) write('cfg-ota-current', status.current_version);
        if (status.target_version !== undefined) write('cfg-ota-target', status.target_version || '—');
        if (status.message !== undefined) {
            otaMessage = status.message || '';
            write('cfg-ota-message', otaMessage || '—');
        }
        const normalizedState = String(state || '').toLowerCase();
        if (status.update_available !== undefined) {
            otaUpdateAvailable = sysinfoBoolean(status.update_available) || normalizedState === 'update_available';
            write('cfg-ota-available', otaUpdateAvailable ? 'yes' : 'no');
        } else if (normalizedState === 'update_available') {
            otaUpdateAvailable = true;
            write('cfg-ota-available', 'yes');
        } else if (normalizedState === 'up_to_date') {
            otaUpdateAvailable = false;
            write('cfg-ota-available', 'no');
        }
        if (status.busy !== undefined) otaBusy = sysinfoBoolean(status.busy);
        if (normalizedState === 'reboot_scheduled') {
            otaAwaitingReconnect = true;
        } else if (otaAwaitingReconnect &&
                (normalizedState === 'idle' || normalizedState === 'up_to_date' ||
                    normalizedState === 'update_available' || normalizedState === 'checking')) {
            completeOtaReconnect();
            return;
        }
        evaluateOtaTracking({ ota_state: state });
        syncOtaUpdateButton();
        syncFirmwareUpdateNotice();
        syncOtaModalDescription();
    }

    function reportOtaCheckFailure() {
        applyOtaStatus({
            state: 'error',
            update_available: false,
            busy: false,
            message: 'Unable to check for updates'
        });
    }

    function currentOtaCheckTransport() {
        if (conn.mode === 'ble') return 'ble';
        if (conn.mode === 'mqtt') return 'mqtt';
        return '';
    }

    function runOtaCheck({ manual = false } = {}) {
        if (conn.mode === 'demo') return;
        const transport = currentOtaCheckTransport();
        if (!manual && transport && otaCheckTransport === transport) return;
        if (otaState === 'checking' && manual) return;
        otaState = 'checking';
        otaMessage = '';
        syncFirmwareUpdateNotice();
        if (transport === 'ble' && bleClient && typeof bleClient.otaCheck === 'function') {
            if (!manual) otaCheckTransport = 'ble';
            bleClient.otaCheck(otaBleOptions()).catch((error) => {
                console.warn('Silent OTA check failed:', error);
                reportOtaCheckFailure();
            });
            return;
        }
        if (transport !== 'mqtt' || !monitorIsMqttLive()) return;
        if (!manual) otaCheckTransport = 'mqtt';
        monitorPublishCommand(otaCommandFields('ota_check'), {
            pendingMessage: '',
            statusFn: () => {}
        }).catch((error) => {
            console.warn('Silent OTA check failed:', error);
            reportOtaCheckFailure();
        });
    }

    function startSilentOtaCheck() {
        runOtaCheck();
    }

    function startManualOtaCheck() {
        runOtaCheck({ manual: true });
    }

    function otaOpen(returnFocus) {
        const modal = $('.js-ota-modal');
        otaModalReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        document.body.classList.add('modal-open');
        modal.querySelector('.modal-card').focus();
    }

    function otaClose(restoreFocus = true) {
        const modal = $('.js-ota-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        document.body.classList.remove('modal-open');
        if (restoreFocus && otaModalReturnFocus && otaModalReturnFocus.isConnected) {
            otaModalReturnFocus.focus();
        }
        otaModalReturnFocus = null;
    }

    async function cfgOtaStart() {
        if (conn.mode === 'demo') return;
        otaActionPending = true;
        syncOtaUpdateButton();
        const description = $('.js-ota-modal') && $('.js-ota-modal').querySelector('.modal-description');
        if (monitorIsMqttLive() && conn.mode !== 'ble') {
            try {
                await monitorPublishCommand(otaCommandFields('ota_start'), {
                    pendingMessage: 'Starting firmware update…',
                    statusFn: (message) => { if (description) description.textContent = message; }
                });
                otaBusy = true;
                beginOtaTracking();
                toast('OTA update started.');
            } catch (error) {
                toast(error.message);
            }
            otaActionPending = false;
            syncOtaUpdateButton();
            syncFirmwareUpdateNotice();
            return;
        }
        const ok = await cfgApply('ota_start', 'OTA update started.',
            () => window.ESPectreBleClient.buildOtaStartCommand(otaBleOptions()));
        otaActionPending = false;
        if (ok) {
            otaBusy = true;
            if (conn.mode === 'ble') beginOtaTracking();
        }
        syncOtaUpdateButton();
        syncFirmwareUpdateNotice();
    }

    function configureInit() {
        $('.js-wifi-save').addEventListener('click', cfgSaveWifi);
        $('.js-wifi-clear').addEventListener('click', cfgClearWifi);
        const startBle = $('.js-cfg-start-ble');
        if (startBle) startBle.addEventListener('click', monitorStartBle);
        $('.js-mqtt-save').addEventListener('click', cfgSaveMqtt);
        $('.js-mqtt-clear').addEventListener('click', cfgClearMqtt);
        $('.js-dev-save').addEventListener('click', cfgSaveDevice);
        $('.js-ota-start').addEventListener('click', cfgOtaStart);
        const otaChannel = document.getElementById('ota-channel');
        if (otaChannel) {
            otaChannel.addEventListener('change', () => {
                if (conn.mode === null) return;
                startManualOtaCheck();
            });
        }
        $$('.js-ota-close').forEach((button) => {
            button.addEventListener('click', () => otaClose());
        });
        $('.js-ota-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) otaClose();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !$('.js-ota-modal').hidden) otaClose();
        });
    }

    /* ================================================================ game */

    const TOTAL_ROUNDS = 5;
    const game = {
        phase: 'idle',   // idle | hold | strike | done
        round: 0,
        score: 0,
        best: null,
        holdTimer: null,
        cooldownTimer: null,
        strikeAt: 0,
        strikeTimeout: null
    };

    function gameSet(selector, value) {
        const el = $(selector);
        if (el) el.textContent = value;
    }

    function gameOrb(state) {
        const orb = $('.js-game-orb');
        if (orb) orb.className = 'game-orb js-game-orb' + (state ? ' is-' + state : '');
    }

    function gameMsg(message) {
        gameSet('.js-game-msg', message);
    }

    function reportGameAbandon(reason) {
        if (game.phase === 'idle' || game.phase === 'done') return;
        track('game_abandon', {
            input_mode: connectionInputMode(),
            rounds: game.round,
            reason
        });
        gameReset();
    }

    function gameReset() {
        clearTimeout(game.holdTimer);
        clearTimeout(game.cooldownTimer);
        clearTimeout(game.strikeTimeout);
        game.phase = 'idle';
        game.round = 0;
        game.score = 0;
        gameOrb('');
        gameMsg('Stand still. Move fast. React to survive.');
        gameSet('.js-game-round', '—');
        gameSet('.js-game-ms', '—');
        gameSet('.js-game-score', '0');
        const start = $('.js-game-start');
        if (start) start.textContent = 'Start game';
    }

    function gameNextRound() {
        game.round += 1;
        if (game.round > TOTAL_ROUNDS) {
            game.phase = 'done';
            gameOrb('');
            gameMsg('Game over — final score ' + game.score + '. Play again?');
            $('.js-game-start').textContent = 'Play again';
            track('game_over', {
                input_mode: connectionInputMode(),
                score: game.score,
                rounds: TOTAL_ROUNDS,
                ...(game.best !== null ? { best_time: game.best } : {})
            });
            return;
        }
        game.phase = 'hold';
        gameSet('.js-game-round', game.round + ' / ' + TOTAL_ROUNDS);
        gameOrb('hold');
        gameMsg('Round ' + game.round + ' — the Spectre is watching... stay perfectly still.');
        $('.js-game-hint').textContent = conn.mode === 'demo'
            ? 'Demo mode: stay still, then move the mouse when the Spectre strikes.'
            : 'Freeze. Any motion now counts as a false start.';
        game.holdTimer = setTimeout(() => {
            game.phase = 'strike';
            game.strikeAt = performance.now();
            gameOrb('strike');
            gameMsg('⚡ MOVE! The Spectre strikes!');
            game.strikeTimeout = setTimeout(() => {
                if (game.phase !== 'strike') return;
                gameOrb('fail');
                gameMsg('Too slow — the Spectre hit you.');
                gameSet('.js-game-ms', 'miss');
                gameEndRound();
            }, 2500);
        }, 1800 + Math.random() * 2600);
    }

    function gameEndRound() {
        game.phase = 'cooldown';
        clearTimeout(game.holdTimer);
        clearTimeout(game.cooldownTimer);
        clearTimeout(game.strikeTimeout);
        const waitForDemoSettle = () => {
            const settleThreshold = Math.max(0.18, conn.threshold * 0.7);
            if (demoInputEnergy > 0 || conn.motion || conn.movement >= settleThreshold) {
                game.cooldownTimer = setTimeout(waitForDemoSettle, 220);
                return;
            }
            demoResetMotion();
            gameNextRound();
        };
        if (conn.mode === 'demo') {
            game.cooldownTimer = setTimeout(waitForDemoSettle, 900);
        } else {
            game.cooldownTimer = setTimeout(gameNextRound, 1700);
        }
    }

    function gameOnTelemetry() {
        if (route !== 'game') return;
        if (game.phase === 'hold' && conn.motion) {
            gameOrb('fail');
            gameMsg('False start! The Spectre saw you twitch.');
            gameSet('.js-game-ms', 'false start');
            gameEndRound();
        } else if (game.phase === 'strike' && conn.motion) {
            const ms = Math.round(performance.now() - game.strikeAt);
            const points = Math.max(0, 1500 - ms);
            game.score += points;
            if (game.best === null || ms < game.best) {
                game.best = ms;
                gameSet('.js-game-best', game.best + ' ms');
            }
            gameOrb('win');
            gameMsg('Hit! ' + ms + ' ms → +' + points + ' points');
            gameSet('.js-game-ms', ms + ' ms');
            gameSet('.js-game-score', String(game.score));
            gameEndRound();
        }
    }

    function gameInit() {
        $('.js-game-start').addEventListener('click', () => {
            reportGameAbandon('restart');
            clearTimeout(game.holdTimer);
            clearTimeout(game.cooldownTimer);
            clearTimeout(game.strikeTimeout);
            game.round = 0;
            game.score = 0;
            demoResetMotion();
            gameSet('.js-game-score', '0');
            $('.js-game-start').textContent = 'Restart';
            track('game_start', { input_mode: connectionInputMode() });
            gameNextRound();
        });
    }

    /* ================================================================ init */

    function init() {
        scrollyInit();

        renderBrowserSupport();

        $$('.js-connect-ble').forEach((btn) => btn.addEventListener('click', connectBle));
        $$('.js-start-detection').forEach((btn) => btn.addEventListener('click', () => startDetection()));
        $('.js-header-connect').addEventListener('click', () => {
            if (route === 'configure') {
                connectBle();
                return;
            }
            location.hash = '#configure';
        });
        $$('.js-demo').forEach((btn) => btn.addEventListener('click', connectDemo));
        $('.js-disconnect').addEventListener('click', disconnect);
        $('.js-dropdown-toggle').addEventListener('click', (event) => {
            event.stopPropagation();
            dropdownOpen = !dropdownOpen;
            renderConnection();
        });
        document.addEventListener('click', (event) => {
            if (dropdownOpen && !event.target.closest('.conn')) {
                dropdownOpen = false;
                renderConnection();
            }
        });
        configureInit();
        flashInit();
        monitorInit();
        thereminInit();
        gameInit();

        document.addEventListener('click', interceptCanonicalLinks);
        $('.skip-link').addEventListener('click', (event) => {
            event.preventDefault();
            focusRouteContent();
        });
        document.addEventListener('mousemove', demoTrackMouse, { passive: true });
        window.addEventListener('hashchange', onHashChange);
        setRoute((location.hash || '#home').slice(1), { force: true, focus: false });
    }

    document.addEventListener('espectre:analytics-enabled', () => {
        if (window.trackRouteView) window.trackRouteView(route, { sendPageView: false });
        if (conn.readyState) markToolReady(conn.readyState);
        if (monitor.readyState) markMonitorReady(monitor.readyState);
        if (conn.mode === 'ble' && bleClient) cfgRefreshSysinfo();
        if (route === 'flash') flashRefresh();
    });
    window.addEventListener('pagehide', (event) => {
        if (event.persisted) return;
        reportGameAbandon('page_exit');
        if (conn.mode) teardownConnection('page_exit');
        else monitorStopAll('page_exit');
    });
    document.addEventListener('DOMContentLoaded', init);
})();
