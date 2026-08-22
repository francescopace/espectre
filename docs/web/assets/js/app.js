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
    const MqttProtocolClient = window.ESPectreMqttClient;
    if (!MqttProtocolClient) throw new Error('ESPectre MQTT protocol client is unavailable');

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    // analytics.js is optional: the app must work with it blocked or absent.
    const track = (name, params) => window.trackEvent ? window.trackEvent(name, params) : false;
    const errorType = (error) => (error && error.name) || 'Error';
    const toolNameForRoute = (routeName) => routeRegistry.groupOf(routeName) === 'tools'
        ? routeName
        : 'monitor';
    const activeToolName = () => toolNameForRoute(route);
    const LEGACY_TOOL_ROUTES = Object.freeze({
        ble: 'configure',
        mqtt: 'monitor',
        device: 'configure'
    });
    const LOCAL_DEVELOPMENT_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);
    const MQTT_PRESETS = Object.freeze({
        home_assistant: Object.freeze({
            configure: Object.freeze({
                host: 'homeassistant.local', port: '1883', hostPlaceholder: 'homeassistant.local'
            }),
            monitor: Object.freeze({
                host: 'homeassistant.local', port: '9001', path: '/mqtt', tls: false,
                hostPlaceholder: 'homeassistant.local'
            })
        }),
        lan_broker: Object.freeze({
            configure: Object.freeze({
                host: '', port: '1883', hostPlaceholder: 'broker.local or 192.168.1.20'
            }),
            monitor: Object.freeze({
                host: 'localhost', port: '9001', path: '/mqtt', tls: false,
                hostPlaceholder: 'localhost or broker.local'
            })
        }),
        emqx_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'deployment-id.ala.region.emqxsl.com', port: '8883',
                hostPlaceholder: 'deployment-id.ala.region.emqxsl.com',
                locked: Object.freeze(['port'])
            }),
            monitor: Object.freeze({
                host: 'deployment-id.ala.region.emqxsl.com', port: '8084', path: '/mqtt', tls: true,
                hostPlaceholder: 'deployment-id.ala.region.emqxsl.com',
                locked: Object.freeze(['port', 'path', 'tls'])
            })
        }),
        hivemq_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'cluster-id.s1.region.hivemq.cloud', port: '8883',
                hostPlaceholder: 'cluster-id.s1.region.hivemq.cloud',
                locked: Object.freeze(['port'])
            }),
            monitor: Object.freeze({
                host: 'cluster-id.s1.region.hivemq.cloud', port: '8884', path: '/mqtt', tls: true,
                hostPlaceholder: 'cluster-id.s1.region.hivemq.cloud',
                locked: Object.freeze(['port', 'path', 'tls'])
            })
        }),
        flespi: Object.freeze({
            configure: Object.freeze({
                host: 'mqtt.flespi.io', port: '8883', hostPlaceholder: 'mqtt.flespi.io',
                locked: Object.freeze(['host', 'port'])
            }),
            monitor: Object.freeze({
                host: 'mqtt.flespi.io', port: '443', path: '/mqtt', tls: true,
                hostPlaceholder: 'mqtt.flespi.io',
                locked: Object.freeze(['host', 'port', 'path', 'tls'])
            })
        }),
        cloud_broker: Object.freeze({
            configure: Object.freeze({
                host: 'cluster.example.com', port: '', hostPlaceholder: 'cluster.example.com'
            }),
            monitor: Object.freeze({
                host: 'cluster.example.com', port: '', path: '/mqtt', tls: true,
                hostPlaceholder: 'cluster.example.com'
            })
        })
    });
    const SECURE_CLOUD_MQTT_PRESETS = new Set([
        'emqx_cloud', 'hivemq_cloud', 'flespi', 'cloud_broker'
    ]);
    const MQTT_FORM_DEFAULTS = {
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

    const EVALUATION_INTERVAL_MS_DEFAULT = 250;
    const PUBLISH_INTERVAL_MS_DEFAULT = 1000;
    const CSI_TARGET_PPS_DEFAULT = 100;
    const CONFIG_VERIFICATION_INITIAL_DELAY_MS = 250;
    const CONFIG_VERIFICATION_RETRY_MS = 1500;
    const CONFIG_VERIFICATION_MAX_ATTEMPTS = 4;
    const OTA_TRACKING_TIMEOUT_MS = 120000;

    const conn = {
        mode: null,             // 'ble' | 'mqtt' | 'demo'
        status: 'disconnected', // disconnected | connecting | connected
        movement: 0,
        threshold: 0.5,
        motion: false,
        evaluationIntervalMs: 0,
        publishIntervalMs: 0,
        csiTargetPps: 0,
        deviceName: '',
        deviceId: '',
        generatedName: '',
        deviceLabel: '',
        deviceConfigSupported: false,
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
    const LIVE_EXPERIENCE_ROUTES = new Set(['game', 'theremin']);
    let pendingLiveDestination = '';
    const deviceNameEditorState = {
        configure: { editing: false, savePending: false },
        monitor: { editing: false, savePending: false }
    };
    let lastTrackedProfile = null;
    let wifiBandPolicyAvailable = false;
    let currentWifiBandPolicy = '2g';
    let otaUpdateAvailable = false;
    let otaBusy = false;
    let otaState = '';
    let otaMessage = '';
    let otaTargetVersion = '';
    let otaSupported = null;
    let otaDefaultChannel = '';
    let otaChannelChanged = false;
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

    function connectionIntentRoute() {
        return pendingLiveDestination || route;
    }

    function rememberLiveDestination(routeName = route) {
        if (LIVE_EXPERIENCE_ROUTES.has(routeName)) pendingLiveDestination = routeName;
    }

    function completeLiveConnectionNavigation() {
        const destination = pendingLiveDestination;
        pendingLiveDestination = '';
        if (destination && routeRegistry.has(destination)) {
            if (route !== destination) location.hash = '#' + destination;
            return;
        }
        if (route === 'monitor' || route === 'configure') {
            setDeviceView('live', { focus: true });
        }
    }

    function rememberConnectionOrigin() {
        const origin = connectionIntentRoute();
        conn.toolName = toolNameForRoute(origin);
        conn.entryPoint = origin;
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
            applyRemoteThreshold(threshold);
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
        applySensingCadence(snapshot);
    }

    function positiveInt(value) {
        const n = Number(value);
        return Number.isInteger(n) && n > 0 ? n : 0;
    }

    function evaluationIntervalMs() {
        return conn.evaluationIntervalMs || EVALUATION_INTERVAL_MS_DEFAULT;
    }

    function publishIntervalMs() {
        return conn.publishIntervalMs || PUBLISH_INTERVAL_MS_DEFAULT;
    }

    function csiTargetPps() {
        return conn.csiTargetPps || CSI_TARGET_PPS_DEFAULT;
    }

    function resetSensingCadence() {
        conn.evaluationIntervalMs = 0;
        conn.publishIntervalMs = 0;
        conn.csiTargetPps = 0;
    }

    function applySensingCadence(snapshot) {
        if (!snapshot || typeof snapshot !== 'object') return;
        const evaluation = positiveInt(snapshot.evaluation_interval_ms);
        const publish = positiveInt(snapshot.publish_interval_ms);
        const pps = positiveInt(snapshot.csi_target_pps);
        if (evaluation) conn.evaluationIntervalMs = evaluation;
        if (publish) conn.publishIntervalMs = publish;
        if (pps) conn.csiTargetPps = pps;
        syncDiagnosticsPolling();
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
        if (data.supports_device_config !== undefined) {
            conn.deviceConfigSupported = sysinfoBoolean(data.supports_device_config);
        }
        if (data.device_label || data.device_name) {
            conn.deviceName = data.device_label || data.device_name;
        }
        if (data.chip) conn.chip = String(data.chip).toUpperCase();
        if (data.firmware_version || data.version) {
            conn.firmwareVersion = data.firmware_version || data.version;
        }
    }

    function renderDeviceIdentity(identity = conn) {
        const write = (selector, value) => {
            $$(selector).forEach((el) => { el.textContent = value || '—'; });
        };
        write('.js-menu-chip', identity.chip);
        write('.js-menu-device-id', identity.deviceId);
        write('.js-menu-firmware', identity.firmwareVersion);
    }

    function formatDeviceIdentityLine(chip, deviceId, firmware) {
        const parts = [];
        if (chip) parts.push('Chip ' + chip);
        if (deviceId) parts.push('Device ID ' + deviceId);
        if (firmware) parts.push('Firmware ' + firmware);
        return parts.join(' · ');
    }

    function deviceNameEditorElements(surface) {
        return {
            editor: $(`.js-${surface}-name-editor`),
            trigger: $(`.js-${surface}-name-trigger`),
            display: $(`.js-${surface}-name-display`),
            input: $(`.js-${surface}-name-input`)
        };
    }

    function renderDeviceNameEditor(surface) {
        const state = deviceNameEditorState[surface];
        const { editor, trigger, display, input } = deviceNameEditorElements(surface);
        if (!state || !editor || !trigger || !display || !input) return;

        const displayName = conn.deviceLabel || conn.generatedName || conn.deviceId
            || conn.deviceName || 'ESPectre';
        const mqttCanEdit = conn.mode === 'mqtt' && monitorIsMqttLive()
            && (!monitor.commandCatalogReady || monitor.commands.has('set_device_label'));
        const canEdit = conn.status === 'connected'
            && (conn.mode === 'ble' || conn.mode === 'demo' || mqttCanEdit)
            && conn.deviceConfigSupported;
        display.textContent = displayName;
        trigger.disabled = !canEdit || state.savePending;
        trigger.setAttribute('aria-label', conn.deviceLabel ? 'Edit device name' : 'Set device name');
        trigger.title = canEdit ? 'Click to edit the device name' : '';
        trigger.hidden = state.editing;
        input.hidden = !state.editing;
        input.disabled = state.savePending;
        if (!state.editing) input.value = conn.deviceLabel || '';
        editor.setAttribute('aria-busy', String(state.savePending));
        if (surface === 'configure') {
            const identity = $('.js-configure-device-banner-sub');
            if (identity) {
                identity.textContent = formatDeviceIdentityLine(
                    conn.chip,
                    conn.deviceLabel ? conn.deviceId : '',
                    conn.firmwareVersion
                ) || '—';
            }
        }
    }

    function renderConfigureDeviceNameEditor() {
        renderDeviceNameEditor('configure');
    }

    function renderMonitorDeviceNameEditor() {
        renderDeviceNameEditor('monitor');
    }

    function startDeviceNameEdit(surface) {
        const state = deviceNameEditorState[surface];
        const { trigger, input } = deviceNameEditorElements(surface);
        if (!state || !trigger || !input || trigger.disabled || state.savePending) return;
        state.editing = true;
        input.value = conn.deviceLabel || '';
        renderDeviceNameEditor(surface);
        requestAnimationFrame(() => {
            input.focus();
            input.select();
        });
    }

    function cancelDeviceNameEdit(surface) {
        const state = deviceNameEditorState[surface];
        if (!state || !state.editing) return;
        state.editing = false;
        renderDeviceNameEditor(surface);
        const { trigger } = deviceNameEditorElements(surface);
        if (trigger) trigger.focus();
    }

    async function saveDeviceNameOnBlur(surface) {
        const state = deviceNameEditorState[surface];
        if (!state || !state.editing || state.savePending) return;
        const { input } = deviceNameEditorElements(surface);
        const label = input ? input.value.trim() : '';
        const previousLabel = conn.deviceLabel;
        const previousName = conn.deviceName;
        state.editing = false;
        if (label === previousLabel) {
            renderDeviceNameEditor(surface);
            return;
        }

        state.savePending = true;
        conn.deviceLabel = label;
        conn.deviceName = label || conn.generatedName || 'ESPectre';
        renderConnection();
        const saved = await cfgSaveDeviceLabel(label);
        if (!saved) {
            conn.deviceLabel = previousLabel;
            conn.deviceName = previousName;
        }
        state.savePending = false;
        renderConnection();
    }

    function startConfigureDeviceNameEdit() {
        startDeviceNameEdit('configure');
    }

    function cancelConfigureDeviceNameEdit() {
        cancelDeviceNameEdit('configure');
    }

    function saveConfigureDeviceNameOnBlur() {
        return saveDeviceNameOnBlur('configure');
    }

    function startMonitorDeviceNameEdit() {
        startDeviceNameEdit('monitor');
    }

    function cancelMonitorDeviceNameEdit() {
        cancelDeviceNameEdit('monitor');
    }

    function saveMonitorDeviceNameOnBlur() {
        return saveDeviceNameOnBlur('monitor');
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
        if (otaAwaitingReconnect) completeOtaReconnect();
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
        resetSensingCadence();
        resetMonitorLiveView();
        resetOtaChannelSelection();
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

    function thresholdControlActive() {
        return document.getElementById('sense-threshold') === document.activeElement;
    }

    function formatThreshold(threshold) {
        return Number(threshold).toFixed(2);
    }

    const GAME_THRESHOLD_DEFAULT = 0.5;
    let gameThresholdOverride = GAME_THRESHOLD_DEFAULT;

    function gameThreshold() {
        return gameThresholdOverride;
    }

    function paintGameThresholdControl() {
        const threshold = gameThreshold();
        const display = formatThreshold(threshold);
        $$('.js-game-threshold-value').forEach((readout) => { readout.textContent = display; });
        renderGameMotionGauge();
        const sliders = [document.getElementById('game-threshold'), ...$$('.js-game-fullscreen-threshold')];
        sliders.filter(Boolean).forEach((slider) => {
            slider.setAttribute('aria-valuetext', display);
            if (slider === document.activeElement || Number(slider.value) === threshold) return;
            slider.value = String(threshold);
        });
    }

    function resetGameThreshold() {
        gameThresholdOverride = GAME_THRESHOLD_DEFAULT;
        paintGameThresholdControl();
    }

    function applyGameThreshold(threshold) {
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) return;
        gameThresholdOverride = threshold;
        paintGameThresholdControl();
        if (route === 'game') {
            gameSetFlight(gameSensingActive());
            gameStartPreview();
        }
    }

    function paintThresholdControls(threshold) {
        const sense = document.getElementById('sense-threshold');
        if (sense && sense !== document.activeElement && Number(sense.value) !== threshold) {
            sense.value = String(threshold);
        }
    }

    function syncThresholdControl(threshold) {
        if (thresholdControlActive()) return;
        paintThresholdControls(threshold);
    }

    function applyRemoteThreshold(threshold) {
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) return;
        if (thresholdControlActive()) return;
        conn.threshold = threshold;
        syncThresholdControl(threshold);
    }

    function applyLocalThreshold(threshold) {
        conn.threshold = threshold;
        paintThresholdControls(threshold);
        renderTelemetry();
    }

    function commitThreshold(threshold) {
        if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
            toast('Threshold must be between 0 and 1.');
            return;
        }
        applyLocalThreshold(threshold);
        runSensingCommand(
            { command: 'set_threshold', threshold },
            'Applying threshold…',
            'Threshold updated.',
            () => { conn.threshold = threshold; renderTelemetry(); }
        );
    }

    function bindThresholdControls() {
        const sense = document.getElementById('sense-threshold');
        const gameSlider = document.getElementById('game-threshold');
        if (sense) {
            sense.addEventListener('change', () => commitThreshold(Number(sense.value)));
        }
        const gameSliders = [gameSlider, ...$$('.js-game-fullscreen-threshold')].filter(Boolean);
        if (!gameSliders.length) return;
        gameSliders.forEach((slider) => {
            slider.addEventListener('input', (event) => applyGameThreshold(Number(event.currentTarget.value)));
            slider.addEventListener('change', (event) => applyGameThreshold(Number(event.currentTarget.value)));
        });
        const fullscreenSlider = $('.js-game-fullscreen-threshold');
        if (!fullscreenSlider) return;
        const applyFullscreenPointerThreshold = (event) => {
            if (event.type === 'pointermove' && !fullscreenSlider.hasPointerCapture(event.pointerId)) return;
            event.preventDefault();
            if (event.type === 'pointerdown') fullscreenSlider.setPointerCapture(event.pointerId);
            const bounds = fullscreenSlider.getBoundingClientRect();
            const threshold = Math.max(0, Math.min(1, (bounds.bottom - event.clientY) / bounds.height));
            fullscreenSlider.value = String(threshold);
            applyGameThreshold(threshold);
        };
        fullscreenSlider.addEventListener('pointerdown', applyFullscreenPointerThreshold);
        fullscreenSlider.addEventListener('pointermove', applyFullscreenPointerThreshold);
    }

    function applyLiveTelemetry(movement, threshold, motionState) {
        markToolReady('telemetry');
        conn.movement = movement;
        applyRemoteThreshold(threshold);
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
            syncFirmwareUpdateNotice();
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
            }
        } catch (error) {
            bleClient = null;
            syncFirmwareUpdateNotice();
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

    function mqttPresetNote(target, presetName) {
        if (target === 'configure') {
            if (presetName === 'home_assistant') {
                return 'Enter the MQTT credentials created for ESPectre.';
            }
            if (presetName === 'lan_broker') {
                return "Use the broker's LAN hostname or IP address.";
            }
            if (presetName === 'emqx_cloud') {
                return 'Replace the template with your EMQX endpoint.';
            }
            if (presetName === 'hivemq_cloud') {
                return 'Replace the template with your HiveMQ endpoint.';
            }
            if (presetName === 'flespi') {
                return 'Use your Flespi token as username; no password.';
            }
            return 'Enter your broker endpoint and credentials.';
        }
        if (presetName === 'home_assistant') {
            return location.protocol === 'https:'
                ? 'Port 9001; HTTPS requires trusted WSS.'
                : 'Home Assistant WebSockets use port 9001.';
        }
        if (presetName === 'lan_broker') {
            return location.protocol === 'https:'
                ? 'Defaults to localhost; HTTPS requires trusted WSS.'
                : 'Defaults to localhost; replace it for another LAN host.';
        }
        if (presetName === 'emqx_cloud') {
            return 'WSS uses port 8084 and path /mqtt.';
        }
        if (presetName === 'hivemq_cloud') {
            return 'WSS uses port 8884 and path /mqtt.';
        }
        if (presetName === 'flespi') {
            return 'Use your Flespi token as username; no password.';
        }
        return 'Enter your broker WSS settings and credentials.';
    }

    function updateMqttPresetNote(target, presetName) {
        const note = target === 'configure'
            ? $('.js-cfg-mqtt-preset-note')
            : $('.js-mon-mqtt-preset-note');
        if (note) note.textContent = mqttPresetNote(target, presetName);
    }

    function applyMqttPresetFieldLocks(target, preset) {
        const fields = target === 'configure'
            ? { host: 'cfg-mqtt-host', port: 'cfg-mqtt-port' }
            : { host: 'mon-host', port: 'mon-port', path: 'mon-path', tls: 'mon-tls' };
        const locked = new Set(preset.locked || []);
        Object.entries(fields).forEach(([name, id]) => {
            const input = document.getElementById(id);
            if (!input) return;
            const isLocked = locked.has(name);
            if (input.type === 'checkbox') input.disabled = isLocked;
            else input.readOnly = isLocked;
            input.toggleAttribute('data-preset-locked', isLocked);
            input.title = isLocked ? 'Set by the selected broker preset' : '';
        });
    }

    function browserBrokerHost(host) {
        return String(host || '')
            .trim()
            .replace(/^mqtts?:\/\//, '')
            .replace(/:\d+$/, '');
    }

    function configuredBrokerPreset(host, port) {
        const normalizedHost = browserBrokerHost(host).toLowerCase();
        if (normalizedHost === 'homeassistant.local'
                && Number(port) === Number(MQTT_PRESETS.home_assistant.configure.port)) {
            return 'home_assistant';
        }
        if (normalizedHost === 'mqtt.flespi.io') return 'flespi';
        if (normalizedHost.endsWith('.hivemq.cloud')) return 'hivemq_cloud';
        if (normalizedHost.endsWith('.emqxsl.com') || normalizedHost.endsWith('.emqx.cloud')) {
            return 'emqx_cloud';
        }
        const localIpv4 = /^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)/.test(normalizedHost);
        if (normalizedHost === 'localhost' || normalizedHost.endsWith('.local') || localIpv4) {
            return 'lan_broker';
        }
        return 'cloud_broker';
    }

    function applyConfigureMqttPreset(presetName, { clearCredentials = true } = {}) {
        const select = document.getElementById('cfg-mqtt-preset');
        const resolvedName = MQTT_PRESETS[presetName] ? presetName : 'cloud_broker';
        const preset = MQTT_PRESETS[resolvedName];
        select.value = resolvedName;
        document.getElementById('cfg-mqtt-host').value = preset.configure.host;
        document.getElementById('cfg-mqtt-host').placeholder = preset.configure.hostPlaceholder;
        document.getElementById('cfg-mqtt-port').value = preset.configure.port;
        document.getElementById('cfg-topic-prefix').value = MQTT_FORM_DEFAULTS.topicPrefix;
        applyMqttPresetFieldLocks('configure', preset.configure);
        if (clearCredentials) {
            document.getElementById('cfg-mqtt-user').value = '';
            document.getElementById('cfg-mqtt-pass').value = '';
        }
        updateMqttPresetNote('configure', select.value);
    }

    function applyMonitorMqttPreset(presetName, { clearCredentials = true } = {}) {
        const select = document.getElementById('mon-mqtt-preset');
        const resolvedName = MQTT_PRESETS[presetName] ? presetName : 'cloud_broker';
        const preset = MQTT_PRESETS[resolvedName];
        select.value = resolvedName;
        document.getElementById('mon-host').value = preset.monitor.host;
        document.getElementById('mon-host').placeholder = preset.monitor.hostPlaceholder;
        document.getElementById('mon-port').value = preset.monitor.port;
        document.getElementById('mon-path').value = preset.monitor.path;
        document.getElementById('mon-tls').checked = preset.monitor.tls;
        document.getElementById('mon-topic-prefix').value = MQTT_FORM_DEFAULTS.topicPrefix;
        applyMqttPresetFieldLocks('monitor', preset.monitor);
        if (clearCredentials) {
            document.getElementById('mon-user').value = '';
            document.getElementById('mon-pass').value = '';
        }
        updateMqttPresetNote('monitor', select.value);
    }

    function applyConfigureMqttToMonitor() {
        const host = document.getElementById('cfg-mqtt-host');
        const user = document.getElementById('cfg-mqtt-user');
        const pass = document.getElementById('cfg-mqtt-pass');
        const prefix = document.getElementById('cfg-topic-prefix');
        const presetName = document.getElementById('cfg-mqtt-preset').value;
        const monHost = document.getElementById('mon-host');
        const monUser = document.getElementById('mon-user');
        const monPass = document.getElementById('mon-pass');
        const monPrefix = document.getElementById('mon-topic-prefix');
        const monDevice = document.getElementById('mon-device');
        if (MQTT_PRESETS[presetName]) {
            applyMonitorMqttPreset(presetName, { clearCredentials: false });
            if (host && browserBrokerHost(host.value)) {
                monHost.value = browserBrokerHost(host.value);
            }
        } else if (host && host.value.trim()) {
            applyMonitorMqttPreset('cloud_broker', { clearCredentials: false });
            monHost.value = browserBrokerHost(host.value);
        }
        monUser.value = user ? user.value.trim() : '';
        monPass.value = pass ? pass.value : '';
        if (prefix && prefix.value.trim()) {
            monPrefix.value = prefix.value.trim().replace(/\/+$/, '');
        }
        const deviceId = conn.deviceId.trim();
        if (deviceId && deviceId !== '—') monDevice.value = deviceId;
        // Presets map device TCP settings to browser WebSocket defaults where a stable mapping exists.
        // Provider presets supply stable ports and paths; account-specific hostnames are copied without the MQTT scheme.
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
        completeLiveConnectionNavigation();
        toast('Sensing is live.');
    }

    async function startDetection() {
        rememberLiveDestination();
        if (conn.mode === 'demo') {
            completeLiveConnectionNavigation();
            return;
        }
        applyConfigureMqttToMonitor();
        const nextDevice = document.getElementById('mon-device').value.trim();
        if (conn.mode === 'mqtt' && monitorIsMqttLive()
                && (!nextDevice || nextDevice === monitor.boundDeviceId)) {
            completeLiveConnectionNavigation();
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
            const mqttPreset = configuredBrokerPreset(snapshot.mqtt_host, snapshot.mqtt_port);
            document.getElementById('cfg-mqtt-preset').value = mqttPreset;
            updateMqttPresetNote('configure', mqttPreset);
            applyMqttPresetFieldLocks('configure', MQTT_PRESETS[mqttPreset].configure);
            const mqttUser = document.getElementById('cfg-mqtt-user');
            if (mqttUser) mqttUser.value = snapshot.mqtt_username || '';
            set('cfg-topic-prefix', snapshot.topic_prefix || MQTT_FORM_DEFAULTS.topicPrefix);
            const mqttPass = document.getElementById('cfg-mqtt-pass');
            if (mqttPass) mqttPass.value = '';
        }
        applyDeviceIdentity(snapshot);
        if (conn.mode === 'ble') otaSupported = false;
        else if (snapshot.supports_ota !== undefined) otaSupported = sysinfoBoolean(snapshot.supports_ota);
        applySensingSnapshot(snapshot);
        syncFirmwareUpdateNotice();
        evaluateConfigVerification(snapshot);
        setConnectionDot('.js-wifi-status-dot', snapshot.wifi_connected);
        setConnectionDot('.js-mqtt-status-dot', snapshot.mqtt_connected);

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
        rememberLiveDestination();
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
                'ota_status', 'ota_check', 'set_ble', 'set_device_label'
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
                device_id: '3cf79180d3a0aca4',
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
            completeLiveConnectionNavigation();
            monitorResetChart();
            let t = 0;
            const demoTickSec = evaluationIntervalMs() / 1000;
            demoTimer = setInterval(() => {
                t += demoTickSec;
                const gameDemoActive = route === 'game' && game.phase !== 'idle' && game.phase !== 'done';
                const idle = 0.035 + Math.sin(t * 0.8) * 0.01 + Math.sin(t * 1.9) * 0.004;
                const gameManualFlight = route === 'game' && game.manualFlight ? 1 : 0;
                const target = Math.min(1, idle + Math.max(demoInputEnergy, gameManualFlight) * 0.95);
                const smoothingTau = gameDemoActive ? 0.29 : 0.49;
                const smoothing = 1 - Math.exp(-demoTickSec / smoothingTau);
                conn.movement += (target - conn.movement) * smoothing;
                conn.movement = Math.max(0.01, Math.min(1, conn.movement));
                const energyTau = gameDemoActive ? 0.33 : 0.49;
                demoInputEnergy *= Math.exp(-demoTickSec / energyTau);
                if (demoInputEnergy < 0.01) demoInputEnergy = 0;
                monitorFeed(
                    conn.movement,
                    conn.threshold,
                    conn.movement >= conn.threshold ? 'motion' : 'idle'
                );
                applyLiveTelemetry(conn.movement, conn.threshold, conn.movement >= conn.threshold ? 1 : 0);
            }, evaluationIntervalMs());
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
        const protocol = monitor.protocol;
        monitor.client = null;
        monitor.protocol = null;
        monitor.closing = true;
        if (protocol) protocol.close();
        if (client) client.end(true);
        monitor.closing = false;
        monitor.commands.clear();
        monitor.commandCatalogReady = false;
        monitor.bleRequested = false;
        monitor.handoffReady = false;
        monitor.boundDeviceId = '';
        if (monitor.discoveryTimer) {
            clearTimeout(monitor.discoveryTimer);
            monitor.discoveryTimer = 0;
        }
        if (monitor.connectionTimer) {
            clearTimeout(monitor.connectionTimer);
            monitor.connectionTimer = 0;
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
        renderConnection();
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
        resetSensingCadence();
        conn.deviceName = '';
        conn.deviceId = '';
        conn.generatedName = '';
        conn.deviceLabel = '';
        conn.deviceConfigSupported = false;
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
        Object.values(deviceNameEditorState).forEach((state) => {
            state.editing = false;
            state.savePending = false;
        });
        pendingLiveDestination = '';
        lastTrackedProfile = null;
        otaUpdateAvailable = false;
        otaBusy = false;
        otaState = '';
        otaMessage = '';
        otaTargetVersion = '';
        otaSupported = null;
        resetOtaChannelSelection();
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
        const usbConnected = Boolean(flash.usbDialog);
        const displayedConnected = usbConnected || connected;
        const displayedConnecting = !usbConnected && conn.status === 'connecting';
        const displayedMode = usbConnected ? 'usb' : conn.mode;
        const live = hasLiveDetection();
        const bleConnecting = conn.status === 'connecting'
            && !!bleClient
            && !bleClient.connected;
        const bleSetup = connected && conn.mode === 'ble';
        const mqttConnectionPending = monitorConnectionPending();
        const mqttSession = live || (monitor.handoffReady && monitorIsMqttLive());

        $('.js-conn-disconnected').hidden = displayedConnected || displayedConnecting;
        $('.js-conn-connecting').hidden = !displayedConnecting;
        $('.js-conn-connected').hidden = !displayedConnected;
        $('.js-dropdown').hidden = !(displayedConnected && dropdownOpen);
        $('.js-dropdown-toggle').setAttribute('aria-expanded', String(displayedConnected && dropdownOpen));
        $('.js-demo-tag').hidden = displayedMode !== 'demo';
        const transportTag = $('.js-transport-tag');
        if (transportTag) {
            const transportLabels = { ble: 'BLE', mqtt: 'MQTT', usb: 'USB' };
            transportTag.textContent = transportLabels[displayedMode] || '';
            transportTag.hidden = !displayedConnected || !transportLabels[displayedMode];
        }

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
        const edit = $('.js-device-edit-connectivity');
        const startSensing = document.querySelector('[data-page="configure"] .js-start-detection');
        if (configureOnboarding) configureOnboarding.hidden = bleSetup || conn.mode === 'demo';
        if (configureWorkspace) configureWorkspace.hidden = !(bleSetup || conn.mode === 'demo');
        if (monitorOnboarding) monitorOnboarding.hidden = mqttSession;
        if (monitorWorkspace) monitorWorkspace.hidden = !mqttSession;
        if (connectivitySetup) connectivitySetup.hidden = !(bleSetup || conn.mode === 'demo');
        if (startSensing) startSensing.disabled = monitor.closingBleForLive;
        if (edit) {
            edit.hidden = false;
            edit.disabled = false;
            edit.textContent = mqttConnectionPending ? 'Cancel connection' : 'Edit connectivity';
        }

        $$('.js-device-name').forEach((el) => { el.textContent = conn.deviceName || 'ESPectre'; });
        const displayedIdentity = usbConnected ? flashUsbIdentity() : conn;
        $$('.js-connection-device-name').forEach((el) => {
            el.textContent = displayedIdentity.deviceName || 'ESPectre';
        });
        $$('.js-device-banner-sub').forEach((el) => { el.textContent = conn.deviceBannerSub; });
        renderConfigureDeviceNameEditor();
        renderMonitorDeviceNameEditor();
        renderDeviceIdentity(displayedIdentity);
        const deviceIdLabel = $('.js-menu-device-id-label');
        if (deviceIdLabel) deviceIdLabel.textContent = usbConnected ? 'USB VID:PID' : 'Device ID';
        const firmwareLabel = $('.js-menu-firmware-label');
        if (firmwareLabel) firmwareLabel.textContent = usbConnected ? 'Target firmware' : 'Firmware';
        const usbNote = $('.js-usb-port-note');
        if (usbNote) usbNote.hidden = !usbConnected;
        const disconnectButton = $('.js-disconnect');
        if (disconnectButton) disconnectButton.hidden = usbConnected;
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
        if (live && route === 'game' && !game.ctx) requestAnimationFrame(gameResizeCanvas);
    }

    function renderTelemetry() {
        const pct = Math.round(energyFraction() * 100) + '%';
        const duration = (evaluationIntervalMs() / 1000) + 's';
        $$('.js-energy-fill').forEach((el) => {
            el.style.transitionDuration = duration;
            el.style.width = pct;
        });
        $$('.js-motion-label').forEach((el) => {
            el.textContent = conn.motion ? 'MOTION' : 'IDLE';
            el.classList.toggle('motion', conn.motion);
        });
        renderGameMotionGauge();
    }

    function renderGameMotionGauge() {
        const fill = $('.js-game-motion-fill');
        const marker = $('.js-game-motion-threshold');
        const gauge = $('.js-game-motion-gauge');
        if (fill) {
            fill.style.height = Math.round(energyFraction() * 100) + '%';
            fill.style.transitionDuration = (evaluationIntervalMs() / 1000) + 's';
        }
        if (marker) marker.style.bottom = Math.round(gameThreshold() * 100) + '%';
        if (gauge) gauge.classList.toggle('is-active', conn.movement >= gameThreshold());
    }

    /* ============================================================= routing */

    function focusRouteContent(routeName = route) {
        const page = $(`[data-page="${routeName}"]`);
        if (!page) return;
        const target = page.querySelector('h1') || page;
        if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
    }

    function applyRoute({ focus = true } = {}) {
        const routeAtStart = route;
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
        if (route === 'game') {
            requestAnimationFrame(() => {
                gameResizeCanvas();
                gameSetFlight(gameSensingActive());
                gameStartPreview();
            });
        }
        const contentPromise = $(`[data-page="${routeAtStart}"] .js-static-content`)
            ? loadStaticContent(routeAtStart)
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
        if (focus) {
            contentPromise.finally(() => {
                if (route === routeAtStart) focusRouteContent(routeAtStart);
            });
        }
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
        if (pendingLiveDestination) {
            if (LIVE_EXPERIENCE_ROUTES.has(target)) pendingLiveDestination = target;
            else if (target !== 'monitor' && target !== 'configure') pendingLiveDestination = '';
        }
        if (previousRoute === 'game' && target !== 'game') {
            gameExitFullscreen();
            reportGameAbandon('route_change');
        }
        if (target === 'game' && previousRoute !== 'game') resetGameThreshold();
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
            if (window.initSdkDownloadVersions) window.initSdkDownloadVersions(container);
            if (window.initCodeTabs) window.initCodeTabs(container);
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
    let scrollyKeyTargetScene = null;
    let scrollyKeyTargetTimer = null;
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
            const isActive = Number(el.dataset.scene) === scene;
            el.classList.toggle('is-active', isActive);
            if (el.classList.contains('js-scrolly-caption')) {
                el.toggleAttribute('inert', !isActive);
                el.setAttribute('aria-hidden', String(!isActive));
            }
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

    function scrollyHandleKeydown(event) {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        const target = event.target;
        if (target instanceof Element && target.closest('a, button, input, select, textarea, [contenteditable="true"]')) return;

        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null) return;
        const rect = section.getBoundingClientRect();
        if (rect.bottom <= 0 || rect.top >= window.innerHeight) return;

        const sceneCount = $$('.js-scrolly-scene').length;
        const currentScene = scrollyKeyTargetScene === null
            ? scrollySceneFromPosition(section, sceneCount)
            : scrollyKeyTargetScene;
        const direction = event.key === 'ArrowDown' ? 1 : -1;
        const nextScene = Math.min(sceneCount - 1, Math.max(0, currentScene + direction));
        if (nextScene === currentScene) return;

        event.preventDefault();
        scrollyKeyTargetScene = nextScene;
        clearTimeout(scrollyKeyTargetTimer);
        scrollyKeyTargetTimer = setTimeout(() => { scrollyKeyTargetScene = null; }, 500);

        const travel = Math.max(1, rect.height - window.innerHeight);
        const sectionTop = window.scrollY + rect.top;
        const sceneProgress = (nextScene + 0.5) / sceneCount;
        window.scrollTo({
            top: sectionTop + (travel * sceneProgress),
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
        });
    }

    function scrollyInit() {
        window.addEventListener('scroll', queueScrollyRender, { passive: true });
        window.addEventListener('resize', queueScrollyRender);
        document.addEventListener('keydown', scrollyHandleKeydown);
        renderScrolly();
    }

    /* =============================================================== flash */

    const flash = {
        manifests: {}, installUrl: null, badgeChecked: false,
        installerObserver: null, watchedDialogs: new WeakSet(), catalogReports: new Set(),
        downloadReady: false, detectedChip: '', supportedChipLabels: [], modalReturnFocus: null,
        refreshRequest: 0, targetVersion: '', usbDialog: null, usbPortInfo: null,
        usbReleaseTimer: null
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

    function flashUsbId(info) {
        if (!info || (!info.usbVendorId && !info.usbProductId)) return '';
        const hex = (value) => Number(value || 0).toString(16).toUpperCase().padStart(4, '0');
        return hex(info.usbVendorId) + ':' + hex(info.usbProductId);
    }

    function flashUsbIdentity() {
        const chip = String(flash.detectedChip || '').toUpperCase();
        const portId = flashUsbId(flash.usbPortInfo);
        return {
            deviceName: chip || portId || 'USB device',
            chip,
            deviceId: portId,
            firmwareVersion: flash.targetVersion
        };
    }

    function releaseUsbConnection(dialog) {
        if (flash.usbDialog !== dialog) return;
        clearTimeout(flash.usbReleaseTimer);
        flash.usbReleaseTimer = null;
        flash.usbDialog = null;
        flash.usbPortInfo = null;
        dropdownOpen = false;
        renderConnection();
        syncFirmwareUpdateNotice();
    }

    function scheduleUsbConnectionRelease(dialog) {
        if (flash.usbDialog !== dialog) return;
        clearTimeout(flash.usbReleaseTimer);
        flash.usbReleaseTimer = setTimeout(() => releaseUsbConnection(dialog), 100);
    }

    function activateUsbConnection(dialog) {
        flash.usbDialog = dialog;
        try {
            flash.usbPortInfo = dialog.port && typeof dialog.port.getInfo === 'function'
                ? dialog.port.getInfo()
                : null;
        } catch (error) {
            flash.usbPortInfo = null;
        }
        dialog.addEventListener('closed', () => scheduleUsbConnectionRelease(dialog), { once: true });
        renderConnection();
        syncFirmwareUpdateNotice();
    }

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

    function flashNextActionLink(label, className) {
        const link = document.createElement('a');
        link.href = '#flash';
        link.className = className;
        link.textContent = label;
        return link;
    }

    function flashHideMatterQr() {
        const status = $('.js-matter-status');
        const result = $('.js-matter-result');
        matterClose(false);
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
            const readQr = flashNextActionLink('Read the onboarding QR over USB', 'js-matter-read');
            if (!browserSupport.flash) {
                readQr.setAttribute('aria-disabled', 'true');
                readQr.title = flashUnsupportedMessage();
            }
            note.replaceChildren(
                'After flashing Matter, commission the device with a Matter controller.',
                document.createElement('br'),
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
        const requestId = ++flash.refreshRequest;
        const selectedChannel = channelSel.value;
        flash.downloadReady = false;
        flash.targetVersion = '';

        try {
            const manifest = await flashLoadManifest(selectedChannel);
            if (requestId !== flash.refreshRequest) return;
            const frontendsMap = flashManifestFrontends(manifest);

            const frontends = Object.entries(frontendsMap)
                .sort(([a], [b]) => byPreferredOrder(FRONTEND_ORDER, a, b));
            const successKey = selectedChannel + ':success';
            if (!flash.catalogReports.has(successKey)) {
                const reported = track('firmware_catalog', {
                    channel: selectedChannel,
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
            flash.targetVersion = manifest.version || '';
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
            if (requestId !== flash.refreshRequest) return;
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
            const failureKey = selectedChannel + ':failure';
            if (!flash.catalogReports.has(failureKey)) {
                const reported = track('firmware_catalog', {
                    channel: selectedChannel,
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
        if (trigger) trigger.setAttribute('aria-disabled', 'true');
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
            status.textContent = 'Onboarding codes read from the device.';
            matterOpen(trigger);
            track('matter_qr_read', { result: 'success' });
        } catch (error) {
            status.textContent = error.message || 'Unable to read the Matter QR code.';
            track('matter_qr_read', { result: 'failure', error_type: errorType(error) });
        } finally {
            if (port && (port.readable || port.writable)) {
                await port.close().catch(() => {});
            }
            if (trigger) trigger.setAttribute('aria-disabled', 'false');
        }
    }

    function syncModalOpenState() {
        const openModal = $$('.modal-backdrop').find((modal) => !modal.hidden);
        document.body.classList.toggle('modal-open', Boolean(openModal));
        Array.from(document.body.children).forEach((child) => {
            if (!(child instanceof HTMLElement)) return;
            const shouldBeInert = Boolean(openModal) && child !== openModal;
            if (shouldBeInert && !child.inert) {
                child.inert = true;
                child.dataset.modalInert = 'true';
            } else if (!shouldBeInert && child.dataset.modalInert === 'true') {
                child.inert = false;
                delete child.dataset.modalInert;
            }
        });
    }

    function matterOpen(returnFocus) {
        const modal = $('.js-matter-modal');
        flash.modalReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        modal.querySelector('.modal-card').focus();
    }

    function matterClose(restoreFocus = true) {
        const modal = $('.js-matter-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (restoreFocus && flash.modalReturnFocus && flash.modalReturnFocus.isConnected) {
            flash.modalReturnFocus.focus();
        }
        flash.modalReturnFocus = null;
    }

    /**
     * Shows the latest published release in the hero badge. The release
     * manifest is staged by CI from the GitHub release tag, so it is already
     * the newest version and needs no API call. The badge is decorative:
     * it stays hidden when the manifest is unavailable.
     */
    async function updateReleaseBadge() {
        if (flash.badgeChecked) return;
        flash.badgeChecked = true;
        try {
            const manifest = await flashLoadManifest('release');
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

    function flashDialogText(root) {
        if (!root) return '';
        let text = root.textContent || '';
        root.querySelectorAll('*').forEach((element) => {
            if (element.shadowRoot) text += ' ' + flashDialogText(element.shadowRoot);
        });
        return text;
    }

    function watchFirmwareInstallDialog(dialog) {
        if (flash.watchedDialogs.has(dialog)) return;
        flash.watchedDialogs.add(dialog);
        activateUsbConnection(dialog);
        let started = false;
        let reported = false;
        let shadowObserver = null;
        let inspectTimer = null;
        const observedRoots = new WeakSet();

        const markStarted = () => {
            if (started) return;
            started = true;
            track('firmware_install_start', flashParams());
        };
        const report = (result) => {
            if (reported) return;
            reported = true;
            track('firmware_install_result', { ...flashParams(), result });
            if (shadowObserver) shadowObserver.disconnect();
            clearInterval(inspectTimer);
            inspectTimer = null;
        };
        const setDetectedChip = (chip) => {
            const normalized = String(chip || '').toUpperCase();
            if (!normalized || flash.detectedChip === normalized) return;
            flash.detectedChip = normalized;
            renderConnection();
        };
        const inspect = () => {
            observeRoot(dialog.shadowRoot);
            const text = flashDialogText(dialog.shadowRoot);
            // The vendored ESP Web Tools version exposes completion before its
            // final Next screen. Text checks keep the listener resilient if
            // those internal state properties change in a future upgrade.
            const installState = dialog._installState && dialog._installState.state;
            if (dialog._installState && dialog._installState.chipFamily) {
                setDetectedChip(dialog._installState.chipFamily);
            }
            const found = text.match(FLASH_CHIP_FOUND_RE);
            if (found) setDetectedChip(found[1]);
            if (dialog._installConfirmed === true
                    || /Preparing installation|Erasing device|Writing progress:/i.test(text)) {
                markStarted();
            }
            const unsupported = text.match(FLASH_CHIP_UNSUPPORTED_RE);
            if (unsupported) {
                setDetectedChip(unsupported[1]);
                flashStatus(flashUnsupportedBoardMessage(unsupported[1]), 'is-error');
                report('unsupported');
                return;
            }
            if (installState === 'finished' || /Installation complete!/i.test(text)) report('success');
            else if (installState === 'error' || /Installation failed/i.test(text)) report('failure');
        };
        const observeRoot = (root) => {
            if (!root || observedRoots.has(root)) return;
            observedRoots.add(root);
            shadowObserver.observe(root, {
                childList: true, subtree: true, characterData: true
            });
            root.querySelectorAll('*').forEach((element) => {
                if (element.shadowRoot) observeRoot(element.shadowRoot);
            });
        };
        const attach = () => {
            if (!dialog.shadowRoot || shadowObserver) return false;
            shadowObserver = new MutationObserver(inspect);
            inspectTimer = setInterval(inspect, 250);
            inspect();
            return true;
        };
        if (!attach()) [0, 50, 200].forEach((delay) => setTimeout(attach, delay));

        const removalObserver = new MutationObserver(() => {
            if (dialog.isConnected) return;
            removalObserver.disconnect();
            scheduleUsbConnectionRelease(dialog);
            if (started && !reported) report('cancelled');
            if (shadowObserver) shadowObserver.disconnect();
            clearInterval(inspectTimer);
            inspectTimer = null;
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
            track('firmware_installer_open', flashParams());
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
            const action = event.target.closest('.js-matter-read');
            if (!action) return;
            event.preventDefault();
            if (action.getAttribute('aria-disabled') === 'true') return;
            matterReadQr();
        });
        $$('.js-matter-close').forEach((button) => {
            button.addEventListener('click', () => matterClose());
        });
        $('.js-matter-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) matterClose();
        });
        if (browserSupport.flash) observeFirmwareInstaller();
    }

    /* ============================================================= monitor */

    const MONITOR_CHART_WINDOW_MS = 60 * 1000;
    const MONITOR_CALIBRATION_FALLBACK_MS = 45 * 1000;
    const MONITOR_CALIBRATION_SAFETY_MS = 90 * 1000;
    const MONITOR_DEMO_CALIBRATION_MS = 2500;
    const MONITOR_DISCOVERY_TIMEOUT_MS = 2000;
    const MONITOR_CONNECTION_TIMEOUT_MS = 10000;

    function monitorChartMaxPoints() {
        return Math.max(2, Math.ceil(MONITOR_CHART_WINDOW_MS / evaluationIntervalMs()) + 2);
    }

    function monitorChartCoalesceMs() {
        return Math.max(16, Math.min(100, Math.floor(evaluationIntervalMs() / 2)));
    }

    function monitorTelemetryStaleMs() {
        return Math.max(publishIntervalMs(), evaluationIntervalMs() * 6);
    }

    const monitor = {
        client: null,
        protocol: null,
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
        commands: new Set(),
        commandCatalogReady: false,
        bleRequested: false,
        handoffReady: false,
        closingBleForLive: false,
        diagTimer: null,
        diagIntervalMs: 0,
        calibrating: false,
        calibrationTimer: null,
        boundDeviceId: '',
        discoveryActive: false,
        discoveredDevices: {},
        discoveryPrefix: '',
        discoveryTopics: [],
        discoveryTimer: 0,
        connectionTimer: 0,
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
        const prefixValid = !!prefix && !prefix.startsWith('/') && !/[+#\0]/.test(prefix);
        const fields = [
            [hostInput, !!host && !/\s|:\/\/|\//.test(host), 'Enter a valid broker host.'],
            [portInput, !!port && Number.isInteger(portNumber)
                && portNumber >= 1 && portNumber <= 65535, 'Enter a port from 1 to 65535.'],
            [prefixInput, prefixValid, 'Enter a topic prefix without a leading slash or MQTT wildcards.'],
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
            device
        };
    }

    function monitorIsMqttLive() {
        return monitor.inputMode === 'mqtt' && !!monitor.client;
    }

    function monitorConnectionPending() {
        return monitor.closingBleForLive || Boolean(
            monitor.client
            && (!monitor.connectedAt || monitor.discoveryActive || conn.status === 'connecting')
        );
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

    function ingestMqttMessage({ suffix, text, data }) {
        if (monitor.boundDeviceId && conn.deviceId && monitor.boundDeviceId !== conn.deviceId) return;
        switch (suffix) {
            case 'commands/catalog':
                if (!data || !Array.isArray(data.commands)) return;
                monitor.commands = new Set(data.commands);
                monitor.commandCatalogReady = true;
                otaSupported = monitor.commands.has('ota_check') && monitor.commands.has('ota_start');
                renderConfigureDeviceNameEditor();
                renderMonitorDeviceNameEditor();
                syncSensingControls();
                syncOtaUpdateButton();
                syncFirmwareUpdateNotice();
                return;
            case 'info':
                applyDeviceInfo(data);
                return;
            case 'status': {
                const online = data.online === true;
                handleOtaDeviceAvailability(online);
                if (!online && !otaAwaitingReconnect && otaState !== 'reboot_scheduled') {
                    toast('The broker is connected, but the device is offline.');
                    monitorStatus('Device offline. Waiting for it to reconnect…');
                }
                return;
            }
            case 'ota/state':
                applyOtaStatus(data);
                return;
            case 'stats':
                if (!data || typeof data !== 'object'
                        || !['traffic_tx_pps', 'csi_callback_pps', 'free_memory_kb']
                            .some((key) => data[key] !== undefined)) return;
                markMonitorReady('diagnostics');
                monitorStats(data);
                if (data.traffic_tx_pps === undefined) {
                    monitorDiagStatus('Diagnostics received — this firmware does not expose the extended fields.');
                }
                return;
            case 'telemetry': {
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
            case 'ha/movement/state': {
                if (monitorHasFreshTelemetry()) return;
                const movement = Number(text);
                if (!Number.isFinite(movement)) return;
                applyMqttLiveTelemetry(movement, conn.threshold, conn.motion ? 'motion' : 'idle');
                return;
            }
            case 'ha/threshold/state':
                applyRemoteThreshold(Number(text));
                renderTelemetry();
                return;
            case 'ha/motion/state': {
                if (monitorHasFreshTelemetry()) return;
                const motion = text === 'ON' || text === '1' || text === 'motion';
                applyMqttLiveTelemetry(conn.movement, conn.threshold, motion ? 'motion' : 'idle');
                return;
            }
            case 'ha/detector/state':
                document.getElementById('sense-detector').value = text;
                if (text !== 'lightweight') setCalibrationBusy(false);
                else syncSensingControls();
                return;
            case 'ha/calibrate/state': {
                const calibrating = text === 'ON' || text === '1';
                setCalibrationBusy(calibrating);
                if (calibrating) scheduleCalibrationIdle(MONITOR_CALIBRATION_SAFETY_MS);
                return;
            }
            case 'ha/motion_on_hits/state':
                document.getElementById('sense-motion-on').value = text;
                return;
            case 'ha/motion_off_hits/state':
                document.getElementById('sense-motion-off').value = text;
                return;
            case 'ha/csi_traffic_mode/state':
                applyCsiTrafficModeSelect(text);
                return;
            case 'ha/traffic_generator_mode/state':
                document.getElementById('sense-generator-mode').value = text;
                return;
            default:
                return;
        }
    }

    function syncMonitorDemoButton() {
        const demo = $('.js-mon-demo');
        const ble = $('.js-mon-ble');
        const connect = $('.js-mon-connect');
        const mqttLive = monitorIsMqttLive();
        const mqttConnected = mqttLive && conn.status === 'connected';
        const connectionPending = monitorConnectionPending();
        if (demo) demo.hidden = mqttLive;
        if (ble) ble.hidden = !mqttLive;
        if (connect) {
            connect.disabled = mqttConnected;
            connect.textContent = mqttConnected ? 'Connected'
                : connectionPending ? 'Cancel connection'
                : 'Connect broker';
        }
    }

    function monitorHasFreshTelemetry() {
        return monitor.lastTelemetryAt > 0
            && (Date.now() - monitor.lastTelemetryAt) < monitorTelemetryStaleMs();
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
        if (last && now - last.at < monitorChartCoalesceMs()) {
            last.m = movement;
            last.t = threshold;
            last.at = now;
            last.on = motion;
        } else {
            monitor.points.push({ m: movement, t: threshold, at: now, on: motion });
        }
        const oldest = now - MONITOR_CHART_WINDOW_MS;
        while (monitor.points.length
                && (monitor.points[0].at < oldest || monitor.points.length > monitorChartMaxPoints())) {
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
        const tickMs = 10 * 1000;
        const labelEvery = width >= 420 ? 1 : 2;
        for (let age = MONITOR_CHART_WINDOW_MS; age >= 0; age -= tickMs) {
            const px = Math.max(0.5, Math.min(width - 0.5, x(now - age)));
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, plotH);
            ctx.stroke();
            const ticks = age / tickMs;
            if (ticks % labelEvery !== 0 && ticks !== 0) continue;
            const label = age === 0 ? 'now' : `−${age / 1000}s`;
            ctx.textAlign = age === 0 ? 'right' : (age === MONITOR_CHART_WINDOW_MS ? 'left' : 'center');
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
        choice.replaceChildren();
    }

    function recordDiscoveredMqttDevice(topic, payload) {
        const prefix = monitor.discoveryPrefix;
        if (!prefix) return;
        let message;
        try {
            message = MqttProtocolClient.parseDiscoveryMessage(prefix, topic, payload);
        } catch (error) {
            return;
        }
        if (!message) return;
        const { data, deviceId: topicId, suffix } = message;
        const device = monitor.discoveredDevices[topicId] || {
            topic_id: topicId,
            device_id: topicId
        };
        if (data.device_id) device.device_id = String(data.device_id);
        if (suffix === 'info') {
            ['device_name', 'device_label', 'frontend', 'chip'].forEach((key) => {
                if (data[key]) device[key] = data[key];
            });
        } else if (suffix === 'status' && 'online' in data) {
            device.online = data.online === true;
        }
        monitor.discoveredDevices[topicId] = device;
    }

    function monitorDeviceChipLabel(chip) {
        const value = String(chip || '').trim().toUpperCase().replace(/[-_]/g, '');
        if (!value) return 'Unknown chip';
        if (value === 'ESP32') return value;
        return value.replace(/^ESP32/, '');
    }

    function monitorDeviceStatus(device) {
        if (device.online === true) return { dotClass: 'dot-ok', label: 'Online' };
        if (device.online === false) return { dotClass: 'dot-error', label: 'Offline' };
        return { dotClass: 'dot-idle', label: 'Status unknown' };
    }

    function populateMonitorDevicePicker(devices) {
        const picker = $('.js-mon-device-picker');
        const choice = document.getElementById('mon-device-choice');
        if (!choice) return;
        choice.replaceChildren();
        devices.forEach((device) => {
            const deviceId = device.topic_id || device.device_id;
            const chip = monitorDeviceChipLabel(device.chip);
            const label = device.device_label || device.device_name || 'unnamed';
            const status = monitorDeviceStatus(device);
            const option = document.createElement('button');
            option.type = 'button';
            option.className = 'device-choice-option';
            option.dataset.deviceId = deviceId;
            option.setAttribute(
                'aria-label',
                `${status.label}, ${chip}, ${label}, device ID ${device.device_id}`
            );

            const dot = document.createElement('span');
            dot.className = `dot ${status.dotClass}`;
            dot.setAttribute('aria-hidden', 'true');
            const chipText = document.createElement('span');
            chipText.className = 'device-choice-chip';
            chipText.textContent = chip;
            const nameText = document.createElement('span');
            nameText.className = 'device-choice-name';
            nameText.textContent = label;
            const idText = document.createElement('span');
            idText.className = 'device-choice-id';
            idText.textContent = device.device_id;
            option.append(dot, chipText, nameText, idText);
            choice.appendChild(option);
        });
        if (picker) picker.hidden = false;
        choice.querySelector('.device-choice-option')?.focus({ preventScroll: true });
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

    function monitorShowDeviceSelection() {
        if (conn.mode !== 'ble' && conn.status === 'connecting') {
            setStatus('disconnected');
            return;
        }
        renderConnection();
    }

    function monitorFinishDiscovery(client) {
        monitor.discoveryActive = false;
        monitorUnsubscribeDiscovery(client);
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
            monitorShowDeviceSelection();
            return;
        }
        const deviceInput = document.getElementById('mon-device');
        monitorStatus('No devices discovered. Enter a device ID.');
        if (deviceInput) {
            markMonitorFieldError(deviceInput, 'Enter a device ID.');
            deviceInput.focus({ preventScroll: true });
        }
        monitorShowDeviceSelection();
    }

    function monitorStartDiscovery(client, prefix) {
        resetMonitorDevicePicker();
        monitorUnsubscribeDiscovery(client);
        monitor.discoveryActive = true;
        monitor.discoveredDevices = {};
        monitor.discoveryPrefix = prefix;
        monitor.protocol.setTopicPrefix(prefix);
        monitor.protocol.setDevice('');
        monitor.discoveryTopics = MqttProtocolClient.discoveryTopics(prefix);
        monitorStatus('Scanning MQTT for devices…');
        toast('Scanning MQTT for devices…');
        syncMonitorDemoButton();
        client.subscribe(monitor.discoveryTopics, (error) => {
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
        monitor.protocol.setTopicPrefix(prefix);
        monitor.protocol.setDevice(device);
        const subscriptionTopic = monitor.protocol.subscriptionTopic;
        monitor.boundDeviceId = device;
        monitor.inputMode = 'mqtt';
        client.subscribe(subscriptionTopic, async (error) => {
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
            if (pendingLiveDestination || route === 'monitor' || route === 'configure') {
                setDeviceView('live');
            }
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
        });
    }

    async function monitorConnect() {
        const connection = validateMonitorConnection();
        if (!connection) {
            monitor.closingBleForLive = false;
            track('tool_connection', {
                tool_name: 'monitor', entry_point: connectionIntentRoute(),
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
                tool_name: 'monitor', entry_point: connectionIntentRoute(),
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
        if (conn.status === 'disconnected') {
            rememberConnectionOrigin();
            setStatus('connecting');
        }
        monitorStopAll('replaced');
        monitor.closing = false;
        resetMonitorLiveView();
        monitor.boundDeviceId = device || '';
        monitor.handoffReady = false;
        monitor.startedAt = Date.now();
        monitor.entryPoint = connectionIntentRoute();
        monitor.readyState = '';
        monitor.readyAt = 0;
        monitor.readyTracked = false;
        monitor.brokerUrl = url;
        monitorStatus('Connecting to ' + url + ' …');
        toast('Connecting to the broker…');
        // The URL is not tracked: it would carry the user's broker address.
        track('tool_connection', {
            tool_name: 'monitor', entry_point: connectionIntentRoute(),
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
        monitor.protocol = new MqttProtocolClient(client, { topicPrefix: prefix });
        renderConnection();
        monitor.connectionTimer = setTimeout(() => {
            if (monitor.client !== client || monitor.connectedAt) return;
            monitorStatus('Connection failed: broker WebSocket timed out.');
            monitor.closingBleForLive = false;
            track('tool_connection', {
                tool_name: 'monitor',
                entry_point: monitor.entryPoint,
                transport: 'mqtt_websocket',
                result: 'failure',
                error_type: 'ConnectionTimeout'
            });
            monitorStopAll('connection_timeout');
            if (conn.mode === 'ble') setDeviceView('connectivity');
            else if (conn.status === 'connecting') setStatus('disconnected');
        }, MONITOR_CONNECTION_TIMEOUT_MS);
        monitor.protocol.on('message', ingestMqttMessage);
        monitor.protocol.on('protocol-error', (error) => {
            console.warn('Ignored malformed ESPectre MQTT payload:', error.message);
        });
        client.on('connect', () => {
            if (monitor.client !== client) return;
            clearTimeout(monitor.connectionTimer);
            monitor.connectionTimer = 0;
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
            monitor.protocol.ingest(topic, payload);
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
        if (!monitorIsMqttLive() || !monitor.protocol?.baseTopic) {
            const error = new Error('Connect through the broker before changing the device.');
            statusFn(error.message);
            return Promise.reject(error);
        }
        statusFn(pendingMessage);
        return monitor.protocol.publishCommand(fields, { timeoutMs });
    }

    function diagnosticsRequestPending() {
        return monitor.protocol?.hasPendingCommand('stats') || false;
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
        monitor.diagIntervalMs = 0;
    }

    function syncDiagnosticsPolling() {
        const canPoll = diagnosticsPanelOpen()
            && (conn.mode === 'demo' || monitorIsMqttLive());
        if (!canPoll) {
            stopDiagnosticsPolling();
            return;
        }
        const interval = publishIntervalMs();
        if (monitor.diagTimer && monitor.diagIntervalMs === interval) return;
        stopDiagnosticsPolling();
        monitorRequestStats();
        monitor.diagIntervalMs = interval;
        monitor.diagTimer = setInterval(monitorRequestStats, interval);
    }

    async function monitorRequestStats() {
        if (!diagnosticsPanelOpen()) {
            stopDiagnosticsPolling();
            return;
        }
        if (conn.mode === 'demo') {
            monitorStats({
                traffic_tx_pps: csiTargetPps(),
                csi_callback_pps: Math.max(1, csiTargetPps() - 4),
                csi_filtered_pps: 6,
                csi_admitted_pps: Math.max(1, csiTargetPps() - 16),
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

    function monitorCancelConnection() {
        monitor.closingBleForLive = false;
        track('tool_connection', {
            tool_name: 'monitor',
            entry_point: monitor.entryPoint,
            transport: 'mqtt_websocket',
            result: 'cancelled'
        });
        monitorStopAll('cancelled');
        monitorStatus('Connection cancelled.');
        if (conn.mode === 'ble') setDeviceView('connectivity');
        else {
            setStatus('disconnected');
            renderConnection();
        }
    }

    function monitorEditOrCancel() {
        if (monitorConnectionPending()) {
            monitorCancelConnection();
            return;
        }
        monitorStartBle();
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
        const presetName = document.getElementById('mon-mqtt-preset').value;
        updateMqttPresetNote('monitor', presetName);
        applyMqttPresetFieldLocks('monitor', MQTT_PRESETS[presetName].monitor);
        $('.js-mon-connect').addEventListener('click', () => {
            if (monitorConnectionPending()) {
                monitorCancelConnection();
                return;
            }
            monitorConnect();
        });
        document.getElementById('mon-mqtt-preset').addEventListener('change', (event) => {
            applyMonitorMqttPreset(event.currentTarget.value);
        });
        ['mon-host', 'mon-port', 'mon-topic-prefix', 'mon-device', 'mon-path'].forEach((id) => {
            const input = document.getElementById(id);
            input.addEventListener('input', () => clearMonitorFieldError(input));
        });
        const deviceChoice = document.getElementById('mon-device-choice');
        if (deviceChoice) {
            deviceChoice.addEventListener('click', (event) => {
                const selected = event.target.closest('.device-choice-option');
                if (!selected || !deviceChoice.contains(selected)) return;
                monitorSelectDevice(selected.dataset.deviceId);
            });
        }
        const diagnostics = $('.device-live-diagnostics');
        if (diagnostics) {
            diagnostics.addEventListener('toggle', syncDiagnosticsPolling);
        }
        $('.js-device-edit-connectivity').addEventListener('click', monitorEditOrCancel);
        $('.js-monitor-name-trigger').addEventListener('click', startMonitorDeviceNameEdit);
        const nameInput = $('.js-monitor-name-input');
        nameInput.addEventListener('blur', () => { saveMonitorDeviceNameOnBlur(); });
        nameInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                nameInput.blur();
            } else if (event.key === 'Escape') {
                event.preventDefault();
                cancelMonitorDeviceNameEdit();
            }
        });
        $$('.js-firmware-update-notice').forEach((button) => {
            button.addEventListener('click', (event) => otaOpen(event.currentTarget));
        });
        bindThresholdControls();
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

    const theremin = { ctx: null, osc: null, gain: null, raf: null, smoothed: 0, lastAt: 0 };

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
        theremin.lastAt = 0;
        $('.js-th-toggle').textContent = '⏹ Stop sound';
        const loop = () => {
            const nowMs = performance.now();
            const dt = theremin.lastAt ? Math.min(0.08, (nowMs - theremin.lastAt) / 1000) : 1 / 60;
            theremin.lastAt = nowMs;
            const f = energyFraction();
            const tau = evaluationIntervalMs() / 2000;
            const alpha = 1 - Math.exp(-dt / tau);
            theremin.smoothed += (f - theremin.smoothed) * alpha;
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
        theremin.lastAt = 0;
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
            if (pending.attempts >= CONFIG_VERIFICATION_MAX_ATTEMPTS) {
                finishConfigVerification('unconfirmed', 'VerificationTimeout');
            } else {
                requestConfigVerification();
            }
        }, CONFIG_VERIFICATION_RETRY_MS);
    }

    function beginConfigVerification(action, verify) {
        if (pendingConfigVerification) finishConfigVerification('unconfirmed', 'Superseded');
        pendingConfigVerification = { action, verify, attempts: 0, timer: null };
        pendingConfigVerification.timer = setTimeout(
            requestConfigVerification,
            CONFIG_VERIFICATION_INITIAL_DELAY_MS
        );
    }

    function evaluateConfigVerification(snapshot) {
        const pending = pendingConfigVerification;
        if (!pending) return;
        clearTimeout(pending.timer);
        if (pending.verify(snapshot)) {
            finishConfigVerification('success');
        } else if (pending.attempts >= CONFIG_VERIFICATION_MAX_ATTEMPTS) {
            finishConfigVerification('unconfirmed', 'VerificationMismatch');
        } else {
            pending.timer = setTimeout(requestConfigVerification, CONFIG_VERIFICATION_RETRY_MS);
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
        const enteredHost = cfgValue('cfg-mqtt-host').trim();
        const presetName = cfgValue('cfg-mqtt-preset');
        const host = SECURE_CLOUD_MQTT_PRESETS.has(presetName)
                && enteredHost && !/^mqtts?:\/\//i.test(enteredHost)
            ? 'mqtts://' + enteredHost
            : enteredHost;
        const username = cfgValue('cfg-mqtt-user').trim();
        const password = cfgValue('cfg-mqtt-pass');
        if (!host || !cfgValue('cfg-mqtt-port')) {
            cfgValidationFailed('set_mqtt', 'MQTT needs a host and port.');
            return;
        }
        if (!browserBrokerHost(host)) {
            cfgValidationFailed('set_mqtt', 'Complete the MQTT broker address after mqtts://.');
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
        applyConfigureMqttPreset('home_assistant');
    }

    async function cfgClearMqtt() {
        const ok = await cfgApply(
            'clear_mqtt', 'MQTT settings cleared.', () => 'CLEAR_MQTT_CONFIG',
            (snapshot) => !snapshot.mqtt_host);
        if (ok) applyConfigureMqttDefaults();
    }

    async function cfgSaveDeviceLabel(label) {
        if (conn.mode === 'mqtt') {
            try {
                window.ESPectreBleClient.buildDeviceLabelCommand(label);
            } catch (error) {
                if (error && error.name === 'ESPectreValidationError') {
                    cfgValidationFailed('set_device', error.message);
                    return false;
                }
                throw error;
            }
            try {
                await monitorPublishCommand(
                    { command: 'set_device_label', device_label: label },
                    { pendingMessage: 'Updating device name…', statusFn: () => {} }
                );
                toast('Device name saved.');
                track('configure_change', { action: 'set_device', result: 'accepted' });
                return true;
            } catch (error) {
                toast('Write failed: ' + (error.message || error));
                track('configure_change', {
                    action: 'set_device', result: 'failure', error_type: errorType(error)
                });
                return false;
            }
        }
        return cfgApply('set_device', 'Device name saved.',
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

    function beginOtaTracking() {
        if (otaTracking) finishOtaTracking('unconfirmed', 'Superseded', otaTracking.lastState);
        otaTracking = { startedAt: Date.now(), attempts: 0, lastState: 'starting' };
        clearTimeout(otaPollTimer);
        otaPollTimer = setTimeout(() => {
            finishOtaTracking('unconfirmed', 'StatusTimeout', otaTracking?.lastState);
        }, OTA_TRACKING_TIMEOUT_MS);
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
        const version = conn.firmwareVersion || (latestDeviceInfo && latestDeviceInfo.firmware_version) || '';
        if (otaTargetVersion && version !== otaTargetVersion) {
            setOtaModalDescription('Device is back online. Verifying the updated firmware version…');
            return;
        }
        otaAwaitingReconnect = false;
        otaBusy = false;
        finishOtaTracking('success', null, 'reconnected');
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
        otaTargetVersion = '';
        otaCheckTransport = '';
        maybeStartSilentOtaCheck();
    }

    function handleOtaDeviceAvailability(online) {
        if (!online) {
            if (otaBusy || otaTracking || otaState === 'reboot_scheduled') {
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
        return value;
    }

    function resetOtaChannelSelection() {
        otaDefaultChannel = '';
        otaChannelChanged = false;
        const el = document.getElementById('ota-channel');
        if (el) el.value = 'release';
    }

    function applyOtaDefaultChannel(channel) {
        const normalized = String(channel || '').trim().toLowerCase();
        if (!['release', 'preview', 'develop'].includes(normalized)) return;
        if (!otaDefaultChannel) otaDefaultChannel = normalized;
        if (otaChannelChanged) return;
        const el = document.getElementById('ota-channel');
        if (el) el.value = otaDefaultChannel;
    }

    function otaCommandFields(command) {
        const channel = selectedOtaChannel();
        return channel ? { command, channel } : { command };
    }

    function syncOtaUpdateButton() {
        const button = $('.js-ota-start');
        if (!button) return;
        button.disabled = conn.mode !== 'mqtt' || !monitorIsMqttLive()
            || otaActionPending || otaBusy || !otaUpdateAvailable;
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
            el.hidden = Boolean(flash.usbDialog) || Boolean(bleClient)
                || conn.mode === 'ble' || otaSupported === false;
        });
    }

    function applyOtaStatus(status) {
        if (!status || typeof status !== 'object') return;
        applyOtaDefaultChannel(status.default_channel || (!otaDefaultChannel ? status.channel : ''));
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
        if (status.target_version !== undefined) {
            const targetVersion = String(status.target_version || '');
            if (targetVersion || !otaBusy) otaTargetVersion = targetVersion;
            write('cfg-ota-target', targetVersion || '—');
        }
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
        maybeStartSilentOtaCheck();
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
        return conn.mode === 'mqtt' && monitorIsMqttLive() ? 'mqtt' : '';
    }

    function runOtaCheck({ manual = false } = {}) {
        if (conn.mode === 'demo') return;
        const transport = currentOtaCheckTransport();
        if (!transport) return;
        if (!manual && transport && otaCheckTransport === transport) return;
        if (otaState === 'checking' && manual) return;
        otaState = 'checking';
        otaMessage = '';
        syncFirmwareUpdateNotice();
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

    function maybeStartSilentOtaCheck() {
        if (!otaDefaultChannel || otaBusy) return;
        if (['checking', 'downloading', 'applying', 'reboot_scheduled'].includes(otaState)) return;
        startSilentOtaCheck();
    }

    function startManualOtaCheck() {
        runOtaCheck({ manual: true });
    }

    function otaOpen(returnFocus) {
        const modal = $('.js-ota-modal');
        otaModalReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        modal.querySelector('.modal-card').focus();
    }

    function otaClose(restoreFocus = true) {
        const modal = $('.js-ota-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (restoreFocus && otaModalReturnFocus && otaModalReturnFocus.isConnected) {
            otaModalReturnFocus.focus();
        }
        otaModalReturnFocus = null;
    }

    async function cfgOtaStart() {
        if (conn.mode !== 'mqtt' || !monitorIsMqttLive()) return;
        otaActionPending = true;
        syncOtaUpdateButton();
        const description = $('.js-ota-modal') && $('.js-ota-modal').querySelector('.modal-description');
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
    }

    function configureInit() {
        const presetName = document.getElementById('cfg-mqtt-preset').value;
        updateMqttPresetNote('configure', presetName);
        applyMqttPresetFieldLocks('configure', MQTT_PRESETS[presetName].configure);
        $('.js-wifi-save').addEventListener('click', cfgSaveWifi);
        $('.js-wifi-clear').addEventListener('click', cfgClearWifi);
        const startBle = $('.js-cfg-start-ble');
        if (startBle) startBle.addEventListener('click', monitorStartBle);
        $('.js-mqtt-save').addEventListener('click', cfgSaveMqtt);
        $('.js-mqtt-clear').addEventListener('click', cfgClearMqtt);
        document.getElementById('cfg-mqtt-preset').addEventListener('change', (event) => {
            applyConfigureMqttPreset(event.currentTarget.value);
        });
        $('.js-configure-name-trigger').addEventListener('click', startConfigureDeviceNameEdit);
        const nameInput = $('.js-configure-name-input');
        nameInput.addEventListener('blur', () => { saveConfigureDeviceNameOnBlur(); });
        nameInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                nameInput.blur();
            } else if (event.key === 'Escape') {
                event.preventDefault();
                cancelConfigureDeviceNameEdit();
            }
        });
        $('.js-ota-start').addEventListener('click', cfgOtaStart);
        const otaChannel = document.getElementById('ota-channel');
        if (otaChannel) {
            otaChannel.addEventListener('change', () => {
                if (conn.mode === null) return;
                otaChannelChanged = true;
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
            if (event.key !== 'Escape') return;
            if (!$('.js-matter-modal').hidden) matterClose();
            else if (!$('.js-ota-modal').hidden) otaClose();
        });
    }

    /* ================================================================ game */

    const GAME_ORB_POINTS = 100;
    const GAME_START_DELAY_MS = 700;
    const gameGhostImage = new Image();
    gameGhostImage.decoding = 'async';
    gameGhostImage.src = '/assets/images/brand/espectre-logo.svg';
    const gameFactoryImage = new Image();
    gameFactoryImage.decoding = 'async';
    gameFactoryImage.src = '/assets/images/game/hardware-factory.png';
    const gameAudio = {
        context: null,
        master: null,
        music: null,
        motionOscillator: null,
        motionGain: null,
        musicTimer: null,
        nextNoteAt: 0,
        noteIndex: 0,
        motionSmoothed: 0,
        enabled: true
    };
    const GAME_MUSIC_NOTES = [220, 0, 330, 0, 262, 0, 392, 0, 247, 0, 370, 0, 294, 0, 440, 0];

    const game = {
        phase: 'idle',   // idle | ready | running | done
        score: 0,
        orbs: 0,
        best: 0,
        distance: 0,
        scrollX: 0,
        displayDistance: -1,
        elapsed: 0,
        raf: null,
        previewRaf: null,
        readyTimer: null,
        lastFrameAt: 0,
        previewLastFrameAt: 0,
        nextSpawn: 0,
        flightActive: false,
        manualFlight: false,
        width: 840,
        height: 340,
        dpr: 1,
        ctx: null,
        player: { x: 0, y: 0, size: 54, vy: 0, grounded: true },
        entities: [],
        particles: [],
        hitFlash: 0
    };

    function gameSet(selector, value) {
        const el = $(selector);
        if (el) el.textContent = value;
    }

    function gameMsg(message) {
        gameSet('.js-game-msg', message);
    }

    function gameAudioEnsure() {
        if (!gameAudio.enabled) return null;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return null;
        if (!gameAudio.context) {
            gameAudio.context = new AudioContext();
            gameAudio.master = gameAudio.context.createGain();
            gameAudio.master.gain.value = 0.32;
            gameAudio.master.connect(gameAudio.context.destination);
            gameAudio.music = gameAudio.context.createGain();
            gameAudio.music.gain.value = 0.22;
            gameAudio.music.connect(gameAudio.master);
            gameAudio.motionGain = gameAudio.context.createGain();
            gameAudio.motionGain.gain.value = 0.0001;
            gameAudio.motionGain.connect(gameAudio.master);
        }
        gameAudio.context.resume().catch(() => {});
        return gameAudio.context;
    }

    function gameTone(frequency, start, duration, {
        type = 'sine',
        gain = 0.1,
        endFrequency = frequency,
        destination = null
    } = {}) {
        const context = gameAudio.context;
        if (!context || !frequency) return;
        const oscillator = context.createOscillator();
        const envelope = context.createGain();
        oscillator.type = type;
        oscillator.frequency.setValueAtTime(frequency, start);
        oscillator.frequency.exponentialRampToValueAtTime(Math.max(1, endFrequency), start + duration);
        envelope.gain.setValueAtTime(0.0001, start);
        envelope.gain.exponentialRampToValueAtTime(gain, start + Math.min(0.025, duration * 0.2));
        envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
        oscillator.connect(envelope);
        envelope.connect(destination || gameAudio.master);
        oscillator.start(start);
        oscillator.stop(start + duration + 0.03);
    }

    function gamePlaySound(kind) {
        const context = gameAudioEnsure();
        if (!context) return;
        const now = context.currentTime + 0.01;
        if (kind === 'start') {
            gameTone(392, now, 0.09, { type: 'triangle', gain: 0.09, endFrequency: 523 });
            gameTone(523, now + 0.1, 0.14, { type: 'triangle', gain: 0.1, endFrequency: 659 });
        } else if (kind === 'orb') {
            gameTone(740, now, 0.06, { type: 'sine', gain: 0.08, endFrequency: 880 });
            gameTone(1047, now + 0.065, 0.12, { type: 'sine', gain: 0.07, endFrequency: 1319 });
        } else if (kind === 'hit') {
            gameTone(176, now, 0.28, { type: 'sawtooth', gain: 0.12, endFrequency: 48 });
            gameTone(88, now + 0.04, 0.35, { type: 'triangle', gain: 0.12, endFrequency: 34 });
        }
    }

    function gameScheduleMusic() {
        if (!gameAudio.context || gameAudio.musicTimer === null || game.phase === 'done') return;
        const context = gameAudio.context;
        const stepSeconds = 0.22;
        while (gameAudio.nextNoteAt < context.currentTime + 0.65) {
            const note = GAME_MUSIC_NOTES[gameAudio.noteIndex % GAME_MUSIC_NOTES.length];
            if (note) {
                gameTone(note, gameAudio.nextNoteAt, 0.16, {
                    type: 'triangle',
                    gain: 0.055,
                    endFrequency: note * 1.004,
                    destination: gameAudio.music
                });
            }
            if (gameAudio.noteIndex % 4 === 0) {
                gameTone(55, gameAudio.nextNoteAt, 0.18, {
                    type: 'sine',
                    gain: 0.065,
                    endFrequency: 52,
                    destination: gameAudio.music
                });
            }
            gameAudio.noteIndex += 1;
            gameAudio.nextNoteAt += stepSeconds;
        }
        gameAudio.musicTimer = setTimeout(gameScheduleMusic, 180);
    }

    function gameStartMusic() {
        const context = gameAudioEnsure();
        if (!context || gameAudio.musicTimer !== null) return;
        gameAudio.music.gain.cancelScheduledValues(context.currentTime);
        gameAudio.music.gain.setValueAtTime(0.0001, context.currentTime);
        gameAudio.music.gain.exponentialRampToValueAtTime(0.22, context.currentTime + 0.12);
        gameAudio.noteIndex = 0;
        gameAudio.nextNoteAt = context.currentTime + 0.04;
        gameAudio.musicTimer = setTimeout(gameScheduleMusic, 0);
    }

    function gameStartMotionSound() {
        const context = gameAudioEnsure();
        if (!context || gameAudio.motionOscillator) return;
        gameAudio.motionOscillator = context.createOscillator();
        gameAudio.motionOscillator.type = 'sine';
        gameAudio.motionOscillator.frequency.value = 100;
        gameAudio.motionOscillator.connect(gameAudio.motionGain);
        gameAudio.motionOscillator.start();
    }

    function gameUpdateMotionSound(dt) {
        if (!gameAudio.context || !gameAudio.motionOscillator || !gameAudio.motionGain) return;
        const tau = evaluationIntervalMs() / 2000;
        const alpha = 1 - Math.exp(-dt / tau);
        gameAudio.motionSmoothed += (energyFraction() - gameAudio.motionSmoothed) * alpha;
        const context = gameAudio.context;
        const now = context.currentTime;
        const frequency = 96 * Math.pow(2, gameAudio.motionSmoothed * 1.8);
        const audible = (game.phase === 'ready' || game.phase === 'running') && game.flightActive;
        gameAudio.motionOscillator.frequency.setTargetAtTime(frequency, now, 0.06);
        gameAudio.motionGain.gain.setTargetAtTime(
            audible ? 0.014 + gameAudio.motionSmoothed * 0.05 : 0.0001,
            now,
            0.08
        );
    }

    function gameStopMusic() {
        clearTimeout(gameAudio.musicTimer);
        gameAudio.musicTimer = null;
        if (gameAudio.context && gameAudio.music) {
            const now = gameAudio.context.currentTime;
            gameAudio.music.gain.cancelScheduledValues(now);
            gameAudio.music.gain.setValueAtTime(Math.max(0.0001, gameAudio.music.gain.value), now);
            gameAudio.music.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);
        }
        if (gameAudio.context && gameAudio.motionGain) {
            gameAudio.motionGain.gain.setTargetAtTime(0.0001, gameAudio.context.currentTime, 0.06);
        }
    }

    function gameRenderSoundControl() {
        const button = $('.js-game-sound');
        if (!button) return;
        button.textContent = gameAudio.enabled ? 'Sound on' : 'Sound off';
        button.setAttribute('aria-label', gameAudio.enabled ? 'Mute game audio' : 'Enable game audio');
        button.setAttribute('aria-pressed', String(gameAudio.enabled));
    }

    function gameToggleSound() {
        gameAudio.enabled = !gameAudio.enabled;
        if (!gameAudio.enabled) gameStopMusic();
        else if (game.phase === 'ready' || game.phase === 'running') {
            gameStartMusic();
            gameStartMotionSound();
        }
        gameRenderSoundControl();
    }

    function gameSetPhase(phase, badge) {
        game.phase = phase;
        const screen = $('.game-screen');
        if (screen) screen.dataset.phase = phase;
        gameSet('.js-game-badge', badge);
        const play = $('.js-game-start');
        if (play) play.hidden = phase === 'ready' || phase === 'running';
        gameSyncFullscreenButton();
    }

    function gameFullscreenElement() {
        return document.fullscreenElement || document.webkitFullscreenElement || null;
    }

    function gameSyncFullscreenButton() {
        const screen = $('.game-screen');
        const button = $('.js-game-fullscreen');
        if (!screen || !button) return;
        const supported = Boolean(
            (screen.requestFullscreen || screen.webkitRequestFullscreen)
            && (document.exitFullscreen || document.webkitExitFullscreen)
        );
        const active = gameFullscreenElement() === screen;
        button.hidden = !supported;
        button.textContent = active ? 'Exit full screen' : 'Full screen';
        button.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
        button.setAttribute('aria-pressed', String(active));
    }

    function gameExitFullscreen() {
        const screen = $('.game-screen');
        if (!screen || gameFullscreenElement() !== screen) return;
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (!exit) return;
        Promise.resolve(exit.call(document)).catch(() => {});
    }

    async function gameToggleFullscreen() {
        const screen = $('.game-screen');
        if (!screen) return;
        if (gameFullscreenElement() === screen) {
            gameExitFullscreen();
            return;
        }
        const request = screen.requestFullscreen || screen.webkitRequestFullscreen;
        if (!request) return;
        try {
            await request.call(screen);
        } catch (error) {
            toast('Full screen is unavailable.');
        }
    }

    function gameOnFullscreenChange() {
        gameSyncFullscreenButton();
        requestAnimationFrame(gameResizeCanvas);
        if (gameFullscreenElement() === $('.game-screen')) {
            $('.js-game-canvas').focus({ preventScroll: true });
        }
    }

    function gameGroundY() {
        return game.height * 0.79;
    }

    function gamePlayerSize() {
        return Math.max(38, Math.min(58, game.height * 0.17));
    }

    function gameFlightY() {
        const size = game.player.size || gamePlayerSize();
        return Math.max(game.height * 0.14, gameGroundY() - size - game.height * 0.34);
    }

    function gameResetPlayer() {
        const size = gamePlayerSize();
        game.player = {
            x: game.width * 0.14,
            y: gameGroundY() - size,
            size,
            vy: 0,
            grounded: true
        };
    }

    function gameResizeCanvas() {
        const canvas = $('.js-game-canvas');
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return;

        const oldWidth = game.width;
        const oldHeight = game.height;
        const oldGround = gameGroundY();
        const oldSize = game.player.size;
        const air = oldGround - (game.player.y + oldSize);
        game.width = rect.width;
        game.height = rect.height;
        game.dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.round(game.width * game.dpr);
        canvas.height = Math.round(game.height * game.dpr);
        game.ctx = canvas.getContext('2d');
        if (!game.ctx) return;
        game.ctx.setTransform(game.dpr, 0, 0, game.dpr, 0, 0);

        const scaleX = oldWidth > 0 ? game.width / oldWidth : 1;
        const scaleY = oldHeight > 0 ? game.height / oldHeight : 1;
        game.scrollX *= scaleX;
        const size = gamePlayerSize();
        game.player.x = game.width * 0.14;
        game.player.size = size;
        const maxAir = Math.max(0, gameGroundY() - size - gameFlightY());
        game.player.y = gameGroundY() - size - Math.min(maxAir, Math.max(0, air * scaleY));
        game.entities.forEach((entity) => {
            entity.x *= scaleX;
            if (entity.kind === 'orb') {
                entity.y = gameOrbY(entity.lane);
                entity.radius = Math.max(5, Math.min(8, size * 0.12));
            } else {
                const dimensions = gameObstacleDimensions(entity.obstacleKind);
                entity.w = dimensions.w;
                entity.h = dimensions.h;
                entity.y = gameObstacleY(entity.obstacleKind, entity.h);
            }
        });
        gameDraw();
    }

    function gameScore() {
        return Math.floor(game.distance) + game.orbs * GAME_ORB_POINTS;
    }

    function gameUpdateStats() {
        game.score = gameScore();
        gameSet('.js-game-score', String(game.score));
        gameSet('.js-game-orbs', String(game.orbs));
        game.displayDistance = Math.floor(game.distance);
        gameSet('.js-game-distance', game.displayDistance + ' m');
        gameSet('.js-game-best', String(game.best));
    }

    function reportGameAbandon(reason) {
        if (game.phase === 'idle' || game.phase === 'done') return;
        track('game_abandon', {
            input_mode: connectionInputMode(),
            score: game.score,
            distance: Math.floor(game.distance),
            reason
        });
        gameReset();
    }

    function gameReset() {
        clearTimeout(game.readyTimer);
        cancelAnimationFrame(game.raf);
        cancelAnimationFrame(game.previewRaf);
        game.raf = null;
        game.previewRaf = null;
        game.score = 0;
        game.orbs = 0;
        game.distance = 0;
        game.scrollX = 0;
        game.elapsed = 0;
        game.entities = [];
        game.particles = [];
        game.hitFlash = 0;
        game.flightActive = false;
        game.manualFlight = false;
        gameAudio.motionSmoothed = 0;
        gameStopMusic();
        gameSetPhase('idle', 'READY');
        gameMsg('Move to fly. Stay quiet to descend. Distance earns points, and orbs add bonuses.');
        gameUpdateStats();
        const start = $('.js-game-start');
        if (start) start.textContent = 'Start game';
        gameResetPlayer();
        gameDraw();
        gameStartPreview();
    }

    function gameObstacleDimensions(kind) {
        const size = game.player.size || gamePlayerSize();
        if (kind === 'aerial_spikes') return { w: size * 1.12, h: size * 0.7 };
        if (kind === 'gate') return { w: size * 0.76, h: size * 0.86 };
        return { w: size * 1.08, h: size * 0.56 };
    }

    function gameObstacleY(kind, height) {
        if (kind === 'aerial_spikes') return gameFlightY() + game.player.size * 0.1;
        return gameGroundY() - height;
    }

    function gameAddObstacle(kind, x) {
        const dimensions = gameObstacleDimensions(kind);
        game.entities.push({
            kind: 'obstacle',
            obstacleKind: kind,
            x,
            y: gameObstacleY(kind, dimensions.h),
            ...dimensions
        });
    }

    function gameOrbY(lane) {
        const size = game.player.size || gamePlayerSize();
        return lane === 'high'
            ? gameFlightY() + size * 0.5
            : gameGroundY() - size * 0.46;
    }

    function gameAddOrb(x, lane) {
        game.entities.push({
            kind: 'orb',
            x,
            y: gameOrbY(lane),
            lane,
            radius: Math.max(5, Math.min(8, game.player.size * 0.12)),
            phase: Math.random() * Math.PI * 2
        });
    }

    function gameSpawnCourse() {
        const size = game.player.size;
        const startX = game.width + size;
        const pattern = Math.random();
        if (pattern < 0.42) {
            const obstacleX = startX + size * 3.2;
            gameAddObstacle(Math.random() < 0.56 ? 'spikes' : 'gate', obstacleX);
            for (let i = 0; i < 4; i += 1) {
                gameAddOrb(obstacleX - size * (2.8 - i * 0.72), 'high');
            }
        } else if (pattern < 0.72) {
            const obstacleX = startX + size * 3.2;
            gameAddObstacle('aerial_spikes', obstacleX);
            for (let i = 0; i < 4; i += 1) {
                gameAddOrb(obstacleX - size * (2.8 - i * 0.72), 'low');
            }
        } else if (pattern < 0.86) {
            for (let i = 0; i < 5; i += 1) {
                gameAddOrb(startX + i * size * 0.68, 'low');
            }
        } else {
            for (let i = 0; i < 5; i += 1) {
                gameAddOrb(startX + i * size * 0.7, 'high');
            }
        }
    }

    function gameRectsOverlap(a, b) {
        return a.x < b.x + b.w && a.x + a.w > b.x
            && a.y < b.y + b.h && a.y + a.h > b.y;
    }

    function gameOrbTouchesPlayer(orb, player) {
        const closestX = Math.max(player.x, Math.min(orb.x, player.x + player.w));
        const closestY = Math.max(player.y, Math.min(orb.y, player.y + player.h));
        const dx = orb.x - closestX;
        const dy = orb.y - closestY;
        return dx * dx + dy * dy <= orb.radius * orb.radius;
    }

    function gameBurst(orb) {
        for (let i = 0; i < 9; i += 1) {
            const angle = (Math.PI * 2 * i) / 9;
            const speed = 28 + Math.random() * 48;
            game.particles.push({
                x: orb.x,
                y: orb.y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 0.42
            });
        }
    }

    function gameFinish() {
        game.hitFlash = 1;
        game.best = Math.max(game.best, game.score);
        gamePlaySound('hit');
        gameStopMusic();
        gameSetPhase('done', 'GAME OVER');
        gameMsg('Obstacle hit — ' + game.score + ' points over ' + Math.floor(game.distance) + ' m.');
        gameUpdateStats();
        $('.js-game-start').textContent = 'Play again';
        track('game_over', {
            input_mode: connectionInputMode(),
            score: game.score,
            orbs: game.orbs,
            distance: Math.floor(game.distance)
        });
    }

    function gameSetFlight(active) {
        game.flightActive = Boolean(active);
        if (game.phase === 'running') {
            gameSet('.js-game-badge', game.flightActive ? 'FLY' : 'GLIDE');
        }
    }

    function gameUpdatePlayer(dt) {
        const ground = gameGroundY();
        const player = game.player;
        const targetY = game.flightActive ? gameFlightY() : ground - player.size;
        const responseSeconds = game.flightActive ? 0.15 : 0.18;
        const blend = 1 - Math.exp(-dt / responseSeconds);
        const previousY = player.y;
        player.y += (targetY - player.y) * blend;
        player.vy = dt > 0 ? (player.y - previousY) / dt : 0;
        player.grounded = Math.abs(player.y - (ground - player.size)) < 1;
    }

    function gamePreviewFrame(now) {
        const previewing = route === 'game' && conn.mode
            && (game.phase === 'idle' || game.phase === 'ready');
        if (!previewing) {
            game.previewRaf = null;
            return;
        }
        const dt = Math.min(0.05, Math.max(0, (now - game.previewLastFrameAt) / 1000));
        game.previewLastFrameAt = now;
        game.elapsed += dt;
        gameUpdatePlayer(dt);
        gameUpdateMotionSound(dt);
        gameDraw();
        game.previewRaf = requestAnimationFrame(gamePreviewFrame);
    }

    function gameStartPreview() {
        if (game.previewRaf || route !== 'game' || !conn.mode
                || (game.phase !== 'idle' && game.phase !== 'ready')) return;
        game.previewLastFrameAt = performance.now();
        game.previewRaf = requestAnimationFrame(gamePreviewFrame);
    }

    function gameStopPreview() {
        cancelAnimationFrame(game.previewRaf);
        game.previewRaf = null;
    }

    function gameUpdate(dt) {
        game.elapsed += dt;
        gameUpdatePlayer(dt);
        gameUpdateMotionSound(dt);
        const player = game.player;
        const speed = Math.max(190, game.width * 0.32)
            + Math.min(game.width * 0.2, game.elapsed * 5.2);

        const travel = speed * dt;
        game.scrollX += travel;
        game.distance += travel / 45;
        game.nextSpawn -= travel;
        if (game.nextSpawn <= 0) {
            gameSpawnCourse();
            game.nextSpawn = Math.max(
                game.width * 0.58,
                speed * (1.75 + Math.random() * 0.45)
            );
        }

        const hitbox = {
            x: player.x + player.size * 0.18,
            y: player.y + player.size * 0.12,
            w: player.size * 0.64,
            h: player.size * 0.76
        };
        let obstacleHit = false;
        game.entities.forEach((entity) => {
            entity.x -= travel;
            if (entity.kind === 'orb') {
                entity.phase += dt * 5;
                if (!entity.collected && gameOrbTouchesPlayer(entity, hitbox)) {
                    entity.collected = true;
                    game.orbs += 1;
                    gameBurst(entity);
                    gamePlaySound('orb');
                    gameUpdateStats();
                }
                return;
            }
            const obstacleHitbox = {
                x: entity.x + 3,
                y: entity.y + 4,
                w: Math.max(1, entity.w - 6),
                h: Math.max(1, entity.h - 4)
            };
            if (gameRectsOverlap(hitbox, obstacleHitbox)) obstacleHit = true;
        });
        game.entities = game.entities.filter((entity) => {
            const width = entity.kind === 'orb' ? entity.radius * 2 : entity.w;
            return !entity.collected && entity.x + width > -24;
        });
        game.particles.forEach((particle) => {
            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;
            particle.vy += game.height * 1.05 * dt;
            particle.life -= dt;
        });
        game.particles = game.particles.filter((particle) => particle.life > 0);
        const distance = Math.floor(game.distance);
        const score = gameScore();
        if (score !== game.score) {
            game.score = score;
            gameSet('.js-game-score', String(score));
        }
        if (distance !== game.displayDistance) {
            game.displayDistance = distance;
            gameSet('.js-game-distance', distance + ' m');
        }
        if (obstacleHit) gameFinish();
    }

    function gameSensingActive() {
        return conn.movement >= gameThreshold();
    }

    function gameOnTelemetry() {
        if (route !== 'game') return;
        gameSetFlight(gameSensingActive());
        gameStartPreview();
    }

    function gameRoundedRect(ctx, x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
    }

    function gameDrawFactoryBackdrop(ctx, width, height) {
        if (!gameFactoryImage.complete || !gameFactoryImage.naturalWidth) return;
        const imageWidth = gameFactoryImage.naturalWidth;
        const imageHeight = gameFactoryImage.naturalHeight;
        const canvasRatio = width / height;
        const imageRatio = imageWidth / imageHeight;
        let sourceX = 0;
        let sourceY = 0;
        let sourceWidth = imageWidth;
        let sourceHeight = imageHeight;
        if (imageRatio > canvasRatio) {
            sourceWidth = imageHeight * canvasRatio;
            const margin = imageWidth - sourceWidth;
            sourceX = margin * (0.5 + Math.sin(game.scrollX * 0.0007) * 0.5);
        } else if (imageRatio < canvasRatio) {
            sourceHeight = imageWidth / canvasRatio;
            sourceY = Math.max(0, imageHeight - sourceHeight);
        }
        ctx.save();
        ctx.globalAlpha = 0.86;
        ctx.drawImage(
            gameFactoryImage,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight,
            0,
            0,
            width,
            height
        );
        ctx.restore();
    }

    function gameDrawFactoryParallax(ctx, ground) {
        const span = game.width + 180;
        ctx.save();
        ctx.globalAlpha = 0.34;
        for (let i = 0; i < 7; i += 1) {
            const x = ((i * 211 - game.scrollX * 0.12) % span + span) % span - 90;
            const y = ground * (0.2 + (i % 3) * 0.13);
            const panelWidth = 42 + (i % 3) * 18;
            const panelHeight = 20 + (i % 2) * 10;
            ctx.fillStyle = '#101b47';
            gameRoundedRect(ctx, x, y, panelWidth, panelHeight, 4);
            ctx.fill();
            ctx.strokeStyle = i % 2 ? 'rgba(121, 139, 255, .75)' : 'rgba(85, 211, 211, .52)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x + 5, y + panelHeight * 0.5);
            ctx.lineTo(x + panelWidth - 8, y + panelHeight * 0.5);
            ctx.stroke();
        }
        ctx.restore();

        ctx.save();
        ctx.globalAlpha = 0.24;
        ctx.strokeStyle = '#6075dd';
        ctx.lineWidth = 1;
        for (let i = 0; i < 9; i += 1) {
            const x = ((i * 173 - game.scrollX * 0.18) % span + span) % span - 70;
            ctx.beginPath();
            ctx.moveTo(x, ground * 0.56);
            ctx.lineTo(x + 24, ground * 0.56);
            ctx.lineTo(x + 34, ground * 0.64);
            ctx.stroke();
        }
        ctx.restore();
    }

    function gameDrawBackground(ctx) {
        const width = game.width;
        const height = game.height;
        const ground = gameGroundY();
        const sky = ctx.createLinearGradient(0, 0, 0, height);
        sky.addColorStop(0, '#070810');
        sky.addColorStop(0.72, '#111328');
        sky.addColorStop(1, '#080911');
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, width, height);
        gameDrawFactoryBackdrop(ctx, width, height);
        gameDrawFactoryParallax(ctx, ground);

        ctx.save();
        ctx.globalAlpha = 0.58;
        for (let i = 0; i < 15; i += 1) {
            const span = width + 90;
            const x = ((i * 137 - game.scrollX * 0.24) % span + span) % span - 45;
            const y = 24 + ((i * 53) % Math.max(40, ground - 88));
            ctx.fillStyle = i % 3 === 0 ? '#9eb0ff' : '#5369d8';
            ctx.beginPath();
            ctx.arc(x, y, i % 3 === 0 ? 1.7 : 1, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        const floor = ctx.createLinearGradient(0, ground, 0, height);
        floor.addColorStop(0, '#20243a');
        floor.addColorStop(0.12, '#121522');
        floor.addColorStop(1, '#08090e');
        ctx.fillStyle = floor;
        ctx.fillRect(0, ground, width, height - ground);
        ctx.fillStyle = '#6677e5';
        ctx.globalAlpha = 0.52;
        ctx.fillRect(0, ground, width, 2);
        ctx.globalAlpha = 1;

        ctx.fillStyle = 'rgba(80, 105, 222, .3)';
        ctx.fillRect(0, ground + 4, width, Math.max(8, height * 0.04));
        for (let i = 0; i < 9; i += 1) {
            const span = width / 8;
            const x = ((i * span - game.scrollX) % width + width) % width;
            ctx.fillStyle = 'rgba(184, 197, 255, .28)';
            ctx.fillRect(x, ground + 7, Math.max(8, span * 0.34), 2);
        }

        ctx.strokeStyle = 'rgba(111, 138, 230, .22)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 8; i += 1) {
            const span = width / 7;
            const x = ((i * span - game.scrollX) % width + width) % width;
            ctx.beginPath();
            ctx.moveTo(x, ground + 5);
            ctx.lineTo(x - height * 0.17, height);
            ctx.stroke();
        }
    }

    function gameDrawOrb(ctx, entity) {
        const y = entity.y + Math.sin(entity.phase) * 2.2;
        ctx.save();
        ctx.shadowColor = 'rgba(255, 194, 91, .82)';
        ctx.shadowBlur = 13;
        const glow = ctx.createRadialGradient(
            entity.x - entity.radius * 0.35,
            y - entity.radius * 0.4,
            entity.radius * 0.1,
            entity.x,
            y,
            entity.radius
        );
        glow.addColorStop(0, '#fff5c0');
        glow.addColorStop(0.34, '#f8c86d');
        glow.addColorStop(1, '#bd7138');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(entity.x, y, entity.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    function gameDrawChip(ctx, x, y, width, height, { hanging = false, label = '' } = {}) {
        const pinLength = Math.max(3, Math.min(7, width * 0.16));
        const pinCount = Math.max(2, Math.floor(height / Math.max(7, height * 0.2)));
        ctx.save();
        ctx.shadowColor = 'rgba(76, 123, 238, .52)';
        ctx.shadowBlur = 10;
        const body = ctx.createLinearGradient(x, y, x + width, y + height);
        body.addColorStop(0, '#5868a8');
        body.addColorStop(0.35, '#222a55');
        body.addColorStop(1, '#11162e');
        ctx.fillStyle = body;
        gameRoundedRect(ctx, x, y, width, height, Math.min(6, width * 0.16));
        ctx.fill();
        ctx.strokeStyle = '#98a8ff';
        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1;
        gameRoundedRect(ctx, x + 1, y + 1, width - 2, height - 2, Math.min(5, width * 0.13));
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.fillStyle = '#090d20';
        gameRoundedRect(ctx, x + width * 0.22, y + height * 0.2, width * 0.56, height * 0.58, 3);
        ctx.fill();
        ctx.strokeStyle = 'rgba(92, 231, 228, .7)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 3; i += 1) {
            const traceY = y + height * (0.34 + i * 0.15);
            ctx.beginPath();
            ctx.moveTo(x + width * 0.08, traceY);
            ctx.lineTo(x + width * 0.22, traceY);
            ctx.lineTo(x + width * 0.28, traceY + (i - 1) * 2);
            ctx.stroke();
        }
        if (label) {
            ctx.fillStyle = '#c2ccff';
            ctx.globalAlpha = 0.78;
            ctx.font = `bold ${Math.max(5, Math.min(8, width * 0.2))}px system-ui`;
            ctx.textAlign = 'center';
            ctx.fillText(label, x + width * 0.5, y + height * 0.62);
            ctx.globalAlpha = 1;
        }
        ctx.fillStyle = '#b8c8ff';
        for (let i = 0; i < pinCount; i += 1) {
            const pinY = y + height * ((i + 0.5) / pinCount) - 1;
            ctx.fillRect(x - pinLength, pinY, pinLength, 2);
            ctx.fillRect(x + width, pinY, pinLength, 2);
        }
        if (hanging) {
            ctx.fillStyle = '#c1ccff';
            const bottomPins = Math.max(2, Math.floor(width / 9));
            for (let i = 0; i < bottomPins; i += 1) {
                const pinX = x + width * ((i + 0.5) / bottomPins) - 1;
                ctx.fillRect(pinX, y + height, 2, pinLength);
            }
        }
        ctx.restore();
    }

    function gameDrawObstacle(ctx, entity) {
        if (entity.obstacleKind === 'aerial_spikes') {
            ctx.save();
            ctx.strokeStyle = 'rgba(160, 177, 255, .75)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(entity.x + entity.w * 0.5, 0);
            ctx.lineTo(entity.x + entity.w * 0.5, entity.y + entity.h * 0.16);
            ctx.stroke();
            gameDrawChip(ctx, entity.x, entity.y + entity.h * 0.16, entity.w, entity.h * 0.74, {
                hanging: true,
                label: 'IO',
            });
            ctx.restore();
            return;
        }

        if (entity.obstacleKind === 'gate') {
            gameDrawChip(ctx, entity.x, entity.y, entity.w, entity.h, { label: 'ESP' });
            return;
        }

        gameDrawChip(ctx, entity.x, entity.y, entity.w, entity.h, { label: 'IC' });
    }

    function gameDrawPlayer(ctx) {
        const player = game.player;
        const canFloat = game.phase === 'idle' || game.phase === 'ready' || game.phase === 'running';
        const bobAmplitude = game.flightActive ? 2.6 : (player.grounded ? 1.2 : 1.8);
        const bob = canFloat
            ? Math.sin(game.elapsed * (game.flightActive ? 7.5 : 10)) * bobAmplitude
            : 0;
        const rotation = game.phase === 'done'
            ? -0.34
            : Math.max(-0.18, Math.min(0.18, player.vy / (game.height * 4)));
        ctx.save();
        ctx.globalAlpha = 0.38;
        ctx.fillStyle = '#05060b';
        ctx.beginPath();
        ctx.ellipse(
            player.x + player.size * 0.5,
            gameGroundY() + 5,
            player.size * (player.grounded ? 0.42 : 0.3),
            player.size * 0.11,
            0,
            0,
            Math.PI * 2
        );
        ctx.fill();
        ctx.restore();

        if (game.flightActive && game.phase === 'running') {
            ctx.save();
            const thrust = ctx.createLinearGradient(0, player.y + player.size * 0.62, 0, player.y + player.size * 1.28);
            thrust.addColorStop(0, 'rgba(112, 133, 255, .44)');
            thrust.addColorStop(1, 'rgba(112, 133, 255, 0)');
            ctx.fillStyle = thrust;
            ctx.beginPath();
            ctx.ellipse(
                player.x + player.size * 0.5,
                player.y + player.size * 0.9,
                player.size * 0.24,
                player.size * 0.42,
                0,
                0,
                Math.PI * 2
            );
            ctx.fill();
            ctx.restore();
        }

        if (game.phase === 'running') {
            for (let i = 2; i > 0; i -= 1) {
                ctx.save();
                ctx.globalAlpha = 0.07 * (3 - i);
                ctx.translate(-i * player.size * 0.25, 0);
                if (gameGhostImage.complete && gameGhostImage.naturalWidth) {
                    ctx.drawImage(gameGhostImage, player.x, player.y + bob, player.size, player.size);
                }
                ctx.restore();
            }
        }

        ctx.save();
        ctx.translate(player.x + player.size / 2, player.y + player.size / 2 + bob);
        ctx.rotate(rotation);
        ctx.shadowColor = game.phase === 'done' ? 'rgba(255, 103, 109, .7)' : 'rgba(86, 111, 255, .65)';
        ctx.shadowBlur = game.phase === 'done' ? 20 : 15;
        if (gameGhostImage.complete && gameGhostImage.naturalWidth) {
            ctx.drawImage(gameGhostImage, -player.size / 2, -player.size / 2, player.size, player.size);
        } else {
            ctx.fillStyle = '#4b7bee';
            gameRoundedRect(ctx, -player.size / 2, -player.size / 2, player.size, player.size, player.size * 0.42);
            ctx.fill();
        }
        ctx.restore();
    }

    function gameDraw() {
        const ctx = game.ctx;
        if (!ctx) return;
        ctx.setTransform(game.dpr, 0, 0, game.dpr, 0, 0);
        ctx.clearRect(0, 0, game.width, game.height);
        gameDrawBackground(ctx);
        game.entities.forEach((entity) => {
            if (entity.kind === 'orb') gameDrawOrb(ctx, entity);
            else gameDrawObstacle(ctx, entity);
        });
        game.particles.forEach((particle) => {
            ctx.save();
            ctx.globalAlpha = Math.min(1, particle.life * 3);
            ctx.fillStyle = '#ffd078';
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 2.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });
        gameDrawPlayer(ctx);
        if (game.hitFlash > 0) {
            ctx.fillStyle = 'rgba(213, 72, 79, .18)';
            ctx.fillRect(0, 0, game.width, game.height);
        }
    }

    function gameFrame(now) {
        if (game.phase !== 'running') {
            game.raf = null;
            gameDraw();
            return;
        }
        const dt = Math.min(0.05, Math.max(0, (now - game.lastFrameAt) / 1000));
        game.lastFrameAt = now;
        gameUpdate(dt);
        gameDraw();
        if (game.phase === 'running') game.raf = requestAnimationFrame(gameFrame);
        else game.raf = null;
    }

    function gameStart() {
        const restartingFromGameOver = game.phase === 'done';
        reportGameAbandon('restart');
        clearTimeout(game.readyTimer);
        cancelAnimationFrame(game.raf);
        gameResizeCanvas();
        game.score = 0;
        game.orbs = 0;
        game.distance = 0;
        game.scrollX = 0;
        game.elapsed = 0;
        game.entities = [];
        game.particles = [];
        game.hitFlash = 0;
        game.manualFlight = false;
        if (restartingFromGameOver) gameResetPlayer();
        gameSetFlight(gameSensingActive());
        gameSpawnCourse();
        game.nextSpawn = Math.max(game.width * 0.58, Math.max(190, game.width * 0.32) * 1.9);
        gameUpdateStats();
        gameSetPhase('ready', 'GET READY');
        gameStartMusic();
        gameStartMotionSound();
        gamePlaySound('start');
        gameStartPreview();
        gameMsg('The Spectre is taking off…');
        $('.js-game-canvas').focus({ preventScroll: true });
        gameDraw();
        track('game_start', { input_mode: connectionInputMode() });
        game.readyTimer = setTimeout(() => {
            if (game.phase !== 'ready') return;
            gameSetPhase('running', game.flightActive ? 'FLY' : 'GLIDE');
            gameStopPreview();
            gameMsg('Move for the high lane. Stay quiet for the low lane. Orbs add 100 points.');
            game.lastFrameAt = performance.now();
            game.raf = requestAnimationFrame(gameFrame);
        }, GAME_START_DELAY_MS);
    }

    function gameDemoFlight(active, event) {
        if (conn.mode !== 'demo' || !['idle', 'ready', 'running'].includes(game.phase)) return;
        if (event) event.preventDefault();
        game.manualFlight = active;
        demoInputEnergy = active ? 1 : 0;
        gameSetFlight(active);
        gameStartPreview();
    }

    function gameInit() {
        const canvas = $('.js-game-canvas');
        $('.js-game-start').addEventListener('click', gameStart);
        $('.js-game-fullscreen').addEventListener('click', gameToggleFullscreen);
        $('.js-game-sound').addEventListener('click', gameToggleSound);
        canvas.addEventListener('pointerdown', (event) => {
            if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
            gameDemoFlight(true, event);
        });
        canvas.addEventListener('pointerup', (event) => gameDemoFlight(false, event));
        canvas.addEventListener('pointercancel', (event) => gameDemoFlight(false, event));
        document.addEventListener('keydown', (event) => {
            if (route !== 'game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
            if (document.activeElement !== canvas) return;
            gameDemoFlight(true, event);
        });
        document.addEventListener('keyup', (event) => {
            if (route !== 'game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
            if (document.activeElement !== canvas) return;
            gameDemoFlight(false, event);
        });
        window.addEventListener('resize', gameResizeCanvas);
        document.addEventListener('fullscreenchange', gameOnFullscreenChange);
        document.addEventListener('webkitfullscreenchange', gameOnFullscreenChange);
        gameGhostImage.addEventListener('load', gameDraw);
        gameFactoryImage.addEventListener('load', gameDraw);
        gameRenderSoundControl();
        gameSyncFullscreenButton();
        gameResetPlayer();
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
            rememberLiveDestination();
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
