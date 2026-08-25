/*
 * ESPectre - Website app shell
 *
 * Hash routing and a persistent session shared by every page. Configure uses
 * the local Direct HTTP transport. Relay is reserved for a future remote
 * connection mode and is not implemented yet.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const routeRegistry = window.ESPectreRoutes;
    if (!routeRegistry) throw new Error('ESPectre route registry is unavailable');
    const sitePolicy = window.ESPectreSite;
    if (!sitePolicy) throw new Error('ESPectre site policy is unavailable');
    const browserSupport = window.ESPectreBrowserSupport && window.ESPectreBrowserSupport.current;
    if (!browserSupport) throw new Error('ESPectre browser capability policy is unavailable');
    const DirectProtocolClient = window.ESPectreDirectClient;
    if (!DirectProtocolClient) throw new Error('ESPectre Direct HTTP client is unavailable');

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    // analytics.js is optional: the app must work with it blocked or absent.
    const track = (name, params) => window.trackEvent ? window.trackEvent(name, params) : false;
    const errorType = (error) => (error && (error.code || error.name)) || 'Error';
    const toolNameForRoute = (routeName) => routeRegistry.groupOf(routeName) === 'tools'
        ? (routeRegistry.get(routeName)?.analyticsName || routeName)
        : 'monitor';
    const activeToolName = () => toolNameForRoute(route);
    const LEGACY_TOOL_ROUTES = Object.freeze({ device: 'tool-configure' });
    const MQTT_PRESETS = Object.freeze({
        home_assistant: Object.freeze({
            configure: Object.freeze({
                host: 'homeassistant.local', port: '1883', hostPlaceholder: 'homeassistant.local'
            })
        }),
        lan_broker: Object.freeze({
            configure: Object.freeze({
                host: '', port: '1883', hostPlaceholder: 'broker.local or 192.168.1.20'
            })
        }),
        emqx_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'deployment-id.ala.region.emqxsl.com', port: '8883',
                hostPlaceholder: 'deployment-id.ala.region.emqxsl.com',
                locked: Object.freeze(['port'])
            })
        }),
        hivemq_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'cluster-id.s1.region.hivemq.cloud', port: '8883',
                hostPlaceholder: 'cluster-id.s1.region.hivemq.cloud',
                locked: Object.freeze(['port'])
            })
        }),
        flespi: Object.freeze({
            configure: Object.freeze({
                host: 'mqtt.flespi.io', port: '8883', hostPlaceholder: 'mqtt.flespi.io',
                locked: Object.freeze(['host', 'port'])
            })
        }),
        cloud_broker: Object.freeze({
            configure: Object.freeze({
                host: 'cluster.example.com', port: '', hostPlaceholder: 'cluster.example.com'
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
            if (!sitePolicy.isLoopbackHostname(location.hostname)) throw error;
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
    const DIRECT_RECONNECT_DELAYS_MS = Object.freeze([500, 1500, 3000]);
    const DIRECT_ENDPOINT_STORAGE_KEY = 'espectre.direct.endpoints.v1';
    const DIRECT_ENDPOINT_LIMIT = 8;
    const DIRECT_FULL_DEVICE_ID = /^[0-9a-f]{16}$/;
    const DIRECT_SHORT_DEVICE_ID = /^[0-9a-f]{6}$/;
    const DIRECT_CANONICAL_HOSTNAME = /^espectre-([0-9a-f]{16})\.local$/;

    class DirectConnectElement extends HTMLElement {
        connectedCallback() {
            if (this.dataset.rendered === 'true') return;
            const template = document.getElementById('direct-connect-template');
            if (!template) throw new Error('Direct connection template is unavailable');
            const surface = this.dataset.surface || 'configure';
            const fragment = template.content.cloneNode(true);
            const inputId = `${surface}-direct-endpoint`;
            const input = fragment.querySelector('.js-direct-endpoint');
            const label = fragment.querySelector('.js-direct-endpoint-label');
            input.id = inputId;
            label.htmlFor = inputId;
            this.replaceChildren(fragment);
            this.dataset.rendered = 'true';
        }
    }

    if (!customElements.get('espectre-direct-connect')) {
        customElements.define('espectre-direct-connect', DirectConnectElement);
    }

    class ConnectionPickerElement extends HTMLElement {
        connectedCallback() {
            if (this.dataset.rendered === 'true') return;
            const template = document.getElementById('connection-picker-template');
            if (!template) throw new Error('Connection picker template is unavailable');
            const surface = this.dataset.surface || 'monitor';
            const fragment = template.content.cloneNode(true);
            const fieldset = fragment.querySelector('.transport-choice');
            fieldset.setAttribute('aria-label', `${surface.replace('-', ' ')} connection method`);
            fragment.querySelectorAll('input[data-connection-mode]').forEach((radio) => {
                const mode = radio.dataset.connectionMode;
                const id = `${surface}-transport-${mode}`;
                radio.id = id;
                radio.name = `${surface}-transport`;
                fragment.querySelector(`label[data-connection-label="${mode}"]`).htmlFor = id;
            });
            fragment.querySelectorAll('[data-connection-panel]').forEach((panel) => {
                panel.dataset.connectionSurface = surface;
            });
            const direct = fragment.querySelector('espectre-direct-connect');
            direct.dataset.surface = surface;
            if (this.dataset.openView) direct.dataset.openView = this.dataset.openView;
            this.replaceChildren(fragment);
            this.querySelectorAll('input[data-connection-mode]').forEach((radio) => {
                radio.addEventListener('change', () => {
                    if (radio.checked) this.select(radio.value);
                });
            });
            this.dataset.rendered = 'true';
            this.select('direct');
        }

        select(mode) {
            const selectedMode = ['direct', 'demo', 'relay'].includes(mode) ? mode : 'direct';
            const radio = this.querySelector(`input[data-connection-mode="${selectedMode}"]`);
            if (radio) radio.checked = true;
            this.querySelectorAll('[data-connection-panel]').forEach((panel) => {
                panel.hidden = panel.dataset.connectionPanel !== selectedMode;
            });
        }
    }

    if (!customElements.get('espectre-connection-picker')) {
        customElements.define('espectre-connection-picker', ConnectionPickerElement);
    }

    const conn = {
        mode: null,             // 'direct' | 'demo'
        status: 'disconnected', // disconnected | connecting | connected
        movement: 0,
        threshold: 0.5,
        motion: false,
        evaluationIntervalMs: 0,
        publishIntervalMs: 0,
        csiTargetPps: 0,
        csiTrafficMode: '',
        deviceName: '',
        deviceId: '',
        generatedName: '',
        deviceLabel: '',
        deviceConfigSupported: false,
        frontend: '',
        chip: '',
        firmwareVersion: '',
        endpoint: '',
        deviceBannerSub: '—',
        connectedAt: 0,
        startedAt: 0,
        toolName: '',
        entryPoint: '',
        readyState: '',
        readyAt: 0,
        readyTracked: false
    };

    let directClient = null;
    let directDiscoveryClient = null;
    let directDiscoveryGeneration = 0;
    let directReconnectTimer = 0;
    let directReconnectAttempt = 0;
    let configClearReturnFocus = null;
    let configClearResolve = null;
    let demoTimer = null;
    let demoInputEnergy = 0;
    const demoPointer = { x: null, y: null, t: 0 };
    let route = 'home';
    const LIVE_EXPERIENCE_ROUTES = new Set(['tool-game', 'tool-theremin']);
    let pendingLiveDestination = '';
    const deviceNameEditorState = {
        configure: { editing: false, savePending: false },
        monitor: { editing: false, savePending: false }
    };
    let lastTrackedProfile = null;
    let currentWifiBssid = '';
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
            const capability = element.dataset.capability;
            if (!Object.prototype.hasOwnProperty.call(snapshot, capability)) return;
            element.hidden = !sysinfoBoolean(snapshot[capability]);
        });
        $$('[data-capability-unavailable]').forEach((element) => {
            const capability = element.dataset.capabilityUnavailable;
            if (!Object.prototype.hasOwnProperty.call(snapshot, capability)) return;
            element.hidden = sysinfoBoolean(snapshot[capability]);
        });
        $$('[data-capability-any]').forEach((element) => {
            const capabilities = element.dataset.capabilityAny.split(/\s+/).filter(Boolean);
            if (!capabilities.some((key) => Object.prototype.hasOwnProperty.call(snapshot, key))) return;
            element.hidden = !capabilities.some((key) => sysinfoBoolean(snapshot[key]));
        });
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
        if (route === 'tool-monitor' || route === 'tool-configure') {
            setDeviceView('live', { focus: true });
        }
    }

    function rememberConnectionOrigin() {
        const origin = connectionIntentRoute();
        conn.toolName = toolNameForRoute(origin);
        conn.entryPoint = toolNameForRoute(origin);
        conn.startedAt = Date.now();
        conn.readyState = '';
        conn.readyAt = 0;
        conn.readyTracked = false;
    }

    function connectionParams() {
        return {
            tool_name: conn.toolName || activeToolName(),
            entry_point: conn.entryPoint || activeToolName()
        };
    }

    function connectionTransport() {
        if (conn.mode === 'direct') return 'direct_http';
        if (conn.mode === 'demo') return 'simulation';
        return 'direct_http';
    }

    function connectionInputMode() {
        if (conn.mode === 'demo') return 'demo';
        return 'direct';
    }

    function hasLiveDetection() {
        return conn.status === 'connected' && ['demo', 'direct'].includes(conn.mode);
    }

    function setDeviceView(view, { focus = false } = {}) {
        const targetRoute = view === 'connectivity' ? 'tool-configure' : 'tool-monitor';
        activeDeviceView = targetRoute === 'tool-configure' ? 'connectivity' : 'live';
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
        if (targetRoute === 'tool-monitor') {
            monitorResizeChart();
        }
        syncDiagnosticsPolling();
    }

    function applyCsiTrafficModeSelect(value) {
        const select = document.getElementById('sense-csi-mode');
        if (!select || !value) {
            return;
        }
        const normalized = value;
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
            syncSensingControls();
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
            conn.csiTrafficMode = snapshot.csi_traffic_mode;
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
        $$('[data-device-command]').forEach((panel) => {
            const supported = conn.mode === 'demo'
                || !monitor.commandCatalogReady
                || monitor.commands.has(panel.dataset.deviceCommand);
            const lightweightOnly = panel.dataset.deviceCommand === 'recalibrate' && detector !== 'lightweight';
            panel.hidden = !supported || lightweightOnly;
            panel.querySelectorAll('button, input, select').forEach((control) => {
                const calibrating = panel.dataset.deviceCommand === 'recalibrate' && monitor.calibrating;
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
        if (data.frontend) conn.frontend = String(data.frontend);
        if (data.device_name || data.name) conn.generatedName = data.device_name || data.name;
        if (data.device_label !== undefined) conn.deviceLabel = data.device_label;
        if (data.supports_device_config !== undefined) {
            conn.deviceConfigSupported = sysinfoBoolean(data.supports_device_config);
        }
        if (data.device_label || data.device_name || data.name) {
            conn.deviceName = data.device_label || data.device_name || data.name;
        }
        if (data.chip) conn.chip = String(data.chip).toUpperCase();
        if (data.firmware_version || data.firmware || data.version) {
            conn.firmwareVersion = data.firmware_version || data.firmware || data.version;
        }
    }

    function renderDeviceIdentity(identity = conn) {
        const write = (selector, value) => {
            $$(selector).forEach((el) => { el.textContent = value || '—'; });
        };
        write('.js-menu-frontend', formatFrontendLabel(identity.frontend));
        write('.js-menu-chip', identity.chip);
        write('.js-menu-device-id', identity.deviceId);
        write('.js-menu-firmware', identity.firmwareVersion);
    }

    function formatFrontendLabel(frontend) {
        const value = String(frontend || '');
        const labels = { native: 'Native', esphome: 'ESPHome', matter: 'Matter' };
        return labels[value.toLowerCase()] || value;
    }

    function formatDeviceIdentityLine(frontend, chip, deviceId, firmware) {
        const parts = [];
        if (frontend) parts.push('Frontend ' + formatFrontendLabel(frontend));
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
        const canEdit = conn.status === 'connected'
            && (conn.mode === 'direct' || conn.mode === 'demo')
            && conn.deviceConfigSupported;
        display.textContent = displayName;
        trigger.disabled = !canEdit || state.savePending;
        trigger.setAttribute('aria-label', canEdit
            ? (conn.deviceLabel ? 'Edit device name' : 'Set device name')
            : 'Device name (read-only)');
        trigger.title = canEdit ? 'Click to edit the device name' : 'Managed by the connected firmware';
        trigger.hidden = state.editing;
        input.hidden = !state.editing;
        input.disabled = state.savePending;
        if (!state.editing) input.value = conn.deviceLabel || '';
        editor.setAttribute('aria-busy', String(state.savePending));
        if (surface === 'configure') {
            const identity = $('.js-configure-device-banner-sub');
            if (identity) {
                identity.textContent = formatDeviceIdentityLine(
                    conn.frontend,
                    conn.chip,
                    conn.deviceId,
                    conn.firmwareVersion
                ) || '—';
            }
        }
    }

    function renderConfigureDeviceNameEditor() {
        renderDeviceNameEditor('configure');
    }

    function renderConfigureAvailability() {
        const esphome = conn.frontend.toLowerCase() === 'esphome';
        const matter = conn.frontend.toLowerCase() === 'matter';
        const nameNote = $('.js-configure-name-unavailable');
        const wifiNote = $('.js-wifi-unavailable');
        const mqttNote = $('.js-mqtt-unavailable');
        if (nameNote) {
            nameNote.hidden = conn.status !== 'connected' || conn.mode === 'demo'
                || conn.deviceConfigSupported;
            nameNote.textContent = 'This firmware exposes its device name as read-only in Direct Configure.';
        }
        if (wifiNote) {
            wifiNote.textContent = esphome || matter
                ? 'This firmware exposes Wi-Fi status and access-point pinning, but its owner manages network credentials.'
                : 'This firmware does not expose Wi-Fi details through Direct Configure.';
        }
        if (mqttNote) {
            mqttNote.textContent = esphome
                ? 'This ESPHome firmware does not expose MQTT configuration through Direct Configure. Use the native ESPHome API, or configure MQTT in the adopted YAML.'
                : 'This firmware does not expose MQTT configuration through Direct Configure.';
        }
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
            data.frontend || conn.frontend,
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
        conn.deviceId = next;
        const switched = previous && previous !== next;
        if (!switched) return;
        monitor.handoffReady = false;
        resetSensingCadence();
        resetMonitorLiveView();
        resetOtaChannelSelection();
        otaCheckTransport = '';
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
        if (route === 'tool-game') {
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

    /* --------------------------------------------------------- Direct mode */

    function directEndpointInput() {
        return document.querySelector(`[data-page="${route}"] .js-direct-endpoint`)
            || document.querySelector('.js-direct-endpoint');
    }

    function syncDirectEndpointInputs(target) {
        $$('.js-direct-endpoint').forEach((input) => {
            if (input && target) input.value = target;
        });
    }

    function privateIpv4Address(value) {
        const octets = value.split('.');
        if (octets.length !== 4 || octets.some((octet) => !/^(0|[1-9][0-9]{0,2})$/.test(octet))) {
            return false;
        }
        const values = octets.map(Number);
        if (values.some((octet) => octet > 255)) return false;
        return values[0] === 10
            || (values[0] === 172 && values[1] >= 16 && values[1] <= 31)
            || (values[0] === 192 && values[1] === 168);
    }

    function directTargetForEndpoint(endpoint) {
        const url = new URL(DirectProtocolClient.normalizeEndpoint(endpoint));
        const canonical = url.hostname.toLowerCase().match(DIRECT_CANONICAL_HOSTNAME);
        if (canonical) return canonical[1];
        if (privateIpv4Address(url.hostname)) return url.hostname;
        if (url.hostname.includes(':')) return url.hostname;
        return url.hostname;
    }

    function parseDirectTarget(value) {
        const input = String(value || '').trim().toLowerCase();
        if (!input) throw new Error('Enter a private IP address or device ID.');
        if (DIRECT_FULL_DEVICE_ID.test(input)) {
            return {
                display: input,
                endpoint: DirectProtocolClient.normalizeEndpoint(`espectre-${input}.local`),
                deviceId: input,
                shortId: '',
                discoveryFallback: true
            };
        }
        if (DIRECT_SHORT_DEVICE_ID.test(input)) {
            return { display: input, endpoint: '', deviceId: '', search: input, shortId: input };
        }
        if (privateIpv4Address(input)) {
            return {
                display: input,
                endpoint: DirectProtocolClient.normalizeEndpoint(input),
                deviceId: '',
                shortId: ''
            };
        }
        try {
            const endpoint = DirectProtocolClient.normalizeEndpoint(input);
            return {
                display: directTargetForEndpoint(endpoint), endpoint, deviceId: '', shortId: ''
            };
        } catch (_error) {
            if (/^[a-z][a-z0-9+.-]*:\/\//i.test(input)
                || /^\d{1,3}(?:\.\d{1,3}){3}$/.test(input)) {
                throw new Error('Use a private device IP address on the current LAN.');
            }
            if (input.length <= 63 && [...input].every((character) => character >= ' ')) {
                return { display: input, endpoint: '', deviceId: '', search: input, shortId: '' };
            }
            throw new Error('Enter a private IP address, device ID, or device name.');
        }
    }

    function storedDirectEndpoints() {
        try {
            const parsed = JSON.parse(localStorage.getItem(DIRECT_ENDPOINT_STORAGE_KEY) || '[]');
            if (!Array.isArray(parsed)) return [];
            return parsed.filter((value) => typeof value === 'string').flatMap((value) => {
                try { return [DirectProtocolClient.normalizeEndpoint(value)]; } catch (_error) { return []; }
            }).slice(0, DIRECT_ENDPOINT_LIMIT);
        } catch (_error) {
            return [];
        }
    }

    function writeStoredDirectEndpoints(endpoints) {
        try {
            localStorage.setItem(DIRECT_ENDPOINT_STORAGE_KEY, JSON.stringify(endpoints.slice(0, DIRECT_ENDPOINT_LIMIT)));
        } catch (_error) {
            // Private browsing and locked-down storage must not block Direct mode.
        }
        renderStoredDirectEndpoints();
    }

    function renderStoredDirectEndpoints() {
        const list = document.getElementById('direct-remembered-endpoints');
        if (!list) return;
        list.replaceChildren(...storedDirectEndpoints().map((endpoint) => {
            const option = document.createElement('option');
            option.value = directTargetForEndpoint(endpoint);
            return option;
        }));
    }

    function rememberDirectEndpoint(endpoint) {
        const endpoints = storedDirectEndpoints().filter((value) => value !== endpoint);
        writeStoredDirectEndpoints([endpoint, ...endpoints]);
    }

    function consumeDirectHandoff() {
        const params = new URLSearchParams(location.search);
        const directTarget = params.get('target') || '';
        if (!directTarget) return;
        try {
            const target = parseDirectTarget(directTarget);
            syncDirectEndpointInputs(target.display);
            const radio = document.getElementById('monitor-transport-direct');
            if (radio) radio.checked = true;
            history.replaceState(null, '', location.pathname + location.hash);
        } catch (_error) {
            history.replaceState(null, '', location.pathname + location.hash);
            toast('The device handoff contained an invalid local device.');
        }
    }

    function directCapabilitiesSnapshot(capabilities) {
        const methods = new Set((capabilities.commands || []).map((item) => item?.name).filter(Boolean));
        const sections = new Set(capabilities.config_sections || []);
        monitor.commands = methods;
        monitor.commandCatalogReady = true;
        return {
            supports_wifi_status: sections.has('wifi'),
            supports_wifi_bssid: methods.has('set_wifi_bssid')
                && methods.has('clear_wifi_bssid')
                && methods.has('scan_wifi_access_points') && methods.has('wifi_access_points'),
            supports_wifi_clear: methods.has('clear_wifi_config'),
            supports_mqtt_config: methods.has('set_mqtt_config'),
            supports_device_config: methods.has('set_device_label'),
            supports_runtime_threshold: methods.has('set_threshold'),
            supports_runtime_motion_hits: methods.has('set_motion_hits'),
            supports_runtime_detector: methods.has('set_detector'),
            supports_manual_recalibration: methods.has('recalibrate'),
            supports_traffic_control: methods.has('set_csi_traffic_mode')
                && methods.has('set_traffic_generator_mode'),
            supports_ota: methods.has('ota_status')
        };
    }

    function applyDirectConfig(config) {
        const device = config.device || {};
        const runtime = config.runtime || {};
        const wifi = config.wifi || {};
        const mqtt = config.mqtt || {};
        applySysinfo({
            device_label: config.device_label ?? device.device_label,
            wifi_configured: wifi.configured,
            wifi_connected: wifi.connected,
            wifi_ssid: wifi.ssid,
            wifi_band: wifi.band,
            wifi_channel: wifi.channel,
            wifi_rssi_dbm: wifi.rssi_dbm,
            wifi_bssid: wifi.bssid,
            wifi_apply_state: wifi.apply_state,
            wifi_apply_message: wifi.apply_message,
            mqtt_configured: mqtt.configured,
            mqtt_host: mqtt.host,
            mqtt_port: mqtt.port,
            mqtt_topic_prefix: mqtt.topic_prefix,
            mqtt_username_configured: mqtt.username_configured
        });
        applySensingSnapshot(runtime);
    }

    function applyRuntimeStatus(status) {
        if (!status || status.calibrating === undefined) return;
        const calibrating = sysinfoBoolean(status.calibrating);
        setCalibrationBusy(calibrating);
        if (calibrating) scheduleCalibrationIdle(MONITOR_CALIBRATION_SAFETY_MS);
    }

    function ingestDirectEvent(name, data) {
        if (name === 'telemetry') {
            applySensingCadence(data);
            applyLiveTelemetry(
                Number(data.movement_score ?? data.movement ?? 0),
                Number(data.threshold ?? conn.threshold),
                data.motion_state ?? data.motion
            );
            monitorFeed(conn.movement, conn.threshold, data.motion_state ?? data.motion);
            monitorResizeChart();
            return;
        }
        if (name === 'info' || name === 'status' || name === 'capabilities') {
            applySysinfo(data);
            if (name === 'status') applyRuntimeStatus(data);
            return;
        }
        if (name === 'config') {
            applyDirectConfig(data);
            return;
        }
        if (name === 'diagnostics') {
            markMonitorReady('diagnostics');
            monitorStats(data);
            return;
        }
        if (name === 'ota_status') applyOtaStatus(data);
        if (name === 'fault') toast(data.message || 'The device reported a runtime fault.');
    }

    function makeDirectClient(endpoint) {
        const client = new DirectProtocolClient(endpoint);
        client.on('event', ingestDirectEvent);
        client.on('protocol-error', (error) => console.warn('Ignored invalid Direct frame:', error.message));
        client.on('close', ({ expected }) => {
            if (expected || conn.mode !== 'direct') return;
            scheduleDirectReconnect(client);
        });
        return client;
    }

    function directPageOriginKind() {
        return sitePolicy.directOriginKind(location);
    }

    function directBrowserGuidance() {
        if (location.protocol !== 'https:') {
            return 'Local development mode: the firmware must explicitly allow HTTP loopback Origins. Development builds accept localhost on any port.';
        }
        if (browserSupport.hostedDirect === 'targeted'
            && ['windows', 'linux'].includes(browserSupport.operatingSystem)) {
            const platform = browserSupport.operatingSystem === 'windows' ? 'Windows' : 'Linux';
            return `Direct is supported on ${platform} desktop Chrome. Automatic discovery depends on the operating system's mDNS configuration; if it fails, connect using the device's current private IP address.`;
        }
        if (browserSupport.hostedDirect === 'targeted') return '';
        return 'Direct compatibility is not guaranteed in this browser. Use Chrome 151 or later on macOS, Windows, or native Linux, and connect by private IP where available.';
    }

    function renderDirectBrowserGuidance() {
        $$('.js-direct-browser-note').forEach((note) => {
            const message = directBrowserGuidance();
            note.textContent = message;
            note.hidden = !message;
        });
    }

    function directConnectionErrorMessage(error, endpoint, permissionState = 'unavailable') {
        const code = error?.code || 'connection_failed';
        let url;
        try { url = new URL(endpoint); } catch (_error) { url = null; }
        const localName = Boolean(url?.hostname.endsWith('.local'));
        const hostedCleartext = location.protocol === 'https:' && url?.protocol === 'http:';
        if (code === 'local_network_denied' || permissionState === 'denied') {
            return 'Local network access is blocked for this site. Open the browser site settings, allow Local network access, and retry. If it is already allowed there, also allow Chrome in the operating system Local Network privacy settings.';
        }
        if (code === 'timeout') {
            return localName
                ? 'The device did not answer in time. Confirm it is online and on this LAN, then retry Auto-discovery or enter its current IP address.'
                : 'The device did not answer in time. Confirm it is online, on this LAN, and that the IP address is current.';
        }
        if (code === 'subprotocol_mismatch' || code === 'unsupported_version'
            || code === 'invalid_capabilities' || code === 'invalid_envelope') {
            return 'The device answered with an incompatible Direct protocol. Confirm that the portal and Native firmware belong to the same supported release.';
        }
        if (code === 'connection_failed' || code === 'closed') {
            if (directPageOriginKind() === 'other') {
                return 'The device may have rejected this page Origin. Use https://espectre.dev, https://test.espectre.dev, or a loopback development portal explicitly enabled in the firmware.';
            }
            if (directPageOriginKind() === 'loopback') {
                return 'A local HTTP portal does not require a Local network access prompt. Confirm that this is a development firmware with loopback Origins enabled, reflash if it predates any-port localhost support, close other ESPectre tabs, and retry.';
            }
            if (hostedCleartext && browserSupport.hostedDirect === 'unsupported') {
                return 'This browser blocks a hosted HTTPS page from opening the device cleartext HTTP. Open this portal in supported desktop Chrome.';
            }
            if (hostedCleartext && permissionState === 'prompt') {
                return 'The browser is waiting for Local network access. Retry, allow the permission prompt for this site, and keep the device on the same LAN.';
            }
            const addressHelp = localName
                ? 'Retry Auto-discovery or enter the device IP address. '
                : 'Confirm the device IP address. ';
            return `The browser could not open the local Direct connection. ${addressHelp}Close other ESPectre tabs, allow Local network access when prompted, and retry. The device may be offline or at its two-client limit.`;
        }
        return error?.message || 'Direct HTTP connection failed.';
    }

    function setDirectConnectionHelp(message = '') {
        const help = directEndpointInput()?.closest('.device-connect-card')?.querySelector('.js-direct-help');
        if (!help) return;
        const copy = help.querySelector('.js-direct-help-copy');
        if (copy) copy.textContent = message;
        help.hidden = !message;
    }

    function setDirectConnectionStatus(message = '') {
        const status = directEndpointInput()?.closest('.device-connect-card')?.querySelector('.js-direct-status');
        if (!status) return;
        status.textContent = message;
        status.hidden = !message;
    }

    function directDiscoveryPanel(button) {
        return button?.closest('.device-connect-card')?.querySelector('.js-direct-discovery') || null;
    }

    function cancelDirectDiscovery({ clear = false } = {}) {
        directDiscoveryGeneration += 1;
        const client = directDiscoveryClient;
        directDiscoveryClient = null;
        client?.close();
        $$('.js-direct-discover').forEach((button) => {
            button.disabled = false;
            button.setAttribute('aria-disabled', 'false');
        });
        if (clear) {
            $$('.js-direct-discovery').forEach((panel) => {
                panel.hidden = true;
                panel.replaceChildren();
            });
        }
    }

    function discoveredPeerChipLabel(chip) {
        return String(chip || '').toUpperCase().replace(/^ESP32([A-Z]\d)$/, 'ESP32-$1');
    }

    function createDiscoveryDeviceButton({
        deviceId,
        displayDeviceId = deviceId,
        displayName,
        frontend,
        chip,
        endpoint = '',
        className = '',
        actionText = 'Connect →',
        ariaLabel
    }) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `btn-ghost direct-discovery-device ${className}`.trim();
        button.dataset.deviceId = deviceId;
        if (endpoint) button.dataset.endpoint = endpoint;
        button.setAttribute('aria-label', ariaLabel);
        const heading = document.createElement('span');
        heading.className = 'direct-discovery-device-heading';
        const name = document.createElement('strong');
        name.className = 'direct-discovery-device-name';
        name.textContent = displayName;
        const action = document.createElement('span');
        action.className = 'direct-discovery-device-action';
        action.setAttribute('aria-hidden', 'true');
        action.textContent = actionText;
        heading.append(name, action);
        const metadata = document.createElement('span');
        metadata.className = 'direct-discovery-device-meta';
        for (const [label, value, valueClass = ''] of [
            ['Frontend', frontend || 'Unknown'],
            ['Hardware', chip || 'Unknown'],
            ['Device ID', displayDeviceId, 'mono']
        ]) {
            const field = document.createElement('span');
            field.className = 'direct-discovery-device-field';
            const fieldLabel = document.createElement('span');
            fieldLabel.className = 'direct-discovery-device-label';
            fieldLabel.textContent = label;
            const fieldValue = document.createElement('span');
            fieldValue.className = `direct-discovery-device-value ${valueClass}`.trim();
            fieldValue.textContent = value;
            field.append(fieldLabel, fieldValue);
            metadata.appendChild(field);
        }
        button.append(heading, metadata);
        return button;
    }

    function renderDiscoveredPeers(panel, result, { selectionRequired = false } = {}) {
        panel.replaceChildren();
        const summary = document.createElement('p');
        summary.className = 'direct-discovery-summary';
        summary.textContent = selectionRequired
            ? `${result.devices.length} matching devices found. Select one to connect.`
            : result.devices.length
            ? `${result.devices.length} local device${result.devices.length === 1 ? '' : 's'} found${result.truncated ? ' (partial result)' : ''}.`
            : 'No compatible devices answered. You can still enter a private IP address or device ID.';
        panel.appendChild(summary);
        if (!result.devices.length) return;
        const list = document.createElement('ul');
        list.className = 'direct-discovery-list';
        for (const peer of result.devices) {
            const item = document.createElement('li');
            const displayName = peer.name || `ESPectre ${peer.device_id.slice(-6)}`;
            const frontend = formatFrontendLabel(peer.frontend);
            const chip = discoveredPeerChipLabel(peer.chip);
            const shortId = peer.device_id.slice(-6);
            const button = createDiscoveryDeviceButton({
                deviceId: peer.device_id,
                displayDeviceId: shortId,
                displayName,
                frontend,
                chip,
                endpoint: peer.endpoints[0],
                ariaLabel: `Connect to ${displayName}, ${frontend}, ${chip}, device ID ${shortId}`
            });
            item.appendChild(button);
            list.appendChild(item);
        }
        panel.appendChild(list);
    }

    function directDiscoveryFailureMessage(error, permissionState) {
        if (error?.code === 'local_network_denied' || permissionState === 'denied') {
            return 'Local network access is blocked for this site. Allow it in the browser site settings, then retry. If it is already allowed there, also allow Chrome in the operating system Local Network privacy settings.';
        }
        if (error?.code === 'unsupported_crypto') {
            return 'Auto-discovery requires Web Crypto, which is unavailable in this browser. Enter the device\'s current private IP address.';
        }
        if (error?.code === 'unsupported_capability') {
            return 'The responder does not support Auto-discovery. Enter the device\'s current private IP address.';
        }
        if (error?.code === 'timeout') {
            return 'Auto-discovery timed out. mDNS may be disabled, or multicast may be filtered or isolated. Enter the current private IP address from the Improv link or router lease table.';
        }
        if (error?.code === 'invalid_envelope' || error?.code === 'unsupported_version') {
            return 'Auto-discovery reached an incompatible responder. Update that device, or enter the target device\'s current private IP address.';
        }
        if (error?.code === 'invalid_peer_result' || error?.code === 'frame_too_large') {
            return 'The responder returned an invalid discovery result, so no device was used. Enter the target device\'s trusted private IP address.';
        }
        if (error?.code === 'connection_failed') {
            return 'Auto-discovery could not reach a responder. Multicast may be isolated, or mDNS may be disabled. Enter the current private IP address from the Improv link or router lease table.';
        }
        return 'Auto-discovery is unavailable on this network. Enter the device\'s current private IP address; device IDs and names still depend on mDNS.';
    }

    async function queryLocalPeers(onProgress = () => {}) {
        const client = makeDirectClient(DirectProtocolClient.createDiscoveryEndpoint());
        directDiscoveryClient = client;
        try {
            onProgress('Looking for compatible ESPectre devices…');
            try {
                return await client.discoverPeersBootstrap();
            } catch (error) {
                error.discoveryStage = 'query';
                throw error;
            }
        } finally {
            if (directDiscoveryClient === client) directDiscoveryClient = null;
            client.close();
        }
    }

    async function discoverLocalPeers(button) {
        cancelDirectDiscovery({ clear: true });
        const panel = directDiscoveryPanel(button);
        if (!panel) return;
        const generation = directDiscoveryGeneration;
        panel.hidden = false;
        panel.textContent = 'Starting Auto-discovery…';
        button.disabled = true;
        button.setAttribute('aria-disabled', 'true');
        track('local_discovery', { tool_name: activeToolName(), result: 'attempt' });
        try {
            const result = await queryLocalPeers((message) => {
                if (generation === directDiscoveryGeneration) panel.textContent = message;
            });
            if (generation !== directDiscoveryGeneration) return;
            setDirectConnectionHelp();
            renderDiscoveredPeers(panel, result);
            track('local_discovery', {
                tool_name: activeToolName(), result: result.devices.length ? 'success' : 'empty',
                device_count: result.devices.length, truncated: result.truncated
            });
        } catch (error) {
            if (generation !== directDiscoveryGeneration) return;
            const permissionState = await localNetworkAccessState();
            if (generation !== directDiscoveryGeneration) return;
            panel.textContent = directDiscoveryFailureMessage(error, permissionState);
            track('local_discovery', {
                tool_name: activeToolName(), result: 'failure', error_type: errorType(error)
            });
        } finally {
            if (generation === directDiscoveryGeneration) {
                button.disabled = false;
                button.setAttribute('aria-disabled', 'false');
            }
        }
    }

    async function localNetworkAccessState() {
        const detectState = window.ESPectreBrowserSupport.localNetworkAccessState;
        return typeof detectState === 'function'
            ? detectState(navigator) : 'unavailable';
    }

    function cancelDirectReconnect() {
        clearTimeout(directReconnectTimer);
        directReconnectTimer = 0;
        directReconnectAttempt = 0;
    }

    function scheduleDirectReconnect(client) {
        if (directClient !== client || conn.mode !== 'direct' || directReconnectTimer) return;
        if (directReconnectAttempt >= DIRECT_RECONNECT_DELAYS_MS.length) {
            directClient = null;
            teardownConnection('reconnect_failed');
            toast('Direct HTTP disconnected. Enter the device address to reconnect.');
            return;
        }
        const delay = DIRECT_RECONNECT_DELAYS_MS[directReconnectAttempt++];
        setStatus('connecting');
        directReconnectTimer = setTimeout(async () => {
            directReconnectTimer = 0;
            if (directClient !== client || conn.mode !== 'direct') return;
            try {
                await client.connect({ timeoutMs: 5000 });
                await client.handshake({ timeoutMs: 5000 });
                if (directClient !== client || conn.mode !== 'direct') {
                    client.close();
                    return;
                }
                await refreshDirectDevice();
                directReconnectAttempt = 0;
                setStatus('connected');
                toast('Direct HTTP reconnected.');
                if (pendingConfigVerification) requestConfigVerification();
            } catch (error) {
                client.close();
                const permissionState = await localNetworkAccessState();
                if (error?.code === 'local_network_denied' || permissionState === 'denied') {
                    directClient = null;
                    teardownConnection('local_network_denied');
                    const message = directConnectionErrorMessage(error, client.endpoint, permissionState);
                    setDirectConnectionHelp(message);
                    toast(message);
                    return;
                }
                scheduleDirectReconnect(client);
            }
        }, delay);
    }

    async function refreshDirectDevice() {
        if (!directClient?.connected) return;
        const supportsOta = directClient.capabilities?.commands?.some((item) => item.name === 'ota_status');
        // Keep local-network requests serial. Chrome may still be resolving its
        // Local Network Access grant when the Direct stream and handshake have
        // just completed, and rejects a concurrent fan-out before CORS runs.
        const info = await directClient.request('info');
        const status = await directClient.request('status');
        const config = await directClient.request('config');
        const diagnostics = activeToolName() === 'configure'
            ? await directClient.request('diagnostics')
            : null;
        const otaStatus = supportsOta ? await directClient.request('ota_status') : null;
        applySysinfo({
            ...directCapabilitiesSnapshot(directClient.capabilities),
            ...info,
            ...status,
            ...(diagnostics || {})
        });
        applyDirectConfig(config);
        if (otaStatus) applyOtaStatus(otaStatus);
    }

    async function resolveDiscoveredTarget(target, input) {
        const description = target.deviceId
            ? `device ID ${target.deviceId}`
            : target.shortId ? `device ID …${target.shortId}` : `device name “${target.search}”`;
        setDirectConnectionHelp();
        setDirectConnectionStatus(`Looking for ${description} on this LAN.`);
        track('local_discovery', { tool_name: activeToolName(), result: 'attempt' });
        let result;
        try {
            result = await queryLocalPeers();
        } catch (error) {
            const permissionState = await localNetworkAccessState();
            throw new Error(directDiscoveryFailureMessage(error, permissionState));
        } finally {
            setDirectConnectionStatus();
        }
        const query = target.search.toLowerCase();
        const matches = result.devices.filter((peer) => target.deviceId
            ? peer.device_id === target.deviceId
            : target.shortId ? peer.device_id.endsWith(target.shortId)
                : [peer.name, peer.instance].some((value) => String(value || '').toLowerCase().includes(query)));
        track('local_discovery', {
            tool_name: activeToolName(), result: matches.length === 1 ? 'success' : (matches.length ? 'multiple' : 'empty'),
            device_count: result.devices.length, truncated: result.truncated
        });
        if (matches.length === 0) {
            throw new Error(`No matching ${description} was found. Retry Auto-discovery, or use the device's current private IP address.`);
        }
        if (matches.length > 1) {
            const panel = directDiscoveryPanel(input);
            if (panel) {
                panel.hidden = false;
                renderDiscoveredPeers(panel, { ...result, devices: matches }, { selectionRequired: true });
            }
            setDirectConnectionHelp();
            return null;
        }
        const peer = matches[0];
        return { display: peer.device_id, endpoint: peer.endpoints[0], deviceId: peer.device_id };
    }

    async function connectDirect({ endpoint, deviceId, openView } = {}) {
        cancelDirectDiscovery();
        if (directClient || conn.status !== 'disconnected') return;
        const input = directEndpointInput();
        let target;
        try {
            if (endpoint) {
                const normalizedEndpoint = DirectProtocolClient.normalizeEndpoint(endpoint);
                target = {
                    display: deviceId || directTargetForEndpoint(normalizedEndpoint),
                    endpoint: normalizedEndpoint,
                    deviceId: deviceId || ''
                };
            } else {
                target = parseDirectTarget(input?.value || '');
                if (target.search) target = await resolveDiscoveredTarget(target, input);
                if (!target) return;
            }
        } catch (error) {
            setDirectConnectionHelp(error.message);
            toast(error.message);
            input?.setAttribute('aria-invalid', 'true');
            return;
        }
        let normalizedEndpoint;
        try {
            normalizedEndpoint = DirectProtocolClient.normalizeEndpoint(target.endpoint);
        } catch (error) {
            toast(error.message);
            input?.setAttribute('aria-invalid', 'true');
            return;
        }
        input?.removeAttribute('aria-invalid');
        setDirectConnectionHelp();
        rememberConnectionOrigin();
        track('tool_connection', {
            ...connectionParams(), transport: 'direct_http', result: 'attempt'
        });
        setStatus('connecting');
        try {
            const client = makeDirectClient(normalizedEndpoint);
            directClient = client;
            cancelDirectReconnect();
            await client.connect();
            await client.handshake();
            if (directClient !== client) return;
            conn.mode = 'direct';
            conn.endpoint = normalizedEndpoint;
            conn.deviceBannerSub = normalizedEndpoint;
            conn.connectedAt = Date.now();
            syncDirectEndpointInputs(target.display);
            rememberDirectEndpoint(normalizedEndpoint);
            await refreshDirectDevice();
            if ((openView || (route === 'tool-monitor' ? 'live' : 'connectivity')) === 'live') {
                await client.request('set_sensing', { enabled: true });
            }
            setStatus('connected');
            setDirectConnectionHelp();
            if (route === 'tool-raw-csi') {
                rawCsiUseConnection();
            } else if (!LIVE_EXPERIENCE_ROUTES.has(route)) {
                const view = openView || (route === 'tool-monitor' ? 'live' : 'connectivity');
                setDeviceView(view);
                if (view === 'connectivity' && monitor.commands.has('scan_wifi_access_points')) {
                    void cfgRefreshWifiAccessPoints();
                }
            }
            track('tool_connection', {
                ...connectionParams(), transport: 'direct_http', result: 'success'
            });
            markToolReady('info');
            if (pendingLiveDestination) completeLiveConnectionNavigation();
        } catch (error) {
            directClient?.close();
            directClient = null;
            setStatus('disconnected');
            if (target.discoveryFallback) {
                try {
                    const discoveredTarget = await resolveDiscoveredTarget({
                        deviceId: target.deviceId,
                        search: target.deviceId,
                        shortId: ''
                    }, input);
                    if (discoveredTarget) {
                        return connectDirect({
                            endpoint: discoveredTarget.endpoint,
                            deviceId: discoveredTarget.deviceId,
                            openView
                        });
                    }
                    return;
                } catch (fallbackError) {
                    error = fallbackError;
                }
            }
            track('tool_connection', {
                ...connectionParams(), transport: 'direct_http', result: 'failure',
                error_type: errorType(error)
            });
            const message = error?.code
                ? directConnectionErrorMessage(error, normalizedEndpoint, await localNetworkAccessState())
                : error.message;
            setDirectConnectionHelp(message);
            toast(message);
        }
    }

    function applyConfigureMqttCredentialPolicy(presetName) {
        const username = document.getElementById('cfg-mqtt-user');
        const password = document.getElementById('cfg-mqtt-pass');
        const isFlespi = presetName === 'flespi';
        username.placeholder = isFlespi ? 'Enter a new token' : 'Keep saved username';
        password.placeholder = isFlespi ? 'Flespi does not use one' : 'Keep saved password';
        password.toggleAttribute('data-preset-locked', isFlespi);
        password.title = isFlespi ? 'Flespi does not use a password' : '';
        password.disabled = isFlespi;
        if (isFlespi) password.value = '';
        syncConfigureMqttCredentialMode();
    }

    function syncConfigureMqttCredentialMode() {
        const clear = document.getElementById('cfg-mqtt-credentials-clear').checked;
        const username = document.getElementById('cfg-mqtt-user');
        const password = document.getElementById('cfg-mqtt-pass');
        const isFlespi = document.getElementById('cfg-mqtt-preset').value === 'flespi';
        username.disabled = clear;
        password.disabled = clear || isFlespi;
        username.placeholder = clear ? 'No username' : isFlespi ? 'Enter a new token' : 'Keep saved username';
        password.placeholder = clear ? 'No password' : isFlespi ? 'Flespi does not use one' : 'Keep saved password';
        if (clear) {
            username.value = '';
            password.value = '';
        }
    }

    function applyMqttPresetFieldLocks(_target, preset) {
        const fields = { host: 'cfg-mqtt-host', port: 'cfg-mqtt-port' };
        const locked = new Set(preset.locked || []);
        Object.entries(fields).forEach(([name, id]) => {
            const input = document.getElementById(id);
            if (!input) return;
            const isLocked = locked.has(name);
            input.readOnly = isLocked;
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
        applyConfigureMqttCredentialPolicy(resolvedName);
        if (clearCredentials) {
            document.getElementById('cfg-mqtt-user').value = '';
            document.getElementById('cfg-mqtt-pass').value = '';
            document.getElementById('cfg-mqtt-credentials-clear').checked = false;
        }
        syncConfigureMqttCredentialMode();
    }

    async function startDetection(preferredTransport = '') {
        rememberLiveDestination();
        if (conn.mode === 'demo') {
            completeLiveConnectionNavigation();
            return;
        }
        if (conn.mode === 'direct' && directClient?.connected) {
            try {
                await directClient.request('set_sensing', { enabled: true });
                setDeviceView('live');
                completeLiveConnectionNavigation();
                toast('Sensing is live over Direct HTTP.');
            } catch (error) {
                toast(error.message);
            }
            return;
        }
        if (preferredTransport === 'direct') {
            selectMonitorTransport('direct');
            location.hash = '#tool-monitor';
            return;
        }
        selectMonitorTransport('direct');
        location.hash = '#tool-monitor';
    }

    function applySysinfo(snapshot) {
        if (conn.mode === 'direct' && conn.toolName === 'configure'
                && (snapshot.frontend || snapshot.chip || snapshot.proto_version)) {
            markToolReady('info');
        }
        applyConfigureCapabilities(snapshot);
        const frontend = snapshot.frontend || conn.frontend;
        const chip = snapshot.chip ? String(snapshot.chip).toUpperCase() : conn.chip;
        const proto = snapshot.proto_version || snapshot.espectre_protocol_version || '';
        const firmware = snapshot.firmware_version || snapshot.firmware || snapshot.version || conn.firmwareVersion;
        const deviceIdentity = formatDeviceIdentityLine(
            frontend,
            chip,
            snapshot.device_id || conn.deviceId,
            firmware
        ) || '—';
        conn.chip = chip;
        conn.firmwareVersion = firmware;
        conn.deviceBannerSub = conn.mode === 'direct'
            ? deviceIdentity
            : [deviceIdentity, conn.endpoint].filter((value) => value && value !== '—').join(' · ') || '—';

        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== '') {
                if (el.tagName === 'INPUT' || el.tagName === 'SELECT') el.value = value;
                else el.textContent = value;
            }
        };
        const setConfigurationStatus = (surface, connectedValue, configuredValue) => {
            const status = $(`.js-${surface}-status`);
            const dot = $(`.js-${surface}-status-dot`);
            const text = $(`.js-${surface}-status-text`);
            if (!status || !dot || !text) return;
            let state;
            let label;
            if (configuredValue !== undefined && !sysinfoBoolean(configuredValue)) {
                state = 'not-configured';
                label = 'Not configured';
            } else if (connectedValue !== undefined) {
                const connected = sysinfoBoolean(connectedValue);
                state = connected ? 'connected' : 'disconnected';
                label = connected ? 'Connected' : 'Disconnected';
            } else if (configuredValue !== undefined && text.textContent === 'Checking…') {
                state = 'configured';
                label = 'Configured';
            } else {
                return;
            }
            status.dataset.state = state;
            text.textContent = label;
            dot.classList.toggle('dot-idle', state === 'not-configured' || state === 'configured');
            dot.classList.toggle('dot-ok', state === 'connected');
            dot.classList.toggle('dot-error', state === 'disconnected');
        };
        if (snapshot.wifi_ssid !== undefined) set('cfg-ssid', snapshot.wifi_ssid || '');
        if (snapshot.wifi_band !== undefined) {
            set('cfg-wifi-band', snapshot.wifi_band === '2g' ? '2.4 GHz'
                : snapshot.wifi_band === '5g' ? '5 GHz' : 'Unknown');
        }
        if (snapshot.wifi_channel !== undefined) {
            set('cfg-channel', Number(snapshot.wifi_channel) > 0 ? snapshot.wifi_channel : 'Unknown');
        }
        if (snapshot.wifi_bssid !== undefined) {
            currentWifiBssid = String(snapshot.wifi_bssid || '').toUpperCase();
            const bssid = document.getElementById('cfg-bssid');
            if (bssid && ![...bssid.options].some((option) => option.value === currentWifiBssid)
                    && currentWifiBssid) {
                bssid.add(new Option(`${currentWifiBssid} · pinned`, currentWifiBssid));
            }
            if (bssid) bssid.value = currentWifiBssid;
        }
        if (snapshot.mqtt_host) {
            set('cfg-mqtt-host', snapshot.mqtt_host);
            set('cfg-mqtt-port', snapshot.mqtt_port);
            const mqttPreset = configuredBrokerPreset(snapshot.mqtt_host, snapshot.mqtt_port);
            document.getElementById('cfg-mqtt-preset').value = mqttPreset;
            applyMqttPresetFieldLocks('configure', MQTT_PRESETS[mqttPreset].configure);
            applyConfigureMqttCredentialPolicy(mqttPreset);
            set('cfg-topic-prefix', snapshot.mqtt_topic_prefix || snapshot.topic_prefix || MQTT_FORM_DEFAULTS.topicPrefix);
            const mqttPass = document.getElementById('cfg-mqtt-pass');
            if (mqttPass) mqttPass.value = '';
        }
        applyDeviceIdentity(snapshot);
        if (snapshot.supports_ota !== undefined) otaSupported = sysinfoBoolean(snapshot.supports_ota);
        applySensingSnapshot(snapshot);
        syncFirmwareUpdateNotice();
        evaluateConfigVerification(snapshot);
        setConfigurationStatus('wifi', snapshot.wifi_connected, snapshot.wifi_configured);
        setConfigurationStatus('mqtt', snapshot.mqtt_connected, snapshot.mqtt_configured);

        // Real hardware only: demo values would pollute the adoption report.
        if (conn.mode === 'direct' && snapshot.frontend && snapshot.chip) {
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

    function connectDemo(openView = '') {
        if (conn.status !== 'disconnected') return;
        if (openView === 'live') rememberLiveDestination();
        rememberConnectionOrigin();
        track('tool_demo_start', connectionParams());
        setStatus('connecting');
        setTimeout(() => {
            conn.mode = 'demo';
            otaSupported = false;
            syncFirmwareUpdateNotice();
            conn.deviceName = 'Demo Device';
            conn.deviceBannerSub = '—';
            conn.threshold = 0.5;
            conn.movement = 0.04;
            conn.connectedAt = Date.now();
            setStatus('connected');
            markToolReady('telemetry');
            monitor.commands = new Set([
                'set_threshold', 'set_motion_hits', 'set_detector', 'recalibrate',
                'set_csi_traffic_mode', 'set_traffic_generator_mode', 'diagnostics',
                'set_device_label'
            ]);
            monitor.commandCatalogReady = true;
            applySysinfo({
                chip: 'esp32-c5',
                frontend: 'native',
                proto_version: '1.0',
                firmware_version: '3.0.0-dev',
                supports_wifi_bssid: 'true',
                supports_wifi_clear: 'true',
                supports_mqtt_config: 'true',
                supports_device_config: 'true',
                supports_extended_diagnostics: 'true',
                supports_ota: 'false',
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
                mqtt_connected: 'true',
                wifi_configured: 'true',
                wifi_ssid: 'HomeNet',
                wifi_band: '5g',
                wifi_channel: '10',
                wifi_bssid: '',
                mqtt_host: 'homeassistant.local',
                mqtt_port: '1883',
                mqtt_username: 'mqtt',
                topic_prefix: 'espectre/v1/devices',
                device_id: '3cf79180d3a0aca4',
                device_name: 'Demo Device',
                device_label: 'Demo Device',
                motion_hits: '4/3'
            });
            if (openView === 'live') completeLiveConnectionNavigation();
            monitorResetChart();
            let t = 0;
            const demoTickSec = evaluationIntervalMs() / 1000;
            demoTimer = setInterval(() => {
                t += demoTickSec;
                const gameDemoActive = route === 'tool-game' && game.phase !== 'idle' && game.phase !== 'done';
                const idle = 0.035 + Math.sin(t * 0.8) * 0.01 + Math.sin(t * 1.9) * 0.004;
                const gameManualFlight = route === 'tool-game' && game.manualFlight ? 1 : 0;
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
        cancelDirectDiscovery({ clear: true });
        cancelDirectReconnect();
        void rawCsiStop();
        const client = directClient;
        directClient = null;
        client?.close();
        teardownConnection('user');
    }

    function resetMonitorSession() {
        monitor.commands.clear();
        monitor.commandCatalogReady = false;
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
        renderConnection();
    }

    function teardownConnection(reason = 'route_change') {
        cancelDirectDiscovery({ clear: true });
        cancelDirectReconnect();
        void rawCsiStop();
        monitor.switchingTransport = false;
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
                transport: previousMode === 'demo' ? 'simulation' : 'direct_http',
                input_mode: previousMode === 'demo' ? 'demo' : 'direct',
                reason,
                duration_seconds: durationSeconds
            });
        }
        clearInterval(demoTimer);
        demoTimer = null;
        directClient?.close();
        directClient = null;
        demoInputEnergy = 0;
        demoPointer.x = null;
        demoPointer.y = null;
        demoPointer.t = 0;
        resetMonitorSession();
        conn.mode = null;
        conn.movement = 0;
        conn.motion = false;
        resetSensingCadence();
        conn.deviceName = '';
        conn.deviceId = '';
        conn.generatedName = '';
        conn.deviceLabel = '';
        conn.deviceConfigSupported = false;
        conn.frontend = '';
        conn.chip = '';
        conn.firmwareVersion = '';
        conn.endpoint = '';
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
        if (previousMode === 'demo') {
            $$('espectre-connection-picker').forEach((picker) => picker.select('direct'));
        }
        setStatus('disconnected');
    }

    /* ----------------------------------------------------------- rendering */

    let dropdownOpen = false;

    function flashUnsupportedMessage() {
        if (browserSupport.mobile) {
            return 'USB flashing is not available on mobile. Use desktop Chrome or Edge.';
        }
        return 'USB flashing is not available in this browser. Use desktop Chrome or Edge.';
    }

    function renderBrowserSupport() {
        const directConnecting = conn.status === 'connecting' && !!directClient && !directClient.connected;
        $$('.js-connect-direct').forEach((button) => {
            button.disabled = directConnecting;
            button.setAttribute('aria-disabled', String(button.disabled));
            const label = button.querySelector('.js-connect-label');
            if (label) {
                if (!label.dataset.supportedLabel) label.dataset.supportedLabel = label.textContent;
                label.textContent = directConnecting ? 'Connecting…' : label.dataset.supportedLabel;
            }
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
        const matterReadButton = $('.js-matter-read');
        if (matterReadButton) {
            matterReadButton.disabled = !browserSupport.flash;
            matterReadButton.setAttribute('aria-disabled', String(!browserSupport.flash));
            matterReadButton.title = browserSupport.flash ? '' : flashUnsupportedMessage();
        }
    }

    function renderConnection() {
        const connected = conn.status === 'connected';
        const usbConnected = Boolean(flash.usbDialog);
        const displayedConnected = usbConnected || connected;
        const displayedConnecting = !usbConnected && conn.status === 'connecting';
        const displayedMode = usbConnected ? 'usb' : conn.mode;
        const live = hasLiveDetection();
        const directSetup = connected && conn.mode === 'direct';

        $('.js-conn-disconnected').hidden = displayedConnected || displayedConnecting;
        $('.js-conn-connecting').hidden = !displayedConnecting;
        $('.js-conn-connected').hidden = !displayedConnected;
        $('.js-dropdown').hidden = !(displayedConnected && dropdownOpen);
        $('.js-dropdown-toggle').setAttribute('aria-expanded', String(displayedConnected && dropdownOpen));
        $('.js-demo-tag').hidden = displayedMode !== 'demo';
        const transportTag = $('.js-transport-tag');
        if (transportTag) {
            const transportLabels = { direct: 'HTTP', usb: 'USB' };
            transportTag.textContent = transportLabels[displayedMode] || '';
            transportTag.hidden = !displayedConnected || !transportLabels[displayedMode];
        }

        $$('.js-needs-conn').forEach((el) => { el.hidden = connected; });
        $$('.js-has-conn').forEach((el) => { el.hidden = !connected; });
        $$('.js-needs-live').forEach((el) => { el.hidden = live; });
        $$('.js-has-live').forEach((el) => { el.hidden = !live; });
        const showLiveEnergy = live;
        $$('.js-live-energy').forEach((el) => { el.hidden = !showLiveEnergy; });
        const configureOnboarding = $('.js-configure-onboarding');
        const configureWorkspace = $('.js-configure-workspace');
        const monitorOnboarding = $('.js-monitor-onboarding');
        const monitorWorkspace = $('.js-monitor-workspace');
        const connectivitySetup = $('.js-connectivity-setup');
        const edit = $('.js-device-edit-connectivity');
        const startSensing = document.querySelector('[data-page="tool-configure"] .js-start-detection');
        if (configureOnboarding) configureOnboarding.hidden = directSetup || conn.mode === 'demo';
        if (configureWorkspace) configureWorkspace.hidden = !(directSetup || conn.mode === 'demo');
        if (monitorOnboarding) monitorOnboarding.hidden = live;
        if (monitorWorkspace) monitorWorkspace.hidden = !live;
        rawCsiUseConnection();
        if (connectivitySetup) connectivitySetup.hidden = !(directSetup || conn.mode === 'demo');
        if (startSensing) startSensing.disabled = monitor.switchingTransport;
        if (edit) {
            edit.hidden = false;
            edit.disabled = false;
            edit.textContent = 'Edit connectivity';
        }

        $$('.js-device-name').forEach((el) => { el.textContent = conn.deviceName || 'ESPectre'; });
        const displayedIdentity = usbConnected ? flashUsbIdentity() : conn;
        $$('.js-connection-device-name').forEach((el) => {
            el.textContent = displayedIdentity.deviceName || 'ESPectre';
        });
        $$('.js-device-banner-sub').forEach((el) => { el.textContent = conn.deviceBannerSub; });
        renderConfigureDeviceNameEditor();
        renderMonitorDeviceNameEditor();
        renderConfigureAvailability();
        renderDeviceIdentity(displayedIdentity);
        const frontendRow = $('.js-menu-frontend-row');
        if (frontendRow) frontendRow.hidden = usbConnected;
        const deviceIdLabel = $('.js-menu-device-id-label');
        if (deviceIdLabel) deviceIdLabel.textContent = usbConnected ? 'USB VID:PID' : 'Device ID';
        const firmwareLabel = $('.js-menu-firmware-label');
        if (firmwareLabel) firmwareLabel.textContent = usbConnected ? 'Target firmware' : 'Firmware';
        const usbNote = $('.js-usb-port-note');
        if (usbNote) usbNote.hidden = !usbConnected;
        const disconnectButton = $('.js-disconnect');
        if (disconnectButton) disconnectButton.hidden = usbConnected;
        $$('.js-direct-chip').forEach((chip) => {
            chip.classList.toggle('ready', connected && conn.mode === 'direct');
            chip.textContent = connected && conn.mode === 'direct' ? 'HTTP · READY' : 'HTTP';
        });

        syncSensingControls();
        syncDiagnosticsPolling();
        renderBrowserSupport();
        renderTelemetry();
        syncDemoToast();
        if (live && route === 'tool-game' && !game.ctx) requestAnimationFrame(gameResizeCanvas);
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
        if (route !== 'tool-theremin') thereminStop();
        if (route === 'tool-monitor') monitorResizeChart();
        if (route === 'tool-raw-csi') rawCsiUseConnection();
        if (route === 'tool-game') {
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
        if (route === 'tool-flash') {
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
        if (window.trackRouteView && route !== 'tool-raw-csi') window.trackRouteView(route);
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
        cancelDirectDiscovery({ clear: true });
        const previousRoute = route;
        if (previousRoute === 'tool-raw-csi' && target !== 'tool-raw-csi') {
            void rawCsiStop();
        }
        if (pendingLiveDestination) {
            if (LIVE_EXPERIENCE_ROUTES.has(target)) pendingLiveDestination = target;
            else if (target !== 'tool-monitor' && target !== 'tool-configure') pendingLiveDestination = '';
        }
        if (previousRoute === 'tool-game' && target !== 'tool-game') {
            gameExitFullscreen();
            reportGameAbandon('route_change');
        }
        if (target === 'tool-game' && previousRoute !== 'tool-game') resetGameThreshold();
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
            if (route === 'tools') renderConnection();
            if (window.initPageTocs) window.initPageTocs(container);
            if (window.initSdkDownloadVersions) window.initSdkDownloadVersions(container);
            if (window.initPublishedReleaseTags) window.initPublishedReleaseTags(container);
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
        detectedChip: '', supportedChipLabels: [], modalReturnFocus: null,
        refreshRequest: 0, targetVersion: '', usbDialog: null, usbPortInfo: null,
        usbReleaseTimer: null
    };

    /*
     * Presentation order for the Flash selectors and installer manifest.
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

    function flashSetFrontendActions(frontendKey) {
        const matterReadButton = $('.js-matter-read');
        matterReadButton.hidden = frontendKey !== 'matter';
        if (frontendKey !== 'matter') flashHideMatterQr();
    }

    async function flashRefresh() {
        const frontendSel = document.getElementById('flash-frontend');
        const channelSel = document.getElementById('flash-channel');
        const summary = $('.js-flash-summary');
        const installButton = $('.js-flash-install');
        const requestId = ++flash.refreshRequest;
        const selectedChannel = channelSel.value;
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
                flashSetFrontendActions(frontendSel.value);
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
            flashSetFrontendActions(frontendSel.value);

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
            flashSetFrontendActions('');
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
        if (trigger) {
            trigger.disabled = true;
            trigger.setAttribute('aria-disabled', 'true');
        }
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
            if (trigger) {
                trigger.disabled = !browserSupport.flash;
                trigger.setAttribute('aria-disabled', String(!browserSupport.flash));
            }
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
        };
        const setDetectedChip = (chip) => {
            const normalized = String(chip || '').toUpperCase();
            if (!normalized || flash.detectedChip === normalized) return;
            flash.detectedChip = normalized;
            renderConnection();
        };
        const customizeNativeDeviceLink = () => {
            if (document.getElementById('flash-frontend').value !== 'native' || !dialog.shadowRoot) return;
            dialog.shadowRoot.querySelectorAll('[slot="headline"]').forEach((headline) => {
                const label = headline.textContent.trim();
                if (label !== 'Visit Device' && label !== 'Configure Device') return;
                if (label === 'Visit Device') headline.textContent = 'Configure Device';
                const item = headline.closest('ew-list-item');
                if (!item || !item.href) return;
                try {
                    const returnedUrl = new URL(item.href, location.href);
                    const configureUrl = new URL('/tools/configure/', location.origin);
                    configureUrl.search = returnedUrl.search;
                    const destination = configureUrl.toString();
                    if (item.href !== destination) item.href = destination;
                    if (item.target !== '_self') item.target = '_self';
                } catch (_error) {
                    // Leave malformed device URLs untouched so ESP Web Tools can report them.
                }
            });
        };
        const inspect = () => {
            observeRoot(dialog.shadowRoot);
            customizeNativeDeviceLink();
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
        $('.js-matter-read').addEventListener('click', matterReadQr);
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

    function monitorChartMaxPoints() {
        return Math.max(2, Math.ceil(MONITOR_CHART_WINDOW_MS / evaluationIntervalMs()) + 2);
    }

    function monitorChartCoalesceMs() {
        return Math.max(16, Math.min(100, Math.floor(evaluationIntervalMs() / 2)));
    }

    const monitor = {
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
        commands: new Set(),
        commandCatalogReady: false,
        switchingTransport: false,
        diagTimer: null,
        diagIntervalMs: 0,
        diagRequestPending: false,
        calibrating: false,
        calibrationTimer: null
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
            transport: 'simulation',
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
                transport: 'simulation',
                input_mode: monitor.inputMode,
                reason,
                duration_seconds: Math.max(0, Math.round((Date.now() - monitor.connectedAt) / 1000))
            });
        }
        resetMonitorSession();
    }

    function monitorPublishCommand(fields, {
        pendingMessage = 'Sending command…',
        statusFn = monitorStatus,
        timeoutMs = 8000
    } = {}) {
        if (conn.mode === 'direct' && directClient?.connected) {
            const { command, ...params } = fields;
            statusFn(pendingMessage);
            return directClient.request(command, params, { timeoutMs });
        }
        const error = new Error('Connect through Direct HTTP before changing the device.');
        statusFn(error.message);
        return Promise.reject(error);
    }

    function diagnosticsRequestPending() {
        return conn.mode === 'direct' && monitor.diagRequestPending;
    }

    function diagnosticsPanelOpen() {
        const panel = $('.device-live-diagnostics');
        const workspace = $('.js-monitor-workspace');
        return !!(panel && panel.open && route === 'tool-monitor' && workspace && !workspace.hidden);
    }

    function stopDiagnosticsPolling() {
        if (!monitor.diagTimer) return;
        clearInterval(monitor.diagTimer);
        monitor.diagTimer = null;
        monitor.diagIntervalMs = 0;
    }

    function syncDiagnosticsPolling() {
        const canPoll = diagnosticsPanelOpen()
            && (conn.mode === 'demo' || conn.mode === 'direct');
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
        const direct = conn.mode === 'direct';
        if (direct && !directClient?.connected) return;
        try {
            if (direct) monitor.diagRequestPending = true;
            const response = await monitorPublishCommand({ command: 'diagnostics' }, {
                pendingMessage: '',
                statusFn: () => {}
            });
            const data = direct ? response : response?.data;
            if (data && typeof data === 'object') {
                markMonitorReady('diagnostics');
                monitorStats(data);
                monitorDiagStatus('');
            }
        } catch (error) {
            if (diagnosticsPanelOpen()) monitorDiagStatus(error.message);
        } finally {
            if (direct) monitor.diagRequestPending = false;
        }
    }

    function monitorOpenConnectivity() {
        setDeviceView('connectivity');
        renderConnection();
    }

    function monitorEditOrCancel() {
        monitorOpenConnectivity();
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

    function selectMonitorTransport(mode) {
        document.querySelector('espectre-connection-picker[data-surface="monitor"]')?.select(mode);
    }

    function monitorInit() {
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

    const utf8Length = (value) => new TextEncoder().encode(String(value)).byteLength;

    function directConfigSnapshot(config) {
        const device = config.device || {};
        const wifi = config.wifi || {};
        const mqtt = config.mqtt || {};
        return {
            device_label: config.device_label ?? device.device_label ?? '',
            wifi_ssid: wifi.ssid || '',
            wifi_band: wifi.band || '',
            wifi_bssid: wifi.bssid || '',
            wifi_apply_state: wifi.apply_state || '',
            mqtt_host: mqtt.host || '',
            mqtt_port: mqtt.port || 0,
            topic_prefix: mqtt.topic_prefix || ''
        };
    }

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
        if (!pending) return;
        if (directClient && conn.mode === 'direct' && conn.status === 'connecting') {
            pending.timer = setTimeout(requestConfigVerification, CONFIG_VERIFICATION_RETRY_MS);
            return;
        }
        if (!directClient?.connected) {
            finishConfigVerification('unconfirmed', 'VerificationUnavailable');
            return;
        }
        pending.attempts += 1;
        directClient.request('config').then((config) => {
            applyDirectConfig(config);
            evaluateConfigVerification(directConfigSnapshot(config));
        }).catch(() => finishConfigVerification('unconfirmed', 'VerificationRequestFailed'));
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

    async function cfgApply(action, successMessage, method, params = {}, verify) {
        if (conn.mode === 'demo') {
            toast(successMessage + ' (demo — nothing written)');
            return true;
        }
        if (!directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', { action, result: 'failure', error_type: 'NotConnected' });
            return false;
        }
        try {
            await directClient.request(method, params);
            toast(successMessage);
            track('configure_change', { action, result: 'accepted' });
            if (verify) beginConfigVerification(action, verify);
            return true;
        } catch (error) {
            toast('Update failed: ' + (error.message || error));
            track('configure_change', { action, result: 'failure', error_type: errorType(error) });
            return false;
        }
    }

    function cfgValidationFailed(action, message) {
        toast(message);
        track('configure_change', { action, result: 'validation_failure' });
    }

    async function cfgRefreshDevice() {
        if (conn.mode === 'direct' && directClient?.connected) {
            try {
                await refreshDirectDevice();
            } catch (error) {
                console.warn('Direct device refresh failed:', error);
            }
        }
    }

    function renderWifiAccessPoints(snapshot = {}) {
        const select = document.getElementById('cfg-bssid');
        const status = $('.js-wifi-scan-status');
        const scanButton = $('.js-wifi-scan');
        if (!select || !scanButton) return;
        const accessPoints = Array.isArray(snapshot.access_points) ? snapshot.access_points : [];
        const options = [new Option('Automatic (strongest available)', '')];
        accessPoints.forEach((accessPoint) => {
            const bssid = String(accessPoint?.bssid || '').toUpperCase();
            const rssi = Number(accessPoint?.rssi_dbm);
            if (!/^[0-9A-F]{2}(?::[0-9A-F]{2}){5}$/.test(bssid)
                    || !Number.isInteger(rssi)) return;
            options.push(new Option(`${bssid} · ${rssi} dBm`, bssid));
        });
        if (currentWifiBssid && !options.some((option) => option.value === currentWifiBssid)) {
            options.push(new Option(`${currentWifiBssid} · pinned`, currentWifiBssid));
        }
        select.replaceChildren(...options);
        select.value = currentWifiBssid;
        const scanning = snapshot.scanning === true;
        select.disabled = scanning;
        scanButton.disabled = scanning;
        if (status) {
            status.textContent = snapshot.message
                || (accessPoints.length ? `${accessPoints.length} access point${accessPoints.length === 1 ? '' : 's'} found.`
                    : 'No matching access points found. Automatic selection remains available.');
        }
    }

    async function cfgRefreshWifiAccessPoints() {
        if (conn.mode === 'demo') {
            renderWifiAccessPoints({
                scanning: false,
                message: '2 access points found. (demo)',
                access_points: [
                    { bssid: 'E6:FA:C4:20:19:DE', channel: 6, rssi_dbm: -43 },
                    { bssid: 'A2:11:7C:09:88:31', channel: 11, rssi_dbm: -67 }
                ]
            });
            return;
        }
        if (!directClient?.connected || !monitor.commands.has('scan_wifi_access_points')) return;
        renderWifiAccessPoints({ scanning: true, message: 'Scanning access points…' });
        try {
            await directClient.request('scan_wifi_access_points');
            for (let attempt = 0; attempt < 20; attempt += 1) {
                await new Promise((resolve) => setTimeout(resolve, 350));
                const snapshot = await directClient.request('wifi_access_points');
                renderWifiAccessPoints(snapshot);
                if (!snapshot.scanning) return;
            }
            renderWifiAccessPoints({ scanning: false, message: 'Access point scan timed out. Try again.' });
        } catch (error) {
            renderWifiAccessPoints({
                scanning: false,
                message: 'Access point scan failed: ' + (error.message || error)
            });
        }
    }

    async function cfgSaveWifi() {
        const bssid = cfgValue('cfg-bssid').trim().toUpperCase();
        const method = bssid ? 'set_wifi_bssid' : 'clear_wifi_bssid';
        const params = bssid ? { bssid } : {};
        await cfgApply(
            method,
            bssid ? 'Access point saved; station reconnecting.' : 'Automatic access-point selection saved; station reconnecting.',
            method, params,
            (snapshot) => String(snapshot.wifi_bssid || '').toUpperCase() === bssid);
    }

    const CONFIG_CLEAR_DIALOGS = Object.freeze({
        wifi: Object.freeze({
            kicker: 'Primary connection',
            title: 'Reset Wi-Fi configuration?',
            description: 'This removes the provisioned Wi-Fi network and password from the device.',
            warning: 'The device will disconnect. Provision it again over Improv Serial to restore network access.',
            confirm: 'Reset Wi-Fi'
        }),
        mqtt: Object.freeze({
            kicker: 'MQTT integration',
            title: 'Remove MQTT configuration?',
            description: 'This removes the broker host, port, username, password, and topic prefix.',
            warning: 'Wi-Fi stays connected, but MQTT automations and Home Assistant discovery will stop.',
            confirm: 'Remove MQTT'
        })
    });

    function closeConfigClearDialog(confirmed = false) {
        const modal = $('.js-config-clear-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (configClearReturnFocus && configClearReturnFocus.isConnected) configClearReturnFocus.focus();
        configClearReturnFocus = null;
        const resolve = configClearResolve;
        configClearResolve = null;
        if (resolve) resolve(confirmed);
    }

    function openConfigClearDialog(kind, returnFocus) {
        const copy = CONFIG_CLEAR_DIALOGS[kind];
        if (!copy) return Promise.resolve(false);
        if (configClearResolve) closeConfigClearDialog(false);
        const modal = $('.js-config-clear-modal');
        $('.js-config-clear-kicker').textContent = copy.kicker;
        $('.js-config-clear-title').textContent = copy.title;
        $('.js-config-clear-description').textContent = copy.description;
        $('.js-config-clear-warning').textContent = copy.warning;
        $('.js-config-clear-confirm').textContent = copy.confirm;
        configClearReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        $('.js-config-clear-confirm').focus();
        return new Promise((resolve) => { configClearResolve = resolve; });
    }

    async function cfgClearWifi() {
        if (conn.mode !== 'demo' && !directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', {
                action: 'clear_wifi', result: 'failure', error_type: 'NotConnected'
            });
            return;
        }
        if (!await openConfigClearDialog('wifi', document.activeElement)) return;
        if (conn.mode === 'demo') {
            toast('Wi-Fi configuration removed. (demo — nothing written)');
            return;
        }
        let result = 'accepted';
        let message = 'Wi-Fi configuration removed. Provision the device again via Improv Serial.';
        try {
            await directClient.request('clear_wifi_config', {}, { timeoutMs: 3000 });
        } catch (error) {
            if (error?.code !== 'timeout' && error?.code !== 'closed') {
                toast('Update failed: ' + (error.message || error));
                track('configure_change', {
                    action: 'clear_wifi', result: 'failure', error_type: errorType(error)
                });
                return;
            }
            result = 'unconfirmed';
            message = 'Wi-Fi removal sent. The device disconnected as expected; provision it again via Improv Serial.';
        }
        ['cfg-ssid', 'cfg-wifi-band', 'cfg-channel', 'cfg-bssid'].forEach((id) => {
            document.getElementById(id).value = '';
        });
        track('configure_change', { action: 'clear_wifi', result });
        toast(message);
        teardownConnection('wifi_cleared');
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
        const clearCredentials = document.getElementById('cfg-mqtt-credentials-clear').checked;
        if (!host || !cfgValue('cfg-mqtt-port')) {
            cfgValidationFailed('set_mqtt', 'MQTT needs a host and port.');
            return;
        }
        if (!browserBrokerHost(host)) {
            cfgValidationFailed('set_mqtt', 'Complete the MQTT broker address after mqtts://.');
            return;
        }
        const port = Number(cfgValue('cfg-mqtt-port'));
        const topicPrefix = cfgValue('cfg-topic-prefix').trim().replace(/\/+$/, '');
        if (!Number.isInteger(port) || port < 1 || port > 65535 || !topicPrefix || /[+#\0]/.test(topicPrefix)) {
            cfgValidationFailed('set_mqtt', 'Use a port from 1 to 65535 and a topic prefix without MQTT wildcards.');
            return;
        }
        const mqttParams = { host, port, topic_prefix: topicPrefix };
        if (username || clearCredentials) mqttParams.username = clearCredentials ? '' : username;
        if (password || clearCredentials) mqttParams.password = clearCredentials ? '' : password;
        const ok = await cfgApply('set_mqtt', 'MQTT settings saved.',
            'set_mqtt_config', mqttParams,
            (snapshot) => snapshot.mqtt_host === host && Number(snapshot.mqtt_port) === port);
        if (ok) {
            document.getElementById('cfg-mqtt-user').value = '';
            document.getElementById('cfg-mqtt-pass').value = '';
            document.getElementById('cfg-mqtt-credentials-clear').checked = false;
            syncConfigureMqttCredentialMode();
        }
    }

    function applyConfigureMqttDefaults() {
        applyConfigureMqttPreset('home_assistant');
    }

    async function cfgClearMqtt() {
        if (conn.mode !== 'demo' && !directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', {
                action: 'clear_mqtt', result: 'failure', error_type: 'NotConnected'
            });
            return;
        }
        if (!await openConfigClearDialog('mqtt', document.activeElement)) return;
        const ok = await cfgApply(
            'clear_mqtt', 'MQTT settings cleared.', 'clear_mqtt_config', {},
            (snapshot) => !snapshot.mqtt_host);
        if (ok) applyConfigureMqttDefaults();
    }

    async function cfgSaveDeviceLabel(label) {
        if (typeof label !== 'string' || /[\r\n\0]/.test(label) || utf8Length(label) > 32) {
            cfgValidationFailed('set_device', 'Device name must be one line and at most 32 UTF-8 bytes.');
            return false;
        }
        return cfgApply('set_device', 'Device name saved.', 'set_device_label',
            { device_label: label },
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
        const otaTransportReady = conn.mode === 'direct' && directClient?.connected;
        button.disabled = !otaTransportReady
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
            el.hidden = conn.mode === 'demo' || Boolean(flash.usbDialog) || otaSupported === false;
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
        if (conn.mode === 'direct' && directClient?.connected) return 'direct';
        return '';
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
        if (!manual) otaCheckTransport = transport;
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
        if (conn.mode === 'demo') return;
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
        if (!currentOtaCheckTransport()) return;
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
        applyMqttPresetFieldLocks('configure', MQTT_PRESETS[presetName].configure);
        applyConfigureMqttCredentialPolicy(presetName);
        $('.js-wifi-save').addEventListener('click', cfgSaveWifi);
        $('.js-wifi-scan').addEventListener('click', cfgRefreshWifiAccessPoints);
        $('.js-wifi-clear').addEventListener('click', cfgClearWifi);
        $('.js-mqtt-save').addEventListener('click', cfgSaveMqtt);
        $('.js-mqtt-clear').addEventListener('click', cfgClearMqtt);
        document.getElementById('cfg-mqtt-credentials-clear').addEventListener('change', syncConfigureMqttCredentialMode);
        document.getElementById('cfg-mqtt-preset').addEventListener('change', (event) => {
            applyConfigureMqttPreset(event.currentTarget.value);
        });
        $$('.js-config-clear-cancel').forEach((button) => {
            button.addEventListener('click', () => closeConfigClearDialog(false));
        });
        $('.js-config-clear-confirm').addEventListener('click', () => closeConfigClearDialog(true));
        $('.js-config-clear-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) closeConfigClearDialog(false);
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
            else if (!$('.js-config-clear-modal').hidden) closeConfigClearDialog(false);
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
        const previewing = route === 'tool-game' && conn.mode
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
        if (game.previewRaf || route !== 'tool-game' || !conn.mode
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
        if (route !== 'tool-game') return;
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

    /* =========================================================== raw CSI */

    const RAW_CSI_V8_HEADER_BYTES = 64;
    const RAW_CSI_VISUAL_HISTORY = 720;
    const RAW_CSI_PHASE_HISTORY = 72;
    const RAW_CSI_IQ_WINDOW_US = 1000000;
    const RAW_CSI_VISUAL_STEP_US = 10000;
    const RAW_CSI_CHANNEL_GHOST_GAIN = 5;
    const RAW_CSI_PHASE_TRAIL_GAIN = 5;
    const RAW_CSI_SELECTED_SUBCARRIERS = Object.freeze([4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]);
    const RAW_CSI_LIVE_SUBCARRIERS = Object.freeze(
        Array.from({ length: 57 }, (_unused, index) => index + 4).filter((index) => index !== 32));
    const RAW_CSI_VISUALIZATIONS = Object.freeze({
        'channel-heatmap': Object.freeze({
            title: 'Channel heatmap',
            description: 'Brightness shows normalized amplitude; cyan and coral reveal movement around the slow baseline.',
            badge: 'LIVE',
            ariaLabel: 'Combined CSI amplitude and motion heatmap over time'
        }),
        'rf-waterfall': Object.freeze({
            title: 'RF waterfall',
            description: 'Recent channel profiles recede through time while movement disturbs the surface.',
            badge: 'LIVE',
            ariaLabel: 'Perspective waterfall of recent CSI channel profiles'
        }),
        'channel-ghost': Object.freeze({
            title: 'Channel ghost',
            description: 'The current channel departs from its slow baseline with 5× visual gain; color preserves the deviation sign.',
            badge: 'LIVE',
            ariaLabel: 'Current normalized CSI channel profile compared with its baseline'
        }),
        'iq-constellation': Object.freeze({
            title: 'I/Q constellation',
            description: 'Recent raw Espressif I/Q samples from the 12 production subcarriers over a one-second window.',
            badge: 'LIVE',
            ariaLabel: 'Recent raw CSI I and Q constellation samples by subcarrier'
        }),
        'phase-trails': Object.freeze({
            title: 'Sanitized phase trails',
            description: 'Experimental relative I/Q phase with 5× trail spread after common packet rotation and linear phase ramp are removed.',
            badge: 'EXPERIMENTAL',
            ariaLabel: 'Experimental sanitized CSI phase constellation trails'
        })
    });
    const rawCsi = {
        sessionClient: null,
        controller: null,
        demoTimer: null,
        demoFresh: 0,
        running: false,
        parser: null,
        visualization: 'channel-heatmap',
        profiles: [],
        deltas: [],
        timestampsUs: [],
        phaseHistory: [],
        iqHistory: [],
        iqTimestampsUs: [],
        packetArrivalTimes: [],
        baseline: null,
        latestProfile: null,
        latestDelta: null,
        lastCaptureTicksUs: 0,
        lastVisualTicksUs: 0,
        renderFrame: 0,
        resizeObserver: null
    };

    function rawCsiStatus(message, error = false) {
        const status = $('.js-raw-csi-status');
        if (!status) return;
        status.textContent = message;
        status.hidden = !message;
        status.classList.toggle('is-error', error);
    }

    function rawCsiDirectReady() {
        return conn.mode === 'direct' && conn.status === 'connected' && Boolean(directClient?.connected);
    }

    function rawCsiSetAvailable(available) {
        const unavailable = $('.js-raw-csi-unavailable');
        const workspace = $('.js-raw-csi-workspace');
        if (unavailable) unavailable.hidden = available;
        if (workspace) workspace.hidden = !available;
    }

    function rawCsiUseConnection() {
        const onboarding = $('.js-raw-csi-onboarding');
        const externalHint = $('.js-raw-csi-external-hint');
        if (conn.status !== 'connected' || !['direct', 'demo'].includes(conn.mode)) {
            if (externalHint) externalHint.hidden = true;
            if (onboarding) onboarding.hidden = false;
            $('.js-raw-csi-unavailable').hidden = true;
            $('.js-raw-csi-workspace').hidden = true;
            return false;
        }
        if (onboarding) onboarding.hidden = true;
        if (conn.mode === 'demo') {
            if (externalHint) externalHint.hidden = true;
            rawCsiSetAvailable(true);
            rawCsiStatus('Demo connected. Start the simulated CSI stream when ready.');
            return true;
        }
        const available = directClient?.capabilities?.features?.raw_csi === true;
        rawCsiSetAvailable(available);
        if (externalHint) externalHint.hidden = !available || conn.csiTrafficMode !== 'external';
        if (available) {
            rawCsiStatus(conn.csiTrafficMode === 'external'
                ? 'Connected in external traffic mode. The stream stays idle until UDP marker traffic reaches the device.'
                : 'Connected. Start the ephemeral stream when ready.');
        }
        return available;
    }

    function rawCsiSetRunning(running) {
        rawCsi.running = running;
        const start = $('.js-raw-csi-start');
        const stop = $('.js-raw-csi-stop');
        if (start) start.disabled = running;
        if (stop) stop.disabled = !running;
    }

    function rawCsiCounter(selector, value) {
        const element = $(selector);
        if (element) element.textContent = typeof value === 'bigint'
            ? value.toLocaleString('en-US') : Number(value).toLocaleString('en-US');
    }

    function rawCsiUpdatePacketRate(received) {
        const now = performance.now();
        if (received) rawCsi.packetArrivalTimes.push(now);
        const cutoff = now - 1000;
        while (rawCsi.packetArrivalTimes[0] <= cutoff) rawCsi.packetArrivalTimes.shift();
        rawCsiCounter('.js-raw-pps', rawCsi.packetArrivalTimes.length);
    }

    function rawCsiPushBounded(collection, value, limit) {
        collection.push(value);
        if (collection.length > limit) collection.shift();
    }

    function rawCsiLiveSubcarriers(length) {
        if (length === 64) return RAW_CSI_LIVE_SUBCARRIERS;
        return Array.from({ length }, (_unused, index) => index);
    }

    function rawCsiResetVisualization() {
        rawCsi.profiles.length = 0;
        rawCsi.deltas.length = 0;
        rawCsi.timestampsUs.length = 0;
        rawCsi.phaseHistory.length = 0;
        rawCsi.iqHistory.length = 0;
        rawCsi.iqTimestampsUs.length = 0;
        rawCsi.packetArrivalTimes.length = 0;
        rawCsiCounter('.js-raw-pps', 0);
        rawCsi.baseline = null;
        rawCsi.latestProfile = null;
        rawCsi.latestDelta = null;
        rawCsi.lastCaptureTicksUs = 0;
        rawCsi.lastVisualTicksUs = 0;
        rawCsiScheduleRender();
    }

    function rawCsiNormalizeProfile(amplitudes) {
        const profile = new Float32Array(amplitudes.length);
        const liveSubcarriers = rawCsiLiveSubcarriers(amplitudes.length);
        let sum = 0;
        let count = 0;
        liveSubcarriers.forEach((index) => {
            if (amplitudes[index] <= 0) return;
            sum += amplitudes[index];
            count += 1;
        });
        const mean = count ? sum / count : 1;
        liveSubcarriers.forEach((index) => {
            profile[index] = amplitudes[index] / Math.max(mean, 1e-6);
        });
        return profile;
    }

    function rawCsiUpdateBaseline(profile, captureTicksUs) {
        if (!rawCsi.baseline || rawCsi.baseline.length !== profile.length) {
            rawCsi.baseline = profile.slice();
            rawCsi.lastCaptureTicksUs = captureTicksUs;
            return new Float32Array(profile.length);
        }
        const elapsedUs = rawCsi.lastCaptureTicksUs > 0 && captureTicksUs > rawCsi.lastCaptureTicksUs
            ? captureTicksUs - rawCsi.lastCaptureTicksUs : RAW_CSI_VISUAL_STEP_US;
        const alpha = Math.max(0.0002, Math.min(0.25, 1 - Math.exp(-elapsedUs / 5000000)));
        const delta = new Float32Array(profile.length);
        const liveSubcarriers = rawCsiLiveSubcarriers(profile.length);
        liveSubcarriers.forEach((index) => {
            const baseline = rawCsi.baseline[index];
            delta[index] = Math.log((profile[index] + 0.05) / (baseline + 0.05));
            rawCsi.baseline[index] = baseline + alpha * (profile[index] - baseline);
        });
        rawCsi.lastCaptureTicksUs = captureTicksUs;
        return delta;
    }

    function rawCsiSanitizedPhase(iValues, qValues, profile) {
        if (profile.length !== 64) return null;
        const residualReal = new Float32Array(profile.length - 1);
        const residualImag = new Float32Array(profile.length - 1);
        let commonReal = 0;
        let commonImag = 0;
        for (let left = 4; left < 60; left += 1) {
            if (left === 31 || left === 32) continue;
            const right = left + 1;
            const real = iValues[left] * iValues[right] + qValues[left] * qValues[right];
            const imag = qValues[left] * iValues[right] - iValues[left] * qValues[right];
            const magnitude = Math.hypot(real, imag);
            if (magnitude <= 1e-6) continue;
            residualReal[left] = real / magnitude;
            residualImag[left] = imag / magnitude;
            commonReal += residualReal[left];
            commonImag += residualImag[left];
        }
        const commonMagnitude = Math.hypot(commonReal, commonImag);
        if (commonMagnitude <= 1e-6) return null;
        const commonUnitReal = commonReal / commonMagnitude;
        const commonUnitImag = commonImag / commonMagnitude;
        const result = new Float32Array(RAW_CSI_SELECTED_SUBCARRIERS.length * 2);
        RAW_CSI_SELECTED_SUBCARRIERS.forEach((subcarrier, index) => {
            const left = subcarrier === 60 ? 59 : subcarrier;
            const real = residualReal[left];
            const imag = residualImag[left];
            const sanitizedReal = real * commonUnitReal + imag * commonUnitImag;
            const sanitizedImag = imag * commonUnitReal - real * commonUnitImag;
            const radius = 0.32 + 0.68 * Math.min(1, profile[subcarrier] / 2);
            result[index * 2] = sanitizedReal * radius;
            result[index * 2 + 1] = sanitizedImag * radius;
        });
        return result;
    }

    function rawCsiIngestVisualFrame(amplitudes, iValues, qValues, captureTicksUs) {
        const profile = rawCsiNormalizeProfile(amplitudes);
        const delta = rawCsiUpdateBaseline(profile, captureTicksUs);
        rawCsi.latestProfile = profile;
        rawCsi.latestDelta = delta;
        if (rawCsi.lastVisualTicksUs > 0
                && captureTicksUs - rawCsi.lastVisualTicksUs < RAW_CSI_VISUAL_STEP_US) {
            rawCsiScheduleRender();
            return;
        }
        rawCsiPushBounded(rawCsi.profiles, profile, RAW_CSI_VISUAL_HISTORY);
        rawCsiPushBounded(rawCsi.deltas, delta, RAW_CSI_VISUAL_HISTORY);
        rawCsiPushBounded(rawCsi.timestampsUs, captureTicksUs, RAW_CSI_VISUAL_HISTORY);
        const sanitizedPhase = rawCsiSanitizedPhase(iValues, qValues, profile);
        if (sanitizedPhase) {
            rawCsiPushBounded(rawCsi.phaseHistory, sanitizedPhase, RAW_CSI_PHASE_HISTORY);
        }
        const iq = new Float32Array(iValues.length * 2);
        iValues.forEach((value, index) => {
            iq[index * 2] = value;
            iq[index * 2 + 1] = qValues[index];
        });
        rawCsi.iqHistory.push(iq);
        rawCsi.iqTimestampsUs.push(captureTicksUs);
        while (rawCsi.iqHistory.length > 1
                && captureTicksUs - rawCsi.iqTimestampsUs[0] > RAW_CSI_IQ_WINDOW_US) {
            rawCsi.iqHistory.shift();
            rawCsi.iqTimestampsUs.shift();
        }
        rawCsi.lastVisualTicksUs = captureTicksUs;
        rawCsiScheduleRender();
    }

    function rawCsiCanvasContext() {
        const canvas = $('.js-raw-visualization');
        const context = canvas?.getContext('2d');
        return canvas && context ? { canvas, context } : null;
    }

    function rawCsiResizeVisualization() {
        const canvas = $('.js-raw-visualization');
        const stage = canvas?.closest('.raw-csi-visualization-stage');
        const width = Math.round(stage?.clientWidth || 0);
        if (!canvas || width < 100) return;
        const height = window.matchMedia('(max-width: 620px)').matches
            ? 260 : Math.min(420, Math.round(width * 420 / 960));
        if (canvas.width === width && canvas.height === height) return;
        canvas.width = width;
        canvas.height = height;
        canvas.style.height = `${height}px`;
        rawCsiScheduleRender();
    }

    function rawCsiClearCanvas(context, canvas) {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = '#05070d';
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    function rawCsiDrawEmpty(context, canvas, message = 'Start the stream to reveal the channel.') {
        rawCsiClearCanvas(context, canvas);
        context.fillStyle = 'rgba(255, 255, 255, .48)';
        context.font = '500 15px "JetBrains Mono", monospace';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(message, canvas.width / 2, canvas.height / 2);
    }

    function rawCsiMotionColor(value, alpha = 1) {
        const intensity = Math.min(1, Math.abs(value));
        const base = [12, 10, 31];
        const target = value < 0 ? [54, 215, 255] : [255, 91, 118];
        const channels = base.map((channel, index) => Math.round(
            channel + (target[index] - channel) * intensity));
        return `rgba(${channels[0]}, ${channels[1]}, ${channels[2]}, ${alpha})`;
    }

    function rawCsiChannelColor(amplitude, delta) {
        const level = Math.max(0, Math.min(1, amplitude / 2.2));
        const motion = Math.sqrt(Math.min(1, Math.abs(delta) / 0.32));
        const base = [Math.round(8 + level * 46), Math.round(8 + level * 34), Math.round(24 + level * 126)];
        const target = delta < 0 ? [42, 220, 255] : [255, 74, 105];
        const channels = base.map((channel, index) => Math.round(
            channel + (target[index] - channel) * motion));
        return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
    }

    function rawCsiDrawHeatmap(context, canvas) {
        if (!rawCsi.profiles.length) {
            rawCsiDrawEmpty(context, canvas);
            return;
        }
        rawCsiClearCanvas(context, canvas);
        const left = 58;
        const top = 24;
        const width = canvas.width - left - 22;
        const height = canvas.height - top - 48;
        context.fillStyle = '#09091c';
        context.fillRect(left, top, width, height);
        const columnWidth = width / RAW_CSI_VISUAL_HISTORY;
        const startX = left + width - rawCsi.profiles.length * columnWidth;
        rawCsi.profiles.forEach((profile, profileIndex) => {
            const x0 = Math.floor(startX + profileIndex * columnWidth);
            const x1 = Math.ceil(startX + (profileIndex + 1) * columnWidth);
            profile.forEach((value, subcarrier) => {
                context.fillStyle = rawCsiChannelColor(
                    value, rawCsi.deltas[profileIndex]?.[subcarrier] || 0);
                const y0 = Math.floor(top + subcarrier * height / profile.length);
                const y1 = Math.ceil(top + (subcarrier + 1) * height / profile.length);
                context.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
            });
        });
        context.strokeStyle = 'rgba(255, 255, 255, .15)';
        context.strokeRect(left + 0.5, top + 0.5, width - 1, height - 1);
        context.fillStyle = 'rgba(255, 255, 255, .55)';
        context.font = '12px "JetBrains Mono", monospace';
        context.textAlign = 'right';
        context.textBaseline = 'middle';
        context.fillText('−32', left - 10, top + 4);
        context.fillText('0', left - 10, top + height / 2);
        context.fillText('+31', left - 10, top + height - 4);
        context.textAlign = 'left';
        context.textBaseline = 'alphabetic';
        context.fillText('SUBCARRIER', left, canvas.height - 14);
        context.textAlign = 'right';
        context.fillText('RECENT TIME →', left + width, canvas.height - 14);
        RAW_CSI_SELECTED_SUBCARRIERS.forEach((subcarrier) => {
            const y = top + (subcarrier + 0.5) * height / rawCsi.profiles[0].length;
            context.fillStyle = 'rgba(255, 255, 255, .72)';
            context.fillRect(left - 4, y - 1, 4, 2);
        });
    }

    function rawCsiDrawWaterfall(context, canvas) {
        if (!rawCsi.profiles.length) {
            rawCsiDrawEmpty(context, canvas);
            return;
        }
        rawCsiClearCanvas(context, canvas);
        const profiles = rawCsi.profiles.slice(-48);
        const deltas = rawCsi.deltas.slice(-profiles.length);
        const centerX = canvas.width / 2;
        const backY = 54;
        const frontY = canvas.height - 58;
        const maximumSpan = canvas.width - 110;
        context.strokeStyle = 'rgba(111, 91, 220, .16)';
        context.lineWidth = 1;
        for (let line = 0; line <= 8; line += 1) {
            const x = centerX - maximumSpan / 2 + line * maximumSpan / 8;
            context.beginPath();
            context.moveTo(centerX + (x - centerX) * 0.62, backY);
            context.lineTo(x, frontY);
            context.stroke();
        }
        profiles.forEach((profile, profileIndex) => {
            const depth = profiles.length === 1 ? 1 : profileIndex / (profiles.length - 1);
            const yBase = backY + depth * (frontY - backY);
            const span = maximumSpan * (0.62 + depth * 0.38);
            const xStart = centerX - span / 2;
            let energy = 0;
            RAW_CSI_LIVE_SUBCARRIERS.forEach((subcarrier) => {
                energy += Math.abs(deltas[profileIndex][subcarrier]);
            });
            energy /= RAW_CSI_LIVE_SUBCARRIERS.length;
            const active = Math.sqrt(Math.min(1, energy / 0.08));
            const alpha = 0.16 + depth * 0.7;
            const red = Math.round(112 + active * 143);
            const green = Math.round(68 + active * 10);
            const blue = Math.round(255 - active * 155);
            context.strokeStyle = `rgba(${red}, ${green}, ${blue}, ${alpha})`;
            context.lineWidth = profileIndex === profiles.length - 1 ? 2.8 : 1 + active * 0.55;
            context.shadowColor = profileIndex === profiles.length - 1
                ? `rgba(${red}, ${green}, ${blue}, .9)` : 'transparent';
            context.shadowBlur = profileIndex === profiles.length - 1 ? 14 : 0;
            [[4, 31], [33, 60]].forEach(([start, end]) => {
                context.beginPath();
                for (let subcarrier = start; subcarrier <= end; subcarrier += 1) {
                    const frequencyPosition = (subcarrier - 4) / 56;
                    const x = xStart + frequencyPosition * span;
                    const y = yBase - (Math.max(0, Math.min(2.4, profile[subcarrier])) - 1) * 31;
                    if (subcarrier === start) context.moveTo(x, y);
                    else context.lineTo(x, y);
                }
                context.stroke();
            });
        });
        context.shadowBlur = 0;
        context.fillStyle = 'rgba(255, 255, 255, .55)';
        context.font = '12px "JetBrains Mono", monospace';
        context.textAlign = 'left';
        context.fillText('PAST', centerX - maximumSpan * 0.31, backY - 18);
        context.fillText('NOW', centerX - maximumSpan / 2, frontY + 28);
        context.textAlign = 'right';
        context.fillText('SUBCARRIER →', centerX + maximumSpan / 2, frontY + 28);
        context.textAlign = 'center';
        context.fillStyle = 'rgba(255, 255, 255, .48)';
        context.fillText('QUIET VIOLET  ·  MOTION CORAL', centerX, canvas.height - 14);
        RAW_CSI_SELECTED_SUBCARRIERS.forEach((subcarrier) => {
            const x = centerX - maximumSpan / 2 + (subcarrier - 4) * maximumSpan / 56;
            context.fillStyle = 'rgba(255, 255, 255, .7)';
            context.fillRect(x - 1, frontY + 5, 2, 5);
        });
    }

    function rawCsiDrawChannelGhost(context, canvas) {
        if (!rawCsi.latestProfile || !rawCsi.baseline) {
            rawCsiDrawEmpty(context, canvas);
            return;
        }
        rawCsiClearCanvas(context, canvas);
        const left = 62;
        const right = canvas.width - 28;
        const top = 38;
        const bottom = canvas.height - 58;
        const middle = (top + bottom) / 2;
        const profileScale = Math.min(108, (bottom - top) * 0.36);
        const yForValue = (value) => middle
            - (Math.max(0, Math.min(2.4, value)) - 1) * profileScale;
        const amplifiedValue = (subcarrier) => rawCsi.baseline[subcarrier]
            + (rawCsi.latestProfile[subcarrier] - rawCsi.baseline[subcarrier])
                * RAW_CSI_CHANNEL_GHOST_GAIN;
        context.strokeStyle = 'rgba(122, 105, 210, .18)';
        context.lineWidth = 1;
        [0.5, 1, 1.5, 2].forEach((value) => {
            const y = yForValue(value);
            context.beginPath();
            context.moveTo(left, y);
            context.lineTo(right, y);
            context.stroke();
        });
        [[4, 31], [33, 60]].forEach(([start, end]) => {
            for (let subcarrier = start; subcarrier < end; subcarrier += 1) {
                const next = subcarrier + 1;
                const x0 = left + (subcarrier - 4) * (right - left) / 56;
                const x1 = left + (next - 4) * (right - left) / 56;
                const current0 = yForValue(amplifiedValue(subcarrier));
                const current1 = yForValue(amplifiedValue(next));
                const baseline0 = yForValue(rawCsi.baseline[subcarrier]);
                const baseline1 = yForValue(rawCsi.baseline[next]);
                const delta = ((rawCsi.latestDelta[subcarrier] || 0) + (rawCsi.latestDelta[next] || 0)) / 2;
                context.fillStyle = rawCsiMotionColor(delta / 0.22, 0.58);
                context.beginPath();
                context.moveTo(x0, baseline0);
                context.lineTo(x1, baseline1);
                context.lineTo(x1, current1);
                context.lineTo(x0, current0);
                context.closePath();
                context.fill();
            }
            context.setLineDash([7, 7]);
            context.strokeStyle = 'rgba(255, 255, 255, .4)';
            context.lineWidth = 1.4;
            context.beginPath();
            for (let subcarrier = start; subcarrier <= end; subcarrier += 1) {
                const x = left + (subcarrier - 4) * (right - left) / 56;
                const y = yForValue(rawCsi.baseline[subcarrier]);
                if (subcarrier === start) context.moveTo(x, y);
                else context.lineTo(x, y);
            }
            context.stroke();
            context.setLineDash([]);
            context.strokeStyle = '#8f7aff';
            context.lineWidth = 2.4;
            context.shadowColor = 'rgba(107, 196, 255, .55)';
            context.shadowBlur = 10;
            context.beginPath();
            for (let subcarrier = start; subcarrier <= end; subcarrier += 1) {
                const x = left + (subcarrier - 4) * (right - left) / 56;
                const y = yForValue(amplifiedValue(subcarrier));
                if (subcarrier === start) context.moveTo(x, y);
                else context.lineTo(x, y);
            }
            context.stroke();
        });
        context.shadowBlur = 0;
        RAW_CSI_SELECTED_SUBCARRIERS.forEach((subcarrier) => {
            const x = left + (subcarrier - 4) * (right - left) / 56;
            const y = yForValue(amplifiedValue(subcarrier));
            context.fillStyle = '#d9d2ff';
            context.beginPath();
            context.arc(x, y, 3, 0, 2 * Math.PI);
            context.fill();
        });
        context.fillStyle = 'rgba(255, 255, 255, .55)';
        context.font = '12px "JetBrains Mono", monospace';
        context.textAlign = 'left';
        context.fillText('— CURRENT', left, canvas.height - 18);
        context.fillStyle = 'rgba(255, 255, 255, .4)';
        context.fillText('┄ BASELINE', left + 118, canvas.height - 18);
        context.textAlign = 'right';
        context.fillStyle = 'rgba(255, 255, 255, .55)';
        context.fillText('5× DEVIATION', right, top - 12);
        context.fillText('SUBCARRIER →', right, canvas.height - 18);
    }

    function rawCsiDrawIqConstellation(context, canvas) {
        if (!rawCsi.iqHistory.length) {
            rawCsiDrawEmpty(context, canvas);
            return;
        }
        rawCsiClearCanvas(context, canvas);
        const latest = rawCsi.iqHistory[rawCsi.iqHistory.length - 1];
        const subcarrierCount = latest.length / 2;
        const selectedSubcarriers = RAW_CSI_SELECTED_SUBCARRIERS
            .filter((subcarrier) => subcarrier < subcarrierCount);
        const absoluteValues = [];
        rawCsi.iqHistory.forEach((sample) => {
            selectedSubcarriers.forEach((subcarrier) => {
                absoluteValues.push(
                    Math.abs(sample[subcarrier * 2]), Math.abs(sample[subcarrier * 2 + 1]));
            });
        });
        absoluteValues.sort((left, right) => left - right);
        const percentileIndex = Math.min(absoluteValues.length - 1,
            Math.floor(absoluteValues.length * 0.98));
        const extent = Math.max(12, Math.min(128,
            (absoluteValues[percentileIndex] || 0) * 1.12));
        const panelSize = Math.min(canvas.height - 58, canvas.width - 30);
        const top = (canvas.height - panelSize) / 2;
        const centerX = canvas.width / 2;
        const centerY = top + panelSize / 2;
        const halfSpan = panelSize / 2;
        const pointPosition = (sample, subcarrier) => ({
            x: Math.max(-1, Math.min(1, sample[subcarrier * 2] / extent)) * halfSpan,
            y: Math.max(-1, Math.min(1, sample[subcarrier * 2 + 1] / extent)) * halfSpan
        });
        const left = centerX - halfSpan;
        context.fillStyle = '#09091c';
        context.fillRect(left, top, panelSize, panelSize);
        context.strokeStyle = 'rgba(121, 105, 219, .2)';
        context.lineWidth = 1;
        [0.25, 0.5, 0.75].forEach((fraction) => {
            const offset = fraction * panelSize;
            context.beginPath();
            context.moveTo(left + offset, top);
            context.lineTo(left + offset, top + panelSize);
            context.moveTo(left, top + offset);
            context.lineTo(left + panelSize, top + offset);
            context.stroke();
        });
        context.strokeStyle = 'rgba(255, 255, 255, .25)';
        context.strokeRect(left + 0.5, top + 0.5, panelSize - 1, panelSize - 1);
        selectedSubcarriers.forEach((subcarrier, subcarrierIndex) => {
            const hue = 188 + subcarrierIndex * 12;
            rawCsi.iqHistory.forEach((sample, historyIndex) => {
                const depth = (historyIndex + 1) / rawCsi.iqHistory.length;
                const point = pointPosition(sample, subcarrier);
                context.fillStyle = `hsla(${hue}, 94%, 68%, ${0.05 + depth * depth * 0.36})`;
                context.fillRect(centerX + point.x - 1.2, centerY - point.y - 1.2, 2.4, 2.4);
            });
            const point = pointPosition(latest, subcarrier);
            context.fillStyle = `hsl(${hue} 94% 72%)`;
            context.shadowColor = `hsl(${hue} 94% 62%)`;
            context.shadowBlur = 8;
            context.beginPath();
            context.arc(centerX + point.x, centerY - point.y, 3.8, 0, 2 * Math.PI);
            context.fill();
        });
        context.shadowBlur = 0;
        context.fillStyle = 'rgba(255, 255, 255, .58)';
        context.font = '12px "JetBrains Mono", monospace';
        context.textAlign = 'center';
        context.fillText('12 PRODUCTION SUBCARRIERS · 1 SECOND', centerX, top - 10);
        context.textAlign = 'right';
        context.fillText('I →', left + panelSize, top + panelSize + 18);
        context.textAlign = 'left';
        context.fillText('Q ↑', left + 6, top + 16);
        context.fillStyle = 'rgba(255, 255, 255, .38)';
        context.fillText(`±${Math.ceil(extent)}`, left + 6, top + panelSize - 8);
        if (canvas.width >= 620) {
            selectedSubcarriers.forEach((subcarrier, index) => {
                const leftSide = index < selectedSubcarriers.length / 2;
                const row = index % (selectedSubcarriers.length / 2);
                const x = leftSide ? left - 118 : left + panelSize + 54;
                const y = top + 48 + row * Math.min(48, (panelSize - 72) / 5);
                const hue = 188 + index * 12;
                context.fillStyle = `hsl(${hue} 94% 70%)`;
                context.beginPath();
                context.arc(x, y - 4, 4, 0, 2 * Math.PI);
                context.fill();
                context.fillStyle = 'rgba(255, 255, 255, .48)';
                context.font = '11px "JetBrains Mono", monospace';
                context.textAlign = 'left';
                context.fillText(`SC ${subcarrier}`, x + 10, y);
            });
        }
    }

    function rawCsiDrawPhaseTrails(context, canvas) {
        if (!rawCsi.phaseHistory.length) {
            rawCsiDrawEmpty(context, canvas);
            return;
        }
        rawCsiClearCanvas(context, canvas);
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2 - 5;
        const radius = Math.min(canvas.width, canvas.height) * 0.36;
        context.strokeStyle = 'rgba(121, 105, 219, .22)';
        context.lineWidth = 1;
        [0.33, 0.66, 1].forEach((scale) => {
            context.beginPath();
            context.arc(centerX, centerY, radius * scale, 0, 2 * Math.PI);
            context.stroke();
        });
        context.beginPath();
        context.moveTo(centerX - radius, centerY);
        context.lineTo(centerX + radius, centerY);
        context.moveTo(centerX, centerY - radius);
        context.lineTo(centerX, centerY + radius);
        context.stroke();
        const centroids = RAW_CSI_SELECTED_SUBCARRIERS.map((_subcarrier, subcarrierIndex) => {
            let centroidReal = 0;
            let centroidImag = 0;
            rawCsi.phaseHistory.forEach((sample) => {
                centroidReal += sample[subcarrierIndex * 2];
                centroidImag += sample[subcarrierIndex * 2 + 1];
            });
            return {
                real: centroidReal / rawCsi.phaseHistory.length,
                imag: centroidImag / rawCsi.phaseHistory.length
            };
        });
        const amplifiedPoint = (subcarrierIndex, phase) => {
            const centroid = centroids[subcarrierIndex];
            let real = centroid.real
                + (phase[subcarrierIndex * 2] - centroid.real) * RAW_CSI_PHASE_TRAIL_GAIN;
            let imag = centroid.imag
                + (phase[subcarrierIndex * 2 + 1] - centroid.imag) * RAW_CSI_PHASE_TRAIL_GAIN;
            const magnitude = Math.hypot(real, imag);
            if (magnitude > 1.08) {
                real *= 1.08 / magnitude;
                imag *= 1.08 / magnitude;
            }
            return { real, imag };
        };
        RAW_CSI_SELECTED_SUBCARRIERS.forEach((subcarrier, subcarrierIndex) => {
            const hue = 188 + subcarrierIndex * 12;
            context.beginPath();
            rawCsi.phaseHistory.forEach((phase, historyIndex) => {
                const point = amplifiedPoint(subcarrierIndex, phase);
                const x = centerX + point.real * radius;
                const y = centerY - point.imag * radius;
                if (historyIndex === 0) context.moveTo(x, y);
                else context.lineTo(x, y);
            });
            context.strokeStyle = `hsla(${hue}, 92%, 68%, .62)`;
            context.lineWidth = 1.8;
            context.stroke();
            const latest = rawCsi.phaseHistory[rawCsi.phaseHistory.length - 1];
            const latestPoint = amplifiedPoint(subcarrierIndex, latest);
            const x = centerX + latestPoint.real * radius;
            const y = centerY - latestPoint.imag * radius;
            context.fillStyle = `hsl(${hue} 94% 70%)`;
            context.shadowColor = `hsl(${hue} 94% 60%)`;
            context.shadowBlur = 12;
            context.beginPath();
            context.arc(x, y, 5, 0, 2 * Math.PI);
            context.fill();
        });
        context.shadowBlur = 0;
        context.fillStyle = 'rgba(255, 255, 255, .5)';
        context.font = '12px "JetBrains Mono", monospace';
        context.textAlign = 'left';
        context.fillText('5× TRAIL SPREAD', 12, 20);
        context.textAlign = 'right';
        context.fillText('RELATIVE I', Math.min(canvas.width - 8, centerX + radius + 62), centerY + 4);
        context.textAlign = 'center';
        context.fillText('RELATIVE Q', centerX, centerY - radius - 18);
        context.fillText('CFO/STO-REDUCED PHASE · NOT POSITION', centerX, canvas.height - 16);
    }

    function rawCsiRender() {
        rawCsi.renderFrame = 0;
        const surface = rawCsiCanvasContext();
        if (!surface) return;
        const { canvas, context } = surface;
        if (rawCsi.visualization === 'channel-heatmap') rawCsiDrawHeatmap(context, canvas);
        else if (rawCsi.visualization === 'rf-waterfall') rawCsiDrawWaterfall(context, canvas);
        else if (rawCsi.visualization === 'channel-ghost') rawCsiDrawChannelGhost(context, canvas);
        else if (rawCsi.visualization === 'iq-constellation') rawCsiDrawIqConstellation(context, canvas);
        else if (rawCsi.visualization === 'phase-trails') rawCsiDrawPhaseTrails(context, canvas);
    }

    function rawCsiScheduleRender() {
        if (rawCsi.renderFrame) return;
        rawCsi.renderFrame = requestAnimationFrame(rawCsiRender);
    }

    function rawCsiSelectVisualization(value) {
        const visualization = RAW_CSI_VISUALIZATIONS[value]
            ? value : 'channel-heatmap';
        const metadata = RAW_CSI_VISUALIZATIONS[visualization];
        rawCsi.visualization = visualization;
        const select = $('.js-raw-visualization-select');
        const title = $('.js-raw-visualization-title');
        const description = $('.js-raw-visualization-description');
        const badge = $('.js-raw-visualization-badge');
        const canvas = $('.js-raw-visualization');
        if (select) select.value = visualization;
        if (title) title.textContent = metadata.title;
        if (description) description.textContent = metadata.description;
        if (badge) badge.textContent = metadata.badge;
        if (canvas) canvas.setAttribute('aria-label', metadata.ariaLabel);
        rawCsiScheduleRender();
    }

    function rawCsiConsumeRecord(record, streamSequence) {
        if (!record.byteLength) return;
        if (record.byteLength < RAW_CSI_V8_HEADER_BYTES) {
            throw new Error('Device sent an unsupported CSI record.');
        }
        const view = new DataView(record.buffer, record.byteOffset, record.byteLength);
        const headerLength = view.getUint8(3);
        const subcarriers = view.getUint16(10, true);
        const csiLength = view.getUint16(12, true);
        if (view.getUint16(0, true) !== 0x4353 || view.getUint8(2) !== 8
            || headerLength !== RAW_CSI_V8_HEADER_BYTES || csiLength !== subcarriers * 2
            || headerLength + csiLength > record.byteLength) {
            throw new Error('Device sent a malformed CSI V8 record.');
        }
        const expectedRecordSequence = streamSequence > 0xFFFFFFFFn
            ? 0xFFFFFFFF : Number(streamSequence);
        if (view.getUint32(6, true) !== expectedRecordSequence) {
            throw new Error('Device sent mismatched raw CSI sequence numbers.');
        }
        const amplitudes = new Float32Array(subcarriers);
        const iValues = new Float32Array(subcarriers);
        const qValues = new Float32Array(subcarriers);
        for (let index = 0, offset = headerLength;
            offset < headerLength + csiLength; index += 1, offset += 2) {
            // Espressif CSI stores each complex sample as [imaginary, real].
            qValues[index] = view.getInt8(offset);
            iValues[index] = view.getInt8(offset + 1);
            amplitudes[index] = Math.hypot(iValues[index], qValues[index]);
        }
        rawCsiCounter('.js-raw-rssi', view.getInt8(43));
        rawCsiCounter('.js-raw-channel', view.getUint8(42));
        const capturedTicksUs = Number(view.getBigUint64(22, true))
            || rawCsi.lastCaptureTicksUs + RAW_CSI_VISUAL_STEP_US;
        rawCsiIngestVisualFrame(amplitudes, iValues, qValues, capturedTicksUs);
    }

    function rawCsiAppend(chunk) {
        if (!rawCsi.parser) throw new Error('Raw CSI parser is not initialized.');
        rawCsi.parser.append(chunk).forEach((frame) => {
            rawCsiCounter('.js-raw-fresh', frame.freshRecordTotal);
            rawCsiCounter('.js-raw-dropped', frame.rawDropTotal);
            rawCsiCounter('.js-raw-backpressure', frame.sendBackpressureTotal);
            rawCsiConsumeRecord(frame.record, frame.streamSequence);
            rawCsiUpdatePacketRate(true);
        });
    }

    function rawCsiDemoFrame(targetPps, intervalMs, startedAtMs) {
        const elapsedSec = (performance.now() - startedAtMs) / 1000;
        const amplitudes = new Float32Array(64);
        const iValues = new Float32Array(64);
        const qValues = new Float32Array(64);
        const motion = 0.08 + conn.movement * 0.92;
        for (let index = 0; index < amplitudes.length; index += 1) {
            const channelShape = 34 + 8 * Math.sin(index * 0.31) + 5 * Math.cos(index * 0.13);
            const disturbance = motion * 18 * Math.sin(elapsedSec * 5.2 + index * 0.19);
            const amplitude = Math.max(4, channelShape + disturbance);
            const phase = index * 0.23 + elapsedSec * (0.7 + motion * 1.8);
            iValues[index] = Math.cos(phase) * amplitude;
            qValues[index] = Math.sin(phase) * amplitude;
            amplitudes[index] = amplitude;
        }
        rawCsiIngestVisualFrame(amplitudes, iValues, qValues, Math.round(performance.now() * 1000));
        rawCsi.demoFresh += Math.max(1, Math.round(targetPps * intervalMs / 1000));
        rawCsiCounter('.js-raw-pps', targetPps);
        rawCsiCounter('.js-raw-fresh', rawCsi.demoFresh);
        rawCsiCounter('.js-raw-rssi', Math.round(-50 + motion * 7));
        rawCsiCounter('.js-raw-channel', 6);
    }

    function rawCsiStartDemo(targetPps) {
        const intervalMs = Math.max(10, Math.round(1000 / targetPps));
        const startedAtMs = performance.now();
        rawCsi.demoFresh = 0;
        rawCsiResetVisualization();
        ['.js-raw-fresh', '.js-raw-dropped', '.js-raw-backpressure']
            .forEach((selector) => rawCsiCounter(selector, 0));
        rawCsiSetRunning(true);
        rawCsiStatus(`Streaming simulated CSI at ${targetPps} target packets/s.`);
        rawCsiDemoFrame(targetPps, intervalMs, startedAtMs);
        rawCsi.demoTimer = setInterval(
            () => rawCsiDemoFrame(targetPps, intervalMs, startedAtMs), intervalMs);
    }

    async function rawCsiStop() {
        const client = rawCsi.sessionClient;
        rawCsi.sessionClient = null;
        clearInterval(rawCsi.demoTimer);
        rawCsi.demoTimer = null;
        rawCsi.demoFresh = 0;
        rawCsi.controller?.abort('raw stream stopped');
        rawCsi.controller = null;
        rawCsiSetRunning(false);
        rawCsi.parser = null;
        rawCsi.packetArrivalTimes.length = 0;
        rawCsiCounter('.js-raw-pps', 0);
        if (client?.rawSessionId && client.connected) {
            try { await client.request('stop_raw_stream', {}, { timeoutMs: 3000 }); } catch (_error) { /* abort also releases the device session */ }
        }
    }

    async function rawCsiStart() {
        const client = directClient;
        if (rawCsi.running || conn.status !== 'connected') return;
        if (conn.mode === 'demo') {
            rawCsiStartDemo(100);
            return;
        }
        if (!rawCsiDirectReady() || client.capabilities?.features?.raw_csi !== true) return;
        rawCsiSetRunning(true);
        rawCsi.sessionClient = client;
        rawCsiStatus('Starting raw CSI stream…');
        try {
            const session = await client.request('start_raw_stream');
            rawCsi.parser = new window.ESPectreRawCsiV2Parser(session.session_id);
            rawCsiResetVisualization();
            const controller = new AbortController();
            rawCsi.controller = controller;
            const response = await fetch(client.rawEndpoint, {
                method: 'GET',
                headers: {
                    Accept: 'application/octet-stream',
                    Authorization: `Bearer ${session.session_id}`
                },
                cache: 'no-store',
                signal: controller.signal,
                targetAddressSpace: 'local'
            });
            if (!response.ok || !response.body) throw new Error(`Raw stream returned HTTP ${response.status}.`);
            rawCsiStatus('Streaming every classified CSI frame received from the configured traffic generator.');
            const reader = response.body.getReader();
            while (!controller.signal.aborted) {
                const { value, done } = await reader.read();
                if (done) break;
                rawCsiAppend(value);
            }
            if (!controller.signal.aborted) throw new Error('Raw stream ended unexpectedly.');
        } catch (error) {
            if (!rawCsi.controller?.signal.aborted) rawCsiStatus(error.message, true);
        } finally {
            if (rawCsi.running) await rawCsiStop();
        }
    }

    function rawCsiChooseDevice() {
        disconnect();
        directEndpointInput()?.focus();
    }

    function rawCsiInit() {
        $('.js-raw-csi-choose-device')?.addEventListener('click', rawCsiChooseDevice);
        $('.js-raw-csi-start')?.addEventListener('click', rawCsiStart);
        $('.js-raw-csi-stop')?.addEventListener('click', () => rawCsiStop());
        $('.js-raw-visualization-select')?.addEventListener('change', (event) => {
            rawCsiSelectVisualization(event.target.value);
        });
        const stage = $('.raw-csi-visualization-stage');
        if (stage && typeof ResizeObserver !== 'undefined') {
            rawCsi.resizeObserver = new ResizeObserver(rawCsiResizeVisualization);
            rawCsi.resizeObserver.observe(stage);
        } else {
            window.addEventListener('resize', rawCsiResizeVisualization);
        }
        rawCsiSelectVisualization(rawCsi.visualization);
        rawCsiResizeVisualization();
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
            if (route !== 'tool-game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
            if (document.activeElement !== canvas) return;
            gameDemoFlight(true, event);
        });
        document.addEventListener('keyup', (event) => {
            if (route !== 'tool-game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
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
        renderDirectBrowserGuidance();
        renderStoredDirectEndpoints();
        consumeDirectHandoff();

        $$('.js-connect-direct').forEach((btn) => btn.addEventListener('click', () => connectDirect({
            openView: btn.closest('espectre-direct-connect')?.dataset.openView
        })));
        $$('.js-direct-discover').forEach((btn) => btn.addEventListener('click', () => discoverLocalPeers(btn)));
        $$('.js-direct-discovery').forEach((panel) => panel.addEventListener('click', (event) => {
            const button = event.target.closest('.direct-discovery-device');
            if (!button?.dataset.endpoint) return;
            const input = button.closest('.device-connect-card')?.querySelector('input[list="direct-remembered-endpoints"]');
            if (input) input.value = button.dataset.deviceId;
            connectDirect({
                endpoint: button.dataset.endpoint,
                deviceId: button.dataset.deviceId,
                openView: button.closest('espectre-direct-connect')?.dataset.openView
            });
        }));
        $$('.js-start-detection').forEach((btn) => btn.addEventListener('click', () => {
            startDetection(btn.dataset.liveTransport || '');
        }));
        $('.js-header-connect').addEventListener('click', () => {
            selectMonitorTransport('direct');
            if (route === 'tool-monitor') {
                document.getElementById('monitor-direct-endpoint')?.focus();
                return;
            }
            pendingLiveDestination = '';
            location.hash = '#tool-monitor';
        });
        $$('.js-demo').forEach((btn) => btn.addEventListener('click', () => {
            connectDemo(btn.closest('espectre-connection-picker')?.dataset.openView || '');
        }));
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
        rawCsiInit();

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
        if (window.trackRouteView && route !== 'tool-raw-csi') {
            window.trackRouteView(route, { sendPageView: false });
        }
        if (conn.readyState) markToolReady(conn.readyState);
        if (monitor.readyState) markMonitorReady(monitor.readyState);
        if (conn.mode === 'direct' && directClient) cfgRefreshDevice();
        if (route === 'tool-flash') flashRefresh();
    });
    window.addEventListener('pagehide', (event) => {
        if (event.persisted) return;
        void rawCsiStop();
        reportGameAbandon('page_exit');
        if (conn.mode) teardownConnection('page_exit');
        else monitorStopAll('page_exit');
    });
    document.addEventListener('DOMContentLoaded', init);
})();
