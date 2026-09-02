/*
 * ESPectre - Device session
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    /* ==================================================== shared connection */

    const EVALUATION_INTERVAL_MS_DEFAULT = 250;
    const DIAGNOSTICS_POLL_INTERVAL_MS = 1000;
    const CSI_TARGET_PPS_DEFAULT = 100;
    const CONFIG_VERIFICATION_INITIAL_DELAY_MS = 250;
    const CONFIG_VERIFICATION_RETRY_MS = 1500;
    const CONFIG_VERIFICATION_MAX_ATTEMPTS = 4;
    const WIFI_BSSID_VERIFICATION_TIMEOUT_MS = 75 * 1000;
    const OTA_TRACKING_TIMEOUT_MS = 120000;
    const DIRECT_RECONNECT_DELAYS_MS = Object.freeze([500, 1500, 3000]);
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
    let demoSysinfoSnapshot = null;
    let demoInputEnergy = 0;
    let connectionCalloutTimer = null;
    let directCalloutVisible = false;
    const DIRECT_CALLOUT_DURATION_MS = 4000;
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
        const enteringDirectConnection = status === 'connected'
            && conn.status !== 'connected'
            && conn.mode === 'direct';
        if (status !== 'connected') clearDirectConnectionCallout();
        conn.status = status;
        if (enteringDirectConnection) showDirectConnectionCallout();
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
            if (route !== destination) window.navigateToRoute(destination);
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
            window.navigateToRoute(targetRoute);
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
        const detectorSelect = document.getElementById('sense-detector');
        const motionOnInput = document.getElementById('sense-motion-on');
        const motionOffInput = document.getElementById('sense-motion-off');
        const trafficGeneratorSelect = document.getElementById('sense-generator-mode');
        if (Number.isFinite(threshold)) {
            applyRemoteThreshold(threshold);
        }
        if (detector && detectorSelect) {
            detectorSelect.value = detector;
            syncSensingControls();
        }
        if (motionHits.length === 2 && motionOnInput && motionOffInput) {
            motionOnInput.value = motionHits[0];
            motionOffInput.value = motionHits[1];
        }
        if (snapshot.motion_on_hits !== undefined && motionOnInput) {
            motionOnInput.value = snapshot.motion_on_hits;
        }
        if (snapshot.motion_off_hits !== undefined && motionOffInput) {
            motionOffInput.value = snapshot.motion_off_hits;
        }
        if (snapshot.csi_traffic_mode) {
            conn.csiTrafficMode = snapshot.csi_traffic_mode;
            applyCsiTrafficModeSelect(snapshot.csi_traffic_mode);
        }
        if ((snapshot.traffic_mode || snapshot.traffic_generator_mode) && trafficGeneratorSelect) {
            trafficGeneratorSelect.value = snapshot.traffic_mode || snapshot.traffic_generator_mode;
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

    function csiTargetPps() {
        return conn.csiTargetPps || CSI_TARGET_PPS_DEFAULT;
    }

    function resetSensingCadence() {
        conn.evaluationIntervalMs = 0;
        conn.csiTargetPps = 0;
    }

    function applySensingCadence(snapshot) {
        if (!snapshot || typeof snapshot !== 'object') return;
        const evaluation = positiveInt(snapshot.evaluation_interval_ms);
        const pps = positiveInt(snapshot.csi_target_pps);
        if (evaluation) conn.evaluationIntervalMs = evaluation;
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
        $$('.live-sensing-controls details.config-advanced').forEach((details) => {
            details.hidden = ![...details.querySelectorAll('[data-device-command]')]
                .some((panel) => !panel.hidden);
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
            button.textContent = monitor.calibrating ? 'Calibrating…' : 'Calibrate';
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
        const labels = { native: 'Native', esphome: 'ESPHome', matter: 'Matter', micro: 'Micro-ESPectre' };
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
            nameNote.textContent = 'The installed firmware does not allow the device name to be changed here.';
        }
        if (wifiNote) {
            wifiNote.textContent = esphome || matter
                ? 'You can view Wi-Fi status and choose an access point here, but this firmware manages the network name and password elsewhere.'
                : 'The installed firmware does not make Wi-Fi details available here.';
        }
        if (mqttNote) {
            mqttNote.textContent = esphome
                ? 'This ESPHome firmware manages MQTT in its ESPHome configuration. Change it through the ESPHome API or the adopted YAML file.'
                : 'The installed firmware does not allow MQTT settings to be changed here.';
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
        if (!conn.mode || conn.status !== 'connected') return;
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
            window.gameSetFlight(window.gameSensingActive());
            window.gameStartPreview();
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
            toast('Motion threshold must be between 0 and 1.');
            return;
        }
        applyLocalThreshold(threshold);
        runSensingCommand(
            { command: 'set_threshold', threshold },
            'Saving motion threshold…',
            'Motion threshold updated.',
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
        if (typeof window.gameOnTelemetry === 'function') window.gameOnTelemetry();
    }

    /* --------------------------------------------------------- Direct mode */

    async function startDetection(preferredTransport = '') {
        rememberLiveDestination();
        if (conn.mode === 'demo') {
            completeLiveConnectionNavigation();
            return;
        }
        if (conn.mode === 'direct' && directClient?.connected) {
            try {
                if (directSupportsCommand('set_sensing')) {
                    await directClient.request('set_sensing', { enabled: true });
                }
                setDeviceView('live');
                completeLiveConnectionNavigation();
                toast('Live motion started.');
            } catch (error) {
                console.warn('Could not start live motion:', error);
                toast('Live motion could not start. Check the connection and try again.');
            }
            return;
        }
        if (preferredTransport === 'direct') {
            selectMonitorTransport('direct');
            window.navigateToRoute('tool-monitor');
            return;
        }
        selectMonitorTransport('direct');
        window.navigateToRoute('tool-monitor');
    }

    function applySysinfo(snapshot) {
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
        if (snapshot.csi_profile !== undefined) {
            const profile = String(snapshot.csi_profile || '').trim().toLowerCase();
            set('cfg-wifi-phy', profile ? profile.toUpperCase() : 'Unknown');
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
            const mqttPresetSelect = document.getElementById('cfg-mqtt-preset');
            if (mqttPresetSelect) {
                mqttPresetSelect.value = mqttPreset;
                applyMqttPresetFieldLocks('configure', MQTT_PRESETS[mqttPreset].configure);
                applyConfigureMqttCredentialPolicy(mqttPreset);
            }
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
            demoSysinfoSnapshot = {
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
                evaluation_interval_ms: '250',
                wifi_connected: 'true',
                mqtt_connected: 'true',
                wifi_configured: 'true',
                wifi_ssid: 'HomeNet',
                wifi_band: '5g',
                wifi_channel: '48',
                csi_profile: 'vht20',
                wifi_bssid: '',
                mqtt_host: 'homeassistant.local',
                mqtt_port: '1883',
                mqtt_username: 'mqtt',
                topic_prefix: 'espectre/v1/devices',
                device_id: '3cf79180d3a0aca4',
                device_name: 'Demo Device',
                device_label: 'Demo Device',
                motion_hits: '4/3'
            };
            applySysinfo(demoSysinfoSnapshot);
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

    /* ----------------------------------------------------- shared teardown */

    function disconnect() {
        cancelDirectDiscovery({ clear: true });
        cancelDirectReconnect();
        if (typeof window.rawCsiStop === 'function') void window.rawCsiStop('user');
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
        if (typeof window.rawCsiStop === 'function') void window.rawCsiStop(reason);
        monitor.switchingTransport = false;
        if (otaTracking) finishOtaTracking('unconfirmed', 'ClientDisconnected');
        if (pendingConfigVerification) {
            finishConfigVerification('unconfirmed', 'ClientDisconnected');
        }
        if (typeof window.reportGameAbandon === 'function') {
            window.reportGameAbandon('disconnect');
        }
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
        demoSysinfoSnapshot = null;
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
        if (typeof window.gameReset === 'function') window.gameReset();
        if (typeof window.thereminStop === 'function') window.thereminStop();
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
                ? 'USB'
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
        if (typeof window.rawCsiUseConnection === 'function') window.rawCsiUseConnection();
        if (connectivitySetup) connectivitySetup.hidden = !(directSetup || conn.mode === 'demo');
        if (startSensing) startSensing.disabled = monitor.switchingTransport;
        if (edit) {
            edit.hidden = false;
            edit.disabled = false;
            edit.textContent = 'Device settings';
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
        });

        syncSensingControls();
        syncDiagnosticsPolling();
        syncFirmwareUpdateNotice();
        renderBrowserSupport();
        renderTelemetry();
        syncConnectionCallout();
        if (live && route === 'tool-game' && typeof window.gameResizeCanvas === 'function') {
            requestAnimationFrame(window.gameResizeCanvas);
        }
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
