/*
 * ESPectre - Monitor tool
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    /* ============================================================= monitor */

    const MONITOR_CHART_WINDOW_MS = 60 * 1000;
    const MONITOR_CALIBRATION_FALLBACK_MS = 45 * 1000;
    const MONITOR_CALIBRATION_SAFETY_MS = 90 * 1000;
    const MONITOR_DEMO_CALIBRATION_MS = 2500;
    // Menu entry that reveals the typed MAC field; never sent to the device.
    const RAW_TARGET_MANUAL = 'manual';

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
        sensingFields: null,
        switchingTransport: false,
        diagTimer: null,
        diagIntervalMs: 0,
        diagRequestPending: false,
        calibrating: false,
        calibrationTimer: null,
        rawTargetSavePending: null,
        rawTargetDiscovery: null
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
        const session = directResourceSessions.get(directClient);
        if (session) {
            session.faults.length = 0;
            session.motion = null;
        }
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
        resetMonitorReportView();
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
        if (stateEl) {
            stateEl.textContent = motion ? 'MOTION' : 'IDLE';
            stateEl.classList.toggle('motion', motion);
        }
        const movementEl = $('.js-mon-move');
        if (movementEl) movementEl.textContent = movement.toFixed(3);
        monitorQueueChart();
    }

    function monitorStat(value, digits, suffix) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
        return Number(value).toFixed(digits) + suffix;
    }

    function monitorSetStat(selector, value, digits, suffix) {
        const element = $(selector);
        if (element) element.textContent = monitorStat(value, digits, suffix);
    }

    function monitorStats(data) {
        monitorSetStat('.js-mon-traffic', data.traffic_tx_pps, 1, ' pps');
        monitorSetStat('.js-mon-traffic-rx', data.traffic_rx_pps, 1, ' pps');
        monitorSetStat('.js-mon-callbacks', data.csi_callback_pps, 1, ' pps');
        monitorSetStat('.js-mon-accepted', data.csi_accepted_pps, 1, ' pps');
        monitorSetStat('.js-mon-occupancy', data.csi_occupancy == null ? null : data.csi_occupancy * 100, 1, '%');
        monitorSetStat('.js-mon-rssi', data.wifi_rssi_dbm, 0, ' dBm');
        monitorSetStat('.js-mon-heap', data.free_memory_kb, 1, ' KiB');
        monitorSetStat('.js-mon-loop', data.loop_time_ms, 2, ' ms');
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
        ctx.font = '10px ui-monospace, "SFMono-Regular", Consolas, monospace';
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
        if (!canvas) return;
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
        pendingMessage = 'Saving setting…',
        statusFn = monitorStatus,
        timeoutMs = 8000
    } = {}) {
        if (conn.mode === 'direct' && directClient?.connected) {
            const { command, ...params } = fields;
            statusFn(pendingMessage);
            if (command === 'read_diagnostics') {
                return directClient.request('get', 'diagnostics', params, { timeoutMs });
            }
            if (command === 'recalibrate') {
                return directClient.request('post', 'sensing/calibrations', {}, { timeoutMs });
            }
            if (command === 'check_ota') {
                return directClient.request('post', 'ota/checks', params, { timeoutMs });
            }
            if (command === 'start_ota') {
                return directClient.request('post', 'ota/updates', params, { timeoutMs });
            }
            return directClient.request('patch', 'sensing', params, { timeoutMs });
        }
        const error = new Error('Connect to the device before changing its settings.');
        statusFn(error.message);
        return Promise.reject(error);
    }

    function reportSensingChange(action, result, error) {
        track('sensing_change', {
            ...connectionParams(),
            transport: connectionTransport(),
            input_mode: connectionInputMode(),
            action,
            result,
            ...(error ? { error_type: errorType(error) } : {})
        });
    }

    function diagnosticsRequestPending() {
        return conn.mode === 'direct' && monitor.diagRequestPending;
    }

    function diagnosticsPanelOpen() {
        const panel = $('.device-live-diagnostics');
        const workspace = $('.js-monitor-workspace');
        return !!(panel && panel.open && !document.hidden
            && route === 'tool-monitor' && workspace && !workspace.hidden);
    }

    function stopDiagnosticsPolling() {
        if (!monitor.diagTimer) return;
        clearInterval(monitor.diagTimer);
        monitor.diagTimer = null;
        monitor.diagIntervalMs = 0;
    }

    function syncDiagnosticsPolling() {
        const canPoll = !occupancyComparison.busy() && diagnosticsPanelOpen()
            && (conn.mode === 'demo' || conn.mode === 'direct');
        if (!canPoll) {
            stopDiagnosticsPolling();
            return;
        }
        const interval = DIAGNOSTICS_POLL_INTERVAL_MS;
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
            const targetPps = csiTargetPps();
            monitorStats({
                traffic_tx_pps: targetPps + 5,
                traffic_rx_pps: targetPps + 3,
                csi_callback_pps: Math.max(1, targetPps - 4),
                csi_accepted_pps: Math.max(1, targetPps - 10),
                csi_occupancy: Math.max(1, targetPps - 16) / targetPps,
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
            const response = await monitorPublishCommand({ command: 'read_diagnostics', fields: [
                'traffic_tx_pps', 'traffic_rx_pps', 'csi_callback_pps', 'csi_accepted_pps',
                'csi_occupancy', 'wifi_rssi_dbm', 'free_memory_kb', 'loop_time_ms'
            ] }, {
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
            reportSensingChange('recalibrate', 'accepted');
            toast(result.message || 'Calibration started. Keep the room still.');
            scheduleCalibrationIdle(MONITOR_CALIBRATION_FALLBACK_MS);
        } catch (error) {
            console.warn('Calibration failed:', error);
            reportSensingChange('recalibrate', 'failure', error);
            toast('Calibration could not start. Check the connection and try again.');
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
            reportSensingChange(fields.command, 'accepted');
            toast(result.message || successMessage);
        } catch (error) {
            console.warn('Sensing setting update failed:', error);
            reportSensingChange(fields.command, 'failure', error);
            toast('The setting could not be saved. Check the connection and try again.');
        }
    }

    function selectMonitorTransport(mode) {
        document.querySelector('espectre-connection-picker[data-surface="monitor"]')?.select(mode);
    }

    function syncRawTrafficTargetControls() {
        const refresh = document.getElementById('sense-raw-target-refresh');
        if (refresh) {
            const scanning = monitor.rawTargetDiscovery !== null;
            refresh.disabled = scanning || (conn.mode !== 'demo' && (!directClient?.connected
                || !directClient.capabilities?.resources?.includes('devices')));
            refresh.classList.toggle('is-scanning', scanning);
            refresh.setAttribute('aria-busy', String(scanning));
        }
        const manual = document.getElementById('sense-raw-target-manual');
        if (manual) {
            manual.hidden = document.getElementById('sense-raw-target')?.value !== RAW_TARGET_MANUAL;
            if (manual.hidden) {
                manual.value = '';
                manual.setAttribute('aria-invalid', 'false');
            }
        }
    }

    // The device rejects its own MAC; group and zero addresses can never answer.
    function normalizeRawTargetMac(text) {
        const octets = String(text).trim().toLowerCase().split(/[:-]/);
        if (octets.length !== 6 || octets.some((octet) => !/^[0-9a-f]{2}$/.test(octet))) return null;
        if (parseInt(octets[0], 16) & 1 || octets.every((octet) => octet === '00')) return null;
        return octets.join(':');
    }

    function rawTrafficTargetValue() {
        return document.getElementById('sense-raw-target')?.value || 'auto';
    }

    function renderRawTrafficTargets(peers, target = rawTrafficTargetValue()) {
        const select = document.getElementById('sense-raw-target');
        if (!select) return;
        const options = [new Option('Automatic (AP)', 'auto')];
        // Results live only in this menu; each click replaces them with a fresh search.
        if (peers === undefined) {
            options.push(...[...select.options].filter((option) => option.dataset.discovered === 'true'));
        } else {
            const seen = new Set();
            for (const peer of peers) {
                const mac = peer.mac_address;
                if (peer.device_id === conn.deviceId || !mac || seen.has(mac)) continue;
                seen.add(mac);
                const option = new Option(`${peer.name || `ESPectre ${peer.device_id.slice(-6)}`} · ${mac}`, mac);
                option.dataset.discovered = 'true';
                options.push(option);
            }
        }
        // While a MAC is being typed, the saved destination stays listed.
        const kept = target === RAW_TARGET_MANUAL ? select.dataset.configuredTarget : target;
        if (kept && !options.some((option) => option.value === kept)) options.push(new Option(kept, kept));
        options.push(new Option('Enter a MAC address…', RAW_TARGET_MANUAL));
        select.replaceChildren(...options);
        select.value = target;
        syncRawTrafficTargetControls();
    }

    function cancelRawTrafficDiscovery({ clear = true, keepManual = false } = {}) {
        monitor.rawTargetDiscovery?.abort();
        monitor.rawTargetDiscovery = null;
        if (clear) {
            const select = document.getElementById('sense-raw-target');
            // Hiding the page must not drop a MAC the user left to copy from another tab.
            const typing = keepManual && select?.value === RAW_TARGET_MANUAL;
            if (select && typing) {
                renderRawTrafficTargets([]);
            } else if (select) {
                select.dataset.dirty = '';
                renderRawTrafficTargets([], select.dataset.configuredTarget || 'auto');
            }
        }
        syncRawTrafficTargetControls();
    }

    async function monitorRefreshRawTrafficTargets() {
        const client = directClient;
        const deviceId = conn.deviceId;
        const mode = conn.mode;
        if (monitor.rawTargetDiscovery || document.hidden || route !== 'tool-monitor') return;
        if (mode !== 'demo' && (!client?.connected || !client.capabilities?.resources?.includes('devices'))) return;
        const controller = new AbortController();
        monitor.rawTargetDiscovery = controller;
        const current = () => monitor.rawTargetDiscovery === controller && !controller.signal.aborted
            && directClient === client && conn.deviceId === deviceId && conn.mode === mode
            && route === 'tool-monitor' && !document.hidden;
        renderRawTrafficTargets([]);
        try {
            let peers;
            if (mode === 'demo') {
                await new Promise((resolve) => setTimeout(resolve, 700));
                peers = [{ device_id: '2222222222222222', name: 'Demo sensor', mac_address: '02:11:22:33:44:55' }];
            } else {
                const result = DirectProtocolClient.validatePeerDiscoveryResult(await client.request('get', 'devices', {}, {
                    signal: controller.signal, timeoutMs: 10000, retryTransportErrors: false
                }));
                peers = result.devices;
            }
            if (!current()) return;
            renderRawTrafficTargets(peers);
            const select = document.getElementById('sense-raw-target');
            if (![...(select?.options || [])].some((option) => option.dataset.discovered === 'true')) {
                toast('No other ESPectre sensor with a station MAC was found.');
            }
        } catch (error) {
            if (!current()) return;
            console.warn('Device discovery failed:', error);
            toast('The search failed. Check the connection and try again.');
        } finally {
            if (monitor.rawTargetDiscovery === controller) {
                monitor.rawTargetDiscovery = null;
                syncRawTrafficTargetControls();
            }
        }
    }

    function applyRawTrafficSettings(snapshot) {
        const select = document.getElementById('sense-raw-target');
        if (!select || snapshot.traffic_generator_target_mac === undefined) return;
        const target = String(snapshot.traffic_generator_target_mac).toLowerCase();
        if (select.dataset.deviceId !== conn.deviceId) {
            cancelRawTrafficDiscovery({ clear: false });
            select.dataset.dirty = '';
            renderRawTrafficTargets([], target);
        }
        select.dataset.deviceId = conn.deviceId;
        select.dataset.configuredTarget = target;
        const draft = rawTrafficTargetValue();
        if (!select.dataset.dirty || draft === target) {
            renderRawTrafficTargets(undefined, target);
            select.dataset.dirty = '';
        }
        syncRawTrafficTargetControls();
        evaluateConfigVerification({ traffic_generator_target_mac: target });
    }

    async function monitorChangeRawTarget() {
        const select = document.getElementById('sense-raw-target');
        const selection = select.value;
        if (!selection) return false;
        select.dataset.dirty = 'true';
        syncRawTrafficTargetControls();
        const manual = document.getElementById('sense-raw-target-manual');
        let target = selection;
        if (selection === RAW_TARGET_MANUAL) {
            if (!manual?.value.trim()) {
                manual?.focus();
                return false;
            }
            target = normalizeRawTargetMac(manual.value);
            if (!target) {
                manual.setAttribute('aria-invalid', 'true');
                toast('Enter a device MAC address, such as 02:11:22:33:44:55.');
                return false;
            }
            manual.setAttribute('aria-invalid', 'false');
        }
        const client = directClient;
        const deviceId = conn.deviceId;
        const mode = conn.mode;
        const sameSession = () => directClient === client && conn.deviceId === deviceId && conn.mode === mode;
        // Preserve edit order when a previous autosave is still in flight.
        const saving = (monitor.rawTargetSavePending || Promise.resolve()).then(async () => {
            if (!sameSession()) return false;
            const ok = await cfgApply('update_sensing', 'Raw Wi-Fi destination saved.',
                'patch', 'sensing', { traffic_generator_target_mac: target },
                (snapshot) => String(snapshot.traffic_generator_target_mac).toLowerCase() === target);
            if (!sameSession()) return false;
            if (ok && mode === 'demo') demoSysinfoSnapshot.traffic_generator_target_mac = target;
            if (ok) select.dataset.configuredTarget = target;
            if (ok && select.value === selection) {
                select.dataset.dirty = '';
                renderRawTrafficTargets(undefined, target);
            }
            return ok;
        });
        monitor.rawTargetSavePending = saving.catch(() => false);
        return saving;
    }

    function monitorChangeSampleSpacing() {
        const csiSampleSpacing = document.getElementById('sense-sample-spacing').value;
        return runSensingCommand(
            { command: 'update_sensing', csi_sample_spacing: csiSampleSpacing },
            'Saving packet spacing…',
            'Packet spacing updated.',
            () => { demoSysinfoSnapshot.csi_sample_spacing = csiSampleSpacing; }
        );
    }

    /* ============================================== diagnostic report */

    let monitorReportSampling = null;
    let monitorReportJson = '';

    function calculateRateStats(values) {
        const numbers = (values || [])
            .filter((v) => typeof v === 'number' && Number.isFinite(v))
            .sort((a, b) => a - b);
        if (!numbers.length) return { min: null, median: null, max: null };
        const min = numbers[0];
        const max = numbers[numbers.length - 1];
        const mid = Math.floor(numbers.length / 2);
        const median = numbers.length % 2 !== 0 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) / 2;
        return {
            min: Number(min.toFixed(4)),
            median: Number(median.toFixed(4)),
            max: Number(max.toFixed(4))
        };
    }

    function isMulticastOrBroadcastMac(mac) {
        return (parseInt(mac.slice(0, 2), 16) & 1) === 1;
    }

    function isMulticastOrLoopbackIp(ip) {
        if (ip.startsWith('127.') || ip === '::1' || ip === '0.0.0.0' || ip === '255.255.255.255') return true;
        const first = parseInt(ip.split('.')[0], 10);
        if (first >= 224 && first <= 239) return true;
        return false;
    }

    function redactDiagnosticReport(data) {
        if (!data || typeof data !== 'object') return data;

        const terms = new Map();
        for (const [value, replacement] of [
            [data.wifi?.ssid, '<redacted_ssid>'],
            [data.wifi?.sta_ssid, '<redacted_ssid>'],
            [data.wifi?.ap_ssid, '<redacted_ssid>'],
            [data.device?.name, '<redacted_name>'],
            [data.device?.label, '<redacted_label>'],
            [data.device?.hostname, '<redacted_hostname>'],
            [data.device?.device_id, '<redacted_id>']
        ]) {
            if (typeof value !== 'string' || !value.length) continue;
            const term = value.toLowerCase();
            if (!terms.has(term)) terms.set(term, replacement);
        }

        const secretKey = /password|passphrase|secret|token|api_key|private_key|credentials/i;
        function collectSecrets(obj, withinSecret = false) {
            for (const [key, val] of Object.entries(obj)) {
                const secret = withinSecret || secretKey.test(key);
                if (secret && typeof val === 'string' && val) terms.set(val.toLowerCase(), '[REDACTED]');
                if (val && typeof val === 'object') collectSecrets(val, secret);
            }
        }
        collectSecrets(data);

        // Match once, longest identity first, so short names cannot alter a longer match or a placeholder.
        const identities = [...terms.keys()].sort((a, b) => b.length - a.length)
            .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        const identityAvailable = Boolean(data.device?.name && (data.wifi?.ssid || data.wifi?.sta_ssid));
        const patterns = [
            /(?<ipv6>(?<![\da-f:])(?:[\da-f]{0,4}:){2,}[\da-f:]*(?:\.[\da-f]+)*(?:%[\w-]+(?:\.[\w-]+)*)?)(?![\da-f:])/.source,
            /(?<mac>\b(?:[\da-f]{2}[:-]){5}[\da-f]{2}\b)/.source,
            /(?<ip>\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b)/.source,
            /(?<hostname>\b(?:[a-z\d][-a-z\d]*\.)+local\b)/.source
        ];
        const addresses = new RegExp('^(?:' + patterns.join('|') + ')$', 'i');
        const sensitive = new RegExp([...patterns, ...identities].join('|'), 'gi');
        const replacement = (match, ipv6, mac, ip, hostname) => {
            if (ipv6) {
                // A colon-separated MAC or timestamp can also match the IPv6 candidate.
                if (/^(?:[\da-f]{2}:){5}[\da-f]{2}$/i.test(ipv6)) mac = ipv6;
                else {
                    const address = ipv6.split('%')[0];
                    try {
                        const host = new URL('http://[' + address + ']/').hostname;
                        return host === '[::1]' || host === '[::]' || host.startsWith('[ff')
                            ? address : '<redacted_ipv6>';
                    } catch {
                        if (address.includes('::') || address.split(':').length >= 8) return '<redacted_ipv6>';
                        return terms.get(match.toLowerCase()) ?? match;
                    }
                }
            }
            if (mac) {
                if (isMulticastOrBroadcastMac(mac)) return mac;
                return mac.slice(0, 5) + (mac[2] + '**').repeat(4);
            }
            if (ip) return isMulticastOrLoopbackIp(ip) ? ip : ip.split('.').slice(0, 2).join('.') + '.x.x';
            if (hostname) return '<redacted_hostname>.local';
            return terms.get(match.toLowerCase());
        };

        // These fields contain protocol tokens or report metadata even if a user picks the same name.
        const structuredStrings = new Set([
            'capabilities.resources', 'capabilities.events', 'capabilities.protocol_version',
            'capabilities.operations.name', 'capabilities.operations.method', 'capabilities.operations.path',
            'capabilities.operations.fields', 'sampling.status', 'timestamp', 'fault_events.timestamp',
            'device.firmware_version', 'diagnostics.esp_idf_version', 'diagnostics.firmware_build_date',
            'diagnostics.recent_logs.level', 'last_motion.state',
            'device.chip', 'device.frontend', 'device.firmware', 'device.csi_profile', 'health.status',
            'wifi.band', 'wifi.apply_state', 'sensing.mode', 'sensing.detector', 'sensing.traffic_generator_mode',
            'sensing.csi_sample_spacing'
        ]);

        const transform = (key, val, path) => {
            const lowerKey = key.toLowerCase();

            if (lowerKey === 'credentials' || lowerKey.endsWith('_credentials')) {
                return undefined;
            }

            if (secretKey.test(lowerKey)) {
                if (typeof val === 'boolean') return val;
                return val == null ? val : '[REDACTED]';
            }

            if (path === 'device.name' && typeof val === 'string') return '<redacted_name>';
            if (path === 'device.label' && typeof val === 'string') return '<redacted_label>';
            if (lowerKey === 'hostname' && typeof val === 'string') return '<redacted_hostname>';
            if (['ssid', 'sta_ssid', 'ap_ssid'].includes(lowerKey) && typeof val === 'string') return '<redacted_ssid>';

            if (lowerKey === 'device_id' && typeof val === 'string') {
                return val.length > 8 ? `${val.slice(0, 4)}...${val.slice(-4)}` : '<redacted_id>';
            }

            if (typeof val === 'string') {
                if (structuredStrings.has(path)) return val;
                if (!identityAvailable && !addresses.test(val)) return '[REDACTED]';
                return val.replace(sensitive, replacement);
            }

            if (Array.isArray(val)) {
                return val.map((item) => transform(key, item, path)).filter((item) => item !== undefined);
            }

            if (val && typeof val === 'object') {
                const result = {};
                for (const [k, v] of Object.entries(val)) {
                    const transformed = transform(k, v, path ? path + '.' + k : k);
                    if (transformed !== undefined) result[k] = transformed;
                }
                return result;
            }

            return val;
        };

        return transform('', data, '');
    }

    function monitorCancelDiagnosticReport({ silent = false } = {}) {
        if (!monitorReportSampling) return;
        monitorReportSampling.cancel(silent);
    }

    function resetMonitorReportView() {
        monitorCancelDiagnosticReport({ silent: true });
        monitorReportJson = '';
        const progressEl = $('.js-mon-report-progress');
        const resultEl = $('.js-mon-report-result');
        const previewEl = $('.js-mon-report-preview');
        const startBtn = $('.js-mon-report-start');
        if (progressEl) progressEl.hidden = true;
        if (resultEl) resultEl.hidden = true;
        if (previewEl) previewEl.textContent = '';
        if (startBtn) startBtn.disabled = false;
    }

    function monitorStartDiagnosticReport({
        sampleDurationSec = 30,
        sampleIntervalMs = 1000
    } = {}) {
        if (monitorReportSampling || occupancyComparison.busy()) return Promise.resolve(null);
        const mode = conn.mode;
        const direct = mode === 'direct';
        const client = directClient;
        if (!direct && mode !== 'demo') {
            toast('Connect to a device before generating a diagnostic report.');
            return Promise.resolve(null);
        }
        if (direct && !client?.connected) {
            toast('Direct HTTP connection is not ready. Connect and try again.');
            return Promise.resolve(null);
        }

        const session = direct ? directResourceSession(client) : null;
        monitorReportJson = '';
        const startBtn = $('.js-mon-report-start');
        if (startBtn) startBtn.disabled = true;

        const progressEl = $('.js-mon-report-progress');
        const trackEl = $('.js-mon-report-track');
        const barEl = $('.js-mon-report-bar');
        const statusEl = $('.js-mon-report-status');
        const resultEl = $('.js-mon-report-result');
        const previewEl = $('.js-mon-report-preview');
        const copyBtn = $('.js-mon-report-copy');

        if (progressEl) progressEl.hidden = false;
        if (resultEl) resultEl.hidden = true;
        if (barEl) barEl.style.width = '0%';
        if (trackEl) trackEl.setAttribute('aria-valuenow', '0');
        if (statusEl) statusEl.textContent = `Sampling diagnostics (0/${sampleDurationSec}s)…`;

        const samples = {
            csi_occupancy: [],
            csi_accepted_pps: [],
            csi_filtered_pps: [],
            csi_hw_error_pps: [],
            csi_callback_pps: [],
            generator_pps: [],
            wifi_rssi_dbm: []
        };

        let startedAt = Date.now();
        const durationMs = sampleDurationSec * 1000;
        let samplesCollected = 0;
        const controller = new AbortController();

        return new Promise((resolve) => {
            const samplingState = {
                timer: null,
                cancel(silent = false) {
                    if (samplingState.timer) clearTimeout(samplingState.timer);
                    controller.abort();
                    monitorReportSampling = null;
                    occupancyComparison.sync();
                    if (progressEl) progressEl.hidden = true;
                    if (startBtn) {
                        startBtn.disabled = false;
                        if (typeof startBtn.focus === 'function') startBtn.focus();
                    }
                    if (!silent) toast('Diagnostic report cancelled.');
                    resolve(null);
                }
            };
            monitorReportSampling = samplingState;
            occupancyComparison.sync();

            const isCurrent = () => {
                if (monitorReportSampling !== samplingState) return false;
                if (conn.mode === mode && (!direct || (client.connected && directClient === client
                    && directResourceSessions.get(client) === session))) return true;
                samplingState.cancel(true);
                toast('Device disconnected while sampling diagnostics.');
                return false;
            };

            let available;
            let sampleFields = Object.keys(samples);
            const readResource = async (name, params = {}) => {
                try {
                    return await client.request('get', name, params, {
                        timeoutMs: 5000,
                        signal: controller.signal
                    });
                } catch {
                    return { error: 'unavailable' };
                }
            };

            const pollSample = async () => {
                if (!isCurrent()) return;
                if (direct && !available) {
                    const catalog = await readResource('diagnostics');
                    if (!isCurrent()) return;
                    available = new Set(Array.isArray(catalog?.fields) ? catalog.fields.map(f => f.name) : []);
                    sampleFields = available.size ? sampleFields.filter(field => available.has(field))
                        : ['csi_occupancy', 'csi_accepted_pps', 'csi_filtered_pps', 'csi_hw_error_pps'];
                    startedAt = Date.now();
                }
                const pollStartedAt = Date.now();
                let values;
                if (direct) {
                    try {
                        if (sampleFields.length) values = await client.request('get', 'diagnostics', { fields: sampleFields }, {
                            timeoutMs: Math.min(2000, sampleIntervalMs), signal: controller.signal
                        });
                    } catch {
                        // Keep sampling after a missed read.
                    }
                    if (!isCurrent()) return;
                } else {
                    const targetPps = csiTargetPps();
                    values = {
                        csi_accepted_pps: Math.max(1, targetPps - 10),
                        csi_occupancy: Math.max(1, targetPps - 16) / targetPps,
                        csi_filtered_pps: 0,
                        csi_hw_error_pps: 0
                    };
                }
                let collected = false;
                for (const [field, readings] of Object.entries(samples)) {
                    if (Number.isFinite(values?.[field])) {
                        readings.push(values[field]);
                        if (field.startsWith('csi_')) collected = true;
                    }
                }
                if (collected) samplesCollected++;
                const elapsedMs = Date.now() - startedAt;
                const pct = Math.min(100, Math.round(100 * elapsedMs / durationMs));
                if (barEl) barEl.style.width = `${pct}%`;
                if (trackEl) trackEl.setAttribute('aria-valuenow', String(pct));
                if (statusEl) statusEl.textContent = `Sampling diagnostics (${Math.min(Math.floor(elapsedMs / 1000), sampleDurationSec)}/${sampleDurationSec}s)…`;

                const remainingMs = durationMs - elapsedMs;
                if (remainingMs <= 0) {
                    await finishReport();
                } else {
                    const untilNextPoll = Math.max(0, sampleIntervalMs - (Date.now() - pollStartedAt));
                    samplingState.timer = setTimeout(pollSample, Math.min(untilNextPoll, remainingMs));
                }
            };

            const finishReport = async () => {
                if (monitorReportSampling !== samplingState) return;
                if (statusEl) statusEl.textContent = 'Reading device resources…';

                let capabilities = {};
                let device = {};
                let health = {};
                let sensing = {};
                let wifi = {};
                const diagnostics = {};
                const reportFields = [
                    'wifi_channel', 'free_memory_kb', 'minimum_free_memory_kb', 'loop_time_ms',
                    'csi_callbacks_total', 'csi_accepted_total', 'csi_filtered_total', 'csi_hw_error_total',
                    'csi_admitted_pps', 'csi_missing_slots_pps', 'csi_excess_pps', 'csi_stale_pps'
                ];

                if (direct) {
                    [capabilities, device, health, sensing, wifi] = await Promise.all([
                        readResource('capabilities'),
                        readResource('device'),
                        readResource('health'),
                        readResource('sensing'),
                        readResource('wifi')
                    ]);
                    if (!isCurrent()) return;
                    const selected = reportFields.filter(field => available.has(field));
                    for (let i = 0; i < selected.length; i += 4) {
                        const fields = selected.slice(i, i + 4);
                        const values = await readResource('diagnostics', { fields });
                        if (!isCurrent()) return;
                        for (const field of fields) diagnostics[field] = values?.[field] ?? null;
                    }
                } else if (mode === 'demo') {
                    capabilities = { protocol_version: '1.0', resources: ['device', 'health', 'sensing', 'wifi', 'diagnostics'] };
                    device = {
                        name: 'Demo sensor',
                        device_id: conn.deviceId || '2222222222222222',
                        chip: 'esp32s3',
                        firmware_version: '1.0.0'
                    };
                    health = { status: 'healthy', uptime: 3600, free_memory_kb: 161.4 };
                    sensing = { detector: 'lightweight', csi_target_pps: csiTargetPps(), motion_on_hits: 3, motion_off_hits: 5 };
                    wifi = { ssid: 'DemoNetwork', sta_ip: '192.168.1.50', sta_mac: '30:ae:a4:07:0a:24', wifi_channel: 6, password_configured: true };
                }

                const comparison = occupancyComparison.report();
                const rawReport = {
                    timestamp: new Date().toISOString(),
                    sampling: {
                        duration_seconds: sampleDurationSec,
                        status: samplesCollected ? 'complete' : 'no_samples',
                        samples_collected: samplesCollected,
                        rates: Object.fromEntries(Object.entries(samples)
                            .filter(([name]) => name !== 'wifi_rssi_dbm')
                            .map(([name, readings]) => [name, calculateRateStats(readings)])),
                        wifi_rssi_dbm: calculateRateStats(samples.wifi_rssi_dbm)
                    },
                    diagnostics,
                    ...(comparison ? { occupancy_comparison: comparison } : {}),
                    last_motion: session?.motion ?? null,
                    fault_events: session ? session.faults.slice() : [],
                    capabilities,
                    device,
                    health,
                    sensing,
                    wifi
                };

                const browser = window.ESPectreBrowserSupport?.current;
                const revision = document.querySelector?.('meta[name="espectre-site-revision"]')?.content;
                const asset = document.querySelector?.('script[src*="/assets/js/monitor-tool.js"]')?.src;
                const sanitized = {
                    privacy: 'Generated locally. Network and device identifiers are masked. Review before sharing.',
                    source: mode,
                    client: {
                        site: ['espectre.dev', 'test.espectre.dev'].includes(window.location?.hostname)
                            ? window.location.hostname : 'local_or_custom',
                        site_revision: /^[a-f0-9]{40}$/i.test(revision) ? revision : null,
                        monitor_revision: asset?.match(/[?&]v=([a-f0-9]{12})(?:&|$)/i)?.[1] ?? null,
                        browser: browser?.browser ?? null,
                        browser_version: browser?.browserVersion ?? null
                    },
                    ...redactDiagnosticReport(rawReport)
                };
                monitorReportJson = JSON.stringify(sanitized, null, 2);

                if (previewEl) previewEl.textContent = monitorReportJson;
                if (progressEl) progressEl.hidden = true;
                if (resultEl) resultEl.hidden = false;
                if (startBtn) startBtn.disabled = false;
                if (copyBtn && typeof copyBtn.focus === 'function') copyBtn.focus();
                monitorReportSampling = null;
                occupancyComparison.sync();
                toast(samplesCollected ? 'Diagnostic report ready.' : 'No diagnostic samples received. The report contains device state only.');
                resolve(sanitized);
            };

            pollSample();
        });
    }

    async function monitorCopyDiagnosticReport() {
        if (!monitorReportJson) {
            toast('No diagnostic report available to copy.');
            return;
        }
        const fenced = '```json\n' + monitorReportJson + '\n```';
        const copyBtn = $('.js-mon-report-copy');
        try {
            if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(fenced);
            } else if (typeof document !== 'undefined') {
                const ta = document.createElement('textarea');
                ta.value = fenced;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                try {
                    if (!document.execCommand('copy')) throw new Error('Copy was declined');
                } finally {
                    document.body.removeChild(ta);
                }
            }
            if (copyBtn) {
                const prevText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    if (copyBtn.textContent === 'Copied!') copyBtn.textContent = prevText;
                }, 2000);
            }
            toast('Fenced JSON copied to clipboard.');
        } catch (err) {
            console.warn('Clipboard write failed:', err);
            toast('Failed to copy to clipboard.');
        }
    }

    function monitorDownloadDiagnosticReport() {
        if (!monitorReportJson) {
            toast('No diagnostic report available to download.');
            return;
        }
        try {
            const blob = new Blob([monitorReportJson], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `espectre-diagnostic-report-${Date.now()}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(url), 0);
            toast('Diagnostic report downloaded.');
        } catch (err) {
            console.warn('Download failed:', err);
            toast('Failed to download report.');
        }
    }

    function monitorInit() {
        occupancyComparison.init();
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
        bindThresholdControls();
        document.getElementById('sense-detector').addEventListener('change', () => {
            const detector = document.getElementById('sense-detector').value;
            syncSensingControls();
            runSensingCommand(
                { command: 'update_sensing', detector },
                'Changing detection mode…',
                'Detection mode updated.'
            );
        });
        const applyMotionHits = () => {
            const motionOnHits = Number(document.getElementById('sense-motion-on').value);
            const motionOffHits = Number(document.getElementById('sense-motion-off').value);
            if (![motionOnHits, motionOffHits].every((value) => Number.isInteger(value) && value >= 1 && value <= 20)) {
                toast('Use a whole number from 1 to 20 for each advanced motion setting.');
                return;
            }
            runSensingCommand(
                { command: 'update_sensing', motion_on_hits: motionOnHits, motion_off_hits: motionOffHits },
                'Saving advanced motion settings…',
                'Advanced motion settings updated.',
                () => setDemoMotionHits(motionOnHits, motionOffHits)
            );
        };
        document.getElementById('sense-motion-on').addEventListener('change', applyMotionHits);
        document.getElementById('sense-motion-off').addEventListener('change', applyMotionHits);
        $('.js-sense-recalibrate').addEventListener('click', beginCalibration);
        document.getElementById('sense-generator-mode').addEventListener('change', () => {
            const trafficGeneratorMode = document.getElementById('sense-generator-mode').value;
            syncSensingControls();
            runSensingCommand(
                { command: 'update_sensing', traffic_generator_mode: trafficGeneratorMode },
                'Changing the Wi-Fi traffic source…',
                'Wi-Fi traffic source updated.',
                () => {
                    demoSysinfoSnapshot.traffic_mode = trafficGeneratorMode;
                    applySensingSnapshot({ traffic_generator_mode: trafficGeneratorMode });
                }
            );
        });
        document.getElementById('sense-sample-spacing').addEventListener('change', monitorChangeSampleSpacing);
        document.getElementById('sense-raw-target').addEventListener('change', monitorChangeRawTarget);
        document.getElementById('sense-raw-target-manual').addEventListener('change', monitorChangeRawTarget);
        document.getElementById('sense-raw-target-refresh').addEventListener('click', monitorRefreshRawTrafficTargets);
        $('.js-mon-report-start')?.addEventListener('click', () => monitorStartDiagnosticReport());
        $('.js-mon-report-cancel')?.addEventListener('click', () => monitorCancelDiagnosticReport());
        $('.js-mon-report-copy')?.addEventListener('click', () => monitorCopyDiagnosticReport());
        $('.js-mon-report-download')?.addEventListener('click', () => monitorDownloadDiagnosticReport());
        window.addEventListener('resize', monitorResizeChart);
    }
