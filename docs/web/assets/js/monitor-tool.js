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
        rawTargetDiscovery: null,
        sessionFaults: []
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
        resetMonitorReportView();
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
        const canPoll = diagnosticsPanelOpen()
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

    function recordMonitorFault(data) {
        if (!monitor.sessionFaults) monitor.sessionFaults = [];
        monitor.sessionFaults.push({
            at: Date.now(),
            timestamp: new Date().toISOString(),
            fault: data && typeof data === 'object' ? { ...data } : data
        });
    }

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
        const clean = mac.toLowerCase();
        if (clean === 'ff:ff:ff:ff:ff:ff' || clean === 'ff-ff-ff-ff-ff-ff') return true;
        if (clean.startsWith('01:00:5e') || clean.startsWith('01-00-5e') || clean.startsWith('33:33')) return true;
        return false;
    }

    function isMulticastOrLoopbackIp(ip) {
        if (ip === '127.0.0.1' || ip === '::1' || ip === '0.0.0.0' || ip === '255.255.255.255') return true;
        const first = parseInt(ip.split('.')[0], 10);
        if (first >= 224 && first <= 239) return true;
        return false;
    }

    function redactMacAddress(value) {
        return value.replace(/\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b/g, (match) => {
            if (isMulticastOrBroadcastMac(match)) return match;
            const sep = match.includes(':') ? ':' : '-';
            const parts = match.split(sep);
            return `${parts[0]}${sep}${parts[1]}${sep}**${sep}**${sep}**${sep}**`;
        });
    }

    function redactIpv4Address(value) {
        return value.replace(/\b((?:1\d\d|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.){3}(?:1\d\d|2[0-4]\d|25[0-5]|[1-9]\d|\d)\b/g, (match) => {
            if (isMulticastOrLoopbackIp(match)) return match;
            const parts = match.split('.');
            return `${parts[0]}.${parts[1]}.x.x`;
        });
    }

    function redactDiagnosticReport(data) {
        if (!data || typeof data !== 'object') return data;

        const sensitiveTerms = [];
        const addTerm = (val, replacement) => {
            if (typeof val === 'string' && val.trim().length >= 3) {
                const escaped = val.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                sensitiveTerms.push({ pattern: new RegExp(escaped, 'gi'), replacement });
            }
        };

        if (data.wifi && typeof data.wifi === 'object') {
            addTerm(data.wifi.ssid, '<redacted_ssid>');
            addTerm(data.wifi.sta_ssid, '<redacted_ssid>');
            addTerm(data.wifi.ap_ssid, '<redacted_ssid>');
        }
        if (data.device && typeof data.device === 'object') {
            addTerm(data.device.name, '<redacted_name>');
            addTerm(data.device.label, '<redacted_label>');
            addTerm(data.device.hostname, '<redacted_hostname>');
        }

        const sanitizeText = (str) => {
            if (typeof str !== 'string') return str;
            let out = str;
            for (const { pattern, replacement } of sensitiveTerms) {
                out = out.replace(pattern, replacement);
            }
            out = out.replace(/\b[a-zA-Z0-9][-a-zA-Z0-9]*\.local\b/gi, '<redacted_hostname>.local');
            out = redactMacAddress(out);
            out = redactIpv4Address(out);
            return out;
        };

        const transform = (key, val) => {
            const lowerKey = (key || '').toLowerCase();

            if (lowerKey === 'credentials' || lowerKey.endsWith('_credentials')) {
                return undefined;
            }

            if (lowerKey.includes('password') || lowerKey.includes('passphrase') || lowerKey.includes('secret') || lowerKey.includes('token') || lowerKey.includes('api_key') || lowerKey.includes('private_key')) {
                if (typeof val === 'boolean') return val;
                if (val && typeof val === 'string') return '[REDACTED]';
                if (val && typeof val === 'object') return undefined;
                return val;
            }

            if (lowerKey === 'name' && typeof val === 'string') return '<redacted_name>';
            if (lowerKey === 'label' && typeof val === 'string') return '<redacted_label>';
            if (lowerKey === 'hostname' && typeof val === 'string') return '<redacted_hostname>';
            if (lowerKey === 'ssid' && typeof val === 'string') return '<redacted_ssid>';

            if (lowerKey === 'device_id' && typeof val === 'string' && val.length >= 8) {
                return `${val.slice(0, 4)}...${val.slice(-4)}`;
            }

            if (typeof val === 'string') {
                return sanitizeText(val);
            }

            if (Array.isArray(val)) {
                return val.map((item) => transform(key, item)).filter((item) => item !== undefined);
            }

            if (val && typeof val === 'object') {
                const result = {};
                for (const [k, v] of Object.entries(val)) {
                    const transformed = transform(k, v);
                    if (transformed !== undefined) result[k] = transformed;
                }
                return result;
            }

            return val;
        };

        return transform('root', data);
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
        if (monitorReportSampling) return Promise.resolve(null);
        const mode = conn.mode;
        const direct = mode === 'direct';
        if (!direct && mode !== 'demo') {
            toast('Connect to a device before generating a diagnostic report.');
            return Promise.resolve(null);
        }
        if (direct && !directClient?.connected) {
            toast('Direct HTTP connection is not ready. Connect and try again.');
            return Promise.resolve(null);
        }

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
            csi_hw_error_pps: []
        };

        let elapsedSec = 0;
        const controller = new AbortController();

        return new Promise((resolve) => {
            const samplingState = {
                controller,
                timer: null,
                cancel(silent = false) {
                    if (samplingState.timer) clearInterval(samplingState.timer);
                    controller.abort();
                    monitorReportSampling = null;
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

            const pollSample = async () => {
                if (monitorReportSampling !== samplingState) return;
                if (direct && !directClient?.connected) {
                    samplingState.cancel({ silent: true });
                    toast('Device disconnected while sampling diagnostics.');
                    return;
                }
                if (direct && directClient?.connected) {
                    try {
                        const res = await directClient.request('get', 'diagnostics', {
                            fields: ['csi_occupancy', 'csi_accepted_pps', 'csi_filtered_pps', 'csi_hw_error_pps']
                        }, { timeoutMs: Math.min(2000, sampleIntervalMs), signal: controller.signal });
                        if (res && typeof res === 'object') {
                            if (typeof res.csi_occupancy === 'number') samples.csi_occupancy.push(res.csi_occupancy);
                            if (typeof res.csi_accepted_pps === 'number') samples.csi_accepted_pps.push(res.csi_accepted_pps);
                            if (typeof res.csi_filtered_pps === 'number') samples.csi_filtered_pps.push(res.csi_filtered_pps);
                            if (typeof res.csi_hw_error_pps === 'number') samples.csi_hw_error_pps.push(res.csi_hw_error_pps);
                        }
                    } catch {
                        // Stale or dropped polling interval sample is ignored
                    }
                } else if (mode === 'demo') {
                    const targetPps = csiTargetPps();
                    samples.csi_accepted_pps.push(Math.max(1, targetPps - 10));
                    samples.csi_occupancy.push(Math.max(1, targetPps - 16) / targetPps);
                    samples.csi_filtered_pps.push(0);
                    samples.csi_hw_error_pps.push(0);
                }
                const pct = Math.min(100, Math.round((elapsedSec / sampleDurationSec) * 100));
                if (barEl) barEl.style.width = `${pct}%`;
                if (trackEl) trackEl.setAttribute('aria-valuenow', String(pct));
                if (statusEl) statusEl.textContent = `Sampling diagnostics (${Math.min(elapsedSec, sampleDurationSec)}/${sampleDurationSec}s)…`;

                if (elapsedSec >= sampleDurationSec) {
                    if (samplingState.timer) clearInterval(samplingState.timer);
                    samplingState.timer = null;
                    await finishReport();
                    return;
                }
                elapsedSec += Math.round(sampleIntervalMs / 1000) || 1;
            };

            const finishReport = async () => {
                if (monitorReportSampling !== samplingState) return;
                if (statusEl) statusEl.textContent = 'Reading device resources…';

                let capabilities = {};
                let device = {};
                let health = {};
                let sensing = {};
                let wifi = {};

                if (direct && directClient?.connected) {
                    const readResource = async (name) => {
                        try {
                            return await directClient.request('get', name, {}, {
                                timeoutMs: 5000,
                                signal: controller.signal
                            });
                        } catch (err) {
                            return { error: err.message || String(err) };
                        }
                    };
                    [capabilities, device, health, sensing, wifi] = await Promise.all([
                        readResource('capabilities'),
                        readResource('device'),
                        readResource('health'),
                        readResource('sensing'),
                        readResource('wifi')
                    ]);
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

                const rawReport = {
                    timestamp: new Date().toISOString(),
                    client: {
                        site: typeof window !== 'undefined' && window.location ? window.location.origin : '',
                        user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : ''
                    },
                    sampling: {
                        duration_seconds: sampleDurationSec,
                        samples_collected: Math.max(
                            samples.csi_occupancy.length,
                            samples.csi_accepted_pps.length,
                            samples.csi_filtered_pps.length,
                            samples.csi_hw_error_pps.length
                        ),
                        rates: {
                            csi_occupancy: calculateRateStats(samples.csi_occupancy),
                            csi_accepted_pps: calculateRateStats(samples.csi_accepted_pps),
                            csi_filtered_pps: calculateRateStats(samples.csi_filtered_pps),
                            csi_hw_error_pps: calculateRateStats(samples.csi_hw_error_pps)
                        }
                    },
                    fault_events: monitor.sessionFaults ? monitor.sessionFaults.slice() : [],
                    capabilities,
                    device,
                    health,
                    sensing,
                    wifi
                };

                const sanitized = redactDiagnosticReport(rawReport);
                monitorReportJson = JSON.stringify(sanitized, null, 2);

                if (previewEl) previewEl.textContent = monitorReportJson;
                if (progressEl) progressEl.hidden = true;
                if (resultEl) resultEl.hidden = false;
                if (startBtn) startBtn.disabled = false;
                if (copyBtn && typeof copyBtn.focus === 'function') copyBtn.focus();
                monitorReportSampling = null;
                toast('Diagnostic report ready.');
                resolve(sanitized);
            };

            pollSample();
            samplingState.timer = setInterval(pollSample, sampleIntervalMs);
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
                document.execCommand('copy');
                document.body.removeChild(ta);
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
