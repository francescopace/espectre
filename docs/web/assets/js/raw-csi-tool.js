/*
 * ESPectre - Raw CSI tool
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    /* =========================================================== raw CSI */

    const RAW_CSI_V8_HEADER_BYTES = 64;
    const RAW_CSI_VISUAL_HISTORY = 720;
    const RAW_CSI_PHASE_HISTORY = 72;
    const RAW_CSI_IQ_WINDOW_US = 2000000;
    const RAW_CSI_IQ_EXTENT = 128;
    const RAW_CSI_VISUAL_STEP_US = 33333;
    const RAW_CSI_RENDER_INTERVAL_MS = 1000 / 30;
    const RAW_CSI_CHANNEL_GHOST_GAIN = 5;
    const RAW_CSI_PHASE_TRAIL_GAIN = 5;
    const RAW_CSI_SELECTED_SUBCARRIERS = Object.freeze([4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60]);
    const RAW_CSI_LIVE_SUBCARRIERS = Object.freeze(
        Array.from({ length: 57 }, (_unused, index) => index + 4).filter((index) => index !== 32));
    const RAW_CSI_VISUALIZATIONS = Object.freeze({
        'channel-heatmap': Object.freeze({
            title: 'Channel heatmap',
            description: 'Brightness shows signal strength. Cyan and coral show changes around the recent baseline.',
            badge: 'LIVE',
            ariaLabel: 'Combined CSI amplitude and motion heatmap over time'
        }),
        'rf-waterfall': Object.freeze({
            title: 'RF waterfall',
            description: 'Recent signal profiles move into the distance while movement changes the surface.',
            badge: 'LIVE',
            ariaLabel: 'Perspective waterfall of recent CSI channel profiles'
        }),
        'channel-ghost': Object.freeze({
            title: 'Channel ghost',
            description: 'The current signal is compared with its recent baseline. Changes are enlarged 5× so they are easier to see.',
            badge: 'LIVE',
            ariaLabel: 'Current normalized CSI channel profile compared with its baseline'
        }),
        'iq-constellation': Object.freeze({
            title: 'I/Q constellation',
            description: 'Recent raw Espressif I/Q samples from the 12 production subcarriers over a two-second window.',
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
        metricsTimer: null,
        demoFresh: 0,
        state: 'idle',
        generation: 0,
        startRequest: null,
        stopPromise: null,
        parser: null,
        visualization: 'channel-heatmap',
        profiles: [],
        deltas: [],
        timestampsUs: [],
        phaseHistory: [],
        iqHistory: [],
        iqTimestampsUs: [],
        baseline: null,
        latestProfile: null,
        latestDelta: null,
        lastCaptureTicksUs: 0,
        lastVisualTicksUs: 0,
        lastRenderAt: 0,
        renderFrame: 0,
        metricsWindowStartedAt: 0,
        metricsReceived: 0,
        metricsRssiSum: 0,
        metricsRssiSamples: 0,
        metricsSnrSum: 0,
        metricsSnrSamples: 0,
        metricsCaptureIntervalMsSum: 0,
        metricsCaptureIntervalSamples: 0,
        metricsLastCaptureTicksUs: 0,
        metricsFresh: 0,
        metricsDropped: 0,
        metricsBackpressure: 0,
        heatmapSurface: null,
        heatmapPixels: null,
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
        const unavailable = $('.js-raw-csi-unavailable');
        const workspace = $('.js-raw-csi-workspace');
        const externalHint = $('.js-raw-csi-external-hint');
        if (conn.status !== 'connected' || !['direct', 'demo'].includes(conn.mode)) {
            if (externalHint) externalHint.hidden = true;
            if (onboarding) onboarding.hidden = false;
            if (unavailable) unavailable.hidden = true;
            if (workspace) workspace.hidden = true;
            return false;
        }
        if (onboarding) onboarding.hidden = true;
        if (conn.mode === 'demo') {
            if (externalHint) externalHint.hidden = true;
            rawCsiSetAvailable(true);
            rawCsiStatus('Demo ready. Start the simulated signal stream when you are ready.');
            return true;
        }
        const rawCapability = directClient?.capabilities?.raw_csi;
        const available = directClient?.capabilities?.features?.raw_csi === true
            && rawCapability?.protocol_version === 1
            && rawCapability?.marker === '👻';
        rawCsiSetAvailable(available);
        if (externalHint) externalHint.hidden = !available || conn.csiTrafficMode !== 'external';
        if (available) {
            rawCsiStatus(conn.csiTrafficMode === 'external'
                ? 'Connected. This device is waiting for its external Wi-Fi traffic source before data appears.'
                : 'Connected. Start the temporary signal stream when you are ready. Nothing is uploaded or stored.');
        }
        return available;
    }

    function rawCsiSetState(state) {
        rawCsi.state = state;
        const toggle = $('.js-raw-csi-toggle');
        if (!toggle) return;
        const idle = state === 'idle';
        toggle.textContent = state === 'stopping' ? 'Stopping…' : idle ? 'Start' : 'Stop';
        toggle.disabled = state === 'stopping';
        toggle.classList.toggle('btn-primary', idle);
        toggle.classList.toggle('btn-secondary', !idle);
    }

    function rawCsiCounter(selector, value) {
        const element = $(selector);
        if (element) element.textContent = typeof value === 'bigint'
            ? value.toLocaleString('en-US') : Number(value).toLocaleString('en-US');
    }

    function rawCsiLabel(selector, value) {
        const element = $(selector);
        if (element) element.textContent = value || '—';
    }

    function rawCsiCollectSignalSample(rssi, noiseFloor) {
        rawCsi.metricsRssiSum += rssi;
        rawCsi.metricsRssiSamples += 1;
        if (noiseFloor < 0) {
            rawCsi.metricsSnrSum += rssi - noiseFloor;
            rawCsi.metricsSnrSamples += 1;
        }
    }

    function rawCsiCollectRadioMetrics(view) {
        const rssi = view.getInt8(43);
        const noiseFloor = view.getInt8(44);
        rawCsiCollectSignalSample(rssi, noiseFloor);
    }

    function rawCsiCollectCaptureInterval(captureTicksUs) {
        const previousTicksUs = rawCsi.metricsLastCaptureTicksUs;
        const intervalMs = previousTicksUs > 0 && captureTicksUs > previousTicksUs
            ? (captureTicksUs - previousTicksUs) / 1000 : 0;
        if (intervalMs > 0) {
            rawCsi.metricsCaptureIntervalMsSum += intervalMs;
            rawCsi.metricsCaptureIntervalSamples += 1;
        }
        rawCsi.metricsLastCaptureTicksUs = captureTicksUs;
    }

    function rawCsiResetMetricWindow(now = performance.now()) {
        rawCsi.metricsWindowStartedAt = now;
        rawCsi.metricsReceived = 0;
        rawCsi.metricsRssiSum = 0;
        rawCsi.metricsRssiSamples = 0;
        rawCsi.metricsSnrSum = 0;
        rawCsi.metricsSnrSamples = 0;
        rawCsi.metricsCaptureIntervalMsSum = 0;
        rawCsi.metricsCaptureIntervalSamples = 0;
    }

    function rawCsiFlushMetrics() {
        const now = performance.now();
        const elapsedMs = Math.max(1, now - rawCsi.metricsWindowStartedAt);
        rawCsiCounter('.js-raw-pps', Math.round(rawCsi.metricsReceived * 1000 / elapsedMs));
        rawCsiCounter('.js-raw-fresh', rawCsi.metricsFresh);
        rawCsiCounter('.js-raw-dropped', rawCsi.metricsDropped);
        rawCsiCounter('.js-raw-backpressure', rawCsi.metricsBackpressure);
        rawCsiLabel('.js-raw-capture-interval', rawCsi.metricsCaptureIntervalSamples > 0
            ? (rawCsi.metricsCaptureIntervalMsSum / rawCsi.metricsCaptureIntervalSamples).toFixed(1)
            : '—');
        rawCsiLabel('.js-raw-rssi', rawCsi.metricsRssiSamples > 0
            ? (rawCsi.metricsRssiSum / rawCsi.metricsRssiSamples).toFixed(1)
            : '—');
        rawCsiLabel('.js-raw-snr', rawCsi.metricsSnrSamples > 0
            ? (rawCsi.metricsSnrSum / rawCsi.metricsSnrSamples).toFixed(1)
            : '—');
        rawCsiResetMetricWindow(now);
    }

    function rawCsiStartMetrics() {
        clearInterval(rawCsi.metricsTimer);
        rawCsi.metricsFresh = 0;
        rawCsi.metricsDropped = 0;
        rawCsi.metricsBackpressure = 0;
        rawCsi.metricsLastCaptureTicksUs = 0;
        rawCsiResetMetricWindow();
        ['.js-raw-pps', '.js-raw-fresh', '.js-raw-dropped', '.js-raw-backpressure']
            .forEach((selector) => rawCsiCounter(selector, 0));
        ['.js-raw-capture-interval', '.js-raw-rssi', '.js-raw-snr']
            .forEach((selector) => rawCsiLabel(selector, '—'));
        rawCsi.metricsTimer = setInterval(rawCsiFlushMetrics, 1000);
    }

    function rawCsiStopMetrics() {
        clearInterval(rawCsi.metricsTimer);
        rawCsi.metricsTimer = null;
        rawCsi.metricsReceived = 0;
        rawCsiCounter('.js-raw-pps', 0);
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
        rawCsi.baseline = null;
        rawCsi.latestProfile = null;
        rawCsi.latestDelta = null;
        rawCsi.lastCaptureTicksUs = 0;
        rawCsi.lastVisualTicksUs = 0;
        rawCsi.lastRenderAt = 0;
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
        context.font = '500 15px ui-monospace, "SFMono-Regular", Consolas, monospace';
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

    function rawCsiWriteChannelPixel(pixels, offset, amplitude, delta) {
        const level = Math.max(0, Math.min(1, amplitude / 2.2));
        const motion = Math.sqrt(Math.min(1, Math.abs(delta) / 0.32));
        const baseRed = 8 + level * 46;
        const baseGreen = 8 + level * 34;
        const baseBlue = 24 + level * 126;
        const negative = delta < 0;
        pixels[offset] = Math.round(baseRed + ((negative ? 42 : 255) - baseRed) * motion);
        pixels[offset + 1] = Math.round(baseGreen + ((negative ? 220 : 74) - baseGreen) * motion);
        pixels[offset + 2] = Math.round(baseBlue + ((negative ? 255 : 105) - baseBlue) * motion);
        pixels[offset + 3] = 255;
    }

    function rawCsiWaterfallColor(active, alpha = 1) {
        const intensity = Math.max(0, Math.min(1, active));
        const hue = Math.round(205 - intensity * 157);
        const saturation = Math.round(88 + intensity * 8);
        const lightness = Math.round(61 + intensity * 4);
        return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
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
        const rows = rawCsi.profiles[0].length;
        if (!rawCsi.heatmapSurface) rawCsi.heatmapSurface = document.createElement('canvas');
        const surface = rawCsi.heatmapSurface;
        if (surface.width !== RAW_CSI_VISUAL_HISTORY || surface.height !== rows) {
            surface.width = RAW_CSI_VISUAL_HISTORY;
            surface.height = rows;
            rawCsi.heatmapPixels = null;
        }
        const surfaceContext = surface.getContext('2d');
        if (!rawCsi.heatmapPixels) {
            rawCsi.heatmapPixels = surfaceContext.createImageData(RAW_CSI_VISUAL_HISTORY, rows);
        }
        const pixels = rawCsi.heatmapPixels;
        const data = pixels.data;
        for (let offset = 0; offset < data.length; offset += 4) {
            data[offset] = 9;
            data[offset + 1] = 9;
            data[offset + 2] = 28;
            data[offset + 3] = 255;
        }
        const startColumn = RAW_CSI_VISUAL_HISTORY - rawCsi.profiles.length;
        rawCsi.profiles.forEach((profile, profileIndex) => {
            profile.forEach((value, subcarrier) => {
                const offset = (subcarrier * RAW_CSI_VISUAL_HISTORY
                    + startColumn + profileIndex) * 4;
                rawCsiWriteChannelPixel(
                    data, offset, value, rawCsi.deltas[profileIndex]?.[subcarrier] || 0);
            });
        });
        surfaceContext.putImageData(pixels, 0, 0);
        context.imageSmoothingEnabled = false;
        context.drawImage(surface, left, top, width, height);
        context.imageSmoothingEnabled = true;
        context.strokeStyle = 'rgba(255, 255, 255, .15)';
        context.strokeRect(left + 0.5, top + 0.5, width - 1, height - 1);
        context.fillStyle = 'rgba(255, 255, 255, .55)';
        context.font = '12px ui-monospace, "SFMono-Regular", Consolas, monospace';
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
        context.strokeStyle = 'rgba(77, 156, 255, .14)';
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
            context.strokeStyle = rawCsiWaterfallColor(active, alpha);
            context.lineWidth = profileIndex === profiles.length - 1 ? 2.8 : 1 + active * 0.55;
            context.shadowColor = profileIndex === profiles.length - 1
                ? rawCsiWaterfallColor(active, .9) : 'transparent';
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
        context.font = '12px ui-monospace, "SFMono-Regular", Consolas, monospace';
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
        context.font = '12px ui-monospace, "SFMono-Regular", Consolas, monospace';
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
        const extent = RAW_CSI_IQ_EXTENT;
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
            rawCsi.iqHistory.forEach((sample) => {
                const point = pointPosition(sample, subcarrier);
                context.fillStyle = `hsl(${hue} 94% 68%)`;
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
        context.font = '12px ui-monospace, "SFMono-Regular", Consolas, monospace';
        context.textAlign = 'center';
        context.fillText('12 PRODUCTION SUBCARRIERS · 2 SECONDS', centerX, top - 10);
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
                context.font = '11px ui-monospace, "SFMono-Regular", Consolas, monospace';
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
        context.font = '12px ui-monospace, "SFMono-Regular", Consolas, monospace';
        context.textAlign = 'left';
        context.fillText('5× TRAIL SPREAD', 12, 20);
        context.textAlign = 'right';
        context.fillText('RELATIVE I', Math.min(canvas.width - 8, centerX + radius + 62), centerY + 4);
        context.textAlign = 'center';
        context.fillText('RELATIVE Q', centerX, centerY - radius - 18);
        context.fillText('CFO/STO-REDUCED PHASE · NOT POSITION', centerX, canvas.height - 16);
    }

    function rawCsiRender(timestamp) {
        rawCsi.renderFrame = 0;
        if (timestamp - rawCsi.lastRenderAt < RAW_CSI_RENDER_INTERVAL_MS) {
            rawCsi.renderFrame = requestAnimationFrame(rawCsiRender);
            return;
        }
        rawCsi.lastRenderAt = timestamp;
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
        rawCsiCollectRadioMetrics(view);
        const capturedTicksUs = Number(view.getBigUint64(22, true))
            || rawCsi.lastCaptureTicksUs + RAW_CSI_VISUAL_STEP_US;
        rawCsiCollectCaptureInterval(capturedTicksUs);
        rawCsiIngestVisualFrame(amplitudes, iValues, qValues, capturedTicksUs);
    }

    function rawCsiAppend(chunk) {
        if (!rawCsi.parser) throw new Error('Raw CSI parser is not initialized.');
        rawCsi.parser.append(chunk).forEach((frame) => {
            rawCsi.metricsFresh = frame.freshRecordTotal;
            rawCsi.metricsDropped = frame.rawDropTotal;
            rawCsi.metricsBackpressure = frame.sendBackpressureTotal;
            rawCsiConsumeRecord(frame.record, frame.streamSequence);
            rawCsi.metricsReceived += 1;
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
        const captureTicksUs = Math.round(performance.now() * 1000);
        rawCsiCollectCaptureInterval(captureTicksUs);
        rawCsiIngestVisualFrame(amplitudes, iValues, qValues, captureTicksUs);
        rawCsi.demoFresh += Math.max(1, Math.round(targetPps * intervalMs / 1000));
        rawCsi.metricsFresh = rawCsi.demoFresh;
        rawCsi.metricsReceived += 1;
        const rssi = Math.round(-50 + motion * 7);
        const noiseFloor = -96;
        rawCsiCollectSignalSample(rssi, noiseFloor);
    }

    function rawCsiStartDemo(targetPps) {
        const intervalMs = Math.max(10, Math.round(1000 / targetPps));
        const startedAtMs = performance.now();
        rawCsi.demoFresh = 0;
        rawCsiResetVisualization();
        rawCsiStartMetrics();
        rawCsiSetState('running');
        rawCsiStatus(`Simulated signal stream running at a target of ${targetPps} packets per second.`);
        rawCsiDemoFrame(targetPps, intervalMs, startedAtMs);
        rawCsi.demoTimer = setInterval(
            () => rawCsiDemoFrame(targetPps, intervalMs, startedAtMs), intervalMs);
    }

    async function rawCsiStop(expectedGeneration = rawCsi.generation) {
        if (expectedGeneration !== rawCsi.generation || rawCsi.state === 'idle') return;
        if (rawCsi.state === 'stopping') return rawCsi.stopPromise;
        const stopGeneration = ++rawCsi.generation;
        const client = rawCsi.sessionClient;
        const pendingStart = rawCsi.startRequest;
        clearInterval(rawCsi.demoTimer);
        rawCsi.demoTimer = null;
        rawCsi.demoFresh = 0;
        rawCsiStopMetrics();
        rawCsi.controller?.abort('raw stream stopped');
        rawCsi.controller = null;
        rawCsiSetState('stopping');
        rawCsi.parser = null;
        rawCsi.stopPromise = (async () => {
            try { await pendingStart; } catch (_error) { /* a failed start has no device session to release */ }
            if (client?.rawSessionId && client.connected) {
                try { await client.request('stop_raw_stream', {}, { timeoutMs: 3000 }); } catch (_error) { /* abort also releases the device session */ }
            }
            if (rawCsi.generation !== stopGeneration) return;
            rawCsi.sessionClient = null;
            rawCsi.startRequest = null;
            rawCsi.stopPromise = null;
            rawCsiSetState('idle');
        })();
        return rawCsi.stopPromise;
    }

    async function rawCsiStart() {
        const client = directClient;
        if (rawCsi.state !== 'idle' || conn.status !== 'connected') return;
        const generation = ++rawCsi.generation;
        if (conn.mode === 'demo') {
            rawCsiStartDemo(100);
            return;
        }
        if (!rawCsiDirectReady() || client.capabilities?.features?.raw_csi !== true) return;
        rawCsiSetState('starting');
        rawCsi.sessionClient = client;
        rawCsiStatus('Starting the signal stream…');
        try {
            const startRequest = client.request('start_raw_stream');
            rawCsi.startRequest = startRequest;
            const session = await startRequest;
            if (rawCsi.startRequest === startRequest) rawCsi.startRequest = null;
            if (rawCsi.generation !== generation || rawCsi.state !== 'starting') return;
            rawCsi.parser = new window.ESPectreRawCsiParser(session.session_id);
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
            if (rawCsi.generation !== generation || rawCsi.state !== 'starting') return;
            if (!response.ok || !response.body) throw new Error(`Raw stream returned HTTP ${response.status}.`);
            rawCsiStartMetrics();
            rawCsiSetState('running');
            rawCsiStatus('Live signal data is arriving. Normal motion sensing will resume when you stop the stream.');
            const reader = response.body.getReader();
            while (rawCsi.generation === generation && !controller.signal.aborted) {
                const { value, done } = await reader.read();
                if (done) break;
                if (rawCsi.generation !== generation) break;
                rawCsiAppend(value);
            }
            if (rawCsi.generation === generation && !controller.signal.aborted) {
                throw new Error('Raw stream ended unexpectedly.');
            }
        } catch (error) {
            if (rawCsi.generation === generation && !rawCsi.controller?.signal.aborted) {
                console.warn('Raw CSI stream failed:', error);
                rawCsiStatus('The signal stream stopped unexpectedly. Stop it, then try again.', true);
            }
        } finally {
            if (rawCsi.generation === generation
                    && rawCsi.state !== 'idle' && rawCsi.state !== 'stopping') {
                await rawCsiStop(generation);
            }
        }
    }

    function rawCsiChooseDevice() {
        disconnect();
        directEndpointInput()?.focus();
    }

    function rawCsiToggle() {
        if (rawCsi.state === 'idle') void rawCsiStart();
        else if (rawCsi.state !== 'stopping') void rawCsiStop();
    }

    function rawCsiInit() {
        $('.js-raw-csi-choose-device')?.addEventListener('click', rawCsiChooseDevice);
        $('.js-raw-csi-toggle')?.addEventListener('click', rawCsiToggle);
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
