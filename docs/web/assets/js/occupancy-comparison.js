// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
'use strict';

const occupancyComparison = (() => {
    const generators = ['ping', 'dns', 'dns_tcp', 'wifi_raw'];
    const spacings = ['half_slot', 'distinct'];
    const sampleMs = 30000;
    const settleMs = 3000;
    const stepLabels = { waiting: 'Waiting', active: 'In progress', done: 'Done', skipped: 'Skipped', stopped: 'Stopped', pending: 'Needs recovery' };
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    const key = id => 'espectre.occupancy.restore.' + id;
    let active = null;
    let view = null;

    function supported(client) {
        const caps = client?.capabilities;
        const op = caps?.operations?.find(o => o.name === 'update_sensing');
        return !!op && caps.features?.csi_sample_spacing === true
            && caps.resources?.includes('diagnostics')
            && ['detector', 'threshold', 'traffic_generator_mode', 'csi_sample_spacing']
                .every(field => !op.fields || op.fields.includes(field));
    }

    function saved(id) {
        try { return JSON.parse(localStorage.getItem(key(id))); }
        catch { return null; }
    }

    function same(run) {
        return directClient === run.client && run.client.connected && conn.mode === 'direct'
            && conn.deviceId === run.id;
    }

    function check(run) {
        if (!same(run)) throw new Error('Reconnect to the original device to restore its settings.');
        if (run.cancelled && !run.restoring) throw new Error('Comparison cancelled.');
    }

    async function request(run, method, resource, data = {}) {
        check(run);
        const result = await run.client.request(method, resource, data, { timeoutMs: 4500 });
        check(run);
        return result;
    }

    const transient = error => ['timeout', 'connection_failed', 'http_503', 'http_429'].includes(error.code);

    async function read(run) {
        for (let attempt = 0; ; attempt++) {
            try { return await request(run, 'get', 'sensing'); }
            catch (error) {
                if (!transient(error) || attempt >= 3) throw error;
                await pause(1000);
            }
        }
    }

    function matches(state, fields) {
        return Object.entries(fields).every(([k, v]) => k === 'threshold'
            ? Number.isFinite(state[k]) && Math.abs(state[k] - v) < 0.00001 : state[k] === v);
    }

    function config(state) {
        return { detector: state.detector, threshold: state.threshold,
            traffic_generator_mode: state.traffic_generator_mode, csi_sample_spacing: state.csi_sample_spacing };
    }

    function pair(state) {
        return { traffic_generator_mode: state.traffic_generator_mode, csi_sample_spacing: state.csi_sample_spacing };
    }

    function label(fields) {
        const source = { ping: 'Ping', dns: 'DNS (UDP)', dns_tcp: 'DNS (TCP)', wifi_raw: 'Wi-Fi raw', external: 'External' };
        return source[fields.traffic_generator_mode] + ' / ' + (fields.csi_sample_spacing === 'half_slot' ? 'Half slot' : 'Distinct timestamps');
    }

    async function change(run, fields) {
        if (!matches(await read(run), fields)) {
            try { await request(run, 'patch', 'sensing', fields); }
            catch (error) {
                // A lost acknowledgement can follow a successful change of traffic source.
                if (!transient(error) || !matches(await read(run), fields)) throw error;
            }
            if (!matches(await read(run), fields)) throw new Error('Device settings could not be verified.');
        }
    }

    async function restore(run) {
        run.restoring = true;
        const original = run.original;
        await change(run, pair(original));
        await change(run, { detector: original.detector });
        // Lightweight can replace a session threshold at the end of calibration.
        const deadline = Date.now() + 90000;
        while ((await read(run)).calibrating) {
            status(run, 'Waiting for the original detector to finish calibration…');
            if (Date.now() >= deadline) throw new Error('Calibration still running. Retry restoration.');
            await pause(1000);
        }
        await change(run, { threshold: original.threshold });
        if (!matches(await read(run), original)) throw new Error('Restoration could not be verified.');
        localStorage.removeItem(key(run.id));
    }

    const percent = value => (100 * value).toFixed(1) + '%';

    function duration(ms) {
        const seconds = Math.max(0, Math.ceil(ms / 1000));
        return seconds < 60 ? seconds + 's' : Math.floor(seconds / 60) + 'm ' + seconds % 60 + 's';
    }

    function renderMeasurement(measurement) {
        $('.js-occ-measure').hidden = !measurement;
        if (!measurement) return;
        const { end, remainingPairs, samples, mayConfirm } = measurement;
        const progress = $('.js-occ-progress');
        const left = end == null ? sampleMs : Math.max(0, end - Date.now());
        progress.max = sampleMs;
        if (end == null) progress.removeAttribute('value');
        else progress.value = sampleMs - left;
        $('.js-occ-sample-count').textContent = end == null ? 'Waiting for the device' : samples + (samples === 1 ? ' fresh sample' : ' fresh samples');
        $('.js-occ-clock').textContent = end == null ? 'Applying settings and waiting for fresh CSI.'
            : duration(left) + ' left in this 30-second measurement';
        const remaining = left + remainingPairs * (sampleMs + settleMs);
        const estimate = duration(remaining) + (mayConfirm ? '–' + duration(remaining + 2 * (sampleMs + settleMs)) : '');
        $('.js-occ-eta').textContent = 'Est. testing time left: ' + estimate;
    }

    function sync() {
        const panel = $('.js-occ-panel');
        if (!panel) return;
        const live = conn.mode === 'direct' && directClient?.connected && !!conn.deviceId;
        const pending = live && saved(conn.deviceId);
        panel.hidden = !live || (!pending && !supported(directClient));
        const busy = !!active;
        const controls = $('.live-sensing-controls');
        if (controls) controls.inert = busy && same(active);
        const dialog = $('.js-occ-dialog');
        if (dialog?.open && live && (panel.hidden || (view && conn.deviceId !== view.id))) dialog.close();
        if (panel.hidden && !dialog?.open) return;
        const passive = $('.js-mon-report-start');
        if (passive) passive.disabled = busy || !!monitorReportSampling;
        const current = !live || (view?.client === directClient && view?.id === conn.deviceId) ? view : null;
        const opener = $('.js-occ-start');
        opener.textContent = busy && same(active) ? 'View progress' : pending ? 'View recovery' : current ? 'View results' : 'Auto-tune';
        opener.disabled = (busy && !same(active)) || (!!monitorReportSampling && !current && !pending);
        $('.js-occ-again').hidden = !live || busy || !!pending || !current;
        $('.js-occ-again').disabled = !!monitorReportSampling;
        $('.js-occ-cancel').hidden = !busy || !same(active) || active.restoring || active.action !== 'compare';
        $('.js-occ-cancel').disabled = !!active?.cancelled;
        $('.js-occ-restore').hidden = !pending || busy;
        $('.js-occ-apply').hidden = !live || !current?.winner || !!pending || busy;
        $('.js-occ-apply').disabled = !!monitorReportSampling;
        $('.js-occ-restore').disabled = !!monitorReportSampling;
        let message = current?.status || (pending ? 'A previous comparison needs restoration.' : '');
        if (!live) message = current && saved(current.id)
            ? 'Connection lost. Reconnect to the same device and choose Restore settings.'
            : 'Device disconnected. Close this window to reconnect.';
        if ($('.js-occ-status').textContent !== message) $('.js-occ-status').textContent = message;
        $('.js-occ-steps').hidden = !current?.steps;
        for (const [step, state] of Object.entries(current?.steps || {})) {
            const badge = $('.js-occ-' + step);
            badge.dataset.state = state;
            badge.textContent = stepLabels[state];
        }
        renderMeasurement(live && !current?.cancelled ? current?.measurement : null);
        $('.js-occ-detail').hidden = !current?.rows.length;
        $('.js-occ-results').textContent = (current?.rows || []).map(row =>
            (row.repeat ? 'Confirmation · ' : '') + label(row) + (row.samples ? ' — median ' + percent(row.median) + ', P10 ' + percent(row.p10)
                + ' (' + row.samples + ' samples)' : ' — ' + row.reason)
        ).join('\n');
    }

    function status(run, message) {
        if (run.status === message) return;
        run.status = message;
        sync();
    }

    function score(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const quantile = q => {
            const pos = (sorted.length - 1) * q;
            return sorted[Math.floor(pos)] + (sorted[Math.ceil(pos)] - sorted[Math.floor(pos)]) * (pos % 1);
        };
        return { samples: values.length, median: quantile(0.5), p10: quantile(0.1) };
    }

    // Require two percentage points above the baseline. Use the lower tail for near ties.
    function improves(candidate, baseline) {
        if (!candidate?.samples || candidate.ready === false || !baseline?.samples) return false;
        return candidate.median - baseline.median >= 0.02 - 1e-9
            || (Math.abs(candidate.median - baseline.median) <= 0.005 + 1e-9
                && candidate.p10 - baseline.p10 >= 0.02 - 1e-9);
    }

    function bestOccupancy(rows) {
        const highest = Math.max(...rows.map(row => row.median));
        return rows.filter(row => highest - row.median <= 0.005 + 1e-9)
            .sort((a, b) => b.p10 - a.p10 || b.median - a.median)[0];
    }

    function comparableRaw(row, reference) {
        return row.traffic_generator_mode === 'wifi_raw' && row.median > 0
            && reference.median - row.median <= 0.01 + 1e-9
            && reference.p10 - row.p10 <= 0.05 + 1e-9;
    }

    function chooseCandidate(rows) {
        const original = rows[0];
        if (!original.samples) return;
        const measured = rows.filter(row => row.ready && row.samples);
        const leader = bestOccupancy(measured);
        const gains = measured.filter(row => improves(row, original));
        const preferred = bestOccupancy(measured.filter(row => comparableRaw(row, leader)
            && (row === original || gains.includes(row)
                || (original.traffic_generator_mode !== 'wifi_raw' && comparableRaw(row, original)))));
        const best = preferred || bestOccupancy(gains);
        if (best !== original) return best;
    }

    async function measure(run, settings, phase, remainingPairs) {
        run.measurement = { remainingPairs, samples: 0, end: null, mayConfirm: run.steps.test === 'active' };
        status(run, phase + ' · ' + label(settings));
        try {
            await change(run, settings);
        } catch (error) {
            if (['unsupported', 'invalid_params', 'http_400', 'http_422'].includes(error.code))
                return { ...settings, reason: 'Unsupported on this device' };
            throw error;
        }
        // The longest supported CSI window is 2 s. Allow one more diagnostic tick for the cache.
        await pause(settleMs);
        const readyDeadline = Date.now() + 15000;
        while (!(await read(run)).ready && Date.now() < readyDeadline) {
            await pause(1000);
        }
        const values = [];
        let previous;
        const end = Date.now() + sampleMs;
        run.measurement.end = end;
        renderMeasurement(run.measurement);
        do {
            const started = Date.now();
            let d = {};
            try { d = await request(run, 'get', 'diagnostics', { fields: ['csi_occupancy'] }); }
            catch (error) { if (!transient(error)) throw error; }
            if (Number.isFinite(d.timestamp_ms) && previous !== d.timestamp_ms
                    && Number.isFinite(d.csi_occupancy) && d.csi_occupancy >= 0 && d.csi_occupancy <= 1) {
                previous = d.timestamp_ms;
                values.push(d.csi_occupancy);
            }
            run.measurement.samples = values.length;
            renderMeasurement(run.measurement);
            await pause(Math.max(0, Math.min(1000 - (Date.now() - started), end - Date.now())));
        } while (Date.now() < end);
        const final = await read(run);
        if (!matches(final, { ...settings, detector: 'high_accuracy', csi_target_pps: run.target }))
            throw new Error('Sensing settings changed during the measurement.');
        return values.length >= 20 ? { ...settings, ...score(values), ready: final.ready } : { ...settings, reason: 'Insufficient fresh samples' };
    }

    async function compare(run) {
        const state = await read(run);
        if (!state.enabled || state.mode !== 'sensing' || state.calibrating
            || !['lightweight', 'high_accuracy'].includes(state.detector) || !Number.isFinite(state.threshold)
            || ![...generators, 'external'].includes(state.traffic_generator_mode)
            || !spacings.includes(state.csi_sample_spacing) || !(state.csi_target_pps > 0))
            throw new Error('Start sensing and wait for calibration before comparing settings.');
        const catalog = await request(run, 'get', 'diagnostics');
        if (!catalog.fields?.some(field => field.name === 'csi_occupancy'))
            throw new Error('Occupancy is unavailable on this device.');
        run.original = config(state);
        run.target = state.csi_target_pps;
        localStorage.setItem(key(run.id), JSON.stringify(run.original));
        await change(run, { detector: 'high_accuracy' });
        const initial = pair(state);
        const modes = generators.filter(mode => mode !== 'wifi_raw' || !/esp32-?c6/i.test(conn.chip));
        if (state.traffic_generator_mode === 'external') modes.push('external');
        const pairs = [initial, ...modes.flatMap(mode => spacings.map(spacing =>
            ({ traffic_generator_mode: mode, csi_sample_spacing: spacing }))).filter(item => !matches(initial, item))];
        run.steps.prepare = 'done';
        run.steps.test = 'active';
        for (const [index, settings] of pairs.entries()) {
            const phase = index === 0 ? 'Measuring your original pair (1 of ' + pairs.length + ')'
                : 'Testing pair ' + (index + 1) + ' of ' + pairs.length;
            run.rows.push(await measure(run, settings, phase, pairs.length - index - 1));
            sync();
        }
        run.steps.test = 'done';
        const baseline = run.rows[0];
        const best = chooseCandidate(run.rows);
        run.steps.confirm = best ? 'active' : 'skipped';
        if (!best) return baseline.samples ? 'Keep the original settings. No clear improvement.' : 'Baseline unavailable. No recommendation.';
        const repeatBase = await measure(run, initial, 'Rechecking original settings (1 of 2)', 1);
        const repeatBest = await measure(run, pair(best), 'Confirming candidate (2 of 2)', 0);
        run.steps.confirm = 'done';
        const confirmation = [repeatBase, repeatBest];
        if (best.traffic_generator_mode === 'wifi_raw') {
            // Retain competing sources so a weak baseline cannot hide a drop below ping or DNS.
            confirmation.push(...run.rows.filter(row => row !== baseline && row.traffic_generator_mode !== 'wifi_raw'));
        }
        run.rows.push({ ...repeatBase, repeat: true }, { ...repeatBest, repeat: true });
        if (chooseCandidate(confirmation) !== repeatBest) return 'Keep the original settings. The result did not repeat.';
        run.winner = pair(best);
        const reason = comparableRaw(repeatBest, repeatBase) && !improves(repeatBest, repeatBase)
            ? 'Preferred at comparable CSI occupancy: ' : 'Recommended for CSI occupancy: ';
        return reason + label(best) + '. Confirmed median '
            + percent(repeatBest.median) + ' vs ' + percent(repeatBase.median) + ' originally. P10 '
            + percent(repeatBest.p10) + ' vs ' + percent(repeatBase.p10) + '.';
    }

    async function start(action) {
        const id = conn.deviceId;
        const client = directClient;
        if (globalThis.navigator?.locks && client && id) {
            return navigator.locks.request(key(id), { ifAvailable: true }, lock => {
                if (conn.deviceId !== id || directClient !== client) return;
                if (lock) return execute(action);
                if ($('.js-occ-dialog')?.open) $('.js-occ-dialog').close();
                toast('Another tab is comparing or restoring settings on this device.');
            });
        }
        return execute(action);
    }

    async function execute(action) {
        if (active || monitorReportSampling || !directClient?.connected || conn.mode !== 'direct' || !conn.deviceId) return;
        const previous = view;
        const run = { client: directClient, id: conn.deviceId, action, rows: [], status: '' };
        const checkpoint = saved(run.id);
        if (action !== 'restore' && (!supported(run.client) || checkpoint)) return;
        if (action === 'restore' && !checkpoint) return;
        if (action === 'compare') {
            run.steps = { prepare: 'active', test: 'waiting', confirm: 'waiting', recover: 'waiting' };
            run.status = 'Checking the device and saving your current settings.';
        } else {
            run.status = action === 'apply' ? 'Applying and verifying the recommended settings…' : 'Restoring original settings…';
        }
        active = view = run;
        stopDiagnosticsPolling();
        sync();
        try {
            if (action === 'restore') {
                run.original = checkpoint;
                await restore(run);
                run.status = 'Original settings restored.';
            } else if (action === 'apply') {
                if (!previous?.winner || previous.client !== run.client || previous.id !== run.id
                    || !matches(await read(run), previous.original)) throw new Error('Settings changed. Run the comparison again.');
                run.rows = previous.rows;
                run.target = previous.target;
                run.complete = previous.complete;
                run.original = previous.original;
                localStorage.setItem(key(run.id), JSON.stringify(run.original));
                await change(run, previous.winner);
                localStorage.removeItem(key(run.id));
                run.applied = previous.winner;
                run.status = 'Applied and verified: ' + label(previous.winner) + '.';
            } else {
                run.status = await compare(run);
                run.complete = true;
            }
        } catch (error) {
            run.winner = null;
            run.status = error.message || 'Comparison could not finish.';
        } finally {
            run.measurement = null;
            if (run.steps) {
                for (const step of ['prepare', 'test', 'confirm']) {
                    if (run.steps[step] === 'active') run.steps[step] = 'stopped';
                    if (run.steps[step] === 'waiting') run.steps[step] = 'skipped';
                }
                run.steps.recover = 'skipped';
            }
            if (action !== 'restore' && saved(run.id)) {
                const result = run.status;
                try {
                    if (run.steps) run.steps.recover = 'active';
                    run.restoring = true;
                    status(run, 'Restoring original settings…');
                    await restore(run);
                    if (run.steps) run.steps.recover = 'done';
                    run.status = result + ' Original settings restored.';
                } catch {
                    if (run.steps) run.steps.recover = 'pending';
                    run.winner = null;
                    run.status = 'Restoration pending. Reconnect to this device and select Restore settings.';
                }
            }
            active = null;
            sync();
            syncDiagnosticsPolling();
        }
        return run;
    }

    function init() {
        const dialog = $('.js-occ-dialog');
        $('.js-occ-start')?.addEventListener('click', () => {
            if (!dialog.open) dialog.showModal();
            if (!active && !saved(conn.deviceId) && !(view && same(view))) return start('compare');
            sync();
        });
        $('.js-occ-close')?.addEventListener('click', () => dialog.close());
        $('.js-occ-again')?.addEventListener('click', () => start('compare'));
        for (const action of ['restore', 'apply']) {
            $('.js-occ-' + action)?.addEventListener('click', () => start(action));
        }
        $('.js-occ-cancel')?.addEventListener('click', () => {
            if (active && !active.restoring) {
                active.cancelled = true;
                status(active, 'Cancelling the current measurement…');
            }
        });
        sync();
    }
    function report() {
        if (active || !view || !same(view) || !view.rows.length) return null;
        return { detector: 'high_accuracy', sample_duration_seconds: sampleMs / 1000, csi_target_pps: view.target,
            complete: !!view.complete, measurements: view.rows,
            recommended: view.winner || view.applied || null, applied: view.applied || null,
            original_pair: pair(view.original), restoration_pending: !!saved(view.id) };
    }

    return { init, sync, start, report, busy: () => !!active };
})();
