/*
 * ESPectre - Home Assistant traffic generator panel contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../tools/ha_traffic_generator_addon/static/panel.js', import.meta.url), 'utf8');

class Element {
    children = [];
    listeners = {};
    textContent = '';
    classList = { toggle() {} };
    addEventListener(name, callback) { this.listeners[name] = callback; }
    dispatch(name) { this.listeners[name]?.({ target: this }); }
    setAttribute(name, value) { this[name] = value; }
    append(...children) { children.forEach(child => this.insertBefore(child, null)); }
    replaceChildren(...children) { [...this.children].forEach(child => child.remove()); this.append(...children); }
    insertBefore(child, next) {
        child.remove();
        this.children.splice(next ? this.children.indexOf(next) : this.children.length, 0, child);
        child.parent = this;
    }
    remove() {
        if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1);
        this.parent = null;
    }
    showModal() { this.open = true; }
    close(value) { this.open = false; this.returnValue = value; this.dispatch('close'); }
}

function device(id = 'sensor-a') {
    const field = value => ({ value, status: 'available', entity_id: 'sensor.sample' });
    return { id, name: id, can_control: true, can_refresh: true, can_set_spacing: true,
        modes: ['ping', 'dns', 'wifi_raw', 'external'],
        fields: { source: field('ping'), spacing: field('half_slot'), generator: field(0), traffic: field(5), traffic_rx: field(100),
            accepted: field(99), occupancy: field(98), rssi: field(-54) } };
}

function fixture(count = 1) {
    const elements = new Map();
    const document = Object.assign(new Element(), {
        hidden: false,
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, new Element());
            return elements.get(id);
        },
        createElement: () => new Element(),
    });
    const window = new Element(), timers = new Map(), calls = [], sockets = [];
    const devices = Array.from({ length: count }, (_, index) => device(`sensor-${index}`));
    let now = 0, timerId = 0, hold = null, fail = false, running = true, results = {};
    const generator = () => ({ running, rate_pps: 100, sent_packets: calls.length,
        send_errors: 0, targets: ['239.255.0.1'], port: 5555 });
    class Socket {
        constructor(url) {
            assert.equal(url.href, 'ws://test/ingress/api/stream');
            sockets.push(this);
            setImmediate(() => { if (!this.closed) this.onopen?.(); });
        }
        send(data) {
            assert.deepEqual(JSON.parse(data), { csrf: 'test-csrf' });
            this.push();
        }
        push(payload = { devices }) {
            if (!this.closed) this.onmessage?.({ data: JSON.stringify({ generator: generator(), ...payload }) });
        }
        close() { this.closed = true; this.onclose?.(); }
    }
    const context = vm.createContext({
        document, window, URL, AbortSignal, WebSocket: Socket, location: { pathname: '/ingress/', origin: 'http://test' },
        performance: { now: () => now },
        setTimeout(callback, delay) { timers.set(++timerId, { callback, at: now + delay }); return timerId; },
        clearTimeout: id => timers.delete(id),
        async fetch(url, options) {
            const body = options.body && JSON.parse(options.body);
            calls.push({ path: url.pathname, body, headers: options.headers });
            if (hold) { const wait = hold; hold = null; await wait; }
            const option = body && (body.spacing ?? body.mode);
            if (fail === true || (body && (body.action ?? option) === fail)) { fail = false; throw new Error('offline'); }
            if (typeof fail === 'number') { const status = fail; fail = false; return { ok: false, status, headers: new Map() }; }
            if (url.pathname.endsWith('/generator')) running = body.action === 'start';
            else if (body) devices.forEach(row => {
                if (body.device_ids.includes(row.id) && !results[row.id]) row.fields[body.spacing ? 'spacing' : 'source'].value = option;
            });
            return { ok: true, json: async () => structuredClone(/\/(status|generator)$/.test(url.pathname) ? {
                csrf: 'test-csrf', generator: { running, rate_pps: 100, sent_packets: calls.length,
                    send_errors: 0, targets: ['239.255.0.1'], port: 5555 },
            } : { devices, results: body?.device_ids.map(id => ({ id, status: results[id] || 'confirmed', option,
                message: results[id] ? `${results[id]} message` : `Now ${option}.` })) }) };
        },
    });
    vm.runInContext(source, context);
    const flush = () => new Promise(resolve => setImmediate(resolve));
    return {
        document, window, calls, devices, elements, timers, flush, sockets,
        get: id => document.getElementById(id),
        failNext(action = true) { fail = action; },
        resultNext(statuses) { results = statuses; },
        holdNext() { let release; hold = new Promise(resolve => { release = resolve; }); return release; },
        async tick(milliseconds = 0) {
            const end = now + milliseconds;
            for (;;) {
                const next = [...timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
                if (!next) break;
                now = next[1].at;
                timers.delete(next[0]);
                next[1].callback();
                await flush();
            }
            now = end;
            await flush();
        },
    };
}

const writes = fixture => fixture.calls.filter(call => call.body);
const picker = (fixture, row = 0) => fixture.get('devices').children[row].children[8].children[0];
const pick = (fixture, mode, row = 0) => { const select = picker(fixture, row); select.value = mode; select.dispatch('change'); };
const option = (select, mode) => select.children.find(child => child.value === mode);

it('applies pushed diagnostics without HTTP polling, replacing controls, or changing modes', async () => {
    const f = fixture();
    await f.tick();
    const row = f.get('devices').children[0], checkbox = row.children[0].children[0];
    checkbox.checked = true;
    checkbox.dispatch('change');
    f.devices[0].fields.accepted.value = 100;
    f.sockets[0].push();
    assert.equal(f.get('devices').children[0], row);
    assert.equal(row.children[0].children[0], checkbox);
    assert.equal(checkbox.checked, true);
    assert.equal(row.children[2].textContent, '0');
    assert.equal(row.children[3].textContent, '5');
    assert.equal(row.children[4].textContent, '100');
    assert.equal(row.children[5].textContent, '100');
    await f.tick(2000);
    assert.equal(writes(f).length, 0);
    assert.equal(f.calls.length, 1);
});

it('updates the device IP from snapshots and handles missing addresses', async () => {
    const f = fixture();
    await f.tick();
    const ip = f.get('devices').children[0].children[1].children[1];
    assert.equal(ip.textContent, '—');
    f.devices[0].ip_address = '192.168.1.42';
    f.sockets[0].push();
    assert.equal(ip.textContent, '192.168.1.42');
    f.devices[0].ip_address = null;
    f.sockets[0].push();
    assert.equal(ip.textContent, '—');
});

it('shows the chip before the address or unavailable status when known', async () => {
    const f = fixture();
    await f.tick();
    const detail = f.get('devices').children[0].children[1].children[1];
    f.devices[0].chip = 'ESP32-C3';
    f.devices[0].ip_address = '192.168.1.42';
    f.sockets[0].push();
    assert.equal(detail.textContent, 'ESP32-C3 · 192.168.1.42');
    f.devices[0].reason = 'Unavailable';
    f.sockets[0].push();
    assert.equal(detail.textContent, 'ESP32-C3 · Unavailable');
    f.devices[0].chip = null;
    f.sockets[0].push();
    assert.equal(detail.textContent, 'Unavailable');
});

it('shows bulk actions only while devices are selected', async () => {
    const f = fixture();
    await f.tick();
    const footer = f.get('bulk-actions');
    const checkbox = f.get('devices').children[0].children[0].children[0];
    assert.equal(footer.hidden, true);
    checkbox.checked = true;
    checkbox.dispatch('change');
    assert.equal(footer.hidden, false);
    checkbox.checked = false;
    checkbox.dispatch('change');
    assert.equal(footer.hidden, true);
    f.get('select-all').checked = true;
    f.get('select-all').dispatch('change');
    assert.equal(footer.hidden, false);
    f.devices[0].can_control = false;
    f.sockets[0].push();
    assert.equal(footer.hidden, true);
});

it('links device names to the HA device page outside the ingress frame', async () => {
    const f = fixture();
    await f.tick();
    const name = f.get('devices').children[0].children[1].children[0];
    assert.equal(name.href, '/config/devices/device/sensor-0');
    assert.equal(name.target, '_top');
    f.devices[0].name = 'Renamed device';
    f.sockets[0].push();
    assert.equal(name.textContent, 'Renamed device');
    assert.equal(name.href, '/config/devices/device/sensor-0');
});

it('replaces the IP with availability status and restores it on recovery', async () => {
    const f = fixture();
    await f.tick();
    const cell = f.get('devices').children[0].children[1];
    const ip = cell.children[1];
    f.devices[0].reason = 'Traffic ownership entity is unavailable.';
    f.devices[0].can_control = false;
    f.devices[0].ip_address = '192.168.1.42';
    f.sockets[0].push();
    assert.equal(cell.children.length, 2);
    assert.equal(ip.textContent, 'Unavailable');
    assert.equal(ip.title, f.devices[0].reason);
    f.devices[0].ip_address = null;
    f.sockets[0].push();
    assert.equal(ip.textContent, 'Unavailable');
    f.devices[0].reason = null;
    f.devices[0].can_control = true;
    f.devices[0].ip_address = '192.168.1.42';
    f.sockets[0].push();
    assert.equal(ip.textContent, '192.168.1.42');
});

it('updates RSSI in dBm and clears unavailable measurements', async () => {
    const f = fixture();
    await f.tick();
    const cell = f.get('devices').children[0].children[7];
    assert.equal(cell.textContent, '-54 dBm');
    f.devices[0].fields.rssi.value = -61;
    f.sockets[0].push();
    assert.equal(cell.textContent, '-61 dBm');
    f.devices[0].fields.rssi.value = null;
    f.sockets[0].push();
    assert.equal(cell.textContent, '—');
});

it('pauses while hidden or unloaded and resumes when visible', async () => {
    const f = fixture();
    await f.tick();
    f.document.hidden = true;
    f.document.dispatch('visibilitychange');
    assert.equal(f.sockets[0].closed, true);
    const count = f.calls.length;
    await f.tick(10000);
    assert.equal(f.calls.length, count);
    f.document.hidden = false;
    f.document.dispatch('visibilitychange');
    await f.tick();
    assert.equal(f.sockets.length, 2);
    f.window.dispatch('pagehide');
    const after = f.calls.length;
    await f.tick(10000);
    assert.equal(f.calls.length, after);
    f.window.dispatch('pageshow');
    await f.tick();
    assert.equal(f.sockets.length, 3);
});

it('does not open a stream when hidden during slow connection setup', async () => {
    const f = fixture();
    const release = f.holdNext();
    await f.tick();
    const count = f.calls.length;
    await f.tick(5000);
    assert.equal(f.calls.length, count);
    f.window.dispatch('pagehide');
    release();
    await f.flush();
    await f.tick(10000);
    assert.equal(f.calls.length, count);
    assert.equal(f.timers.size, 0);
    assert.equal(f.sockets.length, 0);
});

it('pauses for confirmation and sends a mode command only after applying', async () => {
    const f = fixture();
    await f.tick();
    assert.deepEqual(picker(f).children.map(child => child.value), ['ping', 'dns', 'wifi_raw', 'external']);
    assert.equal(picker(f).value, 'ping');
    pick(f, 'external');
    assert.equal(f.get('confirm').open, true);
    await f.tick(5000);
    assert.equal(writes(f).length, 0);
    f.get('confirm').close('cancel');
    await f.tick();
    assert.equal(writes(f).length, 0);
    assert.equal(picker(f).value, 'ping');
    assert.equal(f.sockets.length, 2);
    pick(f, 'external');
    f.get('confirm').close('apply');
    await f.flush();
    assert.deepEqual(writes(f).at(-1).body, { mode: 'external', device_ids: ['sensor-0'] });
    assert.equal(picker(f).value, 'external');
    await f.tick(2000);
    assert.equal(writes(f).filter(call => call.body.mode === 'external').length, 1);
});

it('waits for status before a mode action and never retries a failed write', async () => {
    const f = fixture();
    await f.tick();
    const release = f.holdNext();
    pick(f, 'external');
    f.get('confirm').close('apply');
    await f.flush();
    assert.equal(writes(f).length, 0);
    release();
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.mode === 'external').length, 1);
    // Fail the write itself, not the preceding status request.
    await f.tick(2000);
    pick(f, 'dns');
    f.failNext('dns');
    f.get('confirm').close('apply');
    await f.flush();
    await f.tick(3000);
    assert.equal(writes(f).filter(call => call.body.mode === 'dns').length, 1);
});

it('recovers automatically after a connection failure', async () => {
    const f = fixture();
    await f.tick();
    f.sockets[0].close();
    assert.equal(f.get('devices').children.length, 0);
    assert.equal(f.get('bulk-apply').disabled, true);
    await f.tick(2000);
    assert.equal(f.get('devices').children.length, 1);
    assert.equal(f.get('notice').textContent, '');
});

it('reports an expired ingress session apart from a lost connection and keeps retrying', async () => {
    const f = fixture();
    await f.tick();
    f.sockets[0].close();
    const lost = f.get('notice').textContent;
    f.failNext(401);
    await f.tick(2000);
    assert.equal(f.sockets.length, 1);
    assert.equal(f.get('notice').textContent, 'Home Assistant session expired. Reload the page.');
    assert.notEqual(lost, f.get('notice').textContent);
    await f.tick(2000);
    assert.equal(f.get('devices').children.length, 1);
    assert.equal(f.get('notice').textContent, '');
});

it('reports an expired ingress session when a command cannot reach Home Assistant', async () => {
    const f = fixture();
    await f.tick();
    pick(f, 'external');
    f.failNext(401);
    f.get('confirm').close('apply');
    await f.flush();
    assert.equal(writes(f).length, 0);
    assert.equal(f.get('notice').textContent, 'Home Assistant session expired. Reload the page.');
    await f.tick(2000);
    f.failNext(401);
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    assert.equal(writes(f).length, 0);
    assert.equal(f.get('running').textContent, 'Disconnected');
    assert.equal(f.get('notice').textContent, 'Home Assistant session expired. Reload the page.');
});

it('displays a large streamed inventory without browser diagnostic commands', async () => {
    const f = fixture(33);
    await f.tick();
    await f.tick(1000);
    assert.equal(f.get('devices').children.length, 33);
    assert.equal(writes(f).length, 0);
});

it('starts and stops only UDP while diagnostics and the panel remain available', async () => {
    const f = fixture();
    await f.tick();
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    assert.equal(writes(f)[0].path, '/ingress/api/generator');
    assert.equal(writes(f)[0].body.action, 'stop');
    assert.equal(option(picker(f), 'external').disabled, true);
    assert.equal(f.get('generator-toggle').disabled, false);
    await f.tick(2000);
    f.devices[0].fields.accepted.value = 101;
    f.sockets.at(-1).push();
    assert.equal(f.get('devices').children[0].children[5].textContent, '101');
    assert.equal(f.devices[0].fields.source.value, 'ping');
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.action === 'start').length, 1);
    assert.equal(option(picker(f), 'external').disabled, false);
});

it('does not retry a failed generator command automatically', async () => {
    const f = fixture();
    await f.tick();
    f.failNext('stop');
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    await f.tick(3000);
    assert.equal(writes(f).filter(call => call.body.action === 'stop').length, 1);
});

it('keeps UDP controls available when only HA diagnostics fail', async () => {
    const f = fixture();
    await f.tick();
    f.sockets[0].push({ devices: [], error: 'HA unavailable' });
    assert.equal(f.get('devices').children.length, 0);
    assert.equal(f.get('generator-toggle').disabled, false);
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.action === 'stop').length, 1);
});

it('keeps only failed or unconfirmed devices selected and clears a late confirmation', async () => {
    const f = fixture(3);
    await f.tick();
    const checked = () => f.get('devices').children.map(row => row.children[0].children[0].checked);
    f.get('select-all').checked = true;
    f.get('select-all').dispatch('change');
    f.resultNext({ 'sensor-1': 'unconfirmed', 'sensor-2': 'error' });
    f.get('bulk-mode').value = 'external';
    f.get('bulk-apply').dispatch('click');
    f.get('confirm').close('apply');
    await f.flush();
    await f.tick();
    assert.deepEqual(writes(f).at(-1).body.device_ids, ['sensor-0', 'sensor-1', 'sensor-2']);
    assert.deepEqual(checked(), [false, true, true]);
    assert.equal(f.get('notice').textContent.split('\n').length, 2);
    assert.equal(f.get('bulk-apply').disabled, false);

    f.resultNext({});
    f.devices[1].fields.source.value = 'external';
    f.sockets.at(-1).push();
    assert.deepEqual(checked(), [false, false, true]);
    assert.equal(f.get('notice').textContent, 'sensor-2: error message');

    f.get('bulk-mode').value = 'external';
    f.get('bulk-apply').dispatch('click');
    f.get('confirm').close('apply');
    await f.flush();
    await f.tick();
    assert.deepEqual(writes(f).at(-1).body.device_ids, ['sensor-2']);
    assert.deepEqual(checked(), [false, false, false]);
    assert.equal(f.get('notice').textContent, '');
});

it('offers each mode a selected device has, plus the previous internal source, in bulk', async () => {
    const f = fixture(2);
    f.devices[1].modes = ['ping', 'dns', 'external'];
    await f.tick();
    f.get('devices').children[1].children[0].children[0].checked = true;
    f.get('devices').children[1].children[0].children[0].dispatch('change');
    assert.deepEqual(f.get('bulk-mode').children.map(child => child.value), ['ping', 'dns', 'external', 'internal']);
    // The empty selection offered only `internal`; the first selection still starts on `external`.
    assert.equal(f.get('bulk-mode').value, 'external');
    f.get('select-all').checked = true;
    f.get('select-all').dispatch('change');
    assert.deepEqual(f.get('bulk-mode').children.map(child => child.value),
        ['ping', 'dns', 'wifi_raw', 'external', 'internal']);
    f.get('bulk-mode').value = 'wifi_raw';
    f.get('bulk-apply').dispatch('click');
    f.get('confirm').close('apply');
    await f.flush();
    assert.deepEqual(writes(f).at(-1).body, { mode: 'wifi_raw', device_ids: ['sensor-0', 'sensor-1'] });
});

it('sets sample spacing without confirmation and keeps the selection for source changes', async () => {
    const f = fixture(2);
    await f.tick();
    const spacing = row => f.get('devices').children[row].children[9].children;
    const checkbox = f.get('devices').children[0].children[0].children[0];
    checkbox.checked = true;
    checkbox.dispatch('change');
    f.devices[1].can_set_spacing = false;
    f.sockets[0].push();
    assert.equal(spacing(0)[0].value, 'half_slot');
    assert.deepEqual([spacing(1)[0].hidden, spacing(1)[1].hidden], [true, false]);

    f.resultNext({ 'sensor-0': 'unconfirmed' });
    spacing(0)[0].value = 'distinct';
    spacing(0)[0].dispatch('change');
    await f.flush();
    await f.tick();
    assert.equal(f.get('confirm').open, undefined);
    assert.equal(writes(f).at(-1).path, '/ingress/api/spacing');
    assert.deepEqual(writes(f).at(-1).body, { spacing: 'distinct', device_ids: ['sensor-0'] });
    assert.equal(checkbox.checked, true);
    assert.equal(f.get('notice').textContent, 'sensor-0: unconfirmed message');

    // The device confirms later; the source value must not be mistaken for it.
    f.resultNext({});
    f.devices[0].fields.spacing.value = 'distinct';
    f.sockets.at(-1).push();
    assert.equal(f.get('notice').textContent, '');
    assert.equal(f.devices[0].fields.source.value, 'ping');
});
