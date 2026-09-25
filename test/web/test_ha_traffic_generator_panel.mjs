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
    return { id, name: id, can_control: true, can_refresh: true,
        fields: { ownership: field('internal'), generator: field(0), traffic: field(5), traffic_rx: field(100),
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
    let now = 0, timerId = 0, hold = null, fail = false, running = true;
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
            if (fail === true || body?.action === fail) { fail = false; throw new Error('offline'); }
            if (url.pathname.endsWith('/generator')) running = body.action === 'start';
            else if (body) devices.forEach(row => {
                if (body.device_ids.includes(row.id)) row.fields.ownership.value = body.action;
            });
            return { ok: true, json: async () => structuredClone(/\/(status|generator)$/.test(url.pathname) ? {
                csrf: 'test-csrf', generator: { running, rate_pps: 100, sent_packets: calls.length,
                    send_errors: 0, targets: ['239.255.0.1'], port: 5555 },
            } : { devices, results: body?.device_ids.map(id => ({ id, status: 'confirmed' })) }) };
        },
    });
    vm.runInContext(source, context);
    const flush = () => new Promise(resolve => setImmediate(resolve));
    return {
        document, window, calls, devices, elements, timers, flush, sockets,
        get: id => document.getElementById(id),
        failNext(action = true) { fail = action; },
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
const modeButton = fixture => fixture.get('devices').children[0].children[8].children[0].children[0];

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
    modeButton(f).dispatch('click');
    assert.equal(f.get('confirm').open, true);
    await f.tick(5000);
    assert.equal(writes(f).length, 0);
    f.get('confirm').close('cancel');
    await f.tick();
    assert.equal(writes(f).length, 0);
    assert.equal(f.sockets.length, 2);
    modeButton(f).dispatch('click');
    f.get('confirm').close('apply');
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.action === 'external').length, 1);
    assert.equal(modeButton(f).disabled, true);
    await f.tick(2000);
    assert.equal(writes(f).filter(call => call.body.action === 'external').length, 1);
});

it('waits for status before a mode action and never retries a failed write', async () => {
    const f = fixture();
    await f.tick();
    const release = f.holdNext();
    modeButton(f).dispatch('click');
    f.get('confirm').close('apply');
    await f.flush();
    assert.equal(writes(f).length, 0);
    release();
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.action === 'external').length, 1);
    // Fail the write itself, not the preceding status request.
    const internal = f.get('devices').children[0].children[8].children[0].children[1];
    internal.dispatch('click');
    f.failNext('internal');
    f.get('confirm').close('apply');
    await f.flush();
    await f.tick(3000);
    assert.equal(writes(f).filter(call => call.body.action === 'internal').length, 1);
});

it('recovers automatically after a connection failure', async () => {
    const f = fixture();
    await f.tick();
    f.sockets[0].close();
    assert.equal(f.get('devices').children.length, 0);
    assert.equal(f.get('external').disabled, true);
    await f.tick(2000);
    assert.equal(f.get('devices').children.length, 1);
    assert.equal(f.get('notice').textContent, '');
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
    assert.equal(modeButton(f).disabled, true);
    assert.equal(f.get('generator-toggle').disabled, false);
    await f.tick(2000);
    f.devices[0].fields.accepted.value = 101;
    f.sockets.at(-1).push();
    assert.equal(f.get('devices').children[0].children[5].textContent, '101');
    assert.equal(f.devices[0].fields.ownership.value, 'internal');
    f.get('generator-toggle').dispatch('click');
    await f.flush();
    assert.equal(writes(f).filter(call => call.body.action === 'start').length, 1);
    assert.equal(modeButton(f).disabled, false);
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
