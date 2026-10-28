/*
 * ESPectre - Standalone traffic generator web UI contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../tools/traffic_generator_docker/static/app.js', import.meta.url), 'utf8');

class Element {
    hidden = false;
    disabled = false;
    checked = false;
    value = '';
    textContent = '';
    children = [];
    listeners = {};
    classList = { toggle() {} };
    addEventListener(name, callback) { this.listeners[name] = callback; }
    async dispatch(name) { await this.listeners[name]?.({ preventDefault() {}, target: this }); }
    focus() {}
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
    async close(value) { this.open = false; this.returnValue = value; await this.dispatch('close'); }
}

function device(id, mode = 'ping', extra = {}) {
    return { id, name: `Sensor ${id}`, chip: 'esp32c6', frontend: 'native', firmware: '3.1.0',
        ip_address: '192.168.1.20', port: 62587, address: '192.168.1.20', source: 'mdns',
        can_control: true, error: null, modes: ['ping', 'dns', 'dns_tcp', 'external'],
        sensing: { traffic_generator_mode: mode, csi_target_pps: 100, csi_traffic_udp_port: 5555,
            csi_traffic_multicast_group: '239.255.0.1' },
        fields: { generator: 0, traffic_tx: 5, traffic_rx: 100, accepted: 99, occupancy: 0.98, rssi: -54 },
        ...extra };
}

const SETTINGS = { targets: ['239.255.0.1'], protocol: 'icmp', port: 5555, rate_pps: 100,
    source_ip: '', multicast_ttl: 8, dscp: 46 };

function fixture({ signedIn = false, devices = [] } = {}) {
    const elements = new Map();
    const document = Object.assign(new Element(), {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, new Element());
            return elements.get(id);
        },
        createElement: () => new Element(),
    });
    const calls = [];
    const state = { signedIn, settings: structuredClone(SETTINGS), reject: null, devices, results: {} };
    const reply = (status, body) => ({ ok: status < 400, status,
        json: async () => body, text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) });
    const status = () => ({ csrf: 'csrf-token', settings: state.settings,
        generator: { running: true, failed: false, error: null, targets: state.settings.targets,
            protocol: state.settings.protocol, port: state.settings.port, rate_pps: state.settings.rate_pps,
            sent_packets: 10, send_errors: 0, replies_received: 4 },
        devices: { devices: state.devices, scanning: false, scan_error: null, scanned_at: null } });
    const fetch = async (url, options) => {
        const path = new URL(url).pathname;
        const body = options.body === undefined ? undefined : JSON.parse(options.body);
        calls.push({ path, method: options.method, headers: options.headers, body });
        if (path === '/api/login') {
            if (body.token !== 'right-token') return reply(401, 'Wrong token.');
            state.signedIn = true;
            return reply(200, { signed_in: true });
        }
        if (!state.signedIn) return reply(401, 'Sign in required.');
        if (path === '/api/settings') {
            if (state.reject) return reply(400, state.reject);
            state.settings = body;
        }
        if (path === '/api/devices/mode') {
            const results = body.device_ids.map(id => ({ id, status: state.results[id] || 'confirmed',
                message: state.results[id] ? `${state.results[id]} message` : 'Done.' }));
            for (const row of state.devices) {
                if (body.device_ids.includes(row.id) && !state.results[row.id]) row.sensing.traffic_generator_mode = body.mode;
            }
            return reply(200, { ...status(), results });
        }
        return reply(200, status());
    };
    const timers = [];
    const context = vm.createContext({
        document, fetch, location: { pathname: '/', origin: 'http://127.0.0.1:8099' },
        URL, AbortSignal, JSON, Number, String, Object, performance: { now: () => 1000 },
        setTimeout: (callback) => timers.push(callback), clearTimeout() {},
    });
    vm.runInContext(source, context);
    const $ = id => document.getElementById(id);
    const settle = () => new Promise(resolve => setTimeout(resolve, 0));
    return { $, calls, state, settle, poll: async () => { await timers.shift()?.(); await settle(); } };
}

it('asks for the token and opens the generator after signing in', async () => {
    const { $, calls, settle } = fixture();
    await settle();
    assert.equal($('sign-in').hidden, false);
    assert.equal($('app').hidden, true);

    $('token').value = 'wrong-token';
    await $('sign-in-form').dispatch('submit');
    assert.equal($('app').hidden, true);
    assert.equal($('notice').textContent, 'Wrong token.');

    $('token').value = 'right-token';
    await $('sign-in-form').dispatch('submit');
    await settle();
    assert.equal($('app').hidden, false);
    assert.equal($('sign-in').hidden, true);
    assert.equal($('token').value, '');
    assert.equal($('targets-input').value, '239.255.0.1');
    assert.equal($('port').disabled, true);
    assert.deepEqual(calls.filter(call => call.path === '/api/login').map(call => call.body.token),
        ['wrong-token', 'right-token']);
});

it('saves every setting with the request token and keeps the ping port', async () => {
    const { $, calls, state, settle } = fixture({ signedIn: true });
    await settle();
    $('targets-input').value = '192.168.1.20, 192.168.1.21\n\n192.168.1.22';
    $('rate_pps').value = '50';
    $('port').value = '';
    await $('settings').dispatch('input');
    await $('settings').dispatch('submit');
    await settle();

    const save = calls.find(call => call.path === '/api/settings');
    assert.equal(save.method, 'POST');
    assert.equal(save.headers['X-ESPectre-CSRF'], 'csrf-token');
    assert.deepEqual(save.body, { ...SETTINGS, targets: ['192.168.1.20', '192.168.1.21', '192.168.1.22'], rate_pps: 50 });
    assert.deepEqual(state.settings, save.body);
    assert.equal($('save').disabled, true);
});

it('keeps a rejected change and its error visible across polls', async () => {
    const { $, state, settle, poll } = fixture({ signedIn: true });
    await settle();
    state.reject = 'targets must contain only IPv4 addresses';
    $('targets-input').value = '2001:db8::1';
    await $('settings').dispatch('input');
    await $('settings').dispatch('submit');
    await settle();
    await poll();

    assert.equal($('notice').textContent, state.reject);
    assert.equal($('targets-input').value, '2001:db8::1');
    assert.equal($('save').disabled, false);

    await $('revert').dispatch('click');
    assert.equal($('targets-input').value, '239.255.0.1');
    assert.equal($('notice').textContent, '');
});

it('returns to sign-in when the session ends', async () => {
    const { $, state, settle, poll } = fixture({ signedIn: true });
    await settle();
    assert.equal($('app').hidden, false);
    state.signedIn = false;
    await poll();
    assert.equal($('app').hidden, true);
    assert.equal($('sign-in').hidden, false);
});

it('switches selected devices, keeps the failed ones selected, and clears a late confirmation', async () => {
    const { $, calls, state, settle, poll } = fixture({ signedIn: true, devices: [device('a'), device('b'), device('c')] });
    await settle();
    const rows = () => $('devices').children;
    const checked = () => rows().map(row => row.children[0].children[0].checked);
    assert.equal(rows().length, 3);
    assert.equal(rows()[0].children[6].textContent, '98%');
    assert.equal(rows()[0].children[8].children[0].value, 'ping');

    $('select-all').checked = true;
    await $('select-all').dispatch('change');
    state.results = { b: 'unconfirmed', c: 'error' };
    $('bulk-mode').value = 'external';
    await $('bulk-apply').dispatch('click');
    assert.equal($('confirm').open, true);
    await $('confirm').close('apply');
    await settle();

    const change = calls.find(call => call.path === '/api/devices/mode');
    assert.deepEqual(change.body, { device_ids: ['a', 'b', 'c'], mode: 'external' });
    assert.equal(change.headers['X-ESPectre-CSRF'], 'csrf-token');
    assert.deepEqual(checked(), [false, true, true]);
    assert.equal($('devices-notice').textContent.split('\n').length, 2);

    state.results = {};
    state.devices[1].sensing.traffic_generator_mode = 'external';
    await poll();
    assert.deepEqual(checked(), [false, false, true]);
    assert.equal($('devices-notice').textContent, 'Sensor c: error message');
});

it('asks before changing one device and restores its source when cancelled', async () => {
    const { $, calls, settle } = fixture({ signedIn: true, devices: [device('a')] });
    await settle();
    const select = $('devices').children[0].children[8].children[0];
    select.value = 'dns';
    await select.dispatch('change');
    assert.equal($('confirm').open, true);
    await $('confirm').close('cancel');
    assert.equal(select.value, 'ping');
    assert.equal(calls.filter(call => call.path === '/api/devices/mode').length, 0);
});

it('warns when an external device does not match the generator', async () => {
    const { $, settle } = fixture({ signedIn: true, devices: [device('a', 'external', { sensing: {
        traffic_generator_mode: 'external', csi_target_pps: 50, csi_traffic_udp_port: 5555,
        csi_traffic_multicast_group: '239.255.0.2' } }), device('b', 'ping')] });
    await settle();
    const hint = row => $('devices').children[row].children[8].children[1].textContent;
    assert.match(hint(0), /50 pps/);
    assert.match(hint(0), /239\.255\.0\.2/);
    assert.equal(hint(1), '');
});

it('offers each device only the sources it supports, and their union in bulk', async () => {
    const { $, settle } = fixture({ signedIn: true, devices: [
        device('a'), device('b', 'wifi_raw', { chip: 'esp32s3', modes: ['ping', 'dns', 'dns_tcp', 'wifi_raw', 'external'] })] });
    await settle();
    const options = select => select.children.map(option => option.value);
    const picker = row => $('devices').children[row].children[8].children[0];
    assert.deepEqual(options(picker(0)), ['ping', 'dns', 'dns_tcp', 'external']);
    assert.deepEqual(options(picker(1)), ['ping', 'dns', 'dns_tcp', 'wifi_raw', 'external']);
    assert.equal(picker(1).value, 'wifi_raw');

    const check = row => $('devices').children[row].children[0].children[0];
    check(0).checked = true;
    await check(0).dispatch('change');
    assert.deepEqual(options($('bulk-mode')), ['ping', 'dns', 'dns_tcp', 'external', 'internal']);
    check(1).checked = true;
    await check(1).dispatch('change');
    assert.deepEqual(options($('bulk-mode')), ['ping', 'dns', 'dns_tcp', 'wifi_raw', 'external', 'internal']);
    assert.equal($('bulk-mode').value, 'external');
});
