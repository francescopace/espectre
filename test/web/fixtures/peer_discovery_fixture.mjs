/*
 * ESPectre - Deterministic peer discovery browser fixture
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

export function peer(overrides = {}) {
    return {
        device_id: '0123456789abcdef',
        instance: 'Kitchen sensor',
        hostname: 'espectre-0123456789abcdef',
        name: 'Kitchen',
        frontend: 'native',
        txt_version: 2,
        protocol_version: 1,
        transport: 'http',
        path: '/espectre/v1/request',
        events: '/espectre/v1/events',
        firmware: '4.0.0-dev',
        chip: 'esp32c3',
        port: 80,
        capabilities: ['config', 'monitor', 'peer_discovery'],
        addresses: ['192.168.1.42'],
        ...overrides
    };
}

export function result(devices, overrides = {}) {
    return {
        schema_version: 2,
        elapsed_ms: 42,
        status: 'complete',
        truncated: false,
        rejected_results: 0,
        devices,
        ...overrides
    };
}

export const peerDiscoveryScenarios = Object.freeze({
    multiFrontend: result([
        peer(),
        peer({ device_id: '1111111111111111', instance: 'Collector', hostname: 'espectre-1111111111111111', name: 'Collector', frontend: 'streamer', chip: 'esp32s3', capabilities: ['collect', 'monitor'], addresses: ['192.168.1.43'] }),
        peer({ device_id: '2222222222222222', instance: 'ESPHome sensor', hostname: 'espectre-2222222222222222', name: 'ESPHome sensor', frontend: 'esphome', chip: 'esp32c6', port: 6054, capabilities: ['config', 'monitor'], addresses: ['192.168.1.44'] }),
        peer({ device_id: '3333333333333333', instance: 'Matter sensor', hostname: 'espectre-3333333333333333', name: 'Matter sensor', frontend: 'matter', chip: 'esp32c3', capabilities: ['config', 'monitor'], addresses: ['192.168.1.45'] })
    ]),
    mixedAddresses: result([
        peer(),
        peer({ device_id: 'fedcba9876543210', instance: 'Office sensor', hostname: 'espectre-fedcba9876543210', name: 'Office', frontend: 'esphome', chip: 'esp32s3', port: 6054, capabilities: ['monitor'], addresses: ['fd12:3456:789a::42'] })
    ]),
    duplicateIdentity: result([peer(), peer({ hostname: 'conflicting-host', addresses: ['192.168.1.43'] })]),
    malformed: result([peer({ path: '/espectre/v1/ws' })]),
    websocket: result([peer({ txt_version: 1, transport: 'websocket', path: '/espectre/v1/ws', events: '' })]),
    oversized: result([peer({ name: 'x'.repeat(64) })]),
    nonLocal: result([peer({ addresses: ['203.0.113.42'] })]),
    malformedAddress: result([peer({ addresses: ['192.168.999.42'] })]),
    partial: result([peer()], { rejected_results: 3 }),
    truncated: result([peer()], { truncated: true, rejected_results: 4 }),
    timeout: result([], { elapsed_ms: 3000, status: 'timeout' })
});
