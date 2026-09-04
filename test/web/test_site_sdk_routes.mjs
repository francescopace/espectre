/*
 * ESPectre - Website documentation route contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { index, read, routeManifest } from './fixtures/site_test_helpers.mjs';

const contentPath = (route) => `docs/web/content${route.staticPath.slice(0, -1)}.html`;

const deviceReportPaths = [
    'docs/performance/ESP32.md',
    'docs/performance/ESP32-C3.md',
    'docs/performance/ESP32-C5.md',
    'docs/performance/ESP32-C6.md',
    'docs/performance/ESP32-S2.md',
    'docs/performance/ESP32-S3.md',
];
const cppFrontends = ['Native', 'ESPHome', 'Matter'];

function committedCppFrontendMeans(profile) {
    const samples = [];
    for (const path of deviceReportPaths) {
        const report = execFileSync('git', ['show', `HEAD:${path}`], { encoding: 'utf8' });
        const sections = report.split(/^### /m);
        for (const frontend of cppFrontends) {
            const section = sections.find((candidate) => candidate.startsWith(`${frontend} ${profile}\n`));
            if (!section || !section.includes('Result: **PASS**')) continue;
            const detectionUs = Number(section.match(/\| Detection average \| ([0-9.]+) us \|/)?.[1]);
            const runtimeCpu = Number(section.match(/\| Runtime load \| ([0-9.]+)% mean \|/)?.[1]);
            assert.ok(Number.isFinite(detectionUs), `${path} must report ${frontend} ${profile} detection time`);
            assert.ok(Number.isFinite(runtimeCpu), `${path} must report ${frontend} ${profile} runtime CPU`);
            samples.push({ detectionUs, runtimeCpu });
        }
    }
    assert.ok(samples.length > 0, `${profile} must have committed C++ frontend results`);
    return {
        detectionMs: samples.reduce((sum, sample) => sum + sample.detectionUs, 0) / samples.length / 1000,
        runtimeCpu: samples.reduce((sum, sample) => sum + sample.runtimeCpu, 0) / samples.length,
    };
}

describe('website documentation route contracts', () => {
    it('publishes every grouped documentation route through SPA and static content', () => {
        for (const route of routeManifest.routes.filter(({ group }) => ['guides', 'sdk'].includes(group))) {
            assert.match(index, new RegExp(`data-page="${route.name}"`));
            const content = read(contentPath(route));
            assert.match(content, new RegExp(`data-page-path="${route.group}"`));
            assert.match(content, /<h1(?:\s[^>]*)?>/);
        }
    });

    it('links every SDK child route from the SDK landing page', () => {
        const sdk = read('docs/web/content/sdk.html');
        const linkedPaths = [...sdk.matchAll(/<a href="(\/sdk\/[^"#?]+\/)" class="doc-link">/g)]
            .map((match) => match[1])
            .sort();
        const registeredPaths = routeManifest.routes
            .filter((route) => route.group === 'sdk')
            .map((route) => route.staticPath)
            .sort();
        assert.deepEqual(linkedPaths, registeredPaths);
    });

    it('publishes distinct C++, HTTP, and MQTT API contracts', () => {
        const cppApi = read('docs/web/content/sdk/api.html');
        const httpApi = read('docs/web/content/sdk/http-api.html');
        const mqttApi = read('docs/web/content/sdk/mqtt-api.html');
        assert.match(cppApi, /data-api-reference-browser/);
        assert.match(cppApi, /data-api-index="\/artifacts\/sdk\/api\/api-index\.json"/);
        for (const resource of ['capabilities', 'events', 'csi']) {
            assert.match(httpApi, new RegExp(`/espectre/v1/${resource}`));
        }
        assert.match(mqttApi, /espectre\/v1\/devices\/&lt;device-id&gt;/);
        assert.match(mqttApi, /commands\/request/);
        assert.match(mqttApi, /commands\/result/);
    });

    it('keeps published detector means aligned with committed campaign reports', () => {
        const detectors = read('docs/web/content/sdk/detectors.html');
        for (const profile of ['Lightweight', 'High Accuracy']) {
            const row = detectors.match(new RegExp(
                `<tr><td>${profile}</td><td>[^<]+</td><td>[^<]+</td><td>[^<]+</td>`
                + '<td>([0-9.]+) ms</td><td>([0-9.]+)%</td></tr>',
            ));
            assert.ok(row, `${profile} comparison row must publish device means`);
            const expected = committedCppFrontendMeans(profile);
            assert.equal(row[1], expected.detectionMs.toFixed(3));
            assert.equal(row[2], expected.runtimeCpu.toFixed(2));
        }
    });

    it('provides intrinsic dimensions for documentation images', () => {
        for (const route of routeManifest.routes.filter(({ group }) => ['guides', 'sdk'].includes(group))) {
            const images = [...read(contentPath(route)).matchAll(/<img\b[^>]*>/g)].map((match) => match[0]);
            for (const image of images) {
                assert.match(image, /\bwidth="\d+"/);
                assert.match(image, /\bheight="\d+"/);
            }
        }
    });
});
