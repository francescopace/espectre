/*
 * ESPectre - Website tool contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { AnsiUp } from '../../docs/web/node_modules/ansi_up/ansi_up.js';
import { flashSource, index, read, routeManifest, toolContent } from './fixtures/site_test_helpers.mjs';

function loadFlashCore(globals = {}) {
    const window = {};
    vm.runInNewContext(flashSource, {
        window, URL, Set, Map, Object, String, Number, Math, RegExp,
        Promise, TextEncoder, TextDecoder, Uint8Array, Blob,
        setTimeout, clearTimeout, setInterval, clearInterval, console,
        ...globals,
    });
    return window.ESPectreFlashCore;
}

describe('website tool contracts', () => {
    it('publishes the firmware and SDK artifact channels', () => {
        const sdk = read('docs/web/content/sdk.html');
        for (const { sdkChannel: channel, path } of routeManifest.sdkChannels) {
            assert.match(toolContent.flash, new RegExp(`<option value="${channel}"`));
            assert.match(
                sdk,
                new RegExp(`href="${path}"[\\s\\S]*?data-sdk-version="${channel}"`)
            );
        }
        assert.match(
            read('docs/web/content/sdk/api.html'),
            /data-api-index="\/artifacts\/sdk\/api\/api-index\.json"/
        );
    });

    it('exposes an accessible Web Serial workflow', () => {
        const flash = toolContent.flash;
        const stages = [...flash.matchAll(/data-flash-step="([^"]+)"/g)]
            .map((match) => match[1]);
        assert.deepEqual(stages.sort(), ['error', 'onboarding', 'review', 'select']);
        assert.match(flash, /<progress[^>]+class="flash-progress js-flash-progress"[^>]+max="100"/);
        assert.match(flash, /class="panel flash-progress-card js-flash-progress-card"[^>]+hidden/);
        assert.match(flash, /class="empty-state connection-card flash-connect-card flash-stage js-flash-stage"/);
        assert.match(flash, /class="panel flash-current-panel"/);
        assert.match(flash, /class="panel flash-change-panel"/);
        assert.match(flash, /class="flash-frontend-switch js-flash-frontend-switch" role="group"/);
        assert.match(flash, /<label for="flash-frontend">[^<]+<\/label>\s*<select id="flash-frontend"/);
        assert.match(flash, /class="js-flash-current-install-slot"/);
        assert.match(flash, /class="js-flash-change-install-slot"/);
        assert.match(flash, /class="flash-option flash-erase-option js-flash-force-erase-wrap"/);
        assert.match(flash, /class="modal-backdrop js-flash-erase-modal" hidden/);
        assert.match(flash, /class="modal-backdrop js-flash-flow-modal" hidden/);
        for (const step of ['onboarding', 'error']) {
            assert.match(
                flash,
                new RegExp(
                    `class="modal-card [^"]*js-flash-stage" data-flash-step="${step}" `
                    + 'role="dialog" aria-modal="true"'
                )
            );
        }
        assert.match(flash, /class="js-flash-wifi-form"/);
        assert.match(flash, /class="btn-secondary btn-sm js-flash-configure-wifi"/);
        assert.match(flash, /class="btn-secondary btn-sm js-flash-show-matter" hidden/);
        assert.match(flash, /class="matter-loading js-matter-loading" role="status"/);
        assert.match(flash, /class="flash-console-output js-flash-console-output"[^>]+tabindex="0"/);
        assert.match(flash, /<details class="panel flash-console js-flash-console" data-flash-session-panel hidden>/);
        assert.match(flash, /js-flash-console-reset"[^>]+disabled/);
    });

    it('keeps nested dialogs modal and restores managed inert state', () => {
        class FakeHTMLElement {
            constructor({ hidden = false } = {}) {
                this.children = [];
                this.dataset = {};
                this.hidden = hidden;
                this.inert = false;
                this.parentElement = null;
            }

            append(...children) {
                for (const child of children) {
                    child.parentElement = this;
                    this.children.push(child);
                }
            }
        }

        const body = new FakeHTMLElement();
        const classes = new Set();
        body.classList = {
            contains: (name) => classes.has(name),
            toggle(name, enabled) {
                if (enabled) classes.add(name);
                else classes.delete(name);
            },
        };
        const navigation = new FakeHTMLElement();
        const main = new FakeHTMLElement();
        const page = new FakeHTMLElement();
        const pageContent = new FakeHTMLElement();
        const modal = new FakeHTMLElement();
        const permanentlyInert = new FakeHTMLElement();
        permanentlyInert.inert = true;
        body.append(navigation, main, permanentlyInert);
        main.append(page);
        page.append(pageContent, modal);
        const nodes = [body, navigation, main, page, pageContent, modal, permanentlyInert];
        const document = {
            body,
            querySelectorAll(selector) {
                if (selector === '.modal-backdrop') return [modal];
                if (selector === '[data-modal-inert="true"]') {
                    return nodes.filter((node) => node.dataset.modalInert === 'true');
                }
                return [];
            },
        };
        const core = loadFlashCore({ document, HTMLElement: FakeHTMLElement });

        core.syncModalOpenState();
        assert.equal(body.classList.contains('modal-open'), true);
        assert.equal(navigation.inert, true);
        assert.equal(main.inert, false);
        assert.equal(pageContent.inert, true);
        assert.equal(modal.inert, false);
        assert.equal(permanentlyInert.inert, true);

        modal.hidden = true;
        core.syncModalOpenState();
        assert.equal(body.classList.contains('modal-open'), false);
        assert.equal(navigation.inert, false);
        assert.equal(pageContent.inert, false);
        assert.equal(permanentlyInert.inert, true);
    });

    it('programs only after download and preserves erase ordering with fake dependencies', async () => {
        const core = loadFlashCore();
        const calls = [];
        const image = new Uint8Array([1, 2, 3]);
        await core.programImage({
            download: async () => { calls.push('download'); return image; },
            erase: true,
            eraseFlash: async () => { calls.push('erase'); },
            writeFlash: async (received) => {
                assert.deepEqual([...received], [...image]);
                calls.push('write');
            },
            onErase: () => calls.push('erase-state'),
            onWrite: () => calls.push('write-state'),
        });
        assert.deepEqual(calls, ['download', 'erase-state', 'erase', 'write-state', 'write']);

        calls.length = 0;
        await core.programImage({
            download: async () => { calls.push('download'); return image; },
            erase: false,
            eraseFlash: async () => calls.push('erase'),
            writeFlash: async () => calls.push('write'),
            onErase: () => calls.push('erase-state'),
            onWrite: () => calls.push('write-state'),
        });
        assert.deepEqual(calls, ['download', 'write-state', 'write']);

        calls.length = 0;
        await assert.rejects(core.programImage({
            download: async () => { calls.push('download'); return image; },
            validate: async () => { calls.push('validate'); throw new Error('session ended'); },
            erase: true,
            eraseFlash: async () => calls.push('erase'),
            writeFlash: async () => calls.push('write'),
            onErase: () => calls.push('erase-state'),
            onWrite: () => calls.push('write-state'),
        }), /session ended/);
        assert.deepEqual(calls, ['download', 'validate']);
    });

    it('starts analytics with the active install attempt and limits terminal results to it', () => {
        const core = loadFlashCore();
        const loader = {};
        let reportedAttempt = null;
        const attempt = core.beginInstallAttempt(true, loader, (activeAttempt) => {
            reportedAttempt = activeAttempt;
        });

        assert.equal(reportedAttempt, attempt);
        assert.equal(attempt.erase, true);
        assert.equal(attempt.loader, loader);
        assert.equal(core.shouldReportInstallResult('flash', attempt), true);
        assert.equal(core.shouldReportInstallResult('flash', null), false);
        assert.equal(core.shouldReportInstallResult('wifi', attempt), false);
    });

    it('keeps the connected-device panels visible beside operation results', () => {
        const { stageVisible } = loadFlashCore();
        for (const activeStep of ['onboarding', 'error']) {
            assert.equal(stageVisible('review', activeStep, true), true);
            assert.equal(stageVisible(activeStep, activeStep, true), true);
        }
        assert.equal(stageVisible('review', 'select', true), false);
        assert.equal(stageVisible('review', 'error', false), false);
        assert.equal(stageVisible('error', 'error', false), true);
    });

    it('hides the serial console until the connection step is complete', () => {
        const { sessionPanelVisible } = loadFlashCore();
        assert.equal(sessionPanelVisible('select', true), false);
        assert.equal(sessionPanelVisible('review', true), true);
        assert.equal(sessionPanelVisible('review', false), false);
    });

    it('matches chip and firmware identity without trusting device URLs', () => {
        const core = loadFlashCore();
        const artifacts = [
            { chip_family: 'ESP32', url: '/firmware-esp32.bin' },
            { chip_family: 'ESP32-C3', url: '/firmware-c3.bin' },
            { chip_family: 'ESP32-C6', url: '/firmware-c6.bin' },
        ];
        assert.equal(core.selectArtifact(artifacts, 'ESP32-D0WDQ6 (revision 1)'), artifacts[0]);
        assert.equal(core.selectArtifact(artifacts, 'esp32-c3'), artifacts[1]);
        assert.equal(core.selectArtifact(artifacts, 'ESP32-C3 (QFN32) (revision v0.4)'), artifacts[1]);
        assert.equal(core.selectArtifact(artifacts, 'ESP8685 (QFN28) (revision v0.4)'), artifacts[1]);
        assert.equal(core.selectArtifact(artifacts, 'ESP32-C6 (revision 1)'), artifacts[2]);
        assert.equal(core.selectArtifact(artifacts, 'esp32c6'), artifacts[2]);
        assert.equal(core.selectArtifact(artifacts, 'ESP32-C61 (revision v0.1)'), null);
        assert.equal(core.selectArtifact(artifacts, 'ESP32-S3'), null);
        assert.equal(core.firmwareMatches(
            'esphome', { firmware: 'ESPectre ESPHome', version: 'v1.2.3' }, '1.2.3'
        ), true);
        assert.equal(core.firmwareMatches(
            'esphome', { firmware: 'francescopace.espectre', version: 'v1.2.3' }, '1.2.3'
        ), true);
        assert.equal(core.firmwareMatches(
            'native', { firmware: 'ESPectre ESPHome', version: '1.2.3' }, '1.2.3'
        ), false);
        assert.equal(core.frontendMatches(
            'native', { firmware: 'ESPectre Native', version: '1.0.0' }
        ), true);
        assert.equal(core.frontendMatches(
            'native', { firmware: 'ESPectre ESPHome', version: '1.0.0' }
        ), false);
        assert.equal(core.currentFrontend({ firmware: 'ESPectre Native' }), 'native');
        assert.equal(core.currentFrontend({ firmware: 'ESPHome ESPectre' }), 'esphome');
        assert.equal(core.currentFrontend({ firmware: 'francescopace.espectre' }), 'esphome');
        assert.equal(core.currentFrontend({ firmware: 'unknown' }), '');

        const local = core.settingsUrl(
            'https://test.espectre.dev',
            'http://192.168.1.5/?target=192.168.1.5&ignored=secret',
            true
        );
        assert.equal(local.origin, 'https://test.espectre.dev');
        assert.equal(local.pathname, '/tools/device-settings/');
        assert.equal(local.search, '?target=192.168.1.5');
        assert.equal(core.settingsUrl(
            'http://localhost:8090', 'https://host.invalid/?target=10.0.0.2', false
        ).search, '');
    });

    it('applies the upstream esptool-js SPI register correction for C5 and C6', async () => {
        const core = loadFlashCore();
        for (const chipName of ['ESP32-C5', 'ESP32-C6']) {
            const loader = {
                chip: null,
                async readFlashId() { return this.chip.SPI_REG_BASE; },
            };
            core.applyEsptoolSpiRegisterFix(loader);
            loader.chip = { CHIP_NAME: chipName, SPI_REG_BASE: 0x60002000 };
            assert.equal(await loader.readFlashId(), 0x60003000);
        }
    });

    it('uses the esp-web-tools hard-reset sequence after flashing', async () => {
        const core = loadFlashCore();
        const calls = [];
        await core.hardResetLoader(
            { after: async (...args) => calls.push(['after', ...args]) },
            { setRTS: async (value) => calls.push(['rts', value]) }
        );
        assert.deepEqual(calls, [['rts', true], ['after']]);
    });

    it('recognizes Matter from its existing serial boot markers', () => {
        const core = loadFlashCore();
        const log = [
            'ESP-ROM:esp32s3-20210327',
            'I (100) cpu_start: Project name:     espectre-matter',
            'I (101) cpu_start: App version:      2.8.0-417-g2b49a9c',
            'I (900) app: MATTER_QR=MT:EXAMPLE',
            'I (901) app: MATTER_MANUAL_CODE=12345678901',
        ].join('\n');
        const identity = core.matterIdentity(log);
        assert.equal(identity.firmware, 'ESPectre Matter');
        assert.equal(identity.version, '2.8.0-417-g2b49a9c');
        assert.equal(identity.chipFamily, 'ESP32-S3');
        const codes = core.matterCodes(log);
        assert.equal(codes.qr, 'MT:EXAMPLE');
        assert.equal(codes.manual, '12345678901');
        assert.equal(core.matterIdentity('ESP-ROM:esp32s3-20210327'), null);
    });

    it('replaces the detected firmware identity after changing firmware type', () => {
        const core = loadFlashCore();
        const installed = core.installedInfo(
            'matter',
            '2.9.0',
            'ESP32-S3',
            'Matter',
            { firmware: 'ESPectre Native', version: '2.8.0', chipFamily: 'ESP32-S3' }
        );
        assert.equal(installed.firmware, 'ESPectre Matter');
        assert.equal(installed.version, '2.9.0');
        assert.equal(installed.chipFamily, 'ESP32-S3');
        assert.equal(core.currentFrontend(installed), 'matter');
    });

    it('normalizes the ESPHome project identity after installation', () => {
        const core = loadFlashCore();
        const installed = core.installedInfo(
            'esphome',
            '2.8.0-417-g2b49a9c',
            'ESP32-S3',
            'ESPHome',
            {
                firmware: 'francescopace.espectre',
                version: '2.8.0-417-g2b49a9c',
                chipFamily: 'ESP32-S3'
            }
        );
        assert.equal(installed.firmware, 'ESPectre ESPHome');
        assert.equal(installed.version, '2.8.0-417-g2b49a9c');
    });

    it('reads the Matter version from the installed ESP-IDF app descriptor', async () => {
        const core = loadFlashCore();
        const table = new Uint8Array(0xC00);
        table[0] = 0xAA;
        table[1] = 0x50;
        table[2] = 0x00;
        table[3] = 0x10;
        new DataView(table.buffer).setUint32(4, 0x20000, true);
        new DataView(table.buffer).setUint32(8, 0x3D0000, true);

        const header = new Uint8Array(0x100);
        header[0] = 0xE9;
        new DataView(header.buffer).setUint32(0x20, 0xABCD5432, true);
        header.set(new TextEncoder().encode('2.8.0-417-g2b49a9c'), 0x30);
        header.set(new TextEncoder().encode('espectre-matter'), 0x50);

        const reads = [];
        const info = await core.readInstalledMatterInfo({
            async readFlash(address, length) {
                reads.push([address, length]);
                return address === 0x8000 ? table : header;
            }
        }, 'ESP32-S3');
        assert.deepEqual(reads, [[0x8000, 0xC00], [0x20000, 0x100]]);
        assert.equal(info.firmware, 'ESPectre Matter');
        assert.equal(info.version, '2.8.0-417-g2b49a9c');
        assert.equal(info.chipFamily, 'ESP32-S3');
    });

    it('matches the CLI flash regions while preserving device data', () => {
        const core = loadFlashCore();
        const factory = new Uint8Array(0x50000).fill(0xFF);
        const partition = (offset, type, subtype, address, size) => {
            factory[offset] = 0xAA;
            factory[offset + 1] = 0x50;
            factory[offset + 2] = type;
            factory[offset + 3] = subtype;
            new DataView(factory.buffer).setUint32(offset + 4, address, true);
            new DataView(factory.buffer).setUint32(offset + 8, size, true);
        };
        factory[0] = 0xE9;
        factory[1] = 0;
        factory[23] = 0;
        partition(0x8000, 0x01, 0x02, 0x9000, 0x5000);
        partition(0x8020, 0x01, 0x00, 0xE000, 0x2000);
        partition(0x8040, 0x01, 0x04, 0x10000, 0x1000);
        partition(0x8060, 0x00, 0x10, 0x20000, 0x10000);
        partition(0x8080, 0x00, 0x11, 0x30000, 0x10000);
        factory[0x20000] = 0xE9;
        factory[0x20001] = 0;
        factory[0x20017] = 0;
        const parts = core.preservedParts({ factory, update: null });
        assert.deepEqual(Array.from(parts, (part) => part.address), [0, 0x8000, 0xE000, 0x20000]);
        assert.deepEqual(Array.from(parts, (part) => part.data.length), [32, 0xC00, 0x2000, 32]);
        assert.equal(parts.some((part) => part.address === 0x9000), false);
        assert.equal(parts.some((part) => part.address === 0x30000), false);
    });

    it('bounds and sanitizes the advanced serial console', () => {
        const core = loadFlashCore();
        const state = { active: false };
        const safe = core.sanitizeConsole(
            '\u001b[31mvisible\u001b[0m\n[[secret:start]]\npassword\n[[secret:end]]\ndone\n',
            state
        );
        assert.equal(safe, 'visible\n[sensitive output redacted]\ndone\n');
        assert.equal(state.active, false);
        assert.equal(core.limitConsole('12345', '67890', 6), '567890');
        assert.equal(core.consoleLimit, 512 * 1024);
    });

    it('renders split ANSI color sequences without exposing terminal HTML', () => {
        const renderer = new AnsiUp();
        renderer.url_allowlist = {};
        assert.equal(renderer.ansi_to_html('\x1b[0;'), '');
        const html = renderer.ansi_to_html('32mready <safe>\x1b[0m');
        assert.match(html, /color:rgb\(0,187,0\)/);
        assert.match(html, /ready &lt;safe&gt;/);
        assert.doesNotMatch(html, /\[0;32m|<safe>/);
    });

    it('uses pinned headless browser dependencies', () => {
        const manifest = JSON.parse(read('docs/web/package.json'));
        assert.equal(manifest.dependencies.ansi_up, '6.0.6');
        assert.equal(manifest.dependencies['esptool-js'], '0.6.1');
        assert.equal(manifest.dependencies['improv-wifi-serial-sdk'], '2.8.0');
    });

    it('offers the shared connection picker from every connected browser tool', () => {
        for (const tool of ['configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.match(
                toolContent[tool],
                new RegExp(`<espectre-connection-picker[^>]*data-surface="${tool}"`)
            );
        }
    });

    it('keeps MQTT configuration values in Device settings', () => {
        const configure = toolContent.configure;
        assert.match(configure, /id="cfg-mqtt-scheme"/);
        assert.match(configure, /id="cfg-mqtt-host"/);
        assert.match(configure, /id="cfg-mqtt-port"/);
        assert.match(configure, /id="cfg-topic-prefix"[^>]*value="espectre\/v1\/devices"/);
        assert.match(configure, /id="cfg-mqtt-credentials-clear"/);
        assert.doesNotMatch(configure, /id="cfg-mqtt-user"[^>]*value=/);
        assert.doesNotMatch(configure, /id="cfg-mqtt-pass"[^>]*value=/);
    });

    it('publishes one Raw CSI visualization selector with stable option values', () => {
        const rawCsi = toolContent['raw-csi'];
        const values = [...rawCsi.matchAll(/<option value="([^"]+)"/g)].map((match) => match[1]);
        assert.deepEqual(values, [
            'subcarrier-amplitudes',
            'csi-amplitude-surface',
            'channel-profile-deviation',
            'iq-constellation',
            'relative-phase-trails',
        ]);
        assert.equal((rawCsi.match(/class="js-raw-visualization"/g) || []).length, 1);
    });

    it('keeps unavailable Relay controls inert', () => {
        const template = index.match(/<template id="connection-picker-template">[\s\S]*?<\/template>/)?.[0] || '';
        const relay = template.match(/data-connection-panel="relay"[\s\S]*?<\/section>/)?.[0] || '';
        assert.match(relay, /class="btn-primary" disabled/);
        assert.doesNotMatch(relay, /<input|<select|js-connect/);
    });
});
