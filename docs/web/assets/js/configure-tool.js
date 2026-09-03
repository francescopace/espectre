/*
 * ESPectre - Configure tool
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    function applyConfigureMqttCredentialPolicy(presetName) {
        const username = document.getElementById('cfg-mqtt-user');
        const password = document.getElementById('cfg-mqtt-pass');
        const isFlespi = presetName === 'flespi';
        username.placeholder = isFlespi ? 'Enter a new token' : 'Keep saved username';
        password.placeholder = isFlespi ? 'Flespi does not use one' : 'Keep saved password';
        password.toggleAttribute('data-preset-locked', isFlespi);
        password.title = isFlespi ? 'Flespi does not use a password' : '';
        password.disabled = isFlespi;
        if (isFlespi) password.value = '';
        syncConfigureMqttCredentialMode();
    }

    function syncConfigureMqttCredentialMode() {
        const clear = document.getElementById('cfg-mqtt-credentials-clear').checked;
        const username = document.getElementById('cfg-mqtt-user');
        const password = document.getElementById('cfg-mqtt-pass');
        const isFlespi = document.getElementById('cfg-mqtt-preset').value === 'flespi';
        username.disabled = clear;
        password.disabled = clear || isFlespi;
        username.placeholder = clear ? 'No username' : isFlespi ? 'Enter a new token' : 'Keep saved username';
        password.placeholder = clear ? 'No password' : isFlespi ? 'Flespi does not use one' : 'Keep saved password';
        if (clear) {
            username.value = '';
            password.value = '';
        }
    }

    function applyMqttPresetFieldLocks(_target, preset) {
        const fields = { scheme: 'cfg-mqtt-scheme', host: 'cfg-mqtt-host', port: 'cfg-mqtt-port' };
        const locked = new Set(preset.locked || []);
        Object.entries(fields).forEach(([name, id]) => {
            const input = document.getElementById(id);
            if (!input) return;
            const isLocked = locked.has(name);
            if (input instanceof HTMLSelectElement) input.disabled = isLocked;
            else input.readOnly = isLocked;
            input.toggleAttribute('data-preset-locked', isLocked);
            input.title = isLocked ? 'Set by the selected broker preset' : '';
        });
    }

    function browserBrokerHost(host) {
        return String(host || '')
            .trim()
            .replace(/^mqtts?:\/\//, '')
            .replace(/:\d+$/, '');
    }

    function configuredBrokerPreset(scheme, host, port) {
        const normalizedHost = browserBrokerHost(host).toLowerCase();
        if (scheme === 'mqtt' && normalizedHost === 'homeassistant.local'
                && Number(port) === Number(MQTT_PRESETS.home_assistant.configure.port)) {
            return 'home_assistant';
        }
        if (normalizedHost === 'mqtt.flespi.io') return 'flespi';
        if (normalizedHost.endsWith('.hivemq.cloud')) return 'hivemq_cloud';
        if (normalizedHost.endsWith('.emqxsl.com') || normalizedHost.endsWith('.emqx.cloud')) {
            return 'emqx_cloud';
        }
        const localIpv4 = /^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)/.test(normalizedHost);
        if (normalizedHost === 'localhost' || normalizedHost.endsWith('.local') || localIpv4) {
            return 'lan_broker';
        }
        return 'cloud_broker';
    }

    function applyConfigureMqttPreset(presetName, { clearCredentials = true } = {}) {
        const select = document.getElementById('cfg-mqtt-preset');
        const resolvedName = MQTT_PRESETS[presetName] ? presetName : 'cloud_broker';
        const preset = MQTT_PRESETS[resolvedName];
        select.value = resolvedName;
        document.getElementById('cfg-mqtt-scheme').value = preset.configure.scheme;
        document.getElementById('cfg-mqtt-host').value = preset.configure.host;
        document.getElementById('cfg-mqtt-host').placeholder = preset.configure.hostPlaceholder;
        document.getElementById('cfg-mqtt-port').value = preset.configure.port;
        document.getElementById('cfg-topic-prefix').value = MQTT_FORM_DEFAULTS.topicPrefix;
        applyMqttPresetFieldLocks('configure', preset.configure);
        applyConfigureMqttCredentialPolicy(resolvedName);
        if (clearCredentials) {
            document.getElementById('cfg-mqtt-user').value = '';
            document.getElementById('cfg-mqtt-pass').value = '';
            document.getElementById('cfg-mqtt-credentials-clear').checked = false;
        }
        syncConfigureMqttCredentialMode();
    }

    /* =============================================================== flash */

    const flash = {
        manifests: {}, installUrl: null,
        installerObserver: null, watchedDialogs: new WeakSet(), catalogReports: new Set(),
        detectedChip: '', supportedChipLabels: [], modalReturnFocus: null,
        refreshRequest: 0, targetVersion: '', usbDialog: null, usbPortInfo: null,
        usbReleaseTimer: null
    };

    /*
     * Presentation order for the Flash selectors and installer manifest.
     * Anything not listed keeps its manifest order and lands after the listed
     * entries, so a new frontend or chip still shows up without touching this code.
     */
    const FRONTEND_ORDER = ['native', 'esphome', 'matter'];
    const CHIP_ORDER = ['esp32', 'esp32s3', 'esp32c3', 'esp32c5', 'esp32c6'];
    const FLASH_CHIP_FOUND_RE = /Initialized\. Found ([A-Z0-9-]+)/i;
    const FLASH_CHIP_UNSUPPORTED_RE = /Your ([A-Z0-9-]+) board is not supported/i;

    function flashUsbId(info) {
        if (!info || (!info.usbVendorId && !info.usbProductId)) return '';
        const hex = (value) => Number(value || 0).toString(16).toUpperCase().padStart(4, '0');
        return hex(info.usbVendorId) + ':' + hex(info.usbProductId);
    }

    function flashUsbIdentity() {
        const chip = String(flash.detectedChip || '').toUpperCase();
        const portId = flashUsbId(flash.usbPortInfo);
        return {
            deviceName: chip || portId || 'USB device',
            chip,
            deviceId: portId,
            firmwareVersion: flash.targetVersion
        };
    }

    function releaseUsbConnection(dialog) {
        if (flash.usbDialog !== dialog) return;
        clearTimeout(flash.usbReleaseTimer);
        flash.usbReleaseTimer = null;
        flash.usbDialog = null;
        flash.usbPortInfo = null;
        dropdownOpen = false;
        renderConnection();
        syncFirmwareUpdateNotice();
    }

    function scheduleUsbConnectionRelease(dialog) {
        if (flash.usbDialog !== dialog) return;
        clearTimeout(flash.usbReleaseTimer);
        flash.usbReleaseTimer = setTimeout(() => releaseUsbConnection(dialog), 100);
    }

    function activateUsbConnection(dialog) {
        flash.usbDialog = dialog;
        try {
            flash.usbPortInfo = dialog.port && typeof dialog.port.getInfo === 'function'
                ? dialog.port.getInfo()
                : null;
        } catch (error) {
            flash.usbPortInfo = null;
        }
        dialog.addEventListener('closed', () => scheduleUsbConnectionRelease(dialog), { once: true });
        renderConnection();
        syncFirmwareUpdateNotice();
    }

    function flashManifestFrontends(manifest) {
        const frontends = manifest && manifest.frontends;
        if (!frontends || typeof frontends !== 'object' || Array.isArray(frontends)) {
            const error = new Error(
                'Firmware catalog format is invalid for browser flashing.'
            );
            error.name = 'FirmwareCatalogFormatError';
            throw error;
        }
        return frontends;
    }

    function byPreferredOrder(order, a, b) {
        const ia = order.indexOf(a);
        const ib = order.indexOf(b);
        if (ia === -1 && ib === -1) return 0;
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
    }

    function flashResolveUrl(url) {
        return url;
    }

    async function flashLoadManifest(channel) {
        if (flash.manifests[channel]) return flash.manifests[channel];
        const response = await fetch(
            '/artifacts/firmware/' + channel + '/firmware-manifest-' + channel + '.json',
            { cache: 'no-store' }
        );
        if (!response.ok) {
            const error = new Error('Unable to load the ' + channel + ' firmware manifest.');
            error.status = response.status;
            throw error;
        }
        const manifest = await response.json();
        flash.manifests[channel] = manifest;
        return manifest;
    }

    function flashReleaseChannelLabel(manifest) {
        const version = String(manifest?.release_tag || manifest?.version || '').replace(/^v/, '');
        return /(?:^|[-.])rc(?:[.-]?\d+)(?:$|[-.])/i.test(version)
            ? 'Release candidate'
            : 'Stable — recommended';
    }

    function flashUpdateReleaseChannelLabel(manifest) {
        const select = document.getElementById('flash-channel');
        const option = select?.querySelector('option[value="release"]');
        if (option) option.textContent = flashReleaseChannelLabel(manifest);
    }

    function flashStatus(message, kind) {
        const el = $('.js-flash-status');
        el.textContent = message;
        el.className = 'flash-status js-flash-status' + (kind ? ' ' + kind : '');
    }

    function formatEnglishList(items) {
        if (items.length === 0) return '';
        if (items.length === 1) return items[0];
        if (items.length === 2) return items[0] + ' and ' + items[1];
        return items.slice(0, -1).join(', ') + ', and ' + items[items.length - 1];
    }

    function flashUnsupportedBoardMessage(chipFamily) {
        const detected = chipFamily
            ? ('This ' + chipFamily + ' board is not supported.')
            : 'This board is not supported.';
        const list = formatEnglishList(flash.supportedChipLabels);
        if (!list) return detected;
        return detected + ' Published firmware is available for ' + list + '.';
    }

    function flashHideMatterQr() {
        const status = $('.js-matter-status');
        const result = $('.js-matter-result');
        matterClose(false);
        if (status) {
            status.hidden = true;
            status.textContent = '';
        }
        if (result) result.hidden = true;
    }

    function flashSetFrontendActions(frontendKey) {
        const matterReadButton = $('.js-matter-read');
        matterReadButton.hidden = frontendKey !== 'matter';
        if (frontendKey !== 'matter') flashHideMatterQr();
    }

    async function flashRefresh() {
        const frontendSel = document.getElementById('flash-frontend');
        const channelSel = document.getElementById('flash-channel');
        const summary = $('.js-flash-summary');
        const installButton = $('.js-flash-install');
        const requestId = ++flash.refreshRequest;
        const selectedChannel = channelSel.value;
        flash.targetVersion = '';

        try {
            const manifest = await flashLoadManifest(selectedChannel);
            if (requestId !== flash.refreshRequest) return;
            if (selectedChannel === 'release') flashUpdateReleaseChannelLabel(manifest);
            const frontendsMap = flashManifestFrontends(manifest);

            const frontends = Object.entries(frontendsMap)
                .sort(([a], [b]) => byPreferredOrder(FRONTEND_ORDER, a, b));
            const successKey = selectedChannel + ':success';
            if (!flash.catalogReports.has(successKey)) {
                const reported = track('firmware_catalog', {
                    channel: selectedChannel,
                    result: 'success',
                    frontend_count: frontends.length,
                    artifact_count: frontends.reduce(
                        (total, [, frontend]) => total + (frontend.artifacts || []).length, 0
                    )
                });
                if (reported) flash.catalogReports.add(successKey);
            }
            const previousFrontend = frontendSel.value;
            frontendSel.innerHTML = '';
            for (const [key, value] of frontends) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = value.label || key;
                frontendSel.appendChild(option);
            }
            if (frontends.some(([key]) => key === previousFrontend)) frontendSel.value = previousFrontend;

            const selectedFrontend = frontendsMap[frontendSel.value];
            const artifacts = ((selectedFrontend || {}).artifacts || [])
                .filter((a) => a.build_type === 'factory' && a.chip_family && a.url)
                .sort((a, b) => byPreferredOrder(CHIP_ORDER, a.chip, b.chip));
            flash.supportedChipLabels = artifacts.map((artifact) => artifact.chip_label);
            if (flash.installUrl) {
                URL.revokeObjectURL(flash.installUrl);
                flash.installUrl = null;
            }
            installButton.removeAttribute('manifest');
            if (!artifacts.length) {
                flashSetFrontendActions(frontendSel.value);
                summary.textContent = 'No firmware matches these options.';
                flashStatus('Choose another firmware type or version.', 'is-error');
                return;
            }

            const frontendLabel = (selectedFrontend || {}).label || frontendSel.value;
            flash.targetVersion = manifest.version || '';
            const installManifest = {
                name: 'ESPectre ' + frontendLabel,
                version: manifest.version,
                builds: artifacts.map((artifact) => ({
                    chipFamily: artifact.chip_family,
                    parts: [{ path: flashResolveUrl(artifact.url), offset: 0 }]
                }))
            };
            flash.installUrl = URL.createObjectURL(
                new Blob([JSON.stringify(installManifest)], { type: 'application/json' })
            );
            installButton.setAttribute('manifest', flash.installUrl);

            summary.replaceChildren();
            const title = document.createElement('strong');
            title.textContent = frontendLabel;
            const detail = document.createTextNode(manifest.release_tag + ' ');
            const channel = document.createElement('span');
            channel.className = 'mono-sub';
            channel.textContent = channelSel.options[channelSel.selectedIndex]?.textContent
                || manifest.channel;
            summary.append(title, document.createElement('br'), detail, channel);
            flashSetFrontendActions(frontendSel.value);

            if (!browserSupport.flash) {
                flashStatus(flashUnsupportedMessage(), 'is-error');
            } else {
                flashStatus('Ready. Connect the board over USB, then install.', 'is-ready');
            }
        } catch (error) {
            if (requestId !== flash.refreshRequest) return;
            flash.supportedChipLabels = [];
            if (flash.installUrl) {
                URL.revokeObjectURL(flash.installUrl);
                flash.installUrl = null;
            }
            installButton.removeAttribute('manifest');
            flashSetFrontendActions('');
            summary.textContent = 'Firmware information is unavailable right now.';
            console.warn('Firmware catalog could not be loaded:', error);
            flashStatus('Check your connection, then try again.', 'is-error');
            const failureKey = selectedChannel + ':failure';
            if (!flash.catalogReports.has(failureKey)) {
                const reported = track('firmware_catalog', {
                    channel: selectedChannel,
                    result: 'failure',
                    error_type: errorType(error)
                });
                if (reported) flash.catalogReports.add(failureKey);
            }
        }
    }

    function matterDelay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function matterResetDevice(port) {
        await port.setSignals({ dataTerminalReady: false, requestToSend: true });
        await matterDelay(100);
        await port.setSignals({ dataTerminalReady: false, requestToSend: false });
    }

    async function matterReadCodes(port, timeoutMs = 20000) {
        const reader = port.readable.getReader();
        const decoder = new TextDecoder();
        const deadline = Date.now() + timeoutMs;
        let input = '';
        try {
            while (Date.now() < deadline) {
                const result = await Promise.race([
                    reader.read(),
                    matterDelay(deadline - Date.now()).then(() => ({ timedOut: true }))
                ]);
                if (result.timedOut || result.done) break;
                input += decoder.decode(result.value, { stream: true });
                if (input.length > 16384) input = input.slice(-8192);
                const qr = input.match(/MATTER_QR=(MT:[A-Z0-9.\-]+)/);
                const manual = input.match(/MATTER_MANUAL_CODE=([0-9]+)/);
                if (qr && manual) return { qr: qr[1], manual: manual[1] };
            }
        } finally {
            await reader.cancel().catch(() => {});
            reader.releaseLock();
        }
        throw new Error('Matter codes were not received. Press reset on the board, then try again.');
    }

    async function matterReadQr() {
        const status = $('.js-matter-status');
        const result = $('.js-matter-result');
        const trigger = $('.js-matter-read');
        if (!browserSupport.flash) {
            status.hidden = false;
            status.textContent = flashUnsupportedMessage();
            track('matter_qr_read', { result: 'unsupported' });
            return;
        }
        try {
            await loadBrowserDependency(
                '/vendor/qrcodejs-1.0.0/qrcode.min.js',
                'https://unpkg.com/qrcodejs@1.0.0/qrcode.min.js'
            );
        } catch (error) {
            status.hidden = false;
            status.textContent = 'The Matter setup code could not be displayed. Refresh the page and try again.';
            track('matter_qr_read', { result: 'failure', error_type: 'QrRendererMissing' });
            return;
        }
        let port;
        if (trigger) {
            trigger.disabled = true;
            trigger.setAttribute('aria-disabled', 'true');
        }
        result.hidden = true;
        status.hidden = false;
        status.textContent = 'Choose the ESPectre USB device, then wait for it to restart.';
        try {
            port = await navigator.serial.requestPort();
            await port.open({ baudRate: 115200 });
            await matterResetDevice(port);
            const codes = await matterReadCodes(port);
            const canvas = $('.js-matter-canvas');
            canvas.innerHTML = '';
            new window.QRCode(canvas, {
                text: codes.qr,
                width: 220,
                height: 220,
                colorDark: '#000000',
                colorLight: '#ffffff',
                correctLevel: window.QRCode.CorrectLevel.M
            });
            $('.js-matter-payload').textContent = codes.qr;
            $('.js-matter-manual').textContent = codes.manual;
            result.hidden = false;
            status.textContent = 'Matter setup codes read from the device.';
            matterOpen(trigger);
            track('matter_qr_read', { result: 'success' });
        } catch (error) {
            console.warn('Matter setup code could not be read:', error);
            status.textContent = error?.name === 'NotFoundError'
                ? 'No USB device was selected.'
                : 'The Matter setup code could not be read. Reset the board, then try again.';
            track('matter_qr_read', { result: 'failure', error_type: errorType(error) });
        } finally {
            if (port && (port.readable || port.writable)) {
                await port.close().catch(() => {});
            }
            if (trigger) {
                trigger.disabled = !browserSupport.flash;
                trigger.setAttribute('aria-disabled', String(!browserSupport.flash));
            }
        }
    }

    function syncModalOpenState() {
        const openModal = $$('.modal-backdrop').find((modal) => !modal.hidden);
        document.body.classList.toggle('modal-open', Boolean(openModal));
        Array.from(document.body.children).forEach((child) => {
            if (!(child instanceof HTMLElement)) return;
            const shouldBeInert = Boolean(openModal) && child !== openModal;
            if (shouldBeInert && !child.inert) {
                child.inert = true;
                child.dataset.modalInert = 'true';
            } else if (!shouldBeInert && child.dataset.modalInert === 'true') {
                child.inert = false;
                delete child.dataset.modalInert;
            }
        });
    }

    function matterOpen(returnFocus) {
        const modal = $('.js-matter-modal');
        flash.modalReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        modal.querySelector('.modal-card').focus();
    }

    function matterClose(restoreFocus = true) {
        const modal = $('.js-matter-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (restoreFocus && flash.modalReturnFocus && flash.modalReturnFocus.isConnected) {
            flash.modalReturnFocus.focus();
        }
        flash.modalReturnFocus = null;
    }

    function flashParams() {
        return {
            frontend: document.getElementById('flash-frontend').value,
            channel: document.getElementById('flash-channel').value,
            chip: flash.detectedChip
        };
    }

    function flashDialogText(root) {
        if (!root) return '';
        let text = root.textContent || '';
        root.querySelectorAll('*').forEach((element) => {
            if (element.shadowRoot) text += ' ' + flashDialogText(element.shadowRoot);
        });
        return text;
    }

    function watchFirmwareInstallDialog(dialog) {
        if (flash.watchedDialogs.has(dialog)) return;
        flash.watchedDialogs.add(dialog);
        activateUsbConnection(dialog);
        let started = false;
        let reported = false;
        let shadowObserver = null;
        let inspectTimer = null;
        const observedRoots = new WeakSet();

        const markStarted = () => {
            if (started) return;
            started = true;
            track('firmware_install_start', flashParams());
        };
        const report = (result) => {
            if (reported) return;
            reported = true;
            track('firmware_install_result', { ...flashParams(), result });
        };
        const setDetectedChip = (chip) => {
            const normalized = String(chip || '').toUpperCase();
            if (!normalized || flash.detectedChip === normalized) return;
            flash.detectedChip = normalized;
            renderConnection();
        };
        const customizeDeviceSettingsLink = () => {
            if (!dialog.shadowRoot) return;
            dialog.shadowRoot.querySelectorAll('[slot="headline"]').forEach((headline) => {
                const label = headline.textContent.trim();
                if (!['Visit Device', 'Configure Device', 'Open device settings'].includes(label)) return;
                if (label !== 'Open device settings') headline.textContent = 'Open device settings';
                const item = headline.closest('ew-list-item');
                if (!item || !item.href) return;
                try {
                    const returnedUrl = new URL(item.href, location.href);
                    const configureUrl = new URL('/', location.origin);
                    configureUrl.search = returnedUrl.search;
                    configureUrl.hash = 'tool-configure';
                    const destination = configureUrl.toString();
                    if (item.href !== destination) item.href = destination;
                    if (item.target !== '_self') item.target = '_self';
                } catch (_error) {
                    // Leave malformed device URLs untouched so ESP Web Tools can report them.
                }
            });
        };
        const inspect = () => {
            observeRoot(dialog.shadowRoot);
            customizeDeviceSettingsLink();
            const text = flashDialogText(dialog.shadowRoot);
            // The vendored ESP Web Tools version exposes completion before its
            // final Next screen. Text checks keep the listener resilient if
            // those internal state properties change in a future upgrade.
            const installState = dialog._installState && dialog._installState.state;
            if (dialog._installState && dialog._installState.chipFamily) {
                setDetectedChip(dialog._installState.chipFamily);
            }
            const found = text.match(FLASH_CHIP_FOUND_RE);
            if (found) setDetectedChip(found[1]);
            if (dialog._installConfirmed === true
                    || /Preparing installation|Erasing device|Writing progress:/i.test(text)) {
                markStarted();
            }
            const unsupported = text.match(FLASH_CHIP_UNSUPPORTED_RE);
            if (unsupported) {
                setDetectedChip(unsupported[1]);
                flashStatus(flashUnsupportedBoardMessage(unsupported[1]), 'is-error');
                report('unsupported');
                return;
            }
            if (installState === 'finished' || /Installation complete!/i.test(text)) report('success');
            else if (installState === 'error' || /Installation failed/i.test(text)) report('failure');
        };
        const observeRoot = (root) => {
            if (!root || observedRoots.has(root)) return;
            observedRoots.add(root);
            shadowObserver.observe(root, {
                childList: true, subtree: true, characterData: true
            });
            root.querySelectorAll('*').forEach((element) => {
                if (element.shadowRoot) observeRoot(element.shadowRoot);
            });
        };
        const attach = () => {
            if (!dialog.shadowRoot || shadowObserver) return false;
            shadowObserver = new MutationObserver(inspect);
            inspectTimer = setInterval(inspect, 250);
            inspect();
            return true;
        };
        if (!attach()) {
            void customElements.whenDefined('ewt-install-dialog').then(() => {
                if (dialog.isConnected) attach();
            });
        }

        const removalObserver = new MutationObserver(() => {
            if (dialog.isConnected) return;
            removalObserver.disconnect();
            scheduleUsbConnectionRelease(dialog);
            if (started && !reported) report('cancelled');
            if (shadowObserver) shadowObserver.disconnect();
            clearInterval(inspectTimer);
            inspectTimer = null;
        });
        removalObserver.observe(document.body, { childList: true, subtree: true });
    }

    function observeFirmwareInstaller() {
        if (flash.installerObserver) return;
        const inspect = () => {
            document.querySelectorAll('ewt-install-dialog').forEach(watchFirmwareInstallDialog);
        };
        flash.installerObserver = new MutationObserver(inspect);
        flash.installerObserver.observe(document.body, { childList: true, subtree: true });
        inspect();
    }

    function flashInit() {
        const selectionType = {
            'flash-frontend': 'frontend', 'flash-channel': 'channel'
        };
        Object.keys(selectionType).forEach((id) => {
            document.getElementById(id).addEventListener('change', () => {
                track('firmware_selection', {
                    selection_type: selectionType[id],
                    frontend: document.getElementById('flash-frontend').value,
                    channel: document.getElementById('flash-channel').value
                });
                flashRefresh();
            });
        });
        $('.js-flash-install').addEventListener('click', (event) => {
            if (!browserSupport.flash) {
                event.preventDefault();
                flashStatus(flashUnsupportedMessage(), 'is-error');
                return;
            }
            flash.detectedChip = '';
            flashStatus('Select the serial port. The installer detects the chip and chooses the matching firmware.');
            track('firmware_installer_open', flashParams());
        });
        $('.js-firmware-releases').addEventListener('click', () => {
            const params = flashParams();
            track('firmware_releases_open', {
                frontend: params.frontend,
                channel: params.channel,
                entry_point: 'flash'
            });
        });
        $('.js-matter-read').addEventListener('click', matterReadQr);
        if (browserSupport.flash) observeFirmwareInstaller();
    }

    /* =========================================================== configure */

    const WIFI_SCAN_POLL_INTERVAL_MS = 350;
    const WIFI_SCAN_TIMEOUT_MS = 30 * 1000;

    function cfgValue(id) {
        return document.getElementById(id).value;
    }

    const utf8Length = (value) => new TextEncoder().encode(String(value)).byteLength;

    function directConfigSnapshot(config) {
        const device = config.device || {};
        const wifi = config.wifi || {};
        const mqtt = config.mqtt || {};
        return {
            device_label: device.label ?? '',
            wifi_ssid: wifi.ssid || '',
            wifi_band: wifi.band || '',
            wifi_bssid: wifi.bssid || '',
            wifi_connected: wifi.connected === true,
            wifi_apply_state: wifi.apply_state || '',
            mqtt_host: mqtt.host || '',
            mqtt_port: mqtt.port || 0,
            topic_prefix: mqtt.topic_prefix || ''
        };
    }

    function finishConfigVerification(result, errorType, message = '') {
        if (!pendingConfigVerification) return;
        clearTimeout(pendingConfigVerification.timer);
        const action = pendingConfigVerification.action;
        pendingConfigVerification = null;
        if (message) toast(message);
        track('configure_change', {
            action,
            result,
            ...(errorType ? { error_type: errorType } : {})
        });
    }

    function configVerificationCanWait(pending) {
        return pending.waitForReconnect && Date.now() < pending.deadlineAt;
    }

    function scheduleConfigVerification(pending) {
        if (pendingConfigVerification !== pending) return;
        clearTimeout(pending.timer);
        pending.timer = setTimeout(requestConfigVerification, CONFIG_VERIFICATION_RETRY_MS);
    }

    async function requestConfigVerification() {
        const pending = pendingConfigVerification;
        if (!pending) return;
        if (directClient && conn.mode === 'direct' && conn.status === 'connecting') {
            if (configVerificationCanWait(pending)) {
                pending.observedDisconnect = true;
                scheduleConfigVerification(pending);
            } else {
                finishConfigVerification(
                    'unconfirmed', 'VerificationTimeout', pending.timeoutMessage);
            }
            return;
        }
        if (!directClient?.connected) {
            if (configVerificationCanWait(pending)) {
                pending.observedDisconnect = true;
                scheduleConfigVerification(pending);
            } else {
                finishConfigVerification(
                    'unconfirmed', 'VerificationUnavailable', pending.timeoutMessage);
            }
            return;
        }
        pending.attempts += 1;
        try {
            const [device, runtime, wifi, mqtt] = await Promise.all([
                directClient.request('get', 'device'),
                directClient.request('get', 'sensing'),
                directClient.request('get', 'wifi'),
                directClient.capabilities?.resources?.includes('mqtt')
                    ? directClient.request('get', 'mqtt') : Promise.resolve({})
            ]);
            const config = { device, runtime, wifi, mqtt };
            if (pendingConfigVerification !== pending) return;
            applyDirectConfig(config);
            evaluateConfigVerification(directConfigSnapshot(config));
        } catch (_error) {
            if (pendingConfigVerification !== pending) return;
            if (configVerificationCanWait(pending)) {
                pending.observedDisconnect = true;
                scheduleConfigVerification(pending);
            } else {
                finishConfigVerification(
                    'unconfirmed', 'VerificationRequestFailed', pending.timeoutMessage);
            }
        }
    }

    function beginConfigVerification(action, verification, acknowledgement = {}) {
        if (pendingConfigVerification) finishConfigVerification('unconfirmed', 'Superseded');
        const normalized = typeof verification === 'function'
            ? { verify: verification }
            : verification;
        const timeoutMs = Number(normalized.timeoutMs) || 0;
        const waitForReconnect = normalized.waitForReconnect === true;
        pendingConfigVerification = {
            action,
            verify: normalized.verify,
            evaluate: normalized.evaluate,
            attempts: 0,
            timer: null,
            waitForReconnect,
            observedDisconnect: waitForReconnect
                && (conn.status === 'connecting' || !directClient?.connected || directReconnectAttempt > 0),
            requiresReconnect: typeof normalized.requiresReconnect === 'function'
                ? normalized.requiresReconnect(acknowledgement?.data || acknowledgement)
                : false,
            deadlineAt: timeoutMs > 0 ? Date.now() + timeoutMs : 0,
            timeoutMessage: normalized.timeoutMessage || ''
        };
        pendingConfigVerification.timer = setTimeout(
            requestConfigVerification,
            Number(normalized.initialDelayMs) || CONFIG_VERIFICATION_INITIAL_DELAY_MS
        );
    }

    function evaluateConfigVerification(snapshot) {
        const pending = pendingConfigVerification;
        if (!pending) return;
        clearTimeout(pending.timer);
        if (pending.evaluate) {
            const outcome = pending.evaluate(snapshot, pending);
            if (outcome) {
                finishConfigVerification(outcome.result, outcome.errorType, outcome.message);
            } else if (pending.deadlineAt > 0 && Date.now() >= pending.deadlineAt) {
                finishConfigVerification(
                    'unconfirmed', 'VerificationTimeout', pending.timeoutMessage);
            } else {
                scheduleConfigVerification(pending);
            }
            return;
        }
        if (pending.verify(snapshot)) {
            finishConfigVerification('success');
        } else if (pending.attempts >= CONFIG_VERIFICATION_MAX_ATTEMPTS) {
            finishConfigVerification('unconfirmed', 'VerificationMismatch');
        } else {
            pending.timer = setTimeout(requestConfigVerification, CONFIG_VERIFICATION_RETRY_MS);
        }
    }

    async function cfgApply(action, successMessage, requestMethod, resource, params = {}, verification) {
        const resolvedSuccessMessage = (result = {}) => (
            typeof successMessage === 'function' ? successMessage(result) : successMessage
        );
        if (conn.mode === 'demo') {
            toast(resolvedSuccessMessage() + ' (demo — nothing written)');
            return true;
        }
        if (!directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', { action, result: 'failure', error_type: 'NotConnected' });
            return false;
        }
        try {
            const result = await directClient.request(requestMethod, resource, params);
            toast(resolvedSuccessMessage(result?.data || result));
            track('configure_change', { action, result: 'accepted' });
            if (verification) beginConfigVerification(action, verification, result?.data || result);
            return true;
        } catch (error) {
            console.warn('Device setting update failed:', error);
            toast('The setting could not be saved. Check the values and try again.');
            track('configure_change', { action, result: 'failure', error_type: errorType(error) });
            return false;
        }
    }

    function cfgValidationFailed(action, message) {
        toast(message);
        track('configure_change', { action, result: 'validation_failure' });
    }

    async function cfgRefreshDevice() {
        if (conn.mode === 'direct' && directClient?.connected) {
            try {
                await refreshDirectDevice();
            } catch (error) {
                console.warn('Direct device refresh failed:', error);
            }
        }
    }

    function renderWifiAccessPoints(snapshot = {}) {
        const select = document.getElementById('cfg-bssid');
        const scanButton = $('.js-wifi-scan');
        if (!select || !scanButton) return;
        const scanning = snapshot.scanning === true;
        if (!scanning && Array.isArray(snapshot.access_points)) {
            const options = [new Option('Automatic (strongest available)', '')];
            snapshot.access_points.forEach((accessPoint) => {
                const bssid = String(accessPoint?.bssid || '').toUpperCase();
                const rssi = Number(accessPoint?.rssi_dbm);
                if (!/^[0-9A-F]{2}(?::[0-9A-F]{2}){5}$/.test(bssid)
                        || !Number.isInteger(rssi)) return;
                options.push(new Option(`${bssid} · ${rssi} dBm`, bssid));
            });
            if (currentWifiBssid && !options.some((option) => option.value === currentWifiBssid)) {
                options.push(new Option(`${currentWifiBssid} · preferred`, currentWifiBssid));
            }
            select.replaceChildren(...options);
        }
        select.value = currentWifiBssid;
        select.disabled = scanning;
        scanButton.disabled = scanning;
        scanButton.classList.toggle('is-scanning', scanning);
        scanButton.setAttribute('aria-busy', String(scanning));
    }

    async function cfgRefreshWifiAccessPoints() {
        if (conn.mode === 'demo') {
            renderWifiAccessPoints({ scanning: true });
            await new Promise((resolve) => setTimeout(resolve, 700));
            if (conn.mode !== 'demo') return;
            renderWifiAccessPoints({
                scanning: false,
                message: '2 access points found. (demo)',
                access_points: [
                    { bssid: 'E6:FA:C4:20:19:DE', channel: 6, rssi_dbm: -43 },
                    { bssid: 'A2:11:7C:09:88:31', channel: 11, rssi_dbm: -67 }
                ]
            });
            return;
        }
        const client = directClient;
        if (!client?.connected || !monitor.commands.has('scan_wifi')) return;
        renderWifiAccessPoints({ scanning: true, message: 'Scanning for nearby access points…' });
        try {
            await client.request('post', 'wifi/scans');
            const deadline = Date.now() + WIFI_SCAN_TIMEOUT_MS;
            while (Date.now() < deadline) {
                await new Promise((resolve) => setTimeout(resolve, WIFI_SCAN_POLL_INTERVAL_MS));
                if (directClient !== client || !client.connected) return;
                const snapshot = await client.request('get', 'wifi/access-points');
                if (directClient !== client || !client.connected) return;
                renderWifiAccessPoints(snapshot);
                if (!snapshot.scanning) return;
            }
            if (directClient !== client || !client.connected) return;
            renderWifiAccessPoints({ scanning: false, message: 'The scan took too long. Try again.' });
        } catch (error) {
            if (directClient !== client || !client.connected) return;
            console.warn('Access point scan failed:', error);
            renderWifiAccessPoints({
                scanning: false,
                message: 'The scan failed. Check the connection and try again.'
            });
        }
    }

    async function cfgSaveWifi() {
        const bssid = cfgValue('cfg-bssid').trim().toUpperCase();
        const action = bssid ? 'set_wifi_bssid' : 'clear_wifi_bssid';
        const requestMethod = bssid ? 'put' : 'delete';
        const params = bssid ? { bssid, force: false } : {};
        const verification = {
            waitForReconnect: true,
            timeoutMs: WIFI_BSSID_VERIFICATION_TIMEOUT_MS,
            initialDelayMs: CONFIG_VERIFICATION_RETRY_MS,
            timeoutMessage: 'The Wi-Fi update could not be confirmed. Reconnect to the device and check its current access point.',
            requiresReconnect: (acknowledgement) => {
                const current = String(acknowledgement?.current_bssid || '').toUpperCase();
                return !bssid || current !== bssid;
            },
            evaluate: (snapshot, pending) => {
                const state = String(snapshot.wifi_apply_state || '').toLowerCase();
                const configuredBssid = String(snapshot.wifi_bssid || '').toUpperCase();
                const matches = configuredBssid === bssid;
                if (state === 'recovery_required') {
                    return {
                        result: 'failure', errorType: 'WifiRecoveryRequired',
                        message: 'The previous Wi-Fi selection could not be restored. Device recovery is required.'
                    };
                }
                if (state === 'rolled_back') {
                    return {
                        result: 'failure', errorType: 'WifiRolledBack',
                        message: 'The requested access point was unavailable. The previous Wi-Fi selection was restored.'
                    };
                }
                if (state === 'applied') {
                    return matches
                        ? { result: 'success', message: 'Wi-Fi access-point selection verified.' }
                        : {
                            result: 'unconfirmed', errorType: 'VerificationMismatch',
                            message: 'The device reconnected with a different Wi-Fi selection.'
                        };
                }
                if (state) return null;
                if (pending.requiresReconnect && !pending.observedDisconnect) return null;
                if (snapshot.wifi_connected && matches) {
                    return { result: 'success', message: 'Wi-Fi access-point selection verified.' };
                }
                if (pending.observedDisconnect && snapshot.wifi_connected && !matches) {
                    return {
                        result: 'failure', errorType: 'WifiRolledBack',
                        message: 'The requested access point was unavailable. The previous Wi-Fi selection was restored.'
                    };
                }
                return null;
            }
        };
        await cfgApply(
            action,
            (acknowledgement) => {
                if (!bssid) return 'Automatic Wi-Fi selection saved. The device is reconnecting.';
                const currentBssid = String(acknowledgement?.current_bssid || '').toUpperCase();
                return currentBssid === bssid
                    ? 'Wi-Fi preference saved. This access point is already active.'
                    : 'Wi-Fi preference saved. The device is reconnecting.';
            },
            requestMethod, 'wifi/bssid', params, verification);
    }

    const CONFIG_CLEAR_DIALOGS = Object.freeze({
        wifi: Object.freeze({
            kicker: 'Wi-Fi connection',
            title: 'Reset Wi-Fi configuration?',
            description: 'This removes the saved Wi-Fi network and password from the device.',
            warning: 'The device will disconnect. Connect it to Wi-Fi again over USB to restore network access.',
            confirm: 'Reset Wi-Fi'
        }),
        mqtt: Object.freeze({
            kicker: 'MQTT integration',
            title: 'Remove MQTT configuration?',
            description: 'This removes the broker host, port, username, password, and topic prefix.',
            warning: 'Wi-Fi stays connected, but MQTT automations and Home Assistant discovery will stop.',
            confirm: 'Remove MQTT'
        })
    });

    function closeConfigClearDialog(confirmed = false) {
        const modal = $('.js-config-clear-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (configClearReturnFocus && configClearReturnFocus.isConnected) configClearReturnFocus.focus();
        configClearReturnFocus = null;
        const resolve = configClearResolve;
        configClearResolve = null;
        if (resolve) resolve(confirmed);
    }

    function openConfigClearDialog(kind, returnFocus) {
        const copy = CONFIG_CLEAR_DIALOGS[kind];
        if (!copy) return Promise.resolve(false);
        if (configClearResolve) closeConfigClearDialog(false);
        const modal = $('.js-config-clear-modal');
        $('.js-config-clear-kicker').textContent = copy.kicker;
        $('.js-config-clear-title').textContent = copy.title;
        $('.js-config-clear-description').textContent = copy.description;
        $('.js-config-clear-warning').textContent = copy.warning;
        $('.js-config-clear-confirm').textContent = copy.confirm;
        configClearReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        $('.js-config-clear-confirm').focus();
        return new Promise((resolve) => { configClearResolve = resolve; });
    }

    async function cfgClearWifi() {
        if (conn.mode !== 'demo' && !directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', {
                action: 'clear_wifi', result: 'failure', error_type: 'NotConnected'
            });
            return;
        }
        if (!await openConfigClearDialog('wifi', document.activeElement)) return;
        if (conn.mode === 'demo') {
            toast('Wi-Fi configuration removed. (demo — nothing written)');
            return;
        }
        let result = 'accepted';
        let message = 'Wi-Fi settings removed. Connect the device to Wi-Fi again over USB.';
        try {
            await directClient.request('delete', 'wifi/credentials', {}, { timeoutMs: 3000 });
        } catch (error) {
            if (error?.code !== 'timeout' && error?.code !== 'closed') {
                console.warn('Wi-Fi reset failed:', error);
                toast('Wi-Fi settings could not be removed. Check the connection and try again.');
                track('configure_change', {
                    action: 'clear_wifi', result: 'failure', error_type: errorType(error)
                });
                return;
            }
            result = 'unconfirmed';
            message = 'Wi-Fi reset sent. The device disconnected as expected; connect it to Wi-Fi again over USB.';
        }
        ['cfg-ssid', 'cfg-channel', 'cfg-wifi-band', 'cfg-wifi-phy', 'cfg-bssid'].forEach((id) => {
            document.getElementById(id).value = '';
        });
        track('configure_change', { action: 'clear_wifi', result });
        toast(message);
        teardownConnection('wifi_cleared');
    }

    async function cfgSaveMqtt() {
        const scheme = cfgValue('cfg-mqtt-scheme');
        const host = cfgValue('cfg-mqtt-host').trim();
        const username = cfgValue('cfg-mqtt-user').trim();
        const password = cfgValue('cfg-mqtt-pass');
        const clearCredentials = document.getElementById('cfg-mqtt-credentials-clear').checked;
        if (!scheme || !host || !cfgValue('cfg-mqtt-port')) {
            cfgValidationFailed('update_mqtt', 'MQTT needs a scheme, host, and port.');
            return;
        }
        if (!['mqtt', 'mqtts'].includes(scheme) || /[\s/?#@\[\]]/.test(host)
                || (host.includes(':') && !/^[0-9a-f:.]+$/i.test(host))) {
            cfgValidationFailed('update_mqtt', 'Enter a host or IP address without a scheme, port, path, or credentials.');
            return;
        }
        const port = Number(cfgValue('cfg-mqtt-port'));
        const topicPrefix = cfgValue('cfg-topic-prefix').trim().replace(/\/+$/, '');
        if (!Number.isInteger(port) || port < 1 || port > 65535 || !topicPrefix || /[+#\0]/.test(topicPrefix)) {
            cfgValidationFailed('update_mqtt', 'Use a port from 1 to 65535 and a topic prefix without MQTT wildcards.');
            return;
        }
        const mqttParams = { scheme, host, port, topic_prefix: topicPrefix };
        if (username || clearCredentials) mqttParams.username = clearCredentials ? '' : username;
        if (password || clearCredentials) mqttParams.password = clearCredentials ? '' : password;
        const ok = await cfgApply('update_mqtt', 'MQTT settings saved.',
            'patch', 'mqtt', mqttParams,
            (snapshot) => snapshot.mqtt_scheme === scheme
                && snapshot.mqtt_host === host && Number(snapshot.mqtt_port) === port);
        if (ok) {
            document.getElementById('cfg-mqtt-user').value = '';
            document.getElementById('cfg-mqtt-pass').value = '';
            document.getElementById('cfg-mqtt-credentials-clear').checked = false;
            syncConfigureMqttCredentialMode();
        }
    }

    function applyConfigureMqttDefaults() {
        applyConfigureMqttPreset('home_assistant');
    }

    async function cfgClearMqtt() {
        if (conn.mode !== 'demo' && !directClient?.connected) {
            toast('ESPectre is not connected.');
            track('configure_change', {
                action: 'clear_mqtt', result: 'failure', error_type: 'NotConnected'
            });
            return;
        }
        if (!await openConfigClearDialog('mqtt', document.activeElement)) return;
        const ok = await cfgApply(
            'clear_mqtt', 'MQTT settings cleared.', 'delete', 'mqtt', {},
            (snapshot) => !snapshot.mqtt_host);
        if (ok) applyConfigureMqttDefaults();
    }

    async function cfgSaveDeviceLabel(label) {
        if (typeof label !== 'string' || /[\r\n\0]/.test(label) || utf8Length(label) > 32) {
            cfgValidationFailed('update_device', 'Use a single-line device name that fits within the 32-byte limit. Shorten names with accented or non-Latin characters if needed.');
            return false;
        }
        return cfgApply('update_device', 'Device name saved.', 'patch', 'device',
            { label },
            (snapshot) => (snapshot.device_label || '') === label);
    }

    function finishOtaTracking(result, errorType, state) {
        if (!otaTracking) return;
        clearTimeout(otaPollTimer);
        const startedAt = otaTracking.startedAt;
        otaTracking = null;
        otaPollTimer = null;
        track('ota_update_result', {
            result,
            ota_state: state || 'unknown',
            duration_ms: Math.max(0, Date.now() - startedAt),
            channel: selectedOtaChannel(),
            ...(errorType ? { error_type: errorType } : {})
        });
    }

    function beginOtaTracking() {
        if (otaTracking) finishOtaTracking('unconfirmed', 'Superseded', otaTracking.lastState);
        otaTracking = { startedAt: Date.now(), attempts: 0, lastState: 'starting' };
        clearTimeout(otaPollTimer);
        otaPollTimer = setTimeout(() => {
            finishOtaTracking('unconfirmed', 'StatusTimeout', otaTracking?.lastState);
        }, OTA_TRACKING_TIMEOUT_MS);
    }

    function evaluateOtaTracking(snapshot) {
        if (!otaTracking || !snapshot.ota_state) return;
        const state = String(snapshot.ota_state).toLowerCase();
        otaTracking.lastState = state;
        if (state === 'reboot_scheduled') {
            finishOtaTracking('success', null, state);
        } else if (state === 'error') {
            finishOtaTracking('failure', 'DeviceReportedError', state);
        }
    }

    function otaModalDescriptionElement() {
        const modal = $('.js-ota-modal');
        return modal ? modal.querySelector('.modal-description') : null;
    }

    function setOtaModalDescription(text) {
        const description = otaModalDescriptionElement();
        if (description && text) description.textContent = text;
    }

    function syncOtaModalDescription() {
        const modal = $('.js-ota-modal');
        if (!modal || modal.hidden) return;
        const state = String(otaState || '').toLowerCase();
        if (state === 'downloading') {
            setOtaModalDescription('Downloading firmware…');
        } else if (state === 'applying') {
            setOtaModalDescription('Applying firmware…');
        } else if (state === 'reboot_scheduled') {
            setOtaModalDescription('Update applied. Waiting for the device to come back online…');
        } else if (state === 'error') {
            setOtaModalDescription(otaMessage && otaMessage !== '—' ? otaMessage : 'Update failed.');
        }
    }

    function completeOtaReconnect() {
        if (!otaAwaitingReconnect) return;
        const version = conn.firmwareVersion || '';
        if (otaTargetVersion && version !== otaTargetVersion) {
            setOtaModalDescription('Device is back online. Verifying the updated firmware version…');
            return;
        }
        otaAwaitingReconnect = false;
        otaBusy = false;
        finishOtaTracking('success', null, 'reconnected');
        applyOtaStatus({
            state: 'idle',
            busy: false,
            current_version: version || undefined,
            target_version: '',
            update_available: false,
            message: version ? ('Back online · firmware ' + version) : 'Back online after update'
        });
        setOtaModalDescription(version
            ? 'Update applied. Device is back online on ' + version + '.'
            : 'Update applied. Device is back online.');
        otaClose();
        toast(version
            ? 'Device is back online on ' + version + '.'
            : 'Device is back online after the update.');
        otaTargetVersion = '';
        otaCheckTransport = '';
        maybeStartSilentOtaCheck();
    }

    function selectedOtaChannel() {
        const el = document.getElementById('ota-channel');
        const value = (el && el.value ? String(el.value) : '').trim();
        return value;
    }

    function resetOtaChannelSelection() {
        otaDefaultChannel = '';
        otaChannelChanged = false;
        const el = document.getElementById('ota-channel');
        if (el) el.value = 'release';
    }

    function applyOtaDefaultChannel(channel) {
        const normalized = String(channel || '').trim().toLowerCase();
        if (!['release', 'preview', 'develop'].includes(normalized)) return;
        if (!otaDefaultChannel) otaDefaultChannel = normalized;
        if (otaChannelChanged) return;
        const el = document.getElementById('ota-channel');
        if (el) el.value = otaDefaultChannel;
    }

    function otaStateLabel(state) {
        const value = String(state || '').toLowerCase();
        const labels = {
            idle: 'Ready',
            checking: 'Checking',
            update_available: 'Update available',
            up_to_date: 'Up to date',
            downloading: 'Downloading',
            applying: 'Installing',
            reboot_scheduled: 'Restarting',
            error: 'Error'
        };
        return labels[value] || String(state || '—').replaceAll('_', ' ');
    }

    function otaCommandFields(command) {
        const channel = selectedOtaChannel();
        return channel ? { command, channel } : { command };
    }

    function syncOtaUpdateButton() {
        const button = $('.js-ota-start');
        if (!button) return;
        const otaTransportReady = conn.mode === 'direct' && directClient?.connected;
        button.disabled = !otaTransportReady
            || otaActionPending || otaBusy || !otaUpdateAvailable;
        button.textContent = otaBusy ? 'Update in progress…' : 'Update device';
    }

    function syncFirmwareUpdateNotice() {
        const target = (document.getElementById('cfg-ota-target')?.textContent || '').trim();
        const state = String(otaState || '').toLowerCase();
        let status = 'idle';
        let copy = 'Checking for updates…';
        if (state === 'checking' || state === 'idle' || state === '') {
            status = 'idle';
            copy = 'Checking for updates…';
        } else if (otaBusy || state === 'downloading' || state === 'applying') {
            status = 'busy';
            copy = 'Updating…';
        } else if (state === 'reboot_scheduled') {
            status = 'busy';
            copy = 'Reboot scheduled';
        } else if (state === 'error') {
            status = 'error';
            copy = otaMessage && otaMessage !== '—' ? otaMessage : 'Unable to check for updates';
        } else if (otaUpdateAvailable || state === 'update_available') {
            status = 'update';
            copy = target && target !== '—'
                ? 'Update available · ' + target
                : 'Update available';
        } else if (state === 'up_to_date') {
            status = 'latest';
            copy = 'Latest';
        }
        $$('.js-firmware-update-copy').forEach((el) => { el.textContent = copy; });
        $$('.js-firmware-update-notice').forEach((el) => {
            el.dataset.otaStatus = status;
            el.hidden = conn.mode === 'demo' || Boolean(flash.usbDialog) || otaSupported === false;
        });
    }

    function applyOtaStatus(status) {
        if (!status || typeof status !== 'object') return;
        applyOtaDefaultChannel(status.default_channel || (!otaDefaultChannel ? status.channel : ''));
        const write = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== '') el.textContent = value;
        };
        const state = status.state;
        if (state) {
            write('cfg-ota-state', otaStateLabel(state));
            otaState = String(state).toLowerCase();
        }
        if (status.current_version) write('cfg-ota-current', status.current_version);
        if (status.target_version !== undefined) {
            const targetVersion = String(status.target_version || '');
            if (targetVersion || !otaBusy) otaTargetVersion = targetVersion;
            write('cfg-ota-target', targetVersion || '—');
        }
        if (status.message !== undefined) {
            otaMessage = status.message || '';
            write('cfg-ota-message', otaMessage || '—');
        }
        const normalizedState = String(state || '').toLowerCase();
        if (status.update_available !== undefined) {
            otaUpdateAvailable = sysinfoBoolean(status.update_available) || normalizedState === 'update_available';
            write('cfg-ota-available', otaUpdateAvailable ? 'Yes' : 'No');
        } else if (normalizedState === 'update_available') {
            otaUpdateAvailable = true;
            write('cfg-ota-available', 'Yes');
        } else if (normalizedState === 'up_to_date') {
            otaUpdateAvailable = false;
            write('cfg-ota-available', 'No');
        }
        if (status.busy !== undefined) otaBusy = sysinfoBoolean(status.busy);
        if (normalizedState === 'reboot_scheduled') {
            otaAwaitingReconnect = true;
        } else if (otaAwaitingReconnect &&
                (normalizedState === 'idle' || normalizedState === 'up_to_date' ||
                    normalizedState === 'update_available' || normalizedState === 'checking')) {
            completeOtaReconnect();
            return;
        }
        evaluateOtaTracking({ ota_state: state });
        syncOtaUpdateButton();
        syncFirmwareUpdateNotice();
        syncOtaModalDescription();
        maybeStartSilentOtaCheck();
    }

    function reportOtaCheckFailure() {
        applyOtaStatus({
            state: 'error',
            update_available: false,
            busy: false,
            message: 'Unable to check for updates'
        });
    }

    function currentOtaCheckTransport() {
        if (conn.mode === 'direct' && directClient?.connected) return 'direct';
        return '';
    }

    function runOtaCheck({ manual = false } = {}) {
        if (conn.mode === 'demo') return;
        const transport = currentOtaCheckTransport();
        if (!transport) return;
        if (!manual && transport && otaCheckTransport === transport) return;
        if (otaState === 'checking' && manual) return;
        otaState = 'checking';
        otaMessage = '';
        syncFirmwareUpdateNotice();
        if (!manual) otaCheckTransport = transport;
        monitorPublishCommand(otaCommandFields('check_ota'), {
            pendingMessage: '',
            statusFn: () => {}
        }).catch((error) => {
            console.warn('Silent OTA check failed:', error);
            reportOtaCheckFailure();
        });
    }

    function startSilentOtaCheck() {
        runOtaCheck();
    }

    function maybeStartSilentOtaCheck() {
        if (!otaDefaultChannel || otaBusy) return;
        if (['checking', 'downloading', 'applying', 'reboot_scheduled'].includes(otaState)) return;
        startSilentOtaCheck();
    }

    function startManualOtaCheck() {
        runOtaCheck({ manual: true });
    }

    function otaOpen(returnFocus) {
        if (conn.mode === 'demo') return;
        const modal = $('.js-ota-modal');
        otaModalReturnFocus = returnFocus || document.activeElement;
        modal.hidden = false;
        syncModalOpenState();
        modal.querySelector('.modal-card').focus();
    }

    function otaClose(restoreFocus = true) {
        const modal = $('.js-ota-modal');
        if (!modal || modal.hidden) return;
        modal.hidden = true;
        syncModalOpenState();
        if (restoreFocus && otaModalReturnFocus && otaModalReturnFocus.isConnected) {
            otaModalReturnFocus.focus();
        }
        otaModalReturnFocus = null;
    }

    async function cfgOtaStart() {
        if (!currentOtaCheckTransport()) return;
        otaActionPending = true;
        syncOtaUpdateButton();
        const description = $('.js-ota-modal') && $('.js-ota-modal').querySelector('.modal-description');
        const analyticsParams = {
            channel: selectedOtaChannel(), transport: 'direct_http', input_mode: 'direct'
        };
        track('ota_update_attempt', { ...analyticsParams, result: 'attempt' });
        try {
            await monitorPublishCommand(otaCommandFields('start_ota'), {
                pendingMessage: 'Starting firmware update…',
                statusFn: (message) => { if (description) description.textContent = message; }
            });
            track('ota_update_attempt', { ...analyticsParams, result: 'accepted' });
            otaBusy = true;
            beginOtaTracking();
            toast('Firmware update started.');
        } catch (error) {
            console.warn('Firmware update failed to start:', error);
            track('ota_update_attempt', {
                ...analyticsParams, result: 'failure', error_type: errorType(error)
            });
            toast('The firmware update could not start. Check the connection and try again.');
        }
        otaActionPending = false;
        syncOtaUpdateButton();
        syncFirmwareUpdateNotice();
    }

    function configureInit() {
        const presetName = document.getElementById('cfg-mqtt-preset').value;
        applyMqttPresetFieldLocks('configure', MQTT_PRESETS[presetName].configure);
        applyConfigureMqttCredentialPolicy(presetName);
        $('.js-wifi-save').addEventListener('click', cfgSaveWifi);
        $('.js-wifi-scan').addEventListener('click', cfgRefreshWifiAccessPoints);
        $('.js-wifi-clear').addEventListener('click', cfgClearWifi);
        $('.js-mqtt-save').addEventListener('click', cfgSaveMqtt);
        $('.js-mqtt-clear').addEventListener('click', cfgClearMqtt);
        document.getElementById('cfg-mqtt-credentials-clear').addEventListener('change', syncConfigureMqttCredentialMode);
        document.getElementById('cfg-mqtt-preset').addEventListener('change', (event) => {
            applyConfigureMqttPreset(event.currentTarget.value);
        });
        $('.js-configure-name-trigger').addEventListener('click', startConfigureDeviceNameEdit);
        const nameInput = $('.js-configure-name-input');
        nameInput.addEventListener('blur', () => { saveConfigureDeviceNameOnBlur(); });
        nameInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                nameInput.blur();
            } else if (event.key === 'Escape') {
                event.preventDefault();
                cancelConfigureDeviceNameEdit();
            }
        });
    }
