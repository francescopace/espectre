/*
 * ESPectre - Direct discovery
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    function directEndpointInput() {
        return document.querySelector(`[data-page="${route}"] .js-direct-endpoint`)
            || document.querySelector('.js-direct-endpoint');
    }

    function syncDirectEndpointInputs(target) {
        $$('.js-direct-endpoint').forEach((input) => {
            if (input && target) input.value = target;
        });
    }

    function privateIpv4Address(value) {
        const octets = value.split('.');
        if (octets.length !== 4 || octets.some((octet) => !/^(0|[1-9][0-9]{0,2})$/.test(octet))) {
            return false;
        }
        const values = octets.map(Number);
        if (values.some((octet) => octet > 255)) return false;
        return values[0] === 10
            || (values[0] === 172 && values[1] >= 16 && values[1] <= 31)
            || (values[0] === 192 && values[1] === 168);
    }

    function directTargetForEndpoint(endpoint) {
        const url = new URL(DirectProtocolClient.normalizeEndpoint(endpoint));
        const canonical = url.hostname.toLowerCase().match(DIRECT_CANONICAL_HOSTNAME);
        if (canonical) return canonical[1];
        if (privateIpv4Address(url.hostname)) return url.hostname;
        if (url.hostname.includes(':')) return url.hostname;
        return url.hostname;
    }

    function parseDirectTarget(value) {
        const input = String(value || '').trim().toLowerCase();
        if (!input) throw new Error('Enter a device IP address, ID, or name.');
        if (DIRECT_FULL_DEVICE_ID.test(input)) {
            return {
                display: input,
                endpoint: DirectProtocolClient.normalizeEndpoint(`espectre-${input}.local`),
                deviceId: input,
                shortId: '',
                discoveryFallback: true
            };
        }
        if (DIRECT_SHORT_DEVICE_ID.test(input)) {
            return { display: input, endpoint: '', deviceId: '', search: input, shortId: input };
        }
        if (privateIpv4Address(input)) {
            return {
                display: input,
                endpoint: DirectProtocolClient.normalizeEndpoint(input),
                deviceId: '',
                shortId: ''
            };
        }
        try {
            const endpoint = DirectProtocolClient.normalizeEndpoint(input);
            return {
                display: directTargetForEndpoint(endpoint), endpoint, deviceId: '', shortId: ''
            };
        } catch (_error) {
            if (/^[a-z][a-z0-9+.-]*:\/\//i.test(input)
                || /^\d{1,3}(?:\.\d{1,3}){3}$/.test(input)) {
                throw new Error('Enter a private device IP address from the same Wi-Fi network.');
            }
            if (input.length <= 63 && [...input].every((character) => character >= ' ')) {
                return { display: input, endpoint: '', deviceId: '', search: input, shortId: '' };
            }
            throw new Error('Enter a device IP address, ID, or name.');
        }
    }

    function storedDirectEndpoints() {
        try {
            const parsed = JSON.parse(localStorage.getItem(DIRECT_ENDPOINT_STORAGE_KEY) || '[]');
            if (!Array.isArray(parsed)) return [];
            return parsed.filter((value) => typeof value === 'string').flatMap((value) => {
                try { return [DirectProtocolClient.normalizeEndpoint(value)]; } catch (_error) { return []; }
            }).slice(0, DIRECT_ENDPOINT_LIMIT);
        } catch (_error) {
            return [];
        }
    }

    function writeStoredDirectEndpoints(endpoints) {
        try {
            localStorage.setItem(DIRECT_ENDPOINT_STORAGE_KEY, JSON.stringify(endpoints.slice(0, DIRECT_ENDPOINT_LIMIT)));
        } catch (_error) {
            // Private browsing and locked-down storage must not block Direct mode.
        }
        renderStoredDirectEndpoints();
    }

    function renderStoredDirectEndpoints() {
        const list = document.getElementById('direct-remembered-endpoints');
        if (!list) return;
        list.replaceChildren(...storedDirectEndpoints().map((endpoint) => {
            const option = document.createElement('option');
            option.value = directTargetForEndpoint(endpoint);
            return option;
        }));
    }

    function rememberDirectEndpoint(endpoint) {
        const endpoints = storedDirectEndpoints().filter((value) => value !== endpoint);
        writeStoredDirectEndpoints([endpoint, ...endpoints]);
    }

    function consumeDirectHandoff() {
        const params = new URLSearchParams(location.search);
        const directTarget = params.get('target') || '';
        if (!directTarget) return;
        try {
            const target = parseDirectTarget(directTarget);
            syncDirectEndpointInputs(target.display);
            const radio = document.getElementById('monitor-transport-direct');
            if (radio) radio.checked = true;
            history.replaceState(null, '', location.pathname + location.hash);
        } catch (_error) {
            history.replaceState(null, '', location.pathname + location.hash);
            toast('The device handoff contained an invalid local device.');
        }
    }

    function directCapabilitiesSnapshot(capabilities) {
        const methods = new Set((capabilities.commands || []).map((item) => item?.name).filter(Boolean));
        const sections = new Set(capabilities.config_sections || []);
        monitor.commands = methods;
        monitor.commandCatalogReady = true;
        return {
            supports_wifi_status: sections.has('wifi'),
            supports_wifi_bssid: methods.has('set_wifi_bssid')
                && methods.has('clear_wifi_bssid')
                && methods.has('scan_wifi_access_points') && methods.has('wifi_access_points'),
            supports_wifi_clear: methods.has('clear_wifi_config'),
            supports_mqtt_config: methods.has('set_mqtt_config'),
            supports_device_config: methods.has('set_device_label'),
            supports_runtime_threshold: methods.has('set_threshold'),
            supports_runtime_motion_hits: methods.has('set_motion_hits'),
            supports_runtime_detector: methods.has('set_detector'),
            supports_manual_recalibration: methods.has('recalibrate'),
            supports_traffic_control: methods.has('set_csi_traffic_mode')
                && methods.has('set_traffic_generator_mode'),
            supports_ota: methods.has('ota_status')
        };
    }

    function directSupportsCommand(name) {
        return Boolean(directClient?.capabilities?.commands?.some((item) => item?.name === name));
    }

    function applyDirectConfig(config) {
        const device = config.device || {};
        const runtime = config.runtime || {};
        const wifi = config.wifi || {};
        const mqtt = config.mqtt || {};
        applySysinfo({
            device_label: config.device_label ?? device.device_label,
            wifi_configured: wifi.configured,
            wifi_connected: wifi.connected,
            wifi_ssid: wifi.ssid,
            wifi_band: wifi.band,
            wifi_channel: wifi.channel,
            wifi_rssi_dbm: wifi.rssi_dbm,
            wifi_bssid: wifi.bssid,
            wifi_apply_state: wifi.apply_state,
            wifi_apply_message: wifi.apply_message,
            mqtt_configured: mqtt.configured,
            mqtt_host: mqtt.host,
            mqtt_port: mqtt.port,
            mqtt_topic_prefix: mqtt.topic_prefix,
            mqtt_username_configured: mqtt.username_configured
        });
        applySensingSnapshot(runtime);
    }

    function applyRuntimeStatus(status) {
        if (!status || status.calibrating === undefined) return;
        const calibrating = sysinfoBoolean(status.calibrating);
        setCalibrationBusy(calibrating);
        if (calibrating) scheduleCalibrationIdle(MONITOR_CALIBRATION_SAFETY_MS);
    }

    function ingestDirectEvent(name, data) {
        if (name === 'telemetry') {
            applySensingCadence(data);
            applyLiveTelemetry(
                Number(data.movement_score ?? data.movement ?? 0),
                Number(data.threshold ?? conn.threshold),
                data.motion_state ?? data.motion
            );
            monitorFeed(conn.movement, conn.threshold, data.motion_state ?? data.motion);
            monitorResizeChart();
            return;
        }
        if (name === 'info' || name === 'status' || name === 'capabilities') {
            applySysinfo(data);
            if (name === 'status') applyRuntimeStatus(data);
            return;
        }
        if (name === 'config') {
            applyDirectConfig(data);
            return;
        }
        if (name === 'diagnostics') {
            markMonitorReady('diagnostics');
            monitorStats(data);
            return;
        }
        if (name === 'ota_status') applyOtaStatus(data);
        if (name === 'fault') toast(data.message || 'The device reported a runtime fault.');
    }

    function makeDirectClient(endpoint) {
        const client = new DirectProtocolClient(endpoint);
        client.on('event', ingestDirectEvent);
        client.on('protocol-error', (error) => console.warn('Ignored invalid Direct frame:', error.message));
        client.on('close', ({ expected }) => {
            if (expected || conn.mode !== 'direct') return;
            scheduleDirectReconnect(client);
        });
        return client;
    }

    function directPageOriginKind() {
        return sitePolicy.directOriginKind(location);
    }

    function directBrowserGuidance() {
        if (location.protocol !== 'https:') {
            return 'Local development mode: the firmware must explicitly allow HTTP loopback Origins. Development builds accept localhost on any port.';
        }
        if (browserSupport.hostedDirect === 'targeted'
            && ['windows', 'linux'].includes(browserSupport.operatingSystem)) {
            const platform = browserSupport.operatingSystem === 'windows' ? 'Windows' : 'Linux';
            return `Local connections are supported in desktop Chrome on ${platform}. If search does not find the device, enter its current IP address.`;
        }
        if (browserSupport.hostedDirect === 'targeted') return '';
        return 'This browser may not connect to local devices. Use Chrome 151 or later on macOS, Windows, or native Linux.';
    }

    function renderDirectBrowserGuidance() {
        $$('.js-direct-browser-note').forEach((note) => {
            const message = directBrowserGuidance();
            note.textContent = message;
            note.hidden = !message;
        });
    }

    function directConnectionErrorMessage(error, endpoint, permissionState = 'unavailable') {
        const code = error?.code || 'connection_failed';
        let url;
        try { url = new URL(endpoint); } catch (_error) { url = null; }
        const localName = Boolean(url?.hostname.endsWith('.local'));
        const hostedCleartext = location.protocol === 'https:' && url?.protocol === 'http:';
        if (code === 'local_network_denied' || permissionState === 'denied') {
            return 'Local network access is blocked. Allow it for this site in Chrome settings, then try again. On macOS, you may also need to allow Chrome in System Settings > Privacy & Security > Local Network.';
        }
        if (code === 'timeout') {
            return localName
                ? 'The device did not respond. Make sure it is powered on and connected to the same Wi-Fi network, then search again or enter its IP address.'
                : 'The device did not respond. Make sure it is powered on, connected to the same Wi-Fi network, and still uses this IP address.';
        }
        if (code === 'subprotocol_mismatch' || code === 'unsupported_version'
            || code === 'invalid_capabilities' || code === 'invalid_envelope') {
            return 'This device is not compatible with the current browser tools. Update its firmware, then try again.';
        }
        if (code === 'connection_failed' || code === 'closed') {
            if (directPageOriginKind() === 'other') {
                return 'The device may have rejected this page Origin. Use https://espectre.dev, https://test.espectre.dev, or a loopback development portal explicitly enabled in the firmware.';
            }
            if (directPageOriginKind() === 'loopback') {
                return 'A local HTTP portal does not require a Local network access prompt. Confirm that this is a development firmware with loopback Origins enabled, reflash if it predates any-port localhost support, close other ESPectre tabs, and retry.';
            }
            if (hostedCleartext && browserSupport.hostedDirect === 'unsupported') {
                return 'This browser cannot connect to ESPectre on your local network. Open this page in supported desktop Chrome.';
            }
            if (hostedCleartext && permissionState === 'prompt') {
                return 'Chrome is waiting for local network permission. Try again, allow access for this site, and keep the device on the same Wi-Fi network.';
            }
            const addressHelp = localName
                ? 'Search again or enter the device IP address. '
                : 'Check the device IP address. ';
            return `The browser could not connect to ESPectre. ${addressHelp}Make sure the device is powered on and connected to the same Wi-Fi network, close other ESPectre tabs, and try again.`;
        }
        console.warn('Local device connection failed:', error);
        return 'The browser could not connect to ESPectre. Check the device and try again.';
    }

    function setDirectConnectionHelp(message = '') {
        const help = directEndpointInput()?.closest('.device-connect-card')?.querySelector('.js-direct-help');
        if (!help) return;
        const copy = help.querySelector('.js-direct-help-copy');
        if (copy) copy.textContent = message;
        help.hidden = !message;
    }

    function setDirectConnectionStatus(message = '') {
        const status = directEndpointInput()?.closest('.device-connect-card')?.querySelector('.js-direct-status');
        if (!status) return;
        status.textContent = message;
        status.hidden = !message;
    }

    function directDiscoveryPanel(button) {
        return button?.closest('.device-connect-card')?.querySelector('.js-direct-discovery') || null;
    }

    function cancelDirectDiscovery({ clear = false } = {}) {
        directDiscoveryGeneration += 1;
        const client = directDiscoveryClient;
        directDiscoveryClient = null;
        client?.close();
        $$('.js-direct-discover').forEach((button) => {
            button.disabled = false;
            button.setAttribute('aria-disabled', 'false');
        });
        if (clear) {
            $$('.js-direct-discovery').forEach((panel) => {
                panel.hidden = true;
                panel.replaceChildren();
            });
        }
    }

    function discoveredPeerChipLabel(chip) {
        return String(chip || '').toUpperCase().replace(/^ESP32([A-Z]\d)$/, 'ESP32-$1');
    }

    function createDiscoveryDeviceButton({
        deviceId,
        displayDeviceId = deviceId,
        displayName,
        frontend,
        chip,
        endpoint = '',
        className = '',
        actionText = 'Connect →',
        ariaLabel
    }) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `btn-ghost direct-discovery-device ${className}`.trim();
        button.dataset.deviceId = deviceId;
        if (endpoint) button.dataset.endpoint = endpoint;
        button.setAttribute('aria-label', ariaLabel);
        const heading = document.createElement('span');
        heading.className = 'direct-discovery-device-heading';
        const name = document.createElement('strong');
        name.className = 'direct-discovery-device-name';
        name.textContent = displayName;
        const action = document.createElement('span');
        action.className = 'direct-discovery-device-action';
        action.setAttribute('aria-hidden', 'true');
        action.textContent = actionText;
        heading.append(name, action);
        const metadata = document.createElement('span');
        metadata.className = 'direct-discovery-device-meta';
        for (const [label, value, valueClass = ''] of [
            ['Frontend', frontend || 'Unknown'],
            ['Hardware', chip || 'Unknown'],
            ['Device ID', displayDeviceId, 'mono']
        ]) {
            const field = document.createElement('span');
            field.className = 'direct-discovery-device-field';
            const fieldLabel = document.createElement('span');
            fieldLabel.className = 'direct-discovery-device-label';
            fieldLabel.textContent = label;
            const fieldValue = document.createElement('span');
            fieldValue.className = `direct-discovery-device-value ${valueClass}`.trim();
            fieldValue.textContent = value;
            field.append(fieldLabel, fieldValue);
            metadata.appendChild(field);
        }
        button.append(heading, metadata);
        return button;
    }

    function renderDiscoveredPeers(panel, result, { selectionRequired = false } = {}) {
        panel.replaceChildren();
        const summary = document.createElement('p');
        summary.className = 'direct-discovery-summary';
        summary.textContent = selectionRequired
            ? `${result.devices.length} matching devices found. Select one to connect.`
            : result.devices.length
            ? `${result.devices.length} local device${result.devices.length === 1 ? '' : 's'} found${result.truncated ? ' (partial result)' : ''}.`
            : 'No compatible devices answered. You can still enter a private IP address or device ID.';
        panel.appendChild(summary);
        if (!result.devices.length) return;
        const list = document.createElement('ul');
        list.className = 'direct-discovery-list';
        for (const peer of result.devices) {
            const item = document.createElement('li');
            const displayName = peer.name || `ESPectre ${peer.device_id.slice(-6)}`;
            const frontend = formatFrontendLabel(peer.frontend);
            const chip = discoveredPeerChipLabel(peer.chip);
            const shortId = peer.device_id.slice(-6);
            const button = createDiscoveryDeviceButton({
                deviceId: peer.device_id,
                displayDeviceId: shortId,
                displayName,
                frontend,
                chip,
                endpoint: peer.endpoints[0],
                ariaLabel: `Connect to ${displayName}, ${frontend}, ${chip}, device ID ${shortId}`
            });
            item.appendChild(button);
            list.appendChild(item);
        }
        panel.appendChild(list);
    }

    function directDiscoveryFailureMessage(error, permissionState) {
        if (error?.code === 'local_network_denied' || permissionState === 'denied') {
            return 'Local network access is blocked. Allow it for this site in Chrome settings, then search again.';
        }
        if (error?.code === 'unsupported_crypto') {
            return 'This browser cannot search for local devices. Enter the device IP address instead.';
        }
        if (error?.code === 'unsupported_capability') {
            return 'This device does not support browser search. Enter its IP address instead.';
        }
        if (error?.code === 'timeout') {
            return 'No device responded in time. Make sure ESPectre is powered on and connected to the same Wi-Fi network, then search again or enter its IP address.';
        }
        if (error?.code === 'invalid_envelope' || error?.code === 'unsupported_version') {
            return 'Search found an incompatible device. Update its firmware, or enter the IP address of another ESPectre device.';
        }
        if (error?.code === 'invalid_peer_result' || error?.code === 'frame_too_large') {
            return 'Search found a device it could not recognize. Enter the IP address of the ESPectre device you trust.';
        }
        if (error?.code === 'connection_failed') {
            return 'Search could not reach any devices. Make sure ESPectre is on the same Wi-Fi network, or enter its IP address.';
        }
        return 'Device search is unavailable on this network. Enter the ESPectre IP address instead.';
    }

    async function queryLocalPeers(onProgress = () => {}) {
        const client = makeDirectClient(DirectProtocolClient.createDiscoveryEndpoint());
        directDiscoveryClient = client;
        try {
            onProgress('Looking for compatible ESPectre devices…');
            try {
                return await client.discoverPeersBootstrap();
            } catch (error) {
                error.discoveryStage = 'query';
                throw error;
            }
        } finally {
            if (directDiscoveryClient === client) directDiscoveryClient = null;
            client.close();
        }
    }

    async function discoverLocalPeers(button) {
        cancelDirectDiscovery({ clear: true });
        const panel = directDiscoveryPanel(button);
        if (!panel) return;
        const generation = directDiscoveryGeneration;
        panel.hidden = false;
        panel.textContent = 'Starting device search…';
        button.disabled = true;
        button.setAttribute('aria-disabled', 'true');
        track('local_discovery', { tool_name: activeToolName(), result: 'attempt' });
        try {
            const result = await queryLocalPeers((message) => {
                if (generation === directDiscoveryGeneration) panel.textContent = message;
            });
            if (generation !== directDiscoveryGeneration) return;
            setDirectConnectionHelp();
            renderDiscoveredPeers(panel, result);
            track('local_discovery', {
                tool_name: activeToolName(), result: result.devices.length ? 'success' : 'empty',
                device_count: result.devices.length, truncated: result.truncated
            });
        } catch (error) {
            if (generation !== directDiscoveryGeneration) return;
            const permissionState = await localNetworkAccessState();
            if (generation !== directDiscoveryGeneration) return;
            panel.textContent = directDiscoveryFailureMessage(error, permissionState);
            track('local_discovery', {
                tool_name: activeToolName(), result: 'failure', error_type: errorType(error)
            });
        } finally {
            if (generation === directDiscoveryGeneration) {
                button.disabled = false;
                button.setAttribute('aria-disabled', 'false');
            }
        }
    }

    async function localNetworkAccessState() {
        const detectState = window.ESPectreBrowserSupport.localNetworkAccessState;
        return typeof detectState === 'function'
            ? detectState(navigator) : 'unavailable';
    }

    function cancelDirectReconnect() {
        clearTimeout(directReconnectTimer);
        directReconnectTimer = 0;
        directReconnectAttempt = 0;
    }

    function scheduleDirectReconnect(client) {
        if (directClient !== client || conn.mode !== 'direct' || directReconnectTimer) return;
        if (directReconnectAttempt >= DIRECT_RECONNECT_DELAYS_MS.length) {
            directClient = null;
            teardownConnection('reconnect_failed');
            toast('The device disconnected. Enter its address to reconnect.');
            return;
        }
        const delay = DIRECT_RECONNECT_DELAYS_MS[directReconnectAttempt++];
        setStatus('connecting');
        directReconnectTimer = setTimeout(async () => {
            directReconnectTimer = 0;
            if (directClient !== client || conn.mode !== 'direct') return;
            try {
                await client.connect({ timeoutMs: 5000 });
                await client.handshake({ timeoutMs: 5000 });
                if (directClient !== client || conn.mode !== 'direct') {
                    client.close();
                    return;
                }
                await refreshDirectDevice();
                directReconnectAttempt = 0;
                setStatus('connected');
                toast('Device reconnected.');
                if (pendingConfigVerification) requestConfigVerification();
            } catch (error) {
                client.close();
                const permissionState = await localNetworkAccessState();
                if (error?.code === 'local_network_denied' || permissionState === 'denied') {
                    directClient = null;
                    teardownConnection('local_network_denied');
                    const message = directConnectionErrorMessage(error, client.endpoint, permissionState);
                    setDirectConnectionHelp(message);
                    toast(message);
                    return;
                }
                scheduleDirectReconnect(client);
            }
        }, delay);
    }

    async function refreshDirectDevice() {
        if (!directClient?.connected) return;
        const supportsOta = directClient.capabilities?.commands?.some((item) => item.name === 'ota_status');
        // Keep local-network requests serial. Chrome may still be resolving its
        // Local Network Access grant when the Direct stream and handshake have
        // just completed, and rejects a concurrent fan-out before CORS runs.
        const info = await directClient.request('info');
        const status = await directClient.request('status');
        const config = await directClient.request('config');
        const diagnostics = activeToolName() === 'configure' && directSupportsCommand('diagnostics')
            ? await directClient.request('diagnostics')
            : null;
        const otaStatus = supportsOta ? await directClient.request('ota_status') : null;
        applySysinfo({
            ...directCapabilitiesSnapshot(directClient.capabilities),
            ...info,
            ...status,
            ...(diagnostics || {})
        });
        applyDirectConfig(config);
        if (otaStatus) applyOtaStatus(otaStatus);
    }

    async function resolveDiscoveredTarget(target, input) {
        const description = target.deviceId
            ? `device ID ${target.deviceId}`
            : target.shortId ? `device ID …${target.shortId}` : `device name “${target.search}”`;
        setDirectConnectionHelp();
        setDirectConnectionStatus(`Looking for ${description} on this Wi-Fi network.`);
        track('local_discovery', { tool_name: activeToolName(), result: 'attempt' });
        let result;
        try {
            result = await queryLocalPeers();
        } catch (error) {
            const permissionState = await localNetworkAccessState();
            throw new Error(directDiscoveryFailureMessage(error, permissionState));
        } finally {
            setDirectConnectionStatus();
        }
        const query = target.search.toLowerCase();
        const matches = result.devices.filter((peer) => target.deviceId
            ? peer.device_id === target.deviceId
            : target.shortId ? peer.device_id.endsWith(target.shortId)
                : [peer.name, peer.instance].some((value) => String(value || '').toLowerCase().includes(query)));
        track('local_discovery', {
            tool_name: activeToolName(), result: matches.length === 1 ? 'success' : (matches.length ? 'multiple' : 'empty'),
            device_count: result.devices.length, truncated: result.truncated
        });
        if (matches.length === 0) {
            throw new Error(`No matching ${description} was found. Search again, or enter the device IP address.`);
        }
        if (matches.length > 1) {
            const panel = directDiscoveryPanel(input);
            if (panel) {
                panel.hidden = false;
                renderDiscoveredPeers(panel, { ...result, devices: matches }, { selectionRequired: true });
            }
            setDirectConnectionHelp();
            return null;
        }
        const peer = matches[0];
        return { display: peer.device_id, endpoint: peer.endpoints[0], deviceId: peer.device_id };
    }

    async function connectDirect({ endpoint, deviceId, openView } = {}) {
        cancelDirectDiscovery();
        if (directClient || conn.status !== 'disconnected') return;
        const input = directEndpointInput();
        let target;
        try {
            if (endpoint) {
                const normalizedEndpoint = DirectProtocolClient.normalizeEndpoint(endpoint);
                target = {
                    display: deviceId || directTargetForEndpoint(normalizedEndpoint),
                    endpoint: normalizedEndpoint,
                    deviceId: deviceId || ''
                };
            } else {
                target = parseDirectTarget(input?.value || '');
                if (target.search) target = await resolveDiscoveredTarget(target, input);
                if (!target) return;
            }
        } catch (error) {
            setDirectConnectionHelp(error.message);
            toast(error.message);
            input?.setAttribute('aria-invalid', 'true');
            return;
        }
        let normalizedEndpoint;
        try {
            normalizedEndpoint = DirectProtocolClient.normalizeEndpoint(target.endpoint);
        } catch (error) {
            console.warn('Invalid local device endpoint:', error);
            toast('This device address is not valid. Enter a private IP address, device ID, or device name.');
            input?.setAttribute('aria-invalid', 'true');
            return;
        }
        input?.removeAttribute('aria-invalid');
        setDirectConnectionHelp();
        rememberConnectionOrigin();
        track('tool_connection', {
            ...connectionParams(), transport: 'direct_http', result: 'attempt'
        });
        setStatus('connecting');
        try {
            const client = makeDirectClient(normalizedEndpoint);
            directClient = client;
            cancelDirectReconnect();
            await client.connect();
            await client.handshake();
            if (directClient !== client) return;
            conn.mode = 'direct';
            conn.endpoint = normalizedEndpoint;
            conn.deviceBannerSub = normalizedEndpoint;
            conn.connectedAt = Date.now();
            syncDirectEndpointInputs(target.display);
            rememberDirectEndpoint(normalizedEndpoint);
            await refreshDirectDevice();
            if ((openView || (route === 'tool-monitor' ? 'live' : 'connectivity')) === 'live'
                && directSupportsCommand('set_sensing')) {
                await client.request('set_sensing', { enabled: true });
            }
            setStatus('connected');
            setDirectConnectionHelp();
            if (route === 'tool-raw-csi') {
                rawCsiUseConnection();
            } else if (!LIVE_EXPERIENCE_ROUTES.has(route)) {
                const view = openView || (route === 'tool-monitor' ? 'live' : 'connectivity');
                setDeviceView(view);
                if (view === 'connectivity' && monitor.commands.has('scan_wifi_access_points')) {
                    void cfgRefreshWifiAccessPoints();
                }
            }
            track('tool_connection', {
                ...connectionParams(), transport: 'direct_http', result: 'success'
            });
            markToolReady('info');
            if (pendingLiveDestination) completeLiveConnectionNavigation();
        } catch (error) {
            directClient?.close();
            directClient = null;
            setStatus('disconnected');
            if (target.discoveryFallback) {
                try {
                    const discoveredTarget = await resolveDiscoveredTarget({
                        deviceId: target.deviceId,
                        search: target.deviceId,
                        shortId: ''
                    }, input);
                    if (discoveredTarget) {
                        return connectDirect({
                            endpoint: discoveredTarget.endpoint,
                            deviceId: discoveredTarget.deviceId,
                            openView
                        });
                    }
                    return;
                } catch (fallbackError) {
                    error = fallbackError;
                }
            }
            track('tool_connection', {
                ...connectionParams(), transport: 'direct_http', result: 'failure',
                error_type: errorType(error)
            });
            const message = directConnectionErrorMessage(
                error, normalizedEndpoint, await localNetworkAccessState());
            setDirectConnectionHelp(message);
            toast(message);
        }
    }
