/*
 * ESPectre - Website app shell
 *
 * Hash routing and a persistent device connection shared by every page. The
 * connection is real Web Bluetooth (espectre-ble.js) when available, with a
 * simulated demo mode as fallback for unsupported browsers or when no
 * hardware is around.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

(function () {
    'use strict';

    const NAV_GROUPS = {
        tools: ['flash', 'configure', 'monitor', 'theremin', 'game'],
        guides: ['guide-hardware', 'guide-setup', 'guide-detection', 'guide-firmware'],
        docs: ['docs-api', 'docs-examples', 'docs-architecture']
    };
    const ROUTES = ['home', 'tools', 'guides', 'docs', 'media', 'roadmap']
        .concat(NAV_GROUPS.tools, NAV_GROUPS.guides, NAV_GROUPS.docs);

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    // analytics.js is optional: the app must work with it blocked or absent.
    const track = (name, params) => window.trackEvent && window.trackEvent(name, params);
    const errorType = (error) => (error && error.name) || 'Error';

    /* ==================================================== shared connection */

    const conn = {
        mode: null,             // 'ble' | 'demo'
        status: 'disconnected', // disconnected | connecting | connected
        movement: 0,
        threshold: 0.5,
        motion: false,
        deviceName: '',
        deviceSub: '—',
        detector: '—',
        connectedAt: 0
    };

    let bleClient = null;
    let demoTimer = null;
    let demoInputEnergy = 0;
    const demoPointer = { x: null, y: null, t: 0 };
    let uptimeTimer = null;
    let route = 'home';
    let lastTrackedProfile = null;

    /*
     * Classic and ML both emit a probability on an absolute 0..1 scale, so
     * the display maps the value directly. Scaling against the threshold
     * would saturate well before 1 and hide how far past it a reading is.
     */
    function energyFraction() {
        return Math.min(1, Math.max(0, conn.movement));
    }

    function setStatus(status) {
        conn.status = status;
        renderConnection();
    }

    /* ------------------------------------------------------------ BLE mode */

    function makeBleClient() {
        const client = new window.ESPectreBleClient();
        client.on('telemetry', (t) => {
            conn.movement = t.movement;
            if (t.threshold > 0) conn.threshold = t.threshold;
            conn.motion = t.motionState !== null
                ? t.motionState === 1
                : t.movement >= conn.threshold;
            renderTelemetry();
            gameOnTelemetry();
        });
        client.on('sysinfo', (snapshot) => applySysinfo(snapshot));
        client.on('disconnect', () => {
            teardownConnection();
            toast('Device disconnected.');
        });
        return client;
    }

    async function connectBle() {
        if (conn.status !== 'disconnected') return;
        if (!window.ESPectreBleClient || !window.ESPectreBleClient.supported) {
            track('tool_connection', {
                tool_name: route, transport: 'bluetooth', result: 'unsupported'
            });
            toast('Web Bluetooth is not available in this browser — starting demo mode.');
            connectDemo();
            return;
        }
        setStatus('connecting');
        track('tool_connection', { tool_name: route, transport: 'bluetooth', result: 'attempt' });
        try {
            bleClient = makeBleClient();
            await bleClient.connect();
            conn.mode = 'ble';
            conn.deviceName = bleClient.name || 'ESPectre';
            conn.deviceSub = 'reading device info…';
            conn.connectedAt = Date.now();
            setStatus('connected');
            startUptime();
            track('tool_connection', { tool_name: route, transport: 'bluetooth', result: 'success' });
            try {
                await bleClient.requestSysinfo();
            } catch (error) {
                console.warn('Sysinfo request failed:', error);
            }
        } catch (error) {
            bleClient = null;
            setStatus('disconnected');
            // The chooser being dismissed is a user choice, not a failure.
            const cancelled = error && error.name === 'NotFoundError';
            track('tool_connection', {
                tool_name: route,
                transport: 'bluetooth',
                result: cancelled ? 'cancelled' : 'failure',
                error_type: errorType(error)
            });
            if (cancelled) return;
            toast(error && error.message ? error.message : 'Bluetooth connection failed.');
        }
    }

    function applySysinfo(snapshot) {
        const chip = (snapshot.chip || '').toUpperCase();
        const frontend = snapshot.frontend || '';
        const proto = snapshot.proto_version || snapshot.espectre_protocol_version || '';
        conn.deviceSub = [chip, frontend, proto && ('proto ' + proto)]
            .filter(Boolean).join(' · ') || '—';
        conn.detector = snapshot.detector || '—';
        if (snapshot.threshold) {
            const parsed = parseFloat(snapshot.threshold);
            if (Number.isFinite(parsed) && parsed > 0) conn.threshold = parsed;
        }

        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== '') {
                if (el.tagName === 'INPUT') el.value = value;
                else el.textContent = value;
            }
        };
        if (snapshot.motion_hits) {
            const parts = String(snapshot.motion_hits).split('/');
            if (parts.length === 2) {
                set('cfg-motion-on', parts[0]);
                set('cfg-motion-off', parts[1]);
            }
        }
        set('cfg-ssid', snapshot.wifi_ssid);
        set('cfg-bssid', snapshot.wifi_bssid);
        set('cfg-channel', snapshot.wifi_channel);
        set('cfg-mqtt-host', snapshot.mqtt_host);
        set('cfg-mqtt-port', snapshot.mqtt_port);
        set('cfg-mqtt-user', snapshot.mqtt_username);
        set('cfg-topic-prefix', snapshot.topic_prefix);
        set('cfg-device-id', snapshot.device_id);
        set('cfg-device-name', snapshot.device_name);
        set('cfg-label', snapshot.device_label);
        set('cfg-ota-state', snapshot.ota_state || '—');
        set('cfg-ota-target', snapshot.ota_target_version || '—');
        set('cfg-ota-message', snapshot.ota_message || '—');
        set('diag-protocol', proto || '—');
        set('diag-chip', chip || '—');
        set('diag-detector', snapshot.detector);
        set('diag-window', snapshot.window);
        set('diag-traffic', snapshot.traffic_rate
            ? snapshot.traffic_rate + ' pkt/s'
            : snapshot.traffic_mode);
        set('diag-publish', snapshot.publish_interval && 'every ' + snapshot.publish_interval + ' pkts');
        set('diag-ota', snapshot.ota_state || '—');
        const boolLabel = (v) => (v === 'true' || v === '1' ? 'connected' : 'disconnected');
        if (snapshot.wifi_connected !== undefined) set('diag-wifi', boolLabel(snapshot.wifi_connected));
        if (snapshot.mqtt_connected !== undefined) set('diag-mqtt', boolLabel(snapshot.mqtt_connected));

        // Real hardware only: demo values would pollute the adoption report.
        if (conn.mode === 'ble' && snapshot.frontend && snapshot.chip) {
            const profile = snapshot.frontend + ':' + snapshot.chip;
            if (profile !== lastTrackedProfile) {
                lastTrackedProfile = profile;
                track('device_profile', {
                    tool_name: route,
                    frontend: snapshot.frontend.toLowerCase(),
                    chip: snapshot.chip.toLowerCase()
                });
            }
        }
        renderConnection();
    }

    /* ----------------------------------------------------------- demo mode */

    function connectDemo() {
        if (conn.status !== 'disconnected') return;
        setStatus('connecting');
        setTimeout(() => {
            conn.mode = 'demo';
            conn.deviceName = 'ESPectre-DEMO';
            conn.deviceSub = 'simulated telemetry';
            conn.detector = 'classic';
            conn.threshold = 0.5;
            conn.movement = 0.04;
            conn.connectedAt = Date.now();
            setStatus('connected');
            startUptime();
            applySysinfo({
                chip: 'esp32-s3',
                frontend: 'native',
                proto_version: '1.4',
                detector: 'classic',
                window: '64',
                traffic_rate: '98',
                publish_interval: '10',
                wifi_connected: 'true',
                mqtt_connected: 'true',
                wifi_ssid: 'HomeNet-5G',
                mqtt_host: '192.168.1.20',
                mqtt_port: '1883',
                mqtt_username: 'mqtt',
                topic_prefix: 'espectre/v1/devices',
                device_id: '0x00007c2c6742bbac',
                device_name: 'ESPectre-DEMO',
                device_label: 'Living Room',
                motion_hits: '4/3',
                ota_state: 'idle',
                ota_target_version: '',
                ota_message: ''
            });
            let t = 0;
            demoTimer = setInterval(() => {
                t += 0.16;
                const gameDemoActive = route === 'game' && game.phase !== 'idle' && game.phase !== 'done';
                const idle = 0.035 + Math.sin(t * 0.8) * 0.01 + Math.sin(t * 1.9) * 0.004;
                const target = Math.min(1, idle + demoInputEnergy * 0.95);
                const smoothing = gameDemoActive ? 0.42 : 0.28;
                conn.movement += (target - conn.movement) * smoothing;
                conn.movement = Math.max(0.01, Math.min(1, conn.movement));
                demoInputEnergy *= gameDemoActive ? 0.62 : 0.72;
                if (demoInputEnergy < 0.01) demoInputEnergy = 0;
                conn.motion = conn.movement >= conn.threshold;
                renderTelemetry();
                gameOnTelemetry();
            }, 160);
        }, 600);
    }

    function demoTrackMouse(event) {
        if (conn.mode !== 'demo') return;
        const now = performance.now();
        if (demoPointer.x !== null) {
            const dt = Math.max(16, now - demoPointer.t);
            const dx = event.clientX - demoPointer.x;
            const dy = event.clientY - demoPointer.y;
            const pxPerSecond = Math.hypot(dx, dy) * 1000 / dt;
            const normalized = Math.min(1, pxPerSecond / 1800);
            demoInputEnergy = Math.max(demoInputEnergy, normalized);
        }
        demoPointer.x = event.clientX;
        demoPointer.y = event.clientY;
        demoPointer.t = now;
    }

    function demoResetMotion() {
        if (conn.mode !== 'demo') return;
        demoInputEnergy = 0;
        conn.movement = 0.04;
        conn.motion = false;
        renderTelemetry();
    }

    /* ----------------------------------------------------- shared teardown */

    function disconnect() {
        if (bleClient) {
            const client = bleClient;
            bleClient = null;
            client.disconnect().catch((error) => console.warn(error));
        }
        teardownConnection();
    }

    function teardownConnection() {
        clearInterval(demoTimer);
        clearInterval(uptimeTimer);
        demoTimer = null;
        bleClient = null;
        demoInputEnergy = 0;
        demoPointer.x = null;
        demoPointer.y = null;
        demoPointer.t = 0;
        conn.mode = null;
        conn.movement = 0;
        conn.motion = false;
        conn.deviceSub = '—';
        conn.detector = '—';
        gameReset();
        thereminStop();
        setStatus('disconnected');
    }

    function startUptime() {
        clearInterval(uptimeTimer);
        uptimeTimer = setInterval(() => {
            const up = Math.floor((Date.now() - conn.connectedAt) / 1000);
            const label = up >= 60 ? Math.floor(up / 60) + 'm ' + (up % 60) + 's' : up + 's';
            $$('.js-uptime').forEach((el) => { el.textContent = label; });
        }, 1000);
    }

    /* ----------------------------------------------------------- rendering */

    let dropdownOpen = false;

    function renderConnection() {
        const connected = conn.status === 'connected';

        $('.js-conn-disconnected').hidden = conn.status !== 'disconnected';
        $('.js-conn-connecting').hidden = conn.status !== 'connecting';
        $('.js-conn-connected').hidden = !connected;
        $('.js-dropdown').hidden = !(connected && dropdownOpen);
        $('.js-demo-tag').hidden = conn.mode !== 'demo';

        $('.js-demo-connected').hidden = !connected;
        $('.js-demo-disconnected').hidden = connected;
        $$('.js-needs-conn').forEach((el) => { el.hidden = connected; });
        $$('.js-has-conn').forEach((el) => { el.hidden = !connected; });

        $$('.js-device-name').forEach((el) => { el.textContent = conn.deviceName || 'ESPectre'; });
        $$('.js-device-sub').forEach((el) => { el.textContent = conn.deviceSub; });
        $$('.js-detector').forEach((el) => { el.textContent = conn.detector; });

        $$('.js-ble-chip').forEach((chip) => {
            chip.classList.toggle('ready', connected);
            chip.textContent = connected ? 'BLE · READY' : 'BLE';
        });

        renderTelemetry();
    }

    function renderTelemetry() {
        const pct = Math.round(energyFraction() * 100) + '%';
        $$('.js-energy-fill').forEach((el) => { el.style.width = pct; });
        $$('.js-energy-val').forEach((el) => { el.textContent = conn.movement.toFixed(2); });
        $$('.js-motion-label').forEach((el) => {
            el.textContent = conn.motion ? 'MOTION' : 'quiet';
            el.classList.toggle('motion', conn.motion);
        });
        // Same absolute scale as the bar, so the marker sits where it belongs.
        const thresholdPct = Math.min(100, Math.max(0, conn.threshold * 100));
        $$('.threshold-mark').forEach((el) => {
            el.style.left = thresholdPct + '%';
            el.title = 'Motion threshold: ' + conn.threshold.toFixed(2);
        });
    }

    /* ============================================================= routing */

    function applyRoute() {
        $$('.js-page').forEach((page) => {
            page.hidden = page.dataset.page !== route;
        });
        $$('[data-route-link]').forEach((link) => {
            const target = link.dataset.routeLink;
            const active = target === route
                || (NAV_GROUPS[target] || []).includes(route);
            link.classList.toggle('active', active);
        });
        window.scrollTo(0, 0);
        if (route !== 'theremin') thereminStop();
        if (route === 'monitor') monitorResizeChart();
        if ($(`[data-page="${route}"] .js-static-content`)) loadStaticContent(route);
        if (route === 'home') updateReleaseBadge();
        if (route === 'flash') flashRefresh();
        // The router owns navigation, so it reports it.
        if (window.trackRouteView) window.trackRouteView(route);
    }

    /**
     * Single entry point for navigation. `force` applies the current route on
     * startup; without it a repeated route is ignored so one navigation never
     * reports two page views.
     */
    function setRoute(next, { force = false } = {}) {
        const target = ROUTES.includes(next) ? next : 'home';
        if (!force && target === route) return;
        route = target;
        dropdownOpen = false;
        applyRoute();
        renderConnection();
    }

    function onHashChange() {
        setRoute((location.hash || '#home').slice(1));
    }

    /* ======================================================= static content */

    /*
     * Guides, docs, media, and the roadmap live in shared HTML fragments,
     * which also build their canonical static pages. The SPA fetches each
     * fragment on first visit so text is not duplicated and the device
     * connection survives.
     */
    const STATIC_PAGE_ROUTES = {
        '/guides/': 'guides',
        '/guides/hardware/': 'guide-hardware',
        '/guides/setup/': 'guide-setup',
        '/guides/detection/': 'guide-detection',
        '/guides/custom-firmware/': 'guide-firmware',
        '/docs/': 'docs',
        '/docs/api/': 'docs-api',
        '/docs/examples/': 'docs-examples',
        '/docs/architecture/': 'docs-architecture',
        '/media/': 'media',
        '/roadmap/': 'roadmap'
    };
    const staticContentCache = new Map();

    async function loadStaticContent(route) {
        const container = $(`[data-page="${route}"] .js-static-content`);
        if (!container || container.dataset.loaded === 'true') return;
        const contentUrl = container.dataset.contentUrl;
        try {
            if (!staticContentCache.has(contentUrl)) {
                const response = await fetch(contentUrl);
                if (!response.ok) throw new Error('HTTP ' + response.status);
                staticContentCache.set(contentUrl, await response.text());
            }
            container.innerHTML = staticContentCache.get(contentUrl);
            container.dataset.loaded = 'true';
        } catch (error) {
            console.warn('Static content fetch failed:', error);
            container.innerHTML = '<p class="guide-loading">This page could not be loaded. '
                + '<a href="' + container.dataset.staticUrl + '">Open the standalone page</a>.</p>';
        }
    }

    /*
     * Fragments and content cards link to the canonical static URLs. Inside
     * the app those clicks become hash navigation, so the page never reloads
     * and an active connection survives; modified clicks (new tab) are left
     * to the browser. Root-anchored hash links are normalized for the same
     * reason: from /index.html they would otherwise reload onto /.
     */
    function interceptCanonicalLinks(event) {
        if (event.defaultPrevented || event.button !== 0) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        const link = event.target.closest('a[href]');
        if (!link) return;
        const href = link.getAttribute('href');
        if (STATIC_PAGE_ROUTES[href]) {
            event.preventDefault();
            location.hash = '#' + STATIC_PAGE_ROUTES[href];
        } else if (href.startsWith('/#')) {
            event.preventDefault();
            location.hash = href.slice(1);
        }
    }

    /* =============================================================== toast */

    let toastTimer = null;

    function toast(message) {
        const el = $('.js-toast');
        el.textContent = message;
        el.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
    }

    /* ====================================================== scroll narrative */

    let activeScrollyScene = -1;
    let scrollyFrame = null;
    let scrollyWheelAwaitingImpulse = false;
    let scrollyWheelDirection = 0;
    let scrollyWheelLowMagnitude = Infinity;
    let scrollyWheelLastAt = 0;
    let scrollyTouchStartY = null;
    let scrollyTouchCaptured = false;
    let scrollyTouchAdvanced = false;
    let heroFrameTimer = null;
    const SCROLLY_WHEEL_MIN_DELTA = 4;
    const SCROLLY_WHEEL_MIN_IMPULSE = 12;
    const SCROLLY_WHEEL_ACCELERATION = 3;
    const SCROLLY_WHEEL_RESTART_GAP = 180;
    const SCROLLY_WHEEL_RESTART_DELTA = 40;
    const SCROLLY_TOUCH_THRESHOLD = 28;
    const HERO_FRAME_HOLD = 2000;

    function scrollySceneFromPosition(section, sceneCount) {
        const rect = section.getBoundingClientRect();
        const travel = Math.max(1, rect.height - window.innerHeight);
        const progress = Math.min(1, Math.max(0, -rect.top / travel));
        return Math.min(sceneCount - 1, Math.floor(progress * sceneCount));
    }

    function scrollToScrollyScene(section, scene, sceneCount) {
        const rect = section.getBoundingClientRect();
        const sectionTop = window.scrollY + rect.top;
        const travel = Math.max(1, rect.height - window.innerHeight);
        const sceneStep = travel / sceneCount;
        window.scrollTo(0, sectionTop + sceneStep * scene + (scene > 0 ? 1 : 0));
    }

    function stopHeroFrameSequence() {
        clearTimeout(heroFrameTimer);
        heroFrameTimer = null;
    }

    function startHeroFrameSequence() {
        const media = $('.hero-media');
        stopHeroFrameSequence();
        media.classList.remove('is-connected');
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            media.classList.add('is-connected');
            return;
        }
        heroFrameTimer = setTimeout(() => {
            media.classList.add('is-connected');
            heroFrameTimer = null;
        }, HERO_FRAME_HOLD);
    }

    function setScrollyScene(scene) {
        if (scene === activeScrollyScene) return;
        activeScrollyScene = scene;

        $$('.js-scrolly-scene, .js-scrolly-caption, .js-scrolly-marker').forEach((el) => {
            el.classList.toggle('is-active', Number(el.dataset.scene) === scene);
        });
        $('.scrolly-stage').classList.toggle('is-intro', scene === 0);
        if (scene === 0) startHeroFrameSequence();
        else stopHeroFrameSequence();
        if (scene > 0) $('.js-scrolly-current').textContent = String(scene).padStart(2, '0');
    }

    function renderScrolly() {
        scrollyFrame = null;
        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null) return;

        const rect = section.getBoundingClientRect();
        const sceneCount = $$('.js-scrolly-scene').length;
        let scene = scrollySceneFromPosition(section, sceneCount);
        const stageEngaged = rect.top <= 1 && rect.bottom >= window.innerHeight - 1;

        if (stageEngaged && activeScrollyScene >= 0 && Math.abs(scene - activeScrollyScene) > 1) {
            scene = activeScrollyScene + Math.sign(scene - activeScrollyScene);
            scrollToScrollyScene(section, scene, sceneCount);
        }
        setScrollyScene(scene);
    }

    function queueScrollyRender() {
        if (scrollyFrame !== null) return;
        scrollyFrame = requestAnimationFrame(renderScrolly);
    }

    function scrollyStageEngaged(section) {
        const rect = section.getBoundingClientRect();
        return rect.top <= 1 && rect.bottom >= window.innerHeight - 1;
    }

    function stepScrolly(direction) {
        const section = $('.js-scrolly');
        const sceneCount = $$('.js-scrolly-scene').length;
        const current = activeScrollyScene >= 0
            ? activeScrollyScene
            : scrollySceneFromPosition(section, sceneCount);
        const next = Math.max(0, Math.min(sceneCount - 1, current + direction));
        if (next === current) return false;

        setScrollyScene(next);
        scrollToScrollyScene(section, next, sceneCount);
        return true;
    }

    function scrollyWheelImpulseReady(direction, magnitude) {
        const now = Date.now();
        const initialImpulse = !scrollyWheelAwaitingImpulse
            && magnitude >= SCROLLY_WHEEL_MIN_DELTA;
        const reversedImpulse = scrollyWheelAwaitingImpulse
            && direction !== scrollyWheelDirection
            && magnitude >= SCROLLY_WHEEL_MIN_IMPULSE;
        const acceleratedImpulse = scrollyWheelAwaitingImpulse
            && direction === scrollyWheelDirection
            && magnitude >= SCROLLY_WHEEL_MIN_IMPULSE
            && magnitude >= scrollyWheelLowMagnitude * SCROLLY_WHEEL_ACCELERATION;
        const restartedWheel = scrollyWheelAwaitingImpulse
            && now - scrollyWheelLastAt >= SCROLLY_WHEEL_RESTART_GAP
            && magnitude >= SCROLLY_WHEEL_RESTART_DELTA;
        const ready = initialImpulse || reversedImpulse || acceleratedImpulse || restartedWheel;

        scrollyWheelLastAt = now;
        if (ready) {
            scrollyWheelAwaitingImpulse = true;
            scrollyWheelDirection = direction;
            scrollyWheelLowMagnitude = magnitude;
        } else {
            scrollyWheelLowMagnitude = Math.min(scrollyWheelLowMagnitude, magnitude);
        }
        return ready;
    }

    function scrollyWheel(event) {
        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null || !scrollyStageEngaged(section)) return;

        const magnitude = Math.abs(event.deltaY);
        if (magnitude === 0) return;
        const direction = Math.sign(event.deltaY);
        if (!scrollyWheelImpulseReady(direction, magnitude)) {
            event.preventDefault();
            return;
        }

        const sceneCount = $$('.js-scrolly-scene').length;
        const atStart = activeScrollyScene === 0 && direction < 0;
        const atEnd = activeScrollyScene === sceneCount - 1 && direction > 0;
        if (atStart) return;
        if (atEnd) {
            const rect = section.getBoundingClientRect();
            const sectionTop = window.scrollY + rect.top;
            const travel = Math.max(1, rect.height - window.innerHeight);
            event.preventDefault();
            window.scrollTo(0, sectionTop + travel + Math.max(2, magnitude));
            return;
        }

        event.preventDefault();
        stepScrolly(direction);
    }

    function scrollyKeydown(event) {
        const interactive = event.target.closest
            && event.target.closest('a, button, input, select, textarea, [contenteditable]');
        if (event.repeat || interactive) return;

        let direction = 0;
        if (event.key === ' ') direction = event.shiftKey ? -1 : 1;
        else if (['ArrowDown', 'PageDown'].includes(event.key)) direction = 1;
        else if (['ArrowUp', 'PageUp'].includes(event.key)) direction = -1;
        if (!direction) return;

        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null || !scrollyStageEngaged(section)) return;

        const sceneCount = $$('.js-scrolly-scene').length;
        const atStart = activeScrollyScene === 0 && direction < 0;
        const atEnd = activeScrollyScene === sceneCount - 1 && direction > 0;
        if (atStart || atEnd) return;

        event.preventDefault();
        stepScrolly(direction);
    }

    function scrollyTouchStart(event) {
        const section = $('.js-scrolly');
        const interactive = event.target.closest
            && event.target.closest('a, button, input, select, textarea, [contenteditable]');
        if (
            interactive
            || event.touches.length !== 1
            || !section
            || section.offsetParent === null
            || !scrollyStageEngaged(section)
        ) {
            scrollyTouchCaptured = false;
            scrollyTouchAdvanced = false;
            return;
        }
        scrollyTouchStartY = event.touches[0].clientY;
        scrollyTouchCaptured = true;
        scrollyTouchAdvanced = false;
    }

    function scrollyTouchMove(event) {
        if (!scrollyTouchCaptured || scrollyTouchStartY === null || event.touches.length !== 1) return;
        const distance = scrollyTouchStartY - event.touches[0].clientY;
        if (Math.abs(distance) < SCROLLY_TOUCH_THRESHOLD) {
            event.preventDefault();
            return;
        }

        const direction = Math.sign(distance);
        const sceneCount = $$('.js-scrolly-scene').length;
        const atStart = activeScrollyScene === 0 && direction < 0;
        const atEnd = activeScrollyScene === sceneCount - 1 && direction > 0;
        if (atStart || atEnd) {
            scrollyTouchCaptured = false;
            return;
        }

        event.preventDefault();
        if (!scrollyTouchAdvanced && stepScrolly(direction)) scrollyTouchAdvanced = true;
    }

    function scrollyTouchEnd() {
        scrollyTouchCaptured = false;
        scrollyTouchAdvanced = false;
        scrollyTouchStartY = null;
    }

    function scrollyInit() {
        window.addEventListener('scroll', queueScrollyRender, { passive: true });
        window.addEventListener('resize', queueScrollyRender);
        window.addEventListener('wheel', scrollyWheel, { passive: false });
        window.addEventListener('keydown', scrollyKeydown);
        window.addEventListener('touchstart', scrollyTouchStart, { passive: true });
        window.addEventListener('touchmove', scrollyTouchMove, { passive: false });
        window.addEventListener('touchend', scrollyTouchEnd, { passive: true });
        window.addEventListener('touchcancel', scrollyTouchEnd, { passive: true });
        renderScrolly();
    }

    /* =============================================================== flash */

    const flash = { manifests: {}, installUrl: null, badgeChecked: false };

    /*
     * Presentation order for the Flash selectors. Anything not listed keeps
     * its manifest order and lands after the listed entries, so a new
     * frontend or chip still shows up without touching this code.
     */
    const FRONTEND_ORDER = ['native', 'esphome', 'matter'];
    const CHIP_ORDER = ['esp32', 'esp32s3'];

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
            '/flash/firmware/' + channel + '/firmware-manifest-' + channel + '.json',
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

    function flashStatus(message, kind) {
        const el = $('.js-flash-status');
        el.textContent = message;
        el.className = 'flash-status js-flash-status' + (kind ? ' ' + kind : '');
    }

    async function flashRefresh() {
        const frontendSel = document.getElementById('flash-frontend');
        const channelSel = document.getElementById('flash-channel');
        const chipSel = document.getElementById('flash-chip');
        const summary = $('.js-flash-summary');
        const download = $('.js-flash-download');
        const installButton = $('.js-flash-install');

        try {
            const manifest = await flashLoadManifest(channelSel.value);

            const frontends = Object.entries(manifest.frontends || {})
                .sort(([a], [b]) => byPreferredOrder(FRONTEND_ORDER, a, b));
            const previousFrontend = frontendSel.value;
            frontendSel.innerHTML = '';
            for (const [key, value] of frontends) {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = value.label || key;
                frontendSel.appendChild(option);
            }
            if (frontends.some(([key]) => key === previousFrontend)) frontendSel.value = previousFrontend;

            $('.js-matter-panel').hidden = frontendSel.value !== 'matter';

            const artifacts = ((manifest.frontends[frontendSel.value] || {}).artifacts || [])
                .filter((a) => a.build_type === 'factory')
                .sort((a, b) => byPreferredOrder(CHIP_ORDER, a.chip, b.chip));
            const previousChip = chipSel.value;
            chipSel.innerHTML = '';
            for (const artifact of artifacts) {
                const option = document.createElement('option');
                option.value = artifact.chip;
                option.textContent = artifact.chip_label;
                chipSel.appendChild(option);
            }
            if (artifacts.some((a) => a.chip === previousChip)) chipSel.value = previousChip;

            const artifact = artifacts.find((a) => a.chip === chipSel.value);
            if (flash.installUrl) {
                URL.revokeObjectURL(flash.installUrl);
                flash.installUrl = null;
            }
            if (!artifact) {
                summary.textContent = 'No matching firmware was found for the selected combination.';
                flashStatus('Change the selection or use the manual download.', 'is-error');
                download.href = 'https://github.com/francescopace/espectre/releases';
                return;
            }

            const installManifest = {
                name: 'ESPectre ' + (manifest.frontends[frontendSel.value].label || frontendSel.value) + ' ' + artifact.chip_label,
                version: manifest.version,
                builds: [{
                    chipFamily: artifact.chip_family,
                    parts: [{ path: flashResolveUrl(artifact.url), offset: 0 }]
                }]
            };
            flash.installUrl = URL.createObjectURL(
                new Blob([JSON.stringify(installManifest)], { type: 'application/json' })
            );
            installButton.setAttribute('manifest', flash.installUrl);

            summary.innerHTML =
                '<strong>' + artifact.chip_label + '</strong><br>' +
                manifest.frontends[frontendSel.value].label + ' · ' + manifest.release_tag +
                ' <span class="mono-sub">(' + manifest.channel + ')</span>';
            download.href = flashResolveUrl(artifact.url);
            download.textContent = 'Download binary';

            if (!('serial' in navigator)) {
                flashStatus('This browser does not expose Web Serial. Download the binary and flash manually.', 'is-error');
            } else {
                flashStatus('Ready. Connect the board over USB, then install.', 'is-ready');
            }
        } catch (error) {
            summary.textContent = 'Firmware metadata is currently unavailable.';
            flashStatus(error.message, 'is-error');
            track('firmware_catalog', {
                channel: channelSel.value, result: 'failure', error_type: errorType(error)
            });
        }
    }

    /* ------------------------------------------------ Matter QR over USB */

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
        const button = $('.js-matter-read');
        const result = $('.js-matter-result');
        if (!('serial' in navigator)) {
            status.textContent = 'Web Serial is not available in this browser.';
            track('matter_qr_read', { result: 'unsupported' });
            return;
        }
        if (typeof window.QRCode !== 'function') {
            status.textContent = 'The local QR renderer could not be loaded.';
            track('matter_qr_read', { result: 'failure', error_type: 'QrRendererMissing' });
            return;
        }
        let port;
        button.disabled = true;
        result.hidden = true;
        status.textContent = 'Choose the ESPectre serial port, then wait for the device to restart.';
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
            status.textContent = 'This QR is stored on the device and remains the same after normal updates.';
            track('matter_qr_read', { result: 'success' });
        } catch (error) {
            status.textContent = error.message || 'Unable to read the Matter QR code.';
            track('matter_qr_read', { result: 'failure', error_type: errorType(error) });
        } finally {
            if (port && (port.readable || port.writable)) {
                await port.close().catch(() => {});
            }
            button.disabled = false;
        }
    }

    /**
     * Shows the latest published release in the hero badge. The stable
     * manifest is staged by CI from the GitHub release tag, so it is already
     * the newest version and needs no API call. The badge is decorative:
     * it stays hidden when the manifest is unavailable.
     */
    async function updateReleaseBadge() {
        if (flash.badgeChecked) return;
        flash.badgeChecked = true;
        try {
            const manifest = await flashLoadManifest('stable');
            const version = String(manifest.release_tag || manifest.version || '').replace(/^v/, '');
            if (!version) return;
            $('.js-release-text').textContent = 'v' + version + ' available';
            $('.js-release-badge').hidden = false;
        } catch (error) {
            if (error && error.status !== 404) {
                console.warn('Release badge unavailable:', error);
            }
        }
    }

    function flashInit() {
        const selectionType = {
            'flash-frontend': 'frontend', 'flash-channel': 'channel', 'flash-chip': 'chip'
        };
        Object.keys(selectionType).forEach((id) => {
            document.getElementById(id).addEventListener('change', () => {
                track('firmware_selection', {
                    selection_type: selectionType[id],
                    frontend: document.getElementById('flash-frontend').value,
                    channel: document.getElementById('flash-channel').value,
                    chip: document.getElementById('flash-chip').value
                });
                flashRefresh();
            });
        });
        const flashParams = () => ({
            frontend: document.getElementById('flash-frontend').value,
            channel: document.getElementById('flash-channel').value,
            chip: document.getElementById('flash-chip').value
        });
        $('.js-flash-install').addEventListener('click', () => {
            track('firmware_install_start', flashParams());
        });
        $('.js-flash-download').addEventListener('click', () => {
            track('firmware_download', flashParams());
        });
        $('.js-matter-read').addEventListener('click', matterReadQr);
    }

    /* ============================================================= monitor */

    const monitor = {
        client: null,
        demoTimer: null,
        demoT: 0,
        demoMove: 0.05,
        points: [],
        maxPoints: 120
    };

    function monitorStatus(message) {
        $('.js-mon-status').textContent = message;
    }

    function monitorFeed(movement, threshold, state, deviceId) {
        monitor.points.push({ m: movement, t: threshold });
        if (monitor.points.length > monitor.maxPoints) monitor.points.shift();
        const motion = state === 'motion';
        const stateEl = $('.js-mon-state');
        stateEl.textContent = motion ? 'MOTION' : 'IDLE';
        stateEl.classList.toggle('motion', motion);
        $('.js-mon-move').textContent = movement.toFixed(3);
        $('.js-mon-thr').textContent = threshold.toFixed(3);
        if (deviceId) $('.js-mon-dev').textContent = deviceId;
        monitorDrawChart();
    }

    function monitorDrawChart() {
        const canvas = $('.js-mon-chart');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        if (monitor.points.length < 2) return;

        const styles = getComputedStyle(document.documentElement);
        const accent = styles.getPropertyValue('--accent').trim() || '#4f6bff';
        const dim = styles.getPropertyValue('--dim').trim() || '#888';
        const maxValue = Math.max(
            0.1,
            ...monitor.points.map((p) => Math.max(p.m, p.t))
        ) * 1.15;
        const stepX = width / (monitor.maxPoints - 1);
        const y = (v) => height - (v / maxValue) * (height - 8) - 4;
        const x0 = width - (monitor.points.length - 1) * stepX;

        ctx.lineWidth = 1;
        ctx.strokeStyle = dim;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        monitor.points.forEach((p, i) => {
            const px = x0 + i * stepX;
            i === 0 ? ctx.moveTo(px, y(p.t)) : ctx.lineTo(px, y(p.t));
        });
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.lineWidth = 2;
        ctx.strokeStyle = accent;
        ctx.beginPath();
        monitor.points.forEach((p, i) => {
            const px = x0 + i * stepX;
            i === 0 ? ctx.moveTo(px, y(p.m)) : ctx.lineTo(px, y(p.m));
        });
        ctx.stroke();
    }

    function monitorResizeChart() {
        const canvas = $('.js-mon-chart');
        const rect = canvas.getBoundingClientRect();
        if (rect.width > 0 && canvas.width !== Math.round(rect.width)) {
            canvas.width = Math.round(rect.width);
            monitorDrawChart();
        }
    }

    function monitorStopAll() {
        if (monitor.client) {
            monitor.client.end(true);
            monitor.client = null;
        }
        clearInterval(monitor.demoTimer);
        monitor.demoTimer = null;
    }

    function monitorConnect() {
        if (typeof window.mqtt === 'undefined') {
            monitorStatus('MQTT library not loaded yet — try again in a moment.');
            track('tool_connection', {
                tool_name: 'monitor', transport: 'mqtt_websocket', result: 'unsupported'
            });
            return;
        }
        const host = document.getElementById('mon-host').value.trim();
        const port = document.getElementById('mon-port').value.trim() || '9001';
        const path = document.getElementById('mon-path').value.trim() || '/mqtt';
        const tls = document.getElementById('mon-tls').checked;
        const base = document.getElementById('mon-topic').value.trim().replace(/\/+$/, '');
        if (!host || !base) {
            monitorStatus('Set the broker host and a device base topic first.');
            track('tool_connection', {
                tool_name: 'monitor', transport: 'mqtt_websocket', result: 'validation_failure'
            });
            return;
        }
        monitorStopAll();
        monitor.points = [];
        const url = (tls ? 'wss://' : 'ws://') + host + ':' + port + path;
        monitorStatus('Connecting to ' + url + ' …');
        // The URL is not tracked: it would carry the user's broker address.
        track('tool_connection', {
            tool_name: 'monitor', transport: 'mqtt_websocket', result: 'attempt'
        });
        const client = window.mqtt.connect(url, {
            username: document.getElementById('mon-user').value || undefined,
            password: document.getElementById('mon-pass').value || undefined,
            clientId: 'espectre-mockup-' + Math.random().toString(16).slice(2, 8),
            connectTimeout: 8000,
            reconnectPeriod: 0
        });
        monitor.client = client;
        client.on('connect', () => {
            client.subscribe(base + '/telemetry', (error) => {
                monitorStatus(error
                    ? 'Subscribe failed: ' + error.message
                    : 'Live — subscribed to ' + base + '/telemetry');
                track('tool_connection', {
                    tool_name: 'monitor',
                    transport: 'mqtt_websocket',
                    result: error ? 'subscription_failure' : 'success',
                    ...(error ? { error_type: errorType(error) } : {})
                });
            });
        });
        client.on('message', (topic, payload) => {
            try {
                const data = JSON.parse(payload.toString());
                monitorFeed(
                    Number(data.movement_score ?? data.movement) || 0,
                    Number(data.threshold) || 0,
                    data.motion_state || data.state,
                    data.device_id
                );
            } catch (error) { /* ignore malformed payloads */ }
        });
        client.on('error', (error) => {
            monitorStatus('Connection failed: ' + error.message);
            track('tool_connection', {
                tool_name: 'monitor',
                transport: 'mqtt_websocket',
                result: 'failure',
                error_type: errorType(error)
            });
            monitorStopAll();
        });
        client.on('close', () => {
            if (monitor.client === client) monitorStatus('Disconnected from broker.');
        });
    }

    function monitorDemo() {
        monitorStopAll();
        monitor.points = [];
        monitorStatus('Demo feed — simulated node, no broker involved.');
        monitor.demoTimer = setInterval(() => {
            monitor.demoT += 0.5;
            let m = monitor.demoMove + (Math.random() - 0.5) * 0.04;
            if (Math.random() < 0.05) m += 0.5 + Math.random() * 0.35;
            m += Math.sin(monitor.demoT * 0.4) * 0.01;
            monitor.demoMove = Math.max(0.01, Math.min(1, m * 0.85));
            monitorFeed(
                monitor.demoMove,
                0.5,
                monitor.demoMove >= 0.5 ? 'motion' : 'idle',
                '0x00007c2c6742bbac'
            );
        }, 500);
    }

    function monitorInit() {
        $('.js-mon-connect').addEventListener('click', monitorConnect);
        $('.js-mon-demo').addEventListener('click', monitorDemo);
        window.addEventListener('resize', monitorResizeChart);
    }

    /* ============================================================ theremin */

    const theremin = { ctx: null, osc: null, gain: null, raf: null, smoothed: 0 };

    function thereminStart() {
        if (theremin.ctx) return;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            toast('Web Audio is not available in this browser.');
            return;
        }
        theremin.ctx = new AudioCtx();
        theremin.osc = theremin.ctx.createOscillator();
        theremin.gain = theremin.ctx.createGain();
        theremin.osc.type = document.getElementById('th-wave').value;
        theremin.osc.frequency.value = 140;
        theremin.gain.gain.value = 0;
        theremin.osc.connect(theremin.gain).connect(theremin.ctx.destination);
        theremin.osc.start();
        $('.js-th-toggle').textContent = '⏹ Stop sound';
        const loop = () => {
            const f = energyFraction();
            theremin.smoothed += (f - theremin.smoothed) * 0.12;
            const freq = 140 * Math.pow(2, theremin.smoothed * 2.6);
            const now = theremin.ctx.currentTime;
            theremin.osc.frequency.setTargetAtTime(freq, now, 0.05);
            theremin.gain.gain.setTargetAtTime(0.02 + theremin.smoothed * 0.35, now, 0.08);
            $('.js-th-freq').textContent = Math.round(freq);
            theremin.raf = requestAnimationFrame(loop);
        };
        loop();
    }

    function thereminStop() {
        if (!theremin.ctx) return;
        cancelAnimationFrame(theremin.raf);
        theremin.osc.stop();
        theremin.ctx.close();
        theremin.ctx = null;
        theremin.osc = null;
        theremin.gain = null;
        theremin.smoothed = 0;
        const toggle = $('.js-th-toggle');
        if (toggle) toggle.textContent = '▶ Start sound';
        const freq = $('.js-th-freq');
        if (freq) freq.textContent = '—';
    }

    function thereminInit() {
        $('.js-th-toggle').addEventListener('click', () => {
            const starting = !theremin.ctx;
            starting ? thereminStart() : thereminStop();
            track('theremin_configuration', {
                control: 'playback', setting_value: starting ? 'start' : 'stop'
            });
        });
        document.getElementById('th-wave').addEventListener('change', (event) => {
            if (theremin.osc) theremin.osc.type = event.target.value;
            track('theremin_configuration', {
                control: 'waveform', setting_value: event.target.value
            });
        });
    }

    /* =========================================================== configure */

    function cfgValue(id) {
        return document.getElementById(id).value;
    }

    /**
     * Builds and writes a control command, reporting the outcome as
     * configure_change. `buildCommand` runs even in demo mode so the
     * library's validation gives the same feedback with or without a
     * device; nothing is written and nothing is tracked in demo, because
     * no device is involved.
     */
    async function cfgApply(action, successMessage, buildCommand) {
        let command;
        try {
            command = buildCommand();
        } catch (error) {
            if (error && error.name === 'ESPectreValidationError') {
                cfgValidationFailed(action, error.message);
                return false;
            }
            throw error;
        }
        if (conn.mode === 'demo') {
            toast(successMessage + ' (demo — nothing written)');
            return true;
        }
        if (!bleClient) {
            toast('ESPectre is not connected.');
            track('configure_change', { action, result: 'failure', error_type: 'NotConnected' });
            return false;
        }
        try {
            await bleClient.writeControl(command);
            toast(successMessage);
            track('configure_change', { action, result: 'success' });
            return true;
        } catch (error) {
            toast('Write failed: ' + (error.message || error));
            track('configure_change', { action, result: 'failure', error_type: errorType(error) });
            return false;
        }
    }

    function cfgValidationFailed(action, message) {
        toast(message);
        track('configure_change', { action, result: 'validation_failure' });
    }

    async function cfgRefreshSysinfo() {
        if (conn.mode === 'ble' && bleClient) {
            try {
                await bleClient.requestSysinfo();
            } catch (error) {
                console.warn('Sysinfo request failed:', error);
            }
        }
    }

    async function cfgSaveWifi() {
        const ssid = cfgValue('cfg-ssid').trim();
        const password = cfgValue('cfg-wifi-pass');
        // Site policy: the form requires a password even though the protocol
        // allows open networks; format validation belongs to the library.
        if (!ssid || !password) {
            cfgValidationFailed('set_wifi', 'Wi-Fi needs both SSID and password.');
            return;
        }
        const ok = await cfgApply('set_wifi', 'Wi-Fi credentials saved; station reconnecting.',
            () => window.ESPectreBleClient.buildWifiConfigCommand({
                ssid,
                password,
                bssid: cfgValue('cfg-bssid').trim(),
                channel: Number(cfgValue('cfg-channel') || 0)
            }));
        if (ok) document.getElementById('cfg-wifi-pass').value = '';
    }

    async function cfgClearWifi() {
        const ok = await cfgApply('clear_wifi', 'Wi-Fi credentials cleared.', () => 'CLEAR_WIFI');
        if (ok) {
            ['cfg-ssid', 'cfg-wifi-pass', 'cfg-bssid', 'cfg-channel'].forEach((id) => {
                document.getElementById(id).value = '';
            });
        }
    }

    async function cfgSaveMqtt() {
        const host = cfgValue('cfg-mqtt-host').trim();
        const username = cfgValue('cfg-mqtt-user').trim();
        const password = cfgValue('cfg-mqtt-pass');
        // Site policy: credentials are required here; the library accepts
        // anonymous brokers.
        if (!host || !username || !password || !cfgValue('cfg-mqtt-port')) {
            cfgValidationFailed('set_mqtt', 'MQTT needs host, port, username, and password.');
            return;
        }
        const ok = await cfgApply('set_mqtt', 'MQTT settings saved.',
            () => window.ESPectreBleClient.buildMqttConfigCommand({
                host,
                port: Number(cfgValue('cfg-mqtt-port')),
                username,
                password,
                topicPrefix: cfgValue('cfg-topic-prefix').trim() || undefined
            }));
        if (ok) document.getElementById('cfg-mqtt-pass').value = '';
    }

    async function cfgClearMqtt() {
        const ok = await cfgApply('clear_mqtt', 'MQTT settings cleared.', () => 'CLEAR_MQTT_CONFIG');
        if (ok) document.getElementById('cfg-mqtt-pass').value = '';
    }

    async function cfgSaveDevice() {
        const ok = await cfgApply('set_device', 'Device label saved.',
            () => window.ESPectreBleClient.buildDeviceLabelCommand(cfgValue('cfg-label').trim()));
        if (ok) cfgRefreshSysinfo();
    }

    async function cfgClearDevice() {
        const ok = await cfgApply('clear_device', 'Device config reset.', () => 'CLEAR_DEVICE_CONFIG');
        if (ok) {
            document.getElementById('cfg-label').value = '';
            cfgRefreshSysinfo();
        }
    }

    async function cfgSaveMotionHits() {
        const ok = await cfgApply('set_motion_hits', 'Motion hit thresholds saved.',
            () => window.ESPectreBleClient.buildMotionHitsCommand({
                motionOnHits: Number(cfgValue('cfg-motion-on')),
                motionOffHits: Number(cfgValue('cfg-motion-off'))
            }));
        if (ok) cfgRefreshSysinfo();
    }

    async function cfgOtaStatus() {
        const ok = await cfgApply('ota_status', 'OTA status requested.',
            () => window.ESPectreBleClient.buildOtaStatusCommand());
        if (ok) cfgRefreshSysinfo();
    }

    async function cfgOtaCheck() {
        const ok = await cfgApply('ota_check', 'OTA check started.',
            () => window.ESPectreBleClient.buildOtaCheckCommand());
        if (ok) cfgRefreshSysinfo();
    }

    async function cfgOtaStart() {
        const ok = await cfgApply('ota_start', 'OTA update started.',
            () => window.ESPectreBleClient.buildOtaStartCommand());
        if (ok) cfgRefreshSysinfo();
    }

    function configureInit() {
        $('.js-wifi-save').addEventListener('click', cfgSaveWifi);
        $('.js-wifi-clear').addEventListener('click', cfgClearWifi);
        $('.js-mqtt-save').addEventListener('click', cfgSaveMqtt);
        $('.js-mqtt-clear').addEventListener('click', cfgClearMqtt);
        $('.js-dev-save').addEventListener('click', cfgSaveDevice);
        $('.js-dev-clear').addEventListener('click', cfgClearDevice);
        $('.js-motion-save').addEventListener('click', cfgSaveMotionHits);
        $('.js-ota-status').addEventListener('click', cfgOtaStatus);
        $('.js-ota-check').addEventListener('click', cfgOtaCheck);
        $('.js-ota-start').addEventListener('click', cfgOtaStart);
    }

    /* ================================================================ game */

    const TOTAL_ROUNDS = 5;
    const game = {
        phase: 'idle',   // idle | hold | strike | done
        round: 0,
        score: 0,
        best: null,
        holdTimer: null,
        cooldownTimer: null,
        strikeAt: 0,
        strikeTimeout: null
    };

    function gameSet(selector, value) {
        const el = $(selector);
        if (el) el.textContent = value;
    }

    function gameOrb(state) {
        const orb = $('.js-game-orb');
        if (orb) orb.className = 'game-orb js-game-orb' + (state ? ' is-' + state : '');
    }

    function gameMsg(message) {
        gameSet('.js-game-msg', message);
    }

    function gameReset() {
        clearTimeout(game.holdTimer);
        clearTimeout(game.cooldownTimer);
        clearTimeout(game.strikeTimeout);
        game.phase = 'idle';
        game.round = 0;
        game.score = 0;
        gameOrb('');
        gameMsg('Stand still. Move fast. React to survive.');
        gameSet('.js-game-round', '—');
        gameSet('.js-game-ms', '—');
        gameSet('.js-game-score', '0');
        const start = $('.js-game-start');
        if (start) start.textContent = 'Start game';
    }

    function gameNextRound() {
        game.round += 1;
        if (game.round > TOTAL_ROUNDS) {
            game.phase = 'done';
            gameOrb('');
            gameMsg('Game over — final score ' + game.score + '. Play again?');
            $('.js-game-start').textContent = 'Play again';
            track('game_over', {
                input_mode: conn.mode === 'demo' ? 'demo' : 'bluetooth',
                score: game.score,
                rounds: TOTAL_ROUNDS,
                ...(game.best !== null ? { best_time: game.best } : {})
            });
            return;
        }
        game.phase = 'hold';
        gameSet('.js-game-round', game.round + ' / ' + TOTAL_ROUNDS);
        gameOrb('hold');
        gameMsg('Round ' + game.round + ' — the Spectre is watching... stay perfectly still.');
        $('.js-game-hint').textContent = conn.mode === 'demo'
            ? 'Demo mode: stay still, then move the mouse when the Spectre strikes.'
            : 'Freeze. Any motion now counts as a false start.';
        game.holdTimer = setTimeout(() => {
            game.phase = 'strike';
            game.strikeAt = performance.now();
            gameOrb('strike');
            gameMsg('⚡ MOVE! The Spectre strikes!');
            game.strikeTimeout = setTimeout(() => {
                if (game.phase !== 'strike') return;
                gameOrb('fail');
                gameMsg('Too slow — the Spectre hit you.');
                gameSet('.js-game-ms', 'miss');
                gameEndRound();
            }, 2500);
        }, 1800 + Math.random() * 2600);
    }

    function gameEndRound() {
        game.phase = 'cooldown';
        clearTimeout(game.holdTimer);
        clearTimeout(game.cooldownTimer);
        clearTimeout(game.strikeTimeout);
        const waitForDemoSettle = () => {
            const settleThreshold = Math.max(0.18, conn.threshold * 0.7);
            if (demoInputEnergy > 0 || conn.motion || conn.movement >= settleThreshold) {
                game.cooldownTimer = setTimeout(waitForDemoSettle, 220);
                return;
            }
            demoResetMotion();
            gameNextRound();
        };
        if (conn.mode === 'demo') {
            game.cooldownTimer = setTimeout(waitForDemoSettle, 900);
        } else {
            game.cooldownTimer = setTimeout(gameNextRound, 1700);
        }
    }

    function gameOnTelemetry() {
        if (route !== 'game') return;
        if (game.phase === 'hold' && conn.motion) {
            gameOrb('fail');
            gameMsg('False start! The Spectre saw you twitch.');
            gameSet('.js-game-ms', 'false start');
            gameEndRound();
        } else if (game.phase === 'strike' && conn.motion) {
            const ms = Math.round(performance.now() - game.strikeAt);
            const points = Math.max(0, 1500 - ms);
            game.score += points;
            if (game.best === null || ms < game.best) {
                game.best = ms;
                gameSet('.js-game-best', game.best + ' ms');
            }
            gameOrb('win');
            gameMsg('Hit! ' + ms + ' ms → +' + points + ' points');
            gameSet('.js-game-ms', ms + ' ms');
            gameSet('.js-game-score', String(game.score));
            gameEndRound();
        }
    }

    function gameInit() {
        $('.js-game-start').addEventListener('click', () => {
            clearTimeout(game.holdTimer);
            clearTimeout(game.cooldownTimer);
            clearTimeout(game.strikeTimeout);
            game.round = 0;
            game.score = 0;
            demoResetMotion();
            gameSet('.js-game-score', '0');
            $('.js-game-start').textContent = 'Restart';
            track('game_start', { input_mode: conn.mode === 'demo' ? 'demo' : 'bluetooth' });
            gameNextRound();
        });
    }

    /* ================================================================ init */

    function init() {
        scrollyInit();

        $$('.js-connect').forEach((btn) => btn.addEventListener('click', connectBle));
        $$('.js-demo').forEach((btn) => btn.addEventListener('click', connectDemo));
        $('.js-disconnect').addEventListener('click', disconnect);
        $('.js-dropdown-toggle').addEventListener('click', (event) => {
            event.stopPropagation();
            dropdownOpen = !dropdownOpen;
            renderConnection();
        });
        document.addEventListener('click', (event) => {
            if (dropdownOpen && !event.target.closest('.conn')) {
                dropdownOpen = false;
                renderConnection();
            }
        });
        configureInit();
        flashInit();
        monitorInit();
        thereminInit();
        gameInit();

        document.addEventListener('click', interceptCanonicalLinks);
        document.addEventListener('mousemove', demoTrackMouse, { passive: true });
        window.addEventListener('hashchange', onHashChange);
        setRoute((location.hash || '#home').slice(1), { force: true });
    }

    document.addEventListener('DOMContentLoaded', init);
})();
