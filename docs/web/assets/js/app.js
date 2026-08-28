/*
 * ESPectre - Website app shell
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';


    const routeRegistry = window.ESPectreRoutes;
    if (!routeRegistry) throw new Error('ESPectre route registry is unavailable');
    const sitePolicy = window.ESPectreSite;
    if (!sitePolicy) throw new Error('ESPectre site policy is unavailable');
    const browserSupport = window.ESPectreBrowserSupport && window.ESPectreBrowserSupport.current;
    if (!browserSupport) throw new Error('ESPectre browser capability policy is unavailable');
    const DirectProtocolClient = window.ESPectreDirectClient;
    if (!DirectProtocolClient) throw new Error('ESPectre Direct HTTP client is unavailable');

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => Array.from(document.querySelectorAll(sel));

    // analytics.js is optional: the app must work with it blocked or absent.
    const track = (name, params) => window.trackEvent ? window.trackEvent(name, params) : false;
    const errorType = (error) => (error && (error.code || error.name)) || 'Error';
    const toolNameForRoute = (routeName) => routeRegistry.groupOf(routeName) === 'tools'
        ? (routeRegistry.get(routeName)?.analyticsName || routeName)
        : 'monitor';
    const activeToolName = () => toolNameForRoute(route);
    const LEGACY_TOOL_ROUTES = Object.freeze({ device: 'tool-configure' });
    const MQTT_PRESETS = Object.freeze({
        home_assistant: Object.freeze({
            configure: Object.freeze({
                host: 'homeassistant.local', port: '1883', hostPlaceholder: 'homeassistant.local'
            })
        }),
        lan_broker: Object.freeze({
            configure: Object.freeze({
                host: '', port: '1883', hostPlaceholder: 'broker.local or 192.168.1.20'
            })
        }),
        emqx_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'deployment-id.ala.region.emqxsl.com', port: '8883',
                hostPlaceholder: 'deployment-id.ala.region.emqxsl.com',
                locked: Object.freeze(['port'])
            })
        }),
        hivemq_cloud: Object.freeze({
            configure: Object.freeze({
                host: 'cluster-id.s1.region.hivemq.cloud', port: '8883',
                hostPlaceholder: 'cluster-id.s1.region.hivemq.cloud',
                locked: Object.freeze(['port'])
            })
        }),
        flespi: Object.freeze({
            configure: Object.freeze({
                host: 'mqtt.flespi.io', port: '8883', hostPlaceholder: 'mqtt.flespi.io',
                locked: Object.freeze(['host', 'port'])
            })
        }),
        cloud_broker: Object.freeze({
            configure: Object.freeze({
                host: 'cluster.example.com', port: '', hostPlaceholder: 'cluster.example.com'
            })
        })
    });
    const SECURE_CLOUD_MQTT_PRESETS = new Set([
        'emqx_cloud', 'hivemq_cloud', 'flespi', 'cloud_broker'
    ]);
    const MQTT_FORM_DEFAULTS = {
        topicPrefix: 'espectre/v1/devices'
    };

    const dependencyPromises = new Map();
    const browserDependencyPromises = new Map();

    function loadScriptOnce(src, { module = false } = {}) {
        if (dependencyPromises.has(src)) return dependencyPromises.get(src);
        const promise = new Promise((resolve, reject) => {
            const existing = document.querySelector(`script[src="${src}"]`);
            if (existing && existing.dataset.loaded === 'true') {
                resolve();
                return;
            }
            const script = existing || document.createElement('script');
            if (module) script.type = 'module';
            script.src = src;
            script.addEventListener('load', () => {
                script.dataset.loaded = 'true';
                resolve();
            }, { once: true });
            script.addEventListener('error', () => {
                script.remove();
                reject(new Error(`Unable to load ${src}`));
            }, { once: true });
            if (!existing) document.head.appendChild(script);
        });
        dependencyPromises.set(src, promise);
        promise.catch(() => dependencyPromises.delete(src));
        return promise;
    }

    function loadBrowserDependency(localSrc, developmentCdnSrc, options = {}) {
        if (browserDependencyPromises.has(localSrc)) {
            return browserDependencyPromises.get(localSrc);
        }
        const promise = loadScriptOnce(localSrc, options).catch((error) => {
            if (!sitePolicy.isLoopbackHostname(location.hostname)) throw error;
            console.warn(`Local dependency unavailable; using development CDN fallback: ${developmentCdnSrc}`);
            return loadScriptOnce(developmentCdnSrc, options);
        });
        browserDependencyPromises.set(localSrc, promise);
        promise.catch(() => browserDependencyPromises.delete(localSrc));
        return promise;
    }

    /* ============================================================= routing */

    function focusRouteContent(routeName = route) {
        const page = $(`[data-page="${routeName}"]`);
        if (!page) return;
        const target = page.querySelector('h1') || page;
        if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
    }

    function focusRouteAnchor(routeName, encodedTargetId) {
        let targetId = '';
        try {
            targetId = decodeURIComponent(encodedTargetId);
        } catch (error) {
            return false;
        }
        const page = $(`[data-page="${routeName}"]`);
        const target = page && document.getElementById(targetId);
        if (!target || !page.contains(target)) return false;
        target.scrollIntoView();
        if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
        return true;
    }

    let pendingRouteAnchor = '';

    function consumeRouteAnchorHandoff() {
        const url = new URL(location.href);
        const anchor = url.searchParams.get('anchor') || '';
        if (!anchor) return;
        pendingRouteAnchor = anchor;
        url.searchParams.delete('anchor');
        history.replaceState(null, '', url.pathname + url.search + url.hash);
    }

    function applyRoute({ focus = true } = {}) {
        const routeAtStart = route;
        const anchorAtStart = pendingRouteAnchor;
        pendingRouteAnchor = '';
        $$('.js-page').forEach((page) => {
            const current = page.dataset.page === route;
            page.hidden = !current;
            if (current) page.id = 'main-content';
            else page.removeAttribute('id');
        });
        $$('[data-route-link]').forEach((link) => {
            const target = link.dataset.routeLink;
            const active = target === route
                || routeRegistry.groupOf(route) === target;
            link.classList.toggle('active', active);
            if (active) link.setAttribute('aria-current', 'page');
            else link.removeAttribute('aria-current');
        });
        document.title = window.getRouteTitle
            ? window.getRouteTitle(route)
            : 'ESPectre — Wi-Fi motion sensing';
        window.scrollTo(0, 0);
        if (route !== 'tool-theremin') thereminStop();
        const contentPromise = $(`[data-page="${routeAtStart}"] .js-static-content`)
            ? prepareRouteContent(routeAtStart)
            : Promise.resolve(true);
        if (route === 'home') updateReleaseBadge();
        contentPromise.then((ready) => {
            if (!ready || route !== routeAtStart) return;
            renderBrowserSupport();
            renderDirectBrowserGuidance();
            renderStoredDirectEndpoints();
            renderConnection();
            consumeDirectHandoff();
            if (routeAtStart === 'tool-monitor') monitorResizeChart();
            if (routeAtStart === 'tool-raw-csi') rawCsiUseConnection();
            if (routeAtStart === 'tool-game') {
                void gameLoadFactoryImage();
                requestAnimationFrame(() => {
                    gameResizeCanvas();
                    gameSetFlight(gameSensingActive());
                    gameStartPreview();
                });
            }
            if (routeAtStart === 'tool-flash') {
                if (browserSupport.flash) {
                    loadBrowserDependency(
                        '/vendor/esp-web-tools-10.4.0/install-button.js',
                        'https://unpkg.com/esp-web-tools@10.4.0/dist/web/install-button.js?module',
                        { module: true }
                    ).catch((error) => {
                        console.warn('USB installer could not be loaded:', error);
                        flashStatus('The USB installer could not be loaded. Refresh the page and try again.', 'is-error');
                    });
                }
                flashRefresh();
            }
            if (focus || anchorAtStart) {
                if (!anchorAtStart || !focusRouteAnchor(routeAtStart, anchorAtStart)) {
                    focusRouteContent(routeAtStart);
                }
            }
        });
        // The router owns navigation, so it reports it.
        if (window.trackRouteView) window.trackRouteView(route);
    }

    function clearApiReferenceLocation(previousRoute, nextRoute) {
        if (previousRoute !== 'sdk-api' || nextRoute === 'sdk-api') return;
        const url = new URL(location.href);
        if (!url.searchParams.has('api') && !url.searchParams.has('member')) return;
        url.searchParams.delete('api');
        url.searchParams.delete('member');
        history.replaceState(history.state, '', url.pathname + url.search + url.hash);
    }

    /**
     * Single entry point for navigation. `force` applies the current route on
     * startup; without it a repeated route is ignored so one navigation never
     * reports two page views.
     */
    function setRoute(next, { force = false, focus = true } = {}) {
        const remapped = LEGACY_TOOL_ROUTES[next] || next;
        const target = routeRegistry.has(remapped) ? remapped : 'home';
        if (!force && target === route) return;
        cancelDirectDiscovery({ clear: true });
        const previousRoute = route;
        clearApiReferenceLocation(previousRoute, target);
        if (previousRoute === 'tool-raw-csi' && target !== 'tool-raw-csi') {
            void rawCsiStop();
        }
        if (pendingLiveDestination) {
            if (LIVE_EXPERIENCE_ROUTES.has(target)) pendingLiveDestination = target;
            else if (target !== 'tool-monitor' && target !== 'tool-configure') pendingLiveDestination = '';
        }
        if (previousRoute === 'tool-game' && target !== 'tool-game') {
            gameExitFullscreen();
            reportGameAbandon('route_change');
        }
        if (target === 'tool-game' && previousRoute !== 'tool-game') resetGameThreshold();
        route = target;
        dropdownOpen = false;
        applyRoute({ focus });
        renderConnection();
    }

    function onHashChange() {
        setRoute((location.hash || '#home').slice(1));
    }

    /* ======================================================= static content */

    /*
     * Tools, guides, docs, media, and the roadmap live in shared HTML
     * fragments, which also build their canonical static pages. The SPA
     * fetches each fragment on first visit so content is not duplicated and
     * the device connection survives.
    */
    const staticContentCache = new Map();
    const staticContentLoads = new Map();
    const initializedToolRoutes = new Set();
    const toolInitializers = Object.freeze({
        'tool-flash': flashInit,
        'tool-configure': configureInit,
        'tool-monitor': monitorInit,
        'tool-raw-csi': rawCsiInit,
        'tool-theremin': thereminInit,
        'tool-game': gameInit
    });

    function loadStaticContent(route) {
        const container = $(`[data-page="${route}"] .js-static-content`);
        if (!container || container.dataset.loaded === 'true') return Promise.resolve(true);
        if (staticContentLoads.has(route)) return staticContentLoads.get(route);
        const contentUrl = container.dataset.contentUrl;
        const load = (async () => {
            try {
                if (!staticContentCache.has(contentUrl)) {
                    const response = await fetch(contentUrl);
                    if (!response.ok) throw new Error('HTTP ' + response.status);
                    staticContentCache.set(contentUrl, await response.text());
                }
                container.innerHTML = staticContentCache.get(contentUrl);
                if (window.initPageTocs) window.initPageTocs(container);
                if (window.initPagePaths) window.initPagePaths(container);
                if (window.initSdkDownloadVersions) window.initSdkDownloadVersions(container);
                if (window.initPublishedReleaseTags) window.initPublishedReleaseTags(container);
                if (window.initCodeTabs) window.initCodeTabs(container);
                if (window.initApiReferenceBrowsers) window.initApiReferenceBrowsers(container);
                container.dataset.loaded = 'true';
                return true;
            } catch (error) {
                console.warn('Static content fetch failed:', error);
                container.innerHTML = '<p class="guide-loading">This page could not be loaded. '
                    + '<a href="' + container.dataset.staticUrl + '">Open the standalone page</a>.</p>';
                return false;
            }
        })();
        staticContentLoads.set(route, load);
        load.finally(() => staticContentLoads.delete(route));
        return load;
    }

    async function prepareRouteContent(route) {
        const ready = await loadStaticContent(route);
        const initializer = toolInitializers[route];
        if (!ready || !initializer || initializedToolRoutes.has(route)) return ready;
        initializer();
        initializedToolRoutes.add(route);
        if (conn.mode === 'demo' && demoSysinfoSnapshot) {
            applySysinfo(demoSysinfoSnapshot);
        }
        return true;
    }

    /*
     * Fragments and content cards link to the canonical static URLs. Inside
     * the app those clicks become hash navigation, so the page never reloads
     * and an active connection survives; modified clicks (new tab) are left
     * to the browser. Root-anchored hash links are normalized for the same
     * reason: from /index.html they would otherwise reload onto /. Same-page
     * anchors scroll within the active SPA page without replacing its route.
     */
    function interceptCanonicalLinks(event) {
        if (event.defaultPrevented || event.button !== 0) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        const link = event.target.closest('a[href]');
        if (!link) return;
        const href = link.getAttribute('href');
        const staticTarget = routeRegistry.staticTargetForHref(href, location.href);
        if (staticTarget) {
            event.preventDefault();
            const routeHash = '#' + staticTarget.route;
            if (location.hash !== routeHash) {
                pendingRouteAnchor = staticTarget.anchor;
                location.hash = routeHash;
            } else if (staticTarget.anchor) {
                loadStaticContent(staticTarget.route).finally(() => {
                    if (route !== staticTarget.route) return;
                    if (!focusRouteAnchor(staticTarget.route, staticTarget.anchor)) {
                        focusRouteContent(staticTarget.route);
                    }
                });
            }
        } else if (href.startsWith('/#')) {
            event.preventDefault();
            location.hash = href.slice(1);
        } else if (href.startsWith('#') && href.length > 1) {
            const page = $(`[data-page="${route}"]`);
            const targetId = href.slice(1);
            let decodedTargetId = '';
            try {
                decodedTargetId = decodeURIComponent(targetId);
            } catch (error) {
                return;
            }
            const target = page && document.getElementById(decodedTargetId);
            if (!target || !page.contains(target)) return;
            event.preventDefault();
            focusRouteAnchor(route, targetId);
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

    function clearDirectConnectionCallout() {
        clearTimeout(connectionCalloutTimer);
        connectionCalloutTimer = null;
        directCalloutVisible = false;
    }

    function showDirectConnectionCallout() {
        clearDirectConnectionCallout();
        directCalloutVisible = true;
        connectionCalloutTimer = setTimeout(() => {
            connectionCalloutTimer = null;
            directCalloutVisible = false;
            syncConnectionCallout();
        }, DIRECT_CALLOUT_DURATION_MS);
    }

    function syncConnectionCallout() {
        const el = $('.js-connection-callout');
        if (!el) return;
        const demo = conn.mode === 'demo' && conn.status === 'connected';
        const direct = conn.mode === 'direct' && conn.status === 'connected' && directCalloutVisible;
        const title = $('.js-connection-callout-title');
        const message = $('.js-connection-callout-message');
        if (title) title.textContent = demo ? 'Demo mode' : 'Device connected';
        if (message) {
            message.textContent = demo
                ? 'Move the pointer to simulate motion.'
                : 'ESPectre is ready to use.';
        }
        el.hidden = (!demo && !direct) || dropdownOpen;
    }

    /* ====================================================== scroll narrative */

    let activeScrollyScene = -1;
    let scrollyFrame = null;
    let scrollyKeyTargetScene = null;
    let scrollyKeyTargetTimer = null;
    let heroFrameTimer = null;
    const HERO_FRAME_HOLD = 2000;

    function scrollySceneFromPosition(section, sceneCount) {
        const rect = section.getBoundingClientRect();
        const travel = Math.max(1, rect.height - window.innerHeight);
        const progress = Math.min(1, Math.max(0, -rect.top / travel));
        return Math.min(sceneCount - 1, Math.floor(progress * sceneCount));
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

        const scenes = $$('.js-scrolly-scene');
        const useMobileAsset = window.matchMedia('(max-width: 720px)').matches;
        [scene, scene + 1].forEach((index) => {
            const image = scenes[index] && scenes[index].querySelector('img[data-src]');
            if (!image) return;
            image.src = useMobileAsset && image.dataset.srcMobile
                ? image.dataset.srcMobile
                : image.dataset.src;
            image.removeAttribute('data-src');
            image.removeAttribute('data-src-mobile');
        });

        $$('.js-scrolly-scene, .js-scrolly-caption, .js-scrolly-marker').forEach((el) => {
            const isActive = Number(el.dataset.scene) === scene;
            el.classList.toggle('is-active', isActive);
            if (el.classList.contains('js-scrolly-caption')) {
                el.toggleAttribute('inert', !isActive);
                el.setAttribute('aria-hidden', String(!isActive));
            }
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

        const sceneCount = $$('.js-scrolly-scene').length;
        setScrollyScene(scrollySceneFromPosition(section, sceneCount));
    }

    function queueScrollyRender() {
        if (scrollyFrame !== null) return;
        scrollyFrame = requestAnimationFrame(renderScrolly);
    }

    function scrollyHandleKeydown(event) {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        const target = event.target;
        if (target instanceof Element && target.closest('a, button, input, select, textarea, [contenteditable="true"]')) return;

        const section = $('.js-scrolly');
        if (!section || section.offsetParent === null) return;
        const rect = section.getBoundingClientRect();
        if (rect.bottom <= 0 || rect.top >= window.innerHeight) return;

        const sceneCount = $$('.js-scrolly-scene').length;
        const currentScene = scrollyKeyTargetScene === null
            ? scrollySceneFromPosition(section, sceneCount)
            : scrollyKeyTargetScene;
        const direction = event.key === 'ArrowDown' ? 1 : -1;
        const nextScene = Math.min(sceneCount - 1, Math.max(0, currentScene + direction));
        if (nextScene === currentScene) return;

        event.preventDefault();
        scrollyKeyTargetScene = nextScene;
        clearTimeout(scrollyKeyTargetTimer);
        scrollyKeyTargetTimer = setTimeout(() => { scrollyKeyTargetScene = null; }, 500);

        const travel = Math.max(1, rect.height - window.innerHeight);
        const sectionTop = window.scrollY + rect.top;
        const sceneProgress = (nextScene + 0.5) / sceneCount;
        window.scrollTo({
            top: sectionTop + (travel * sceneProgress),
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
        });
    }

    function scrollyInit() {
        window.addEventListener('scroll', queueScrollyRender, { passive: true });
        window.addEventListener('resize', queueScrollyRender);
        document.addEventListener('keydown', scrollyHandleKeydown);
        renderScrolly();
    }

    function sharedDialogsInit() {
        $$('.js-matter-close').forEach((button) => {
            button.addEventListener('click', () => matterClose());
        });
        $('.js-matter-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) matterClose();
        });
        $$('.js-config-clear-cancel').forEach((button) => {
            button.addEventListener('click', () => closeConfigClearDialog(false));
        });
        $('.js-config-clear-confirm').addEventListener('click', () => closeConfigClearDialog(true));
        $('.js-config-clear-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) closeConfigClearDialog(false);
        });
        $('.js-ota-start').addEventListener('click', cfgOtaStart);
        document.getElementById('ota-channel').addEventListener('change', () => {
            if (conn.mode === null) return;
            otaChannelChanged = true;
            startManualOtaCheck();
        });
        $$('.js-ota-close').forEach((button) => {
            button.addEventListener('click', () => otaClose());
        });
        $('.js-ota-modal').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) otaClose();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            if (!$('.js-matter-modal').hidden) matterClose();
            else if (!$('.js-config-clear-modal').hidden) closeConfigClearDialog(false);
            else if (!$('.js-ota-modal').hidden) otaClose();
        });
    }

    function sharedToolControlsInit() {
        document.addEventListener('click', (event) => {
            if (!(event.target instanceof Element)) return;
            const connectButton = event.target.closest('.js-connect-direct');
            if (connectButton) {
                void connectDirect({
                    openView: connectButton.closest('espectre-direct-connect')?.dataset.openView
                });
                return;
            }
            const discoveryButton = event.target.closest('.js-direct-discover');
            if (discoveryButton) {
                void discoverLocalPeers(discoveryButton);
                return;
            }
            const discoveredDevice = event.target.closest('.direct-discovery-device');
            if (discoveredDevice?.dataset.endpoint) {
                const input = discoveredDevice.closest('.device-connect-card')
                    ?.querySelector('input[list="direct-remembered-endpoints"]');
                if (input) input.value = discoveredDevice.dataset.deviceId;
                void connectDirect({
                    endpoint: discoveredDevice.dataset.endpoint,
                    deviceId: discoveredDevice.dataset.deviceId,
                    openView: discoveredDevice.closest('espectre-direct-connect')?.dataset.openView
                });
                return;
            }
            const startButton = event.target.closest('.js-start-detection');
            if (startButton) {
                void startDetection(startButton.dataset.liveTransport || '');
                return;
            }
            const demoButton = event.target.closest('.js-demo');
            if (demoButton) {
                connectDemo(demoButton.closest('espectre-connection-picker')?.dataset.openView || '');
                return;
            }
            const firmwareButton = event.target.closest('.js-firmware-update-notice');
            if (firmwareButton) otaOpen(firmwareButton);
        });
    }

    /* ================================================================ init */

    function init() {
        scrollyInit();

        renderBrowserSupport();
        renderDirectBrowserGuidance();
        renderStoredDirectEndpoints();
        consumeRouteAnchorHandoff();
        sharedDialogsInit();
        sharedToolControlsInit();
        $('.js-header-connect').addEventListener('click', () => {
            selectMonitorTransport('direct');
            if (route === 'tool-monitor') {
                document.getElementById('monitor-direct-endpoint')?.focus();
                return;
            }
            pendingLiveDestination = '';
            location.hash = '#tool-monitor';
        });
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
        document.addEventListener('click', interceptCanonicalLinks);
        $('.skip-link').addEventListener('click', (event) => {
            event.preventDefault();
            focusRouteContent();
        });
        document.addEventListener('mousemove', demoTrackMouse, { passive: true });
        window.addEventListener('hashchange', onHashChange);
        setRoute((location.hash || '#home').slice(1), { force: true, focus: false });
    }

    document.addEventListener('espectre:analytics-enabled', () => {
        if (window.trackRouteView) window.trackRouteView(route, { sendPageView: false });
        if (conn.readyState) markToolReady(conn.readyState);
        if (monitor.readyState) markMonitorReady(monitor.readyState);
        if (conn.mode === 'direct' && directClient) cfgRefreshDevice();
        if (route === 'tool-flash') {
            void prepareRouteContent(route).then((ready) => {
                if (ready && route === 'tool-flash') flashRefresh();
            });
        }
    });
    window.addEventListener('pagehide', (event) => {
        if (event.persisted) return;
        void rawCsiStop();
        reportGameAbandon('page_exit');
        if (conn.mode) teardownConnection('page_exit');
        else monitorStopAll('page_exit');
    });
    document.addEventListener('DOMContentLoaded', init);
