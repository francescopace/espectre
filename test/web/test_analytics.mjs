/*
 * ESPectre - Website analytics unit tests
 *
 * Copyright 2026 Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const analyticsSource = readFileSync(
    new URL('../../docs/web/assets/js/analytics.js', import.meta.url), 'utf8'
);
const routeRegistrySource = readFileSync(
    new URL('../../docs/web/assets/js/route-registry.js', import.meta.url), 'utf8'
);
const routeManifest = JSON.parse(readFileSync(
    new URL('../../docs/web/routes.json', import.meta.url), 'utf8'
));
const testExports = `
globalThis.__analyticsTest = {
    analyticsAllowedHere, applyConsent, commitConsent, disableAnalytics, enableAnalytics, getRouteTitle, getSiteSection,
    initializeConsentControls, routePath, sanitizeAnalyticsEvent, sendRoutePageView,
    sendStaticPageView, trackCampaignConversion, trackEvent, trackRouteView,
    consentKey: CONSENT_KEY, adsId: ADS_ID, gaId: GA_MEASUREMENT_ID, conversions: ADS_CONVERSIONS,
    enabled: () => analyticsEnabled
};`;

function analyticsContext({
    hostname = 'espectre.dev', path = '/', hash = '', search = '', staticPage = false,
    navigatorValues = {}, storedConsent = null, preferences = null, storageBlocked = false, preloadManifest = true
} = {}) {
    const appendedScripts = [];
    const listeners = new Map();
    const storage = new Map();
    if (storedConsent !== null) storage.set('espectre.analytics.consent.v1', storedConsent);
    if (preferences !== null) storage.set('espectre.consent.v2', JSON.stringify(preferences));
    const controls = Object.fromEntries(['accept', 'reject', 'customize', 'save', 'analytics', 'campaigns', 'preferences'].map((name) => [
        `.js-consent-${name}`, {
            checked: false, hidden: true, handlers: {}, attributes: {}, focused: false,
            addEventListener(event, handler) { this.handlers[event] = handler; },
            setAttribute(name, value) { this.attributes[name] = value; },
            focus() { this.focused = true; },
            click() { this.handlers.click?.({ currentTarget: this }); }
        }
    ]));
    const consentBanner = { hidden: true };
    const document = {
        cookie: '',
        title: 'Test page | ESPectre',
        documentElement: {
            hasAttribute: (name) => name === 'data-static-page' && staticPage,
            dataset: staticPage ? { siteSection: 'documentation' } : {}
        },
        head: { appendChild: (node) => appendedScripts.push(node) },
        createElement: (tagName) => ({ tagName, addEventListener() {} }),
        querySelector: (selector) => selector === '.js-consent-banner' ? consentBanner : controls[selector] || null,
        querySelectorAll: (selector) => controls[selector] ? [controls[selector]] : [],
        addEventListener: (name, callback) => listeners.set(name, callback),
        dispatchEvent: (event) => listeners.get(event.type)?.(event)
    };
    const location = {
        hostname,
        pathname: path,
        hash,
        search,
        origin: `https://${hostname}`,
        get href() { return this.origin + this.pathname + this.search + this.hash; }
    };
    const window = {
        location,
        localStorage: {
            getItem: (key) => { if (storageBlocked) throw Error('blocked'); return storage.get(key) ?? null; },
            setItem: (key, value) => { if (storageBlocked) throw Error('blocked'); storage.set(key, value); },
            removeItem: (key) => storage.delete(key)
        },
        history: {
            state: null,
            replaceState(_state, _title, url) {
                const next = new URL(url, location.origin);
                location.pathname = next.pathname; location.search = next.search; location.hash = next.hash;
            }
        }
    };
    if (preloadManifest) window.ESPectreRouteManifest = routeManifest;
    const context = vm.createContext({
        console, Date, Map, Set, URL, URLSearchParams, Object, navigator: navigatorValues, document, location,
        fetch: async (url) => ({
            ok: url === '/routes.json',
            status: url === '/routes.json' ? 200 : 404,
            json: async () => routeManifest
        }),
        CustomEvent: class CustomEvent { constructor(type) { this.type = type; } },
        window, globalThis: null
    });
    context.globalThis = context;
    vm.runInContext(routeRegistrySource + analyticsSource + testExports, context);
    return {
        api: context.__analyticsTest,
        sitePolicy: context.window.ESPectreSite,
        window,
        appendedScripts,
        consentBanner,
        controls, storage, document,
        listeners
    };
}

describe('analytics privacy boundary', () => {
    it('uses one site policy for production, validation, and loopback hosts', () => {
        const { sitePolicy } = analyticsContext();
        const locationFor = (hostname, protocol = 'https:') => ({ hostname, protocol });

        assert.equal(sitePolicy.analyticsAllowed(locationFor('espectre.dev')), true);
        assert.equal(sitePolicy.analyticsAllowed(locationFor('test.espectre.dev')), true);
        assert.equal(sitePolicy.analyticsAllowed(locationFor('localhost', 'http:')), true);
        assert.equal(sitePolicy.analyticsAllowed(locationFor('example.test')), false);
        assert.equal(sitePolicy.analyticsDebug(locationFor('espectre.dev')), false);
        assert.equal(sitePolicy.analyticsDebug(locationFor('test.espectre.dev')), true);
        assert.equal(sitePolicy.directOriginKind(locationFor('espectre.dev')), 'production');
        assert.equal(sitePolicy.directOriginKind(locationFor('test.espectre.dev')), 'validation');
        assert.equal(sitePolicy.directOriginKind(locationFor('localhost', 'http:')), 'loopback');
        assert.equal(sitePolicy.directOriginKind(locationFor('localhost')), 'other');
    });

    it('does not enable or load GA outside production', () => {
        const { api, appendedScripts, window } = analyticsContext({ hostname: 'example.test' });
        api.enableAnalytics();
        assert.equal(api.enabled(), false);
        assert.equal(appendedScripts.length, 0);
        assert.equal(window.dataLayer, undefined);
    });

    it('enables debug collection on the hosted test site and loopback after consent', () => {
        for (const hostname of ['test.espectre.dev', 'localhost']) {
            const enabled = analyticsContext({ hostname });
            enabled.api.enableAnalytics({ sendPageView: false });
            assert.equal(enabled.api.enabled(), true);
            const config = enabled.window.dataLayer.find((entry) => entry[0] === 'config');
            assert.equal(config[2].debug_mode, true);
        }
    });

    it('shows consent on the hosted test site without loading GA before a choice', () => {
        const { api, appendedScripts, consentBanner, window } = analyticsContext({
            hostname: 'test.espectre.dev'
        });
        api.initializeConsentControls();
        assert.equal(consentBanner.hidden, false);
        assert.equal(appendedScripts.length, 0);
        assert.equal(window.dataLayer, undefined);
    });

    it('uses denied-by-default consent and disables advertising signals', () => {
        const { api, appendedScripts, window } = analyticsContext();
        api.enableAnalytics({ sendPageView: false });

        assert.equal(api.enabled(), true);
        assert.equal(appendedScripts.length, 1);
        assert.match(appendedScripts[0].src, /googletagmanager\.com\/gtag\/js\?id=G-S0NQNG0V11/);
        assert.equal(window.dataLayer[0][0], 'consent');
        assert.equal(window.dataLayer[0][1], 'default');
        assert.equal(window.dataLayer[0][2].analytics_storage, 'denied');
        assert.equal(window.dataLayer[1][1], 'update');
        assert.equal(window.dataLayer[1][2].analytics_storage, 'granted');
        const config = window.dataLayer.find((entry) => entry[0] === 'config');
        assert.equal(config[2].send_page_view, false);
        assert.equal(config[2].allow_google_signals, false);
        assert.equal(config[2].allow_ad_personalization_signals, false);
    });

    it('gates custom events until analytics is enabled', () => {
        const { api, window } = analyticsContext();
        api.trackEvent('firmware_catalog', { result: 'success' });
        assert.equal(window.dataLayer, undefined);
        api.enableAnalytics();
        const before = window.dataLayer.length;
        api.trackEvent('firmware_catalog', { result: 'success' });
        assert.equal(window.dataLayer.length, before + 1);
        assert.equal(window.dataLayer.at(-1)[0], 'event');
        assert.equal(window.dataLayer.at(-1)[1], 'firmware_catalog');
    });

    it('preserves only public campaign slugs across navigation and delayed consent', () => {
        const { api, appendedScripts, window } = analyticsContext({
            search: '?utm_source=github&utm_medium=social&utm_campaign=rc3&utm_content=readme'
                + '&utm_id=release-3&target=192.168.1.10&password=secret&utm_term=private'
        });
        window.location.search = '';
        window.location.pathname = '/tools/flash/';
        assert.equal(appendedScripts.length, 0);
        assert.equal(window.dataLayer, undefined);
        api.enableAnalytics();
        const config = window.dataLayer.find((entry) => entry[0] === 'config')[2];
        assert.equal(config.campaign_source, 'github');
        assert.equal(config.campaign_medium, 'social');
        assert.equal(config.campaign_name, 'rc3');
        assert.equal(config.campaign_content, 'readme');
        assert.equal(config.campaign_id, 'release-3');
        assert.equal(config.campaign_term, undefined);
        assert.equal(config.page_location, 'https://espectre.dev/tools/flash/');
        assert.doesNotMatch(JSON.stringify(window.dataLayer), /192\.168|secret|private/);
        api.disableAnalytics();
        assert.equal(api.trackEvent('firmware_usb_result', { result: 'identified' }), false);
    });

    it('rejects campaign URLs, addresses, free text, and oversized values', () => {
        const { api, window } = analyticsContext({
            search: '?utm_source=user%40example.com&utm_medium=https://example.com'
                + '&utm_campaign=private%20text&utm_id=192.168.1.2&utm_content=' + 'a'.repeat(65)
        });
        api.enableAnalytics({ sendPageView: false });
        const config = window.dataLayer.find((entry) => entry[0] === 'config')[2];
        assert.equal(Object.keys(config).some((key) => key.startsWith('campaign_')), false);
    });

    it('keeps reliability diagnostics bounded and excludes private payloads', () => {
        const { api, window } = analyticsContext();
        const cases = [
            ['firmware_usb_result', { result: 'cancelled', stage: 'port_selection', flow: 'flash' }],
            ['wifi_provision_result', { result: 'failure', stage: 'provision', error_type: 'NetworkError' }],
            ['tool_load_error', { tool_name: 'game', stage: 'script', error_type: 'TypeError' }],
            ['local_discovery', { method: 'target', result: 'empty', network_permission: 'granted' }],
            ['firmware_install_result', { stage: 'download', result: 'failure', error_type: 'FirmwareSignatureError', firmware_version: '3.0.0-rc3' }]
        ];
        for (const [name, params] of cases) {
            assert.equal(api.trackEvent(name, params), false);
        }
        api.enableAnalytics({ sendPageView: false });
        for (const [name, params] of cases) {
            api.trackEvent(name, { ...params, duration_ms: 125, ssid: 'private', message: 'secret', endpoint: '192.168.1.1' });
            const event = window.dataLayer.at(-1)[2];
            for (const [key, value] of Object.entries(params)) assert.equal(event[key], value);
            assert.equal(event.analytics_version, '2');
            assert.doesNotMatch(JSON.stringify(event), /private|secret|192\.168/);
        }
        const rejected = api.sanitizeAnalyticsEvent('local_discovery', {
            method: 'private device name', network_permission: 'private', duration_ms: Infinity
        });
        assert.equal(rejected.method, 'unknown');
        assert.equal(rejected.network_permission, 'unknown');
        assert.equal(rejected.duration_ms, undefined);
        for (const transport of ['direct_http', 'simulation', 'serial']) {
            assert.equal(api.sanitizeAnalyticsEvent('tool_connection', { transport }).transport, transport);
        }
        assert.equal(api.sanitizeAnalyticsEvent('tool_connection', { transport: '192.168.1.1' }).transport, 'unknown');
    });

    it('allows the documented tool contract and rejects device-controlled analytics data', () => {
        const { api, window } = analyticsContext();
        api.enableAnalytics({ sendPageView: false });

        assert.equal(api.trackEvent('device_profile', {
            tool_name: 'monitor',
            entry_point: 'monitor',
            frontend: 'native',
            chip: 'ESP32-C3',
            detector: 'lightweight',
            protocol_version: '1',
            firmware_version: '2.8.0-403-g7801312',
            device_id: '0123456789abcdef',
            mqtt_host: 'broker.local'
        }), true);
        const profile = window.dataLayer.at(-1)[2];
        assert.equal(profile.firmware_version, '2.8.0-dev');
        assert.equal(profile.chip, 'ESP32-C3');
        assert.equal(profile.protocol_version, '1');
        assert.equal(Object.hasOwn(profile, 'device_id'), false);
        assert.equal(Object.hasOwn(profile, 'mqtt_host'), false);
        assert.equal(
            api.sanitizeAnalyticsEvent('device_profile', { firmware_version: '0.0.0-main' })
                .firmware_version,
            '0.0.0-main'
        );
        assert.equal(
            api.sanitizeAnalyticsEvent(
                'device_profile', { firmware_version: '3.0.0-rc1-12-gabcdef0' }
            ).firmware_version,
            '3.0.0-dev'
        );

        api.trackEvent('configure_change', {
            action: 'update_mqtt',
            result: 'failure',
            error_type: 'password=user@example.com'
        });
        assert.equal(window.dataLayer.at(-1)[2].error_type, 'unknown');

        api.trackEvent('device_profile', {
            firmware_version: 'device-123-secret',
            frontend: 'native',
            chip: 'esp32-c3'
        });
        assert.equal(window.dataLayer.at(-1)[2].firmware_version, 'unknown');

        const before = window.dataLayer.length;
        assert.equal(api.trackEvent('unregistered_device_event', { payload: 'secret' }), false);
        assert.equal(window.dataLayer.length, before);
    });

    it('does not duplicate a page view when consent is accepted twice', () => {
        const { api, window } = analyticsContext();
        api.enableAnalytics();
        api.enableAnalytics();
        const pageViews = window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        );
        assert.equal(pageViews.length, 1);
    });

    it('reports a returning visitor before tool events and deduplicates SPA startup', () => {
        const spa = analyticsContext({ storedConsent: 'granted', hash: '#tool-monitor' });
        spa.api.initializeConsentControls();
        spa.api.trackEvent('tool_connection', { result: 'success' });
        spa.api.trackRouteView('tool-monitor');
        const events = spa.window.dataLayer.filter((entry) => entry[0] === 'event');
        assert.equal(events[0][1], 'page_view');
        assert.equal(events[0][2].page_path, '/tools/monitor/');
        assert.equal(spa.window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        ).length, 1);

        const staticPage = analyticsContext({ storedConsent: 'granted', staticPage: true });
        staticPage.api.initializeConsentControls();
        assert.equal(staticPage.window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        ).length, 1);
    });

    it('grants consent again and sends one page view after a withdrawal', () => {
        const { api, window } = analyticsContext();
        api.enableAnalytics();
        api.disableAnalytics();
        api.enableAnalytics();
        const consentUpdates = window.dataLayer.filter(
            (entry) => entry[0] === 'consent' && entry[1] === 'update'
        );
        assert.equal(consentUpdates.at(-1)[2].analytics_storage, 'granted');
        const pageViews = window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        );
        assert.equal(pageViews.length, 2);
    });
});

describe('analytics route metadata', () => {
    it('uses the shared route resolver for legacy device entries without a false home view', () => {
        for (const path of ['/', '/index.html']) {
            const { api, window } = analyticsContext({ path, hash: '#device', storedConsent: 'granted' });
            api.initializeConsentControls();
            api.trackRouteView('tool-configure');
            const views = window.dataLayer.filter((entry) => entry[0] === 'event' && entry[1] === 'page_view');
            assert.equal(views.length, 1);
            assert.equal(views[0][2].page_path, window.ESPectreRoutes.get('tool-configure').staticPath);
        }
        const staticPage = analyticsContext({ path: '/guides/setup/', hash: '#device', staticPage: true });
        assert.equal(staticPage.api.getRouteTitle(), staticPage.window.ESPectreRoutes.title('guide-setup'));
    });

    it('reports views before early events and keeps later route changes', () => {
        const { api, window } = analyticsContext({ hash: '#tool-monitor' });
        api.enableAnalytics({ sendPageView: false });
        api.trackEvent('tool_connection', { result: 'success' });
        api.trackRouteView('tool-monitor');
        api.trackRouteView('guide-detection');
        api.trackRouteView('tool-monitor');
        const events = window.dataLayer.filter((entry) => entry[0] === 'event');
        assert.equal(events[0][1], 'page_view');
        assert.deepEqual(Array.from(events.filter((entry) => entry[1] === 'page_view'),
            (entry) => entry[2].page_path),
            ['/tools/monitor/', '/guides/detection/', '/tools/monitor/']);
        for (const entry of window.dataLayer.filter((entry) => entry[0] === 'config')) {
            assert.equal(entry[2].send_page_view, false);
        }
    });

    it('waits for the directly fetched manifest before using route metadata', async () => {
        const { api, listeners, window } = analyticsContext({
            path: '/tools/monitor/', preloadManifest: false, storedConsent: 'granted'
        });
        await listeners.get('DOMContentLoaded')();
        api.trackRouteView('tool-monitor');
        const pageViews = window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        );
        assert.equal(pageViews.length, 1);
        assert.equal(pageViews[0][2].page_path, '/tools/monitor/');
        assert.equal(pageViews[0][2].content_group, 'monitor');
    });

    it('uses stable content groups and route paths', () => {
        const { api, window } = analyticsContext({ hash: '#tool-configure' });
        assert.equal(api.getSiteSection('tool-configure'), 'configure');
        assert.equal(api.getSiteSection('tool-monitor'), 'monitor');
        assert.equal(api.getSiteSection('guide-setup'), 'documentation');
        assert.equal(api.getSiteSection('sdk-api'), 'documentation');
        assert.equal(api.getSiteSection('sdk-http-api'), 'documentation');
        assert.equal(api.getSiteSection('sdk-mqtt-api'), 'documentation');
        assert.equal(api.getSiteSection('sdk-detectors'), 'documentation');
        assert.equal(window.ESPectreRoutes.guideNameForPath('/guides/detectors/'), 'detectors');
        assert.equal(window.ESPectreRoutes.guideNameForPath('/guides/future-wifi-sensing/'), 'future-wifi-sensing');
        assert.equal(window.ESPectreRoutes.guideNameForPath('/sdk/api/'), '');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/sdk/api/'), 'api');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/sdk/http-api/'), 'http-api');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/sdk/mqtt-api/'), 'mqtt-api');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/sdk/detectors/'), 'detectors');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/sdk/'), 'overview');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/artifacts/sdk/release/'), 'sdk_release');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/artifacts/sdk/preview/'), 'sdk_preview');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/artifacts/sdk/develop/'), 'sdk_develop');
        assert.equal(window.ESPectreRoutes.documentNameForPath('/guides/detection/'), '');
        assert.equal(api.getSiteSection('privacy'), 'privacy');
        assert.equal(api.routePath('home'), '/');
        assert.equal(api.routePath('tool-flash'), '/tools/flash/');
    });

    it('reports canonical static paths without query parameters', () => {
        const { api, window } = analyticsContext({
            path: '/guides/setup/', staticPage: true
        });
        api.enableAnalytics({ sendPageView: false });
        api.sendStaticPageView();
        const event = window.dataLayer.at(-1);
        assert.equal(event[1], 'page_view');
        assert.equal(event[2].page_path, '/guides/setup/');
        assert.equal(event[2].page_location, 'https://espectre.dev/guides/setup/');
    });

    it('records 404 suggestions only with consent and keeps the requested page location', () => {
        for (const storedConsent of [null, 'denied', 'granted']) {
            const { api, window } = analyticsContext({
                path: '/documentation/setup/', staticPage: true, storedConsent
            });
            api.initializeConsentControls();
            api.trackEvent('select_404_suggestion', { source_url: 'private', destination: 'private' });
            const events = window.dataLayer?.filter((entry) => entry[1] === 'select_404_suggestion') || [];
            assert.equal(events.length, storedConsent === 'granted' ? 1 : 0);
            if (storedConsent === 'granted') {
                assert.deepEqual(Object.keys(events[0][2]), ['send_to', 'content_group', 'analytics_version']);
                const pageView = window.dataLayer.find((entry) => entry[1] === 'page_view');
                assert.equal(pageView[2].page_location, 'https://espectre.dev/documentation/setup/');
            }
        }
    });

    it('updates the Google tag configuration before a virtual page view', () => {
        const { api, window } = analyticsContext({ hash: '#tool-monitor' });
        api.enableAnalytics({ sendPageView: false });
        api.sendRoutePageView('tool-monitor');
        const update = window.dataLayer.at(-2);
        const pageView = window.dataLayer.at(-1);
        assert.equal(update[0], 'config');
        assert.equal(update[2].update, true);
        assert.equal(update[2].page_location, 'https://espectre.dev/tools/monitor/');
        assert.equal(update[2].page_title, api.getRouteTitle('tool-monitor'));
        assert.equal(update[2].content_group, 'monitor');
        assert.equal(pageView[1], 'page_view');
    });

    it('reports a tool capability after late consent without another page view', () => {
        const { api, window } = analyticsContext({
            hash: '#tool-flash', navigatorValues: { serial: {} }
        });
        api.enableAnalytics();
        const before = window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        ).length;
        api.trackRouteView('tool-flash', { sendPageView: false });
        const capability = window.dataLayer.at(-1);
        assert.equal(capability[1], 'tool_capability');
        assert.equal(capability[2].tool_name, 'flash');
        assert.equal(capability[2].result, 'available');
        const after = window.dataLayer.filter(
            (entry) => entry[0] === 'event' && entry[1] === 'page_view'
        ).length;
        assert.equal(after, before);
    });

    it('reports the CSI visualizer SPA route through its canonical page path', () => {
        const { api, window } = analyticsContext({ path: '/tools/csi-visualizer/' });
        api.enableAnalytics({ sendPageView: false });
        api.sendRoutePageView('tool-raw-csi');
        const pageView = window.dataLayer.at(-1);
        assert.equal(pageView[1], 'page_view');
        assert.equal(pageView[2].page_path, '/tools/csi-visualizer/');
        assert.equal(pageView[2].content_group, 'raw-csi');
    });
});

describe('analytics automatic events', () => {
    it('records the commercial contact action only with consent and without email data', async () => {
        const fragment = readFileSync(new URL('../../docs/web/content/licensing.html', import.meta.url), 'utf8');
        const anchor = fragment.match(/<a\b[^>]*\bdata-analytics-contact="commercial"[^>]*>/)?.[0];
        assert.ok(anchor, 'The commercial contact action must expose its analytics intent');
        const href = anchor.match(/\bhref="([^"]+)"/)?.[1];
        assert.ok(href?.startsWith('mailto:'));
        const { api, listeners, window } = analyticsContext({ path: '/licensing/' });
        await listeners.get('DOMContentLoaded')();
        const link = {
            href,
            dataset: { analyticsContact: 'commercial' },
            getAttribute: () => href
        };
        const click = () => listeners.get('click')({
            target: { closest: (selector) => selector === 'a[href]' ? link : null }
        });

        click();
        assert.equal(window.dataLayer, undefined);
        api.enableAnalytics();
        const before = window.dataLayer.length;
        click();
        assert.equal(window.dataLayer.length, before + 1);
        assert.equal(window.dataLayer.at(-1)[1], 'click_commercial_contact');
        assert.deepEqual(Object.keys(window.dataLayer.at(-1)[2]).sort(), ['analytics_version', 'content_group', 'send_to']);

        api.disableAnalytics();
        const afterWithdrawal = window.dataLayer.length;
        click();
        assert.equal(window.dataLayer.length, afterWithdrawal);
    });

    it('tracks contact and public tool links added after analytics initialization', async () => {
        const { api, consentBanner, listeners, window } = analyticsContext();
        api.enableAnalytics({ sendPageView: false });
        await listeners.get('DOMContentLoaded')();

        const click = listeners.get('click');
        const dispatchLink = (href, { cookieSettings = false, dataset = {} } = {}) => {
            const link = {
                href,
                dataset,
                getAttribute: () => href,
                querySelector: () => null,
                textContent: href
            };
            let prevented = false;
            click({
                target: {
                    closest: (selector) => {
                        if (selector === '.js-cookie-settings') return cookieSettings ? link : null;
                        if (selector === 'a[href]') return link;
                        return null;
                    }
                },
                preventDefault: () => { prevented = true; }
            });
            return prevented;
        };

        dispatchLink('mailto:contact@espectre.dev?subject=Product');
        assert.equal(window.dataLayer.at(-1)[1], 'click_contact');
        dispatchLink('mailto:security@espectre.dev');
        assert.equal(window.dataLayer.at(-1)[1], 'click_security');
        dispatchLink('https://espectre.dev/artifacts/sdk/release/sdk.zip', {
            dataset: { sdkChannel: 'release', sdkFormat: 'zip' }
        });
        assert.equal(window.dataLayer.at(-1)[1], 'sdk_download');
        assert.equal(window.dataLayer.at(-1)[2].channel, 'release');
        assert.equal(window.dataLayer.at(-1)[2].format, 'zip');
        dispatchLink('https://espectre.dev/tools/device-settings/');
        assert.equal(window.dataLayer.at(-1)[1], 'select_tool');
        assert.equal(window.dataLayer.at(-1)[2].tool_name, 'configure');
        dispatchLink('https://espectre.dev/#guides');
        assert.equal(window.dataLayer.at(-1)[1], 'select_guide');
        assert.equal(window.dataLayer.at(-1)[2].guide_name, 'overview');
        dispatchLink('https://espectre.dev/#sdk-api');
        assert.equal(window.dataLayer.at(-1)[1], 'select_documentation');
        assert.equal(window.dataLayer.at(-1)[2].document_name, 'api');
        consentBanner.hidden = true;
        assert.equal(dispatchLink('/privacy/#cookie-settings', { cookieSettings: true }), true);
        assert.equal(consentBanner.hidden, false);
    });
});


describe('campaign consent and conversion boundaries', () => {
    const events = (window, name) => (window.dataLayer || []).filter((row) => row[0] === 'event' && row[1] === name);
    const preferences = (analytics, campaigns) => ({ version: 2, analytics, campaigns, updatedAt: Date.now() });

    it('offers independent unchecked preferences and does not load tags before a choice', () => {
        const { api, controls, appendedScripts, window } = analyticsContext();
        api.initializeConsentControls();
        controls['.js-consent-customize'].click();
        assert.equal(controls['.js-consent-preferences'].hidden, false);
        assert.equal(controls['.js-consent-customize'].attributes['aria-expanded'], 'true');
        assert.equal(controls['.js-consent-analytics'].checked, false);
        assert.equal(controls['.js-consent-campaigns'].checked, false);
        assert.equal(controls['.js-consent-analytics'].focused, true);
        assert.equal(appendedScripts.length, 0);
        assert.equal(window.dataLayer, undefined);
    });

    it('accepts both purposes with one tag and keeps all personalization disabled', () => {
        const { api, controls, window, storage, appendedScripts } = analyticsContext();
        api.initializeConsentControls();
        controls['.js-consent-accept'].click();
        const choice = JSON.parse(storage.get(api.consentKey));
        assert.equal(choice.analytics, true);
        assert.equal(choice.campaigns, true);
        assert.ok(choice.updatedAt <= Date.now());
        assert.equal(appendedScripts.length, 1);
        const configurations = window.dataLayer.filter((row) => row[0] === 'config');
        assert.ok(configurations.some((row) => row[1] === api.gaId));
        assert.ok(configurations.some((row) => row[1] === api.adsId));
        for (const row of configurations) assert.equal(row[2].allow_ad_personalization_signals, false);
        for (const row of window.dataLayer.filter((row) => row[0] === 'consent')) {
            assert.equal(row[2].ad_personalization, 'denied');
        }
        assert.equal(events(window, 'conversion').length, 0);
        controls['.js-consent-accept'].click();
        assert.equal(appendedScripts.length, 1);
        assert.equal(events(window, 'page_view').length, 1);
    });

    it('rejects without loading Google and discards click identifiers without storing them', () => {
        const { api, controls, window, storage, appendedScripts } = analyticsContext({ search: '?gclid=Click-123&utm_source=google' });
        api.initializeConsentControls();
        controls['.js-consent-reject'].click();
        assert.equal(window.dataLayer, undefined);
        assert.equal(appendedScripts.length, 0);
        assert.equal(window.espectreCampaignSearch(), '');
        assert.equal(window.location.search, '?utm_source=google');
        assert.doesNotMatch(JSON.stringify([...storage]), /Click-123/);
        assert.equal(api.trackCampaignConversion('sdk_download'), false);
    });

    it('measures campaigns without enabling Analytics or device-event collection', () => {
        const { api, controls, window, appendedScripts } = analyticsContext();
        api.initializeConsentControls();
        controls['.js-consent-campaigns'].checked = true;
        controls['.js-consent-save'].click();
        assert.equal(api.enabled(), false);
        assert.equal(api.trackEvent('device_profile', { chip: 'esp32c6' }), false);
        assert.equal(events(window, 'page_view').length, 0);
        assert.equal(window.dataLayer.some((row) => row[0] === 'config' && row[1] === api.gaId), false);
        assert.equal(appendedScripts.length, 1);
        assert.ok(appendedScripts[0].src.endsWith(api.adsId));
        for (const [name, destination] of Object.entries(api.conversions)) {
            assert.equal(api.trackCampaignConversion(name), true);
            const data = events(window, 'conversion').at(-1)[2];
            assert.equal(data.send_to, destination);
            assert.equal(data.value, undefined);
            assert.equal(data.currency, undefined);
            assert.equal(data.allow_ad_personalization_signals, false);
        }
        assert.equal(api.trackCampaignConversion('device_profile'), false);
    });

    it('routes Analytics events exclusively to GA4 and never enables Ads for analytics-only consent', () => {
        const { api, window } = analyticsContext({ preferences: preferences(true, false) });
        api.initializeConsentControls();
        api.trackEvent('sdk_download', { channel: 'release', format: 'zip' });
        api.trackRouteView('sdk');
        for (const row of (window.dataLayer || []).filter((row) => row[0] === 'event')) assert.equal(row[2].send_to, api.gaId);
        assert.equal(api.trackCampaignConversion('sdk_download'), false);
        assert.equal(window.dataLayer.some((row) => row[0] === 'config' && row[1] === api.adsId), false);
    });

    it('retains only valid click identifiers through delayed SPA consent', () => {
        const { api, window, storage } = analyticsContext({
            search: '?gclid=Click-123&wbraid=BraId_456&gbraid=bad%40email.test&target=192.168.1.5&secret=private'
        });
        assert.equal(window.espectreCampaignSearch(), 'gclid=Click-123&wbraid=BraId_456');
        window.location.pathname = '/licensing/'; window.location.search = `?${window.espectreCampaignSearch()}`;
        assert.equal(storage.size, 0);
        assert.equal(window.dataLayer, undefined);
        api.commitConsent({ analytics: false, campaigns: true });
        api.trackCampaignConversion('click_commercial_contact');
        assert.equal(events(window, 'conversion')[0][2].page_location, 'https://espectre.dev/licensing/?gclid=Click-123&wbraid=BraId_456');
        assert.doesNotMatch(JSON.stringify(window.dataLayer), /bad@|192\.168|secret|private/);
        assert.doesNotMatch(JSON.stringify([...storage]), /Click-123|BraId_456/);
    });

    it('revokes each purpose independently and clears its cookies and click IDs', () => {
        const { api, window, document } = analyticsContext({ search: '?gclid=Click-123' });
        api.commitConsent({ analytics: true, campaigns: true });
        document.cookie = '_gcl_aw=click';
        api.commitConsent({ analytics: true, campaigns: false });
        assert.equal(api.trackCampaignConversion('sdk_download'), false);
        assert.equal(api.trackEvent('sdk_download'), true);
        assert.match(document.cookie, /_gcl_aw=; Max-Age=0/);
        assert.equal(window.espectreCampaignSearch(), '');
        document.cookie = '_ga_ABC=analytics';
        api.commitConsent({ analytics: false, campaigns: false });
        assert.equal(api.trackEvent('sdk_download'), false);
        assert.equal(window[`ga-disable-${api.gaId}`], true);
        assert.match(document.cookie, /_ga_ABC=; Max-Age=0/);
        const last = window.dataLayer.filter((row) => row[0] === 'consent').at(-1)[2];
        for (const value of Object.values(last)) assert.equal(value, 'denied');
    });

    it('does not upgrade legacy Analytics consent to campaign consent', () => {
        const { api, window, consentBanner, controls } = analyticsContext({ storedConsent: 'granted' });
        api.initializeConsentControls();
        assert.equal(api.enabled(), true);
        assert.equal(consentBanner.hidden, false);
        assert.equal(controls['.js-consent-analytics'].checked, true);
        assert.equal(controls['.js-consent-campaigns'].checked, false);
        assert.equal(api.trackCampaignConversion('sdk_download'), false);
        assert.equal(window.dataLayer.some((row) => row[0] === 'config' && row[1] === api.adsId), false);
    });

    it('restores valid choices but fails closed on expired or malformed preferences', () => {
        for (const value of [preferences(false, true), preferences(true, false), preferences(false, false)]) {
            const { api, consentBanner } = analyticsContext({ preferences: value });
            api.initializeConsentControls();
            assert.equal(consentBanner.hidden, true);
            assert.equal(api.enabled(), value.analytics);
            assert.equal(api.trackCampaignConversion('sdk_download'), value.campaigns);
        }
        for (const value of [{}, { ...preferences(true, true), updatedAt: 1 }, { ...preferences(true, true), analytics: 'true' }]) {
            const { api, consentBanner, appendedScripts } = analyticsContext({ preferences: value });
            api.initializeConsentControls();
            assert.equal(consentBanner.hidden, false);
            assert.equal(appendedScripts.length, 0);
        }
    });

    it('keeps consent usable when browser storage is blocked', () => {
        const { api, controls, consentBanner } = analyticsContext({ storageBlocked: true });
        api.initializeConsentControls();
        assert.equal(consentBanner.hidden, false);
        controls['.js-consent-accept'].click();
        assert.equal(api.enabled(), true);
        controls['.js-consent-reject'].click();
        assert.equal(api.enabled(), false);
    });

    it('never sends Google Ads conversions from a debug host', () => {
        for (const hostname of ['test.espectre.dev', 'localhost', '127.0.0.1']) {
            const { api, window, appendedScripts } = analyticsContext({ hostname });
            api.commitConsent({ analytics: false, campaigns: true });
            assert.equal(appendedScripts.length, 0);
            assert.equal(api.trackCampaignConversion('sdk_download'), false);
            api.commitConsent({ analytics: true, campaigns: true });
            assert.equal(window.dataLayer.some((row) => row[0] === 'config' && row[1] === api.adsId), false);
            assert.equal(events(window, 'conversion').length, 0);
        }
    });
});
