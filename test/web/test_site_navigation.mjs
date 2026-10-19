/*
 * ESPectre - Website navigation contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { index, read, roadmapContent, routeBootstrap, routeManifest, routeRegistry, styles } from './fixtures/site_test_helpers.mjs';

describe('website navigation contracts', () => {
    it('redirects legacy doc entries to their mapped destinations on the current host', () => {
        const entries = [...routeManifest.legacyRedirects.filter((entry) => entry.doc),
            ...routeManifest.legacyNotFound.map((entry) => ({ ...entry, href: entry.path }))];
        for (const entry of entries) {
            for (const origin of ['https://espectre.dev', 'https://test.espectre.dev']) {
                for (const path of ['/', '/index.html']) {
                    for (const doc of [entry.doc, entry.doc.toLowerCase()]) {
                        const destinations = [];
                        const location = new URL(`${origin}${path}?doc=${doc}&private=value#old-heading`);
                        location.replace = (href) => destinations.push(href);
                        const window = { location, ESPectreRouteManifest: routeManifest };
                        runInNewContext(routeRegistry, { URL, window });
                        assert.equal(window.ESPectreRoutes.redirecting, true);
                        assert.deepEqual(destinations, [new URL(entry.href, origin).href]);
                        assert.equal(new URL(destinations[0]).origin, origin);
                    }
                }
            }
        }
    });

    it('rejects removed documents that collide with current routes or doc identifiers', () => {
        const retired = routeManifest.legacyNotFound[0];
        const current = routeManifest.routes.find((route) => route.staticPath !== '/');
        const legacyDoc = routeManifest.legacyRedirects.find((entry) => entry.doc).doc;
        for (const entry of [{ ...retired, path: current.staticPath }, { ...retired, doc: legacyDoc }]) {
            const destinations = [];
            const location = new URL(`/?doc=${retired.doc}`, routeManifest.siteOrigin);
            location.replace = (href) => destinations.push(href);
            const window = { location, ESPectreRouteManifest: { ...routeManifest, legacyNotFound: [entry] } };
            assert.throws(() => runInNewContext(routeRegistry, { URL, window }), /legacy/);
            assert.deepEqual(destinations, []);
        }
    });

    it('leaves unrelated routes and ambiguous or unknown doc parameters alone', () => {
        const doc = routeManifest.legacyRedirects.find((entry) => entry.doc).doc;
        for (const path of ['/', '/index.html', '/?doc=', '/?doc=constructor',
            '/?doc=https://example.com/', `/?doc=${doc}&doc=${doc}`, `/guides/setup/?doc=${doc}`]) {
            const destinations = [];
            const location = new URL(path, routeManifest.siteOrigin);
            location.replace = (href) => destinations.push(href);
            const window = { location, ESPectreRouteManifest: routeManifest };
            runInNewContext(routeRegistry, { URL, window });
            assert.equal(window.ESPectreRoutes.redirecting, false);
            assert.deepEqual(destinations, []);
        }
    });

    it('loads the current SDK reference, selects symbols, and rejects invalid manifests', async () => {
        const source = read('docs/web/assets/js/navigation.js');
        const functions = source.slice(source.indexOf('    const apiReferenceManifestPromises'),
            source.indexOf('    function apiReferenceEntryLabel'))
            + source.slice(source.indexOf('    async function loadApiReference'),
                source.indexOf('    function initApiReferenceBrowsers'));
        const commit = 'a'.repeat(40);
        const version = '3.0.0-rc2+build';
        const params = new URLSearchParams({ api: 'controller', member: 'setup' });
        const index = '/artifacts/sdk/api/api-index.json';
        const base = '/artifacts/sdk/api/';
        const manifest = { sdk_version: version, source_commit: commit, default: 'index', entries: [{ refid: 'index' }] };
        for (const scenario of ['selected', 'latest', 'missing', 'invalid-commit', 'incomplete', 'invalid-version']) {
            const requests = [];
            const shown = [];
            const warnings = [];
            const attributes = new Map();
            const content = {
                replaceChildren() {}, setAttribute: (name, value) => attributes.set(name, value),
                removeAttribute: (name) => attributes.delete(name),
            };
            const label = {};
            const filter = {};
            const browser = {
                dataset: { apiIndex: index },
                querySelector: (selector) => ({
                    '[data-api-reference-version]': label,
                    '[data-api-reference-filter]': filter,
                    '[data-api-reference-content]': content,
                })[selector],
            };
            const location = new URL(`https://espectre.dev/sdk/api/?${params}`);
            if (scenario === 'latest') location.search = '';
            const context = {
                URL, URLSearchParams, browser, console: { warn: (...args) => warnings.push(args) },
                window: { location, history: { state: {}, pushState: (_state, _unused, url) => { location.href = new URL(url, location).href; } } },
                fetch: async (url) => {
                    requests.push(url);
                    return { ok: scenario !== 'missing', status: 404,
                        json: async () => ({
                            ...manifest,
                            source_commit: scenario === 'invalid-commit' ? 'short-sha' : commit,
                            sdk_version: scenario === 'invalid-version' ? 'unknown' : version,
                            entries: scenario === 'incomplete' ? [] : manifest.entries,
                        }) };
                },
                renderApiReferencePicker() {}, refreshApiReferenceToc() {}, closeApiReferencePicker() {},
                showApiReference: async (_browser, refid, member) => shown.push({ refid, member }),
            };
            await runInNewContext(functions + '\nloadApiReference(browser);', context);
            if (['selected', 'latest'].includes(scenario)) {
                assert.equal(browser.apiReferenceManifest.sdk_version, version);
                assert.equal(filter.disabled, false);
                assert.equal(browser.apiReferenceBase, base);
                assert.equal(label.title, commit);
                assert.deepEqual(requests, [index]);
                assert.deepEqual(shown, [{ refid: scenario === 'latest' ? 'index' : 'controller', member: scenario === 'latest' ? '' : 'setup' }]);
                if (scenario === 'selected') {
                    runInNewContext("updateApiReferenceLocation('listener', 'on_ready');", context);
                    assert.equal(location.searchParams.get('api'), 'listener');
                    assert.equal(location.searchParams.get('member'), 'on_ready');
                }
            } else {
                assert.equal(browser.apiReferenceManifest, null);
                assert.equal(filter.disabled, true);
                assert.equal(shown.length, 0);
                assert.equal(warnings.length, 1);
                assert.equal(attributes.has('aria-busy'), false);
                assert.deepEqual(requests, [index]);
            }
        }
    });

    it('maps legacy redirects to existing routes on the same site', () => {
        for (const entry of routeManifest.legacyRedirects) {
            const destination = new URL(entry.href, routeManifest.siteOrigin);
            assert.equal(destination.origin, routeManifest.siteOrigin);
            assert.ok(routeManifest.routes.some((route) => route.staticPath === destination.pathname));
        }
    });

    it('rejects redirect destinations outside the current site before navigating', () => {
        const entry = routeManifest.legacyRedirects.find((redirect) => redirect.doc);
        for (const href of [
            'https://github.com/francescopace/espectre/blob/develop/CONTRIBUTING.md',
            'https://example.com/', '//example.com/', '/\t/example.com/',
            '/\\example.com/', 'javascript:alert(1)'
        ]) {
            const destinations = [];
            const location = new URL(`/?doc=${entry.doc}`, routeManifest.siteOrigin);
            location.replace = (destination) => destinations.push(destination);
            const window = { location, ESPectreRouteManifest: {
                ...routeManifest, legacyRedirects: [{ ...entry, href }]
            } };
            assert.throws(() => runInNewContext(routeRegistry, { URL, window }),
                /Invalid legacy redirect destination/);
            assert.deepEqual(destinations, []);
        }
    });

    it('shares the published Release tag between home and roadmap badges and preserves the Flash links', async () => {
        const source = read('docs/web/assets/js/navigation.js');
        const start = source.indexOf('    let publishedReleaseTagPromise;');
        const end = source.indexOf('    function activateCodeTab(', start);
        const scenarios = [
            [{ release_tag: '3.1.0', version: '3.1.0-1-gabcdef0' }, 'v3.1.0'],
            [{ release_tag: 'v3.1.0-rc2', version: '3.1.0-rc2' }, 'v3.1.0-rc2'],
            [{ version: '3.1.0' }, 'v3.1.0'],
            [{ version: 'v3.1.0' }, 'v3.1.0'],
            [{}, null],
            [null, null],
            [new Error('Network unavailable'), null],
            [new SyntaxError('Invalid JSON'), null],
        ];
        for (const [manifest, expectedTag] of scenarios) {
            const badges = [{ hidden: true }, { hidden: false }];
            const labels = badges.map((badge) => ({
                dataset: {},
                textContent: '',
                closest: (selector) => {
                    assert.equal(selector, '.release-badge');
                    return badge;
                }
            }));
            const requests = [];
            const context = {
                document: {
                    querySelectorAll: (selector) => {
                        assert.equal(selector, '[data-published-release-tag]:not([data-published-release-tag-initialized])');
                        return labels.filter((label) => !label.dataset.publishedReleaseTagInitialized);
                    }
                },
                fetch: async (url, options) => {
                    requests.push(url);
                    assert.equal(url, '/artifacts/firmware/release/firmware-manifest-release.json');
                    assert.equal(options.cache, 'no-store');
                    if (manifest instanceof Error && !(manifest instanceof SyntaxError)) throw manifest;
                    return {
                        ok: manifest !== null,
                        status: manifest === null ? 404 : 200,
                        json: async () => {
                            if (manifest instanceof SyntaxError) throw manifest;
                            return manifest;
                        }
                    };
                }
            };
            await runInNewContext(source.slice(start, end) + `
                initPublishedReleaseTags();
                publishedReleaseTag().catch(() => {});
            `, context);
            assert.equal(badges[0].hidden, expectedTag === null);
            assert.equal(badges[1].hidden, false);
            for (const label of labels) {
                if (expectedTag) {
                    assert.ok(label.textContent.startsWith(expectedTag));
                    assert.equal(label.title, label.textContent);
                }
                else assert.equal(label.textContent, '');
            }
            runInNewContext('initPublishedReleaseTags();', context);
            assert.equal(requests.length, 1);
        }
        for (const content of [index, roadmapContent]) {
            const badge = content.match(/<a\b[^>]*class="[^"]*\brelease-badge\b[^"]*"[^>]*>[\s\S]*?<\/a>/)?.[0];
            assert.match(badge, /href="\/tools\/flash\/"/);
            assert.match(badge, /<span\b[^>]*\bdata-published-release-tag(?:\s|>)/);
        }
    });

    it('keeps one route registry aligned with the SPA pages and static paths', () => {
        const registeredRoutes = routeManifest.routes.map((route) => route.name).sort();
        const pageRoutes = [...index.matchAll(/<main\b[^>]*\bdata-page="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredRoutes, pageRoutes);

        for (const route of routeManifest.routes.filter(({ group }) => group === 'tools')) {
            assert.match(index, new RegExp(`data-page="${route.name}"[\\s\\S]*?<div class="js-static-content">`));
            const content = read(`docs/web/content${route.staticPath.slice(0, -1)}.html`);
            assert.match(content, /class="tool-static-entry"/);
            assert.match(content, /class="tool-interactive"/);
        }
    });

    it('keeps static page browser titles aligned with the route registry', () => {
        const window = { ESPectreRouteManifest: routeManifest };
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        assert.equal(index.match(/<title>([^<]*)<\/title>/)?.[1], window.ESPectreRoutes.title('home'));
        assert.equal(
            index.match(/property="og:title" content="([^"]*)"/)?.[1],
            window.ESPectreRoutes.title('home')
        );
        assert.equal(
            index.match(/name="twitter:title" content="([^"]*)"/)?.[1],
            window.ESPectreRoutes.title('home')
        );
    });

    it('lists the guides in the section menu in the same order as the guides index', () => {
        const window = { ESPectreRouteManifest: routeManifest };
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        const menuPaths = [...window.ESPectreRoutes.membersOf('guides')]
            .filter((entry) => entry.name !== 'guides')
            .map((entry) => entry.staticPath);
        const indexPaths = [...read('docs/web/content/guides.html')
            .matchAll(/<a href="(\/guides\/[^"#?]+\/)" class="guide-card">/g)]
            .map((match) => match[1]);
        assert.deepEqual(menuPaths, indexPaths);
    });

    it('shows the guides menu only on the guides listed in it', () => {
        const source = read('docs/web/assets/js/navigation.js');
        const functions = source.slice(source.indexOf('    const pagePathLabels'), source.indexOf('    let pageTocUpdateFrame'))
            + source.slice(source.indexOf('    function currentPagePathRoute'), source.indexOf('    function ensurePageTocs'));
        const window = { ESPectreRouteManifest: routeManifest };
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        const element = () => ({ dataset: {}, append() {}, setAttribute() {} });
        const context = { Object, window, document: { createElement: element } };
        runInNewContext(functions, context);
        for (const route of routeManifest.routes.filter(({ group }) => group === 'guides')) {
            const appended = [];
            context.buildPagePath({
                dataset: { pagePath: 'guides' },
                querySelector: () => null,
                closest: () => ({ dataset: { page: route.name } }),
                append: (node) => appended.push(node),
            });
            assert.equal(appended.length, route.pathLabel ? 1 : 0, route.name);
        }
    });

    it('uses canonical paths for static pages and SPA navigation', () => {
        const mainNavigationPaths = routeManifest.navigation.main
            .map((name) => routeManifest.routes.find((route) => route.name === name)?.staticPath);
        const relativePaths = (html) => [...html.matchAll(/<a href="(\/(?:[^"]*\/)?)"/g)]
            .map((match) => match[1]);
        for (const source of [index, read('docs/web/404.html')]) {
            const mainNavigation = source.match(/<nav class="main-nav"[^>]*>([\s\S]*?)<\/nav>/)?.[1] || '';
            const exploreLinks = source.match(/<div class="home-resource-links">([\s\S]*?)<\/div>/)?.[1] || '';
            assert.deepEqual(relativePaths(mainNavigation), mainNavigationPaths);
            assert.deepEqual(relativePaths(exploreLinks), mainNavigationPaths.slice(1));
            for (const route of routeManifest.routes.filter(({ navLabel }) => navLabel)) {
                assert.match(mainNavigation, new RegExp(`<a href="${route.staticPath}"[^>]*>${route.navLabel}</a>`));
                assert.match(exploreLinks, new RegExp(`<a href="${route.staticPath}"><strong>${route.navLabel}</strong>`));
            }
        }
    });

    it('keeps the Home hero out of the first paint while a deep SPA route boots', () => {
        const attributes = new Map();
        const document = {
            documentElement: {
                dataset: {},
                setAttribute: (name, value) => attributes.set(name, value),
                removeAttribute: (name) => attributes.delete(name),
            },
        };
        const window = {
            location: { hash: '#guide-setup' },
        };
        window.self = window;
        window.top = window;
        runInNewContext(routeBootstrap, { document, URL, window });
        assert.equal(attributes.has('data-spa-booting'), true);
        assert.match(styles, /html\[data-spa-booting\] \.js-page\[data-page="home"\] \{ display: none; \}/);
        assert.match(index, /<script src="\/assets\/js\/route-bootstrap\.js\?v=[a-f0-9]{12}"><\/script>/);
        assert.ok(
            index.indexOf('route-bootstrap.js') < index.indexOf('espectre-direct.js'),
            'the route bootstrap must run before deferred application scripts'
        );
    });

    it('keeps the interactive portal inert when embedded by another origin', () => {
        const attributes = new Map([['data-frame-guard', '']]);
        const document = {
            documentElement: {
                dataset: {},
                setAttribute: (name, value) => attributes.set(name, value),
                removeAttribute: (name) => attributes.delete(name),
            },
        };
        const window = { location: { hash: '#tool-configure' } };
        window.self = window;
        window.top = {};
        runInNewContext(routeBootstrap, { document, URL, window });
        assert.equal(attributes.has('data-frame-guard'), true);
        assert.equal(attributes.has('data-spa-booting'), false);
        assert.match(index, /<html[^>]+data-frame-guard/);
        assert.match(index, /<style>html\[data-frame-guard\] \{ visibility: hidden; \}<\/style>/);
        assert.match(index, /<noscript><style>html\[data-frame-guard\] \{ visibility: visible; \}<\/style><\/noscript>/);
        assert.match(styles, /html\[data-frame-guard\] \{ visibility: hidden; \}/);
    });

    it('resolves canonical page anchors before entering the SPA', () => {
        const window = { ESPectreRouteManifest: routeManifest };
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        const routes = window.ESPectreRoutes;
        assert.equal(routes.siteOrigin, routeManifest.siteOrigin);
        const target = routes.staticTargetForHref(
            '/guides/setup/#setup-native-discovery',
            'https://test.espectre.dev/'
        );

        assert.equal(target.route, 'guide-setup');
        assert.equal(target.anchor, 'setup-native-discovery');
        assert.equal(
            routes.staticTargetForHref('https://example.com/guides/setup/', 'https://test.espectre.dev/'),
            null
        );
        assert.equal(
            routes.staticTargetForHref('/guides/setup/?source=external', 'https://test.espectre.dev/'),
            null
        );
    });

    it('normalizes explicit HTML entries and static-page clicks to root SPA hashes', () => {
        const navigation = read('docs/web/assets/js/navigation.js');
        const listeners = new Map();
        const replacements = [];
        const assignments = [];
        const location = {
            pathname: '/index.html',
            href: 'https://espectre.dev/index.html#contact',
            search: '',
            hash: '#contact',
            assign: (href) => assignments.push(href),
        };
        const history = {
            state: { source: 'test' },
            replaceState: (state, title, href) => replacements.push(String(href)),
        };
        const document = {
            documentElement: { hasAttribute: (name) => name === 'data-static-page' },
            addEventListener: (type, listener) => listeners.set(type, listener),
            querySelectorAll: () => [],
        };
        const window = {
            location,
            history,
            matchMedia: () => ({ matches: false, addEventListener: () => {} }),
            ESPectreRoutes: {
                staticTargetForHref: (href) => {
                    if (href === '/contact/') return { route: 'contact', anchor: '' };
                    if (href === '/roadmap/#roadmap-research-title') {
                        return { route: 'roadmap', anchor: 'roadmap-research-title' };
                    }
                    return null;
                }
            },
        };
        runInNewContext(navigation, { document, URL, URLSearchParams, window });

        assert.deepEqual(replacements, ['https://espectre.dev/#contact']);
        let prevented = false;
        listeners.get('click')({
            defaultPrevented: false,
            button: 0,
            metaKey: false,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            target: {
                closest: () => ({
                    target: '',
                    getAttribute: () => '/contact/',
                }),
            },
            preventDefault: () => { prevented = true; },
        });
        assert.equal(prevented, true);
        listeners.get('click')({
            defaultPrevented: false,
            button: 0,
            metaKey: false,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            target: {
                closest: () => ({
                    target: '',
                    getAttribute: () => '/roadmap/#roadmap-research-title',
                }),
            },
            preventDefault: () => {},
        });
        location.pathname = '/tools/device-settings/';
        location.href = 'https://espectre.dev/tools/device-settings/?target=192.168.1.42';
        location.search = '?target=192.168.1.42';
        listeners.get('click')({
            defaultPrevented: false,
            button: 0,
            metaKey: false,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            target: {
                closest: () => ({
                    target: '',
                    getAttribute: () => '/#tool-configure',
                }),
            },
            preventDefault: () => {},
        });
        assert.deepEqual(assignments, [
            '/#contact',
            '/?anchor=roadmap-research-title#roadmap',
            '/?target=192.168.1.42#tool-configure'
        ]);
    });

    it('returns a refreshed SPA route to the app shell without redirecting direct visits', () => {
        function runBootstrap({ navigationType, rememberedRoute }) {
            const replacements = [];
            const location = {
                pathname: '/tools/monitor/',
                href: 'https://espectre.dev/tools/monitor/?target=192.168.1.42#diagnostics',
                search: '?target=192.168.1.42',
                hash: '#diagnostics',
                replace: (href) => replacements.push(href),
            };
            const document = {
                documentElement: {
                    dataset: { spaRoute: 'tool-monitor' },
                    setAttribute: () => {},
                },
            };
            const window = {
                location,
                history: {
                    state: rememberedRoute ? { espectreRoute: rememberedRoute } : null,
                    replaceState: () => {},
                },
                performance: { getEntriesByType: () => [{ type: navigationType }] },
            };
            runInNewContext(routeBootstrap, { document, URL, window });
            return replacements;
        }

        const reload = runBootstrap({
            navigationType: 'reload',
            rememberedRoute: 'tool-monitor',
        });
        assert.deepEqual(reload, [
            '/?target=192.168.1.42&anchor=diagnostics#tool-monitor'
        ]);

        const directVisit = runBootstrap({
            navigationType: 'navigate',
            rememberedRoute: 'tool-monitor',
        });
        assert.deepEqual(directVisit, []);

        const standaloneReload = runBootstrap({
            navigationType: 'reload',
            rememberedRoute: '',
        });
        assert.deepEqual(standaloneReload, []);
    });

});

describe('asynchronous route rendering', () => {
    function harness({ booting = false, deferPreparation = false } = {}) {
        const source = read('docs/web/assets/js/app.js');
        const pending = [];
        const preparing = [];
        const pages = ['home', 'guides', 'sdk'].map((name) => ({
            dataset: { page: name }, hidden: name !== 'home', inert: false,
            removeAttribute(name) { delete this[name]; },
        }));
        const rootAttributes = new Set(booting ? ['data-spa-booting'] : []);
        const views = [];
        const focused = [];
        const browserRenders = [];
        let preparations = 0;
        const context = {
            route: 'guides', pendingRouteAnchor: '',
            loadStaticContent: () => new Promise((resolve) => pending.push(resolve)),
            $: () => ({}), $$: (selector) => selector === '.js-page' ? pages : [],
            document: {
                documentElement: {
                    hasAttribute: (name) => rootAttributes.has(name),
                    removeAttribute: (name) => rootAttributes.delete(name),
                    toggleAttribute: (name, force) => {
                        if (force) rootAttributes.add(name);
                        else rootAttributes.delete(name);
                    },
                },
            },
            routeRegistry: { groupOf: () => '' },
            window: { scrollTo() {}, trackRouteView: (route) => views.push(route) },
            syncRouteMetadata() {},
            prepareRouteContent: (route) => {
                preparations++;
                if (!deferPreparation) return Promise.resolve(true);
                return new Promise((resolve) => preparing.push(resolve));
            },
            renderBrowserSupport: () => browserRenders.push(context.route),
            renderDirectBrowserGuidance() {},
            renderConnection() {}, consumeDirectHandoff() {},
            focusRouteContent: (route) => focused.push(route), focusRouteAnchor: () => false,
        };
        const render = runInNewContext(source.slice(source.indexOf('    let routeRenderSequence'),
            source.indexOf('    function clearApiReferenceLocation')) + '\napplyRoute;', context);
        return {
            context, render, pending, preparing, pages, rootAttributes, views, focused, browserRenders,
            preparations: () => preparations,
        };
    }

    it('keeps the current page visible until the destination is populated', async () => {
        const h = harness();
        const rendering = h.render();
        assert.deepEqual(h.pages.map((page) => page.hidden), [false, true, true]);
        assert.deepEqual(h.pages.map((page) => page.inert), [true, false, false]);
        assert.ok(h.rootAttributes.has('data-route-pending'));
        assert.deepEqual(h.views, []);
        h.pending.shift()(true);
        await rendering;
        assert.deepEqual(h.pages.map((page) => page.hidden), [true, false, true]);
        assert.deepEqual(h.pages.map((page) => page.inert), [false, false, false]);
        assert.equal(h.rootAttributes.has('data-route-pending'), false);
        assert.equal(h.pages[1].id, 'main-content');
        assert.deepEqual(h.views, ['guides']);
        assert.deepEqual(h.focused, ['guides']);
    });

    it('ignores stale completions even when navigation returns to the same destination', async () => {
        const h = harness();
        const first = h.render();
        h.context.route = 'sdk';
        const second = h.render();
        h.context.route = 'guides';
        const latest = h.render();
        h.pending[0](true);
        h.pending[1](true);
        await Promise.all([first, second]);
        assert.deepEqual(h.pages.map((page) => page.hidden), [false, true, true]);
        assert.deepEqual(h.views, []);
        h.pending[2](true);
        await latest;
        assert.deepEqual(h.views, ['guides']);
        assert.deepEqual(h.focused, ['guides']);
    });

    it('reveals a failed destination without retrying or initializing its tool', async () => {
        const h = harness();
        const rendering = h.render();
        h.pending.shift()(false);
        await rendering;
        assert.equal(h.pages[1].hidden, false);
        assert.equal(h.preparations(), 0);
        assert.deepEqual(h.focused, ['guides']);
    });

    it('keeps the boot loading state until the destination is populated', async () => {
        const h = harness({ booting: true });
        const rendering = h.render({ focus: false });
        assert.equal(h.rootAttributes.has('data-spa-booting'), true);
        assert.equal(h.pages[1].hidden, true);
        assert.equal(h.rootAttributes.has('data-route-pending'), false);
        assert.deepEqual(h.views, []);
        h.pending.shift()(true);
        await rendering;
        assert.deepEqual(h.pages.map((page) => page.hidden), [true, false, true]);
        assert.equal(h.rootAttributes.has('data-spa-booting'), false);
        assert.equal(h.preparations(), 1);
        assert.deepEqual(h.browserRenders, ['guides']);
        assert.deepEqual(h.focused, []);
        assert.deepEqual(h.views, ['guides']);
    });

    it('retains the boot loading state until the latest destination succeeds or fails', async () => {
        const h = harness({ booting: true });
        const first = h.render({ focus: false });
        h.context.route = 'sdk';
        const latest = h.render();
        h.pending.shift()(true);
        await first;
        assert.equal(h.rootAttributes.has('data-spa-booting'), true);
        assert.equal(h.pages[1].hidden, true);
        assert.deepEqual(h.views, []);
        h.pending.shift()(false);
        await latest;
        assert.equal(h.rootAttributes.has('data-spa-booting'), false);
        assert.equal(h.pages[2].hidden, false);
        assert.equal(h.preparations(), 0);
        assert.deepEqual(h.focused, ['sdk']);
        assert.deepEqual(h.views, ['sdk']);
    });

    it('ignores a preparation that finishes after navigation returns to its destination', async () => {
        const h = harness({ deferPreparation: true });
        const first = h.render();
        h.pending.shift()(true);
        await first;
        h.context.route = 'sdk';
        const second = h.render();
        h.context.route = 'guides';
        const latest = h.render();
        h.preparing.shift()(true);
        await Promise.resolve();
        assert.deepEqual(h.browserRenders, []);
        assert.deepEqual(h.focused, []);
        h.pending.shift()(true);
        h.pending.shift()(true);
        await Promise.all([second, latest]);
        h.preparing.shift()(true);
        await Promise.resolve();
        assert.deepEqual(h.browserRenders, ['guides']);
        assert.deepEqual(h.focused, ['guides']);
        assert.deepEqual(h.views, ['guides', 'guides']);
    });
});

describe('consented campaign navigation', () => {
    it('preserves only the measurement owner\'s allowed click identifiers on SPA routes', () => {
        const source = read('docs/web/assets/js/app.js');
        const start = source.indexOf('    function routeHistoryUrl(');
        const end = source.indexOf('    function syncRouteMetadata(', start);
        const context = {
            URL, location: { href: 'https://espectre.dev/sdk/?secret=private&gclid=Click-123' },
            routeRegistry: { get: () => ({ staticPath: '/licensing/' }) },
            window: { espectreCampaignSearch: () => 'gclid=Click-123' }
        };
        const route = runInNewContext(source.slice(start, end) + '\nrouteHistoryUrl;', context);
        assert.equal(route('licensing'), '/licensing/?gclid=Click-123');
        assert.equal(route('licensing', { anchor: 'quote' }), '/licensing/?gclid=Click-123#quote');
        context.window.espectreCampaignSearch = () => '';
        assert.equal(route('licensing'), '/licensing/');
        delete context.window.espectreCampaignSearch;
        assert.equal(route('licensing'), '/licensing/');
    });
});
