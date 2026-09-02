/*
 * ESPectre - Website navigation contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { app, browserSupportSource, directProtocol, GPL_HTML_HEADER, index, read, roadmapContent, routeBootstrap, routeManifest, routeRegistry, security, styles, toolContent, toolFragments, toolsContent } from './fixtures/site_test_helpers.mjs';

describe('website navigation contracts', () => {
    it('keeps one route registry aligned with the SPA pages and static paths', () => {
        assert.match(app, /routeRegistry = await window\.ESPectreRoutesReady/);
        const registeredRoutes = routeManifest.routes.map((route) => route.name).sort();
        const pageRoutes = [...index.matchAll(/<main\b[^>]*\bdata-page="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredRoutes, pageRoutes);

        const registeredStaticPaths = routeManifest.routes
            .filter((route) => route.staticPath !== '/')
            .map((route) => route.staticPath)
            .sort();
        assert.equal(registeredStaticPaths.length, 29);
        assert.match(app, /const contentPath = routeRegistry\.contentPath\(route\)/);
        assert.match(index, /data-page="tools"[\s\S]*?<div class="js-static-content">/);
        assert.doesNotMatch(index.match(/data-page="tools"[\s\S]*?<\/main>/)?.[0] || '', /class="tools-grid"/);
        assert.match(toolsContent, /class="tools-grid"/);
        for (const [tool, path] of [
            ['flash', 'flash'],
            ['configure', 'device-settings'],
            ['monitor', 'monitor'],
            ['raw-csi', 'csi-visualizer'],
            ['theremin', 'theremin'],
            ['game', 'game'],
        ]) {
            assert.match(toolsContent, new RegExp(`href="/tools/${path}/"`));
            assert.match(
                index,
                new RegExp(`data-page="tool-${tool}"[\\s\\S]*?<div class="js-static-content">`)
            );
            assert.match(toolContent[tool], /class="tool-static-entry"/);
            assert.match(toolContent[tool], /class="tool-interactive"/);
        }
        assert.equal((toolsContent.match(/href="\/roadmap\/#roadmap-research-title"/g) || []).length, 2);
        assert.match(roadmapContent, /id="roadmap-research-title"/);
        assert.match(app, /prepareRouteContent\(routeAtStart\)[\s\S]*?renderConnection\(\)/);
        assert.match(app, /const staticContentLoads = new Map\(\)/);
        assert.match(app, /const contentUrl = `\/\$\{contentPath\}`/);
        assert.match(app, /fetch\(contentUrl, \{ cache: 'no-cache' \}\)/);
        assert.match(app, /if \(staticContentLoads\.has\(route\)\) return staticContentLoads\.get\(route\)/);
        assert.match(app, /load\.finally\(\(\) => staticContentLoads\.delete\(route\)\)/);
        assert.match(app, /const toolInitializers = Object\.freeze\([\s\S]*?'tool-game': 'gameInit'/);
        assert.match(app, /await loadToolScript\(route\);[\s\S]*?window\[initializerName\]/);
        assert.match(read('.github/scripts/build_static_pages.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
    });

    it('keeps static page browser titles aligned with the route registry', () => {
        const window = { ESPectreRouteManifest: routeManifest };
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const registeredStaticTitles = new Map(
            window.ESPectreRoutes.all
                .map((name) => window.ESPectreRoutes.get(name))
                .filter((definition) => definition.staticPath !== '/')
                .map((definition) => [definition.staticPath, definition.title])
        );

        assert.equal(index.match(/<title>([^<]*)<\/title>/)?.[1], window.ESPectreRoutes.title('home'));
        assert.equal(
            index.match(/property="og:title" content="([^"]*)"/)?.[1],
            window.ESPectreRoutes.title('home')
        );
        assert.equal(
            index.match(/name="twitter:title" content="([^"]*)"/)?.[1],
            window.ESPectreRoutes.title('home')
        );
        assert.equal(registeredStaticTitles.size, 29);
        assert.match(staticPageBuilder, /ROUTE_MANIFEST = load_manifest\(\)/);
        assert.match(staticPageBuilder, /"title": route\["title"\]/);
    });

    it('uses canonical paths for static pages and SPA navigation', () => {
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const staticRouteNames = routeManifest.routes
            .filter((route) => route.staticPath !== '/')
            .map((route) => route.name);
        const mainNavigationPaths = routeManifest.navigation.main
            .map((name) => routeManifest.routes.find((route) => route.name === name)?.staticPath);
        const relativePaths = (html) => [...html.matchAll(/<a href="(\/(?:[^"]*\/)?)"/g)]
            .map((match) => match[1]);
        for (const source of [index, read('docs/web/404.html')]) {
            const mainNavigation = source.match(/<nav class="main-nav"[^>]*>([\s\S]*?)<\/nav>/)?.[1] || '';
            const exploreLinks = source.match(/<div class="home-resource-links">([\s\S]*?)<\/div>/)?.[1] || '';
            assert.deepEqual(relativePaths(mainNavigation), mainNavigationPaths);
            assert.deepEqual(relativePaths(exploreLinks), mainNavigationPaths.slice(1));
        }
        for (const routeName of staticRouteNames) {
            assert.doesNotMatch(index, new RegExp(`href="(?:/)?#${routeName}"`));
        }
        assert.doesNotMatch(index, /href="#home"/);
        assert.match(index, /href="\/" class="brand" data-route-link="home"/);
        assert.match(index, /href="\/" class="nav-link" data-route-link="home">Home<\/a>/);
        assert.match(index, /href="\/guides\/" class="nav-link" data-route-link="guides"/);
        assert.match(index, /href="\/sdk\/" class="nav-link" data-route-link="sdk"/);
        assert.match(index, /href="\/tools\/" class="nav-link" data-route-link="tools"/);
        for (const source of [
            index,
            read('docs/web/404.html'),
        ]) {
            assert.ok(source.indexOf('href="/sdk/" class="nav-link') < source.indexOf('href="/roadmap/" class="nav-link'));
        }
        assert.ok(
            routeManifest.navigation.main.indexOf('sdk')
            < routeManifest.navigation.main.indexOf('roadmap')
        );
        assert.match(read('.github/scripts/web_page_shell.py'), /for name in manifest\["navigation"\]\["main"\]/);
        assert.match(app, /const staticTarget = routeRegistry\.staticTargetForHref\(href, location\.href\);[\s\S]*?navigateToRoute\(staticTarget\.route/);
        assert.match(app, /history\[replace \? 'replaceState' : 'pushState'\][\s\S]*?routeHistoryUrl\(target/);
        assert.match(app, /window\.addEventListener\('popstate', onPopState\)/);
        assert.match(app, /function syncRouteMetadata\(routeName\)[\s\S]*?ogUrl\.content = canonical[\s\S]*?ogTitle\.content = title[\s\S]*?ogDescription\.content = description[\s\S]*?twitterTitle\.content = title[\s\S]*?twitterDescription\.content = description[\s\S]*?metaDescription\.content = description/);
        assert.match(staticPageBuilder, /<link rel="canonical" href="\{canonical\}">[\s\S]*?<meta property="og:url" content="\{canonical\}">[\s\S]*?<meta property="og:title" content="\{title\}">/);
        assert.match(staticPageBuilder, /<meta name="twitter:title" content="\{title\}">/);
        assert.doesNotMatch(app, /location\.hash = '#' \+ targetRoute/);
        assert.match(app, /previousRoute !== 'sdk-api' \|\| nextRoute === 'sdk-api'[\s\S]*?searchParams\.delete\('api'\)[\s\S]*?searchParams\.delete\('member'\)[\s\S]*?history\.replaceState/);
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
        assert.match(
            app,
            /\$\$\('\.js-page'\)\.forEach[\s\S]*?document\.documentElement\.removeAttribute\('data-spa-booting'\)/
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
        assert.match(app, /navigateToRoute\(staticTarget\.route, \{ anchor: staticTarget\.anchor \}\)/);
        assert.match(app, /focusRouteAnchor\(routeAtStart, anchorAtStart\)/);
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
        assert.match(app, /consumeRouteAnchorHandoff\(\);/);
        assert.match(app, /prepareRouteContent\(routeAtStart\)[\s\S]*?consumeDirectHandoff\(\);/);
    });

    it('returns a refreshed SPA route to the app shell without redirecting direct visits', () => {
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');

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
        assert.match(app, /pushState'[\s\S]*?\{ espectreRoute: target \}/);
        assert.match(staticPageBuilder, /data-spa-route="\{name\}"/);
        assert.match(staticPageBuilder, /route-bootstrap\.js\?v=\{route_bootstrap_version\}"><\/script>/);
        assert.doesNotMatch(staticPageBuilder, /window\.location\.replace/);
    });

});
