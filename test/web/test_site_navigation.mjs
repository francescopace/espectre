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
import { app, browserSupportSource, directProtocol, GPL_HTML_HEADER, index, read, roadmapContent, routeRegistry, security, styles, toolContent, toolFragments, toolsContent } from './fixtures/site_test_helpers.mjs';

describe('website navigation contracts', () => {
    it('keeps one route registry aligned with the SPA pages and static paths', () => {
        assert.match(app, /document\.addEventListener\('DOMContentLoaded', init\);/);
        const registeredRoutes = [...routeRegistry.matchAll(/\{ name: '([^']+)'/g)]
            .map((match) => match[1])
            .sort();
        const pageRoutes = [...index.matchAll(/<main\b[^>]*\bdata-page="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredRoutes, pageRoutes);

        const registeredStaticPaths = [...routeRegistry.matchAll(/staticPath: '([^']+)'/g)]
            .map((match) => match[1])
            .sort();
        const pageStaticPaths = [...index.matchAll(/\bdata-static-url="([^"]+)"/g)]
            .map((match) => match[1])
            .sort();
        assert.deepEqual(registeredStaticPaths, pageStaticPaths);
        assert.match(index, /data-page="tools"[\s\S]*data-content-url="content\/tools\.html\?v=[0-9a-f]{12}" data-static-url="\/tools\/"/);
        assert.doesNotMatch(index.match(/data-page="tools"[\s\S]*?<\/main>/)?.[0] || '', /class="tools-grid"/);
        assert.match(toolsContent, /class="tools-grid"/);
        for (const tool of ['flash', 'configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.match(toolsContent, new RegExp(`href="/tools/${tool}/"`));
            assert.match(
                index,
                new RegExp(`data-page="tool-${tool}"[\\s\\S]*?data-content-url="content/tools/${tool}\\.html\\?v=[0-9a-f]{12}" data-static-url="/tools/${tool}/"`)
            );
            assert.match(toolContent[tool], /class="tool-static-entry"/);
            assert.match(toolContent[tool], /class="tool-interactive"/);
        }
        assert.equal((toolsContent.match(/href="\/roadmap\/#roadmap-research-title"/g) || []).length, 2);
        assert.match(roadmapContent, /id="roadmap-research-title"/);
        assert.match(app, /prepareRouteContent\(routeAtStart\)[\s\S]*?renderConnection\(\)/);
        assert.match(app, /const staticContentLoads = new Map\(\)/);
        assert.match(app, /if \(staticContentLoads\.has\(route\)\) return staticContentLoads\.get\(route\)/);
        assert.match(app, /load\.finally\(\(\) => staticContentLoads\.delete\(route\)\)/);
        assert.match(app, /const toolInitializers = Object\.freeze\([\s\S]*?'tool-game': gameInit/);
        assert.match(read('.github/scripts/build_static_pages.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /route-registry\.js\?v=\{route_registry_version\}" defer>/);
    });

    it('uses canonical paths for static pages and hashes for SPA-only navigation', () => {
        const staticRouteNames = [...routeRegistry.matchAll(/\{ name: '([^']+)'[^\n]+staticPath:/g)]
            .map((match) => match[1]);
        for (const routeName of staticRouteNames) {
            assert.doesNotMatch(index, new RegExp(`href="(?:/)?#${routeName}"`));
        }
        assert.match(index, /href="\/guides\/" class="nav-link" data-route-link="guides"/);
        assert.match(index, /href="\/sdk\/" class="nav-link" data-route-link="sdk"/);
        assert.match(index, /href="\/tools\/" class="nav-link" data-route-link="tools"/);
        for (const source of [
            index,
            read('docs/web/404.html'),
            read('.github/scripts/build_static_pages.py'),
            read('.github/scripts/stage_web_sdk.py'),
        ]) {
            assert.ok(source.indexOf('href="/sdk/" class="nav-link') < source.indexOf('href="/roadmap/" class="nav-link'));
        }
        assert.match(app, /const staticTarget = routeRegistry\.staticTargetForHref\(href, location\.href\);[\s\S]*?location\.hash = routeHash;/);
        assert.match(app, /previousRoute !== 'sdk-api' \|\| nextRoute === 'sdk-api'[\s\S]*?searchParams\.delete\('api'\)[\s\S]*?searchParams\.delete\('member'\)[\s\S]*?history\.replaceState/);
    });

    it('resolves canonical page anchors before entering the SPA', () => {
        const window = {};
        runInNewContext(routeRegistry, { Map, Object, Set, URL, window });
        const routes = window.ESPectreRoutes;
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
        assert.match(app, /pendingRouteAnchor = staticTarget\.anchor/);
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
        runInNewContext(navigation, { document, URL, window });

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
        assert.deepEqual(assignments, [
            '/#contact',
            '/?anchor=roadmap-research-title#roadmap'
        ]);
        assert.match(app, /consumeRouteAnchorHandoff\(\);/);
        assert.match(app, /prepareRouteContent\(routeAtStart\)[\s\S]*?consumeDirectHandoff\(\);/);
    });

});
