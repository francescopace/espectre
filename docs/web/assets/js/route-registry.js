/*
 * ESPectre - Shared website registry
 *
 * This is the single source of truth for SPA route membership, navigation
 * groups, page titles, canonical static paths, and deployment host roles.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const productionHosts = new Set(['espectre.dev', 'www.espectre.dev']);
    const validationHosts = new Set(['test.espectre.dev']);
    const loopbackHosts = new Set(['localhost', '127.0.0.1', '[::1]']);

    function siteKind(locationLike) {
        const hostname = locationLike?.hostname || '';
        if (productionHosts.has(hostname)) return 'production';
        if (validationHosts.has(hostname)) return 'validation';
        if (loopbackHosts.has(hostname)) return 'loopback';
        return 'other';
    }

    function analyticsAllowed(locationLike) {
        return siteKind(locationLike) !== 'other';
    }

    function analyticsDebug(locationLike) {
        return ['validation', 'loopback'].includes(siteKind(locationLike));
    }

    function directOriginKind(locationLike) {
        const kind = siteKind(locationLike);
        if (locationLike?.protocol === 'https:' && ['production', 'validation'].includes(kind)) return kind;
        if (locationLike?.protocol === 'http:' && kind === 'loopback') return kind;
        return 'other';
    }

    const definitions = [
        { name: 'home', title: 'ESPectre — Wi-Fi motion sensing' },
        { name: 'tools', title: 'Browser tools | ESPectre', staticPath: '/tools/' },
        { name: 'tool-flash', group: 'tools', pathLabel: 'Install', title: 'Install ESPectre | ESPectre', staticPath: '/tools/flash/', analyticsName: 'flash' },
        { name: 'tool-configure', group: 'tools', pathLabel: 'Configure', title: 'Device settings | ESPectre', staticPath: '/tools/configure/', analyticsName: 'configure' },
        { name: 'tool-monitor', group: 'tools', pathLabel: 'Monitor', title: 'Live motion | ESPectre', staticPath: '/tools/monitor/', analyticsName: 'monitor' },
        { name: 'tool-raw-csi', group: 'tools', pathLabel: 'Raw CSI', title: 'Raw Wi-Fi signal | ESPectre', staticPath: '/tools/raw-csi/', analyticsName: 'raw-csi' },
        { name: 'tool-game', group: 'tools', pathLabel: 'Game', title: 'Run with the Spectre | ESPectre', staticPath: '/tools/game/', analyticsName: 'game' },
        { name: 'tool-theremin', group: 'tools', pathLabel: 'Theremin', title: 'Motion theremin | ESPectre', staticPath: '/tools/theremin/', analyticsName: 'theremin' },
        { name: 'guides', title: 'Guides | ESPectre', staticPath: '/guides/', analyticsName: 'overview' },
        { name: 'guide-detection', group: 'guides', pathLabel: 'Detection', title: 'How Wi-Fi sensing detects movement | ESPectre', staticPath: '/guides/detection/' },
        { name: 'guide-hardware', group: 'guides', pathLabel: 'Hardware', title: 'Choose your hardware | ESPectre', staticPath: '/guides/hardware/' },
        { name: 'guide-setup', group: 'guides', pathLabel: 'Setup', title: 'Flash and set up your device | ESPectre', staticPath: '/guides/setup/' },
        { name: 'guide-placement', group: 'guides', pathLabel: 'Placement', title: 'Place your ESPectre sensor | ESPectre', staticPath: '/guides/placement/' },
        { name: 'guide-home-assistant', group: 'guides', pathLabel: 'Home Assistant', title: 'Build your Home Assistant dashboard | ESPectre', staticPath: '/guides/home-assistant/' },
        { name: 'guide-detectors', group: 'guides', pathLabel: 'Detector profiles', title: 'Choose your detection profile | ESPectre', staticPath: '/guides/detectors/' },
        { name: 'guide-micropython', group: 'guides', pathLabel: 'MicroPython', title: 'Run ESPectre on MicroPython | ESPectre', staticPath: '/guides/micropython/' },
        { name: 'guide-future-wifi-sensing', group: 'guides', pathLabel: 'Future Wi-Fi sensing', title: 'The future of Wi-Fi sensing | ESPectre', staticPath: '/guides/future-wifi-sensing/', analyticsName: 'future-wifi-sensing' },
        { name: 'sdk', pathLabel: 'SDK', title: 'ESPectre SDK quick guide | ESPectre', staticPath: '/sdk/', analyticsName: 'overview' },
        { name: 'sdk-architecture', group: 'sdk', pathLabel: 'Architecture', title: 'Architecture | ESPectre', staticPath: '/sdk/architecture/' },
        { name: 'sdk-api', group: 'sdk', pathLabel: 'API', title: 'API reference | ESPectre', staticPath: '/sdk/api/' },
        { name: 'sdk-detectors', group: 'sdk', pathLabel: 'Detectors', title: 'Detector architecture | ESPectre', staticPath: '/sdk/detectors/' },
        { name: 'sdk-examples', group: 'sdk', pathLabel: 'Examples', title: 'Examples | ESPectre', staticPath: '/sdk/examples/' },
        { name: 'media', title: 'Media | ESPectre', staticPath: '/media/' },
        { name: 'roadmap', title: 'Roadmap | ESPectre', staticPath: '/roadmap/' },
        { name: 'privacy', title: 'Website privacy and analytics | ESPectre', staticPath: '/privacy/' },
        { name: 'terms', title: 'Terms of use | ESPectre', staticPath: '/terms/' },
        { name: 'legal', title: 'Legal information | ESPectre', staticPath: '/legal/' },
        { name: 'security', title: 'Security and responsible use | ESPectre', staticPath: '/security/' },
        { name: 'licensing', title: 'Commercial licensing | ESPectre', staticPath: '/licensing/' },
        { name: 'contact', title: 'Contact | ESPectre', staticPath: '/contact/' }
    ].map((definition) => Object.freeze({ ...definition }));

    const documentPathNames = new Map([
        ['/artifacts/sdk/release/', 'sdk_release'],
        ['/artifacts/sdk/preview/', 'sdk_preview'],
        ['/artifacts/sdk/develop/', 'sdk_develop']
    ]);

    const byName = new Map(definitions.map((definition) => [definition.name, definition]));
    const emptyGroup = Object.freeze([]);
    const byGroup = new Map();
    definitions.forEach((definition) => {
        if (!definition.group) return;
        if (!byGroup.has(definition.group)) byGroup.set(definition.group, []);
        byGroup.get(definition.group).push(definition);
    });
    byGroup.forEach((members, group) => {
        const root = byName.get(group);
        if (root?.pathLabel) members.unshift(root);
        byGroup.set(group, Object.freeze(members));
    });
    const byStaticPath = new Map(
        definitions
            .filter((definition) => definition.staticPath)
            .map((definition) => [definition.staticPath, definition.name])
    );

    function contentGroup(name) {
        const definition = byName.get(name);
        if (!definition) return 'other';
        if (definition.group === 'tools') return definition.analyticsName || definition.name;
        if (definition.name === 'tools') return 'tools';
        if (definition.group === 'guides' || definition.group === 'sdk') return 'documentation';
        if (definition.name === 'guides' || definition.name === 'sdk' || definition.name === 'roadmap') {
            return 'documentation';
        }
        return definition.name;
    }

    function normalizedPath(path) {
        return path.endsWith('/') ? path : `${path}/`;
    }

    function contentNameForPath(path, prefix, rootName) {
        const definition = byName.get(byStaticPath.get(normalizedPath(path)));
        if (!definition || (definition.name !== rootName && definition.group !== rootName)) return '';
        if (definition.analyticsName) return definition.analyticsName;
        return definition.name.slice(prefix.length);
    }

    function documentNameForPath(path) {
        const normalized = normalizedPath(path);
        return documentPathNames.get(normalized)
            || contentNameForPath(normalized, 'sdk-', 'sdk');
    }

    function staticTargetForHref(href, baseHref) {
        let base;
        let target;
        try {
            base = new URL(baseHref);
            target = new URL(href, base);
        } catch (error) {
            return null;
        }
        if (target.origin !== base.origin || target.search) return null;
        const route = byStaticPath.get(target.pathname);
        if (!route) return null;
        return Object.freeze({
            route,
            anchor: target.hash.length > 1 ? target.hash.slice(1) : ''
        });
    }

    window.ESPectreRoutes = Object.freeze({
        all: Object.freeze(definitions.map((definition) => definition.name)),
        contentGroup,
        documentNameForPath,
        get: (name) => byName.get(name) || null,
        groupOf: (name) => byName.get(name)?.group || '',
        guideNameForPath: (path) => contentNameForPath(path, 'guide-', 'guides'),
        has: (name) => byName.has(name),
        membersOf: (group) => byGroup.get(group) || emptyGroup,
        routeForPath: (path) => byStaticPath.get(path) || '',
        staticTargetForHref,
        title: (name) => byName.get(name)?.title || ''
    });
    window.ESPectreSite = Object.freeze({
        analyticsAllowed,
        analyticsDebug,
        directOriginKind,
        isLoopbackHostname: (hostname) => loopbackHosts.has(hostname),
        kind: siteKind
    });
}());
