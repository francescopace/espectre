/*
 * ESPectre - Shared website route registry
 *
 * This is the single source of truth for SPA route membership, navigation
 * groups, page titles, and canonical static paths.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const definitions = [
        { name: 'home', title: 'ESPectre — Wi-Fi motion sensing' },
        { name: 'tools', title: 'Browser tools | ESPectre' },
        { name: 'flash', group: 'tools', title: 'Flash firmware | ESPectre' },
        { name: 'configure', group: 'tools', title: 'Configure | ESPectre' },
        { name: 'monitor', group: 'tools', title: 'Monitor | ESPectre' },
        { name: 'theremin', group: 'tools', title: 'Motion theremin | ESPectre' },
        { name: 'game', group: 'tools', title: 'Motion game | ESPectre' },
        { name: 'guides', title: 'Guides | ESPectre', staticPath: '/guides/', analyticsName: 'overview' },
        { name: 'guide-hardware', group: 'guides', title: 'Choosing an ESP32 board | ESPectre', staticPath: '/guides/hardware/' },
        { name: 'guide-setup', group: 'guides', title: 'Flash & Wi-Fi setup | ESPectre', staticPath: '/guides/setup/' },
        { name: 'guide-home-assistant', group: 'guides', title: 'Build your Home Assistant dashboard | ESPectre', staticPath: '/guides/home-assistant/' },
        { name: 'guide-placement', group: 'guides', title: 'Sensor placement guide | ESPectre', staticPath: '/guides/placement/' },
        { name: 'guide-detection', group: 'guides', title: 'How Wi-Fi sensing detects movement | ESPectre', staticPath: '/guides/detection/' },
        { name: 'guide-detectors', group: 'guides', title: 'Detection profiles | ESPectre', staticPath: '/guides/detectors/' },
        { name: 'guide-micropython', group: 'guides', title: 'Run ESPectre on MicroPython | ESPectre', staticPath: '/guides/micropython/' },
        { name: 'guide-future-wifi-sensing', group: 'guides', title: 'The future of Wi-Fi sensing | ESPectre', staticPath: '/guides/future-wifi-sensing/', analyticsName: 'future-wifi-sensing' },
        { name: 'sdk', title: 'ESPectre SDK quick guide | ESPectre', staticPath: '/sdk/', analyticsName: 'overview' },
        { name: 'sdk-architecture', group: 'sdk', title: 'Architecture | ESPectre', staticPath: '/sdk/architecture/' },
        { name: 'sdk-api', group: 'sdk', title: 'API orientation | ESPectre', staticPath: '/sdk/api/' },
        { name: 'sdk-detectors', group: 'sdk', title: 'Detector architecture | ESPectre', staticPath: '/sdk/detectors/' },
        { name: 'sdk-examples', group: 'sdk', title: 'Examples | ESPectre', staticPath: '/sdk/examples/' },
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
        ['/artifacts/sdk/develop/', 'sdk_develop'],
        ['/artifacts/sdk/api/', 'api_reference']
    ]);

    const byName = new Map(definitions.map((definition) => [definition.name, definition]));
    const byStaticPath = new Map(
        definitions
            .filter((definition) => definition.staticPath)
            .map((definition) => [definition.staticPath, definition.name])
    );

    function contentGroup(name) {
        const definition = byName.get(name);
        if (!definition) return 'other';
        if (definition.group === 'tools') return definition.name;
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

    window.ESPectreRoutes = Object.freeze({
        all: Object.freeze(definitions.map((definition) => definition.name)),
        contentGroup,
        documentNameForPath,
        get: (name) => byName.get(name) || null,
        groupOf: (name) => byName.get(name)?.group || '',
        guideNameForPath: (path) => contentNameForPath(path, 'guide-', 'guides'),
        has: (name) => byName.has(name),
        routeForPath: (path) => byStaticPath.get(path) || '',
        title: (name) => byName.get(name)?.title || ''
    });
}());
