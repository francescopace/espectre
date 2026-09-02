/*
 * ESPectre - Shared website registry
 *
 * Route data comes from /routes.json. This file owns only browser-side route
 * behavior and the deployment host policy.
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

    function normalizedPath(path) {
        return path.endsWith('/') ? path : `${path}/`;
    }

    function installRouteManifest(manifest) {
        if (!manifest || !Array.isArray(manifest.routes) || !manifest.routes.length) {
            throw new Error('ESPectre route manifest contains no routes');
        }

        const contentGroups = manifest.contentGroups;
        if (!contentGroups || typeof contentGroups !== 'object' || Array.isArray(contentGroups)) {
            throw new Error('ESPectre route manifest contains no content groups');
        }
        const routeNames = manifest.routes.map((definition) => definition.name);
        if (Object.keys(contentGroups).length !== routeNames.length
                || routeNames.some((name) => typeof contentGroups[name] !== 'string' || !contentGroups[name])) {
            throw new Error('ESPectre route manifest content groups do not match its routes');
        }
        const definitions = manifest.routes.map((definition) => Object.freeze({
            ...definition,
            contentGroup: contentGroups[definition.name]
        }));
        const byName = new Map(definitions.map((definition) => [definition.name, definition]));
        const byStaticPath = new Map(
            definitions.map((definition) => [definition.staticPath, definition.name])
        );
        if (byName.size !== definitions.length || byStaticPath.size !== definitions.length) {
            throw new Error('ESPectre route manifest contains duplicate routes');
        }

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

        const sdkChannelPathNames = new Map(
            (manifest.sdkChannels || []).map((sdkChannel) => [sdkChannel.path, sdkChannel.analyticsName])
        );

        function contentGroup(name) {
            const definition = byName.get(name);
            return definition?.contentGroup || 'other';
        }

        function contentNameForPath(path, prefix, rootName) {
            const definition = byName.get(byStaticPath.get(normalizedPath(path)));
            if (!definition || (definition.name !== rootName && definition.group !== rootName)) return '';
            if (definition.analyticsName) return definition.analyticsName;
            return definition.name.slice(prefix.length);
        }

        function documentNameForPath(path) {
            const normalized = normalizedPath(path);
            return sdkChannelPathNames.get(normalized)
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
            if (!route || route === 'home') return null;
            return Object.freeze({
                route,
                anchor: target.hash.length > 1 ? target.hash.slice(1) : ''
            });
        }

        const registry = Object.freeze({
            all: Object.freeze(definitions.map((definition) => definition.name)),
            contentGroup,
            contentPath: (name) => {
                const path = byName.get(name)?.staticPath;
                return path && path !== '/' ? `content${path.replace(/\/$/, '')}.html` : '';
            },
            documentNameForPath,
            get: (name) => byName.get(name) || null,
            groupOf: (name) => byName.get(name)?.group || '',
            guideNameForPath: (path) => contentNameForPath(path, 'guide-', 'guides'),
            has: (name) => byName.has(name),
            membersOf: (group) => byGroup.get(group) || emptyGroup,
            routeForPath: (path) => byStaticPath.get(path) || '',
            siteOrigin: manifest.siteOrigin,
            staticTargetForHref,
            title: (name) => byName.get(name)?.title || ''
        });
        window.ESPectreRoutes = registry;
        return registry;
    }

    window.ESPectreSite = Object.freeze({
        analyticsAllowed,
        analyticsDebug,
        directOriginKind,
        isLoopbackHostname: (hostname) => loopbackHosts.has(hostname),
        kind: siteKind
    });

    if (window.ESPectreRouteManifest) {
        window.ESPectreRoutesReady = Promise.resolve(
            installRouteManifest(window.ESPectreRouteManifest)
        );
        return;
    }

    window.ESPectreRoutesReady = fetch('/routes.json', { cache: 'no-cache' })
        .then((response) => {
            if (!response.ok) throw new Error(`Unable to load route manifest: HTTP ${response.status}`);
            return response.json();
        })
        .then(installRouteManifest);
}());
