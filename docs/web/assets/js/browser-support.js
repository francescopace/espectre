/*
 * ESPectre - Browser capability policy
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function (root) {
    'use strict';

    function detect(navigatorLike) {
        const nav = navigatorLike || {};
        const userAgent = nav.userAgent || '';
        const platform = (nav.userAgentData && nav.userAgentData.platform) || nav.platform || '';
        const ios = /iPad|iPhone|iPod/i.test(userAgent)
            || (platform === 'MacIntel' && Number(nav.maxTouchPoints) > 1);
        const android = /Android/i.test(userAgent);
        const mobile = Boolean(nav.userAgentData && nav.userAgentData.mobile)
            || ios || android || /Mobile/i.test(userAgent);
        const serial = Boolean(nav.serial && typeof nav.serial.requestPort === 'function');
        const browser = /Edg\//i.test(userAgent) ? 'edge'
            : /Firefox\//i.test(userAgent) ? 'firefox'
                : /(?:Chrome|CriOS)\//i.test(userAgent) ? 'chrome'
                    : /Safari\//i.test(userAgent) ? 'safari' : 'other';
        const versionMatch = userAgent.match(browser === 'edge' ? /Edg\/(\d+)/i
            : browser === 'firefox' ? /Firefox\/(\d+)/i
                : browser === 'chrome' ? /(?:Chrome|CriOS)\/(\d+)/i
                    : browser === 'safari' ? /Version\/(\d+)/i : null);
        const browserMajor = versionMatch ? Number(versionMatch[1]) : 0;
        const operatingSystem = ios ? 'ios'
            : android ? 'android'
                : /mac/i.test(platform) ? 'macos'
                    : /win/i.test(platform) ? 'windows'
                        : /linux/i.test(platform) ? 'linux' : 'other';
        const hostedDirect = !mobile && browser === 'chrome' && browserMajor >= 151
            && ['macos', 'windows', 'linux'].includes(operatingSystem)
            ? 'targeted'
            : ['firefox', 'safari'].includes(browser) ? 'unsupported' : 'unclaimed';

        return Object.freeze({
            ios,
            android,
            mobile,
            serial,
            browser,
            browserMajor,
            operatingSystem,
            hostedDirect,
            // ESPectre supports browser flashing only on desktop Chrome or Edge.
            flash: serial && !mobile
        });
    }

    async function localNetworkAccessState(navigatorLike) {
        const permissions = (navigatorLike || {}).permissions;
        if (!permissions || typeof permissions.query !== 'function') return 'unavailable';
        for (const name of ['local-network', 'local-network-access']) {
            try {
                const result = await permissions.query({ name });
                if (['granted', 'prompt', 'denied'].includes(result && result.state)) {
                    return result.state;
                }
            } catch (_error) {
                // Chromium versions before the split support only the legacy alias.
            }
        }
        return 'unavailable';
    }

    root.ESPectreBrowserSupport = Object.freeze({
        detect,
        localNetworkAccessState,
        current: detect(root.navigator)
    });
}(window));
