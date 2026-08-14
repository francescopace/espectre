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
        const bluetooth = !ios && Boolean(
            nav.bluetooth && typeof nav.bluetooth.requestDevice === 'function'
        );
        const serial = Boolean(nav.serial && typeof nav.serial.requestPort === 'function');

        return Object.freeze({
            ios,
            android,
            mobile,
            bluetooth,
            serial,
            // ESPectre supports browser flashing only on desktop Chrome or Edge.
            flash: serial && !mobile
        });
    }

    root.ESPectreBrowserSupport = Object.freeze({
        detect,
        current: detect(root.navigator)
    });
}(window));
