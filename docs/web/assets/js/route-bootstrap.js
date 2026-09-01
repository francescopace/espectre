/*
 * ESPectre - Pre-paint route bootstrap
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const root = document.documentElement;
    const currentRoute = root.dataset.spaRoute || '';
    if (!currentRoute) {
        if (window.location.hash.length > 1) root.setAttribute('data-spa-booting', '');
        return;
    }

    const navigation = window.performance?.getEntriesByType?.('navigation')?.[0];
    const rememberedRoute = window.history.state?.espectreRoute || '';
    if (navigation?.type !== 'reload' || rememberedRoute !== currentRoute) return;

    const destination = new URL('/', window.location.href);
    destination.search = window.location.search;
    if (window.location.hash.length > 1) {
        destination.searchParams.set('anchor', window.location.hash.slice(1));
    }
    destination.hash = currentRoute;
    window.location.replace(destination.pathname + destination.search + destination.hash);
}());
