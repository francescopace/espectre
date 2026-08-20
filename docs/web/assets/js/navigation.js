/*
 * ESPectre - Shared responsive navigation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

(function () {
    'use strict';

    const compactToc = window.matchMedia('(max-width: 720px)');

    function setPageTocMode(toc) {
        toc.open = !compactToc.matches;
    }

    function initPageTocs(root = document) {
        root.querySelectorAll('details.page-toc:not([data-toc-initialized])').forEach((toc) => {
            toc.dataset.tocInitialized = 'true';
            setPageTocMode(toc);
        });
    }

    const sdkVersionPromises = new Map();

    function sdkVersion(channel) {
        if (sdkVersionPromises.has(channel)) return sdkVersionPromises.get(channel);
        const promise = fetch(`/artifacts/sdk/${channel}/sdk-manifest-${channel}.json`)
            .then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            })
            .then((manifest) => {
                if (typeof manifest.version !== 'string' || !manifest.version) {
                    throw new Error('SDK manifest version is unavailable');
                }
                return manifest.version;
            });
        sdkVersionPromises.set(channel, promise);
        return promise;
    }

    function initSdkDownloadVersions(root = document) {
        root.querySelectorAll('[data-sdk-version]:not([data-sdk-version-initialized])').forEach((label) => {
            label.dataset.sdkVersionInitialized = 'true';
            const channel = label.dataset.sdkVersion;
            sdkVersion(channel).then((version) => {
                label.textContent = `Version ${version}`;
            }).catch(() => {
                label.title = `${channel} version unavailable`;
            });
        });
    }

    window.initPageTocs = initPageTocs;
    window.initSdkDownloadVersions = initSdkDownloadVersions;
    compactToc.addEventListener('change', () => {
        document.querySelectorAll('details.page-toc').forEach(setPageTocMode);
    });

    function closeNavigation(toggle, nav) {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        const label = toggle.querySelector('.sr-only');
        if (label) label.textContent = 'Open navigation';
    }

    document.addEventListener('DOMContentLoaded', () => {
        initPageTocs();
        initSdkDownloadVersions();
        const toggle = document.querySelector('.nav-toggle');
        const nav = document.getElementById('main-navigation');
        if (!toggle || !nav) return;

        toggle.addEventListener('click', () => {
            const opening = !nav.classList.contains('is-open');
            nav.classList.toggle('is-open', opening);
            toggle.setAttribute('aria-expanded', String(opening));
            const label = toggle.querySelector('.sr-only');
            if (label) label.textContent = opening ? 'Close navigation' : 'Open navigation';
        });

        nav.addEventListener('click', (event) => {
            if (event.target.closest('a')) closeNavigation(toggle, nav);
        });

        document.addEventListener('click', (event) => {
            document.querySelectorAll('details.sdk-download[open]').forEach((menu) => {
                if (!menu.contains(event.target)) menu.open = false;
            });
            if (!nav.classList.contains('is-open')) return;
            if (nav.contains(event.target) || toggle.contains(event.target)) return;
            closeNavigation(toggle, nav);
        });

        document.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || !nav.classList.contains('is-open')) return;
            closeNavigation(toggle, nav);
            toggle.focus();
        });
    });
})();
