/*
 * ESPectre - Shared responsive navigation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

(function () {
    'use strict';

    function closeNavigation(toggle, nav) {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
        const label = toggle.querySelector('.sr-only');
        if (label) label.textContent = 'Open navigation';
    }

    document.addEventListener('DOMContentLoaded', () => {
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
