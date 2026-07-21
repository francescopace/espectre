/*
 * ESPectre - Analytics
 *
 * Shared Google Analytics event tracking for the single-page app.
 *
 * The site is a SPA, so gtag's automatic page_view fires only once. app.js
 * owns routing and calls trackRouteView on every navigation; routes are
 * reported under their real URL (`/#flash`), grouped by content_group.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

const GA_MEASUREMENT_ID = 'G-S0NQNG0V11';

/* Tool routes double as their own content_group value. */
const TOOL_ROUTES = ['flash', 'configure', 'monitor', 'theremin', 'game'];

function currentRoute() {
    return (window.location.hash || '#home').slice(1) || 'home';
}

/**
 * Maps a route to the standard content_group dimension. Guides and SDK docs
 * share one `documentation` group so they can be compared as a whole.
 */
function getSiteSection(route = currentRoute()) {
    if (route === 'home') return 'home';
    if (TOOL_ROUTES.includes(route)) return route;
    if (route === 'guides' || route.startsWith('guide-')) return 'documentation';
    if (route === 'docs' || route.startsWith('docs-')) return 'documentation';
    return 'other';
}

function routePath(route) {
    return route === 'home' ? '/' : `/#${route}`;
}

function initializeAnalytics() {
    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };

    if (!document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}"]`)) {
        const script = document.createElement('script');
        script.async = true;
        script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
        document.head.appendChild(script);
    }

    window.gtag('js', new Date());
    // Manual page_view: trackRouteView owns it so every route is reported once.
    window.gtag('config', GA_MEASUREMENT_ID, { send_page_view: false });
}

initializeAnalytics();

// ==================== ANALYTICS HELPERS ====================

function trackEvent(eventName, params = {}) {
    window.gtag('event', eventName, {
        content_group: getSiteSection(),
        ...params
    });
}

const CAPABILITY_BY_ROUTE = {
    configure: ['web_bluetooth', 'bluetooth'],
    theremin: ['web_bluetooth', 'bluetooth'],
    game: ['web_bluetooth', 'bluetooth'],
    flash: ['web_serial', 'serial']
};

const reportedCapabilities = new Set();

/**
 * Reports a SPA navigation, plus browser support the first time a tool that
 * needs a hardware API is opened. Called by the router in app.js.
 */
function trackRouteView(route = currentRoute()) {
    const section = getSiteSection(route);
    window.gtag('event', 'page_view', {
        page_location: window.location.origin + routePath(route),
        page_path: routePath(route),
        page_title: document.title,
        content_group: section
    });

    const capability = CAPABILITY_BY_ROUTE[route];
    if (!capability || reportedCapabilities.has(route)) return;
    reportedCapabilities.add(route);
    trackEvent('tool_capability', {
        tool_name: route,
        capability: capability[0],
        result: capability[1] in navigator ? 'available' : 'unavailable'
    });
}

window.trackEvent = trackEvent;
window.trackRouteView = trackRouteView;
window.getSiteSection = getSiteSection;

// ==================== AUTO-TRACKING SETUP ====================

document.addEventListener('DOMContentLoaded', function () {
    /*
     * Whole cards are links, so raw textContent would capture the entire card
     * body. Prefer the card heading when there is one.
     */
    const linkText = (link) => {
        const heading = link.querySelector('h1, h2, h3, .doc-link-title');
        return (heading || link).textContent.trim().replace(/\s+/g, ' ').substring(0, 100);
    };

    // Internal navigation is hash-based, so classify clicks by destination.
    document.addEventListener('click', function (event) {
        const link = event.target.closest('a[href^="#"]');
        if (!link) return;
        const route = link.getAttribute('href').slice(1);
        if (!route) return;

        if (TOOL_ROUTES.includes(route)) {
            trackEvent('select_tool', { tool_name: route, link_text: linkText(link) });
        } else if (route.startsWith('guide-')) {
            trackEvent('select_guide', {
                guide_name: route.replace(/^guide-/, ''),
                link_text: linkText(link)
            });
        } else if (route.startsWith('docs-')) {
            trackEvent('select_documentation', {
                document_name: route.replace(/^docs-/, ''),
                link_text: linkText(link)
            });
        }
    });

    document.querySelectorAll('a[href^="mailto:contact@"]').forEach((link) => {
        link.addEventListener('click', () => trackEvent('click_contact'));
    });
    document.querySelectorAll('a[href^="mailto:security@"]').forEach((link) => {
        link.addEventListener('click', () => trackEvent('click_security'));
    });
});
