/*
 * ESPectre - Privacy-conscious analytics
 *
 * Google Analytics is enabled only on the production hostname and only after
 * explicit consent. The SPA router owns manual page views; generated static
 * pages report their canonical path directly.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

const GA_MEASUREMENT_ID = 'G-S0NQNG0V11';
const ANALYTICS_CONSENT_KEY = 'espectre.analytics.consent.v1';
const PRODUCTION_HOSTS = new Set(['espectre.dev', 'www.espectre.dev']);
const LOCAL_ANALYTICS_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);
const IS_STATIC_PAGE = document.documentElement.hasAttribute('data-static-page');
const STATIC_PAGE_SECTION = document.documentElement.dataset.siteSection || 'documentation';

const CAPABILITY_BY_ROUTE = {
    configure: ['web_bluetooth', 'bluetooth'],
    theremin: ['web_bluetooth', 'bluetooth'],
    game: ['web_bluetooth', 'bluetooth'],
    flash: ['web_serial', 'serial']
};

const reportedCapabilities = new Set();
let analyticsEnabled = false;
let analyticsConfigured = false;

function currentRoute() {
    return (window.location.hash || '#home').slice(1) || 'home';
}

function getRouteTitle(route = currentRoute()) {
    const registeredTitle = window.ESPectreRoutes?.title(route);
    if (registeredTitle) return registeredTitle;
    const conventionalPrefix = ['guide-', 'docs-'].find((prefix) => route.startsWith(prefix));
    if (conventionalPrefix) {
        const slug = route.slice(conventionalPrefix.length);
        const label = slug.replace(/[-_]+/g, ' ').trim();
        if (label) return `${label.charAt(0).toUpperCase()}${label.slice(1)} | ESPectre`;
    }
    return 'ESPectre — Wi-Fi motion sensing';
}

function getSiteSection(route = currentRoute()) {
    if (IS_STATIC_PAGE) return STATIC_PAGE_SECTION;
    return window.ESPectreRoutes?.contentGroup(route) || 'other';
}

function routePath(route) {
    return route === 'home' ? '/' : `/#${route}`;
}

function analyticsAllowedHere() {
    return PRODUCTION_HOSTS.has(window.location.hostname) || localAnalyticsDebugEnabled();
}

function localAnalyticsDebugEnabled() {
    return LOCAL_ANALYTICS_HOSTS.has(window.location.hostname);
}

function storedConsent() {
    try {
        return window.localStorage.getItem(ANALYTICS_CONSENT_KEY);
    } catch (error) {
        return null;
    }
}

function saveConsent(value) {
    try {
        window.localStorage.setItem(ANALYTICS_CONSENT_KEY, value);
    } catch (error) {
        // Consent remains valid for the current page when storage is blocked.
    }
}

function ensureGtagQueue() {
    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
}

function loadGoogleTag() {
    if (document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}"]`)) return;
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
    script.referrerPolicy = 'strict-origin-when-cross-origin';
    document.head.appendChild(script);
}

function updatePageConfig(pageLocation, pageTitle, contentGroup) {
    window.gtag('config', GA_MEASUREMENT_ID, {
        update: true,
        page_location: pageLocation,
        page_title: pageTitle,
        content_group: contentGroup
    });
}

function sendStaticPageView() {
    const pageLocation = window.location.origin + window.location.pathname;
    updatePageConfig(pageLocation, document.title, STATIC_PAGE_SECTION);
    window.gtag('event', 'page_view', {
        page_location: pageLocation,
        page_path: window.location.pathname,
        page_title: document.title,
        content_group: STATIC_PAGE_SECTION
    });
}

function sendRoutePageView(route = currentRoute()) {
    const path = routePath(route);
    const pageLocation = window.location.origin + path;
    const pageTitle = getRouteTitle(route);
    const contentGroup = getSiteSection(route);
    updatePageConfig(pageLocation, pageTitle, contentGroup);
    window.gtag('event', 'page_view', {
        page_location: pageLocation,
        page_path: path,
        page_title: pageTitle,
        content_group: contentGroup
    });
}

function enableAnalytics({ sendPageView = true } = {}) {
    if (!analyticsAllowedHere()) return;
    const wasEnabled = analyticsEnabled;
    ensureGtagQueue();

    if (!analyticsConfigured) {
        window.gtag('consent', 'default', {
            analytics_storage: 'denied',
            ad_storage: 'denied',
            ad_user_data: 'denied',
            ad_personalization: 'denied',
            wait_for_update: 500
        });
    }

    window.gtag('consent', 'update', {
        analytics_storage: 'granted',
        ad_storage: 'denied',
        ad_user_data: 'denied',
        ad_personalization: 'denied'
    });

    if (!analyticsConfigured) {
        window.gtag('js', new Date());
        const config = {
            send_page_view: false,
            allow_google_signals: false,
            allow_ad_personalization_signals: false
        };
        if (localAnalyticsDebugEnabled()) config.debug_mode = true;
        window.gtag('config', GA_MEASUREMENT_ID, config);
        loadGoogleTag();
        analyticsConfigured = true;
    }

    analyticsEnabled = true;
    if (sendPageView && !wasEnabled) {
        if (IS_STATIC_PAGE) sendStaticPageView();
        else sendRoutePageView();
    }
    if (!wasEnabled) document.dispatchEvent(new CustomEvent('espectre:analytics-enabled'));
}

function clearAnalyticsCookies() {
    document.cookie.split(';').forEach((entry) => {
        const name = entry.split('=')[0].trim();
        if (!name.startsWith('_ga')) return;
        document.cookie = `${name}=; Max-Age=0; Path=/; SameSite=Lax`;
        document.cookie = `${name}=; Max-Age=0; Path=/; Domain=.espectre.dev; SameSite=Lax`;
    });
}

function disableAnalytics() {
    analyticsEnabled = false;
    if (typeof window.gtag === 'function') {
        window.gtag('consent', 'update', {
            analytics_storage: 'denied',
            ad_storage: 'denied',
            ad_user_data: 'denied',
            ad_personalization: 'denied'
        });
    }
    clearAnalyticsCookies();
}

function trackEvent(eventName, params = {}) {
    if (!analyticsEnabled) return false;
    window.gtag('event', eventName, {
        content_group: getSiteSection(),
        ...params
    });
    return true;
}

function trackRouteView(route = currentRoute(), { sendPageView = true } = {}) {
    if (!analyticsEnabled) return;
    if (sendPageView) sendRoutePageView(route);

    const capability = CAPABILITY_BY_ROUTE[route];
    if (!capability || reportedCapabilities.has(route)) return;
    reportedCapabilities.add(route);
    trackEvent('tool_capability', {
        tool_name: route,
        capability: capability[0],
        result: capability[1] in navigator ? 'available' : 'unavailable'
    });
}

function linkText(link) {
    const heading = link.querySelector('h1, h2, h3, .doc-link-title');
    return (heading || link).textContent.trim().replace(/\s+/g, ' ').substring(0, 100);
}

function showConsentBanner() {
    const banner = document.querySelector('.js-consent-banner');
    if (banner) banner.hidden = false;
}

function hideConsentBanner() {
    const banner = document.querySelector('.js-consent-banner');
    if (banner) banner.hidden = true;
}

function initializeConsentControls() {
    document.querySelectorAll('.js-consent-accept').forEach((button) => {
        button.addEventListener('click', () => {
            saveConsent('granted');
            hideConsentBanner();
            enableAnalytics();
        });
    });
    document.querySelectorAll('.js-consent-reject').forEach((button) => {
        button.addEventListener('click', () => {
            saveConsent('denied');
            hideConsentBanner();
            disableAnalytics();
        });
    });
    document.querySelectorAll('.js-cookie-settings').forEach((button) => {
        button.addEventListener('click', showConsentBanner);
    });

    if (!analyticsAllowedHere()) {
        hideConsentBanner();
        return;
    }

    const consent = storedConsent();
    if (consent === 'granted') enableAnalytics();
    else if (consent !== 'denied') showConsentBanner();
}

function initializeAutoTracking() {
    document.addEventListener('click', (event) => {
        const link = event.target.closest('a[href]');
        if (!link) return;

        let url;
        try {
            url = new URL(link.href, window.location.origin);
        } catch (error) {
            return;
        }

        const guideName = url.origin === window.location.origin
            ? window.ESPectreRoutes?.guideNameForPath(url.pathname)
            : '';
        if (guideName) {
            trackEvent('select_guide', {
                guide_name: guideName,
                link_text: linkText(link)
            });
            return;
        }

        const documentName = url.origin === window.location.origin
            ? window.ESPectreRoutes?.documentNameForPath(url.pathname)
            : '';
        if (documentName) {
            trackEvent('select_documentation', {
                document_name: documentName,
                link_text: linkText(link)
            });
            return;
        }

        if (link.dataset.sdkChannel && link.dataset.sdkFormat) {
            trackEvent('sdk_download', {
                channel: link.dataset.sdkChannel,
                format: link.dataset.sdkFormat,
                link_text: linkText(link)
            });
            return;
        }

        const route = url.origin === window.location.origin ? url.hash.replace(/^#/, '') : '';
        if (window.ESPectreRoutes?.groupOf(route) === 'tools') {
            trackEvent('select_tool', { tool_name: route, link_text: linkText(link) });
        } else if (route === 'guides') {
            trackEvent('select_guide', { guide_name: 'overview', link_text: linkText(link) });
        } else if (route === 'docs' || route.startsWith('docs-')) {
            trackEvent('select_documentation', {
                document_name: route === 'docs' ? 'overview' : route.replace(/^docs-/, ''),
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
}

window.trackEvent = trackEvent;
window.trackRouteView = trackRouteView;
window.getSiteSection = getSiteSection;
window.getRouteTitle = getRouteTitle;
window.ESPectreAnalytics = Object.freeze({
    showConsentSettings: showConsentBanner,
    consent: storedConsent
});

document.addEventListener('DOMContentLoaded', () => {
    initializeConsentControls();
    initializeAutoTracking();
});
