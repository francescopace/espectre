/*
 * ESPectre - Analytics
 * 
 * Shared Google Analytics event tracking for all pages.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

const GA_MEASUREMENT_ID = 'G-S0NQNG0V11';

function getSiteSection(pathname = window.location.pathname) {
    if (pathname === '/') return 'home';
    if (pathname.startsWith('/guides/')) return 'guides';
    if (pathname.startsWith('/game/')) return 'game';
    if (pathname.startsWith('/flash/')) return 'flash';
    if (pathname.startsWith('/configure/')) return 'configure';
    if (pathname.startsWith('/monitor/')) return 'monitor';
    if (pathname.startsWith('/theremin/')) return 'theremin';
    return 'other';
}

function initializeAnalytics() {
    window.dataLayer = window.dataLayer || [];
    window.gtag = window.gtag || function() { window.dataLayer.push(arguments); };

    if (!document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}"]`)) {
        const script = document.createElement('script');
        script.async = true;
        script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
        document.head.appendChild(script);
    }

    window.gtag('js', new Date());
    window.gtag('config', GA_MEASUREMENT_ID, {
        content_group: getSiteSection()
    });
}

initializeAnalytics();

// ==================== ANALYTICS HELPERS ====================

function trackEvent(eventName, params = {}) {
    window.gtag('event', eventName, params);
}

window.trackEvent = trackEvent;

// ==================== AUTO-TRACKING SETUP ====================

document.addEventListener('DOMContentLoaded', function() {
    const siteSection = getSiteSection();
    const requiredBrowserCapabilities = {
        configure: ['web_bluetooth', 'bluetooth'],
        flash: ['web_serial', 'serial'],
        game: ['web_bluetooth', 'bluetooth']
    };
    const requiredCapability = requiredBrowserCapabilities[siteSection];
    if (requiredCapability) {
        trackEvent('tool_capability', {
            tool_name: siteSection,
            capability: requiredCapability[0],
            result: requiredCapability[1] in navigator ? 'available' : 'unavailable'
        });
    }

    // Get Started (scroll to get-started; keep #config for old links)
    document.querySelectorAll('a[href="#get-started"], a[href="#config"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_get_started'));
    });
    
    // Contact links
    document.querySelectorAll('a[href^="mailto:contact@"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_contact'));
    });
    
    // Security link
    document.querySelectorAll('a[href^="mailto:security@"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_security'));
    });
    
    // Main product areas
    document.querySelectorAll('a[href^="/flash/"], a[href^="/configure/"], a[href^="/monitor/"], a[href^="/game/"], a[href^="/theremin/"]').forEach(link => {
        link.addEventListener('click', () => {
            trackEvent('select_tool', {
                tool_name: getSiteSection(new URL(link.href, window.location.origin).pathname),
                link_text: link.textContent.trim().replace(/\s+/g, ' ').substring(0, 100)
            });
        });
    });

    // Guide destinations are grouped by their first path segment so the
    // dashboard can compare interest without creating one event name per page.
    document.querySelectorAll('a[href^="/guides/"]').forEach(link => {
        link.addEventListener('click', () => {
            const path = new URL(link.href, window.location.origin).pathname;
            const guideName = path.replace(/^\/guides\/?/, '').replace(/\/$/, '');
            trackEvent('select_guide', {
                guide_name: guideName,
                link_text: link.textContent.trim().replace(/\s+/g, ' ').substring(0, 100)
            });
        });
    });
});
