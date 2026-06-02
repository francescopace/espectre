/*
 * ESPectre - Analytics
 * 
 * Shared Google Analytics event tracking for all pages.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

// ==================== ANALYTICS HELPERS ====================

function trackEvent(eventName, params = {}) {
    if (typeof gtag === 'function') {
        gtag('event', eventName, params);
    }
}

function trackOutboundClick(eventName, url) {
    trackEvent(eventName, { link_url: url });
}

// ==================== AUTO-TRACKING SETUP ====================

document.addEventListener('DOMContentLoaded', function() {
    // GitHub links
    document.querySelectorAll('a[href*="github.com/francescopace/espectre"]').forEach(link => {
        if (link.href.includes('/issues')) {
            link.addEventListener('click', () => trackOutboundClick('click_issues', link.href));
        } else if (link.href.includes('/discussions')) {
            link.addEventListener('click', () => trackOutboundClick('click_discussions', link.href));
        } else {
            link.addEventListener('click', () => trackOutboundClick('click_github', link.href));
        }
    });
    
    // Medium link
    document.querySelectorAll('a[href*="medium.com"]').forEach(link => {
        link.addEventListener('click', () => trackOutboundClick('click_medium', link.href));
    });
    
    // Game link
    document.querySelectorAll('a[href="/game/"], a[href="/game"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_game'));
    });
    
    // Get Started (scroll to config)
    document.querySelectorAll('a[href="#config"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_get_started'));
    });
    
    // Home link (from game page)
    document.querySelectorAll('nav a[href="/"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_home'));
    });
    
    // Contact links
    document.querySelectorAll('a[href^="mailto:contact@"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_contact'));
    });
    
    // Security link
    document.querySelectorAll('a[href^="mailto:security@"]').forEach(link => {
        link.addEventListener('click', () => trackEvent('click_security'));
    });
    
    // LinkedIn link
    document.querySelectorAll('a[href*="linkedin.com"]').forEach(link => {
        link.addEventListener('click', () => trackOutboundClick('click_linkedin', link.href));
    });
    
    // Buy Me a Coffee widget click detection
    const observer = new MutationObserver(function(mutations) {
        const bmcWidget = document.querySelector('#bmc-wbtn');
        if (bmcWidget && !bmcWidget.dataset.tracked) {
            bmcWidget.dataset.tracked = 'true';
            bmcWidget.addEventListener('click', () => trackEvent('click_bmc'));
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});

