/*
 * ESPectre - Shared Components
 * 
 * Injects shared header and footer across all pages
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

function loadHeader(options = {}) {
    const isGamePage = options.isGamePage || false;
    const headerEl = document.getElementById('site-header');
    if (!headerEl) return;

    // Determine home link based on page location
    const homeLink = '/';
    const utmMedium = isGamePage ? 'game.html' : 'index.html';
    
    // First nav item changes based on page
    const firstNavItem = isGamePage 
        ? `<a href="/"><i class="fas fa-home"></i> Home</a>`
        : `<a href="/game/"><i class="fas fa-gamepad"></i> The Game</a>`;

    headerEl.innerHTML = `
        <div class="header-left">
            <a href="${homeLink}" class="logo">
                <span class="logo-icon"><i class="fas fa-wifi"></i></span>
                <span>ESPectre</span>
                <span id="life-ghost" class="logo-icon life-ghost"><i class="fas fa-ghost"></i></span>
            </a>
            <div class="header-controls-group">
                <span class="header-separator">|</span>
                <button id="btn-usb" class="header-control btn-usb" title="Connect ESPectre device via USB"><i class="fab fa-usb"></i></button>
                <button id="btn-mute" class="header-control btn-mute" title="Enable sound"><i class="fas fa-volume-mute"></i></button>
            </div>
        </div>
        <button class="menu-toggle" aria-label="Toggle menu">
            <i class="fas fa-bars"></i>
        </button>
        <nav>
            ${firstNavItem}
            <a href="https://github.com/francescopace/espectre/issues"><i class="fas fa-bug"></i> Issues</a>
            <a href="https://github.com/francescopace/espectre/discussions"><i class="fas fa-comments"></i> Discussions</a>
            <div class="nav-dropdown nav-dropdown-right">
                <a href="#" class="nav-dropdown-toggle"><i class="fab fa-medium"></i> Medium <i class="fas fa-chevron-down"></i></a>
                <div class="nav-dropdown-menu">
                    <a href="https://medium.com/@francesco.pace/how-i-turned-my-wi-fi-into-a-motion-sensor-61a631a9b4ec?sk=c7f79130d78b0545fce4a228a6a79af3&utm_source=espectre.dev&utm_medium=${utmMedium}&utm_campaign=espectre">How I Turned My Wi-Fi Into a Motion Sensor - Part 1</a>
                    <a href="https://medium.com/@francesco.pace/how-i-turned-my-wi-fi-into-a-motion-sensor-part-2-62038130e530?sk=7c8b6f11cf3fcb8d279648016ebff72a&utm_source=espectre.dev&utm_medium=${utmMedium}&utm_campaign=espectre">How I Turned My Wi-Fi Into a Motion Sensor - Part 2</a>
                </div>
            </div>
            <div class="nav-dropdown nav-dropdown-right">
                <a href="#" class="nav-dropdown-toggle"><i class="fas fa-book"></i> Docs <i class="fas fa-chevron-down"></i></a>
                <div class="nav-dropdown-menu">
                    <a href="/documentation/"><i class="fas fa-home"></i> Main</a>
                    <a href="/documentation/setup/"><i class="fas fa-wrench"></i> Setup Guide</a>
                    <a href="/documentation/tuning/"><i class="fas fa-sliders"></i> Tuning Guide</a>
                    <a href="/documentation/changelog/"><i class="fas fa-list"></i> Changelog</a>
                    <a href="/documentation/roadmap/"><i class="fas fa-map"></i> Roadmap</a>
                    <div class="nav-dropdown-divider"></div>
                    <a href="/documentation/algorithms/"><i class="fas fa-square-root-variable"></i> Algorithms</a>
                    <a href="/documentation/micro-espectre/"><i class="fas fa-flask"></i> Micro-ESPectre</a>
                </div>
            </div>
        </nav>
    `;

    // Setup mobile menu toggle
    const menuToggle = headerEl.querySelector('.menu-toggle');
    const nav = headerEl.querySelector('nav');
    if (menuToggle && nav) {
        menuToggle.addEventListener('click', () => {
            nav.classList.toggle('open');
        });
    }

    // Setup dropdown toggles for mobile
    const dropdownToggles = headerEl.querySelectorAll('.nav-dropdown-toggle');
    dropdownToggles.forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                e.preventDefault();
                toggle.parentElement.classList.toggle('open');
            }
        });
    });

    // Close mobile menu when clicking nav links
    const navLinks = headerEl.querySelectorAll('nav a:not(.nav-dropdown-toggle)');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (nav) nav.classList.remove('open');
        });
    });
    
    // Dynamic header - shrink and glow on scroll
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            headerEl.classList.add('scrolled');
        } else {
            headerEl.classList.remove('scrolled');
        }
    });
}

function loadFooter() {
    const footerEl = document.getElementById('site-footer');
    if (!footerEl) return;

    footerEl.innerHTML = `
        <p class="footer-copy">
            <a href="mailto:contact@espectre.dev"><i class="fas fa-envelope"></i> Contact</a>
            <a href="mailto:security@espectre.dev"><i class="fas fa-shield-halved"></i> Security</a>
            <a href="https://linkedin.com/in/francescopace"><i class="fas fa-user"></i> Francesco Pace</a>
        </p>
        <p class="footer-copyright">© 2025 ESPectre · GPLv3 License</p>
    `;
}

