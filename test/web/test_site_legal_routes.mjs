/*
 * ESPectre - Website legal route contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { app, browserSupportSource, directProtocol, GPL_HTML_HEADER, index, read, roadmapContent, routeRegistry, security, styles, toolContent, toolFragments, toolsContent } from './fixtures/site_test_helpers.mjs';

describe('website legal route contracts', () => {
    it('keeps privacy discoverable and serves a real 404 page', () => {
        assert.match(index, /data-page="privacy"/);
        assert.match(index, /data-content-url="content\/privacy\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<div class="footer-links">\s*<a href="\/privacy\/"/);
        assert.match(routeRegistry, /name: 'privacy'.*staticPath: '\/privacy\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/privacy\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/privacy\/"/);
        const sitemap = read('.github/scripts/sitemap.template.xml');
        assert.match(sitemap, /https:\/\/espectre\.dev\/privacy\//);
        assert.doesNotMatch(sitemap, /<(?:changefreq|lastmod)>/);
        assert.match(read('docs/web/content/privacy.html'), /id="cookie-settings"/);
        const notFound = read('docs/web/404.html');
        assert.doesNotMatch(notFound, /http-equiv="refresh"|location\.replace/);
        assert.match(notFound, /<footer class="site-footer">/);
        assert.match(styles, /body \{[\s\S]*?display: flex;[\s\S]*?flex-direction: column;[\s\S]*?min-height: 100dvh;/);
        assert.match(styles, /body > main \{[\s\S]*?width: 100%;[\s\S]*?box-sizing: border-box;[\s\S]*?flex: 1 0 auto;/);
        const sharedFooterBrand = /<div class="footer-brand">\s*<img src="\/assets\/images\/brand\/espectre-logo\.svg(?:\?v=(?:[0-9a-f]{12}|\{logo_version\}))?" alt="" width="23" height="23" aria-hidden="true">/;
        assert.match(index, sharedFooterBrand);
        assert.match(notFound, sharedFooterBrand);
        assert.match(read('.github/scripts/build_static_pages.py'), sharedFooterBrand);
        assert.match(read('.github/scripts/stage_web_sdk.py'), sharedFooterBrand);
        assert.match(styles, /\.footer-brand \{[^}]*color: var\(--text\);/);
        assert.match(notFound, /data-static-page data-site-section="other"/);
        assert.match(notFound, /<a href="\/privacy\/#cookie-settings" class="js-cookie-settings"/);
        assert.match(styles, /\.footer-links a,\s*\.footer-links a:visited \{[\s\S]*?display: inline-flex;[\s\S]*?color: var\(--text\);[\s\S]*?text-decoration: none;/);
        assert.match(styles, /\.footer-links a:hover,\s*\.footer-links a:focus-visible \{ color: var\(--accent\); text-decoration: none; \}/);
        assert.match(read('docs/web/content/privacy.html'), /<h2 id="cookie-settings">/);
        assert.match(read('docs/web/assets/js/analytics.js'), /document\.addEventListener\('click'[\s\S]*?closest\('\.js-cookie-settings'\)[\s\S]*?event\.preventDefault\(\);[\s\S]*?showConsentBanner\(\);/);
        assert.match(notFound, /class="consent-banner js-consent-banner"/);
        const notFoundScripts = [...notFound.matchAll(/<script\b([^>]*)>/g)]
            .map((match) => match[1]);
        assert.deepEqual(
            notFoundScripts.map((attrs) => attrs.match(/src="(\/assets\/js\/[^"?]+)/)[1]),
            [
                '/assets/js/route-registry.js',
                '/assets/js/navigation.js',
                '/assets/js/analytics.js'
            ]
        );
        for (const attrs of notFoundScripts) {
            assert.match(attrs, /\bdefer\b/, `expected defer on ${attrs.trim()}`);
        }
    });

    it('publishes a dedicated commercial licensing page', () => {
        const licensingContent = read('docs/web/content/licensing.html');
        assert.match(index, /data-page="licensing"/);
        assert.match(index, /data-content-url="content\/licensing\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<a href="\/licensing\/"/);
        assert.match(index, /<a href="\/contact\/"/);
        assert.match(routeRegistry, /name: 'licensing'.*staticPath: '\/licensing\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/licensing\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/licensing\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/licensing\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/licensing\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/licensing\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/licensing\/": \(Path\("docs\/web\/content\/licensing\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(licensingContent, /<h1 class="page-title">/);
        assert.match(licensingContent, /mailto:contact@espectre\.dev\?subject=Commercial%20licensing%20inquiry/);
    });

    it('publishes a dedicated contact page from every footer', () => {
        const contactContent = read('docs/web/content/contact.html');
        assert.match(index, /data-page="contact"/);
        assert.match(index, /data-content-url="content\/contact\.html\?v=[0-9a-f]{12}"/);
        assert.match(routeRegistry, /name: 'contact'.*staticPath: '\/contact\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/contact\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/contact\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/contact\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/contact\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/contact\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/contact\/": \(Path\("docs\/web\/content\/contact\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(contactContent, /<h1 class="page-title">/);
        assert.match(contactContent, /mailto:contact@espectre\.dev/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/discussions/);
        assert.match(contactContent, /github\.com\/francescopace\/espectre\/issues/);
        assert.doesNotMatch(contactContent, /mailto:security@espectre\.dev/);
    });

    it('publishes a dedicated security and responsible-use page', () => {
        const securityContent = read('docs/web/content/security.html');
        assert.match(index, /data-page="security"/);
        assert.match(index, /data-content-url="content\/security\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /<a href="\/security\/"/);
        assert.match(routeRegistry, /name: 'security'.*staticPath: '\/security\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/security\.html"/);
        assert.match(read('.github/scripts/build_static_pages.py'), /<a href="\/security\/"/);
        assert.match(read('.github/scripts/stage_web_sdk.py'), /<a href="\/security\/"/);
        assert.match(read('docs/web/404.html'), /<a href="\/security\/"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/security\//);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/security\/": \(Path\("docs\/web\/content\/security\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(securityContent, /<h1 class="page-title">/);
        assert.match(securityContent, /mailto:contact@espectre\.dev\?subject=Responsible%20use%20or%20abuse%20report/);
        assert.match(securityContent, /mailto:security@espectre\.dev/);
        assert.match(securityContent, /github\.com\/francescopace\/espectre\/security/);
        assert.doesNotMatch(securityContent, /github\.com\/francescopace\/espectre\/blob\/main\/SECURITY\.md/);
    });

    it('publishes website terms and current legal information', () => {
        const termsContent = read('docs/web/content/terms.html');
        const legalContent = read('docs/web/content/legal.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        const sdkPageBuilder = read('.github/scripts/stage_web_sdk.py');
        const notFound = read('docs/web/404.html');
        const sitemap = read('.github/scripts/sitemap.template.xml');
        const sitemapBuilder = read('.github/scripts/build_sitemap.py');

        assert.match(index, /data-page="terms"/);
        assert.match(index, /data-content-url="content\/terms\.html\?v=[0-9a-f]{12}"/);
        assert.match(index, /data-page="legal"/);
        assert.match(index, /data-content-url="content\/legal\.html\?v=[0-9a-f]{12}"/);
        assert.match(routeRegistry, /name: 'terms'.*staticPath: '\/terms\/'/);
        assert.match(routeRegistry, /name: 'legal'.*staticPath: '\/legal\/'/);

        for (const source of [index, staticPageBuilder, sdkPageBuilder, notFound]) {
            assert.match(source, /<a href="\/terms\/"/);
            assert.match(source, /<a href="\/legal\/"/);
        }
        assert.match(staticPageBuilder, /"source": "content\/terms\.html"/);
        assert.match(staticPageBuilder, /"source": "content\/legal\.html"/);
        assert.match(sitemap, /https:\/\/espectre\.dev\/terms\//);
        assert.match(sitemap, /https:\/\/espectre\.dev\/legal\//);
        assert.match(sitemapBuilder, /"\/terms\/": \(Path\("docs\/web\/content\/terms\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(sitemapBuilder, /"\/legal\/": \(Path\("docs\/web\/content\/legal\.html"\), STATIC_PAGE_BUILDER\)/);

        assert.match(termsContent, /<h1 class="page-title">/);
        assert.match(legalContent, /<h1 class="page-title">/);
        assert.match(legalContent, /<dl[\s\S]*?href="mailto:contact@espectre\.dev"[\s\S]*?<\/dl>/);
        assert.doesNotMatch(legalContent, /francesco\.pace@espectre\.dev|security@espectre\.dev|href="\/security\/"/);
    });

    it('treats top-level SDK, roadmap, privacy, terms, legal, security, licensing, and contact as pages, not articles', () => {
        const sdkContent = read('docs/web/content/sdk.html');
        const roadmapContent = read('docs/web/content/roadmap.html');
        const privacyContent = read('docs/web/content/privacy.html');
        const termsContent = read('docs/web/content/terms.html');
        const legalContent = read('docs/web/content/legal.html');
        const securityContent = read('docs/web/content/security.html');
        const licensingContent = read('docs/web/content/licensing.html');
        const contactContent = read('docs/web/content/contact.html');
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.ok(sdkContent.startsWith(`${GPL_HTML_HEADER}<div class="docs-quickstart" data-page-toc data-page-path="sdk">`));
        assert.ok(roadmapContent.startsWith(`${GPL_HTML_HEADER}<div class="roadmap-page" data-page-toc>`));
        assert.ok(privacyContent.startsWith(`${GPL_HTML_HEADER}<div class="privacy-page" data-page-toc>`));
        assert.ok(termsContent.startsWith(`${GPL_HTML_HEADER}<div class="terms-page" data-page-toc>`));
        assert.ok(legalContent.startsWith(`${GPL_HTML_HEADER}<div class="legal-page" data-page-toc>`));
        assert.ok(securityContent.startsWith(`${GPL_HTML_HEADER}<div class="security-page" data-page-toc>`));
        assert.ok(licensingContent.startsWith(`${GPL_HTML_HEADER}<div class="licensing-page" data-page-toc>`));
        assert.ok(contactContent.startsWith(`${GPL_HTML_HEADER}<div class="contact-page">`));
        assert.match(index, /<main class="js-page page-narrow" data-page="roadmap"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="privacy"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="terms"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="legal"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="security"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="licensing"/);
        assert.match(index, /<main class="js-page page-narrow" data-page="contact"/);
        assert.doesNotMatch(index, /<main class="js-page page-narrow page-article" data-page="(?:sdk|roadmap|privacy|terms|legal|security|licensing|contact)"/);
        assert.match(staticPageBuilder, /"source": "content\/sdk\.html",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/roadmap\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/privacy\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/terms\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/legal\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/security\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/licensing\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /"source": "content\/contact\.html",[\s\S]*?"main_class": "page-narrow",[\s\S]*?"og_type": "website"/);
        assert.match(staticPageBuilder, /<meta property="og:type" content="\{og_type\}">/);
    });

});
