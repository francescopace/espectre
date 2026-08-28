/*
 * ESPectre - Website SDK route contracts
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

describe('website SDK route contracts', () => {
    it('uses the shared page heading styles on every top-level inner page', () => {
        for (const path of ['guides', 'sdk', 'roadmap', 'privacy', 'terms', 'legal', 'security', 'licensing', 'contact']) {
            const content = read(`docs/web/content/${path}.html`);
            assert.match(content, /<h1 class="page-title">/);
            assert.match(content, /<p class="page-sub">/);
        }
        assert.match(styles, /\.page-title \{ font-size: 40px;/);
        const pageSubRule = styles.match(/\.page-sub \{([^}]*)\}/)?.[1] || '';
        assert.match(pageSubRule, /font-size: 18px;/);
        assert.match(pageSubRule, /line-height: 1\.55;/);
        assert.match(styles, /@media \(max-width: 720px\) \{\s*\.page-title \{ font-size: 36px; \}\s*\.page-sub \{ font-size: 17px; \}/);
    });

    it('uses one shared measure for every inner page', () => {
        const staticPageBuilder = read('.github/scripts/build_static_pages.py');
        assert.match(styles, /\.page-narrow \{[\s\S]*?max-width: 1120px;/);
        assert.match(styles, /\.article \{ width: 100%; \}/);
        const innerPages = [...index.matchAll(/<main class="([^"]*)" data-page="([^"]+)"/g)]
            .filter(([, , page]) => page !== 'home');
        for (const [, classes] of innerPages) {
            assert.match(classes, /(?:^|\s)page-narrow(?:\s|$)/);
        }
        assert.match(staticPageBuilder, /spec\.get\("main_class", "page-narrow page-article"\)/);
    });

    it('gives the SDK landing page a clear start-to-reference hierarchy', () => {
        const sdkContent = read('docs/web/content/sdk.html');
        assert.match(sdkContent, /<h1 class="page-title">/);
        assert.match(sdkContent, /<section class="docs-start" aria-labelledby="docs-start-title">/);
        assert.match(sdkContent, /<section class="docs-section sdk-fit" aria-labelledby="sdk-fit-title">/);
        assert.match(sdkContent, /<section class="docs-section" aria-labelledby="docs-paths-title">/);
        assert.match(sdkContent, /<section class="docs-section" aria-labelledby="docs-quick-start-title">/);
        assert.match(sdkContent, /<section class="docs-next" aria-labelledby="docs-next-title">/);
        assert.match(sdkContent, /class="docs-cover"[\s\S]*?sdk-firmware-pipeline\.avif/);
        assert.match(sdkContent, /href="\/licensing\/"/);
        assert.ok(sdkContent.indexOf('class="docs-start"') < sdkContent.indexOf('class="sdk-fit-grid"'));
        assert.ok(sdkContent.indexOf('class="sdk-fit-grid"') < sdkContent.indexOf('class="docs-steps"'));
        assert.ok(sdkContent.indexOf('class="docs-steps"') < sdkContent.indexOf('class="docs-paths"'));
        assert.ok(sdkContent.indexOf('class="docs-steps"') < sdkContent.indexOf('class="docs-next"'));
        const sdkIndexLinks = [...sdkContent.matchAll(/<a href="(\/sdk\/(?:api|examples|architecture|detectors)\/)" class="doc-link">/g)].map((match) => match[1]);
        assert.deepEqual(sdkIndexLinks, ['/sdk/architecture/', '/sdk/detectors/', '/sdk/api/', '/sdk/examples/']);
        const pathCards = sdkContent.match(/<div class="docs-path(?: docs-path-recommended)?">[\s\S]*?<\/div>/g) || [];
        assert.equal(pathCards.length, 3);
        for (const card of pathCards) {
            assert.match(card, /<h3>/);
        }
    });

    it('documents both public SDK facades and their compatibility boundary', () => {
        const apiContent = read('docs/web/content/sdk/api.html');
        assert.match(apiContent, /href="\/sdk\/api\/\?api=espectre__sdk_8h" data-api-reference-ref="espectre__sdk_8h"/);
        assert.match(apiContent, /href="\/sdk\/api\/\?api=espectre__core__sdk_8h" data-api-reference-ref="espectre__core__sdk_8h"/);
    });

    it('uses the shared SDK path instead of duplicate previous and next cards', () => {
        for (const page of ['architecture', 'api', 'detectors', 'examples']) {
            const content = read(`docs/web/content/sdk/${page}.html`);
            assert.match(content, /data-page-path="sdk"/);
            assert.doesNotMatch(content, /class="article-nav"|doc-link-next/);
        }
    });

    it('publishes the detector architecture through SDK SPA and static routes', () => {
        const detectors = read('docs/web/content/sdk/detectors.html');
        assert.match(detectors, /<h1>/);
        assert.match(detectors, /<h2 id="sdk-detectors-performance"[^>]*>/);
        assert.match(detectors, /DetectionAlgorithm::HIGH_ACCURACY/);
        const detectorGuide = read('docs/web/content/guides/detectors.html');
        assert.match(detectorGuide, /href="\/sdk\/detectors\/"/);
        assert.match(index, /data-page="sdk-detectors"[\s\S]*?data-static-url="\/sdk\/detectors\/"/);
        assert.match(routeRegistry, /name: 'sdk-detectors'.*staticPath: '\/sdk\/detectors\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/sdk\/detectors\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/sdk\/detectors\//);
    });

    it('publishes the detection profile guide through SPA and static routes', () => {
        const guide = read('docs/web/content/guides/detectors.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /<h2 id="detectors-two">/);
        assert.equal((guide.match(/class="profile-comparison-row"/g) || []).length, 5);
        assert.match(guide, /role="tablist"[^>]*aria-label=/);
        assert.equal((guide.match(/data-detector-interface=/g) || []).length, 4);
        assert.match(guide, /id="detectors-native-tab"[^>]*aria-selected="true"/);
        assert.match(guide, /id="detectors-cli"/);
        assert.match(styles, /\.profile-comparison-row \{[\s\S]*?grid-template-columns: minmax\(130px, \.55fr\) repeat\(2, minmax\(0, 1fr\)\);/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.profile-comparison-row \{ grid-template-columns: minmax\(0, 1fr\); gap: 12px; \}/);
        assert.match(index, /data-page="guide-detectors"/);
        assert.match(routeRegistry, /name: 'guide-detectors'.*staticPath: '\/guides\/detectors\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/detectors\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/detectors\//);
    });

    it('publishes the MicroPython contribution and runtime guide', () => {
        const guide = read('docs/web/content/guides/micropython.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /micropython\/micropython\/pull\/18460/);
        assert.match(read('docs/web/content/guides.html'), /micropython-csi-runtime-card\.avif/);
        assert.match(guide, /micropython-csi-runtime-card\.avif/);
        assert.match(index, /data-page="guide-micropython"/);
        assert.match(routeRegistry, /name: 'guide-micropython'.*staticPath: '\/guides\/micropython\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/micropython\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/micropython\//);
    });

    it('publishes the IEEE 802.11bf future sensing guide', () => {
        const guide = read('docs/web/content/guides/future-wifi-sensing.html');
        const guideIndex = read('docs/web/content/guides.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /<h2 id="future-origin">/);
        assert.match(guide, /standards\.ieee\.org\/ieee\/802\.11bf\/11574\//);
        assert.match(guide, /www\.ieee802\.org\/11\/Reports\/tgbf_update\.htm/);
        assert.match(guide, /future-wifi-sensing-card\.avif/);
        assert.match(guideIndex, /href="\/guides\/future-wifi-sensing\/"/);
        assert.match(index, /data-page="guide-future-wifi-sensing"/);
        assert.match(routeRegistry, /name: 'guide-future-wifi-sensing'.*staticPath: '\/guides\/future-wifi-sensing\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/future-wifi-sensing\.html"/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/future-wifi-sensing\//);
    });

    it('publishes the Home Assistant dashboard guide with a distinct cover and collapsible recovery guidance', () => {
        const guide = read('docs/web/content/guides/home-assistant.html');
        const guideIndex = read('docs/web/content/guides.html');
        assert.match(guide, /<h1>/);
        assert.match(guide, /home-assistant-dashboard\.yaml/);
        assert.match(guide, /home-assistant-dashboard-card\.avif/);
        assert.ok(guide.indexOf('id="ha-before"') < guide.indexOf('home-assistant-dashboard-card.avif'));
        assert.ok(guide.indexOf('home-assistant-dashboard-card.avif') < guide.indexOf('id="ha-prefix"'));
        assert.match(guide, /sensor\.espectre_c3_f61093_movement_score/);
        assert.match(guide, /sensor\.espectre_c3_f61093_movement_score_2/);
        assert.match(guide, /<details class="article-details" id="ha-recreate-ids">/);
        assert.match(guide, /homeassistant\/#/);
        assert.match(guideIndex, /href="\/guides\/home-assistant\/"/);
        assert.match(guideIndex, /home-assistant-dashboard-card\.avif/);
        for (const sectionId of ['official-guides', 'external-guides']) {
            assert.match(guideIndex, new RegExp(`<section class="guide-section" aria-labelledby="${sectionId}">`));
            assert.match(guideIndex, new RegExp(`<h2 id="${sectionId}">`));
        }
        assert.equal((guideIndex.match(/class="guides-grid"/g) || []).length, 2);
        assert.ok(guideIndex.indexOf('href="/guides/detection/"') < guideIndex.indexOf('href="/guides/hardware/"'));
        assert.ok(guideIndex.indexOf('href="/guides/placement/"') < guideIndex.indexOf('href="/guides/home-assistant/"'));
        assert.ok(guideIndex.indexOf('href="/guides/home-assistant/"') < guideIndex.indexOf('href="/guides/detectors/"'));
        assert.match(index, /data-page="guide-home-assistant"/);
        assert.match(routeRegistry, /name: 'guide-home-assistant'.*staticPath: '\/guides\/home-assistant\/'/);
        assert.match(read('.github/scripts/build_static_pages.py'), /"source": "content\/guides\/home-assistant\.html"/);
        assert.match(read('.github/scripts/build_sitemap.py'), /"\/guides\/home-assistant\/": \(Path\("docs\/web\/content\/guides\/home-assistant\.html"\), STATIC_PAGE_BUILDER\)/);
        assert.match(read('.github/scripts/sitemap.template.xml'), /https:\/\/espectre\.dev\/guides\/home-assistant\//);
    });

    it('builds complementary side rails from the shared route groups', () => {
        assert.match(read('docs/web/content/sdk.html'), /data-page-toc data-page-path="sdk"/);
        for (const page of ['architecture', 'api', 'detectors', 'examples']) {
            const content = read(`docs/web/content/sdk/${page}.html`);
            assert.match(content, /<article class="article(?: [^"]*)?"[^>]*data-page-toc[^>]*data-page-path="sdk">/);
            assert.doesNotMatch(content, /class="sdk-local-nav"/);
            assert.doesNotMatch(content, /class="page-path"/);
        }
        for (const page of ['detection', 'hardware', 'setup', 'placement', 'home-assistant', 'detectors', 'micropython', 'future-wifi-sensing']) {
            assert.match(read(`docs/web/content/guides/${page}.html`), /data-page-path="guides"/);
        }
        for (const page of ['flash', 'configure', 'monitor', 'raw-csi', 'theremin', 'game']) {
            assert.match(index, new RegExp(`data-page="tool-${page}"[^>]*data-page-path="tools"`));
        }
        assert.match(styles, /@media \(min-width: 1440px\) \{[\s\S]*?\.page-toc \{[\s\S]*?position: fixed;[\s\S]*?left: max\(/);
        assert.match(styles, /@media \(min-width: 1440px\) \{[\s\S]*?\.page-path \{[\s\S]*?position: fixed;[\s\S]*?right: max\([\s\S]*?left: auto;/);
        assert.match(styles, /\.page-path a\[aria-current="page"\]::before/);
        const navigation = read('docs/web/assets/js/navigation.js');
        assert.match(navigation, /function buildPagePath\(container\)/);
        assert.match(navigation, /ESPectreRoutes\?\.membersOf\(group\)/);
        assert.match(navigation, /details\.page-toc, details\.page-path/);
        assert.match(routeRegistry, /membersOf: \(group\) => byGroup\.get\(group\) \|\| emptyGroup/);
    });

    it('uses one cover, practical CLI sections, and the shared path across the official guides', () => {
        const guides = [
            { file: 'detection', cover: 'csi-multipath-room.avif', cli: null, coverAfter: null, coverBefore: 'detection-room', toc: false },
            { file: 'hardware', cover: 'esp32-chip-family-card.avif', cli: null, coverAfter: null, coverBefore: 'hardware-chips', toc: false },
            { file: 'setup', cover: 'flash-connect-usb-card.avif', cli: 'setup-cli', coverAfter: 'setup-requirements', coverBefore: 'setup-firmware', toc: true },
            { file: 'placement', cover: 'sensor-placement-card.avif', cli: null, coverAfter: null, coverBefore: 'placement-link', toc: true },
            { file: 'home-assistant', cover: 'home-assistant-dashboard-card.avif', cli: null, coverAfter: 'ha-before', coverBefore: 'ha-prefix', toc: true },
            { file: 'detectors', cover: 'detection-profiles-card.avif', cli: 'detectors-cli', coverAfter: null, coverBefore: 'detectors-two', toc: true },
            { file: 'micropython', cover: 'micropython-csi-runtime-card.avif', cli: 'micropython-cli', coverAfter: null, coverBefore: 'micropython-upstream', toc: true },
            { file: 'future-wifi-sensing', cover: 'future-wifi-sensing-card.avif', cli: null, coverAfter: null, coverBefore: 'future-origin', toc: true },
        ];
        for (const guide of guides) {
            const path = `docs/web/content/guides/${guide.file}.html`;
            const content = read(path);
            assert.match(content, /<article class="article guide-article" data-page-toc data-page-path="guides">/);
            const firstImage = content.match(/<img\b[^>]*>/)?.[0] || '';
            assert.ok(firstImage.includes(guide.cover), `${guide.file} starts with its guide-card cover`);
            if (guide.coverAfter) {
                assert.ok(content.indexOf(`id="${guide.coverAfter}"`) < content.indexOf(guide.cover), `${guide.file} introduces requirements before its cover`);
            }
            assert.ok(content.indexOf(guide.cover) < content.indexOf(`id="${guide.coverBefore}"`), `${guide.file} positions its cover before the main guide sections`);
            if (guide.toc) {
                assert.match(content, /<details class="page-toc">/);
            }
            if (guide.cli) {
                assert.match(content, new RegExp(`<h[23][^>]*\\bid="${guide.cli}"`), `${guide.file} documents its CLI equivalent`);
            }
            const images = [...content.matchAll(/<img\b[^>]*>/g)].map((match) => match[0]);
            for (const image of images) {
                assert.match(image, /\bwidth="\d+"/);
                assert.match(image, /\bheight="\d+"/);
            }
            assert.doesNotMatch(content, /class="article-nav"|doc-link-next/);
        }
        assert.match(styles, /\.page-toc \{/);
        assert.match(styles, /@media \(min-width: 1440px\) \{[\s\S]*?\.page-toc \{[\s\S]*?position: fixed;[\s\S]*?left: max\(/);
        assert.match(styles, /\.page-toc a\[aria-current="location"\]/);
        assert.match(styles, /@media \(max-width: 720px\) \{[\s\S]*?\.page-toc \{ display: none; \}/);
        assert.doesNotMatch(styles, /\.page-toc, \.page-path \{ display: none; \}/);
        const navigation = read('docs/web/assets/js/navigation.js');
        assert.match(navigation, /matchMedia\('\(max-width: 1439px\)'\)/);
        assert.match(navigation, /function buildPageToc\(container\)/);
        assert.match(navigation, /querySelectorAll\('h2:not\(\[data-toc-exclude\]\)'\)/);
        assert.match(navigation, /heading\.dataset\.tocLabel/);
        assert.match(navigation, /requestAnimationFrame/);
        assert.match(navigation, /const remainingArticleScroll = Math\.max\(0, articleRect\.bottom - window\.innerHeight\);/);
        assert.match(navigation, /Math\.max\(railTop \+ 24, endAwareLine\)/);
        assert.doesNotMatch(navigation, /railFits/);
        assert.match(navigation, /setAttribute\('aria-current', 'location'\)/);
        for (const page of [
            'legal.html',
            'licensing.html',
            'privacy.html',
            'roadmap.html',
            'sdk.html',
            'security.html',
            'terms.html',
            'sdk/api.html',
            'sdk/architecture.html',
            'sdk/detectors.html',
            'sdk/examples.html',
        ]) {
            assert.match(read(`docs/web/content/${page}`), /data-page-toc/, `${page} enables section navigation`);
        }
        assert.match(styles, /\.article-details \{[\s\S]*?border: 1px solid var\(--border\);/);
        assert.match(styles, /\.guide-card img \{[\s\S]*?aspect-ratio: 16 \/ 9;[\s\S]*?object-fit: cover;/);
        assert.doesNotMatch(styles, /\.article-nav/);
        const detectionGuide = read('docs/web/content/guides/detection.html');
        assert.match(detectionGuide, /csi-multipath-room\.avif/);
        assert.match(detectionGuide, /csi-iq-motion\.svg/);
        assert.ok(detectionGuide.indexOf('csi-multipath-room.avif') < detectionGuide.indexOf('id="detection-room"'));
        assert.ok(detectionGuide.indexOf('id="detection-room"') < detectionGuide.indexOf('id="detection-csi"'));
        assert.ok(detectionGuide.indexOf('id="detection-csi"') < detectionGuide.indexOf('id="detection-motion"'));
        assert.ok(detectionGuide.indexOf('id="detection-motion"') < detectionGuide.indexOf('csi-iq-motion.svg'));
    });

});
