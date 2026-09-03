/*
 * ESPectre - Website legal route contracts
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { index, read, routeManifest } from './fixtures/site_test_helpers.mjs';

const route = (name) => routeManifest.routes.find((candidate) => candidate.name === name);

describe('website legal route contracts', () => {
    it('publishes every footer route in the SPA and static footer', () => {
        const notFound = read('docs/web/404.html');
        for (const name of routeManifest.navigation.footer) {
            const definition = route(name);
            assert.ok(definition);
            assert.match(index, new RegExp(`data-page="${name}"`));
            assert.match(index, new RegExp(`href="${definition.staticPath}"`));
            assert.match(notFound, new RegExp(`href="${definition.staticPath}"`));
            const content = read(`docs/web/content/${name}.html`);
            assert.match(content, /<h1 class="page-title">/);
        }
    });

    it('serves a static 404 page without client-side redirection', () => {
        const notFound = read('docs/web/404.html');
        assert.doesNotMatch(notFound, /http-equiv="refresh"|location\.replace/);
        assert.match(notFound, /data-static-page data-site-section="other"/);
        assert.match(notFound, /<footer class="site-footer">/);
        assert.match(notFound, /class="consent-banner js-consent-banner"/);
    });

    it('publishes the documented contact and security destinations', () => {
        const contact = read('docs/web/content/contact.html');
        const security = read('docs/web/content/security.html');
        const licensing = read('docs/web/content/licensing.html');
        assert.match(contact, /mailto:contact@espectre\.dev/);
        assert.match(contact, /github\.com\/francescopace\/espectre\/discussions/);
        assert.match(contact, /github\.com\/francescopace\/espectre\/issues/);
        assert.match(security, /mailto:security@espectre\.dev/);
        assert.match(security, /github\.com\/francescopace\/espectre\/security/);
        assert.match(licensing, /mailto:contact@espectre\.dev\?subject=Commercial%20licensing%20inquiry/);
    });

    it('publishes privacy cookie settings as a stable anchor', () => {
        assert.match(read('docs/web/content/privacy.html'), /id="cookie-settings"/);
        assert.match(read('docs/web/404.html'), /href="\/privacy\/#cookie-settings" class="js-cookie-settings"/);
    });
});
