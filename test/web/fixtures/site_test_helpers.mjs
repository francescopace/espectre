/*
 * ESPectre - Shared website structural test fixtures
 *
 * Copyright 2026 Francesco Pace <francescopace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { readFileSync } from 'node:fs';

export const read = (path) => readFileSync(new URL(`../../../${path}`, import.meta.url), 'utf8');

export const index = read('docs/web/index.html');
export const app = [
    'device-session.js',
    'direct-discovery.js',
    'configure-tool.js',
    'monitor-tool.js',
    'csi-tool.js',
    'game-tool.js',
    'theremin-tool.js',
    'app.js',
].map((name) => read(`docs/web/assets/js/${name}`)).join('\n');
export const directProtocol = read('docs/web/assets/js/espectre-direct.js');
export const browserSupportSource = read('docs/web/assets/js/browser-support.js');
export const routeBootstrap = read('docs/web/assets/js/route-bootstrap.js');
export const routeRegistry = read('docs/web/assets/js/route-registry.js');
export const routeManifest = JSON.parse(read('docs/web/routes.json'));
export const styles = read('docs/web/assets/css/styles.css');
export const security = read('docs/web/content/security.html');
export const toolsContent = read('docs/web/content/tools.html');
export const toolContent = Object.fromEntries(
    ['flash', 'configure', 'monitor', 'raw-csi', 'theremin', 'game']
        .map((slug) => [
            slug,
            read(`docs/web/content/tools/${{
                configure: 'device-settings',
                'raw-csi': 'csi-visualizer',
            }[slug] || slug}.html`)
        ])
);
export const toolFragments = Object.values(toolContent).join('\n');
export const roadmapContent = read('docs/web/content/roadmap.html');
export const GPL_HTML_HEADER = `<!--
  SPDX-License-Identifier: GPL-3.0-only
  Commercial licensing available under separate agreement; see LICENSING.md.
-->
`;
