/*
 * ESPectre - Web Serial bundle configuration
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const webRoot = fileURLToPath(new URL('.', import.meta.url));

const serialConfig = {
    absWorkingDir: webRoot,
    entryPoints: ['headless-entry.js'],
    outfile: 'build/headless-web-serial.js',
    bundle: true,
    format: 'esm',
    platform: 'browser',
    sourcemap: false,
    // Keep /*! license headers beside the code they cover.
    legalComments: 'inline',
    logLevel: 'warning',
};

export default serialConfig;

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    await build(serialConfig);
}
