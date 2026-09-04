/*
 * ESPectre - Web Serial bundle configuration
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import commonjs from '@rollup/plugin-commonjs';
import json from '@rollup/plugin-json';
import { nodeResolve } from '@rollup/plugin-node-resolve';

export default {
    input: 'headless-entry.js',
    output: {
        file: 'build/headless-web-serial.js',
        format: 'es',
        inlineDynamicImports: true,
        sourcemap: false,
    },
    plugins: [nodeResolve({ browser: true }), commonjs(), json()],
};
