/*
 * Browser-only dependency boundary for the ESPectre Web Serial wizard.
 * The Rollup output contains protocol and flashing code, but no third-party UI.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

import { ImprovSerial as BaseImprovSerial } from 'improv-wifi-serial-sdk/dist/serial.js';

export { ESPLoader, Transport } from 'esptool-js';
export { ImprovSerialCurrentState } from 'improv-wifi-serial-sdk/dist/const.js';

const ESPECTRE_IMPROV_GET_MATTER_ONBOARDING = 0x80;

export class ImprovSerial extends BaseImprovSerial {
    async requestMatterOnboarding(timeout) {
        const response = await this._sendRPCWithResponse(
            ESPECTRE_IMPROV_GET_MATTER_ONBOARDING, [], timeout
        );
        return {
            qr: response[0] || '',
            manual: response[1] || '',
        };
    }
}
