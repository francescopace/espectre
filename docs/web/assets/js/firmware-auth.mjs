/*
 * ESPectre - Firmware catalog authentication using the browser Web Crypto API
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

const IDENTITY_FIELDS = ['channel', 'version', 'release_tag', 'commit'];
const ARTIFACT_FIELDS = ['frontend', 'chip', 'chip_family', 'build_type', 'filename', 'size', 'sha256'];

function reject(message) {
    const error = new Error(message);
    error.name = 'FirmwareSignatureError';
    throw error;
}

function decodeBase64(value) {
    if (typeof value !== 'string' || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) {
        reject('Invalid firmware signature encoding.');
    }
    return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

function sameArtifact(left, right) {
    return ARTIFACT_FIELDS.every((field) => left[field] === right[field]);
}

export async function verifyCatalog(manifest, registry, { channel } = {}) {
    if (channel !== undefined && manifest.channel !== channel) reject('The firmware catalog channel does not match.');
    if (!manifest.authentication) {
        reject('The firmware catalog is not signed.');
    }
    const authentication = manifest.authentication;
    if (registry.schema_version !== 1 || !Array.isArray(registry.keys)) reject('Invalid firmware key registry.');
    const trusted = registry.keys.filter((key) => key.id === authentication.key_id && key.algorithm === 'rsa3072');
    if (trusted.length !== 1) reject('The firmware catalog uses an untrusted signing key.');
    const publicKey = await crypto.subtle.importKey(
        'spki', decodeBase64(trusted[0].spki), { name: 'RSA-PSS', hash: 'SHA-256' }, false, ['verify']
    );
    if (publicKey.algorithm.modulusLength !== 3072) reject('Invalid firmware signing key size.');
    const payload = decodeBase64(authentication.payload);
    const valid = await crypto.subtle.verify(
        { name: 'RSA-PSS', saltLength: 32 }, publicKey, decodeBase64(authentication.signature), payload
    );
    if (!valid) reject('The firmware catalog signature is invalid.');
    // Verify the original bytes before parsing; JSON reserialization is not part
    // of the signature contract. URLs may change when CI stages a signed release.
    const signed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(payload));
    if (signed.format !== 'espectre-firmware-v1'
            || !IDENTITY_FIELDS.every((field) => signed[field] === (manifest[field] ?? null))) {
        reject('The firmware catalog identity does not match its signature.');
    }
    const byName = new Map();
    for (const artifact of signed.artifacts) {
        if (byName.has(artifact.filename)) reject('Duplicate signed firmware filename.');
        byName.set(artifact.filename, artifact);
    }
    const retained = new Set();
    for (const [frontend, metadata] of Object.entries(manifest.frontends)) {
        for (const artifact of metadata.artifacts) {
            const expected = byName.get(artifact.filename);
            if (!expected || retained.has(artifact.filename)
                    || !sameArtifact(expected, { ...artifact, frontend })) {
                reject('The firmware metadata does not match the signed catalog.');
            }
            retained.add(artifact.filename);
        }
    }
    return byName;
}

export async function verifyArtifact(inventory, frontend, artifact, bytes) {
    const expected = inventory.get(artifact.filename);
    if (!expected || !sameArtifact(expected, { ...artifact, frontend })
            || !Number.isSafeInteger(expected.size) || expected.size <= 0
            || bytes.byteLength !== expected.size || !/^[a-f0-9]{64}$/.test(expected.sha256)) {
        reject('The downloaded firmware does not match the signed catalog.');
    }
    const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    const actual = Array.from(hash, (byte) => byte.toString(16).padStart(2, '0')).join('');
    if (actual !== expected.sha256) reject('The firmware hash does not match its signature.');
}

export async function authenticateDownload(manifest, frontend, artifacts, fetchBinary, location,
        confirmFailure = () => false) {
    const canConfirm = ['localhost', '127.0.0.1', '[::1]', 'test.espectre.dev'].includes(location.hostname);
    let verificationError;
    const recordFailure = (error) => {
        if (!canConfirm) throw error;
        verificationError ??= error;
    };
    let inventory;
    try {
        let registry = { schema_version: 1, keys: [] };
        if (manifest.authentication) {
            const response = await fetch('/assets/firmware-signing-keys.json', { cache: 'no-store' });
            if (!response.ok) reject('The firmware verification keys could not be loaded.');
            registry = await response.json();
        }
        inventory = await verifyCatalog(manifest, registry);
    } catch (error) {
        recordFailure(error);
    }
    const result = [];
    for (const artifact of artifacts) {
        // Download failures always block installation, even on development hosts.
        const bytes = await fetchBinary(artifact.url);
        if (inventory) {
            try {
                await verifyArtifact(inventory, frontend, artifact, bytes);
            } catch (error) {
                recordFailure(error);
            }
        }
        result.push(bytes);
    }
    // Ask once, after all downloads, and return these exact bytes without retrying.
    if (verificationError && await confirmFailure(verificationError) !== true) throw verificationError;
    return result;
}
