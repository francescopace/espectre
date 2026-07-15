const MANIFEST_URLS = {
    stable: "/flash/firmware/stable/firmware-manifest-stable.json",
    main: "/flash/firmware/main/firmware-manifest-main.json",
};

const CHANNEL_LABELS = {
    stable: "Stable",
    main: "Main",
};

const state = {
    manifests: {},
    installManifestUrl: null,
};

const frontendSelect = document.getElementById("frontend-select");
const channelSelect = document.getElementById("channel-select");
const chipSelect = document.getElementById("chip-select");
const summaryEl = document.getElementById("flash-summary");
const statusEl = document.getElementById("flash-status");
const installButton = document.getElementById("install-button");
const downloadLink = document.getElementById("download-link");
const matterOnboardingEl = document.getElementById("matter-onboarding");
const matterQrButton = document.getElementById("read-matter-qr");
const matterQrStatus = document.getElementById("matter-qr-status");
const matterQrResult = document.getElementById("matter-qr-result");
const matterQrPayload = document.getElementById("matter-qr-payload");
const matterManualCode = document.getElementById("matter-manual-code");
const matterQrCanvas = document.getElementById("matter-qr-canvas");

function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.classList.remove("is-error", "is-ready");
    if (kind) {
        statusEl.classList.add(kind);
    }
}

function clearInstallManifest() {
    if (state.installManifestUrl) {
        URL.revokeObjectURL(state.installManifestUrl);
        state.installManifestUrl = null;
    }
    installButton.setAttribute("manifest", "");
}

async function loadChannelManifest(channel) {
    if (state.manifests[channel]) {
        return state.manifests[channel];
    }

    const response = await fetch(MANIFEST_URLS[channel], { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`Unable to load the ${channel} firmware manifest.`);
    }

    const manifest = await response.json();
    state.manifests[channel] = manifest;
    return manifest;
}

function createOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
}

function getArtifacts(manifest, frontend) {
    return manifest.frontends?.[frontend]?.artifacts ?? [];
}

function populateChipOptions(artifacts) {
    const chips = [...new Map(artifacts.map((artifact) => [artifact.chip, artifact.chip_label])).entries()];
    chipSelect.innerHTML = "";
    chips.forEach(([chip, label]) => chipSelect.appendChild(createOption(chip, label)));
}

function buildInstallManifest(artifact, manifest) {
    return {
        name: `ESPectre ${artifact.frontend_label || frontendSelect.value.toUpperCase()} ${artifact.chip_label}`,
        version: manifest.version,
        builds: [
            {
                chipFamily: artifact.chip_family,
                parts: [
                    {
                        path: artifact.url,
                        offset: 0,
                    },
                ],
            },
        ],
    };
}

function selectArtifact(manifest) {
    const frontend = frontendSelect.value;
    const chip = chipSelect.value;
    const artifacts = getArtifacts(manifest, frontend).filter((artifact) => artifact.build_type === "factory");
    return artifacts.find((artifact) => artifact.chip === chip);
}

function renderArtifact(manifest, artifact) {
    clearInstallManifest();
    matterOnboardingEl.hidden = frontendSelect.value !== "matter";
    matterQrResult.hidden = true;
    const qrRendererAvailable = typeof window.QRCode === "function";
    matterQrButton.disabled = !qrRendererAvailable;
    if (frontendSelect.value === "matter" && !qrRendererAvailable) {
        matterQrStatus.textContent = "The local QR renderer could not be loaded.";
        matterQrStatus.classList.add("is-error");
    } else {
        matterQrStatus.textContent = "Select the same USB port used for flashing. If no data appears, press the board reset button once.";
        matterQrStatus.classList.remove("is-error");
    }

    if (!artifact) {
        summaryEl.textContent = "No matching firmware was found for the selected combination.";
        setStatus("Change the selection or use the manual download fallback.", "is-error");
        downloadLink.href = "https://github.com/francescopace/espectre/releases";
        downloadLink.textContent = "Browse Releases";
        return;
    }

    const installManifest = buildInstallManifest(artifact, manifest);
    state.installManifestUrl = URL.createObjectURL(
        new Blob([JSON.stringify(installManifest)], { type: "application/json" })
    );
    installButton.setAttribute("manifest", state.installManifestUrl);

    summaryEl.innerHTML = `
        <strong>${artifact.chip_label}</strong><br>
        ${manifest.frontends[frontendSelect.value].label} · ${CHANNEL_LABELS[manifest.channel]}<br>
        Release tag: <code>${manifest.release_tag}</code>
    `;

    downloadLink.href = artifact.url;
    downloadLink.innerHTML = '<i class="fas fa-download"></i> Download Binary';

    if (!("serial" in navigator)) {
        setStatus("This browser does not expose the Web Serial API. Use the download button and flash manually.", "is-error");
        return;
    }

    setStatus("Ready to flash. Connect the board over USB, then use the install button.", "is-ready");
}

function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function resetSerialDevice(port) {
    await port.setSignals({ dataTerminalReady: false, requestToSend: true });
    await delay(100);
    await port.setSignals({ dataTerminalReady: false, requestToSend: false });
}

async function readMatterCodes(port, timeoutMilliseconds = 20000) {
    const reader = port.readable.getReader();
    const decoder = new TextDecoder();
    const deadline = Date.now() + timeoutMilliseconds;
    let input = "";

    try {
        while (Date.now() < deadline) {
            const remaining = deadline - Date.now();
            const result = await Promise.race([
                reader.read(),
                delay(remaining).then(() => ({ timedOut: true })),
            ]);
            if (result.timedOut) break;
            if (result.done) break;
            input += decoder.decode(result.value, { stream: true });
            if (input.length > 16384) input = input.slice(-8192);

            const qrMatch = input.match(/MATTER_QR=(MT:[A-Z0-9.\-]+)/);
            const manualMatch = input.match(/MATTER_MANUAL_CODE=([0-9]+)/);
            if (qrMatch && manualMatch) {
                return { qr: qrMatch[1], manual: manualMatch[1] };
            }
        }
    } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
    }
    throw new Error("Matter codes were not received. Press reset on the board, then try again.");
}

async function renderMatterQr(codes) {
    if (typeof window.QRCode !== "function") {
        throw new Error("The local QR renderer could not be loaded.");
    }
    matterQrCanvas.innerHTML = "";
    new window.QRCode(matterQrCanvas, {
        text: codes.qr,
        width: 240,
        height: 240,
        colorDark: "#000000",
        colorLight: "#ffffff",
        correctLevel: window.QRCode.CorrectLevel.M,
    });
    matterQrPayload.textContent = codes.qr;
    matterManualCode.textContent = codes.manual;
    matterQrResult.hidden = false;
}

async function readMatterQr() {
    if (!("serial" in navigator)) {
        matterQrStatus.textContent = "Web Serial is not available in this browser.";
        matterQrStatus.classList.add("is-error");
        return;
    }

    let port;
    matterQrButton.disabled = true;
    matterQrResult.hidden = true;
    matterQrStatus.classList.remove("is-error");
    matterQrStatus.textContent = "Choose the ESPectre serial port, then wait for the device to restart.";
    try {
        port = await navigator.serial.requestPort();
        await port.open({ baudRate: 115200 });
        await resetSerialDevice(port);
        const codes = await readMatterCodes(port);
        await renderMatterQr(codes);
        matterQrStatus.textContent = "This QR is stored on the device and remains the same after normal updates.";
    } catch (error) {
        matterQrStatus.textContent = error.message || "Unable to read the Matter QR code.";
        matterQrStatus.classList.add("is-error");
    } finally {
        if (port?.readable || port?.writable) {
            await port.close().catch(() => {});
        }
        matterQrButton.disabled = false;
    }
}

async function refreshSelections() {
    setStatus("Loading firmware metadata...");

    try {
        const manifest = await loadChannelManifest(channelSelect.value);
        const frontend = frontendSelect.value;
        const frontendArtifacts = getArtifacts(manifest, frontend).filter((artifact) => artifact.build_type === "factory");

        if (frontendArtifacts.length === 0) {
            chipSelect.innerHTML = "";
            renderArtifact(manifest, null);
            return;
        }

        populateChipOptions(frontendArtifacts);
        if ([...chipSelect.options].some((option) => option.value === chipSelect.dataset.selectedValue)) {
            chipSelect.value = chipSelect.dataset.selectedValue;
        }

        renderArtifact(manifest, selectArtifact(manifest));
    } catch (error) {
        clearInstallManifest();
        summaryEl.textContent = "Firmware metadata is currently unavailable.";
        setStatus(error.message, "is-error");
        downloadLink.href = "https://github.com/francescopace/espectre/releases";
        downloadLink.innerHTML = '<i class="fas fa-download"></i> Browse Releases';
    }
}

frontendSelect.addEventListener("change", () => {
    chipSelect.dataset.selectedValue = "";
    refreshSelections();
});

channelSelect.addEventListener("change", () => {
    chipSelect.dataset.selectedValue = "";
    refreshSelections();
});

chipSelect.addEventListener("change", () => {
    chipSelect.dataset.selectedValue = chipSelect.value;
    refreshSelections();
});

matterQrButton.addEventListener("click", readMatterQr);

refreshSelections();
