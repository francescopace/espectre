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

refreshSelections();
