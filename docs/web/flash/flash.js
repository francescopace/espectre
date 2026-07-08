const MANIFEST_URLS = {
    stable: "https://github.com/francescopace/espectre/releases/latest/download/firmware-manifest-stable.json",
    main: "https://github.com/francescopace/espectre/releases/download/snapshot/firmware-manifest-main.json",
};

const CHANNEL_LABELS = {
    stable: "Stable",
    main: "Main",
};

const ALGORITHM_LABELS = {
    classic: "Classic (default)",
    ml: "ML",
};

const state = {
    manifests: {},
    installManifestUrl: null,
};

const frontendSelect = document.getElementById("frontend-select");
const channelSelect = document.getElementById("channel-select");
const chipSelect = document.getElementById("chip-select");
const algorithmField = document.getElementById("algorithm-field");
const algorithmSelect = document.getElementById("algorithm-select");
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

function populateAlgorithmOptions(artifacts) {
    const algorithms = [...new Set(artifacts.map((artifact) => artifact.algorithm).filter(Boolean))];
    const needsAlgorithm = algorithms.length > 1;

    algorithmField.hidden = !needsAlgorithm;
    algorithmSelect.innerHTML = "";

    if (!needsAlgorithm) {
        return;
    }

    algorithms.forEach((algorithm) => {
        algorithmSelect.appendChild(createOption(algorithm, ALGORITHM_LABELS[algorithm] || algorithm.toUpperCase()));
    });
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
    const algorithm = algorithmField.hidden ? null : algorithmSelect.value;
    const artifacts = getArtifacts(manifest, frontend).filter((artifact) => artifact.build_type === "factory");

    return artifacts.find((artifact) => {
        if (artifact.chip !== chip) {
            return false;
        }
        if (!algorithm) {
            return artifact.algorithm === null || artifact.algorithm === "classic";
        }
        return artifact.algorithm === algorithm;
    });
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

    const algorithmLabel = artifact.algorithm ? `, ${ALGORITHM_LABELS[artifact.algorithm] || artifact.algorithm}` : "";
    summaryEl.innerHTML = `
        <strong>${artifact.chip_label}</strong><br>
        ${manifest.frontends[frontendSelect.value].label} · ${CHANNEL_LABELS[manifest.channel]}${algorithmLabel}<br>
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
            algorithmField.hidden = true;
            renderArtifact(manifest, null);
            return;
        }

        populateChipOptions(frontendArtifacts);
        if ([...chipSelect.options].some((option) => option.value === chipSelect.dataset.selectedValue)) {
            chipSelect.value = chipSelect.dataset.selectedValue;
        }

        const chipArtifacts = frontendArtifacts.filter((artifact) => artifact.chip === chipSelect.value);
        populateAlgorithmOptions(chipArtifacts);

        if (!algorithmField.hidden && [...algorithmSelect.options].some((option) => option.value === algorithmSelect.dataset.selectedValue)) {
            algorithmSelect.value = algorithmSelect.dataset.selectedValue;
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
    algorithmSelect.dataset.selectedValue = "";
    refreshSelections();
});

channelSelect.addEventListener("change", () => {
    chipSelect.dataset.selectedValue = "";
    algorithmSelect.dataset.selectedValue = "";
    refreshSelections();
});

chipSelect.addEventListener("change", () => {
    chipSelect.dataset.selectedValue = chipSelect.value;
    algorithmSelect.dataset.selectedValue = "";
    refreshSelections();
});

algorithmSelect.addEventListener("change", () => {
    algorithmSelect.dataset.selectedValue = algorithmSelect.value;
    refreshSelections();
});

refreshSelections();
