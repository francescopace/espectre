const SERVICE_UUID = 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0';
const TELEMETRY_UUID = '119d5cac-48da-4bd9-bfc3-169805868258';
const SYSINFO_UUID = 'c8c89ffa-c401-461f-9ffc-942fa04adfe3';
const CONTROL_UUID = '33ed9214-a8d7-40e8-82d1-c82747dcdc71';

let device = null;
let telemetryChar = null;
let sysinfoChar = null;
let controlChar = null;
let liveTelemetryRequested = true;
let capabilityState = defaultCapabilities();
let sysinfoSnapshotBuffer = [];
let pendingThresholdValue = null;

const el = (id) => document.getElementById(id);

function defaultCapabilities() {
return {
  supportsWifiProvisioning: true,
  supportsMqttConfig: true,
  supportsDeviceConfig: true,
  supportsRuntimeThreshold: true,
  supportsRuntimeDetector: true,
  supportsLiveTelemetry: true,
  supportsExtendedDiagnostics: true
};
}

function knownFrontendCapabilities(frontend) {
if (frontend === 'streamer') {
  return {
    supportsWifiProvisioning: true,
    supportsMqttConfig: false,
    supportsDeviceConfig: true,
    supportsRuntimeThreshold: false,
    supportsRuntimeDetector: false,
    supportsLiveTelemetry: false,
    supportsExtendedDiagnostics: false
  };
}
if (frontend === 'native') {
  return defaultCapabilities();
}
return null;
}

function log(message) {
const line = `[${new Date().toLocaleTimeString()}] ${message}`;
el('eventLog').textContent += `${line}\n`;
el('eventLog').scrollTop = el('eventLog').scrollHeight;
}

function revealLogs() {
el('logsContent').classList.remove('collapsed');
el('logsArrow').classList.add('rotate');
}

function showValidationError(message) {
revealLogs();
log(`error: ${message}`);
window.alert(message);
}

function parseLeadingNumber(value) {
const match = String(value).trim().match(/-?\d+(?:[.,]\d+)?/);
if (!match) {
  return null;
}
const parsed = Number(match[0].replace(',', '.'));
return Number.isFinite(parsed) ? parsed : null;
}

function formatMotionState(value) {
if (value === 1 || value === '1' || value === 'motion') {
  return 'Motion';
}
if (value === 0 || value === '0' || value === 'idle') {
  return 'Idle';
}
return String(value);
}

function updateStateCard(value) {
const stateCard = el('stateCard');
const label = formatMotionState(value);
el('stateValue').textContent = label;
stateCard.classList.remove('idle', 'motion');
if (label === 'Idle') {
  stateCard.classList.add('idle');
} else if (label === 'Motion') {
  stateCard.classList.add('motion');
}
}

function setThresholdPending(value = null) {
pendingThresholdValue = Number.isFinite(value) ? value : null;
el('thresholdCard').classList.toggle('pending', pendingThresholdValue !== null);
}

function setConnected(connected) {
el('statusText').textContent = connected ? 'Connected' : 'Disconnected';
el('statusIndicator').className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
el('connectBtn').textContent = connected ? 'Disconnect' : 'Connect';
el('connectBtn').classList.toggle('btn-primary', !connected);
el('connectBtn').classList.toggle('btn-danger', connected);
ToolPage.setHeaderConnectionStatus(connected);
el('sysinfoBtn').disabled = !connected;
sysinfoSnapshotBuffer = [];
if (!connected) {
  capabilityState = defaultCapabilities();
  setThresholdPending(null);
}
applyCapabilityState(connected);
if (connected) {
  el('configContent').classList.add('collapsed');
  el('configArrow').classList.add('rotate');
}
}

function liveTelemetryEnabled() {
return liveTelemetryRequested;
}

function updateThresholdValue(value, force = false) {
const threshold = parseDecimalInput(value);
if (threshold === null) {
  return null;
}
if (pendingThresholdValue !== null && Math.abs(threshold - pendingThresholdValue) <= 1e-4) {
  setThresholdPending(null);
}
const input = el('thresholdValue');
if (!force && document.activeElement === input) {
  return threshold;
}
input.value = threshold.toFixed(6);
return threshold;
}

function updateDetectorValue(value) {
if (value !== 'classic' && value !== 'ml') {
  return;
}
el('detectorValue').value = value;
el('thresholdValue').max = value === 'ml' ? '1' : '10';
}

function updateMotionLevel(movement, threshold) {
const movementValue = Number(movement);
const thresholdValue = Number(threshold);
if (!Number.isFinite(movementValue) || !Number.isFinite(thresholdValue) || thresholdValue <= 0) {
  el('movementValue').textContent = '-';
  return;
}
el('movementValue').textContent = `${((movementValue / thresholdValue) * 100).toFixed(1)}%`;
}

function updateTrafficDiagnostic() {
const mode = el('diagTraffic').dataset.mode || '';
const rate = el('diagTraffic').dataset.rate || '';
if (!mode && !rate) {
  el('diagTraffic').textContent = '-';
  return;
}
if (mode && rate) {
  el('diagTraffic').textContent = `${mode} ${rate} pps`;
  return;
}
el('diagTraffic').textContent = mode || `${rate} pps`;
}

function updateLiveTelemetryButton() {
const button = el('liveTelemetryBtn');
const state = el('liveTelemetryState');
button.classList.toggle('enabled', liveTelemetryEnabled());
button.classList.toggle('disabled', !liveTelemetryEnabled());
state.textContent = liveTelemetryEnabled() ? 'Enabled' : 'Disabled';
}

function setCapability(name, supported) {
capabilityState[name] = supported;
}

function setCardVisible(cardId, visible) {
el(cardId).classList.toggle('hidden', !visible);
}

function applyFrontendCapabilities(frontend) {
const frontendCapabilities = knownFrontendCapabilities(frontend);
if (!frontendCapabilities) {
  return;
}
capabilityState = {
  ...capabilityState,
  ...frontendCapabilities
};
}

function applyCapabilityState(connected) {
const isConnected = Boolean(connected);
setCardVisible('configurationCard', isConnected);
setCardVisible('logsCard', isConnected);
setCardVisible('wifiProvisioningCard', capabilityState.supportsWifiProvisioning);
setCardVisible('mqttSettingsCard', capabilityState.supportsMqttConfig);
setCardVisible('deviceSettingsCard', capabilityState.supportsDeviceConfig);
setCardVisible('telemetryCard', isConnected && capabilityState.supportsLiveTelemetry);
setCardVisible('diagWifiItem', capabilityState.supportsWifiProvisioning);
setCardVisible('diagMqttItem', capabilityState.supportsMqttConfig);
setCardVisible('diagDetectorItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagWindowItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagLowpassItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagHampelItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagTrafficItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagPublishItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagEvaluationItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagMotionHitsItem', capabilityState.supportsExtendedDiagnostics);
setCardVisible('diagStartupThresholdItem', capabilityState.supportsExtendedDiagnostics);

el('wifiApplyBtn').disabled = !isConnected || !capabilityState.supportsWifiProvisioning;
el('wifiClearBtn').disabled = !isConnected || !capabilityState.supportsWifiProvisioning;
el('mqttApplyBtn').disabled = !isConnected || !capabilityState.supportsMqttConfig;
el('mqttClearBtn').disabled = !isConnected || !capabilityState.supportsMqttConfig;
el('deviceApplyBtn').disabled = !isConnected || !capabilityState.supportsDeviceConfig;
el('deviceClearBtn').disabled = !isConnected || !capabilityState.supportsDeviceConfig;
el('thresholdValue').disabled = !isConnected || !capabilityState.supportsRuntimeThreshold;
setCardVisible('detectorCard', isConnected && capabilityState.supportsRuntimeDetector);
el('detectorValue').disabled = !isConnected || !capabilityState.supportsRuntimeDetector;
el('liveTelemetryBtn').disabled = !capabilityState.supportsLiveTelemetry;
updateLiveTelemetryButton();
}

async function applyLiveTelemetryPreference() {
if (!telemetryChar || !capabilityState.supportsLiveTelemetry) {
  return;
}
if (liveTelemetryEnabled()) {
  await telemetryChar.startNotifications();
  log('Telemetry notifications enabled');
} else {
  await telemetryChar.stopNotifications();
  log('Telemetry notifications disabled');
}
}

function appendSysinfoLog(line) {
el('sysinfoLog').textContent += `${line}\n`;
el('sysinfoLog').scrollTop = el('sysinfoLog').scrollHeight;
}

function applySysinfoLine(line) {
const idx = line.indexOf('=');
if (idx <= 0) {
  return;
}
const key = line.slice(0, idx);
const value = line.slice(idx + 1);
if (key === 'frontend') {
  applyFrontendCapabilities(value);
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_wifi_provisioning') {
  setCapability('supportsWifiProvisioning', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_mqtt_config') {
  setCapability('supportsMqttConfig', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_device_config') {
  setCapability('supportsDeviceConfig', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_runtime_threshold') {
  setCapability('supportsRuntimeThreshold', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_runtime_detector') {
  setCapability('supportsRuntimeDetector', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_live_telemetry') {
  setCapability('supportsLiveTelemetry', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'supports_extended_diagnostics') {
  setCapability('supportsExtendedDiagnostics', value === 'true' || value === '1');
  applyCapabilityState(device && device.gatt && device.gatt.connected);
} else if (key === 'chip') {
  el('diagChip').textContent = value.toUpperCase();
} else if (key === 'device_id') {
  el('deviceIdInput').value = value;
} else if (key === 'device_label') {
  el('deviceLabelInput').value = value;
} else if (key === 'device_name') {
  el('deviceNameInput').value = value;
} else if (key === 'mqtt_host') {
  el('mqttHostInput').value = value;
} else if (key === 'mqtt_port') {
  el('mqttPortInput').value = value;
} else if (key === 'mqtt_username') {
  el('mqttUsernameInput').value = value;
} else if (key === 'topic_prefix') {
  el('topicPrefixInput').value = value;
} else if (key === 'wifi_ssid') {
  el('wifiSsidInput').value = value;
} else if (key === 'wifi_bssid') {
  el('wifiBssidInput').value = value;
} else if (key === 'wifi_channel') {
  el('wifiChannelInput').value = value || '0';
} else if (key === 'threshold') {
  const threshold = parseLeadingNumber(value);
  if (threshold !== null) {
    updateThresholdValue(threshold);
  } else {
    el('thresholdValue').value = value;
  }
}

if (key === 'proto_version' || key === 'espectre_protocol_version') {
  el('diagProtocol').textContent = key === 'espectre_protocol_version' ? value : value;
} else if (key === 'detector') {
  el('diagDetector').textContent = value;
  updateDetectorValue(value);
} else if (key === 'window') {
  el('diagWindow').textContent = value;
} else if (key === 'lowpass') {
  el('diagLowpass').textContent = value;
} else if (key === 'lowpass_cutoff') {
  el('diagLowpass').textContent = `${el('diagLowpass').textContent} ${value} Hz`.trim();
} else if (key === 'hampel') {
  el('diagHampel').textContent = value;
} else if (key === 'hampel_window') {
  el('diagHampel').textContent = `${el('diagHampel').textContent} w=${value}`.trim();
} else if (key === 'hampel_threshold') {
  el('diagHampel').textContent = `${el('diagHampel').textContent} t=${value}`.trim();
} else if (key === 'traffic_mode') {
  el('diagTraffic').dataset.mode = value;
  updateTrafficDiagnostic();
} else if (key === 'traffic_rate') {
  el('diagTraffic').dataset.rate = value;
  updateTrafficDiagnostic();
} else if (key === 'publish_interval') {
  el('diagPublishInterval').textContent = `every ${value} pkts`;
} else if (key === 'evaluation_interval') {
  el('diagEvaluationInterval').textContent = `every ${value} pkts`;
} else if (key === 'wifi_connected') {
  el('diagWifi').textContent = value === 'true' || value === '1' ? 'connected' : 'disconnected';
} else if (key === 'mqtt_connected') {
  el('diagMqtt').textContent = value === 'true' || value === '1' ? 'connected' : 'disconnected';
} else if (key === 'motion_hits') {
  el('diagMotionHits').textContent = value;
} else if (key === 'startup_threshold') {
  el('diagStartupThreshold').textContent = value;
}
}

function appendSysinfo(line) {
appendSysinfoLog(line);
if (!line) {
  return;
}
if (line === 'END') {
  const snapshot = sysinfoSnapshotBuffer;
  sysinfoSnapshotBuffer = [];
  snapshot.forEach(applySysinfoLine);
  return;
}
sysinfoSnapshotBuffer.push(line);
}

async function writeControl(command) {
const payload = new TextEncoder().encode(command);
if (controlChar.writeValueWithResponse) {
  await controlChar.writeValueWithResponse(payload);
} else {
  await controlChar.writeValue(payload);
}
const sensitive = command.startsWith('SET_WIFI_CONFIG:') || command.startsWith('SET_MQTT_CONFIG:');
log(`control -> ${sensitive ? command.split(':', 1)[0] + ':[redacted]' : command}`);
}

function parseDecimalInput(value) {
const parsed = Number(String(value).trim().replace(',', '.'));
return Number.isFinite(parsed) ? parsed : null;
}

function parsePortInput(value) {
const port = Number(String(value).trim());
return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

function hasNonEmptyValue(value) {
return String(value).trim().length > 0;
}

function buildEncodedCommand(prefix, fields) {
const payload = Object.entries(fields).map(([key, value]) => {
  return `${encodeURIComponent(key)}=${encodeURIComponent(String(value ?? ''))}`;
}).join('&');
return `${prefix}${payload}`;
}

function logMissingRequiredFields(sectionLabel, fieldLabels) {
const missingFields = fieldLabels.join(', ');
const suffix = fieldLabels.length === 1 ? 'field is' : 'fields are';
showValidationError(`${sectionLabel} required ${suffix} missing: ${missingFields}`);
}

async function connect() {
if (!navigator.bluetooth) {
  log('Web Bluetooth is not available in this browser.');
  return;
}

log('Opening BLE device picker...');
device = await navigator.bluetooth.requestDevice({
  filters: [{ services: [SERVICE_UUID] }],
  optionalServices: [SERVICE_UUID]
});
device.addEventListener('gattserverdisconnected', () => {
  log('BLE device disconnected');
  setConnected(false);
});

log(`Connecting to ${device.name || device.id}...`);
const server = await device.gatt.connect();
const service = await server.getPrimaryService(SERVICE_UUID);
telemetryChar = await service.getCharacteristic(TELEMETRY_UUID);
sysinfoChar = await service.getCharacteristic(SYSINFO_UUID);
controlChar = await service.getCharacteristic(CONTROL_UUID);

telemetryChar.addEventListener('characteristicvaluechanged', (event) => {
  const dv = event.target.value;
  if (!dv || dv.byteLength < 8) {
    log(`telemetry ignored: ${dv ? dv.byteLength : 0} bytes`);
    return;
  }
  const movement = dv.getFloat32(0, true);
  const threshold = dv.getFloat32(4, true);
  const motionState = dv.byteLength >= 9 ? dv.getUint8(8) : null;
  updateMotionLevel(movement, threshold);
  if (document.activeElement !== el('thresholdValue')) {
    el('thresholdValue').value = Number.isFinite(threshold) ? threshold.toFixed(6) : '';
  }
  if (motionState !== null) {
    updateStateCard(motionState);
  }
});

sysinfoChar.addEventListener('characteristicvaluechanged', (event) => {
  const value = event.target.value;
  const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  const line = new TextDecoder().decode(bytes).trim();
  appendSysinfo(line);
});

if (liveTelemetryEnabled() && capabilityState.supportsLiveTelemetry) {
  await telemetryChar.startNotifications();
  log('Telemetry notifications enabled');
} else {
  log('Telemetry notifications disabled from UI');
}
await sysinfoChar.startNotifications();
log('System info notifications enabled');
setConnected(true);
await writeControl('REQ_SYSINFO');
}

async function disconnect() {
if (device && device.gatt && device.gatt.connected) {
  device.gatt.disconnect();
}
setConnected(false);
}

el('connectBtn').addEventListener('click', async () => {
try {
  if (device && device.gatt && device.gatt.connected) {
    await disconnect();
  } else {
    await connect();
  }
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('sysinfoBtn').addEventListener('click', async () => {
try {
  await writeControl('REQ_SYSINFO');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('liveTelemetryBtn').addEventListener('click', async () => {
try {
  liveTelemetryRequested = !liveTelemetryRequested;
  updateLiveTelemetryButton();
  if (device && device.gatt && device.gatt.connected) {
    await applyLiveTelemetryPreference();
  } else {
    log(liveTelemetryEnabled() ? 'Live BLE telemetry will be enabled on connect' : 'Live BLE telemetry will stay disabled on connect');
  }
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('deviceApplyBtn').addEventListener('click', async () => {
try {
  const deviceLabel = el('deviceLabelInput').value.trim();
  await writeControl(`SET_DEVICE_CONFIG:device_label=${deviceLabel}`);
  await writeControl('REQ_SYSINFO');
  log('Device settings saved');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('deviceClearBtn').addEventListener('click', async () => {
try {
  await writeControl('CLEAR_DEVICE_CONFIG');
  await writeControl('REQ_SYSINFO');
  log('Device settings cleared');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('thresholdValue').addEventListener('change', async () => {
try {
  const threshold = updateThresholdValue(el('thresholdValue').value, true);
  if (threshold === null) {
    showValidationError('invalid threshold value');
    return;
  }
  setThresholdPending(threshold);
  await writeControl(`SET_THRESHOLD:${threshold.toFixed(6)}`);
} catch (error) {
  setThresholdPending(null);
  log(`error: ${error.message}`);
}
});

el('detectorValue').addEventListener('change', async () => {
try {
  const detector = el('detectorValue').value;
  await writeControl(`SET_DETECTOR:${detector}`);
  await writeControl('REQ_SYSINFO');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('wifiApplyBtn').addEventListener('click', async () => {
try {
  const ssid = el('wifiSsidInput').value.trim();
  const password = el('wifiPasswordInput').value;
  const channel = Number(el('wifiChannelInput').value);
  const missingFields = [];

  if (!ssid) {
    missingFields.push('SSID');
  }
  if (!hasNonEmptyValue(password)) {
    missingFields.push('password');
  }
  if (missingFields.length > 0) {
    logMissingRequiredFields('Wi-Fi', missingFields);
    return;
  }
  if (!Number.isInteger(channel) || channel < 0 || channel > 14) {
    showValidationError('channel must be 0..14');
    return;
  }
  const bssid = el('wifiBssidInput').value.trim();
  if (bssid && !/^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$/.test(bssid)) {
    showValidationError('BSSID must match aa:bb:cc:dd:ee:ff');
    return;
  }
  await writeControl(buildEncodedCommand('SET_WIFI_CONFIG:', {
    ssid,
    password,
    bssid,
    channel
  }));
  el('wifiPasswordInput').value = '';
  log('Wi-Fi credentials saved; station reconnecting');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('wifiClearBtn').addEventListener('click', async () => {
try {
  await writeControl('CLEAR_WIFI');
  el('wifiSsidInput').value = '';
  el('wifiPasswordInput').value = '';
  el('wifiPasswordInput').placeholder = 'Password';
  el('wifiBssidInput').value = '';
  el('wifiChannelInput').value = '0';
  log('Wi-Fi credentials cleared; station disconnected');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('mqttApplyBtn').addEventListener('click', async () => {
try {
  const mqttHost = el('mqttHostInput').value.trim();
  const mqttPortRaw = el('mqttPortInput').value;
  const mqttPort = parsePortInput(mqttPortRaw);
  const mqttUsername = el('mqttUsernameInput').value.trim();
  const mqttPassword = el('mqttPasswordInput').value;
  const topicPrefix = el('topicPrefixInput').value.trim() || 'espectre/v1/devices';
  const missingFields = [];

  if (!mqttHost) {
    missingFields.push('broker host');
  }
  if (!hasNonEmptyValue(mqttPortRaw)) {
    missingFields.push('broker port');
  }
  if (!mqttUsername) {
    missingFields.push('username');
  }
  if (!hasNonEmptyValue(mqttPassword)) {
    missingFields.push('password');
  }
  if (missingFields.length > 0) {
    logMissingRequiredFields('MQTT', missingFields);
    return;
  }
  if (mqttPort === null) {
    showValidationError('MQTT port must be 1..65535');
    return;
  }

  await writeControl(buildEncodedCommand('SET_MQTT_CONFIG:', {
    host: mqttHost,
    port: mqttPort,
    username: mqttUsername,
    password: mqttPassword,
    topic_prefix: topicPrefix
  }));
  el('mqttPasswordInput').value = '';
  log('MQTT settings saved');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('mqttClearBtn').addEventListener('click', async () => {
try {
  await writeControl('CLEAR_MQTT_CONFIG');
  el('mqttPasswordInput').value = '';
  log('MQTT settings cleared');
} catch (error) {
  log(`error: ${error.message}`);
}
});

el('clearBtn').addEventListener('click', () => {
el('stateValue').textContent = '-';
el('stateCard').classList.remove('idle', 'motion');
el('movementValue').textContent = '-';
el('thresholdValue').value = '';
setThresholdPending(null);
sysinfoSnapshotBuffer = [];
el('sysinfoLog').textContent = '';
el('eventLog').textContent = '';
});

applyCapabilityState(false);
updateLiveTelemetryButton();
