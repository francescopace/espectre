const TRACKED_CONFIGURE_ACTIONS = new Set([
  'SET_DEVICE_CONFIG', 'CLEAR_DEVICE_CONFIG', 'SET_THRESHOLD', 'SET_DETECTOR',
  'SET_WIFI_CONFIG', 'CLEAR_WIFI', 'SET_MQTT_CONFIG', 'CLEAR_MQTT_CONFIG'
]);
const SYSINFO_CAPABILITY_KEYS = new Set([
  'supports_wifi_provisioning',
  'supports_mqtt_config',
  'supports_device_config',
  'supports_runtime_threshold',
  'supports_runtime_detector',
  'supports_live_telemetry',
  'supports_extended_diagnostics'
]);

let liveTelemetryRequested = true;
let capabilityState = defaultCapabilities();
let capabilitiesResolved = false;
let advertisedCapabilityKeys = new Set();
let analyticsDeviceInfo = {};
let lastTrackedDeviceProfile = null;
let sysinfoSnapshotActive = false;

const ble = new ESPectreBleClient({
  onTelemetry: handleTelemetry,
  onInvalidTelemetry: (byteLength) => log(`telemetry ignored: ${byteLength} bytes`),
  onSysinfoLine: appendSysinfo,
  onSysinfoSnapshot: handleSysinfoSnapshot,
  onDisconnected: () => {
    log('BLE device disconnected');
    setConnected(false);
  }
});

const movementBar = new ESPectreMovementBar({
  root: document.getElementById('movement-bar-vertical'),
  scaleMax: 1,
  thresholdMin: 0,
  thresholdMax: 1,
  onThresholdCommit: async (threshold) => {
    await writeControl(`SET_THRESHOLD:${threshold.toFixed(6)}`);
  }
});
movementBar.setVisible(false);

const el = (id) => document.getElementById(id);

function defaultCapabilities() {
return {
  supportsWifiProvisioning: false,
  supportsMqttConfig: false,
  supportsDeviceConfig: false,
  supportsRuntimeThreshold: false,
  supportsRuntimeDetector: false,
  supportsLiveTelemetry: false,
  supportsExtendedDiagnostics: false
};
}

function log(message) {
const line = `[${new Date().toLocaleTimeString()}] ${message}`;
el('eventLog').textContent += `${line}\n`;
el('eventLog').scrollTop = el('eventLog').scrollHeight;
}

function revealLogs() {
el('logsContent').classList.remove('collapsed');
el('logsArrow').classList.remove('rotate');
const trigger = el('logsArrow') && el('logsArrow').closest('.collapsible-header');
if (trigger) trigger.setAttribute('aria-expanded', 'true');
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

function showConnectionStatus(message, type = '') {
  const status = el('connectionStatus');
  if (!status) return;
  status.textContent = message;
  status.className = `connection-status ${type}`.trim();
}

function updateConnectionUi(connected) {
  const connectionPre = el('connectionPre');
  const connectionReady = el('connectionReady');
  const workspace = el('configureWorkspace');
  if (connectionPre) connectionPre.hidden = connected;
  if (connectionReady) connectionReady.hidden = !connected;
  if (workspace) workspace.hidden = !connected;
  document.body.classList.toggle('configure-connected', connected);
  ToolPage.setHeaderConnectionStatus(connected);

  if (connected) {
    const label = el('deviceName');
    if (label) label.textContent = `${ble.name || 'ESP32'} Connected`;
    showConnectionStatus('', '');
  } else {
    showConnectionStatus('', '');
    movementBar.setMovement(0);
    movementBar.setVisible(false);
  }
}

function setConnected(connected) {
  const connectBtn = el('connectBtn');
  if (connectBtn) connectBtn.disabled = false;
  updateConnectionUi(connected);
  el('sysinfoBtn').disabled = !connected;
  if (!connected) {
    capabilityState = defaultCapabilities();
    capabilitiesResolved = false;
    advertisedCapabilityKeys = new Set();
  }
  applyCapabilityState(connected);
}

function liveTelemetryEnabled() {
return liveTelemetryRequested;
}

function updateDetectorValue(value) {
if (value !== 'classic' && value !== 'ml') {
  return;
}
el('detectorValue').value = value;
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
state.textContent = liveTelemetryEnabled() ? 'On' : 'Off';
}

function setCapability(name, supported) {
capabilityState[name] = supported;
}

function setCardVisible(cardId, visible) {
const node = el(cardId);
if (!node) return;
node.hidden = !visible;
}

function applyCapabilityState(connected) {
const isConnected = Boolean(connected);
const hasConfiguration = capabilitiesResolved && (
  capabilityState.supportsWifiProvisioning ||
  capabilityState.supportsMqttConfig ||
  capabilityState.supportsDeviceConfig ||
  capabilityState.supportsExtendedDiagnostics
);
const hasRuntimeToolbar = capabilitiesResolved && (
  capabilityState.supportsRuntimeDetector ||
  capabilityState.supportsLiveTelemetry
);
const showMovementBar = isConnected && capabilitiesResolved && (
  capabilityState.supportsLiveTelemetry ||
  capabilityState.supportsRuntimeThreshold
);
const capabilityNotice = el('capabilityNotice');
if (!isConnected || hasConfiguration || hasRuntimeToolbar || showMovementBar) {
  capabilityNotice.hidden = true;
} else {
  capabilityNotice.hidden = false;
  capabilityNotice.textContent = capabilitiesResolved
    ? (advertisedCapabilityKeys.size === 0
        ? 'This firmware did not report BLE capability flags. Update the device firmware and reconnect.'
        : 'This device does not expose configurable BLE features.')
    : 'Reading supported BLE features from the device...';
}
setCardVisible('configurationCard', isConnected && hasConfiguration);
setCardVisible('logsCard', isConnected);
setCardVisible('wifiProvisioningCard', isConnected && capabilitiesResolved && capabilityState.supportsWifiProvisioning);
setCardVisible('mqttSettingsCard', isConnected && capabilitiesResolved && capabilityState.supportsMqttConfig);
setCardVisible('deviceSettingsCard', isConnected && capabilitiesResolved && capabilityState.supportsDeviceConfig);
setCardVisible('diagnosticsCard', isConnected && capabilitiesResolved && capabilityState.supportsExtendedDiagnostics);
setCardVisible('runtimeToolbar', isConnected && hasRuntimeToolbar);
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
setCardVisible('detectorField', isConnected && capabilityState.supportsRuntimeDetector);
setCardVisible('liveTelemetryBtn', capabilityState.supportsLiveTelemetry);
el('detectorValue').disabled = !isConnected || !capabilityState.supportsRuntimeDetector;
el('liveTelemetryBtn').disabled = !capabilityState.supportsLiveTelemetry;
movementBar.setVisible(showMovementBar);
movementBar.setInteractive(isConnected && capabilityState.supportsRuntimeThreshold);
updateLiveTelemetryButton();
}

async function applyLiveTelemetryPreference() {
if (!ble.connected || !capabilityState.supportsLiveTelemetry) {
  return;
}
await ble.setTelemetryNotifications(liveTelemetryEnabled());
log(liveTelemetryEnabled() ? 'Telemetry notifications enabled' : 'Telemetry notifications disabled');
}

function appendSysinfoLog(line) {
el('sysinfoLog').textContent += `${line}\n`;
el('sysinfoLog').scrollTop = el('sysinfoLog').scrollHeight;
}

function resetSysinfoSnapshot() {
sysinfoSnapshotActive = false;
capabilityState = defaultCapabilities();
capabilitiesResolved = false;
advertisedCapabilityKeys = new Set();
applyCapabilityState(ble.connected);
}

function handleTelemetry(telemetry) {
  if (Number.isFinite(telemetry.movement)) {
    movementBar.setMovement(telemetry.movement);
  }
  if (Number.isFinite(telemetry.threshold) && !movementBar.isDragging) {
    movementBar.setThreshold(telemetry.threshold);
  }
}

function handleSysinfoSnapshot() {
sysinfoSnapshotActive = false;
capabilitiesResolved = true;
applyCapabilityState(ble.connected);
}

function applySysinfoLine(line) {
const idx = line.indexOf('=');
if (idx <= 0) {
  return;
}
const key = line.slice(0, idx);
const value = line.slice(idx + 1);
if (key === 'frontend' || key === 'chip') {
  analyticsDeviceInfo[key] = value.toLowerCase();
  if (analyticsDeviceInfo.frontend && analyticsDeviceInfo.chip) {
    const profile = `${analyticsDeviceInfo.frontend}:${analyticsDeviceInfo.chip}`;
    if (profile !== lastTrackedDeviceProfile) {
      trackEvent('device_profile', {
        tool_name: 'configure',
        frontend: analyticsDeviceInfo.frontend,
        chip: analyticsDeviceInfo.chip
      });
      lastTrackedDeviceProfile = profile;
    }
  }
}
if (key === 'supports_wifi_provisioning') {
  setCapability('supportsWifiProvisioning', value === 'true' || value === '1');
} else if (key === 'supports_mqtt_config') {
  setCapability('supportsMqttConfig', value === 'true' || value === '1');
} else if (key === 'supports_device_config') {
  setCapability('supportsDeviceConfig', value === 'true' || value === '1');
} else if (key === 'supports_runtime_threshold') {
  setCapability('supportsRuntimeThreshold', value === 'true' || value === '1');
} else if (key === 'supports_runtime_detector') {
  setCapability('supportsRuntimeDetector', value === 'true' || value === '1');
} else if (key === 'supports_live_telemetry') {
  setCapability('supportsLiveTelemetry', value === 'true' || value === '1');
} else if (key === 'supports_extended_diagnostics') {
  setCapability('supportsExtendedDiagnostics', value === 'true' || value === '1');
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
    movementBar.setThreshold(threshold);
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
if (!line || line === 'END') {
  return;
}

if (!sysinfoSnapshotActive || line.startsWith('proto_version=')) {
  resetSysinfoSnapshot();
  sysinfoSnapshotActive = true;
}
applySysinfoLine(line);

const separatorIndex = line.indexOf('=');
const key = separatorIndex > 0 ? line.slice(0, separatorIndex) : '';
if (SYSINFO_CAPABILITY_KEYS.has(key)) {
  advertisedCapabilityKeys.add(key);
  capabilitiesResolved = true;
  applyCapabilityState(ble.connected);
  if (ble.connected && key === 'supports_live_telemetry') {
    applyLiveTelemetryPreference().catch((error) => log(`error: ${error.message}`));
  }
}
}

async function writeControl(command) {
if (command === 'REQ_SYSINFO') {
  resetSysinfoSnapshot();
}
const action = command.split(':', 1)[0];
try {
  await ble.writeControl(command);
} catch (error) {
  if (TRACKED_CONFIGURE_ACTIONS.has(action)) {
    trackEvent('configure_change', {
      action,
      result: 'failure',
      error_type: error.name || 'Error'
    });
  }
  throw error;
}
const sensitive = command.startsWith('SET_WIFI_CONFIG:') || command.startsWith('SET_MQTT_CONFIG:');
log(`control -> ${sensitive ? command.split(':', 1)[0] + ':[redacted]' : command}`);
if (TRACKED_CONFIGURE_ACTIONS.has(action)) {
  trackEvent('configure_change', { action, result: 'success' });
}
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
  if (!ESPectreBleClient.supported) {
    showConnectionStatus('Web Bluetooth is not available in this browser.', 'error');
    log('Web Bluetooth is not available in this browser.');
    ToolPage.showNotification('Web Bluetooth is not available in this browser.', 'error');
    trackEvent('tool_connection', { tool_name: 'configure', transport: 'bluetooth', result: 'unsupported' });
    return;
  }

  const connectBtn = el('connectBtn');
  if (connectBtn) connectBtn.disabled = true;
  showConnectionStatus('Requesting Bluetooth device…', 'connecting');
  log('Opening BLE device picker...');
  trackEvent('tool_connection', { tool_name: 'configure', transport: 'bluetooth', result: 'attempt' });
  try {
    await ble.connect({ telemetry: false, sysinfo: true });
    log(`Connected to ${ble.name || 'ESPectre'}`);
    if (liveTelemetryEnabled() && capabilityState.supportsLiveTelemetry) {
      await ble.setTelemetryNotifications(true);
      log('Telemetry notifications enabled');
    } else {
      log('Telemetry notifications disabled from UI');
    }
    log('System info notifications enabled');
    setConnected(true);
    trackEvent('tool_connection', { tool_name: 'configure', transport: 'bluetooth', result: 'success' });
    await writeControl('REQ_SYSINFO');
  } catch (error) {
    if (connectBtn) connectBtn.disabled = false;
    showConnectionStatus(error.message || 'Bluetooth connection failed.', 'error');
    throw error;
  }
}

async function disconnect() {
  await ble.disconnect();
  setConnected(false);
}

async function toggleBleConnection() {
  try {
    if (ble.connected) {
      await disconnect();
    } else {
      await connect();
    }
  } catch (error) {
    log(`error: ${error.message}`);
    ToolPage.showNotification(`Bluetooth connection failed: ${error.message}`, 'error');
    trackEvent('tool_connection', {
      tool_name: 'configure',
      transport: 'bluetooth',
      result: 'failure',
      error_type: error.name || 'Error'
    });
  }
}

el('connectBtn').addEventListener('click', toggleBleConnection);
el('disconnectBtn').addEventListener('click', () => {
  toggleBleConnection();
});
window.toolPageActions = {
  ...(window.toolPageActions || {}),
  connectBtn: toggleBleConnection
};

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
  if (ble.connected) {
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
el('sysinfoLog').textContent = '';
el('eventLog').textContent = '';
movementBar.setMovement(0);
});

if (!ESPectreBleClient.supported) {
  el('connectBtn').disabled = true;
  showConnectionStatus('Web Bluetooth is not supported. Use Chrome or Edge.', 'error');
}

updateConnectionUi(false);
applyCapabilityState(false);
updateLiveTelemetryButton();
