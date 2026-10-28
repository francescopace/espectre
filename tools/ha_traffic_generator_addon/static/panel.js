/* SPDX-License-Identifier: GPL-3.0-only */
"use strict";

const $ = (id) => document.getElementById(id);
let rows = [], csrf = "", busy = false, pending = null, connectTask = null, reconnectTimer = null, actionMessage = "", pageActive = true;
const selected = new Set();
// Devices whose last change failed or is unconfirmed: id → {option, status, message}.
const outcomes = new Map();
const deviceElements = new Map();
let generatorRunning = null, generatorProtocol = null;
// The bulk source the user picked; it survives selections that do not offer it.
let bulkMode = "external";
let streamSocket = null;
// Labels for the modes the firmware offers; other options show their own name.
const MODE_LABELS = {ping: "Ping", dns: "DNS", dns_tcp: "DNS over TCP", wifi_raw: "Wi-Fi raw", external: "External"};
// `internal` asks each device for the internal source it used before.
const INTERNAL = "internal";
const modeLabel = mode => mode === INTERNAL ? "Previous internal source" : MODE_LABELS[mode] || mode;
const base = new URL(location.pathname.endsWith("/") ? location.pathname : `${location.pathname}/`, location.origin);

async function api(path, body) {
  const response = await fetch(new URL(`api/${path}`, base), {
    method: body ? "POST" : "GET",
    headers: body ? {"Content-Type": "application/json", "X-ESPectre-CSRF": csrf} : {},
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
    signal: AbortSignal.timeout(45000),
  });
  if (!response.ok) {
    let message = `Home Assistant unavailable (${response.status}).`;
    if (response.headers.get("content-type")?.includes("application/json")) {
      message = (await response.json()).error || message;
    }
    throw Object.assign(new Error(message), {status: response.status});
  }
  return response.json();
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function noticeFailure(error, fallback) {
  // Ingress rejects an expired session cookie. Home Assistant normally renews it only on page load.
  const expired = error.status === 401;
  notice(expired ? "Home Assistant session expired. Reload the page." : fallback, true);
  return expired;
}

function controls() {
  $("generator-toggle").disabled = busy || generatorRunning === null;
  const eligible = rows.filter(row => row.can_control);
  $("select-all").disabled = busy || !eligible.length;
  $("select-all").checked = eligible.length > 0 && eligible.every(row => selected.has(row.id));
  $("select-all").indeterminate = eligible.some(row => selected.has(row.id)) && !$("select-all").checked;
  $("selection").textContent = selected.size ? `${selected.size} selected` : "";
  $("bulk-actions").hidden = selected.size === 0;
  // Bulk choices: every mode a selected device offers, plus each device's previous internal source.
  const offered = new Set(eligible.filter(row => selected.has(row.id)).flatMap(row => row.modes));
  const choices = [...Object.keys(MODE_LABELS).filter(mode => offered.has(mode)),
    ...[...offered].filter(mode => !(mode in MODE_LABELS)), INTERNAL];
  setOptions($("bulk-mode"), choices, bulkMode);
  $("bulk-mode").disabled = busy;
  $("bulk-apply").disabled = busy || !eligible.some(row => selected.has(row.id)) ||
    ($("bulk-mode").value === "external" && generatorRunning !== true);
}

function setOptions(select, modes, value) {
  const current = [...select.children].map(option => option.value);
  if (current.join() !== modes.join()) {
    select.replaceChildren(...modes.map(mode => {
      const option = node("option", modeLabel(mode));
      option.value = mode;
      option.disabled = mode === "external" && generatorRunning !== true;
      return option;
    }));
  }
  for (const option of select.children) option.disabled = option.value === "external" && generatorRunning !== true;
  select.value = modes.includes(value) ? value : modes[0] ?? "";
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function sample(cell, field, suffix = "") {
  cell.textContent = field.value === null ? "—" : `${Number(field.value).toLocaleString(undefined, {maximumFractionDigits: 1})}${suffix}`;
  cell.title = field.entity_id || "Entity not exposed by this integration";
}

function deviceElement(id) {
  if (!deviceElements.has(id)) {
    const tr = node("tr"), name = node("a", "", "device-name");
    name.href = `/config/devices/device/${encodeURIComponent(id)}`;
    name.target = "_top";
    const checkCell = node("td"), check = node("input");
    check.type = "checkbox";
    check.addEventListener("change", () => { check.checked ? selected.add(id) : selected.delete(id); controls(); });
    checkCell.append(check);
    const nameCell = node("td"), ip = node("div", "", "device-ip");
    nameCell.append(name, ip);
    const generator = node("td"), traffic = node("td"), trafficRx = node("td");
    const accepted = node("td"), occupancy = node("td"), rssi = node("td");
    const sourceCell = node("td"), source = node("select");
    source.addEventListener("change", () => propose(source.value, [id]));
    sourceCell.append(source);
    tr.append(checkCell, nameCell, generator, traffic, trafficRx, accepted, occupancy, rssi, sourceCell);
    deviceElements.set(id, {tr, check, name, ip, generator, traffic, trafficRx, accepted, occupancy, rssi, source});
  }
  return deviceElements.get(id);
}

// force resets every source picker, even the focused one, after a confirmation closes.
function render(force = false) {
  const ids = new Set(rows.map(row => row.id));
  for (const id of selected) if (!rows.some(row => row.id === id && row.can_control)) selected.delete(id);
  for (const [id, element] of deviceElements) {
    if (!ids.has(id)) { element.tr.remove(); deviceElements.delete(id); }
  }
  // Update cells in place so one-second sampling preserves keyboard focus and selection.
  rows.forEach((row, index) => {
    const element = deviceElement(row.id);
    element.check.checked = selected.has(row.id);
    element.check.disabled = busy || !row.can_control;
    element.check.setAttribute("aria-label", `Select ${row.name}`);
    element.name.textContent = row.name;
    element.name.title = row.area || "";
    const address = row.reason ? "Unavailable" : row.ip_address || "—";
    element.ip.textContent = row.chip ? `${row.chip} · ${address}` : address;
    element.ip.title = row.reason || (row.ip_address ? "IP address reported by Home Assistant" : "IP address not exposed by Home Assistant");
    const current = row.fields.source.value;
    const keep = !force && (pending !== null || document.activeElement === element.source);
    const modes = row.modes.length ? row.modes : current ? [current] : [];
    setOptions(element.source, modes, keep ? element.source.value : current);
    element.source.disabled = busy || !row.can_control;
    element.source.setAttribute("aria-label", `Traffic source: ${row.name}`);
    sample(element.generator, row.fields.generator);
    sample(element.traffic, row.fields.traffic);
    sample(element.trafficRx, row.fields.traffic_rx);
    sample(element.accepted, row.fields.accepted);
    sample(element.occupancy, row.fields.occupancy, "%");
    sample(element.rssi, row.fields.rssi, " dBm");
    if ($("devices").children[index] !== element.tr) {
      $("devices").insertBefore(element.tr, $("devices").children[index] || null);
    }
  });
  controls();
}

function scheduleStream(delay = 0) {
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (!pageActive || document.hidden || busy || pending) {
    const socket = streamSocket;
    streamSocket = null;
    socket?.close();
    return;
  }
  if (!streamSocket && !connectTask) reconnectTimer = setTimeout(connectStream, delay);
}

function connectStream() {
  if (!pageActive || document.hidden || busy || pending || connectTask || streamSocket) return;
  connectTask = openStream().catch(error => {
    noticeFailure(error, "Connection lost. Reconnecting…");
  }).finally(() => {
    connectTask = null;
    scheduleStream(2000);
  });
}

async function openStream() {
  await loadStatus();
  if (!pageActive || document.hidden || busy || pending) return;
  const url = new URL("api/stream", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(url);
  streamSocket = socket;
  socket.onopen = () => {
    if (streamSocket === socket) socket.send(JSON.stringify({csrf}));
  };
  socket.onmessage = event => {
    if (streamSocket !== socket) return;
    const data = JSON.parse(event.data);
    showGenerator(data.generator);
    if (data.devices) {
      showSnapshot(data);
      const message = outcomeMessage();
      notice(message || data.error || (rows.length ? "" : "No ESPectre devices."), Boolean(message || data.error));
    }
  };
  socket.onclose = () => {
    if (streamSocket !== socket) return;
    streamSocket = null;
    generatorRunning = null;
    $("running").textContent = "Disconnected";
    rows = []; selected.clear(); render();
    notice("Connection lost. Reconnecting…", true);
    scheduleStream(2000);
  };
  socket.onerror = () => socket.close();
}

async function loadStatus() {
  try {
    const data = await api("status");
    csrf = data.csrf;
    showGenerator(data.generator);
  } catch (error) {
    generatorRunning = null;
    $("running").textContent = "Disconnected";
    controls();
    throw error;
  }
}

function showGenerator(generator) {
  generatorRunning = generator.running;
  generatorProtocol = generator.protocol;
  $("generator-toggle").textContent = generator.running ? "Stop" : "Start";
  $("generator-toggle").title = generator.running ? "Stop traffic; device modes stay unchanged" : "Start traffic";
  $("running").textContent = generator.running ? "Running" : "Stopped";
  $("rate").textContent = `${generator.rate_pps} pps`;
  $("errors").textContent = generator.send_errors.toLocaleString();
  $("targets").textContent = generator.protocol === "icmp"
    ? `${generator.targets.join(", ")} · ping, ${(generator.replies_received ?? 0).toLocaleString()} replies`
    : `${generator.targets.join(", ")} · UDP ${generator.port}`;
  controls();
}

function showSnapshot(data) {
  rows = data.devices;
  // A late confirmation arrives through the stream: drop its warning and selection.
  for (const [id, outcome] of outcomes) {
    const row = rows.find(row => row.id === id);
    if (outcome.status === "unconfirmed" && row?.fields.source.value === outcome.option) {
      outcomes.delete(id);
      selected.delete(id);
    }
  }
  render();
}

function outcomeMessage() {
  if (actionMessage) return actionMessage;
  const names = new Map(rows.map(row => [row.id, row.name]));
  return [...outcomes].map(([id, outcome]) => `${names.get(id) || id}: ${outcome.message}`).join("\n");
}

function propose(mode, ids) {
  if (busy) return;
  if (mode === "external" && generatorRunning !== true) { render(true); return; }
  ids = ids.filter(id => rows.some(row => row.id === id && row.can_control));
  if (!ids.length) return;
  if (ids.length > 32) { notice("Select up to 32 devices.", true); return; }
  pending = {mode, ids};
  scheduleStream();
  $("confirm-title").textContent = mode === INTERNAL ? "Use the previous internal source?" : `Use ${modeLabel(mode)} traffic?`;
  $("confirm-message").textContent = `${rows.filter(row => ids.includes(row.id)).map(row => row.name).join(", ")}. ` +
    (mode === "external"
      ? `Check targets${generatorProtocol === "icmp" ? "" : ", UDP port,"} and rate. Calibration may restart.`
      : `${mode === INTERNAL ? "Each device goes back to the internal source it used before, or ping. " : ""}Calibration may restart.`);
  $("confirm").returnValue = "cancel";
  $("confirm").showModal();
}

async function execute(mode, ids) {
  busy = true; actionMessage = ""; outcomes.clear(); scheduleStream(); render(); notice("Applying…");
  try {
    // Finish connection setup before sending a command; never retry writes automatically.
    await connectTask;
    await loadStatus();
    const data = await api("action", {mode, device_ids: ids});
    for (const result of data.results) {
      if (["error", "unconfirmed"].includes(result.status)) {
        outcomes.set(result.id, {option: result.option || mode, status: result.status, message: result.message});
      }
    }
    // Keep only the devices that still need attention selected, so one click retries them.
    selected.clear();
    for (const id of outcomes.keys()) selected.add(id);
    showSnapshot(data);
    const message = outcomeMessage();
    notice(message, Boolean(message));
  } catch (error) {
    // A timed-out write may have succeeded. Never retry it automatically.
    rows = []; selected.clear(); outcomes.clear();
    const fallback = "Action not confirmed. Check the device in Home Assistant before retrying.";
    actionMessage = noticeFailure(error, fallback) ? "" : fallback;
  } finally { busy = false; render(true); scheduleStream(); }
}

async function toggleGenerator() {
  if (busy || generatorRunning === null) return;
  const action = generatorRunning ? "stop" : "start";
  busy = true; actionMessage = ""; scheduleStream(); render(); notice("Applying…");
  try {
    await connectTask;
    await loadStatus();
    showGenerator((await api("generator", {action})).generator);
    notice("");
  } catch (error) {
    generatorRunning = null;
    $("running").textContent = "Disconnected";
    const fallback = "Generator action not confirmed. Check its status before retrying.";
    actionMessage = noticeFailure(error, fallback) ? "" : fallback;
  } finally { busy = false; render(); scheduleStream(); }
}

$("generator-toggle").addEventListener("click", toggleGenerator);
// Use the HA route's installed slug, including local/repository prefixes.
try {
  const slug = window.parent.location.pathname.split("/").filter(Boolean).pop();
  if (/^[a-zA-Z0-9_-]+_espectre_traffic_generator$/.test(slug || "")) {
    $("addon-settings").href = `/config/app/${encodeURIComponent(slug)}/config`;
  }
} catch {
  // Standalone previews or cross-origin embeds keep the Apps overview link.
}
$("confirm").addEventListener("close", () => {
  const request = pending; pending = null;
  if ($("confirm").returnValue === "apply" && request) execute(request.mode, request.ids);
  else { render(true); scheduleStream(); }
});
$("select-all").addEventListener("change", event => {
  selected.clear();
  if (event.target.checked) rows.filter(row => row.can_control).forEach(row => selected.add(row.id));
  render();
});
$("bulk-mode").addEventListener("change", () => { bulkMode = $("bulk-mode").value; controls(); });
$("bulk-apply").addEventListener("click", () => propose($("bulk-mode").value, [...selected]));
document.addEventListener("visibilitychange", () => scheduleStream());
window.addEventListener("pagehide", () => { pageActive = false; scheduleStream(); });
window.addEventListener("pageshow", () => { pageActive = true; scheduleStream(); });
scheduleStream();
