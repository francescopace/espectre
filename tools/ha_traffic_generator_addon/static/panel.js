/* SPDX-License-Identifier: GPL-3.0-only */
"use strict";

const $ = (id) => document.getElementById(id);
let rows = [], csrf = "", busy = false, pending = null, connectTask = null, reconnectTimer = null, actionMessage = "", pageActive = true;
const selected = new Set();
const deviceElements = new Map();
let generatorRunning = null;
let streamSocket = null;
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
    throw new Error(message);
  }
  return response.json();
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function controls() {
  $("generator-toggle").disabled = busy || generatorRunning === null;
  const eligible = rows.filter(row => row.can_control);
  $("select-all").disabled = busy || !eligible.length;
  $("select-all").checked = eligible.length > 0 && eligible.every(row => selected.has(row.id));
  $("select-all").indeterminate = eligible.some(row => selected.has(row.id)) && !$("select-all").checked;
  $("selection").textContent = selected.size ? `${selected.size} selected` : "";
  $("bulk-actions").hidden = selected.size === 0;
  for (const name of ["external", "internal"]) {
    $(name).disabled = busy || (name === "external" && generatorRunning !== true) || !eligible.some(row => selected.has(row.id));
  }
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
    const actions = node("td"), buttons = node("div", undefined, "row-actions");
    const modes = {};
    for (const [action, label] of [["external", "External"], ["internal", "Internal"]]) {
      const button = node("button", label);
      button.type = "button";
      button.addEventListener("click", () => propose(action, [id]));
      buttons.append(button);
      modes[action] = button;
    }
    actions.append(buttons);
    tr.append(checkCell, nameCell, generator, traffic, trafficRx, accepted, occupancy, rssi, actions);
    deviceElements.set(id, {tr, check, name, ip, generator, traffic, trafficRx, accepted, occupancy, rssi, modes});
  }
  return deviceElements.get(id);
}

function render() {
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
    for (const [action, button] of Object.entries(element.modes)) {
      button.disabled = busy || !row.can_control || row.fields.ownership.value === action ||
        (action === "external" && generatorRunning !== true);
      button.setAttribute("aria-label", `${button.textContent}: ${row.name}`);
    }
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
  connectTask = openStream().catch(() => {
    notice("Connection lost. Reconnecting…", true);
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
      notice(actionMessage || data.error || (rows.length ? "" : "No ESPectre devices."), Boolean(actionMessage || data.error));
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
  $("generator-toggle").textContent = generator.running ? "Stop" : "Start";
  $("generator-toggle").title = generator.running ? "Stop UDP traffic; device modes stay unchanged" : "Start UDP traffic";
  $("running").textContent = generator.running ? "Running" : "Stopped";
  $("rate").textContent = `${generator.rate_pps} pps`;
  $("errors").textContent = generator.send_errors.toLocaleString();
  $("targets").textContent = `${generator.targets.join(", ")} · UDP ${generator.port}`;
  controls();
}

function showSnapshot(data) {
  rows = data.devices;
  render();
}

function propose(action, ids) {
  if (busy) return;
  if (action === "external" && generatorRunning !== true) return;
  ids = ids.filter(id => rows.some(row => row.id === id && row.can_control));
  if (!ids.length) return;
  if (ids.length > 32) { notice("Select up to 32 devices.", true); return; }
  pending = {action, ids};
  scheduleStream();
  $("confirm-title").textContent = `Use ${action} traffic?`;
  $("confirm-message").textContent = `${rows.filter(row => ids.includes(row.id)).map(row => row.name).join(", ")}. ` +
    (action === "external" ? "Check targets, UDP port, and rate. Calibration may restart." : "Resume internal traffic. Calibration may restart.");
  $("confirm").returnValue = "cancel";
  $("confirm").showModal();
}

async function execute(action, ids) {
  busy = true; actionMessage = ""; scheduleStream(); render(); notice("Applying…");
  try {
    // Finish connection setup before sending a command; never retry writes automatically.
    await connectTask;
    await loadStatus();
    const data = await api("action", {action, device_ids: ids});
    showSnapshot(data);
    const names = new Map(rows.map(row => [row.id, row.name]));
    actionMessage = data.results.filter(result => ["error", "unconfirmed"].includes(result.status))
      .map(result => `${names.get(result.id) || result.id}: ${result.message}`).join("\n");
    notice(actionMessage, Boolean(actionMessage));
  } catch (error) {
    // A timed-out write may have succeeded. Never retry it automatically.
    rows = []; selected.clear();
    actionMessage = "Action not confirmed. Check the device in Home Assistant before retrying.";
    notice(actionMessage, true);
  } finally { busy = false; render(); scheduleStream(); }
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
    actionMessage = "Generator action not confirmed. Check its status before retrying.";
    notice(actionMessage, true);
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
  if ($("confirm").returnValue === "apply" && request) execute(request.action, request.ids);
  else scheduleStream();
});
$("select-all").addEventListener("change", event => {
  selected.clear();
  if (event.target.checked) rows.filter(row => row.can_control).forEach(row => selected.add(row.id));
  render();
});
for (const action of ["external", "internal"]) {
  $(action).addEventListener("click", () => propose(action, [...selected]));
}
document.addEventListener("visibilitychange", () => scheduleStream());
window.addEventListener("pagehide", () => { pageActive = false; scheduleStream(); });
window.addEventListener("pageshow", () => { pageActive = true; scheduleStream(); });
scheduleStream();
