/* SPDX-License-Identifier: GPL-3.0-only */
"use strict";

const $ = (id) => document.getElementById(id);
const NUMBERS = ["port", "rate_pps", "multicast_ttl", "dscp"];
const base = new URL(location.pathname.endsWith("/") ? location.pathname : `${location.pathname}/`, location.origin);
let csrf = "", busy = false, dirty = false, timer = null, settings = null, generator = null, previous = null;
// The outcome of the last change stays visible until the next one.
let outcome = {text: "", error: false};
// Labels in display order; each device lists the sources it offers.
const MODE_LABELS = {ping: "Ping", dns: "DNS", dns_tcp: "DNS over TCP", wifi_raw: "Wi-Fi raw", external: "External"};
// The bulk picker can also send each device back to the internal source it used before.
const INTERNAL = "internal";
const modeLabel = mode => mode === INTERNAL ? "Previous internal source" : MODE_LABELS[mode] || mode;
const FIELDS = ["generator", "traffic_tx", "traffic_rx", "accepted", "occupancy", "rssi"];
let devices = [], scanning = false, deviceBusy = false, pending = null, deviceMessage = "";
const selected = new Set();
const rowElements = new Map();
// Devices whose last change failed or is unconfirmed: id → {mode, status, message}.
const outcomes = new Map();
// The bulk source the user picked; it survives selections that do not offer it.
let bulkMode = "external";

async function api(path, body) {
  const response = await fetch(new URL(`api/${path}`, base), {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : {"Content-Type": "application/json", "X-ESPectre-CSRF": csrf},
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
    credentials: "same-origin",
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    const message = (await response.text()).trim() || `Request failed (${response.status}).`;
    throw Object.assign(new Error(message), {status: response.status});
  }
  return response.json();
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function showView(signedIn) {
  $("sign-in").hidden = signedIn;
  $("app").hidden = !signedIn;
  $("controls").hidden = !signedIn;
  if (!signedIn) {
    clearTimeout(timer);
    timer = null;
    generator = null;
    previous = null;
    $("token").focus();
  }
}

function controls() {
  $("generator-toggle").disabled = busy || generator === null;
  $("save").disabled = busy || !dirty;
  $("revert").disabled = busy || !dirty;
  $("port").disabled = busy || $("protocol").value !== "udp";
}

function count(value) {
  return Number(value).toLocaleString(undefined, {maximumFractionDigits: 1});
}

function showGenerator(next) {
  const now = performance.now();
  if (next.running && previous && next.sent_packets >= previous.sent) {
    $("send-rate").textContent = count((next.sent_packets - previous.sent) / ((now - previous.at) / 1000));
  } else if (!next.running) {
    $("send-rate").textContent = "0";
  }
  previous = {sent: next.sent_packets, at: now};
  generator = next;
  $("running").textContent = next.running ? "Running" : next.failed ? "Failed" : "Stopped";
  $("running").classList.toggle("failed", next.failed);
  $("generator-toggle").textContent = next.running ? "Stop" : "Start";
  $("rate").textContent = `${count(next.rate_pps)} pps`;
  $("sent").textContent = count(next.sent_packets);
  $("errors").textContent = count(next.send_errors);
  $("replies").textContent = count(next.replies_received);
  $("replies-summary").hidden = next.protocol !== "icmp";
  $("targets").textContent = next.protocol === "icmp"
    ? `${next.targets.join(", ")} · ping`
    : `${next.targets.join(", ")} · UDP ${next.port}`;
  controls();
}

function fillForm(values) {
  $("targets-input").value = values.targets.join("\n");
  $("protocol").value = values.protocol;
  for (const name of [...NUMBERS, "source_ip"]) $(name).value = String(values[name]);
  dirty = false;
  controls();
}

function readForm() {
  const integer = (name) => $(name).value.trim() === "" ? null : Number($(name).value);
  const values = {
    targets: $("targets-input").value.split(/[\s,]+/).filter(Boolean),
    protocol: $("protocol").value,
    source_ip: $("source_ip").value.trim(),
  };
  for (const name of NUMBERS) values[name] = integer(name);
  // The port is ignored for ping; keep the saved value so it stays valid.
  if (values.protocol !== "udp" && settings) values.port = settings.port;
  return values;
}

function show(data) {
  csrf = data.csrf;
  settings = data.settings;
  showView(true);
  showGenerator(data.generator);
  if (!dirty) fillForm(settings);
  showDevices(data.devices);
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

// External traffic needs the generator running.
function setOptions(select, modes, value) {
  if ([...select.children].map(option => option.value).join() !== modes.join()) {
    select.replaceChildren(...modes.map(mode => {
      const option = node("option", modeLabel(mode));
      option.value = mode;
      return option;
    }));
  }
  for (const option of select.children) option.disabled = option.value === "external" && generator?.running !== true;
  select.value = modes.includes(value) ? value : modes[0] ?? "";
}

function sample(cell, value, scale = 1, suffix = "") {
  cell.textContent = value === null || value === undefined ? "—" : `${count(Number(value) * scale)}${suffix}`;
}

// Settings that keep a device in external mode from using this generator's traffic.
function hints(device) {
  const sensing = device.sensing;
  if (sensing.traffic_generator_mode !== "external" || !generator) return [];
  const found = [];
  if (sensing.csi_target_pps !== null && sensing.csi_target_pps !== generator.rate_pps) {
    found.push(`Device expects ${sensing.csi_target_pps} pps`);
  }
  if (generator.protocol === "udp" && sensing.csi_traffic_udp_port !== null && sensing.csi_traffic_udp_port !== generator.port) {
    found.push(`Device listens on UDP ${sensing.csi_traffic_udp_port}`);
  }
  const groups = generator.targets.filter(target => Number(target.split(".")[0]) >= 224 && Number(target.split(".")[0]) <= 239);
  if (groups.length && sensing.csi_traffic_multicast_group && !groups.includes(sensing.csi_traffic_multicast_group)) {
    found.push(`Device joins ${sensing.csi_traffic_multicast_group}`);
  }
  return found;
}

function deviceRow(id) {
  if (!rowElements.has(id)) {
    const tr = node("tr"), checkCell = node("td"), check = node("input");
    check.type = "checkbox";
    check.addEventListener("change", () => { check.checked ? selected.add(id) : selected.delete(id); deviceControls(); });
    checkCell.append(check);
    const nameCell = node("td"), name = node("div", "", "device-name"), detail = node("div", "", "device-detail");
    const detailText = node("span"), remove = node("button", "Remove");
    remove.type = "button";
    remove.addEventListener("click", () => deviceChange("devices/remove", {device_id: id}, ""));
    detail.append(detailText, remove);
    nameCell.append(name, detail);
    const cells = FIELDS.map(() => node("td"));
    const modeCell = node("td"), mode = node("select"), hint = node("small", "", "hint");
    mode.addEventListener("change", () => propose(mode.value, [id]));
    modeCell.append(mode, hint);
    tr.append(checkCell, nameCell, ...cells, modeCell);
    rowElements.set(id, {tr, check, name, detailText, remove, cells, mode, hint});
  }
  return rowElements.get(id);
}

// force resets every source picker, even the focused one, after a confirmation closes.
function showDevices(snapshot, force = false) {
  devices = snapshot.devices;
  scanning = snapshot.scanning;
  const ids = new Set(devices.map(device => device.id));
  for (const [id, row] of rowElements) {
    if (!ids.has(id)) { row.tr.remove(); rowElements.delete(id); }
  }
  // A device that reports the requested source later no longer needs attention.
  for (const [id, result] of outcomes) {
    const device = devices.find(device => device.id === id);
    if (!device || (result.status === "unconfirmed" && device.sensing.traffic_generator_mode === result.mode)) {
      outcomes.delete(id);
      selected.delete(id);
    }
  }
  for (const id of selected) if (!devices.some(device => device.id === id && device.can_control)) selected.delete(id);
  devices.forEach((device, index) => {
    const row = deviceRow(device.id);
    row.check.checked = selected.has(device.id);
    row.check.setAttribute("aria-label", `Select ${device.name}`);
    row.name.textContent = device.name;
    row.detailText.textContent = [device.chip, device.address, device.error ? "Unreachable" : ""].filter(Boolean).join(" · ");
    row.detailText.className = device.error ? "unreachable" : "";
    row.detailText.title = device.error || `${device.frontend} ${device.firmware}`.trim();
    row.remove.hidden = device.source !== "manual";
    row.remove.setAttribute("aria-label", `Remove ${device.name}`);
    FIELDS.forEach((name, column) => sample(row.cells[column], device.fields[name],
      name === "occupancy" ? 100 : 1, {occupancy: "%", rssi: " dBm"}[name] || ""));
    const current = device.sensing.traffic_generator_mode || "";
    const keep = !force && (pending !== null || document.activeElement === row.mode);
    setOptions(row.mode, device.modes.length ? device.modes : current ? [current] : [], keep ? row.mode.value : current);
    row.mode.setAttribute("aria-label", `Traffic source: ${device.name}`);
    row.hint.textContent = hints(device).join(" · ");
    if ($("devices").children[index] !== row.tr) $("devices").insertBefore(row.tr, $("devices").children[index] || null);
  });
  deviceControls();
  deviceNotice();
}

function deviceNotice() {
  const names = new Map(devices.map(device => [device.id, device.name]));
  const lines = [...outcomes].map(([id, result]) => `${names.get(id) || id}: ${result.message}`);
  let message = deviceMessage || lines.join("\n");
  if (!message && scanning) message = "Searching the network…";
  if (!message && !devices.length) message = "No devices found. Search again, or add one by its IPv4 address.";
  $("devices-notice").textContent = message;
  $("devices-notice").classList.toggle("error", Boolean(deviceMessage || lines.length));
}

function deviceControls() {
  const eligible = devices.filter(device => device.can_control);
  for (const device of devices) {
    const row = rowElements.get(device.id);
    row.check.disabled = deviceBusy || !device.can_control;
    row.mode.disabled = deviceBusy || !device.can_control;
    row.remove.disabled = deviceBusy;
  }
  $("select-all").disabled = deviceBusy || !eligible.length;
  $("select-all").checked = eligible.length > 0 && eligible.every(device => selected.has(device.id));
  $("select-all").indeterminate = eligible.some(device => selected.has(device.id)) && !$("select-all").checked;
  $("selection").textContent = selected.size ? `${selected.size} selected` : "";
  $("bulk-actions").hidden = selected.size === 0;
  // Bulk choices: every source a selected device offers, plus each device's previous internal source.
  const offered = new Set(eligible.filter(device => selected.has(device.id)).flatMap(device => device.modes));
  setOptions($("bulk-mode"), [...Object.keys(MODE_LABELS).filter(mode => offered.has(mode)), INTERNAL],
    bulkMode);
  $("bulk-apply").disabled = deviceBusy || selected.size === 0 ||
    ($("bulk-mode").value === "external" && generator?.running !== true);
  $("scan").disabled = deviceBusy || scanning;
  $("scan").textContent = scanning ? "Searching…" : "Search";
  $("add").disabled = deviceBusy;
}

function propose(mode, ids) {
  ids = ids.filter(id => devices.some(device => device.id === id && device.can_control));
  if (deviceBusy || !ids.length) return;
  pending = {mode, ids};
  $("confirm-title").textContent = mode === INTERNAL ? "Use the previous internal source?" : `Use ${modeLabel(mode)} traffic?`;
  $("confirm-message").textContent = `${devices.filter(device => ids.includes(device.id)).map(device => device.name).join(", ")}. ` +
    (mode === "external" ? "Check that the device rate and multicast group match the generator. " : "") +
    (mode === INTERNAL ? "Each device goes back to the internal source it used before, or ping. " : "") + "Calibration may restart.";
  $("confirm").returnValue = "cancel";
  $("confirm").showModal();
}

async function deviceChange(path, body, success) {
  if (deviceBusy) return null;
  deviceBusy = true;
  deviceMessage = "";
  deviceControls();
  $("devices-notice").textContent = "Applying…";
  try {
    const data = await api(path, body);
    deviceBusy = false;
    deviceMessage = success;
    show(data);
    return data;
  } catch (error) {
    if (error.status === 401) failure(error, "");
    else deviceMessage = error.status ? error.message : "Change not confirmed. Check the device before retrying.";
    return null;
  } finally {
    deviceBusy = false;
    deviceControls();
    deviceNotice();
  }
}

async function applyMode(mode, ids) {
  outcomes.clear();
  const data = await deviceChange("devices/mode", {device_ids: ids, mode}, "");
  if (!data) {
    showDevices({devices, scanning}, true);
    return;
  }
  for (const result of data.results) {
    if (["error", "unconfirmed"].includes(result.status)) {
      outcomes.set(result.id, {mode: result.option || mode, status: result.status, message: result.message});
    }
  }
  // Keep only the devices that still need attention selected, so one click retries them.
  selected.clear();
  for (const id of outcomes.keys()) selected.add(id);
  showDevices(data.devices, true);
}

function showOutcome(next) {
  if (next.failed) notice(`Traffic stopped: ${next.error}`, true);
  else notice(outcome.text, outcome.error);
}

function failure(error, fallback) {
  if (error.status === 401) {
    showView(false);
    notice("");
    return;
  }
  outcome = {text: error.status ? error.message : fallback, error: true};
  notice(outcome.text, true);
}

function schedule(delay = 2000) {
  clearTimeout(timer);
  timer = document.hidden || $("app").hidden ? null : setTimeout(refresh, delay);
}

async function refresh() {
  try {
    const data = await api("status");
    show(data);
    if (!busy) showOutcome(data.generator);
    schedule();
  } catch (error) {
    failure(error, "Connection lost. Retrying…");
    if (error.status !== 401) {
      generator = null;
      $("running").textContent = "Disconnected";
      controls();
      timer = setTimeout(refresh, 5000);
    }
  }
}

async function change(path, body, success) {
  if (busy) return;
  busy = true;
  outcome = {text: "", error: false};
  controls();
  notice("Applying…");
  try {
    const data = await api(path, body);
    busy = false;
    if (path === "settings") dirty = false;
    show(data);
    outcome = {text: success, error: false};
    showOutcome(data.generator);
  } catch (error) {
    failure(error, "Change not confirmed. Check the status before retrying.");
  } finally {
    busy = false;
    controls();
    schedule();
  }
}

$("sign-in-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("login", {token: $("token").value});
    $("token").value = "";
    notice("");
    await refresh();
  } catch (error) {
    notice(error.message || "Sign-in failed.", true);
  }
});

$("sign-out").addEventListener("click", async () => {
  try {
    await api("logout", {});
  } finally {
    showView(false);
    notice("");
  }
});

$("generator-toggle").addEventListener("click", () => {
  if (generator !== null) change("generator", {action: generator.running ? "stop" : "start"}, "");
});

$("settings").addEventListener("input", () => {
  dirty = true;
  outcome = {text: "", error: false};
  controls();
});

$("settings").addEventListener("submit", (event) => {
  event.preventDefault();
  if (dirty) change("settings", readForm(), "Settings saved. Counters restart from zero.");
});

$("bulk-mode").addEventListener("change", () => { bulkMode = $("bulk-mode").value; deviceControls(); });
$("bulk-apply").addEventListener("click", () => propose($("bulk-mode").value, [...selected]));
$("select-all").addEventListener("change", (event) => {
  selected.clear();
  if (event.target.checked) devices.filter(device => device.can_control).forEach(device => selected.add(device.id));
  showDevices({devices, scanning});
});
$("confirm").addEventListener("close", () => {
  const request = pending;
  pending = null;
  if ($("confirm").returnValue === "apply" && request) applyMode(request.mode, request.ids);
  else showDevices({devices, scanning}, true);
});
$("scan").addEventListener("click", () => deviceChange("devices/scan", {}, ""));
$("add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const address = $("add-address").value.trim();
  if (address && await deviceChange("devices/add", {address}, "")) $("add-address").value = "";
});

$("revert").addEventListener("click", () => {
  if (settings) fillForm(settings);
  outcome = {text: "", error: false};
  notice("");
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearTimeout(timer);
    timer = null;
  } else if (!$("app").hidden) {
    refresh();
  }
});

refresh();
