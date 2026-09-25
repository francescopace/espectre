# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""ESPectre controls and metadata through Home Assistant's internal APIs."""

import asyncio
import contextlib
import ipaddress
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

import aiohttp


HA_WEBSOCKET = "ws://supervisor/core/websocket"
UNAVAILABLE = {"unknown", "unavailable", None}
# Every ESPectre firmware offers these packet modes in its traffic source select,
# whatever the user named it. 3.0.0-rc1 and rc2 add a separate ownership select.
TRAFFIC_SOURCE_MODES = {"ping", "dns", "dns_tcp"}
OWNERSHIP_MODES = {"internal", "external"}
# Integration-owned identifiers, not user-editable entity IDs or display names.
ENTITY_ROLES = {
    "refresh": ("button", "refresh_diagnostics"),
    "generator": ("sensor", "generator_rate"),
    "traffic": ("sensor", "traffic_tx_rate"),
    "traffic_rx": ("sensor", "traffic_rx_rate"),
    "accepted": ("sensor", "csi_accepted_rate"),
    "occupancy": ("sensor", "csi_temporal_occupancy"),
    "rssi": ("sensor", "wifi_rssi"),
    "calibrating": ("binary_sensor", "calibration_active"),
}


class HomeAssistantError(RuntimeError):
    """A bounded HA operation failed; messages never include credentials."""


def entity_role(entity):
    """Recognize MQTT IDs and ESPHome's legacy and device-aware IDs."""
    platform = entity.get("platform")
    domain = entity.get("entity_id", "").partition(".")[0]
    unique_id = str(entity.get("unique_id", ""))
    for role, (expected_domain, suffix) in ENTITY_ROLES.items():
        if domain != expected_domain:
            continue
        if platform == "mqtt" and unique_id.endswith("_" + suffix):
            return role
        if platform == "esphome":
            # ESPHome v3: mac/sub-device/domain/name; v1/v2: mac-domain-name.
            match = re.fullmatch(r"[^/]+/\d+/" + domain + r"/(.+)", unique_id)
            legacy = re.fullmatch(r"[0-9a-fA-F:]+-" + domain + r"-(.+?)(?:@\d+)?", unique_id)
            name = (match or legacy).group(1) if (match or legacy) else ""
            if re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") == suffix:
                return role
    return None


def select_options(entity, states):
    options = states.get(entity.get("entity_id"), {}).get("attributes", {}).get("options")
    if not entity.get("entity_id", "").startswith("select.") or not isinstance(options, list):
        return set()
    return set(options)


def entity_value(entity, states, *, numeric=False):
    if not entity:
        return {"value": None, "status": "missing", "updated_at": None}
    state = states.get(entity["entity_id"], {})
    value = state.get("state")
    status = "disabled" if entity.get("disabled_by") else "available"
    if status == "available" and value in UNAVAILABLE:
        status = "unavailable"
    if status != "available":
        value = None
    elif numeric:
        try:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
            value, status = None, "unavailable"
    return {"value": value, "status": status,
            "updated_at": state.get("last_reported") or state.get("last_updated"),
            "entity_id": entity["entity_id"]}


def device_ip(device):
    """Read a literal IP from HA metadata without resolving or contacting hosts."""
    if device.get("_network_ip"):
        return device["_network_ip"]
    try:
        url = urlsplit(device.get("configuration_url") or "")
        if url.scheme in {"http", "https"} and url.hostname:
            return str(ipaddress.ip_address(url.hostname))
    except ValueError:
        pass
    return None


def network_addresses(devices, addresses):
    """Join HA's DHCP cache to registry devices by MAC, never by display name."""
    for device in devices:
        matches = {addresses[re.sub(r"[^0-9a-f]", "", value.lower())]
                   for kind, value in device.get("connections", []) if kind == "mac"
                   and re.sub(r"[^0-9a-f]", "", value.lower()) in addresses}
        device["_network_ip"] = next(iter(matches)) if len(matches) == 1 else None


def device_chip(device):
    """Use explicit HA hardware metadata, never infer a chip from a device name."""
    chips = []
    for key in ("model", "model_id", "hw_version", "_esphome_model"):
        match = re.search(r"\bESP32(?:[-_ ]?([CSHP]\d+))?\b", str(device.get(key) or ""), re.IGNORECASE)
        if match:
            chips.append("ESP32" + ("-" + match[1].upper() if match[1] else ""))
    variants = set(chips) - {"ESP32"}
    if len(variants) == 1:
        return variants.pop()
    return "ESP32" if chips and not variants else None


def build_inventory(devices, entities, state_list, areas):
    """Return only ESPectre devices, with fail-closed control eligibility."""
    states = {state["entity_id"]: state for state in state_list}
    area_names = {area["area_id"]: area["name"] for area in areas}
    grouped = {}
    for entity in entities:
        if entity.get("device_id"):
            grouped.setdefault(entity["device_id"], []).append(entity)
    result = []
    for device in devices:
        candidates = {}
        for entity in grouped.get(device["id"], []):
            options = select_options(entity, states)
            role = ("source" if TRAFFIC_SOURCE_MODES <= options else
                    "ownership" if options == OWNERSHIP_MODES else entity_role(entity))
            if role:
                candidates.setdefault(role, []).append(entity)
        if "source" not in candidates:
            continue
        roles = {key: values[0] for key, values in candidates.items() if len(values) == 1}
        fields = {key: entity_value(roles.get(key), states, numeric=key in
                  {"generator", "traffic", "traffic_rx", "accepted", "occupancy", "rssi"})
                  for key in ("source", "ownership", *ENTITY_ROLES)}
        # Current firmware offers `external` in the source select itself.
        unified = "ownership" not in candidates
        control_role = "source" if unified else "ownership"
        control = fields[control_role]
        options = select_options(roles.get(control_role, {}), states)
        internal_mode = None
        if unified:
            if control["value"] not in {None, "external"}:
                internal_mode = control["value"]
            fields["ownership"] = dict(control, value=None if control["value"] is None else
                                       ("external" if control["value"] == "external" else "internal"))
        reason = None
        if device.get("disabled_by"):
            reason = "Device is disabled in Home Assistant."
        elif len(candidates[control_role]) > 1:
            reason = "Multiple traffic ownership entities; no control selected."
        elif control["status"] != "available":
            reason = "Traffic ownership entity is " + control["status"] + "."
        elif "external" not in options or control["value"] not in options:
            reason = "Traffic ownership options are not supported."
        # A button may have state 'unknown' before its first press; this is valid.
        refresh = roles.get("refresh", {})
        refresh_state = states.get(refresh.get("entity_id"))
        can_refresh = bool(refresh and not refresh.get("disabled_by") and refresh_state
                           and refresh_state.get("state") != "unavailable"
                           and not device.get("disabled_by"))
        result.append({
            "id": device["id"], "name": device.get("name_by_user") or device.get("name") or device["id"],
            "area": area_names.get(device.get("area_id")),
            "ip_address": device_ip(device),
            "chip": device_chip(device),
            "integration": ", ".join(sorted({e.get("platform", "") for e in
                                            grouped.get(device["id"], [])})),
            "fields": fields, "can_control": reason is None, "reason": reason,
            "can_refresh": can_refresh, "unified_source": unified, "internal_mode": internal_mode,
        })
    return sorted(result, key=lambda row: (row["name"].casefold(), row["id"]))


class Connection:
    def __init__(self, websocket, enrich=None):
        self.websocket = websocket
        self.enrich = enrich
        self.next_id = 0
        self.pending = {}
        self.events = asyncio.Queue(maxsize=1024)
        self.addresses = {}
        self.dhcp_started = False
        self.dhcp_ready = asyncio.Event()
        self.reader = asyncio.create_task(self.read_messages())

    async def read_messages(self):
        try:
            async for frame in self.websocket:
                if frame.type != aiohttp.WSMsgType.TEXT:
                    break
                message = frame.json()
                if message.get("type") == "result":
                    future = self.pending.get(message.get("id"))
                    if future and not future.done():
                        future.set_result(message)
                elif message.get("type") == "event":
                    event = message["event"]
                    if "add" in event:
                        for item in event["add"]:
                            try:
                                address = str(ipaddress.ip_address(item["ip_address"]))
                                mac = re.sub(r"[^0-9a-f]", "", item["mac_address"].lower())
                                if len(mac) == 12:
                                    self.addresses[mac] = address
                            except (ValueError, KeyError, TypeError):
                                continue
                        self.dhcp_ready.set()
                    self.events.put_nowait(event)
        except (aiohttp.ClientError, ValueError, TypeError, asyncio.QueueFull):
            pass
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(HomeAssistantError("Home Assistant connection closed."))
            if self.events.full():
                self.events.get_nowait()
            self.events.put_nowait(None)

    async def close(self):
        self.reader.cancel()
        await asyncio.gather(self.reader, return_exceptions=True)

    async def command(self, command_type, **kwargs):
        self.next_id += 1
        command_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[command_id] = future
        try:
            async with asyncio.timeout(10):
                if self.reader.done():
                    raise HomeAssistantError("Home Assistant connection closed.")
                await self.websocket.send_json({"id": command_id, "type": command_type, **kwargs})
                message = await future
                if not message.get("success"):
                    raise HomeAssistantError("Home Assistant rejected the request.")
                return message.get("result")
        finally:
            self.pending.pop(command_id, None)

    async def inventory(self):
        if not self.dhcp_started:
            self.dhcp_started = True
            try:
                await self.command("dhcp/subscribe_discovery")
                await asyncio.wait_for(self.dhcp_ready.wait(), timeout=2)
            except (HomeAssistantError, TimeoutError):
                # DHCP is optional; controls and other diagnostics remain available.
                pass
        devices = await self.command("config/device_registry/list")
        entities = await self.command("config/entity_registry/list")
        areas = await self.command("config/area_registry/list")
        states = await self.command("get_states")
        network_addresses(devices, self.addresses)
        if self.enrich:
            await self.enrich(devices, entities, states, areas)
        return devices, entities, states, areas


class HomeAssistant:
    def __init__(self, token, *, endpoint=HA_WEBSOCKET, confirmation_timeout=5):
        self.token = token
        self.endpoint = endpoint
        self.confirmation_timeout = confirmation_timeout
        self.models = {}
        # Last internal packet seen per device, restored when leaving external.
        self.internal_modes = {}

    async def enrich(self, session, devices, entities, states, areas):
        """Read ESPHome's original model, which HA may replace with project.name."""
        eligible = {row["id"] for row in build_inventory(devices, entities, states, areas)}
        entries = {}
        for entity in entities:
            if entity.get("platform") == "esphome" and entity.get("config_entry_id"):
                entries.setdefault(entity.get("device_id"), set()).add(entity["config_entry_id"])
        active = {entry for values in entries.values() for entry in values}
        self.models = {key: value for key, value in self.models.items() if key in active}
        base = self.endpoint.rsplit("/", 1)[0].replace("ws:", "http:", 1).replace("wss:", "https:", 1)
        for device in devices:
            candidates = entries.get(device["id"], set()) & set(device.get("config_entries", []))
            if device["id"] not in eligible or device_chip(device) or len(candidates) != 1:
                continue
            entry = next(iter(candidates))
            now = asyncio.get_running_loop().time()
            cached = self.models.get(entry)
            if not cached or cached[0] <= now or cached[1] != device.get("sw_version"):
                model = None
                try:
                    async with session.get(
                        base + "/api/diagnostics/config_entry/" + quote(entry, safe=""),
                        headers={"Authorization": "Bearer " + self.token},
                        timeout=aiohttp.ClientTimeout(total=3), allow_redirects=False,
                    ) as response:
                        if response.status == 200:
                            payload = await response.json()
                            info = payload["data"]["storage_data"]["device_info"]
                            model = device_chip({"model": info["model"]})
                except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError):
                    # Metadata is optional; never prevent controls or leak diagnostics.
                    pass
                cached = (now + (3600 if model else 60), device.get("sw_version"), model)
                self.models[entry] = cached
            device["_esphome_model"] = cached[2]

    @asynccontextmanager
    async def connect(self):
        if not self.token:
            raise HomeAssistantError("Home Assistant API access is not configured. Rebuild or update the app.")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10), trust_env=False) as session:
                async with session.ws_connect(self.endpoint, heartbeat=30, max_msg_size=16 * 1024 * 1024) as websocket:
                    async with asyncio.timeout(10):
                        if (await websocket.receive_json()).get("type") != "auth_required":
                            raise HomeAssistantError("Unexpected Home Assistant authentication response.")
                        await websocket.send_json({"type": "auth", "access_token": self.token})
                        if (await websocket.receive_json()).get("type") != "auth_ok":
                            raise HomeAssistantError("Home Assistant API authorization failed.")
                    async def enrich(devices, entities, states, areas):
                        await self.enrich(session, devices, entities, states, areas)
                    connection = Connection(websocket, enrich)
                    try:
                        yield connection
                    finally:
                        await connection.close()
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError) as error:
            raise HomeAssistantError("Home Assistant is unavailable or timed out. Refresh to check device state.") from error

    async def snapshot(self):
        async with self.connect() as connection:
            return build_inventory(*(await connection.inventory()))

    async def act(self, device_ids, action):
        """Resolve fresh registry entries; never accept entity IDs or services from the browser."""
        async with self.connect() as connection:
            inventory = await connection.inventory()
            rows = {row["id"]: row for row in build_inventory(*inventory)}
            results, pending = [], {}
            for device_id in device_ids:
                row = rows.get(device_id)
                result = {"id": device_id, "status": "error", "message": "Device is no longer available."}
                results.append(result)
                if row is None:
                    continue
                if not row["can_control"]:
                    result["message"] = row["reason"]
                    continue
                if row["fields"]["ownership"]["value"] == action:
                    result.update(status="unchanged", message="Already " + action + ".")
                    continue
                entity_id = row["fields"]["ownership"]["entity_id"]
                option = action
                if row["unified_source"]:
                    if row["internal_mode"]:
                        self.internal_modes[device_id] = row["internal_mode"]
                    option = "external" if action == "external" else self.internal_modes.get(device_id, "ping")
                try:
                    await connection.command("call_service", domain="select", service="select_option",
                                             target={"entity_id": entity_id}, service_data={"option": option})
                except HomeAssistantError:
                    result["message"] = "Home Assistant rejected the action."
                    continue
                result.update(status="unconfirmed", message="Command sent, but the device has not confirmed it. Refresh before retrying.")
                pending[entity_id] = (result, option)
            deadline = asyncio.get_running_loop().time() + self.confirmation_timeout
            while True:
                states = await connection.command("get_states")
                for state in states:
                    if state["entity_id"] in pending and state.get("state") == pending[state["entity_id"]][1]:
                        pending.pop(state["entity_id"])[0].update(status="confirmed", message="Now " + action + ".")
                if not pending or asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.25)
            devices, entities, _, areas = inventory
            return {"results": results, "devices": build_inventory(devices, entities, states, areas),
                    "read_at": datetime.now(timezone.utc).isoformat()}


class HomeAssistantFeed:
    """One HA subscription and diagnostic timer shared by visible panel clients."""

    def __init__(self, ha):
        self.ha = ha
        self.listeners = set()
        self.task = None
        self.latest = {"devices": [], "error": "Connecting to Home Assistant…"}

    def publish(self, payload):
        self.latest = payload
        for queue in self.listeners:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(payload)

    @contextlib.asynccontextmanager
    async def subscribe(self):
        queue = asyncio.Queue(maxsize=1)
        self.listeners.add(queue)
        queue.put_nowait(self.latest)
        if self.task is None:
            self.task = asyncio.create_task(self.run())
        try:
            yield queue
        finally:
            self.listeners.discard(queue)
            if not self.listeners:
                await self.close()

    async def close(self):
        task, self.task = self.task, None
        self.latest = {"devices": [], "error": "Connecting to Home Assistant…"}
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run(self):
        while True:
            try:
                async with self.ha.connect() as connection:
                    await self.session(connection)
            except (HomeAssistantError, aiohttp.ClientError, TimeoutError, ValueError, TypeError):
                self.publish({"devices": [], "error": "Home Assistant disconnected. Reconnecting…"})
                await asyncio.sleep(2)

    async def session(self, connection):
        for event_type in ("state_changed", "device_registry_updated", "entity_registry_updated", "area_registry_updated"):
            await connection.command("subscribe_events", event_type=event_type)
        inventory = await connection.inventory()
        devices, entities, state_list, areas = inventory
        states = {state["entity_id"]: state for state in state_list}
        rows = build_inventory(devices, entities, list(states.values()), areas)
        self.publish({"devices": rows})

        async def diagnostics():
            while True:
                started = asyncio.get_running_loop().time()
                failed = False
                for row in rows:
                    if row["can_refresh"]:
                        try:
                            await connection.command("call_service", domain="button", service="press",
                                                     target={"entity_id": row["fields"]["refresh"]["entity_id"]}, service_data={})
                        except HomeAssistantError:
                            failed = True
                if failed:
                    self.publish({"devices": rows, "error": "Diagnostics unavailable."})
                await asyncio.sleep(max(0.1, 1 - (asyncio.get_running_loop().time() - started)))

        refresh = asyncio.create_task(diagnostics())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(connection.events.get(), timeout=1)
                except TimeoutError:
                    if refresh.done():
                        raise HomeAssistantError("Diagnostic connection failed.") from None
                    continue
                if event is None or refresh.done():
                    raise HomeAssistantError("Home Assistant connection closed.")
                if "add" in event:
                    network_addresses(devices, connection.addresses)
                elif event.get("event_type") != "state_changed":
                    # Registry reads happen only at connection and on registry changes.
                    refresh.cancel()
                    await asyncio.gather(refresh, return_exceptions=True)
                    devices, entities, state_list, areas = await connection.inventory()
                    states = {state["entity_id"]: state for state in state_list}
                else:
                    data = event.get("data", {})
                    entity_id = data.get("entity_id")
                    relevant = {field.get("entity_id") for row in rows for field in row["fields"].values()}
                    if entity_id not in relevant:
                        continue
                    state = data.get("new_state")
                    previous = states.get(entity_id, {})
                    # Events queued during the initial snapshot must not roll it back.
                    timestamp = (state or data.get("old_state") or {}).get("last_updated", "")
                    if timestamp < previous.get("last_updated", ""):
                        continue
                    if state is None:
                        states.pop(entity_id, None)
                    else:
                        states[entity_id] = state
                rows = build_inventory(devices, entities, list(states.values()), areas)
                self.publish({"devices": rows})
                if refresh.done():
                    refresh = asyncio.create_task(diagnostics())
        finally:
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
