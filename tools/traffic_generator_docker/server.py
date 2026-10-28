#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Standalone web UI for the shared external traffic generator and the devices it serves."""

import asyncio
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import sys
import time
from dataclasses import dataclass

from aiohttp import web

ADDON = Path(__file__).resolve().parent.parent / "ha_traffic_generator_addon"
STATIC = Path(__file__).with_name("static")

if __package__:
    from ..ha_traffic_generator_addon.espectre_traffic_generator import (
        DSCP, MULTICAST_TTL, PORT, RATE, ExternalTrafficGenerator)
    from ..ha_traffic_generator_addon.traffic import validate_options
    from .devices import INTERNAL, TRAFFIC_MODES, DeviceRegistry
else:
    sys.path.insert(0, str(ADDON))
    # A repository checkout supplies the CLI's discovery and Direct client; the image copies them.
    REPO_PYTHON = Path(__file__).resolve().parents[2] / "src" / "python"
    if REPO_PYTHON.is_dir():
        sys.path.insert(0, str(REPO_PYTHON))
    from espectre_traffic_generator import DSCP, MULTICAST_TTL, PORT, RATE, ExternalTrafficGenerator
    from traffic import validate_options
    from devices import INTERNAL, TRAFFIC_MODES, DeviceRegistry


# Ping is the recommended source; a new installation has no UDP setups to keep.
DEFAULT_SETTINGS = {"targets": ["239.255.0.1"], "protocol": "icmp", "port": PORT, "rate_pps": RATE,
                    "source_ip": "", "multicast_ttl": MULTICAST_TTL, "dscp": DSCP}
MAX_TARGETS = 32
WEB_PORT = 8099
SESSION_COOKIE = "espectre_session"
SESSION_SECONDS = 30 * 24 * 3600
MAX_SESSIONS = 32
LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 60
MIN_TOKEN_LENGTH = 12
MAX_ACTION_DEVICES = 32
# Devices are read only while someone has the web UI open.
DEVICE_REFRESH_SECONDS = 3
DEVICE_SCAN_SECONDS = 300
VIEWER_TIMEOUT_SECONDS = 15
PUBLIC_ROUTES = {"/", "/logo.png", "/api/login"}


def normalize_settings(settings):
    """Validate settings with the add-on rules; return the stored form."""
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    targets = settings.get("targets")
    if isinstance(targets, list) and len(targets) > MAX_TARGETS:
        raise ValueError(f"list at most {MAX_TARGETS} targets")
    options = validate_options(settings)
    if len(set(options["targets"])) != len(options["targets"]):
        raise ValueError("targets must not repeat")
    return {**options, "source_ip": options["source_ip"] or ""}


def generator_options(settings):
    return {**settings, "source_ip": settings["source_ip"] or None}


def load_settings(path):
    """Read saved settings, filling options added after they were saved."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    if not isinstance(saved, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return normalize_settings({**DEFAULT_SETTINGS, **saved})


def write_private_file(path, text, mode):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_token(data_dir, configured=None):
    """Return the sign-in token: configured, saved, or generated on first start."""
    configured = (os.environ.get("ESPECTRE_UI_TOKEN", "") if configured is None else configured).strip()
    path = data_dir / "ui_token"
    if not configured:
        try:
            configured = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            configured = secrets.token_urlsafe(18)
            write_private_file(path, configured + "\n", 0o600)
            logging.info("Web UI sign-in token: %s (saved in %s)", configured, path)
    if len(configured) < MIN_TOKEN_LENGTH:
        raise ValueError(f"the web UI token needs at least {MIN_TOKEN_LENGTH} characters")
    return configured


@dataclass
class Session:
    csrf: str
    expires: float


class Sessions:
    """In-memory browser sessions; a restart signs everyone out."""

    def __init__(self, token):
        self.token = token
        self.entries = {}
        self.failures = []

    def throttled(self):
        now = time.monotonic()
        self.failures = [moment for moment in self.failures if now - moment < LOGIN_WINDOW_SECONDS]
        return len(self.failures) >= LOGIN_FAILURES

    def matches(self, candidate):
        if hmac.compare_digest(candidate.encode(), self.token.encode()):
            return True
        self.failures.append(time.monotonic())
        return False

    def create(self):
        now = time.monotonic()
        self.entries = {key: value for key, value in self.entries.items() if value.expires > now}
        while len(self.entries) >= MAX_SESSIONS:
            self.entries.pop(min(self.entries, key=lambda key: self.entries[key].expires))
        session_id = secrets.token_urlsafe(32)
        self.entries[session_id] = Session(secrets.token_urlsafe(32), now + SESSION_SECONDS)
        return session_id

    def lookup(self, session_id):
        session = self.entries.get(session_id or "")
        if session is None or session.expires <= time.monotonic():
            return None
        return session


class Controller:
    """Own the running generator and replace it when settings change."""

    def __init__(self, settings, path):
        self.settings = settings
        self.path = path
        self.generator = ExternalTrafficGenerator(**generator_options(settings))
        self.expected_running = False
        self.lock = asyncio.Lock()

    @property
    def failed(self):
        return self.expected_running and not self.generator.running

    def status(self):
        generator = self.generator
        failed = self.failed
        return {"running": generator.running, "failed": failed,
                "error": (generator.error or "The generator stopped.") if failed else None,
                "targets": list(generator.targets), "protocol": generator.protocol,
                "port": generator.port, "rate_pps": generator.rate_pps,
                "sent_packets": generator.sent_packets, "send_errors": generator.send_errors,
                "replies_received": generator.replies_received,
                "sent_by_target": dict(generator.sent_by_target),
                "errors_by_target": dict(generator.errors_by_target)}

    def start(self):
        self.generator.start()
        self.expected_running = True

    def stop(self):
        self.generator.stop()
        self.expected_running = False

    def apply(self, settings):
        # Build the replacement first, so a rejected value leaves traffic untouched.
        generator = ExternalTrafficGenerator(**generator_options(settings))
        write_private_file(self.path, json.dumps(settings, indent=2) + "\n", 0o644)
        self.generator.stop()
        self.generator, self.settings = generator, settings
        if self.expected_running:
            generator.start()


CONTROLLER = web.AppKey("controller", Controller)
SESSIONS = web.AppKey("sessions", Sessions)
SESSION = web.RequestKey("session", Session)
DEVICES = web.AppKey("devices", DeviceRegistry)
VIEWED_AT = web.AppKey("viewed_at", list)
TASKS = web.AppKey("tasks", set)


async def read_json(request):
    if request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType(text="Expected JSON.")
    try:
        return await request.json()
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest(text="Expected a JSON object.") from None


@web.middleware
async def protect(request, handler):
    if request.path not in PUBLIC_ROUTES and not request.path.startswith("/static/"):
        session = request.app[SESSIONS].lookup(request.cookies.get(SESSION_COOKIE))
        if session is None:
            raise web.HTTPUnauthorized(text="Sign in required.")
        if request.method not in {"GET", "HEAD"} and not hmac.compare_digest(
                request.headers.get("X-ESPectre-CSRF", ""), session.csrf):
            raise web.HTTPForbidden(text="Invalid request token. Reload the page.")
        request[SESSION] = session
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return response


def status_response(request, **extra):
    controller = request.app[CONTROLLER]
    request.app[VIEWED_AT][0] = time.monotonic()
    return web.json_response({"generator": controller.status(), "settings": controller.settings,
                              "devices": request.app[DEVICES].snapshot(), "csrf": request[SESSION].csrf,
                              **extra})


async def login(request):
    sessions = request.app[SESSIONS]
    if sessions.throttled():
        raise web.HTTPTooManyRequests(text="Too many failed sign-ins. Wait a minute and try again.")
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != {"token"} or not isinstance(payload["token"], str):
        raise web.HTTPBadRequest(text="Expected a token.")
    if not sessions.matches(payload["token"].strip()):
        logging.warning("Rejected a web UI sign-in from %s", request.remote)
        raise web.HTTPUnauthorized(text="Wrong token.")
    response = web.json_response({"signed_in": True})
    response.set_cookie(SESSION_COOKIE, sessions.create(), max_age=SESSION_SECONDS, path="/",
                        httponly=True, samesite="Strict", secure=request.secure)
    return response


async def logout(request):
    sessions = request.app[SESSIONS]
    sessions.entries.pop(request.cookies.get(SESSION_COOKIE), None)
    response = web.json_response({"signed_in": False})
    response.del_cookie(SESSION_COOKIE, path="/")
    return response


async def status(request):
    return status_response(request)


async def control_generator(request):
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != {"action"} or payload["action"] not in ("start", "stop"):
        raise web.HTTPBadRequest(text="Expected start or stop.")
    controller = request.app[CONTROLLER]
    if controller.lock.locked():
        raise web.HTTPConflict(text="Another change is in progress.")
    async with controller.lock:
        await asyncio.to_thread(controller.start if payload["action"] == "start" else controller.stop)
    logging.info("Traffic generator %s requested from the web UI", payload["action"])
    return status_response(request)


async def update_settings(request):
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != set(DEFAULT_SETTINGS):
        raise web.HTTPBadRequest(text="Expected every setting.")
    try:
        settings = normalize_settings(payload)
    except (ValueError, TypeError) as error:
        raise web.HTTPBadRequest(text=str(error)) from None
    controller = request.app[CONTROLLER]
    if controller.lock.locked():
        raise web.HTTPConflict(text="Another change is in progress.")
    async with controller.lock:
        await asyncio.to_thread(controller.apply, settings)
    logging.info("Settings changed from the web UI: %s", json.dumps(settings))
    return status_response(request)


async def scan_devices(request):
    payload = await read_json(request)
    if payload != {}:
        raise web.HTTPBadRequest(text="Expected an empty object.")
    registry = request.app[DEVICES]
    if not registry.scanning:
        # Searching takes several seconds; the status poll shows the result.
        task = asyncio.create_task(asyncio.to_thread(registry.scan))
        request.app[TASKS].add(task)
        task.add_done_callback(request.app[TASKS].discard)
        await asyncio.sleep(0)
    return status_response(request)


async def add_device(request):
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != {"address"} or not isinstance(payload["address"], str):
        raise web.HTTPBadRequest(text="Expected an address.")
    try:
        device = await asyncio.to_thread(request.app[DEVICES].add, payload["address"][:64])
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error)) from None
    logging.info("Device %s added at %s", device.name, device.ip_address)
    return status_response(request)


async def remove_device(request):
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != {"device_id"} or not isinstance(payload["device_id"], str):
        raise web.HTTPBadRequest(text="Expected a device ID.")
    try:
        await asyncio.to_thread(request.app[DEVICES].remove, payload["device_id"])
    except KeyError:
        raise web.HTTPNotFound(text="Only devices added by address can be removed.") from None
    return status_response(request)


async def set_device_mode(request):
    payload = await read_json(request)
    if not isinstance(payload, dict) or set(payload) != {"device_ids", "mode"}:
        raise web.HTTPBadRequest(text="Expected device_ids and mode.")
    device_ids, mode = payload["device_ids"], payload["mode"]
    if (mode not in (*TRAFFIC_MODES, INTERNAL) or not isinstance(device_ids, list)
            or not 1 <= len(device_ids) <= MAX_ACTION_DEVICES
            or any(not isinstance(value, str) or not value or len(value) > 64 for value in device_ids)
            or len(set(device_ids)) != len(device_ids)):
        raise web.HTTPBadRequest(text=f"Select 1-{MAX_ACTION_DEVICES} unique devices and a supported traffic source.")
    controller = request.app[CONTROLLER]
    if mode == "external" and not controller.generator.running:
        raise web.HTTPConflict(text="Start the traffic generator before switching devices to external.")
    if controller.lock.locked():
        raise web.HTTPConflict(text="Another change is in progress.")
    async with controller.lock:
        results = await asyncio.to_thread(request.app[DEVICES].set_mode, device_ids, mode)
    return status_response(request, results=results)


async def finish_tasks(app):
    tasks = list(app[TASKS])
    await asyncio.gather(*tasks, return_exceptions=True)
    app[TASKS].clear()


async def watch_devices(app):
    """Search at start, then read devices only while someone watches the web UI."""
    registry, next_scan = app[DEVICES], 0.0
    while True:
        now = time.monotonic()
        watched = now - app[VIEWED_AT][0] < VIEWER_TIMEOUT_SECONDS
        if now >= next_scan and (watched or next_scan == 0.0):
            await asyncio.to_thread(registry.scan)
            next_scan = time.monotonic() + DEVICE_SCAN_SECONDS
        elif watched:
            await asyncio.to_thread(registry.refresh_all)
        await asyncio.sleep(DEVICE_REFRESH_SECONDS)


async def index(_request):
    return web.FileResponse(STATIC / "index.html")


async def logo(_request):
    return web.FileResponse(ADDON / "icon.png")


async def shared_stylesheet(_request):
    return web.FileResponse(ADDON / "static" / "panel.css")


def create_app(controller, token, registry):
    app = web.Application(middlewares=[protect], client_max_size=16 * 1024)
    app[CONTROLLER], app[SESSIONS], app[DEVICES] = controller, Sessions(token), registry
    app[VIEWED_AT], app[TASKS] = [float("-inf")], set()
    app.on_shutdown.append(finish_tasks)
    app.router.add_get("/", index)
    app.router.add_get("/logo.png", logo)
    # The add-on stylesheet supplies the shared look; static/app.css adds the forms.
    app.router.add_get("/static/panel.css", shared_stylesheet)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/logout", logout)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/generator", control_generator)
    app.router.add_post("/api/settings", update_settings)
    app.router.add_post("/api/devices/scan", scan_devices)
    app.router.add_post("/api/devices/add", add_device)
    app.router.add_post("/api/devices/remove", remove_device)
    app.router.add_post("/api/devices/mode", set_device_mode)
    app.router.add_static("/static/", STATIC, show_index=False, follow_symlinks=False)
    return app


def web_port():
    port = int(os.environ.get("ESPECTRE_WEB_PORT", WEB_PORT))
    if not 1 <= port <= 65535:
        raise ValueError("ESPECTRE_WEB_PORT must be in the 1-65535 range")
    return port


async def serve():
    data_dir = Path(os.environ.get("ESPECTRE_DATA_DIR", "/data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    host, port = os.environ.get("ESPECTRE_WEB_HOST", "0.0.0.0"), web_port()
    controller = Controller(load_settings(data_dir / "settings.json"), data_dir / "settings.json")
    registry = DeviceRegistry(data_dir / "devices.json")
    app = create_app(controller, load_token(data_dir), registry)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=10)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        controller.start()
        logging.info("Traffic generator started: %s", json.dumps(controller.settings))
        logging.info("Web UI listening on %s:%s", host, port)
        watcher = asyncio.create_task(watch_devices(app))
        app[TASKS].add(watcher)
        next_report = time.monotonic() + 60
        reported_errors = 0
        reported_failure = None
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except TimeoutError:
                pass
            generator = controller.generator
            # A failed generator stays visible in the web UI instead of restarting the container.
            failure = generator if controller.failed and not controller.lock.locked() else None
            if failure is not None and failure is not reported_failure:
                logging.error("Traffic generator stopped: %s", controller.status()["error"])
            reported_failure = failure
            if time.monotonic() >= next_report:
                if generator.send_errors > reported_errors:
                    logging.warning("Send errors: %s in the last minute, %s in total",
                                    generator.send_errors - reported_errors, generator.send_errors)
                reported_errors = generator.send_errors
                next_report = time.monotonic() + 60
    finally:
        for task in app[TASKS]:
            task.cancel()
        await runner.cleanup()
        controller.generator.stop()
        logging.info("External CSI traffic generator stopped after %s packets; send errors: %s",
                     controller.generator.sent_packets, controller.generator.send_errors)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
