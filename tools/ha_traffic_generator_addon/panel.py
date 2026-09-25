#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Ingress-only panel; UDP pacing runs independently in the shared generator."""

import asyncio
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import time
from datetime import datetime, timezone
from dataclasses import dataclass

from aiohttp import web

if __package__:
    from .espectre_traffic_generator import ExternalTrafficGenerator
    from .ha_client import HomeAssistant, HomeAssistantError, HomeAssistantFeed
    from .traffic import load_options
else:
    from espectre_traffic_generator import ExternalTrafficGenerator
    from ha_client import HomeAssistant, HomeAssistantError, HomeAssistantFeed
    from traffic import load_options


INGRESS_ADDRESS = "172.30.32.2"
STATIC = Path(__file__).with_name("static")
HA = web.AppKey("ha", HomeAssistant)
GENERATOR = web.AppKey("generator", ExternalTrafficGenerator)
CSRF = web.AppKey("csrf", str)
ACTION_LOCK = web.AppKey("action_lock", asyncio.Lock)
ALLOWED_PEER = web.AppKey("allowed_peer", str)
FEED = web.AppKey("feed", HomeAssistantFeed)
STREAMS = web.AppKey("streams", set)

@dataclass
class GeneratorState:
    expected_running: bool


GENERATOR_STATE = web.AppKey("generator_state", GeneratorState)


@web.middleware
async def protect_ingress(request, handler):
    # Inspect the socket peer, never a browser-supplied X-Forwarded-For header.
    if request.remote != request.app[ALLOWED_PEER]:
        raise web.HTTPForbidden(text="Home Assistant Ingress is required.")
    if request.method not in {"GET", "HEAD"}:
        if (request.content_type != "application/json" or
                not hmac.compare_digest(request.headers.get("X-ESPectre-CSRF", ""), request.app[CSRF])):
            raise web.HTTPForbidden(text="Invalid request token.")
    try:
        response = await handler(request)
    except HomeAssistantError as error:
        response = web.json_response({"error": str(error)}, status=503)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
    )
    return response


def generator_status(generator):
    return {"running": generator.running, "targets": list(generator.targets),
            "port": generator.port, "rate_pps": generator.rate_pps,
            "sent_packets": generator.sent_packets, "send_errors": generator.send_errors}


async def status(request):
    return web.json_response({"generator": generator_status(request.app[GENERATOR]),
                              "csrf": request.app[CSRF]})


async def devices(request):
    return web.json_response({"devices": await request.app[HA].snapshot(),
                              "read_at": datetime.now(timezone.utc).isoformat()})


async def stream(request):
    websocket = web.WebSocketResponse(heartbeat=20, max_msg_size=1024)
    await websocket.prepare(request)
    try:
        async with asyncio.timeout(5):
            auth = await websocket.receive_json()
        if (not isinstance(auth, dict) or not isinstance(auth.get("csrf"), str)
                or not hmac.compare_digest(auth["csrf"], request.app[CSRF])):
            await websocket.close(code=1008)
            return websocket
    except (TimeoutError, ValueError, TypeError):
        await websocket.close(code=1008)
        return websocket

    request.app[STREAMS].add(websocket)
    async with request.app[FEED].subscribe() as queue:
        async def send_updates():
            try:
                while not websocket.closed:
                    try:
                        payload = dict(await asyncio.wait_for(queue.get(), timeout=1))
                    except TimeoutError:
                        payload = {}
                    payload["generator"] = generator_status(request.app[GENERATOR])
                    await websocket.send_json(payload)
            except Exception:
                await websocket.close()
                raise

        sender = asyncio.create_task(send_updates())
        try:
            async for _message in websocket:
                # Commands use the bounded, CSRF-protected HTTP endpoints only.
                await websocket.close(code=1008)
        finally:
            request.app[STREAMS].discard(websocket)
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
    return websocket


async def close_feed(app):
    sockets = list(app[STREAMS])
    await asyncio.gather(*(socket.close(code=1001) for socket in sockets), return_exceptions=True)
    app[STREAMS].difference_update(sockets)
    await app[FEED].close()


async def action(request):
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest(text="Expected a JSON object.") from None
    if not isinstance(payload, dict) or set(payload) != {"device_ids", "action"}:
        raise web.HTTPBadRequest(text="Expected device_ids and action.")
    device_ids, operation = payload["device_ids"], payload["action"]
    if (not isinstance(operation, str) or operation not in {"internal", "external"}
            or not isinstance(device_ids, list) or not 1 <= len(device_ids) <= 32
            or any(not isinstance(value, str) or not value or len(value) > 128 for value in device_ids)
            or len(set(device_ids)) != len(device_ids)):
        raise web.HTTPBadRequest(text="Select 1–32 unique devices and a supported action.")
    if operation == "external" and not request.app[GENERATOR].running:
        raise web.HTTPConflict(text="The traffic generator is not running.")
    lock = request.app[ACTION_LOCK]
    if lock.locked():
        raise web.HTTPConflict(text="Another action is in progress. Refresh before retrying.")
    async with lock:
        return web.json_response(await request.app[HA].act(device_ids, operation))


async def index(_request):
    return web.FileResponse(STATIC / "index.html")


async def logo(_request):
    return web.FileResponse(Path(__file__).with_name("icon.png"))


async def control_generator(request):
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise web.HTTPBadRequest(text="Expected a JSON object.") from None
    if (not isinstance(payload, dict) or set(payload) != {"action"}
            or payload["action"] not in ("start", "stop")):
        raise web.HTTPBadRequest(text="Expected start or stop.")
    lock = request.app[ACTION_LOCK]
    if lock.locked():
        raise web.HTTPConflict(text="Another action is in progress.")
    async with lock:
        generator = request.app[GENERATOR]
        expected = payload["action"] == "start"
        await asyncio.to_thread(generator.start if expected else generator.stop)
        request.app[GENERATOR_STATE].expected_running = expected
        logging.info("UDP generator %s requested from panel", payload["action"])
        return web.json_response({"generator": generator_status(generator)})


def generator_failed(app):
    return app[GENERATOR_STATE].expected_running and not app[ACTION_LOCK].locked() and not app[GENERATOR].running


def create_app(ha, generator, *, allowed_peer=INGRESS_ADDRESS):
    app = web.Application(middlewares=[protect_ingress], client_max_size=16 * 1024)
    app[HA], app[GENERATOR] = ha, generator
    app[CSRF], app[ACTION_LOCK] = secrets.token_urlsafe(32), asyncio.Lock()
    app[ALLOWED_PEER] = allowed_peer
    app[GENERATOR_STATE] = GeneratorState(generator.running)
    app[FEED] = HomeAssistantFeed(ha)
    app[STREAMS] = set()
    app.on_shutdown.append(close_feed)
    app.router.add_get("/", index)
    app.router.add_get("/logo.png", logo)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/devices", devices)
    app.router.add_get("/api/stream", stream)
    app.router.add_post("/api/action", action)
    app.router.add_post("/api/generator", control_generator)
    # Serve only public assets, not configuration, sources, or /data.
    app.router.add_static("/static/", STATIC, show_index=False, follow_symlinks=False)
    return app


async def serve():
    options = load_options()
    ingress_port = int(os.environ["ESPECTRE_INGRESS_PORT"])
    if not 1 <= ingress_port <= 65535:
        raise ValueError("Supervisor did not assign a valid Ingress port")
    generator = ExternalTrafficGenerator(**options)
    ha = HomeAssistant(os.environ.get("SUPERVISOR_TOKEN", ""))
    app = create_app(ha, generator)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=10)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", ingress_port).start()
        generator.start()
        app[GENERATOR_STATE].expected_running = True
        logging.info("UDP generator started: %s", json.dumps(options))
        logging.info("Panel ready through Home Assistant Ingress on port %s", ingress_port)
        next_report = time.monotonic() + 60
        reported_errors = 0
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except TimeoutError:
                pass
            if generator_failed(app):
                raise RuntimeError("UDP generator stopped unexpectedly")
            if time.monotonic() >= next_report:
                # The panel shows live counters; the log reports only new send errors.
                send_errors = generator.send_errors
                if send_errors > reported_errors:
                    logging.warning("UDP send errors: %s in the last minute, %s in total",
                                    send_errors - reported_errors, send_errors)
                    reported_errors = send_errors
                next_report = time.monotonic() + 60
    finally:
        await runner.cleanup()
        generator.stop()
        logging.info("External CSI traffic generator stopped after %s packets; send errors: %s",
                     generator.sent_packets, generator.send_errors)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
