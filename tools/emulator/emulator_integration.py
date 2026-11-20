#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Exercise Wi-Fi rollback on Native C6 and the motion indicator on ESPHome S3."""
import argparse
import ipaddress
import json
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import emulator_csi_replay as replay


def check_rollback(client, console, log, port):
    # The pinned Espressif cosim_tcp.py implements the 1.2.0 wire framing.
    import cosim_tcp as cosim

    def waiting_timestamp(sock):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            _, payload = cosim.recv(sock)
            if payload is None:
                raise EOFError("emulator closed the co-simulation socket")
            message = cosim.decode(payload)
            if 41 in message:  # EmulatorMessage.time_waiting
                return message[1]
        raise TimeoutError("emulator did not acknowledge the clock grant")

    before = client.request("get", "wifi")
    if before.get("ssid") != replay.EMULATOR_WIFI_SSID or not before.get("connected"):
        raise AssertionError(f"unexpected starting Wi-Fi state: {before}")
    if ipaddress.ip_address(before["ip"]) not in ipaddress.ip_network(replay.EMULATOR_SUBNET):
        raise AssertionError(f"station address outside the configured subnet: {before['ip']}")
    _, candidate = replay.improv_wifi_settings("espectre-absent")
    offset = log.stat().st_size

    def states():
        with log.open("rb") as stream:
            stream.seek(offset)
            return [(f.packet_type, f.data) for f in replay.ImprovFrameParser().feed(stream.read())]

    pending = (replay.ImprovPacketType.CURRENT_STATE, bytes([replay.ImprovState.PROVISIONING]))
    failed = (replay.ImprovPacketType.ERROR_STATE, b"\x03")  # Improv ERROR_UNABLE_TO_CONNECT
    recovered = (replay.ImprovPacketType.CURRENT_STATE, bytes([replay.ImprovState.PROVISIONED]))
    with socket.create_connection(("127.0.0.1", port), timeout=45) as sock:
        cosim.send(sock, cosim.register("ESPectre Wi-Fi rollback", cosim.ld(23, b"")))  # Claim.time
        start = waiting_timestamp(sock)

        def advance(seconds):
            grant = cosim.field(1, cosim.varint(round(seconds * 1e9)))  # TimeGrant.advance_ns
            cosim.send(sock, cosim.ld(40, grant))  # EmulatorMessage.time_grant
            return (waiting_timestamp(sock) - start) / 1e9

        console.write(candidate)
        advance(0.5)
        replay.wait_for(lambda: pending in states(), 5)
        before_timeout = advance(29)
        if abs(before_timeout - 29.5) > 0.001 or failed in states() or recovered in states():
            raise AssertionError("candidate did not remain pending before the 30 s timeout")
        after_timeout = advance(10)
        if abs(after_timeout - 39.5) > 0.001:
            raise AssertionError(f"unexpected clock advance: {after_timeout} s")
        replay.wait_for(lambda: failed in states() and recovered in states(), 5)

    after = client.request("get", "wifi")
    if after.get("apply_state") != "rolled_back" or not after.get("connected") or after["ssid"] != before["ssid"]:
        raise AssertionError(f"previous Wi-Fi connection was not restored: {after}")
    return {"pending_at_s": before_timeout, "restored_by_s": after_timeout,
            "ssid": after["ssid"], "ip": after["ip"], "apply_state": after["apply_state"]}


def write_s3_recording(path):
    packets, segments = replay.build_stream("s3")
    # Give startup calibration 30 quiet seconds, then repeat both edges in a short loop.
    durations = (30, 16, 16)
    with path.open("w") as output:
        for packet in packets:
            if packet.stream_us / 1e6 - segments[packet.segment].start_s >= durations[packet.segment]:
                continue
            # ESP32-S3 hardware presents DC first. The replay fixture is centered on DC.
            native = np.fft.ifftshift(packet.csi)
            print(packet.rssi_dbm, packet.noise_floor_dbm, native.tobytes().hex(), file=output)


def check_led(client, port):
    client.start_events()
    red = None
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock, sock.makefile("rb") as control:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            sock.sendall(b"part strip state\n")
            reply = control.readline().decode().strip()
            if not reply.startswith("ok"):
                raise RuntimeError(f"WS2812 model returned {reply!r}")
            pixel = re.search(r"\b0=#([0-9a-fA-F]{6})\b", reply)
            event = next((event for event in reversed(client.events) if event.name == "motion"), None)
            if pixel and event:
                r, g, b = bytes.fromhex(pixel[1])
                if r and not g and not b and event.data["state"] == "motion":
                    red = pixel[1]
                if red and g and not r and not b and event.data["state"] == "idle":
                    return {"motion_rgb": red, "idle_rgb": pixel[1], "gpio": 48}
            time.sleep(0.25)
    raise AssertionError(f"missing WS2812 motion-to-idle transition (observed red: {red})")


def run(args, tmp):
    password, credentials = replay.improv_wifi_settings()
    led = args.case == "led"
    cmd = [args.emulator, "--firmware", str(args.firmware), "--chip", "esp32s3" if led else "esp32c6",
           "--wifi-ssid", replay.EMULATOR_WIFI_SSID, "--wifi-password", password,
           "--net", replay.user_network(args.direct_port), "--timeout", "500s"]
    endpoint = f"127.0.0.1:{args.control_port}"
    if led:
        recording = tmp / "s3.csi"
        write_s3_recording(recording)
        board = tmp / "board.json"
        board.write_text(json.dumps({"parts": {"strip": {"module": str(args.ws2812.resolve()), "pins": {"DIN": 48}}}}))
        cmd.extend(["--cosim-board", str(board), "--control-tcp", endpoint,
                    "--csi-replay", str(recording), "--csi-replay-loop"])
    else:
        cmd.extend(["--cosim-tcp", endpoint])
    client = replay.DirectClient(f"http://127.0.0.1:{args.direct_port}/espectre/v1", timeout=replay.DIRECT_TIMEOUT_S)
    with args.log.open("wb") as log, subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as emulator:
        console = replay.EmulatorConsole(emulator, log)
        try:
            replay.provision_wifi(console, credentials)
            replay.wait_for(lambda: client.request("get", "health", timeout=2)["status"] == "ok", 60)
            if led:
                return check_led(client, args.control_port)
            return check_rollback(client, console, args.log, args.control_port)
        finally:
            client.close()
            emulator.terminate()
            emulator.wait(timeout=30)
            console.join(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("rollback", "led"))
    parser.add_argument("--emulator", default="esp-emu")
    parser.add_argument("--firmware", required=True, type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ws2812", type=Path, help="ws2812.wasm from the esp-emu examples kit")
    parser.add_argument("--direct-port", type=int, default=18690)
    parser.add_argument("--control-port", type=int, default=5590)
    args = parser.parse_args()
    if args.case == "led" and not args.ws2812:
        parser.error("led requires --ws2812")
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(run(args, Path(directory)), indent=2))
