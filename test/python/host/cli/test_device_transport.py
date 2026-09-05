# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Device Transport contracts."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from types import SimpleNamespace
import pytest
from src.python.espectre_cli.device_transport import (
    DIRECT_MAX_REQUEST_FRAME_SIZE,
    DIRECT_MAX_RESPONSE_FRAME_SIZE,
    DirectClient,
    DirectEventStreamTransportError,
    DirectProtocolError,
    ImprovCommand,
    ImprovFrameParser,
    ImprovPacketType,
    ImprovProtocolError,
    ImprovSerialClient,
    direct_endpoint_from_device_url,
    encode_improv_frame,
    encode_improv_rpc,
    parse_improv_rpc_response,
)
from src.python.espectre_cli import device_transport
from src.python.micro_espectre import protocol
class _FakeHttpResponse:
    status = 200

    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        return json.dumps(self.payload).encode()

    def close(self) -> None:
        self.closed = True

class _FakeEventResponse:
    status = 200

    def __init__(self, lines: list[bytes]):
        self.lines = list(lines)
        self.closed = threading.Event()

    def readline(self, _size: int = -1) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        self.closed.wait(1.0)
        return b""

    def close(self) -> None:
        self.closed.set()

class _ClosingEventResponse(_FakeEventResponse):
    class EventSocket:
        def __init__(self, closed):
            self.closed = closed
            self.timeout = object()
            self.shutdown_mode = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def shutdown(self, mode):
            self.shutdown_mode = mode
            self.closed.set()

    def __init__(self, lines):
        super().__init__(lines)
        self.event_socket = self.EventSocket(self.closed)
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.event_socket))

    def readline(self, _size: int = -1) -> bytes:
        self.closed.wait(1.0)
        raise AttributeError("response stream was closed")


class _GatedEventResponse(_FakeEventResponse):
    def __init__(self, lines: list[bytes]):
        super().__init__(lines)
        self.readable = threading.Event()

    def readline(self, _size: int = -1) -> bytes:
        self.readable.wait(1.0)
        if self.lines:
            return self.lines.pop(0)
        return b""


class _IncompleteEventResponse(_GatedEventResponse):
    def readline(self, _size: int = -1) -> bytes:
        self.readable.wait(1.0)
        raise http.client.IncompleteRead(b"")


def _improv_rpc_response(command: ImprovCommand, values: list[str]) -> bytes:
    encoded = [value.encode() for value in values]
    data = bytes((int(command), sum(len(value) + 1 for value in encoded))) + b"".join(
        bytes((len(value),)) + value for value in encoded
    )
    return encode_improv_frame(ImprovPacketType.RPC_RESPONSE, data)

def test_improv_parser_recovers_fragmented_frames_from_console_noise():
    encoded = encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
    parser = ImprovFrameParser()

    assert parser.feed(b"I (12) boot log\nIM") == []
    assert parser.feed(b"PRO" + encoded[5:8]) == []
    frames = parser.feed(encoded[8:] + b"trailing log")

    assert frames[0].packet_type == ImprovPacketType.CURRENT_STATE
    assert frames[0].data == b"\x04"

def test_improv_parser_rejects_bad_checksum():
    encoded = bytearray(encode_improv_rpc(ImprovCommand.GET_DEVICE_INFO))
    encoded[-2] ^= 0xFF

    with pytest.raises(ImprovProtocolError, match="checksum"):
        ImprovFrameParser().feed(bytes(encoded))

def test_improv_rpc_frame_ends_with_line_feed_after_checksum():
    encoded = encode_improv_rpc(ImprovCommand.GET_CURRENT_STATE)

    assert encoded[-1:] == b"\n"
    assert sum(encoded[:-2]) & 0xFF == encoded[-2]

def test_improv_rpc_response_validates_lengths_and_utf8():
    frame = ImprovFrameParser().feed(
        _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
    )[0]

    command, values = parse_improv_rpc_response(frame.data)

    assert command == ImprovCommand.WIFI_SETTINGS
    assert values == ("http://192.0.2.10",)

    # ESPHome includes Improv SDK's reserved inner checksum slot in the outer
    # serial frame when checksum generation is disabled.
    command, values = parse_improv_rpc_response(frame.data + b"\x00")

    assert command == ImprovCommand.WIFI_SETTINGS
    assert values == ("http://192.0.2.10",)
    with pytest.raises(ImprovProtocolError, match="length"):
        parse_improv_rpc_response(b"\x01\x01\x01x")
    with pytest.raises(ImprovProtocolError, match="length"):
        parse_improv_rpc_response(frame.data + b"\x01")


def test_improv_client_does_not_touch_modem_control_lines():
    class FakeSerial:
        def __init__(self):
            self.dtr = True
            self.rts = True

        def close(self) -> None:
            pass

    serial = FakeSerial()
    client = ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial)

    assert serial.dtr is True
    assert serial.rts is True
    client.close()


def test_improv_client_handles_multiple_frames_in_one_serial_read():
    class FakeSerial:
        def __init__(self, **_kwargs):
            self.writes: list[bytes] = []
            self.closed = False
            device_info = _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"])
            provisioned_url = _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
            self.reads = [
                b"boot log\n"
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x02")
                + device_info
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x03")
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + provisioned_url
            ]

        def read(self, _size: int = 1) -> bytes:
            return self.reads.pop(0) if self.reads else b""

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    serial = FakeSerial()
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial) as client:
        result = client.provision("Lab", "secret", timeout=1.0)

    assert result.endpoint == "http://192.0.2.10"
    assert result.states == ("authorized", "provisioning", "provisioned")
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert len(serial.writes) == 3
    assert serial.closed

def test_improv_client_retries_initial_state_query_after_flash(monkeypatch):
    class FakeClock:
        now = 0.0

        def monotonic(self):
            self.now += 0.75
            return self.now

    class FakeSerial:
        def __init__(self):
            self.writes: list[bytes] = []
            self.responses_sent = False

        def read(self, _size: int = 1) -> bytes:
            if len(self.writes) < 2 or self.responses_sent:
                return b""
            self.responses_sent = True
            return (
                encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x02")
                + _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "s3", "Native"])
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
            )

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            pass

    clock = FakeClock()
    serial = FakeSerial()
    monkeypatch.setattr(device_transport.time, "monotonic", clock.monotonic)

    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial) as client:
        result = client.provision("Lab", "secret", timeout=10.0)

    assert result.endpoint == "http://192.0.2.10"
    assert len(serial.writes) == 4

def test_improv_client_ignores_current_state_url_before_device_info():
    class FakeSerial:
        def __init__(self, **_kwargs):
            self.writes: list[bytes] = []
            self.closed = False
            self.reads = [
                encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + _improv_rpc_response(ImprovCommand.GET_CURRENT_STATE, ["http://192.0.2.10"])
                + _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"])
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x03")
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
            ]

        def read(self, _size: int = 1) -> bytes:
            return self.reads.pop(0) if self.reads else b""

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    serial = FakeSerial()
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial) as client:
        result = client.provision("Lab", "secret", timeout=1.0)

    assert result.endpoint == "http://192.0.2.10"
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert serial.closed

def test_improv_portal_url_builds_direct_endpoint_from_device_ip():
    assert direct_endpoint_from_device_url(
        "https://espectre.dev/tools/device-settings/?target=192.0.2.10"
    ) == "http://192.0.2.10:62587/espectre/v1"
    assert direct_endpoint_from_device_url(
        "https://espectre.dev/tools/device-settings/?target=192.0.2.10%3A62587"
    ) == "http://192.0.2.10:62587/espectre/v1"
    assert direct_endpoint_from_device_url("http://192.0.2.10/custom") == "http://192.0.2.10/espectre/v1"
    with pytest.raises(ValueError, match="invalid device URL"):
        direct_endpoint_from_device_url("ws://192.0.2.10/custom")

def test_direct_client_reads_a_resource_with_get():
    requests: list[object] = []

    def open_request(request: object, **kwargs: object) -> _FakeHttpResponse:
        requests.append((request, kwargs))
        return _FakeHttpResponse({"uptime": 7})

    with DirectClient("http://192.0.2.10/espectre/v1", urlopen_factory=open_request) as client:
        result = client.request("get", "diagnostics")
        assert result == {"uptime": 7}
    request, kwargs = requests[0]
    assert request.method == "GET"
    assert request.full_url == "http://192.0.2.10/espectre/v1/diagnostics"
    assert request.data is None
    assert request.headers["Origin"] == "https://test.espectre.dev"
    assert request.headers["Cache-control"] == "no-store"
    assert kwargs["timeout"] == 8.0

def test_direct_client_collects_canonical_sse_events():
    payload = protocol.build_motion_payload(
        "0123456789abcdef", "micro", 1000, "idle", 0.1, 0.25, "lightweight", 1
    )
    response = _FakeEventResponse(
        [b": connected\n", b"\n", b"event: motion\n", f"data: {json.dumps(payload)}\n".encode(), b"\n"]
    )

    with DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    ) as client:
        client.start_events()
        deadline = time.monotonic() + 1.0
        while not client.events and time.monotonic() < deadline:
            time.sleep(0.01)
        client.stop_events()

    assert len(client.events) == 1
    assert client.events[0].name == "motion"
    assert client.events[0].data == payload


def test_direct_client_types_unexpected_sse_eof_as_transport_loss():
    response = _GatedEventResponse([])
    client = DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    )

    assert client.events_active is False
    client.start_events()
    assert client.events_active is True
    response.readable.set()
    assert response.closed.wait(1.0)
    assert client.events_active is False
    with pytest.raises(DirectEventStreamTransportError):
        client.stop_events()
    assert client.events_active is False


def test_direct_client_types_incomplete_sse_chunk_as_transport_loss():
    response = _IncompleteEventResponse([])
    client = DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    )

    client.start_events()
    response.readable.set()
    assert response.closed.wait(1.0)
    with pytest.raises(DirectEventStreamTransportError):
        client.stop_events()


def test_direct_client_keeps_invalid_sse_json_as_protocol_error():
    response = _GatedEventResponse(
        [b"event: motion\n", b"data: {invalid\n", b"\n"]
    )
    client = DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    )

    client.start_events()
    response.readable.set()
    assert response.closed.wait(1.0)
    with pytest.raises(DirectProtocolError) as raised:
        client.stop_events()
    assert not isinstance(raised.value, DirectEventStreamTransportError)

def test_direct_client_ignores_http_client_close_race():
    response = _ClosingEventResponse([])

    with DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    ) as client:
        client.start_events()
        client.stop_events()

    assert response.event_socket.timeout is None
    assert response.event_socket.shutdown_mode == socket.SHUT_RDWR

def test_direct_client_sends_resource_mutation_body():
    requests: list[object] = []

    def open_request(request: object, **_kwargs: object) -> _FakeHttpResponse:
        requests.append(request)
        return _FakeHttpResponse(
            {"accepted": True, "code": "ok", "message": "sensing updated"}
        )

    with DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=open_request,
    ) as client:
        assert client.request("patch", "sensing", {"enabled": True}) == {
            "accepted": True,
            "code": "ok",
            "message": "sensing updated",
        }

    assert len(requests) == 1
    assert requests[0].method == "PATCH"
    assert requests[0].full_url == "http://192.0.2.10/espectre/v1/sensing"
    assert json.loads(requests[0].data) == {"enabled": True}

@pytest.mark.parametrize("method", ["get", "patch", "post", "put", "delete"])
@pytest.mark.parametrize("failure", [OSError, TimeoutError, http.client.RemoteDisconnected])
def test_persistent_direct_retries_reads_but_never_replays_mutations(monkeypatch, method, failure):
    calls = []
    connections = []

    class Connection:
        sock = None

        def __init__(self, *_args, **_kwargs):
            self.closed = False
            connections.append(self)

        def request(self, verb, path, **_kwargs):
            calls.append((verb, path))

        def getresponse(self):
            if len(calls) == 1:
                raise failure("response lost")
            return _FakeHttpResponse({"online": True})

        def close(self):
            self.closed = True

    monkeypatch.setattr(device_transport.http.client, "HTTPConnection", Connection)
    client = DirectClient("http://192.0.2.10" + device_transport.DIRECT_PATH, persistent_requests=True)
    try:
        if method == "get":
            assert client.request(method, "health") == {"online": True}
            assert len(calls) == 2
        else:
            with pytest.raises(DirectProtocolError):
                client.request(method, "sensing", {"enabled": True})
            assert len(calls) == 1
        assert connections[0].closed is True
    finally:
        client.close()


def test_direct_client_can_pace_requests(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    def open_request(request: object, **_kwargs: object) -> _FakeHttpResponse:
        return _FakeHttpResponse({"status": "ok"})

    monkeypatch.setattr(device_transport.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(device_transport.time, "sleep", fake_sleep)
    with DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=open_request,
        minimum_request_interval_seconds=0.075,
    ) as client:
        client.request("get", "health")
        client.request("get", "diagnostics")

    assert sleeps == [0.075]

def test_direct_client_accepts_response_larger_than_request_limit():
    padding = "x" * (DIRECT_MAX_REQUEST_FRAME_SIZE + 1)
    response = _FakeHttpResponse({"padding": padding})
    response_size = len(json.dumps(response.payload).encode())
    assert DIRECT_MAX_REQUEST_FRAME_SIZE < response_size <= DIRECT_MAX_RESPONSE_FRAME_SIZE

    with DirectClient(
        "http://192.0.2.10/espectre/v1",
        urlopen_factory=lambda *_args, **_kwargs: response,
    ) as client:
        assert client.request("get", "diagnostics") == {"padding": padding}

def test_direct_client_rejects_malformed_mutation_result():
    response = _FakeHttpResponse({"accepted": True})

    with DirectClient("http://192.0.2.10/espectre/v1", urlopen_factory=lambda *_args, **_kwargs: response) as client:
        with pytest.raises(DirectProtocolError, match="code and message"):
            client.request("patch", "sensing", {"enabled": True})
