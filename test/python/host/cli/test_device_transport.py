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


class ImprovConsole:
    """Queue device replies only when the host writes their corresponding RPC."""

    def __init__(self, reply):
        self.reply = reply
        self.input = bytearray()
        self.commands = []
        self.closed = False

    def write(self, data):
        command = ImprovCommand(data[9])
        self.commands.append(command)
        self.input.extend(self.reply(command))
        return len(data)

    def read(self, size):
        chunk = bytes(self.input[:size])
        del self.input[:size]
        return chunk

    def close(self):
        self.closed = True


def _state(value):
    return encode_improv_frame(ImprovPacketType.CURRENT_STATE, bytes([value]))


def test_improv_client_handles_multiple_frames_in_one_serial_read():
    replies = {
        ImprovCommand.GET_CURRENT_STATE: b"boot log\n" + _state(2),
        ImprovCommand.GET_DEVICE_INFO: _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"]),
        ImprovCommand.WIFI_SETTINGS: _state(3) + _state(4) + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"]),
    }
    serial = ImprovConsole(replies.__getitem__)
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: serial) as client:
        result = client.provision("Lab", "secret", timeout=1)

    assert result.endpoint == "http://192.0.2.10"
    assert result.states == ("authorized", "provisioning", "provisioned")
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert len(serial.commands) == 3
    assert serial.closed


@pytest.mark.parametrize("outcome", ["accepted", "lost_completion", "rejected"])
def test_improv_reprovision_with_truncated_usb_replies(monkeypatch, outcome):
    monkeypatch.setattr(device_transport, "IMPROV_OPTIONAL_RESPONSE_TIMEOUT", 0.02)

    def state(value):
        return encode_improv_frame(ImprovPacketType.CURRENT_STATE, bytes([value]))

    class UsbConsole:
        response = b""

        def write(self, packet):
            command = ImprovCommand(packet[9])
            full = _improv_rpc_response(command, ["x" * 73])
            truncated = full[:64] + full[-2:]
            if command == ImprovCommand.GET_CURRENT_STATE:
                self.response = state(4) + truncated
            elif command == ImprovCommand.GET_DEVICE_INFO:
                self.response = truncated
            elif outcome == "lost_completion":
                self.response = state(3)
            elif outcome == "accepted":
                self.response = state(3) + state(4) + truncated
            else:
                self.response = state(4) + encode_improv_frame(ImprovPacketType.ERROR_STATE, bytes([3]))
            return len(packet)

        def read(self, size):
            chunk, self.response = self.response[:size], self.response[size:]
            return chunk

        def close(self):
            pass

    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: UsbConsole()) as client:
        if outcome != "rejected":
            result = client.provision("Lab", "test", timeout=3)
            assert result.states == ("provisioned", "provisioning", "provisioned")
            assert result.endpoint is None
            assert result.device_info == ()
        else:
            with pytest.raises(ImprovProtocolError, match="error 3"):
                client.provision("Lab", "wrong", timeout=1)


def test_improv_client_retries_initial_state_query_after_flash(monkeypatch):
    now = [0.0]
    replies = iter([
        b"", _state(2),
        _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "s3", "Native"]),
        _state(4) + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"]),
    ])
    class BootingConsole(ImprovConsole):
        def read(self, size):
            now[0] += 0.1
            return super().read(size)

    serial = BootingConsole(lambda _: next(replies))
    monkeypatch.setattr(device_transport.time, "monotonic", lambda: now[0])
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: serial) as client:
        result = client.provision("Lab", "secret", timeout=10)

    assert result.endpoint == "http://192.0.2.10"
    assert serial.commands == [ImprovCommand.GET_CURRENT_STATE] * 2 + [ImprovCommand.GET_DEVICE_INFO, ImprovCommand.WIFI_SETTINGS]


def test_improv_client_ignores_current_state_url_before_device_info():
    replies = {
        ImprovCommand.GET_CURRENT_STATE: _state(4) + _improv_rpc_response(ImprovCommand.GET_CURRENT_STATE, ["http://192.0.2.99"]),
        ImprovCommand.GET_DEVICE_INFO: _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"]),
        ImprovCommand.WIFI_SETTINGS: _state(3) + _state(4) + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"]),
    }
    serial = ImprovConsole(replies.__getitem__)
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: serial) as client:
        result = client.provision("Lab", "secret", timeout=1)
    assert result.endpoint == "http://192.0.2.10"
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert serial.closed


@pytest.mark.parametrize("initial_state", [2, 4])
def test_improv_requires_evidence_from_new_settings(monkeypatch, initial_state):
    now = [0.0]
    replies = {
        ImprovCommand.GET_CURRENT_STATE: _state(initial_state),
        ImprovCommand.GET_DEVICE_INFO: _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, []) + _state(4),
        ImprovCommand.WIFI_SETTINGS: _state(4),
    }

    class Console(ImprovConsole):
        def read(self, size):
            now[0] += 0.1
            return super().read(size)

    serial = Console(replies.__getitem__)
    monkeypatch.setattr(device_transport.time, "monotonic", lambda: now[0])
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: serial) as client:
        with pytest.raises(TimeoutError):
            client.provision("New", "wrong", timeout=5)
    assert serial.commands.count(ImprovCommand.GET_CURRENT_STATE) == 1


@pytest.mark.parametrize("tail", ["disconnect", "malformed", "bad_checksum", "invalid_state", "rejected"])
def test_improv_optional_address_after_confirmation(monkeypatch, tail):
    monkeypatch.setattr(device_transport, "IMPROV_OPTIONAL_RESPONSE_TIMEOUT", 0.02)
    damaged_url = encode_improv_frame(ImprovPacketType.RPC_RESPONSE, bytes([ImprovCommand.WIFI_SETTINGS, 7]))
    bad_checksum = bytearray(_state(4))
    bad_checksum[-2] ^= 0xFF
    rejected = encode_improv_frame(ImprovPacketType.ERROR_STATE, bytes([3]))
    ending = {
        "disconnect": b"",
        "malformed": damaged_url,
        "bad_checksum": bytes(bad_checksum),
        "invalid_state": _state(0xFF),
        "rejected": bytes(bad_checksum) + damaged_url + rejected,
    }[tail]
    replies = {
        ImprovCommand.GET_CURRENT_STATE: _state(2),
        ImprovCommand.GET_DEVICE_INFO: _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, []),
        ImprovCommand.WIFI_SETTINGS: _state(3) + _state(4) + ending,
    }

    class Console(ImprovConsole):
        def read(self, size):
            if not self.input and tail == "disconnect":
                raise OSError("USB disconnected")
            return super().read(size)

    serial = Console(replies.__getitem__)
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_: serial) as client:
        if tail == "rejected":
            with pytest.raises(ImprovProtocolError, match="error 3"):
                client.provision("Lab", "test", timeout=1)
        else:
            result = client.provision("Lab", "test", timeout=1)
            assert result.endpoint is None
            assert result.states == ("authorized", "provisioning", "provisioned")


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


@pytest.mark.parametrize("persistent", [False, True])
def test_direct_transport_failure_identifies_method_and_resource(monkeypatch, persistent):
    def fail(*_args, **_kwargs):
        raise TimeoutError("timed out")

    client = DirectClient(
        "http://192.0.2.10" + device_transport.DIRECT_PATH,
        persistent_requests=persistent,
        urlopen_factory=None if persistent else fail,
    )
    if persistent:
        monkeypatch.setattr(client, "_persistent_request", fail)
    with pytest.raises(DirectProtocolError, match="Direct HTTP PATCH sensing request failed: timed out") as error:
        client.request("patch", "sensing", {"detector": "high_accuracy"})
    assert isinstance(error.value.__cause__, TimeoutError)
    assert "192.0.2.10" not in str(error.value)


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
