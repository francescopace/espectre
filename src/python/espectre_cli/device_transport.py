# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Bounded Improv Serial and Direct WebSocket clients shared by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import ipaddress
import json
import time
from typing import Callable, Protocol, Sequence
from urllib.parse import parse_qs, urlsplit, urlunsplit


IMPROV_HEADER = b"IMPROV"
IMPROV_VERSION = 1
IMPROV_MAX_FRAME_SIZE = 265
DIRECT_VERSION = 1
DIRECT_SUBPROTOCOL = "espectre.v1"
DIRECT_PATH = "/espectre/v1/ws"
DIRECT_MAX_REQUEST_FRAME_SIZE = 4096
DIRECT_MAX_RESPONSE_FRAME_SIZE = 8192
DIRECT_MAX_FRAME_SIZE = DIRECT_MAX_REQUEST_FRAME_SIZE
DEFAULT_DIRECT_ORIGIN = "https://test.espectre.dev"


class ImprovPacketType(IntEnum):
    CURRENT_STATE = 0x01
    ERROR_STATE = 0x02
    RPC = 0x03
    RPC_RESPONSE = 0x04


class ImprovCommand(IntEnum):
    WIFI_SETTINGS = 0x01
    GET_CURRENT_STATE = 0x02
    GET_DEVICE_INFO = 0x03


class ImprovState(IntEnum):
    STOPPED = 0x00
    AWAITING_AUTHORIZATION = 0x01
    AUTHORIZED = 0x02
    PROVISIONING = 0x03
    PROVISIONED = 0x04


class ImprovProtocolError(RuntimeError):
    """Raised when the device emits an invalid or rejected Improv frame."""


class DirectProtocolError(RuntimeError):
    """Raised when the Direct peer violates the versioned envelope contract."""


class DirectRequestError(RuntimeError):
    """Raised when firmware returns a correlated Direct error response."""

    def __init__(self, code: str, message: str):
        super().__init__(f"Direct request failed ({code}): {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ImprovFrame:
    packet_type: ImprovPacketType
    data: bytes


@dataclass(frozen=True)
class ImprovProvisioningResult:
    endpoint: str
    states: tuple[str, ...]
    device_info: tuple[str, ...]


@dataclass(frozen=True)
class DirectEvent:
    name: str
    data: dict[str, object]
    host_elapsed_seconds: float


class SerialTransport(Protocol):
    def read(self, size: int = 1) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...


class WebSocketTransport(Protocol):
    subprotocol: str | None

    def send(self, message: str) -> None: ...
    def recv(self, timeout: float | None = None) -> object: ...
    def close(self) -> None: ...


class ImprovFrameParser:
    """Incrementally recover Improv frames from an ESP-IDF console byte stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[ImprovFrame]:
        self._buffer.extend(chunk)
        frames: list[ImprovFrame] = []
        while True:
            header_at = self._buffer.find(IMPROV_HEADER)
            if header_at < 0:
                keep = min(len(self._buffer), len(IMPROV_HEADER) - 1)
                if keep:
                    self._buffer[:] = self._buffer[-keep:]
                else:
                    self._buffer.clear()
                break
            if header_at:
                del self._buffer[:header_at]
            if len(self._buffer) < 9:
                break
            data_length = self._buffer[8]
            frame_length = 10 + data_length
            if frame_length > IMPROV_MAX_FRAME_SIZE:
                raise ImprovProtocolError("Improv frame exceeds the size limit")
            if len(self._buffer) < frame_length:
                break
            raw = bytes(self._buffer[:frame_length])
            del self._buffer[:frame_length]
            if raw[6] != IMPROV_VERSION:
                raise ImprovProtocolError(f"unsupported Improv version {raw[6]}")
            if sum(raw[:-1]) & 0xFF != raw[-1]:
                raise ImprovProtocolError("invalid Improv frame checksum")
            try:
                packet_type = ImprovPacketType(raw[7])
            except ValueError as exc:
                raise ImprovProtocolError(f"unknown Improv packet type {raw[7]}") from exc
            frames.append(ImprovFrame(packet_type, raw[9:-1]))
        return frames


def encode_improv_frame(packet_type: ImprovPacketType, data: bytes = b"") -> bytes:
    if len(data) > 0xFF:
        raise ValueError("Improv frame data exceeds 255 bytes")
    frame = bytearray((*IMPROV_HEADER, IMPROV_VERSION, int(packet_type), len(data)))
    frame.extend(data)
    frame.append(sum(frame) & 0xFF)
    return bytes(frame)


def encode_improv_rpc(command: ImprovCommand, values: Sequence[str] = ()) -> bytes:
    encoded_values = [value.encode("utf-8") for value in values]
    if any(len(value) > 0xFF for value in encoded_values):
        raise ValueError("Improv RPC string exceeds 255 bytes")
    payload = bytearray((int(command), sum(len(value) + 1 for value in encoded_values)))
    for value in encoded_values:
        payload.append(len(value))
        payload.extend(value)
    return encode_improv_frame(ImprovPacketType.RPC, bytes(payload))


def parse_improv_rpc_response(data: bytes) -> tuple[ImprovCommand, tuple[str, ...]]:
    if len(data) < 2 or data[1] != len(data) - 2:
        raise ImprovProtocolError("invalid Improv RPC response length")
    try:
        command = ImprovCommand(data[0])
    except ValueError as exc:
        raise ImprovProtocolError(f"unknown Improv RPC response command {data[0]}") from exc
    values: list[str] = []
    offset = 2
    while offset < len(data):
        length = data[offset]
        offset += 1
        end = offset + length
        if end > len(data):
            raise ImprovProtocolError("truncated Improv RPC response string")
        try:
            values.append(data[offset:end].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ImprovProtocolError("invalid UTF-8 in Improv RPC response") from exc
        offset = end
    return command, tuple(values)


class ImprovSerialClient:
    """Minimal Improv v1 client used only until Wi-Fi provisioning completes."""

    def __init__(
        self,
        port: str,
        *,
        serial_factory: Callable[..., SerialTransport] | None = None,
        baudrate: int = 115200,
    ) -> None:
        if serial_factory is None:
            from serial import Serial

            serial_factory = Serial
        self._serial = serial_factory(port=port, baudrate=baudrate, timeout=0.1, write_timeout=2.0)
        self._parser = ImprovFrameParser()
        self._pending_frames: list[ImprovFrame] = []

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> "ImprovSerialClient":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _send_rpc(self, command: ImprovCommand, values: Sequence[str] = ()) -> None:
        frame = encode_improv_rpc(command, values)
        if self._serial.write(frame) != len(frame):
            raise ImprovProtocolError("short write while sending Improv request")

    def _read_until(
        self,
        predicate: Callable[[ImprovFrame], object | None],
        deadline: float,
    ) -> object:
        while time.monotonic() < deadline:
            if not self._pending_frames:
                chunk = self._serial.read(IMPROV_MAX_FRAME_SIZE)
                if not chunk:
                    continue
                self._pending_frames.extend(self._parser.feed(chunk))
            while self._pending_frames:
                frame = self._pending_frames.pop(0)
                if frame.packet_type == ImprovPacketType.ERROR_STATE:
                    code = frame.data[0] if len(frame.data) == 1 else -1
                    if code:
                        raise ImprovProtocolError(f"Improv device reported error {code}")
                result = predicate(frame)
                if result is not None:
                    return result
        raise TimeoutError("timed out waiting for Improv response")

    def _rpc(self, command: ImprovCommand, deadline: float) -> tuple[str, ...]:
        self._send_rpc(command)

        def matching_response(frame: ImprovFrame) -> tuple[str, ...] | None:
            if frame.packet_type != ImprovPacketType.RPC_RESPONSE:
                return None
            response_command, values = parse_improv_rpc_response(frame.data)
            if response_command == ImprovCommand.GET_CURRENT_STATE and command != response_command:
                # A provisioned device emits its URL immediately after the
                # current-state frame. That response belongs to the preceding
                # state query and may still be queued when the next RPC starts.
                return None
            if response_command != command:
                raise ImprovProtocolError(
                    f"unexpected Improv RPC response {response_command.name}; expected {command.name}"
                )
            return values

        return self._read_until(matching_response, deadline)  # type: ignore[return-value]

    def provision(self, ssid: str, password: str, *, timeout: float) -> ImprovProvisioningResult:
        deadline = time.monotonic() + timeout
        states: list[str] = []

        def current_state(frame: ImprovFrame) -> str | None:
            if frame.packet_type != ImprovPacketType.CURRENT_STATE:
                return None
            if len(frame.data) != 1:
                raise ImprovProtocolError("invalid Improv current-state payload")
            try:
                state = ImprovState(frame.data[0])
            except ValueError as exc:
                raise ImprovProtocolError(f"unknown Improv state {frame.data[0]}") from exc
            states.append(state.name.lower())
            return state.name.lower()

        self._send_rpc(ImprovCommand.GET_CURRENT_STATE)
        self._read_until(current_state, deadline)
        device_info = self._rpc(ImprovCommand.GET_DEVICE_INFO, deadline)
        self._send_rpc(ImprovCommand.WIFI_SETTINGS, (ssid, password))

        endpoint: str | None = None

        def provisioning_result(frame: ImprovFrame) -> str | None:
            nonlocal endpoint
            state = current_state(frame)
            if state is not None:
                return endpoint if state == ImprovState.PROVISIONED.name.lower() and endpoint else None
            if frame.packet_type != ImprovPacketType.RPC_RESPONSE:
                return None
            response_command, values = parse_improv_rpc_response(frame.data)
            if response_command not in {
                ImprovCommand.WIFI_SETTINGS,
                ImprovCommand.GET_CURRENT_STATE,
            }:
                raise ImprovProtocolError(
                    f"unexpected Improv RPC response {response_command.name}; expected WIFI_SETTINGS or GET_CURRENT_STATE"
                )
            if len(values) != 1 or not values[0]:
                raise ImprovProtocolError("Improv Wi-Fi response did not contain one device URL")
            endpoint = values[0]
            return endpoint if ImprovState.PROVISIONED.name.lower() in states else None

        while endpoint is None and time.monotonic() < deadline:
            retry_deadline = min(deadline, time.monotonic() + 2.0)
            try:
                endpoint = self._read_until(provisioning_result, retry_deadline)  # type: ignore[assignment]
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
                # Recover when the one-shot WIFI_SETTINGS completion was lost
                # while the interface associated or the USB console reset.
                self._send_rpc(ImprovCommand.GET_CURRENT_STATE)
        if endpoint is None:
            raise TimeoutError("timed out waiting for Improv provisioning result")
        return ImprovProvisioningResult(endpoint, tuple(states), device_info)


def direct_endpoint_from_device_url(device_url: str) -> str:
    parsed = urlsplit(device_url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise ValueError("Improv returned an invalid device URL")
    target_values = parse_qs(parsed.query, strict_parsing=False).get("target", [])
    if target_values:
        if len(target_values) != 1:
            raise ValueError("Improv returned multiple device targets")
        try:
            address = ipaddress.ip_address(target_values[0])
        except ValueError as error:
            raise ValueError("Improv returned an invalid device target") from error
        host = f"[{address}]" if address.version == 6 else str(address)
        return urlunsplit(("ws", host, DIRECT_PATH, "", ""))
    if parsed.scheme in {"ws", "wss"}:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or DIRECT_PATH, "", ""))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, DIRECT_PATH, "", ""))


class DirectClient:
    """Sequential, correlated Direct v1 client with strict envelope validation."""

    def __init__(
        self,
        endpoint: str,
        *,
        origin: str = DEFAULT_DIRECT_ORIGIN,
        timeout: float = 8.0,
        connect_factory: Callable[..., WebSocketTransport] | None = None,
    ) -> None:
        if connect_factory is None:
            from websockets.sync.client import connect

            connect_factory = connect
        self.endpoint = endpoint
        self.origin = origin
        self.timeout = timeout
        self._started = time.monotonic()
        self._next_id = 1
        self.events: list[DirectEvent] = []
        try:
            self._socket = connect_factory(
                endpoint,
                origin=origin,
                subprotocols=[DIRECT_SUBPROTOCOL],
                open_timeout=timeout,
                ping_interval=None,
                close_timeout=timeout,
                max_size=DIRECT_MAX_RESPONSE_FRAME_SIZE,
                proxy=None,
            )
        except Exception as exc:
            raise DirectProtocolError(f"Direct WebSocket handshake failed: {exc}") from exc
        if self._socket.subprotocol != DIRECT_SUBPROTOCOL:
            self._socket.close()
            raise DirectProtocolError(
                f"Direct subprotocol mismatch: {self._socket.subprotocol!r}"
            )

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "DirectClient":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _decode(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, str):
            raise DirectProtocolError("Direct endpoint sent a non-text frame")
        if len(raw.encode("utf-8")) > DIRECT_MAX_RESPONSE_FRAME_SIZE:
            raise DirectProtocolError("Direct frame exceeds the size limit")
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DirectProtocolError("Direct endpoint sent invalid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("v") != DIRECT_VERSION:
            raise DirectProtocolError("Direct endpoint sent an incompatible envelope")
        return envelope

    def request(
        self,
        method: str,
        params: dict[str, object] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, object]:
        request_id = f"benchmark-{self._next_id}"
        self._next_id += 1
        message = json.dumps(
            {
                "v": DIRECT_VERSION,
                "type": "request",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
            separators=(",", ":"),
        )
        if len(message.encode("utf-8")) > DIRECT_MAX_REQUEST_FRAME_SIZE:
            raise ValueError("Direct request exceeds the size limit")
        self._socket.send(message)
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for Direct response to {method}")
            envelope = self._decode(self._socket.recv(timeout=remaining))
            envelope_type = envelope.get("type")
            if envelope_type == "event":
                name = envelope.get("event")
                data = envelope.get("data")
                if not isinstance(name, str) or not name or not isinstance(data, dict):
                    raise DirectProtocolError("invalid Direct event envelope")
                self.events.append(DirectEvent(name, data, time.monotonic() - self._started))
                continue
            if envelope_type != "response":
                raise DirectProtocolError("Direct endpoint sent an invalid envelope type")
            if envelope.get("id") != request_id:
                raise DirectProtocolError("Direct endpoint sent an unknown response identifier")
            ok = envelope.get("ok")
            if not isinstance(ok, bool):
                raise DirectProtocolError("Direct response is missing its boolean result status")
            if ok:
                result = envelope.get("result")
                if not isinstance(result, dict):
                    raise DirectProtocolError("Direct success response result must be an object")
                if "data" in result:
                    data = result["data"]
                    if not isinstance(data, dict):
                        raise DirectProtocolError("Direct query response data must be an object")
                    return data
                return result
            error = envelope.get("error")
            if not isinstance(error, dict):
                raise DirectProtocolError("Direct error response is missing its error object")
            code = error.get("code")
            error_message = error.get("message")
            if not isinstance(code, str) or not isinstance(error_message, str):
                raise DirectProtocolError("Direct error response fields must be strings")
            raise DirectRequestError(code, error_message)
