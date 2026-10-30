"""Audio device discovery for Windows (WASAPI) using PyAudioWPatch.

Only WASAPI devices are listed: the same physical device also appears under
MME / DirectSound, which would just add duplicates. A loopback device is the
"recording" twin of an output device (speakers / headphones), so capturing it
records whatever Windows plays there, e.g. the other side of a call.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyaudiowpatch as pyaudio


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    kind: str          # "mic" or "loopback"
    channels: int
    rate: int
    is_default: bool


def _wasapi(pa: pyaudio.PyAudio) -> dict:
    try:
        return pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as e:  # WASAPI missing would be very unusual on Windows 10/11
        raise RuntimeError("WASAPI host API not available on this system") from e


def _default_loopback_index(pa: pyaudio.PyAudio, wasapi: dict) -> int | None:
    """Loopback twin of the current default output device, if found."""
    out_idx = wasapi.get("defaultOutputDevice", -1)
    if out_idx < 0:
        return None
    speakers = pa.get_device_info_by_index(out_idx)
    if speakers.get("isLoopbackDevice"):
        return speakers["index"]
    for lb in pa.get_loopback_device_info_generator():
        if speakers["name"] in lb["name"]:
            return lb["index"]
    return None


def list_devices(pa: pyaudio.PyAudio) -> list[Device]:
    wasapi = _wasapi(pa)
    default_mic = wasapi.get("defaultInputDevice", -1)
    default_lb = _default_loopback_index(pa, wasapi)
    devices: list[Device] = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["hostApi"] != wasapi["index"] or info["maxInputChannels"] <= 0:
            continue
        is_lb = bool(info.get("isLoopbackDevice"))
        devices.append(Device(
            index=info["index"],
            name=info["name"],
            kind="loopback" if is_lb else "mic",
            channels=int(info["maxInputChannels"]),
            rate=int(info["defaultSampleRate"]),
            is_default=info["index"] == (default_lb if is_lb else default_mic),
        ))
    return devices


def print_devices(devices: list[Device]) -> None:
    for kind, title in (("mic", "MICROPHONES  (use with --mic)"),
                        ("loopback", "SYSTEM AUDIO / LOOPBACK  (use with --loopback)")):
        print(f"\n{title}")
        rows = [d for d in devices if d.kind == kind]
        if not rows:
            print("  (none found)")
        for d in rows:
            star = "*" if d.is_default else " "
            print(f" {star}[{d.index:>3}] {d.name}  ({d.channels} ch, {d.rate} Hz)")
    print("\n * = current Windows default. Pick the loopback that matches the output "
          "device your call app plays to (e.g. your headphones).")


def resolve(devices: list[Device], index: int, kind: str) -> Device:
    """Return the device with this index, checking it is the expected kind."""
    for d in devices:
        if d.index == index:
            if d.kind != kind:
                raise ValueError(
                    f"Device [{index}] '{d.name}' is a {d.kind} device, not a {kind} device. "
                    "Run with --list-devices to see the right indices.")
            return d
    raise ValueError(f"No WASAPI {kind} device with index {index}. Run with --list-devices.")
