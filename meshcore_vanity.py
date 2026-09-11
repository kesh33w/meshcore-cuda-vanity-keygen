#!/usr/bin/env python3
"""Local, MeshCore-compatible Ed25519 vanity key generator."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import secrets
import selectors
import signal
import subprocess
import stat
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from rare_rules import (
    CUDA_RULE_PROTOCOL_VERSION,
    DEFAULT_RULESET as DEFAULT_RARE_RULESET,
    RareMatch,
    RareRule,
    RareRuleset,
    RuleConfigError,
    load_ruleset,
    select_rules,
)

APP_DIR = Path(__file__).resolve().parent
VERSION_PATH = APP_DIR / "VERSION"
APP_VERSION = VERSION_PATH.read_text(encoding="utf-8").strip() if VERSION_PATH.is_file() else "development"
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("MESHCORE_VANITY_RESULTS_DIR", str(APP_DIR / "results"))
).expanduser().resolve()
DEFAULT_WATCH_PATH = DEFAULT_RESULTS_DIR / "rare-keys.jsonl"
_xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
DEFAULT_GUI_SETTINGS_PATH = Path(
    os.environ.get(
        "MESHCORE_VANITY_SETTINGS_PATH",
        str(
            (Path(_xdg_config_home).expanduser() if _xdg_config_home
             else Path.home() / ".config")
            / "meshcore-vanity-keygen" / "settings.json"
        ),
    )
).expanduser()
REFERENCE_CUDA_RATE = 870_000_000.0
ICON_PATH = APP_DIR / "assets" / "meshcore-vanity-keygen.png"
RARE_LOG_SCHEMA = 4
GUI_SETTINGS_SCHEMA = 1
GUI_SETTINGS_MAX_BYTES = 64 * 1024
RARE_BROWSER_LIMIT = 10_000
RARE_BROWSER_PAGE_SIZE = 500
RARE_READ_BLOCK_SIZE = 64 * 1024
RARE_MAX_RECORD_BYTES = 1024 * 1024
TEMPERATURE_POLL_SECONDS = 3.0
TEMPERATURE_STALE_SECONDS = 12.0
TEMPERATURE_QUERY_TIMEOUT = 1.5
TEMPERATURE_MAX_OUTPUT_BYTES = 64 * 1024
WATCH_WORDS = tuple(
    rule.value for rule in DEFAULT_RARE_RULESET.rules
    if rule.kind == "literal-prefix" and rule.enabled
)
PI_DIGITS = next(
    rule.value for rule in DEFAULT_RARE_RULESET.rules
    if rule.kind == "sequence-prefix" and rule.id == "pi"
)


class Sodium:
    def __init__(self) -> None:
        library = ctypes.util.find_library("sodium") or "libsodium.so.23"
        self.lib = ctypes.CDLL(library)
        self.lib.sodium_init.restype = ctypes.c_int
        self.lib.crypto_scalarmult_ed25519_base_noclamp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.crypto_scalarmult_ed25519_base_noclamp.restype = ctypes.c_int
        self.lib.crypto_sign_verify_detached.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p
        ]
        self.lib.crypto_sign_verify_detached.restype = ctypes.c_int
        self.lib.crypto_sign_ed25519_pk_to_curve25519.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.crypto_sign_ed25519_pk_to_curve25519.restype = ctypes.c_int
        self.lib.crypto_scalarmult_curve25519.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.crypto_scalarmult_curve25519.restype = ctypes.c_int
        if self.lib.sodium_init() < 0:
            raise RuntimeError("libsodium initialization failed")

    def derive_public(self, scalar: bytes) -> bytes:
        public = (ctypes.c_ubyte * 32)()
        secret = (ctypes.c_ubyte * 32).from_buffer_copy(scalar)
        if self.lib.crypto_scalarmult_ed25519_base_noclamp(public, secret) != 0:
            raise RuntimeError("libsodium rejected the Ed25519 scalar")
        return bytes(public)

    def verify(self, signature: bytes, message: bytes, public: bytes) -> bool:
        if len(signature) != 64 or len(public) != 32:
            return False
        signature_buffer = (ctypes.c_ubyte * len(signature)).from_buffer_copy(signature)
        message_buffer = (ctypes.c_ubyte * len(message)).from_buffer_copy(message)
        public_buffer = (ctypes.c_ubyte * len(public)).from_buffer_copy(public)
        return self.lib.crypto_sign_verify_detached(
            signature_buffer, message_buffer, len(message), public_buffer
        ) == 0

    def key_exchange(self, scalar: bytes, peer_public: bytes) -> bytes:
        """Match MeshCore's Ed25519-to-Montgomery shared-secret operation."""
        if len(scalar) != 32 or len(peer_public) != 32:
            raise ValueError("invalid MeshCore key-exchange key length")
        montgomery_public = (ctypes.c_ubyte * 32)()
        peer_buffer = (ctypes.c_ubyte * 32).from_buffer_copy(peer_public)
        if self.lib.crypto_sign_ed25519_pk_to_curve25519(
                montgomery_public, peer_buffer) != 0:
            raise RuntimeError("libsodium rejected the Ed25519 public key")
        shared_secret = (ctypes.c_ubyte * 32)()
        scalar_buffer = (ctypes.c_ubyte * 32).from_buffer_copy(scalar)
        if self.lib.crypto_scalarmult_curve25519(
                shared_secret, scalar_buffer, montgomery_public) != 0:
            raise RuntimeError("libsodium rejected the MeshCore key exchange")
        return bytes(shared_secret)


SODIUM = Sodium()


class SearchCancelled(RuntimeError):
    pass


class SearchWorkerError(RuntimeError):
    """A CPU search worker stopped unexpectedly."""

    pass


def cuda_device_count() -> int:
    """Return CUDA devices visible to the driver; never loads a CUDA toolkit."""
    try:
        cuda = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        count = ctypes.c_int()
        # cuInit(0), then cuDeviceGetCount().
        if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return count.value
    except OSError:
        return 0


def meshcore_keypair() -> tuple[bytes, bytes]:
    """Match MeshCore's ed25519_create_keypair(seed) representation exactly."""
    digest = hashlib.sha512(secrets.token_bytes(32)).digest()
    scalar = bytearray(digest[:32])
    scalar[0] &= 248
    scalar[31] &= 63
    scalar[31] |= 64
    private = bytes(scalar) + digest[32:]
    return SODIUM.derive_public(bytes(scalar)), private


ED25519_ORDER = 2**252 + 27742317777372353535851937790883648493


def sign_expanded(private: bytes, public: bytes, message: bytes) -> bytes:
    """Sign with MeshCore's expanded scalar || nonce-prefix private format."""
    if len(private) != 64 or len(public) != 32:
        raise ValueError("invalid expanded Ed25519 key length")
    scalar = int.from_bytes(private[:32], "little")
    nonce = int.from_bytes(hashlib.sha512(private[32:] + message).digest(), "little") % ED25519_ORDER
    encoded_nonce = SODIUM.derive_public(nonce.to_bytes(32, "little"))
    challenge = int.from_bytes(
        hashlib.sha512(encoded_nonce + public + message).digest(), "little"
    ) % ED25519_ORDER
    response = (nonce + challenge * scalar) % ED25519_ORDER
    return encoded_nonce + response.to_bytes(32, "little")


def meshcore_shared_secret(private: bytes, peer_public: bytes) -> bytes:
    """Calculate the shared secret produced by MeshCore's ed25519_key_exchange."""
    if len(private) != 64:
        raise ValueError("invalid expanded Ed25519 private-key length")
    return SODIUM.key_exchange(private[:32], peer_public)


def verify_expanded_key(private: bytes, public: bytes) -> bool:
    if (len(private) != 64 or len(public) != 32 or private[0] & 7
            or private[31] & 128 or not private[31] & 64
            or public[0] in (0, 255)):
        return False
    message = b"MeshCore vanity key compatibility test"
    try:
        if (SODIUM.derive_public(private[:32]) != public
                or not SODIUM.verify(sign_expanded(private, public, message), message, public)):
            return False
        from_private = meshcore_shared_secret(private, ECDH_TEST_PEER_PUBLIC)
        from_peer = meshcore_shared_secret(ECDH_TEST_PEER_PRIVATE, public)
        return (bool(any(from_private))
                and secrets.compare_digest(from_private, from_peer))
    except (ValueError, RuntimeError):
        return False


@dataclass(frozen=True)
class Result:
    public_key: str
    private_key: str
    attempts: int
    elapsed_seconds: float
    match: str
    backend: str = "cpu"
    engine: Optional[str] = None


def valid_pattern(value: str, label: str) -> str:
    value = value.strip().lower()
    if value and any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must contain hexadecimal characters only")
    if len(value) > 60:
        raise ValueError(f"{label} is too long (maximum is 60 hex characters)")
    return value


def validate_constraints(prefix: str, suffix: str, contains: str) -> None:
    """Reject constraints for which no MeshCore-valid 64-nibble key can exist."""
    assigned: list[Optional[str]] = [None] * 64

    def place(value: str, offset: int) -> bool:
        for index, character in enumerate(value):
            position = offset + index
            if assigned[position] is not None and assigned[position] != character:
                return False
        for index, character in enumerate(value):
            assigned[offset + index] = character
        return True

    if len(prefix) >= 2 and prefix[:2] in ("00", "ff"):
        raise ValueError("prefix begins with a byte MeshCore rejects (00 or ff)")
    if not place(prefix, 0) or not place(suffix, 64 - len(suffix)):
        raise ValueError("prefix and suffix conflict where they overlap")
    if contains:
        possible = False
        for offset in range(65 - len(contains)):
            if all(assigned[offset + index] in (None, character)
                   for index, character in enumerate(contains)):
                possible = True
                break
        if not possible:
            raise ValueError("substring conflicts with the prefix and suffix")


def matches(key: str, prefix: str, suffix: str, contains: str) -> bool:
    return ((not prefix or key.startswith(prefix)) and
            (not suffix or key.endswith(suffix)) and
            (not contains or contains in key))


def search(prefix: str, suffix: str, contains: str, workers: int,
           update: Optional[Callable[[int, float], None]] = None,
           cancel: Optional[threading.Event] = None) -> Result:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    stop = threading.Event()
    found: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
    failures: queue.Queue[BaseException] = queue.Queue(maxsize=1)
    counts = [0] * workers
    start = time.monotonic()

    def worker(index: int) -> None:
        local_count = 0
        try:
            while not stop.is_set() and not (cancel and cancel.is_set()):
                public, private = meshcore_keypair()
                local_count += 1
                public_hex = public.hex()
                # MeshCore rejects these identities when importing the private key.
                if public[0] not in (0, 255) and matches(public_hex, prefix, suffix, contains):
                    try:
                        found.put_nowait((public_hex, private.hex()))
                        stop.set()
                    except queue.Full:
                        pass
                    break
                if local_count % 256 == 0:
                    counts[index] = local_count
        except BaseException as error:
            # A failed daemon worker must wake the coordinator. Otherwise every
            # remaining worker can search forever for an impossible/test target.
            try:
                failures.put_nowait(error)
            except queue.Full:
                pass
            stop.set()
        finally:
            counts[index] = local_count

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for thread in threads:
        thread.start()
    last = 0.0
    try:
        while not stop.wait(0.1):
            if cancel and cancel.is_set():
                stop.set()
                break
            elapsed = time.monotonic() - start
            if update and elapsed - last >= 0.25:
                update(sum(counts), elapsed)
                last = elapsed
    except BaseException:
        stop.set()
        raise
    finally:
        for thread in threads:
            thread.join()
    elapsed = time.monotonic() - start
    attempts = sum(counts)
    if not failures.empty():
        failure = failures.get_nowait()
        raise SearchWorkerError(f"CPU search worker failed: {failure}") from failure
    if found.empty():
        raise SearchCancelled("Search cancelled")
    public_hex, private_hex = found.get_nowait()
    needle = ", ".join(part for part in (prefix and f"prefix {prefix}", suffix and f"suffix {suffix}", contains and f"contains {contains}") if part) or "any valid key"
    return Result(public_hex, private_hex, attempts, elapsed, needle, "cpu")


def _read_small_text(path: Path, limit: int = 128) -> Optional[str]:
    """Read one bounded sysfs value without letting telemetry raise."""
    try:
        with path.open("rb") as file:
            raw = file.read(limit + 1)
        if len(raw) > limit:
            return None
        return raw.decode("ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_millidegrees(path: Path) -> Optional[float]:
    raw = _read_small_text(path)
    if raw is None or re.fullmatch(r"-?[0-9]{1,9}", raw) is None:
        return None
    celsius = int(raw) / 1000.0
    return celsius if -40.0 <= celsius <= 200.0 else None


def read_cpu_package_temperature(
        hwmon_root: Path = Path("/sys/class/hwmon"),
        thermal_root: Path = Path("/sys/class/thermal"),
) -> Optional[float]:
    """Return the hottest CPU package/control sensor, never another device."""
    preferred: list[float] = []
    core_fallbacks: list[float] = []
    cpu_hwmon_names = {
        "coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal",
        "x86_pkg_temp",
    }
    try:
        chips = sorted(hwmon_root.glob("hwmon*"))
    except OSError:
        chips = []
    for chip in chips:
        name = (_read_small_text(chip / "name") or "").lower()
        if name not in cpu_hwmon_names:
            continue
        try:
            inputs = sorted(chip.glob("temp*_input"))
        except OSError:
            continue
        for input_path in inputs:
            match = re.fullmatch(r"temp([0-9]+)_input", input_path.name)
            if match is None:
                continue
            celsius = _read_millidegrees(input_path)
            if celsius is None:
                continue
            label = (
                _read_small_text(chip / f"temp{match.group(1)}_label") or ""
            ).lower()
            is_package = (
                name in {"cpu_thermal", "soc_thermal", "x86_pkg_temp"}
                or (name == "coretemp" and label.startswith("package id"))
                or (name in {"k10temp", "zenpower"}
                    and label in {"tctl", "tdie", "package"})
            )
            (preferred if is_package else core_fallbacks).append(celsius)
    if preferred:
        return max(preferred)

    thermal: list[float] = []
    cpu_zone_types = {
        "x86_pkg_temp", "cpu_thermal", "cpu-thermal", "cpu_therm",
        "soc_thermal", "soc-thermal",
    }
    try:
        zones = sorted(thermal_root.glob("thermal_zone*"))
    except OSError:
        zones = []
    for zone in zones:
        zone_type = (_read_small_text(zone / "type") or "").lower()
        if zone_type not in cpu_zone_types:
            continue
        celsius = _read_millidegrees(zone / "temp")
        if celsius is not None:
            thermal.append(celsius)
    if thermal:
        return max(thermal)
    return max(core_fallbacks) if core_fallbacks else None


def normalize_pci_bus_id(value: object) -> Optional[str]:
    """Normalize CUDA/NVIDIA PCI IDs to dddd:bb:dd.f for safe matching."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
        r"([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        return None
    domain = int(match.group(1), 16)
    if domain > 0xffff:
        return None
    return (
        f"{domain:04x}:{int(match.group(2), 16):02x}:"
        f"{int(match.group(3), 16):02x}.{int(match.group(4))}"
    )


@dataclass(frozen=True)
class NvidiaTemperatureReading:
    index: int
    pci_bus_id: str
    celsius: Optional[float]


@dataclass(frozen=True)
class NvidiaTemperatureReport:
    readings: tuple[NvidiaTemperatureReading, ...]


@dataclass(frozen=True)
class TemperatureSnapshot:
    observed_at: float
    cpu_celsius: Optional[float]
    nvidia: Optional[NvidiaTemperatureReport]


@dataclass
class TemperatureCache:
    cpu_celsius: Optional[float] = None
    cpu_observed_at: float = float("-inf")
    gpu_celsius_by_bus: dict[str, float] = field(default_factory=dict)
    gpu_observed_at_by_bus: dict[str, float] = field(default_factory=dict)
    nvidia_device_count: Optional[int] = None
    nvidia_bus_ids: frozenset[str] = frozenset()
    nvidia_observed_at: float = float("-inf")


def parse_nvidia_temperatures(output: str) -> NvidiaTemperatureReport:
    """Parse bounded CSV keyed by explicit NVIDIA index and PCI bus ID."""
    if not isinstance(output, str) or len(output.encode("utf-8")) > TEMPERATURE_MAX_OUTPUT_BYTES:
        raise ValueError("NVIDIA temperature output is too large")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) > 256:
        raise ValueError("NVIDIA temperature output has too many rows")
    readings: list[NvidiaTemperatureReading] = []
    seen_indices: set[int] = set()
    seen_buses: set[str] = set()
    for row in rows:
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 3 or re.fullmatch(r"[0-9]{1,4}", fields[0]) is None:
            raise ValueError("malformed NVIDIA temperature row")
        index = int(fields[0])
        pci_bus_id = normalize_pci_bus_id(fields[1])
        if pci_bus_id is None or index in seen_indices or pci_bus_id in seen_buses:
            raise ValueError("ambiguous NVIDIA temperature row")
        seen_indices.add(index)
        seen_buses.add(pci_bus_id)
        raw_temperature = fields[2].lower()
        celsius: Optional[float] = None
        if raw_temperature not in {"n/a", "[n/a]", "not supported"}:
            if re.fullmatch(r"-?[0-9]{1,3}(?:\.[0-9])?", fields[2]) is None:
                raise ValueError("malformed NVIDIA temperature value")
            parsed = float(fields[2])
            if not -40.0 <= parsed <= 200.0:
                raise ValueError("implausible NVIDIA temperature value")
            celsius = parsed
        readings.append(NvidiaTemperatureReading(index, pci_bus_id, celsius))
    readings.sort(key=lambda reading: reading.index)
    return NvidiaTemperatureReport(tuple(readings))


def _bounded_command_output(
        command: list[str], timeout: float, max_output_bytes: int,
) -> Optional[str]:
    """Run a fixed argv command with a hard time and stdout memory bound."""
    if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes < 1:
        raise ValueError("command telemetry bounds must be positive")
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        return None
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout
    completed = False

    def kill_process_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

    try:
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
        raw = bytearray()
        eof = False
        while not eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            events = selector.select(remaining)
            if not events:
                return None
            for _key, _mask in events:
                while True:
                    read_size = min(
                        64 * 1024, max_output_bytes + 1 - len(raw),
                    )
                    try:
                        chunk = os.read(descriptor, read_size)
                    except BlockingIOError:
                        break
                    if not chunk:
                        eof = True
                        break
                    raw.extend(chunk)
                    if len(raw) > max_output_bytes:
                        return None
                if eof:
                    break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return None
        if return_code != 0:
            return None
        output = raw.decode("ascii")
        completed = True
        return output
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    finally:
        selector.close()
        if not completed:
            kill_process_group()
        if process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                kill_process_group()
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        process.stdout.close()


def query_nvidia_temperatures(
        timeout: float = TEMPERATURE_QUERY_TIMEOUT,
) -> Optional[NvidiaTemperatureReport]:
    try:
        output = _bounded_command_output(
            ["nvidia-smi",
             "--query-gpu=index,pci.bus_id,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout, TEMPERATURE_MAX_OUTPUT_BYTES,
        )
        if output is None:
            return None
        return parse_nvidia_temperatures(output)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        return None


def sample_hardware_temperatures() -> TemperatureSnapshot:
    return TemperatureSnapshot(
        time.monotonic(), read_cpu_package_temperature(),
        query_nvidia_temperatures(),
    )


def update_temperature_cache(
        cache: TemperatureCache, snapshot: TemperatureSnapshot,
        stale_after: float = TEMPERATURE_STALE_SECONDS,
) -> None:
    """Merge a best-effort sample and expire readings after a grace period."""
    if not math.isfinite(stale_after) or stale_after <= 0:
        raise ValueError("temperature stale interval must be positive")
    now = snapshot.observed_at
    if snapshot.cpu_celsius is not None:
        cache.cpu_celsius = snapshot.cpu_celsius
        cache.cpu_observed_at = now
    if now - cache.cpu_observed_at >= stale_after:
        cache.cpu_celsius = None

    if snapshot.nvidia is not None:
        cache.nvidia_device_count = len(snapshot.nvidia.readings)
        cache.nvidia_bus_ids = frozenset(
            reading.pci_bus_id for reading in snapshot.nvidia.readings
        )
        cache.nvidia_observed_at = now
        for reading in snapshot.nvidia.readings:
            if reading.celsius is not None:
                cache.gpu_celsius_by_bus[reading.pci_bus_id] = reading.celsius
                cache.gpu_observed_at_by_bus[reading.pci_bus_id] = now
    for pci_bus_id, observed_at in tuple(cache.gpu_observed_at_by_bus.items()):
        if now - observed_at >= stale_after:
            cache.gpu_observed_at_by_bus.pop(pci_bus_id, None)
            cache.gpu_celsius_by_bus.pop(pci_bus_id, None)
    if now - cache.nvidia_observed_at >= stale_after:
        cache.nvidia_device_count = None
        cache.nvidia_bus_ids = frozenset()


def format_temperature_status(
        cache: TemperatureCache, selected_device: Optional[int],
        selected_pci_bus: Optional[str], cuda_device_count: int,
) -> str:
    def formatted(value: Optional[float]) -> str:
        return "—" if value is None else f"{value:.0f} °C"

    normalized_bus = normalize_pci_bus_id(selected_pci_bus)
    gpu_celsius = (
        cache.gpu_celsius_by_bus.get(normalized_bus)
        if normalized_bus is not None else None
    )
    # A positional fallback is unambiguous only when both APIs see one GPU.
    if (gpu_celsius is None and cache.nvidia_device_count == 1
            and len(cache.nvidia_bus_ids) == 1 and cuda_device_count <= 1):
        sole_bus = next(iter(cache.nvidia_bus_ids))
        gpu_celsius = cache.gpu_celsius_by_bus.get(sole_bus)
    gpu_name = f"GPU {selected_device}" if selected_device is not None else "GPU"
    return (
        f"CPU package {formatted(cache.cpu_celsius)}  •  "
        f"{gpu_name} {formatted(gpu_celsius)}"
    )


def start_temperature_monitor(
        callback: Callable[[TemperatureSnapshot], None],
        stop_event: threading.Event, interval: float = TEMPERATURE_POLL_SECONDS,
        sampler: Optional[Callable[[], TemperatureSnapshot]] = None,
) -> threading.Thread:
    """Start optional telemetry without ever doing sensor I/O on Tk's thread."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("temperature polling interval must be positive")
    sample = sample_hardware_temperatures if sampler is None else sampler

    def worker() -> None:
        while not stop_event.is_set():
            try:
                snapshot = sample()
            except Exception:
                snapshot = TemperatureSnapshot(time.monotonic(), None, None)
            if stop_event.is_set():
                break
            try:
                callback(snapshot)
            except Exception:
                if stop_event.is_set():
                    break
            if stop_event.wait(interval):
                break

    thread = threading.Thread(
        target=worker, daemon=True, name="meshcore-temperature-monitor",
    )
    thread.start()
    return thread


def drain_callback_queue(
        events: queue.SimpleQueue[
            tuple[Callable[..., None], tuple[object, ...]]
        ],
        limit: int = 256,
        on_error: Optional[Callable[[Exception], None]] = None,
) -> int:
    """Run a bounded UI-event batch without one callback stopping the queue."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("callback drain limit must be a positive integer")
    processed = 0
    for _index in range(limit):
        try:
            callback, arguments = events.get_nowait()
        except queue.Empty:
            break
        try:
            callback(*arguments)
        except Exception as error:
            if on_error is not None:
                try:
                    on_error(error)
                except Exception:
                    pass
        processed += 1
    return processed


def cuda_executable() -> Path:
    return Path(__file__).resolve().with_name("meshcore_cuda_vanity")


CUDA_PROBE_PROTOCOL = "meshcore-cuda-probe-v2"
SUPPORTED_CUDA_PROBE_PROTOCOLS = frozenset((
    "meshcore-cuda-probe-v1", CUDA_PROBE_PROTOCOL,
))
INTERACTIVE_CUDA_ATTEMPTS = {"optimized": 2048, "baseline": 32}
_CUDA_PROBE_CACHE: dict[tuple[object, ...], dict[str, object]] = {}
_CUDA_PROBE_LOCK = threading.Lock()


def cuda_probe(device: int = 0, engine: str = "optimized", *,
               interactive: bool = False, refresh: bool = False,
               timeout: float = 5.0) -> dict[str, object]:
    """Exercise a real kernel and return a validated, key-free readiness report."""
    if device < 0:
        raise ValueError("CUDA device must be zero or greater")
    if engine == "incremental":
        engine = "optimized"
    if engine not in ("optimized", "baseline"):
        raise ValueError("CUDA engine must be optimized or baseline")
    if not isinstance(interactive, bool):
        raise ValueError("CUDA interactive profile must be boolean")
    executable = cuda_executable()
    base: dict[str, object] = {
        "schema": 1,
        "protocol": CUDA_PROBE_PROTOCOL,
        "ready": False,
        "device": device,
        "engine": engine,
    }
    try:
        metadata = executable.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("CUDA engine is not a regular file")
    except OSError as error:
        return {**base, "error": str(error)}
    cache_key = (
        str(executable.resolve()), metadata.st_dev, metadata.st_ino,
        metadata.st_size, metadata.st_mtime_ns, device, engine, interactive,
    )
    with _CUDA_PROBE_LOCK:
        cached = _CUDA_PROBE_CACHE.get(cache_key)
    if cached is not None and not refresh:
        return dict(cached)

    try:
        command = [
            str(executable), "--probe", "--device", str(device),
            "--engine", engine,
        ]
        if interactive:
            command.append("--interactive")
        completed = subprocess.run(
            command,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("probe output is not a JSON object")
        if (isinstance(payload.get("schema"), bool)
                or not isinstance(payload.get("schema"), int)
                or payload.get("schema") != 1
                or payload.get("protocol") not in SUPPORTED_CUDA_PROBE_PROTOCOLS
                or (interactive and payload.get("protocol") != CUDA_PROBE_PROTOCOL)
                or isinstance(payload.get("device"), bool)
                or not isinstance(payload.get("device"), int)
                or payload.get("device") != device or payload.get("engine") != engine
                or not isinstance(payload.get("ready"), bool)):
            raise ValueError("probe response does not match the requested protocol")
        allowed = {
            "schema", "protocol", "ready", "device", "device_name",
            "pci_bus_id", "compute_capability", "engine", "build_fingerprint", "build_arches",
            "threads", "blocks_per_sm", "attempts_per_thread", "max_registers",
            "rare_rule_protocol", "default_ruleset_fingerprint", "error",
        }
        if set(payload) - allowed:
            raise ValueError("probe response contains unsupported fields")
        if (isinstance(payload.get("rare_rule_protocol"), bool)
                or payload.get("rare_rule_protocol") != CUDA_RULE_PROTOCOL_VERSION
                or not isinstance(payload.get("default_ruleset_fingerprint"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(payload["default_ruleset_fingerprint"]),
                ) is None):
            raise ValueError("probe rare-rule compatibility details are malformed")
        if payload["ready"]:
            if completed.returncode != 0:
                raise ValueError("probe reported readiness with a failing exit status")
            if (not isinstance(payload.get("device_name"), str)
                    or not payload["device_name"]
                    or ("pci_bus_id" in payload
                        and normalize_pci_bus_id(payload.get("pci_bus_id")) is None)
                    or not isinstance(payload.get("compute_capability"), str)
                    or re.fullmatch(r"\d+\.\d+", str(payload["compute_capability"])) is None
                    or not isinstance(payload.get("build_fingerprint"), str)
                    or re.fullmatch(r"[0-9a-f]{16}", str(payload["build_fingerprint"])) is None
                    or not isinstance(payload.get("build_arches"), str)
                    or any(isinstance(payload.get(name), bool)
                           or not isinstance(payload.get(name), int)
                           or int(payload[name]) <= 0
                           for name in ("threads", "blocks_per_sm", "attempts_per_thread"))
                    or isinstance(payload.get("max_registers"), bool)
                    or not isinstance(payload.get("max_registers"), int)
                    or int(payload["max_registers"]) < 0):
                raise ValueError("probe readiness details are malformed")
            if (interactive
                    and (int(payload["blocks_per_sm"]) > 4
                         or int(payload["attempts_per_thread"])
                         != INTERACTIVE_CUDA_ATTEMPTS[engine])):
                raise ValueError("probe interactive scheduling details are malformed")
        else:
            if completed.returncode == 0 or not isinstance(payload.get("error"), str):
                raise ValueError("probe failure details are malformed")
        result = {key: payload[key] for key in allowed if key in payload}
        if result.get("ready") is True and "pci_bus_id" in result:
            normalized_bus = normalize_pci_bus_id(result.get("pci_bus_id"))
            assert normalized_bus is not None
            result["pci_bus_id"] = normalized_bus
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as error:
        result = {**base, "error": f"CUDA readiness probe failed: {error}"}

    with _CUDA_PROBE_LOCK:
        # A binary replacement changes the cache key; discard stale entries so
        # long-running GUIs cannot accumulate obsolete probe responses.
        for previous_key in tuple(_CUDA_PROBE_CACHE):
            if (previous_key[0] == cache_key[0]
                    and previous_key[-3:] == cache_key[-3:]
                    and previous_key != cache_key):
                del _CUDA_PROBE_CACHE[previous_key]
        # Initialization and driver failures can be transient. Retaining them
        # would disable CUDA for the lifetime of a long-running GUI.
        if result.get("ready") is True:
            _CUDA_PROBE_CACHE[cache_key] = dict(result)
        else:
            _CUDA_PROBE_CACHE.pop(cache_key, None)
    return result


def cuda_available(device: int = 0, engine: str = "optimized", *,
                   interactive: bool = False, refresh: bool = False) -> bool:
    if cuda_device_count() <= device:
        return False
    return bool(cuda_probe(
        device, engine, interactive=interactive, refresh=refresh,
    ).get("ready"))


def cuda_device_names() -> list[str]:
    """Return user-facing GPU names without requiring the CUDA toolkit."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def signal_process_termination(process: subprocess.Popen[str]) -> bool:
    """Request child termination without leaking a concurrent-exit race."""
    if process.poll() is not None:
        return False
    try:
        process.terminate()
    except ProcessLookupError:
        return False
    return True


def terminate_process(process: subprocess.Popen[str], timeout: float = 2.0) -> None:
    """Terminate and reap a child, escalating to kill after a bounded wait."""
    if not signal_process_termination(process):
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def diagnostics(ruleset: RareRuleset = DEFAULT_RARE_RULESET, *,
                interactive: bool = False) -> dict[str, object]:
    devices = cuda_device_count()
    names = cuda_device_names()
    firmware_vector = firmware_compatibility_test()
    probes = [
        cuda_probe(index, interactive=interactive) for index in range(devices)
    ]
    return {
        "version": APP_VERSION,
        "firmware_vector": firmware_vector,
        "cuda_devices": devices,
        "cuda_names": names,
        "cuda_engine_built": cuda_executable().is_file(),
        "cuda_ready": any(bool(probe.get("ready")) for probe in probes),
        "cuda_probes": probes,
        "results_directory": str(DEFAULT_RESULTS_DIR),
        "rare_ruleset_id": ruleset.ruleset_id,
        "rare_ruleset_fingerprint": ruleset.fingerprint,
    }


def gui_diagnostics(ruleset: RareRuleset = DEFAULT_RARE_RULESET) -> dict[str, object]:
    """Collect every GUI CUDA-engine readiness result without touching Tk."""
    details = diagnostics(ruleset, interactive=True)
    device_count = int(details.get("cuda_devices", 0))
    engine_probes: dict[tuple[int, str], dict[str, object]] = {}
    existing = details.get("cuda_probes", [])
    if isinstance(existing, list):
        for probe in existing:
            if not isinstance(probe, dict):
                continue
            probe_device = probe.get("device")
            probe_engine = probe.get("engine")
            if (isinstance(probe_device, int) and not isinstance(probe_device, bool)
                    and probe_engine in ("optimized", "baseline")):
                engine_probes[(probe_device, str(probe_engine))] = dict(probe)
    for device_index in range(device_count):
        for engine in ("optimized", "baseline"):
            if (device_index, engine) not in engine_probes:
                engine_probes[(device_index, engine)] = cuda_probe(
                    device_index, engine, interactive=True,
                )
    details["cuda_engine_probes"] = engine_probes
    details["cuda_ready"] = any(
        bool(probe.get("ready")) for probe in engine_probes.values()
    )
    return details


def start_gui_diagnostics(
        callback: Callable[[Optional[dict[str, object]], Optional[str]], None],
        ruleset: RareRuleset = DEFAULT_RARE_RULESET,
) -> threading.Thread:
    """Launch GUI discovery and report from a worker through a safe callback."""
    def worker() -> None:
        try:
            callback(gui_diagnostics(ruleset), None)
        except Exception as error:
            callback(None, str(error))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def estimate_attempts(prefix: str, suffix: str, contains: str) -> float:
    """Estimate mean candidates, exactly for prefix/suffix and approximately for contains."""
    assigned: dict[int, str] = {}
    for index, character in enumerate(prefix):
        assigned[index] = character
    for index, character in enumerate(suffix):
        assigned[64 - len(suffix) + index] = character
    fixed_probability = 16.0 ** -len(assigned)
    if not contains:
        probability = fixed_probability
    else:
        conditional = 0.0
        for offset in range(65 - len(contains)):
            if all(assigned.get(offset + index, character) == character
                   for index, character in enumerate(contains)):
                extra = sum(1 for index in range(len(contains)) if offset + index not in assigned)
                conditional += 16.0 ** -extra
        probability = min(1.0, fixed_probability * conditional)
    return 1.0 / max(probability, 16.0 ** -64)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f} seconds"
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    if seconds < 31557600:
        return f"{seconds / 86400:.1f} days"
    return f"{seconds / 31557600:.1f} years"


def search_cuda(prefix: str, suffix: str, contains: str,
                update: Optional[Callable[[int, float], None]] = None,
                watch_path: Optional[Path] = None,
                cancel: Optional[threading.Event] = None,
                watch_update: Optional[Callable[[int, str, str, Path], None]] = None,
                device: int = 0,
                engine: str = "optimized",
                collect_only: bool = False,
                interactive: bool = False,
                process_update: Optional[Callable[[Optional[subprocess.Popen[str]]], None]] = None,
                ruleset: RareRuleset = DEFAULT_RARE_RULESET) -> Result:
    executable = cuda_executable()
    if not executable.is_file():
        raise RuntimeError("CUDA engine is not built; run 'make'")
    if engine == "incremental":
        engine = "optimized"
    if engine not in ("optimized", "baseline"):
        raise ValueError("CUDA engine must be optimized or baseline")
    if not isinstance(interactive, bool):
        raise ValueError("CUDA interactive profile must be boolean")
    command = [str(executable), "--device", str(device), "--engine", engine]
    if interactive:
        command.append("--interactive")
    command.extend(ruleset.cuda_arguments())
    if collect_only:
        if any((prefix, suffix, contains)):
            raise ValueError("Collector mode cannot be combined with a vanity pattern")
        command.append("--collect-only")
    for option, value in (("--prefix", prefix), ("--suffix", suffix), ("--contains", contains)):
        if value:
            command.extend((option, value))
    if update:
        update(0, 0.0)
    if cancel and cancel.is_set():
        raise SearchCancelled("Search cancelled")
    watch_path = watch_path or DEFAULT_WATCH_PATH
    initialize_watch_file(watch_path)
    if cancel and cancel.is_set():
        raise SearchCancelled("Search cancelled")
    process: Optional[subprocess.Popen[str]] = None
    errors: list[str] = []
    stdout = ""
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=1)
        if process_update:
            process_update(process)
        # Process registration and the GUI close path form a handshake: if
        # close won the race before registration, this worker owns cleanup.
        if cancel and cancel.is_set():
            raise SearchCancelled("Search cancelled")
        assert process.stderr is not None
        for line in process.stderr:
            if cancel and cancel.is_set():
                raise SearchCancelled("Search cancelled")
            progress = re.fullmatch(r"PROGRESS (\d+) ([0-9.]+) ([0-9.]+)\s*", line)
            if progress:
                if update:
                    update(int(progress.group(1)), float(progress.group(2)))
            elif line.startswith("WATCH "):
                try:
                    _, rule, public_hex, private_hex = line.strip().split()
                    rule_number = int(rule)
                    record = append_interesting(
                        watch_path, rule_number, public_hex, private_hex, engine,
                        ruleset=ruleset,
                    )
                    if watch_update:
                        watch_update(rule_number, str(record["reason"]), public_hex, watch_path)
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(str(error)) from error
            else:
                errors.append(line.strip())
        assert process.stdout is not None
        stdout = process.stdout.read()
        return_code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            terminate_process(process)
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        if process_update:
            process_update(None)
    # The GUI terminates the CUDA child immediately for responsive cancellation.
    # If that happens between progress lines, the stderr loop ends before its
    # in-loop cancellation check can run. Treat the resulting SIGTERM as the
    # requested cancellation rather than reporting the last startup banner as
    # a CUDA failure.
    if cancel and cancel.is_set():
        raise SearchCancelled("Search cancelled")
    if return_code:
        detail = errors[-1] if errors else f"status {return_code}"
        raise RuntimeError(f"CUDA engine failed: {detail}")
    if collect_only:
        raise RuntimeError("CUDA collector stopped unexpectedly")
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        allowed_fields = {
            "public_key", "private_key", "engine", "attempts", "elapsed_seconds",
        }
        if not isinstance(payload, dict) or set(payload) != allowed_fields:
            raise ValueError("result object has unexpected fields")
        public_hex = payload.get("public_key")
        private_hex = payload.get("private_key")
        attempts = payload.get("attempts")
        elapsed_seconds = payload.get("elapsed_seconds")
        if (not isinstance(public_hex, str)
                or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None
                or not isinstance(private_hex, str)
                or re.fullmatch(r"[0-9a-f]{128}", private_hex) is None
                or payload.get("engine") != engine
                or isinstance(attempts, bool) or not isinstance(attempts, int)
                or attempts < 0
                or isinstance(elapsed_seconds, bool)
                or not isinstance(elapsed_seconds, (int, float))
                or not math.isfinite(float(elapsed_seconds))
                or float(elapsed_seconds) < 0):
            raise ValueError("result fields are malformed")
        public = bytes.fromhex(public_hex)
        private = bytes.fromhex(private_hex)
    except (IndexError, OverflowError, TypeError, ValueError) as error:
        raise RuntimeError("CUDA engine returned an invalid result") from error
    if (public[0] in (0, 255) or not verify_expanded_key(private, public)
            or not matches(public_hex, prefix, suffix, contains)):
        raise RuntimeError("CUDA result failed independent CPU verification")
    needle = ", ".join(part for part in (prefix and f"prefix {prefix}", suffix and f"suffix {suffix}", contains and f"contains {contains}") if part)
    return Result(public_hex, private_hex, attempts,
                  float(elapsed_seconds), needle, "cuda", engine)


def secure_open(path: Path, flags: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"refusing to write non-regular file: {path}")
    return descriptor


def atomic_write_json(record: object, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing to replace non-regular file: {path}")
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}")
    try:
        descriptor = secure_open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def active_rule_ids(ruleset: RareRuleset) -> tuple[str, ...]:
    return tuple(rule.id for rule in ruleset.active_rules)


def rare_rule_short_name(rule: RareRule) -> str:
    if rule.kind == "bookend":
        return "Bookends"
    if rule.kind == "mirror":
        return "Mirrors"
    if rule.kind == "repeat-prefix":
        return "Repeated prefix"
    if rule.kind == "literal-prefix" and rule.value == "1337133713":
        return "1337133713"
    if rule.kind == "sequence-prefix" and rule.id == "pi":
        return "Pi"
    return rule.id


def rare_rule_choice_label(rule: RareRule) -> str:
    length = rule.threshold_length
    if rule.kind == "bookend":
        detail = f"first {length}+ hex characters match the end"
    elif rule.kind == "mirror":
        detail = f"first {length}+ mirror the final characters"
    elif rule.kind == "repeat-prefix":
        detail = (
            f"{length}+ identical starting characters; excludes "
            f"{', '.join(rule.excluded_nibbles)}"
        )
    elif rule.kind == "literal-prefix":
        detail = f"public key begins with {rule.value}"
    else:
        preview = rule.value[:length]
        detail = f"public key begins with {preview} from the {rule.id} sequence"
    return (
        f"{rare_rule_short_name(rule)} — {detail} "
        f"({rule.rarity_bits:.1f} rarity bits)"
    )


def rare_rule_selection_summary(ruleset: RareRuleset) -> str:
    active = ruleset.active_rules
    total = len(ruleset.rules)
    if len(active) == total:
        return f"All {total} selected"
    if len(active) <= 3:
        names = ", ".join(rare_rule_short_name(rule) for rule in active)
        return f"{len(active)} of {total} selected: {names}"
    inactive = [rule for rule in ruleset.rules if not rule.enabled]
    if len(inactive) <= 2:
        names = ", ".join(rare_rule_short_name(rule) for rule in inactive)
        return f"{len(active)} of {total} selected; excluding {names}"
    return f"{len(active)} of {total} selected"


def load_gui_rule_selection(
        base_ruleset: RareRuleset,
        path: Optional[Path] = None,
) -> RareRuleset:
    """Load a bounded built-in GUI preference, falling back safely if stale."""
    settings_path = DEFAULT_GUI_SETTINGS_PATH if path is None else path
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            settings_path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > GUI_SETTINGS_MAX_BYTES):
            os.close(descriptor)
            descriptor = None
            return base_ruleset
        with os.fdopen(descriptor, "rb") as file:
            descriptor = None
            raw = file.read(GUI_SETTINGS_MAX_BYTES + 1)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        return base_ruleset
    if len(raw) > GUI_SETTINGS_MAX_BYTES:
        return base_ruleset

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("duplicate settings field")
            value[name] = item
        return value

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        if (not isinstance(document, dict)
                or set(document) != {
                    "schema_version", "base_ruleset_fingerprint", "enabled_rule_ids",
                }
                or isinstance(document.get("schema_version"), bool)
                or document.get("schema_version") != GUI_SETTINGS_SCHEMA
                or document.get("base_ruleset_fingerprint") != base_ruleset.fingerprint
                or not isinstance(document.get("enabled_rule_ids"), list)):
            return base_ruleset
        return select_rules(base_ruleset, document["enabled_rule_ids"])
    except (
        UnicodeDecodeError, json.JSONDecodeError, RecursionError,
        RuleConfigError, ValueError,
    ):
        return base_ruleset


def save_gui_rule_selection(
        base_ruleset: RareRuleset, selected_ruleset: RareRuleset,
        path: Optional[Path] = None,
) -> None:
    """Persist the non-secret built-in GUI rule choice without editing defaults."""
    settings_path = DEFAULT_GUI_SETTINGS_PATH if path is None else path
    atomic_write_json({
        "schema_version": GUI_SETTINGS_SCHEMA,
        "base_ruleset_fingerprint": base_ruleset.fingerprint,
        "enabled_rule_ids": list(active_rule_ids(selected_ruleset)),
    }, settings_path, overwrite=True)


def save_result(result: Result, path: Path, overwrite: bool = False) -> None:
    atomic_write_json(asdict(result), path, overwrite)


def available_result_path(directory: Path, public_key: str) -> Path:
    base = directory / f"meshcore-identity-{public_key[:12]}.json"
    if not base.exists() and not base.is_symlink():
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base.with_name(f"{base.stem}-{stamp}-{secrets.token_hex(2)}.json")


def default_result_path(public_key: str) -> Path:
    return available_result_path(DEFAULT_RESULTS_DIR, public_key)


def count_interesting(
        path: Path, cancel: Optional[threading.Event] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        block_size: int = RARE_READ_BLOCK_SIZE,
) -> int:
    """Count nonblank entries in a stable size snapshot without blocking appends."""
    if block_size < 1:
        raise ValueError("count block size must be positive")
    try:
        with path.open("rb") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            try:
                snapshot_size = os.fstat(file.fileno()).st_size
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            remaining = snapshot_size
            scanned = 0
            count = 0
            line_has_content = False
            while remaining and not (cancel and cancel.is_set()):
                chunk = file.read(min(block_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                scanned += len(chunk)
                parts = chunk.split(b"\n")
                if len(parts) == 1:
                    line_has_content = line_has_content or bool(parts[0].strip())
                else:
                    if line_has_content or parts[0].strip():
                        count += 1
                    count += sum(1 for part in parts[1:-1] if part.strip())
                    line_has_content = bool(parts[-1].strip())
                if progress:
                    progress(scanned, snapshot_size)
            if remaining == 0 and line_has_content:
                count += 1
            return count
    except FileNotFoundError:
        return 0


def apply_rare_count_snapshot(
        state: dict[str, object], generation: int, total: int,
) -> bool:
    """Apply a background count only if no search invalidated its snapshot."""
    current = state.get("rare_count_generation")
    if (isinstance(current, bool) or not isinstance(current, int)
            or current != generation):
        return False
    state["rare_count"] = total
    return True


def initialize_watch_file(path: Path) -> None:
    """Create the watch file up front so an empty file clearly means zero finds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.close(descriptor)


WATCH_REASONS = DEFAULT_RARE_RULESET.watch_reasons


def interesting_rule(public_hex: str,
                     ruleset: RareRuleset = DEFAULT_RARE_RULESET) -> int:
    """Classify a rare public key independently of the CUDA implementation."""
    return ruleset.classify(public_hex)


def interesting_matches(public_hex: str,
                        ruleset: RareRuleset = DEFAULT_RARE_RULESET) -> list[RareMatch]:
    """Describe every recognized rare property, strongest property first."""
    return list(ruleset.analyze(public_hex))


def infer_legacy_rarity(record: dict[str, object]) -> tuple[int, float]:
    """Infer basic sortable metadata for records written by older releases."""
    reason = str(record.get("reason", ""))
    numbered = re.fullmatch(r"(?:bookend|mirror|repeat-prefix)-(\d{1,2})", reason)
    if numbered:
        length = int(numbered.group(1))
        if not 1 <= length <= 64:
            return 0, 0.0
        alternatives = 14 if reason.startswith("repeat-prefix-") else 1
        return length, round(length * 4.0 - math.log2(alternatives), 3)
    for marker in ("prefix-pi-", "prefix-", "suffix-"):
        if reason.startswith(marker):
            length = len(reason.removeprefix(marker))
            if 1 <= length <= 64:
                return length, float(length * 4)
    return 0, 0.0


def _history_length(value: object, fallback: int = 0) -> int:
    if (isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value <= 64):
        return value
    return fallback


def _history_rarity(value: object, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError):
            pass
        else:
            if math.isfinite(converted) and 0 <= converted <= 256:
                return converted
    return fallback


def _history_attempts(value: object, fallback: str = "0") -> str:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,80}", value):
        return value
    return fallback


def _history_rule_ids(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        return []
    result: list[str] = []
    for rule_id in value:
        if (not isinstance(rule_id, str)
                or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", rule_id) is None
                or rule_id in result):
            return []
        result.append(rule_id)
    return result


def _sanitize_stored_match(value: object) -> Optional[dict[str, object]]:
    if not isinstance(value, dict):
        return None
    reason = value.get("reason")
    kind = value.get("kind")
    length = value.get("length")
    rarity = value.get("rarity_bits")
    attempts = value.get("mean_attempts")
    if (not isinstance(reason, str) or not reason
            or not isinstance(kind, str) or not kind
            or _history_length(length, -1) < 1
            or _history_rarity(rarity, -1.0) < 0
            or not isinstance(attempts, str)
            or re.fullmatch(r"[0-9]{1,80}", attempts) is None):
        return None
    sanitized = dict(value)
    sanitized["length"] = _history_length(length)
    sanitized["rarity_bits"] = _history_rarity(rarity)
    sanitized["mean_attempts"] = _history_attempts(attempts)
    return sanitized


def _stored_rare_analysis(record: dict[str, object]) -> Optional[dict[str, object]]:
    """Return the primary stored match when schema-v2+ analysis is well formed."""
    schema_version = record.get("schema_version")
    matches_found = record.get("matches")
    if (isinstance(schema_version, bool) or not isinstance(schema_version, int)
            or schema_version < 2 or not isinstance(matches_found, list)
            or not matches_found):
        return None
    sanitized = [
        match for item in matches_found
        if (match := _sanitize_stored_match(item)) is not None
    ]
    if not sanitized:
        return None
    record["matches"] = sanitized
    return sanitized[0]


def normalize_interesting_record(
        record: object, ruleset: RareRuleset = DEFAULT_RARE_RULESET,
) -> Optional[dict[str, object]]:
    """Validate a log record and add display metadata without exposing secrets."""
    if not isinstance(record, dict):
        return None
    public_hex = record.get("public_key")
    private_hex = record.get("private_key")
    if (not isinstance(public_hex, str) or not isinstance(private_hex, str)
            or not re.fullmatch(r"[0-9a-f]{64}", public_hex)
            or not re.fullmatch(r"[0-9a-f]{128}", private_hex)):
        return None
    normalized = dict(record)
    normalized["active_rule_ids"] = _history_rule_ids(
        normalized.get("active_rule_ids"),
    )
    stored_primary = _stored_rare_analysis(normalized)
    if stored_primary is not None:
        # A recorded ruleset may later be disabled or changed. Preserve the
        # original analysis as historical truth instead of reclassifying it.
        normalized["reason"] = stored_primary["reason"]
        normalized["match_length"] = stored_primary["length"]
        normalized["rarity_bits"] = stored_primary["rarity_bits"]
        normalized["mean_attempts"] = stored_primary["mean_attempts"]
        return normalized

    analyzed = interesting_matches(public_hex, ruleset)
    if analyzed:
        primary = analyzed[0]
        normalized["reason"] = primary.reason
        normalized["match_length"] = primary.length
        normalized["rarity_bits"] = primary.rarity_bits
        normalized["mean_attempts"] = primary.mean_attempts
        normalized["matches"] = [asdict(match) for match in analyzed]
    else:
        length, rarity_bits = infer_legacy_rarity(normalized)
        normalized["match_length"] = _history_length(
            normalized.get("match_length"), length,
        )
        normalized["rarity_bits"] = _history_rarity(
            normalized.get("rarity_bits"), rarity_bits,
        )
        alternatives = 14 if str(normalized.get("reason", "")).startswith(
            "repeat-prefix-"
        ) else 1
        inferred_attempts = (
            str((16 ** length + alternatives - 1) // alternatives)
            if length else "0"
        )
        normalized["mean_attempts"] = _history_attempts(
            normalized.get("mean_attempts"), inferred_attempts,
        )
        normalized["matches"] = []
    return normalized


def load_interesting_records(
        path: Path, limit: int = RARE_BROWSER_LIMIT,
        progress: Optional[Callable[[int, int], None]] = None,
        block_size: int = RARE_READ_BLOCK_SIZE,
        max_record_bytes: int = RARE_MAX_RECORD_BYTES,
        ruleset: RareRuleset = DEFAULT_RARE_RULESET,
        cancel: Optional[threading.Event] = None,
) -> tuple[list[dict[str, object]], int]:
    """Load newest valid JSONL records by reading backward from the file tail.

    Work and retained memory are bounded by ``limit`` and the bytes needed to
    reach that many valid records. A shared advisory lock prevents observing a
    half-written line from this program's append path. A malformed crash tail
    is skipped and does not hide the preceding valid records.
    """
    if limit < 1:
        return [], 0
    if block_size < 1 or max_record_bytes < 1:
        raise ValueError("tail-reader bounds must be positive")
    if cancel and cancel.is_set():
        return [], 0

    newest_first: list[dict[str, object]] = []
    skipped = 0

    def consume(raw: bytes) -> None:
        nonlocal skipped
        if not raw.strip():
            return
        if len(raw) > max_record_bytes:
            skipped += 1
            return
        try:
            normalized = normalize_interesting_record(json.loads(raw), ruleset)
        except (ValueError, UnicodeDecodeError, RecursionError):
            normalized = None
        if normalized is None:
            skipped += 1
        elif len(newest_first) < limit:
            newest_first.append(normalized)

    try:
        with path.open("rb") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH)
            try:
                total = file.seek(0, os.SEEK_END)
                position = total
                pending = b""
                dropping_oversized = False
                scanned = 0
                last_progress = 0.0
                while (position > 0 and len(newest_first) < limit
                       and not (cancel and cancel.is_set())):
                    size = min(block_size, position)
                    position -= size
                    file.seek(position)
                    chunk = file.read(size)
                    scanned += len(chunk)

                    if dropping_oversized:
                        parts = chunk.split(b"\n")
                        if len(parts) == 1:
                            if progress and time.monotonic() - last_progress >= 0.1:
                                progress(scanned, total)
                                last_progress = time.monotonic()
                            continue
                        # The rightmost fragment belongs to the oversized line
                        # already counted as malformed. Earlier complete lines
                        # in this chunk remain usable.
                        pending = parts[0]
                        candidates = parts[1:-1]
                        dropping_oversized = False
                    else:
                        parts = (chunk + pending).split(b"\n")
                        pending = parts[0]
                        candidates = parts[1:]

                    for raw in reversed(candidates):
                        if cancel and cancel.is_set():
                            break
                        consume(raw)

                    if len(pending) > max_record_bytes:
                        pending = b""
                        dropping_oversized = True
                        skipped += 1

                    now = time.monotonic()
                    if progress and now - last_progress >= 0.1:
                        progress(scanned, total)
                        last_progress = now

                if (position == 0 and not dropping_oversized
                        and not (cancel and cancel.is_set())):
                    consume(pending)
                if progress:
                    progress(scanned, total)
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        if progress:
            progress(0, 0)
    newest_first.reverse()
    return newest_first, skipped


def select_interesting_page(
        records: list[dict[str, object]], query: str, sort_column: str,
        reverse: bool, page: int, page_size: int = RARE_BROWSER_PAGE_SIZE,
) -> tuple[list[dict[str, object]], int, int, int]:
    """Filter, sort, and select one bounded page for the rare-key browser."""
    if page_size < 1:
        raise ValueError("page size must be positive")
    normalized_query = query.strip().lower()
    filtered = [
        record for record in records
        if not normalized_query or normalized_query in " ".join((
            str(record.get("found_at", "")), str(record.get("reason", "")),
            str(record.get("public_key", "")),
            " ".join(_history_rule_ids(record.get("active_rule_ids"))),
        )).lower()
    ]
    if sort_column == "length":
        sort_key = lambda record: _history_length(record.get("match_length", 0))
    elif sort_column == "rarity":
        sort_key = lambda record: _history_rarity(record.get("rarity_bits", 0))
    elif sort_column == "public":
        sort_key = lambda record: str(record.get("public_key", ""))
    elif sort_column == "reason":
        sort_key = lambda record: str(record.get("reason", ""))
    else:
        sort_key = lambda record: str(record.get("found_at", ""))
    filtered.sort(key=sort_key, reverse=reverse)
    pages = max(1, math.ceil(len(filtered) / page_size))
    selected_page = min(max(0, page), pages - 1)
    start = selected_page * page_size
    return filtered[start:start + page_size], len(filtered), selected_page, pages


def append_interesting(path: Path, rule: int, public_hex: str, private_hex: str,
                       engine: str = "optimized", *,
                       ruleset: RareRuleset = DEFAULT_RARE_RULESET) -> dict[str, object]:
    private = bytes.fromhex(private_hex)
    public = bytes.fromhex(public_hex)
    if (rule < 0 or rule >= len(ruleset.watch_reasons)
            or interesting_rule(public_hex, ruleset) != rule
            or len(private) != 64 or len(public) != 32
            or public[0] in (0, 255)
            or not verify_expanded_key(private, public)):
        raise RuntimeError("An incidental CUDA result failed CPU verification")
    matches_found = interesting_matches(public_hex, ruleset)
    if not matches_found:
        raise RuntimeError("An incidental CUDA result failed rarity analysis")
    primary = matches_found[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
    record = {
        "schema_version": RARE_LOG_SCHEMA,
        "found_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger": ruleset.watch_reasons[rule],
        "reason": primary.reason,
        "match_length": primary.length,
        "rarity_bits": primary.rarity_bits,
        "mean_attempts": primary.mean_attempts,
        "matches": [asdict(match) for match in matches_found],
        "public_key": public_hex,
        "private_key": private_hex,
        "backend": "cuda",
        "engine": engine,
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_fingerprint": ruleset.fingerprint,
        "active_rule_ids": list(active_rule_ids(ruleset)),
    }
    with os.fdopen(descriptor, "a", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(json.dumps(record, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return record


TEST_PRIVATE = bytes.fromhex(
    "7065e18fd9fabb70c1ed90dca19907de698c88b709ea146eafd93d9b830c7b60"
    "c4681193c7b9bc39945ba8064104bb618f8fd7a84a0af6f57033d6e8ddcd6471")
TEST_PUBLIC = "1ec77175b0918ed206f9ae04ec136d6d5d4315bb26305427f645b492e9350c10"

# A deterministic, non-secret second identity makes the firmware's two-sided
# ECDH validation reproducible without depending on a device or random input.
_ECDH_TEST_DIGEST = bytearray(
    hashlib.sha512(b"MeshCore vanity ECDH validation peer").digest()
)
_ECDH_TEST_DIGEST[0] &= 248
_ECDH_TEST_DIGEST[31] &= 63
_ECDH_TEST_DIGEST[31] |= 64
ECDH_TEST_PEER_PRIVATE = bytes(_ECDH_TEST_DIGEST)
ECDH_TEST_PEER_PUBLIC = bytes.fromhex(
    "8fca689524405a1529f7ae57f3b11e5e40fe32d73432401914393e9ca4b16b9a"
)
ECDH_TEST_SHARED = bytes.fromhex(
    "86d47e289ad85d9b272a0fd1a739f6931d47ce0be5c5ea5ba6206644b73be228"
)
del _ECDH_TEST_DIGEST


def firmware_compatibility_test() -> bool:
    """Exercise the same derivation, signing, and ECDH checks as firmware."""
    public = bytes.fromhex(TEST_PUBLIC)
    if not verify_expanded_key(TEST_PRIVATE, public):
        return False
    try:
        shared = meshcore_shared_secret(TEST_PRIVATE, ECDH_TEST_PEER_PUBLIC)
        return secrets.compare_digest(shared, ECDH_TEST_SHARED)
    except (ValueError, RuntimeError):
        return False


def self_test() -> bool:
    valid = firmware_compatibility_test()
    print("PASS" if valid else "FAIL",
          "MeshCore firmware derivation, signature, and shared-secret vector")
    return valid


def _run_gui_mainloop(
        mainloop: Callable[[], None], close_window: Callable[[], None],
) -> int:
    """Translate a terminal interrupt into the GUI's normal cleanup path."""
    try:
        mainloop()
    except KeyboardInterrupt:
        close_window()
        return 130
    return 0


def run_gui(
        ruleset: RareRuleset = DEFAULT_RARE_RULESET, *,
        persist_rule_selection: bool = True,
) -> int:
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError:
        print("Tk is unavailable. Install python3-tk or use the command line.", file=sys.stderr)
        return 2

    active_ruleset = (
        load_gui_rule_selection(ruleset) if persist_rule_selection else ruleset
    )
    root = tk.Tk(className="MeshCoreVanityKeygen")
    root.title(f"MeshCore Vanity Key Generator {APP_VERSION}")
    if ICON_PATH.is_file():
        try:
            window_icon = tk.PhotoImage(file=str(ICON_PATH))
            root.iconphoto(True, window_icon)
        except tk.TclError:
            window_icon = None
    root.minsize(780, 690)
    frame = ttk.Frame(root, padding=16)
    frame.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    # Render first. Driver discovery, nvidia-smi, native probes, and the
    # firmware compatibility vector all run after Tk's event loop is ready.
    details: dict[str, object] = {
        "firmware_vector": None, "cuda_devices": 0, "cuda_names": [],
        "cuda_ready": False,
    }
    gpu_count = 0
    gpu_names: list[str] = []
    fields: dict[str, tk.StringVar] = {
        name: tk.StringVar() for name in ("prefix", "suffix", "contains")
    }
    field_entries: dict[str, ttk.Entry] = {}
    for row, (name, value) in enumerate(fields.items()):
        ttk.Label(frame, text=f"{name.title()} (hex)").grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(frame, width=48, textvariable=value)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        field_entries[name] = entry

    estimate = tk.StringVar(value="Enter a hexadecimal pattern to see estimated difficulty.")
    ttk.Label(frame, textvariable=estimate).grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 8))

    workers = tk.IntVar(value=max(1, (os.cpu_count() or 2) - 1))
    ttk.Label(frame, text="CPU workers").grid(row=4, column=0, sticky="w", pady=3)
    ttk.Spinbox(frame, from_=1, to=max(1, os.cpu_count() or 1), width=8,
                textvariable=workers).grid(row=4, column=1, sticky="w")

    device = tk.StringVar()
    device_values = ("Discovering…",)
    device.set(device_values[0])
    ttk.Label(frame, text="CUDA device").grid(row=5, column=0, sticky="w", pady=3)
    device_combo = ttk.Combobox(
        frame, state="disabled", textvariable=device, values=device_values,
    )
    device_combo.grid(row=5, column=1, columnspan=2, sticky="ew")

    cuda_engine = tk.StringVar(value="optimized")
    ttk.Label(frame, text="CUDA engine").grid(row=6, column=0, sticky="w", pady=3)
    engine_combo = ttk.Combobox(
        frame, width=14, state="disabled", textvariable=cuda_engine,
        values=("optimized", "baseline"),
    )
    engine_combo.grid(row=6, column=1, sticky="w")
    collector_mode = tk.BooleanVar(value=False)

    def toggle_collector_mode() -> None:
        collecting = collector_mode.get()
        for entry in field_entries.values():
            entry.configure(state="disabled" if collecting else "normal")
        button.configure(text="Start rare collector" if collecting else "Find vanity key")
        refresh_estimate()

    collector_checkbox = ttk.Checkbutton(
        frame, text="Continuous rare collector", variable=collector_mode,
        command=toggle_collector_mode, state="disabled",
    )
    collector_checkbox.grid(row=6, column=2, sticky="e")

    rare_rule_summary = tk.StringVar(
        value=rare_rule_selection_summary(active_ruleset),
    )
    ttk.Label(frame, text="Rare keys to keep").grid(
        row=7, column=0, sticky="w", pady=3,
    )
    ttk.Label(frame, textvariable=rare_rule_summary).grid(
        row=7, column=1, sticky="w", pady=3,
    )
    rare_rule_button = ttk.Button(frame, text="Choose…", state="disabled")
    rare_rule_button.grid(row=7, column=2, padx=(7, 0))

    output_directory = tk.StringVar(value=str(DEFAULT_RESULTS_DIR))
    ttk.Label(frame, text="Results folder").grid(row=8, column=0, sticky="w", pady=3)
    ttk.Entry(frame, textvariable=output_directory).grid(row=8, column=1, sticky="ew", pady=3)

    def choose_output_directory() -> None:
        selected = filedialog.askdirectory(initialdir=output_directory.get() or str(DEFAULT_RESULTS_DIR))
        if selected:
            output_directory.set(selected)

    ttk.Button(frame, text="Choose…", command=choose_output_directory).grid(row=8, column=2, padx=(7, 0))

    diagnostic_text = tk.StringVar(
        value=(
            "Self-test: checking…  •  CUDA: discovering…  •  "
            f"Devices: …  •  Version: {APP_VERSION}"
        )
    )
    ttk.Label(frame, textvariable=diagnostic_text).grid(
        row=9, column=0, columnspan=3, sticky="w", pady=(8, 3),
    )
    temperature_text = tk.StringVar(value="CPU package —  •  GPU —")
    ttk.Label(frame, text="Temperatures").grid(
        row=10, column=0, sticky="w", pady=3,
    )
    ttk.Label(frame, textvariable=temperature_text).grid(
        row=10, column=1, columnspan=2, sticky="w", pady=3,
    )
    status = tk.StringVar(value="Discovering CUDA devices and checking compatibility…")
    ttk.Label(frame, textvariable=status).grid(row=11, column=0, columnspan=3, sticky="w", pady=3)
    incidental = tk.StringVar(value="Saved rare incidental keys: counting…")
    ttk.Label(frame, textvariable=incidental).grid(row=12, column=0, columnspan=3, sticky="w", pady=(0, 3))
    output = tk.Text(frame, width=86, height=10, state="disabled", wrap="word")
    output.grid(row=13, column=0, columnspan=3, sticky="nsew", pady=5)
    frame.rowconfigure(13, weight=1)
    state: dict[str, object] = {
        "result": None, "cancel": threading.Event(), "searching": False,
        "process": None, "closing": False, "saved_path": None,
        "reveal_private": False, "observed_rate": None,
        "rare_count": None, "rare_count_generation": 0,
        "collector_started": None, "collector_session_count": 0,
        "collector_best_bits": 0.0, "collector_best_reason": None,
        "discovery_pending": True, "discovery_error": None,
        "cuda_probes": {},
    }
    ui_events: queue.SimpleQueue[
        tuple[Callable[..., None], tuple[object, ...]]
    ] = queue.SimpleQueue()
    shutdown_event = threading.Event()
    history_load_cancels: set[threading.Event] = set()
    process_lock = threading.Lock()
    active_process: list[Optional[subprocess.Popen[str]]] = [None]
    temperature_cache = TemperatureCache()
    temperature_update_lock = threading.Lock()
    latest_temperature: list[Optional[TemperatureSnapshot]] = [None]
    temperature_event_pending = [False]

    def open_rare_rule_selector() -> None:
        if state["searching"]:
            return
        chooser = tk.Toplevel(root)
        chooser.title("Rare Keys to Keep")
        chooser.geometry("800x540")
        chooser.minsize(620, 420)
        chooser.transient(root)
        chooser.columnconfigure(0, weight=1)
        chooser.rowconfigure(1, weight=1)
        if ICON_PATH.is_file():
            try:
                chooser.iconphoto(True, window_icon)
            except (tk.TclError, UnboundLocalError):
                pass

        introduction = (
            "Choose which rare identities are saved during vanity searches and "
            "continuous collection. This affects new discoveries only; saved "
            "history is never removed."
        )
        if not persist_rule_selection:
            introduction += " Choices from this custom rules file last for this app session."
        ttk.Label(
            chooser, text=introduction, wraplength=740, justify="left",
            padding=(14, 14, 14, 8),
        ).grid(row=0, column=0, sticky="ew")

        list_frame = ttk.Frame(chooser, padding=(14, 0, 14, 8))
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        background = ttk.Style(chooser).lookup("TFrame", "background") or "#d9d9d9"
        canvas = tk.Canvas(
            list_frame, highlightthickness=0, background=background,
        )
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=canvas.yview,
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        choices = ttk.Frame(canvas)
        choices.columnconfigure(0, weight=1)
        choices_window = canvas.create_window(
            (0, 0), window=choices, anchor="nw",
        )

        def resize_choices(_event: object = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_choices_width(event: object) -> None:
            width = getattr(event, "width", 0)
            if isinstance(width, int) and width > 0:
                canvas.itemconfigure(choices_window, width=width)

        choices.bind("<Configure>", resize_choices)
        canvas.bind("<Configure>", resize_choices_width)
        selected_ids = set(active_rule_ids(active_ruleset))
        choice_vars: dict[str, tk.BooleanVar] = {}
        validation_text = tk.StringVar()
        for row, rule in enumerate(ruleset.rules):
            variable = tk.BooleanVar(value=rule.id in selected_ids)
            choice_vars[rule.id] = variable
            ttk.Checkbutton(
                choices, text=rare_rule_choice_label(rule), variable=variable,
                command=lambda: validation_text.set(""),
            ).grid(row=row, column=0, sticky="w", padx=6, pady=7)

        def set_choices(enabled: bool) -> None:
            for variable in choice_vars.values():
                variable.set(enabled)
            validation_text.set("")

        def restore_configured_defaults() -> None:
            defaults = {rule.id for rule in ruleset.active_rules}
            for rule_id, variable in choice_vars.items():
                variable.set(rule_id in defaults)
            validation_text.set("")

        def close_chooser() -> None:
            try:
                chooser.grab_release()
            except tk.TclError:
                pass
            chooser.destroy()

        def apply_choices() -> None:
            nonlocal active_ruleset
            enabled = [
                rule.id for rule in ruleset.rules if choice_vars[rule.id].get()
            ]
            try:
                selected = select_rules(ruleset, enabled)
            except RuleConfigError as error:
                validation_text.set(str(error).capitalize())
                return
            active_ruleset = selected
            rare_rule_summary.set(rare_rule_selection_summary(selected))
            if persist_rule_selection:
                try:
                    save_gui_rule_selection(ruleset, selected)
                except (OSError, ValueError) as error:
                    messagebox.showwarning(
                        "Selection not remembered",
                        "The selection is active for this session, but could not be "
                        f"saved for the next launch.\n\n{error}",
                        parent=chooser,
                    )
            close_chooser()

        chooser.bind("<Escape>", lambda _event: close_chooser())
        footer = ttk.Frame(chooser, padding=(14, 4, 14, 14))
        footer.grid(row=2, column=0, sticky="ew")
        ttk.Button(
            footer, text="Select all", command=lambda: set_choices(True),
        ).pack(side="left")
        ttk.Button(
            footer, text="Restore configured defaults",
            command=restore_configured_defaults,
        ).pack(side="left", padx=7)
        ttk.Label(footer, textvariable=validation_text).pack(side="left", padx=8)
        ttk.Button(footer, text="Cancel", command=close_chooser).pack(side="right")
        ttk.Button(footer, text="Apply", command=apply_choices).pack(
            side="right", padx=(0, 7),
        )
        chooser.protocol("WM_DELETE_WINDOW", close_chooser)
        chooser.grab_set()
        chooser.focus_set()

    rare_rule_button.configure(command=open_rare_rule_selector, state="normal")

    def post_ui(callback: Callable[..., None], *args: object) -> None:
        """Queue a callback for execution by Tk's main thread."""
        ui_events.put((callback, args))

    def selected_device_index() -> Optional[int]:
        if state["discovery_pending"] or gpu_count < 1:
            return None
        match = re.match(r"(\d+)\s", device.get())
        if match is None:
            return None
        selected = int(match.group(1))
        return selected if 0 <= selected < gpu_count else None

    def selected_cuda_probe() -> Optional[dict[str, object]]:
        selected = selected_device_index()
        probes = state.get("cuda_probes")
        if selected is None or not isinstance(probes, dict):
            return None
        probe = probes.get((selected, cuda_engine.get()))
        return probe if isinstance(probe, dict) else None

    def selected_cuda_ready() -> bool:
        probe = selected_cuda_probe()
        return bool(probe and probe.get("ready"))

    def render_temperatures() -> None:
        selected = selected_device_index()
        probe = selected_cuda_probe()
        pci_bus_id = (
            str(probe.get("pci_bus_id"))
            if isinstance(probe, dict) and probe.get("pci_bus_id") is not None
            else None
        )
        temperature_text.set(format_temperature_status(
            temperature_cache, selected, pci_bus_id, gpu_count,
        ))

    def apply_latest_temperature() -> None:
        with temperature_update_lock:
            snapshot = latest_temperature[0]
            temperature_event_pending[0] = False
        if state["closing"] or snapshot is None:
            return
        update_temperature_cache(temperature_cache, snapshot)
        render_temperatures()

    def queue_temperature_snapshot(snapshot: TemperatureSnapshot) -> None:
        if shutdown_event.is_set():
            return
        with temperature_update_lock:
            latest_temperature[0] = snapshot
            if temperature_event_pending[0]:
                return
            temperature_event_pending[0] = True
        post_ui(apply_latest_temperature)

    def render_diagnostic_summary() -> None:
        if state["discovery_pending"]:
            return
        firmware = details.get("firmware_vector")
        vector_label = "PASS" if firmware is True else "FAIL"
        if gpu_count < 1:
            cuda_label = "not detected (CPU available)"
        else:
            probe = selected_cuda_probe()
            cuda_label = (
                f"{cuda_engine.get()} ready" if probe and probe.get("ready")
                else f"{cuda_engine.get()} unavailable (CPU available)"
            )
        diagnostic_text.set(
            f"Self-test: {vector_label}  •  CUDA: {cuda_label}  •  "
            f"Devices: {gpu_count}  •  Version: {APP_VERSION}"
        )

    def update_backend_controls(*_args: object) -> None:
        if state["closing"]:
            return
        pending = bool(state["discovery_pending"])
        searching = bool(state["searching"])
        rare_rule_button.configure(state="disabled" if searching else "normal")
        render_temperatures()
        if pending or searching:
            button.configure(state="disabled")
            collector_checkbox.configure(state="disabled")
            device_combo.configure(state="disabled")
            engine_combo.configure(state="disabled")
            return
        device_combo.configure(state="readonly" if gpu_count else "disabled")
        engine_combo.configure(state="readonly" if gpu_count else "disabled")
        ready = selected_cuda_ready()
        if collector_mode.get() and not ready:
            collector_mode.set(False)
            toggle_collector_mode()
        collector_checkbox.configure(state="normal" if ready else "disabled")
        button.configure(state="normal")
        render_diagnostic_summary()

    def apply_gui_diagnostics(
            discovered: Optional[dict[str, object]], error_text: Optional[str],
    ) -> None:
        nonlocal gpu_count, gpu_names
        if state["closing"]:
            return
        report = discovered or {
            "firmware_vector": False, "cuda_devices": 0, "cuda_names": [],
            "cuda_ready": False, "cuda_engine_probes": {},
        }
        details.clear()
        details.update(report)
        raw_count = report.get("cuda_devices", 0)
        gpu_count = (
            raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            and raw_count >= 0 else 0
        )
        raw_names = report.get("cuda_names", [])
        gpu_names = (
            [str(name) for name in raw_names] if isinstance(raw_names, list) else []
        )
        raw_probes = report.get("cuda_engine_probes", {})
        state["cuda_probes"] = dict(raw_probes) if isinstance(raw_probes, dict) else {}
        state["discovery_error"] = error_text
        state["discovery_pending"] = False

        values = tuple(
            f"{index} — {gpu_names[index] if index < len(gpu_names) else 'NVIDIA GPU'}"
            for index in range(gpu_count)
        ) or ("None detected",)
        device_combo.configure(values=values)
        device.set(values[0])
        update_backend_controls()
        if error_text:
            status.set(f"Ready for CPU searches; hardware diagnostics failed: {error_text}")
        elif details.get("firmware_vector") is not True:
            status.set("Ready; firmware compatibility self-test failed")
        elif any(
                bool(probe.get("ready"))
                for probe in state["cuda_probes"].values()
                if isinstance(probe, dict)):
            status.set("Ready")
        else:
            status.set("Ready; CUDA unavailable, CPU searches are available")
        refresh_estimate()

    def poll_ui_events() -> None:
        if state["closing"]:
            return
        def report_callback_error(error: Exception) -> None:
            print(
                f"GUI update callback failed: {type(error).__name__}",
                file=sys.stderr,
            )
            if not state["closing"]:
                status.set(
                    "A background interface update failed; the GUI remains usable"
                )

        try:
            # Bound each drain so a burst of progress events cannot starve Tk.
            drain_callback_queue(ui_events, on_error=report_callback_error)
        finally:
            # A failed callback must never permanently stop UI event delivery.
            if not state["closing"]:
                root.after(25, poll_ui_events)

    initial_count_generation = int(state["rare_count_generation"])

    def apply_initial_rare_count(total: int, generation: int) -> None:
        if not apply_rare_count_snapshot(state, generation, total):
            return
        incidental.set(f"Saved rare incidental keys: {total}")

    def count_initial_records() -> None:
        try:
            total = count_interesting(DEFAULT_WATCH_PATH, cancel=shutdown_event)
        except OSError:
            return
        if shutdown_event.is_set():
            return
        post_ui(apply_initial_rare_count, total, initial_count_generation)

    threading.Thread(target=count_initial_records, daemon=True).start()

    def show(text: str) -> None:
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("1.0", text)
        output.configure(state="disabled")

    def selected_results_directory() -> Path:
        value = output_directory.get().strip()
        if not value:
            raise ValueError("Choose a results folder")
        path = Path(value).expanduser().resolve()
        if path.exists() and not path.is_dir():
            raise ValueError("The selected results path is not a folder")
        return path

    def refresh_estimate(*_args: object) -> None:
        try:
            if collector_mode.get():
                estimate.set("Continuous rare collection runs until cancelled; no vanity target is needed.")
                return
            values = {name: valid_pattern(var.get(), name) for name, var in fields.items()}
            if not any(values.values()):
                estimate.set("Enter a hexadecimal pattern to see estimated difficulty.")
                return
            validate_constraints(**values)
            attempts = estimate_attempts(**values)
            rate = state.get("observed_rate")
            if not isinstance(rate, (int, float)) or rate <= 0:
                if state["discovery_pending"]:
                    estimate.set(
                        f"Mean work: {attempts:,.0f} candidates  •  "
                        "GPU discovery in progress"
                    )
                    return
                rate = (
                    REFERENCE_CUDA_RATE if selected_cuda_ready()
                    else max(1, workers.get()) * 20_000
                )
            qualifier = "approximate " if values["contains"] else ""
            estimate.set(
                f"Mean work: {qualifier}{attempts:,.0f} candidates  •  "
                f"Estimated average: {format_duration(attempts / rate)} at {rate:,.0f} keys/s"
            )
        except (ValueError, tk.TclError) as error:
            estimate.set(str(error))

    for variable in fields.values():
        variable.trace_add("write", refresh_estimate)
    workers.trace_add("write", refresh_estimate)

    def backend_selection_changed(_event: object = None) -> None:
        update_backend_controls()
        refresh_estimate()

    device_combo.bind("<<ComboboxSelected>>", backend_selection_changed)
    engine_combo.bind("<<ComboboxSelected>>", backend_selection_changed)

    def render_result() -> None:
        result = state.get("result")
        if not isinstance(result, Result):
            return
        saved_path = state.get("saved_path")
        private = result.private_key if state["reveal_private"] else "•" * 32 + "  (hidden)"
        show(
            f"PUBLIC KEY (64 hex characters):\n{result.public_key}\n\n"
            f"PRIVATE KEY (128 hex characters):\n{private}\n\n"
            f"Saved automatically: {saved_path or 'SAVE FAILED — use Save a copy'}\n\n"
            "Import the private key as MeshCore prv.key, then reboot. Keep the JSON file private."
        )

    def set_result_controls(enabled: bool) -> None:
        new_state = "normal" if enabled else "disabled"
        reveal_button.configure(state=new_state)
        copy_public_button.configure(state=new_state)
        copy_private_button.configure(state=new_state)
        save_button.configure(state=new_state)

    def start_search() -> None:
        if state["discovery_pending"]:
            status.set("CUDA discovery is still in progress; please wait")
            return
        collecting = collector_mode.get()
        search_ruleset = active_ruleset
        selected_device = selected_device_index()
        if selected_device is None:
            selected_device = 0
        selected_engine = cuda_engine.get()
        using_cuda = selected_cuda_ready()
        try:
            values = {name: valid_pattern(var.get(), name) for name, var in fields.items()}
            worker_count = workers.get()
            if not 1 <= worker_count <= max(1, os.cpu_count() or 1):
                raise ValueError("CPU workers must be within the range shown")
            if not collecting and not any(values.values()):
                raise ValueError("Enter a prefix, suffix, or substring to search for")
            if collecting:
                values = {"prefix": "", "suffix": "", "contains": ""}
                if not using_cuda:
                    raise ValueError("Continuous rare collection requires a working CUDA engine")
            else:
                validate_constraints(**values)
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Invalid pattern", str(error))
            return
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.clear()
        state["searching"] = True
        state["result"] = None
        state["saved_path"] = None
        state["reveal_private"] = False
        state["collector_started"] = time.monotonic() if collecting else None
        state["collector_session_count"] = 0
        state["collector_best_bits"] = 0.0
        state["collector_best_reason"] = None
        reveal_button.configure(text="Reveal private key")
        set_result_controls(False)
        try:
            result_directory = selected_results_directory()
            result_directory.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as error:
            state["searching"] = False
            messagebox.showerror("Invalid results folder", str(error))
            return
        state["rare_count_generation"] = int(state["rare_count_generation"]) + 1
        watch_path = result_directory / "rare-keys.jsonl"
        # Counting an arbitrarily large history belongs off Tk's main thread.
        # Keep an already-known default count; otherwise report this session
        # until the browser performs its own bounded background tail load.
        known_total = (state.get("rare_count")
                       if watch_path == DEFAULT_WATCH_PATH else None)
        state["rare_count"] = known_total
        if collecting:
            total_label = f"{known_total}" if isinstance(known_total, int) else "counting deferred"
            incidental.set(f"Rare keys total: {total_label} | this session: 0 | best: none yet")
        else:
            incidental.set(
                f"Saved rare incidental keys: {known_total}"
                if isinstance(known_total, int)
                else "Saved rare incidental keys this session: 0"
            )
        update_backend_controls()
        cancel_button.configure(state="normal")
        activity.start(12)
        if collecting:
            status.set("Starting continuous rare collector…")
            show("Continuous rare-key collection is active. It will run until you press Cancel.\nEvery verified discovery is saved immediately.")
        elif using_cuda:
            status.set("Starting CUDA search…")
            show("CUDA search is active. Longer patterns may take minutes or hours.\nRare incidental keys are being saved while you wait.")
        else:
            status.set("Starting CPU search…")
            show("CPU fallback search is active. Automatic rare-key collection requires CUDA.")

        def display_progress(attempts: int, elapsed: float) -> None:
            rate = attempts / max(elapsed, .001)
            if attempts:
                state["observed_rate"] = rate
            label = "Collecting" if collecting else "Searching"
            status.set(
                f"{label}: {attempts:,} keys, {rate:,.0f} keys/s, {format_duration(elapsed)}"
            )
            refresh_estimate()

        def progress(attempts: int, elapsed: float) -> None:
            post_ui(display_progress, attempts, elapsed)

        def display_watch_progress(reason: str, public_key: str, path: Path) -> None:
            known = state.get("rare_count")
            if isinstance(known, int):
                state["rare_count"] = known + 1
            state["collector_session_count"] = int(
                state.get("collector_session_count", 0)
            ) + 1
            session_count = int(state["collector_session_count"])
            if collecting:
                analyzed = interesting_matches(public_key, search_ruleset)
                rarity_bits = analyzed[0].rarity_bits if analyzed else 0.0
                if rarity_bits > float(state.get("collector_best_bits", 0.0)):
                    state["collector_best_bits"] = rarity_bits
                    state["collector_best_reason"] = reason
                best = state.get("collector_best_reason") or "none yet"
                total_label = (str(state["rare_count"])
                               if isinstance(state.get("rare_count"), int)
                               else "not counted")
                incidental.set(
                    f"Rare keys total: {total_label} | this session: {session_count} | "
                    f"best: {best} ({float(state['collector_best_bits']):.1f} bits)"
                )
            else:
                count_label = (f"total: {state['rare_count']}"
                               if isinstance(state.get("rare_count"), int)
                               else f"this session: {session_count}")
                incidental.set(
                    f"Saved rare incidental keys {count_label} | latest: {reason} | "
                    f"{public_key[:16]}… | saved: {path}"
                )

        def watch_progress(_rule: int, reason: str, public_key: str, path: Path) -> None:
            post_ui(display_watch_progress, reason, public_key, path)

        def display_process(process: Optional[subprocess.Popen[str]]) -> None:
            state["process"] = process

        def process_progress(process: Optional[subprocess.Popen[str]]) -> None:
            # Process lifetime is operational synchronization state, not Tk
            # state. Mirroring it under a lock lets Cancel/Close terminate a
            # just-started child even before the next UI queue poll.
            with process_lock:
                active_process[0] = process
            post_ui(display_process, process)

        def job() -> None:
            try:
                if using_cuda:
                    result = search_cuda(**values, update=progress, cancel=cancel_event,
                                         watch_path=watch_path, watch_update=watch_progress,
                                         device=selected_device,
                                         engine=selected_engine,
                                         interactive=True,
                                         collect_only=collecting,
                                         process_update=process_progress,
                                         ruleset=search_ruleset)
                else:
                    result = search(**values, workers=worker_count, update=progress,
                                    cancel=cancel_event)
                post_ui(complete, result, result_directory)
            except SearchCancelled:
                post_ui(stopped)
            except Exception as error:
                post_ui(failed, str(error))

        # A non-daemon coordinator guarantees that a CUDA child spawned during
        # the close/register race reaches search_cuda's cleanup path.
        threading.Thread(target=job, daemon=False).start()

    def complete(result: Result, result_directory: Path) -> None:
        state["searching"] = False
        activity.stop()
        backend_label = result.backend.upper() + (f"/{result.engine}" if result.engine else "")
        status.set(f"Found with {backend_label} after ≤{result.attempts:,} attempts in {result.elapsed_seconds:.3f}s")
        state["result"] = result
        path = available_result_path(result_directory, result.public_key)
        try:
            save_result(result, path)
            state["saved_path"] = path
        except Exception as error:
            state["saved_path"] = None
            messagebox.showerror(
                "Automatic save failed",
                f"The key is still available in this window. Save a copy before closing.\n\n{error}",
            )
        render_result()
        set_result_controls(True)
        update_backend_controls()
        cancel_button.configure(state="disabled")

    def stopped() -> None:
        state["searching"] = False
        activity.stop()
        status.set("Search cancelled")
        if bool(state.get("collector_started")):
            show(
                f"Rare collector stopped. Saved {state['collector_session_count']} verified "
                "keys during this session; all completed discoveries remain saved."
            )
        else:
            show("Search cancelled. Interesting keys found before cancellation remain saved.")
        update_backend_controls()
        cancel_button.configure(state="disabled")

    def failed(error_text: str) -> None:
        state["searching"] = False
        activity.stop()
        update_backend_controls()
        cancel_button.configure(state="disabled")
        messagebox.showerror("Search failed", error_text)

    def cancel_search() -> None:
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.set()
        with process_lock:
            process = active_process[0]
        if isinstance(process, subprocess.Popen):
            signal_process_termination(process)
        status.set("Stopping search…")
        cancel_button.configure(state="disabled")

    def save_copy() -> None:
        result = state["result"]
        if not isinstance(result, Result):
            messagebox.showinfo("Nothing to save", "Find a key first.")
            return
        saved_path = state.get("saved_path")
        initial_directory = (
            saved_path.parent if isinstance(saved_path, Path) else DEFAULT_RESULTS_DIR
        )
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", initialdir=str(initial_directory),
            initialfile=f"meshcore-identity-{result.public_key[:12]}.json",
        )
        if filename:
            path = Path(filename)
            overwrite = False
            if path.exists():
                overwrite = messagebox.askyesno("Replace file?", f"Replace the existing file?\n\n{path}")
                if not overwrite:
                    return
            try:
                save_result(result, path, overwrite=overwrite)
                messagebox.showinfo("Saved", "Saved atomically with owner-only permissions.")
            except (OSError, ValueError) as error:
                messagebox.showerror("Save failed", str(error))

    def copy_value(private: bool) -> None:
        result = state.get("result")
        if not isinstance(result, Result):
            return
        value = result.private_key if private else result.public_key
        root.clipboard_clear()
        root.clipboard_append(value)
        status.set("Private key copied; clipboard will be cleared in 60 seconds" if private else "Public key copied")
        if private:
            def clear_if_unchanged() -> None:
                try:
                    if root.clipboard_get() == value:
                        root.clipboard_clear()
                        status.set("Private key cleared from clipboard")
                except tk.TclError:
                    pass
            root.after(60_000, clear_if_unchanged)

    def toggle_private() -> None:
        state["reveal_private"] = not bool(state["reveal_private"])
        reveal_button.configure(text="Hide private key" if state["reveal_private"] else "Reveal private key")
        render_result()

    def open_results() -> None:
        try:
            path = selected_results_directory()
            path.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open results", str(error))

    def open_rare_browser() -> None:
        try:
            path = selected_results_directory() / "rare-keys.jsonl"
        except ValueError as error:
            messagebox.showerror("Cannot open rare keys", str(error))
            return

        browser = tk.Toplevel(root)
        browser.title("Saved Rare MeshCore Keys")
        browser.geometry("1180x680")
        browser.minsize(820, 500)
        browser.columnconfigure(0, weight=1)
        browser.rowconfigure(1, weight=1)
        if ICON_PATH.is_file():
            try:
                browser.iconphoto(True, window_icon)
            except (tk.TclError, UnboundLocalError):
                pass

        filter_value = tk.StringVar()
        browser_status = tk.StringVar(value="Loading rare keys…")
        toolbar = ttk.Frame(browser, padding=(10, 10, 10, 5))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Filter").grid(row=0, column=0, padx=(0, 7))
        filter_entry = ttk.Entry(toolbar, textvariable=filter_value)
        filter_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(toolbar, textvariable=browser_status).grid(row=0, column=2, padx=(12, 0))

        table_frame = ttk.Frame(browser, padding=(10, 5))
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        table_font = tkfont.nametofont("TkDefaultFont")
        heading_font = tkfont.nametofont("TkHeadingFont")
        row_height = max(30, table_font.metrics("linespace") + 12)
        browser_style = ttk.Style(browser)
        browser_style.configure("RareKeys.Treeview", font=table_font, rowheight=row_height)
        browser_style.configure(
            "RareKeys.Treeview.Heading", font=heading_font, padding=(8, 6)
        )
        columns = ("found", "reason", "length", "rarity", "public")
        tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse",
            style="RareKeys.Treeview",
        )
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        headings = {
            "found": "Found UTC", "reason": "Strongest match", "length": "Chars",
            "rarity": "Rarity", "public": "Public key",
        }
        widths = {"found": 185, "reason": 245, "length": 80, "rarity": 105, "public": 455}
        for column in columns:
            anchor = "center" if column in ("length", "rarity") else "w"
            tree.heading(column, text=headings[column], anchor=anchor)
            tree.column(
                column, width=widths[column], minwidth=70, anchor=anchor,
                stretch=column == "public",
            )

        detail = tk.Text(browser, height=8, state="disabled", wrap="char")
        detail.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 5))
        browser_state: dict[str, object] = {
            "records": [], "visible": {}, "sort": "found", "reverse": True,
            "private_visible": False, "skipped": 0, "page": 0,
            "pages": 1, "matching": 0, "loading": False,
            "load_generation": 0, "closed": False, "filter_after": None,
            "load_cancel": None,
        }

        def selected_record(require_valid_private: bool = False) -> Optional[dict[str, object]]:
            selection = tree.selection()
            visible = browser_state["visible"]
            assert isinstance(visible, dict)
            record = visible.get(selection[0]) if selection else None
            if not isinstance(record, dict):
                messagebox.showinfo("No selection", "Select a rare key first.", parent=browser)
                return None
            if require_valid_private:
                private_hex = str(record["private_key"])
                public_hex = str(record["public_key"])
                if not verify_expanded_key(bytes.fromhex(private_hex), bytes.fromhex(public_hex)):
                    messagebox.showerror(
                        "Verification failed",
                        "This saved record did not pass independent key verification.",
                        parent=browser,
                    )
                    return None
            return record

        def render_selected(*_args: object) -> None:
            selection = tree.selection()
            visible = browser_state["visible"]
            assert isinstance(visible, dict)
            record = visible.get(selection[0]) if selection else None
            detail.configure(state="normal")
            detail.delete("1.0", "end")
            if isinstance(record, dict):
                private = (str(record["private_key"])
                           if browser_state["private_visible"] else "•" * 32 + "  (hidden)")
                match_items = record.get("matches", [])
                match_names = [
                    str(item.get("reason")) for item in match_items
                    if isinstance(item, dict) and item.get("reason")
                ]
                saved_policy = _history_rule_ids(record.get("active_rule_ids"))
                detail.insert(
                    "1.0",
                    f"Public key: {record['public_key']}\n"
                    f"Private key: {private}\n"
                    f"Rarity: {_history_rarity(record.get('rarity_bits', 0)):.1f} bits"
                    f"  •  Match length: {_history_length(record.get('match_length', 0))}"
                    f"  •  All matches: {', '.join(match_names) or record.get('reason', 'unknown')}\n"
                    f"Saved policy: {', '.join(saved_policy) if saved_policy else 'not recorded'}",
                )
            detail.configure(state="disabled")

        def refresh_view(reset_page: bool = False) -> None:
            if reset_page:
                browser_state["page"] = 0
            records = browser_state["records"]
            assert isinstance(records, list)
            page_records, matching, selected_page, pages = select_interesting_page(
                records, filter_value.get(), str(browser_state["sort"]),
                bool(browser_state["reverse"]), int(browser_state["page"]),
            )
            browser_state["page"] = selected_page
            browser_state["pages"] = pages
            browser_state["matching"] = matching
            children = tree.get_children()
            if children:
                tree.delete(*children)
            visible: dict[str, dict[str, object]] = {}
            for index, record in enumerate(page_records):
                item = f"record-{selected_page}-{index}"
                visible[item] = record
                found_at = str(record.get("found_at", "")).replace("T", " ").replace("Z", "")[:19]
                rarity = f"{_history_rarity(record.get('rarity_bits', 0)):.1f} bits"
                tree.insert("", "end", iid=item, values=(
                    found_at, record.get("reason", "unknown"),
                    _history_length(record.get("match_length", 0)), rarity,
                    record.get("public_key", ""),
                ))
            browser_state["visible"] = visible
            browser_state["private_visible"] = False
            rare_reveal.configure(text="Reveal private")
            children = tree.get_children()
            if children:
                tree.selection_set(children[0])
                tree.focus(children[0])
            skipped = int(browser_state["skipped"])
            note = f" • {skipped} malformed skipped" if skipped else ""
            limited = (f" • newest {RARE_BROWSER_LIMIT:,} retained"
                       if len(records) == RARE_BROWSER_LIMIT else "")
            if matching:
                first = selected_page * RARE_BROWSER_PAGE_SIZE + 1
                last = first + len(page_records) - 1
                range_label = f"{first:,}–{last:,} of {matching:,} matches"
            else:
                range_label = "0 matches"
            browser_status.set(
                f"{range_label} • {len(records):,} loaded{limited}{note}"
            )
            page_status.set(f"Page {selected_page + 1:,} of {pages:,}")
            previous_button.configure(state="normal" if selected_page > 0 else "disabled")
            next_button.configure(
                state="normal" if selected_page + 1 < pages else "disabled"
            )
            render_selected()

        def sort_by(column: str) -> None:
            if browser_state["sort"] == column:
                browser_state["reverse"] = not bool(browser_state["reverse"])
            else:
                browser_state["sort"] = column
                browser_state["reverse"] = column in ("found", "length", "rarity")
            refresh_view(reset_page=True)

        for column in columns:
            tree.heading(column, text=headings[column], command=lambda value=column: sort_by(value))

        def set_loading(loading: bool) -> None:
            browser_state["loading"] = loading
            widget_state = "disabled" if loading else "normal"
            refresh_button.configure(state=widget_state)
            filter_entry.configure(state=widget_state)
            for control in record_action_buttons:
                control.configure(state=widget_state)
            if loading:
                previous_button.configure(state="disabled")
                next_button.configure(state="disabled")

        def load_progress(generation: int, scanned: int, total: int) -> None:
            if (browser_state["closed"]
                    or generation != browser_state["load_generation"]):
                return
            if total:
                browser_status.set(
                    f"Loading newest records… {scanned / total:.0%} of file scanned"
                )
            else:
                browser_status.set("Loading newest records…")

        def finish_load(
                generation: int, records: Optional[list[dict[str, object]]],
                skipped: int, error_text: Optional[str],
                load_cancel: threading.Event,
        ) -> None:
            history_load_cancels.discard(load_cancel)
            if browser_state.get("load_cancel") is load_cancel:
                browser_state["load_cancel"] = None
            if (browser_state["closed"]
                    or generation != browser_state["load_generation"]):
                return
            set_loading(False)
            if error_text is not None:
                browser_status.set("Could not load rare-key history")
                messagebox.showerror("Could not load rare keys", error_text, parent=browser)
                return
            browser_state["records"] = records or []
            browser_state["skipped"] = skipped
            refresh_view(reset_page=True)

        def reload_records() -> None:
            if browser_state["loading"]:
                return
            generation = int(browser_state["load_generation"]) + 1
            browser_state["load_generation"] = generation
            set_loading(True)
            browser_status.set("Loading newest records…")
            load_cancel = threading.Event()
            browser_state["load_cancel"] = load_cancel
            history_load_cancels.add(load_cancel)

            def loader() -> None:
                try:
                    records, skipped = load_interesting_records(
                        path,
                        progress=lambda scanned, total: post_ui(
                            load_progress, generation, scanned, total
                        ),
                        ruleset=ruleset,
                        cancel=load_cancel,
                    )
                except Exception as error:
                    post_ui(
                        finish_load, generation, None, 0, str(error), load_cancel,
                    )
                    return
                post_ui(
                    finish_load, generation, records, skipped, None, load_cancel,
                )

            threading.Thread(target=loader, daemon=True).start()

        def copy_selected(private: bool) -> None:
            record = selected_record(require_valid_private=private)
            if record is None:
                return
            if private and not messagebox.askyesno(
                    "Copy private key?",
                    "The private key grants control of this identity. Copy it to the clipboard?",
                    parent=browser):
                return
            value = str(record["private_key"] if private else record["public_key"])
            browser.clipboard_clear()
            browser.clipboard_append(value)
            browser_status.set("Private key copied; clipboard clears in 60 seconds"
                               if private else "Public key copied")
            if private:
                def clear_if_unchanged() -> None:
                    try:
                        if browser.clipboard_get() == value:
                            browser.clipboard_clear()
                            browser_status.set("Private key cleared from clipboard")
                    except tk.TclError:
                        pass
                browser.after(60_000, clear_if_unchanged)

        def toggle_rare_private() -> None:
            record = selected_record(require_valid_private=True)
            if record is None:
                return
            showing = bool(browser_state["private_visible"])
            if not showing and not messagebox.askyesno(
                    "Reveal private key?",
                    "Anyone who sees this private key can control the identity. Reveal it?",
                    parent=browser):
                return
            browser_state["private_visible"] = not showing
            rare_reveal.configure(text="Hide private" if not showing else "Reveal private")
            render_selected()

        def export_selected() -> None:
            record = selected_record(require_valid_private=True)
            if record is None:
                return
            filename = filedialog.asksaveasfilename(
                parent=browser, defaultextension=".json", initialdir=str(path.parent),
                initialfile=f"meshcore-rare-identity-{str(record['public_key'])[:12]}.json",
            )
            if not filename:
                return
            destination = Path(filename)
            overwrite = False
            if destination.exists() or destination.is_symlink():
                overwrite = messagebox.askyesno(
                    "Replace file?", f"Replace the existing file?\n\n{destination}", parent=browser
                )
                if not overwrite:
                    return
            try:
                atomic_write_json(record, destination, overwrite=overwrite)
                messagebox.showinfo(
                    "Saved", "Rare identity exported with owner-only permissions.", parent=browser
                )
            except (OSError, ValueError) as error:
                messagebox.showerror("Export failed", str(error), parent=browser)

        tree.bind("<<TreeviewSelect>>", lambda _event: (
            browser_state.__setitem__("private_visible", False),
            rare_reveal.configure(text="Reveal private"),
            render_selected(),
        ))

        def schedule_filter(*_args: object) -> None:
            pending = browser_state.get("filter_after")
            if isinstance(pending, str):
                try:
                    browser.after_cancel(pending)
                except tk.TclError:
                    pass
            browser_state["filter_after"] = browser.after(
                180, lambda: refresh_view(reset_page=True)
            )

        filter_value.trace_add("write", schedule_filter)
        browser_controls = ttk.Frame(browser, padding=(10, 5, 10, 10))
        browser_controls.grid(row=3, column=0, sticky="ew")
        refresh_button = ttk.Button(browser_controls, text="Refresh", command=reload_records)
        refresh_button.pack(side="left")
        copy_rare_public = ttk.Button(
            browser_controls, text="Copy public", command=lambda: copy_selected(False)
        )
        copy_rare_public.pack(side="left", padx=6)
        copy_rare_private = ttk.Button(
            browser_controls, text="Copy private", command=lambda: copy_selected(True)
        )
        copy_rare_private.pack(side="left")
        rare_reveal = ttk.Button(browser_controls, text="Reveal private", command=toggle_rare_private)
        rare_reveal.pack(side="left", padx=6)
        export_rare = ttk.Button(
            browser_controls, text="Export selected…", command=export_selected
        )
        export_rare.pack(side="left")
        record_action_buttons = (
            copy_rare_public, copy_rare_private, rare_reveal, export_rare,
        )
        def change_page(offset: int) -> None:
            browser_state["page"] = int(browser_state["page"]) + offset
            refresh_view()

        def close_browser() -> None:
            browser_state["closed"] = True
            browser_state["load_generation"] = int(browser_state["load_generation"]) + 1
            load_cancel = browser_state.get("load_cancel")
            if isinstance(load_cancel, threading.Event):
                load_cancel.set()
            pending = browser_state.get("filter_after")
            if isinstance(pending, str):
                try:
                    browser.after_cancel(pending)
                except tk.TclError:
                    pass
            browser.destroy()

        ttk.Button(browser_controls, text="Close", command=close_browser).pack(side="right")
        next_button = ttk.Button(
            browser_controls, text="Next", command=lambda: change_page(1), state="disabled"
        )
        next_button.pack(side="right", padx=(6, 0))
        previous_button = ttk.Button(
            browser_controls, text="Previous", command=lambda: change_page(-1), state="disabled"
        )
        previous_button.pack(side="right")
        page_status = tk.StringVar(value="Page 1 of 1")
        ttk.Label(browser_controls, textvariable=page_status).pack(
            side="right", padx=8,
        )
        browser.protocol("WM_DELETE_WINDOW", close_browser)
        reload_records()

    activity = ttk.Progressbar(frame, mode="indeterminate", length=620)
    activity.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(7, 3))
    controls = ttk.Frame(frame)
    controls.grid(row=15, column=0, columnspan=3, sticky="ew")
    button = ttk.Button(
        controls, text="Find vanity key", command=start_search, state="disabled",
    )
    button.pack(side="left")
    cancel_button = ttk.Button(controls, text="Cancel", command=cancel_search, state="disabled")
    cancel_button.pack(side="left", padx=7)
    ttk.Button(controls, text="Open results", command=open_results).pack(side="left")
    ttk.Button(controls, text="Rare keys…", command=open_rare_browser).pack(side="left", padx=7)
    save_button = ttk.Button(controls, text="Save a copy…", command=save_copy, state="disabled")
    save_button.pack(side="right")
    copy_private_button = ttk.Button(controls, text="Copy private", command=lambda: copy_value(True), state="disabled")
    copy_private_button.pack(side="right", padx=5)
    copy_public_button = ttk.Button(controls, text="Copy public", command=lambda: copy_value(False), state="disabled")
    copy_public_button.pack(side="right")
    reveal_button = ttk.Button(controls, text="Reveal private key", command=toggle_private, state="disabled")
    reveal_button.pack(side="right", padx=5)

    def close_window() -> None:
        state["closing"] = True
        shutdown_event.set()
        for history_cancel in tuple(history_load_cancels):
            history_cancel.set()
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.set()
        with process_lock:
            process = active_process[0]
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            # Reaping can take up to the timeout, so keep it off Tk's thread.
            # This thread is intentionally non-daemon: the CUDA child cannot
            # outlive a GUI that is closing.
            threading.Thread(
                target=terminate_process, args=(process,), daemon=False,
            ).start()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    refresh_estimate()
    root.after(25, poll_ui_events)
    start_gui_diagnostics(
        lambda discovered, error_text: post_ui(
            apply_gui_diagnostics, discovered, error_text,
        ),
        ruleset,
    )
    start_temperature_monitor(queue_temperature_snapshot, shutdown_event)
    return _run_gui_mainloop(root.mainloop, close_window)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--prefix", default="", help="hexadecimal public-key prefix")
    parser.add_argument("--suffix", default="", help="hexadecimal public-key suffix")
    parser.add_argument("--contains", default="", help="hexadecimal substring anywhere in the public key")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, help="write result JSON with mode 0600")
    parser.add_argument("--watch-output", type=Path, default=DEFAULT_WATCH_PATH,
                        help="append incidental interesting keys here")
    parser.add_argument("--rare-rules", type=Path,
                        help="load an alternate rare-key rules JSON file")
    parser.add_argument("--collect-rare", action="store_true",
                        help="continuously collect rare keys with CUDA until interrupted")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--cuda-engine", choices=("optimized", "baseline", "incremental"),
                        default="optimized", help="CUDA implementation (default: optimized)")
    parser.add_argument("--show-private", action="store_true",
                        help="print the private key to the terminal")
    parser.add_argument("--force", action="store_true",
                        help="allow --output to replace an existing regular file")
    parser.add_argument("--gui", action="store_true", help="open the small desktop GUI")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diagnostics", action="store_true",
                        help="print local readiness information as JSON")
    args = parser.parse_args()
    try:
        ruleset = (load_ruleset(args.rare_rules)
                   if args.rare_rules is not None else DEFAULT_RARE_RULESET)
    except RuleConfigError as error:
        parser.error(str(error))
    if args.self_test:
        return 0 if self_test() else 1
    if args.diagnostics:
        try:
            report = diagnostics(ruleset)
        except KeyboardInterrupt:
            print("\nDiagnostics cancelled.")
            return 130
        print(json.dumps(report, indent=2))
        return 0
    if args.gui:
        return run_gui(
            ruleset, persist_rule_selection=args.rare_rules is None,
        )
    try:
        prefix, suffix, contains = (valid_pattern(getattr(args, name), name) for name in ("prefix", "suffix", "contains"))
        if args.collect_rare and any((prefix, suffix, contains)):
            raise ValueError("--collect-rare cannot be combined with a vanity pattern")
        if not args.collect_rare and not any((prefix, suffix, contains)):
            raise ValueError("Specify --prefix, --suffix, or --contains (or use --gui)")
        if not args.collect_rare:
            validate_constraints(prefix, suffix, contains)
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        if args.device < 0:
            raise ValueError("--device must be zero or greater")
        if args.collect_rare and args.backend == "cpu":
            raise ValueError("--collect-rare requires the CUDA backend")
        if args.collect_rare and (args.output or args.show_private or args.force):
            raise ValueError("--output, --show-private, and --force do not apply to --collect-rare")
        if args.output and (args.output.exists() or args.output.is_symlink()) and not args.force:
            raise ValueError(f"output already exists; choose another path or add --force: {args.output}")
    except ValueError as error:
        parser.error(str(error))
    try:
        cuda_is_ready = (cuda_available(args.device, args.cuda_engine)
                         if args.collect_rare or args.backend != "cpu" else False)
    except KeyboardInterrupt:
        print("\nSearch cancelled.")
        return 130
    use_cuda = (args.collect_rare or args.backend == "cuda"
                or (args.backend == "auto" and cuda_is_ready))
    if use_cuda:
        if not cuda_is_ready:
            try:
                # Failed probes are intentionally not cached, so an explicit
                # CUDA request gets one immediate recovery attempt.
                probe = cuda_probe(args.device, args.cuda_engine)
            except KeyboardInterrupt:
                print("\nSearch cancelled.")
                return 130
            if probe.get("ready") is True:
                cuda_is_ready = True
            else:
                detail = probe.get("error", "selected CUDA engine did not become ready")
                print(f"CUDA unavailable: {detail}", file=sys.stderr)
                return 2
        if args.collect_rare:
            session = {"count": 0, "best_bits": 0.0, "best": "none yet"}

            def collector_progress(attempts: int, elapsed: float) -> None:
                rate = attempts / max(elapsed, .001)
                print(
                    f"\rCollecting: {attempts:,} keys | {rate:,.0f} keys/s | "
                    f"{format_duration(elapsed)} | saved this session: {session['count']} | "
                    f"best: {session['best']}", end="", flush=True,
                )

            def collector_watch(_rule: int, reason: str, public_key: str, path: Path) -> None:
                session["count"] = int(session["count"]) + 1
                analyzed = interesting_matches(public_key, ruleset)
                rarity_bits = analyzed[0].rarity_bits if analyzed else 0.0
                if rarity_bits > float(session["best_bits"]):
                    session["best_bits"] = rarity_bits
                    session["best"] = f"{reason} ({rarity_bits:.1f} bits)"
                print(f"\nSaved {reason}; session total {session['count']} -> {path}")

            print("Using CUDA continuous rare-key collector. Press Ctrl+C to stop.")
            try:
                search_cuda(
                    "", "", "", update=collector_progress, watch_path=args.watch_output,
                    watch_update=collector_watch, device=args.device,
                    engine=args.cuda_engine, collect_only=True, ruleset=ruleset,
                )
            except KeyboardInterrupt:
                print(
                    f"\nCollector stopped. Saved {session['count']} verified rare keys "
                    "during this session."
                )
                return 0
            except RuntimeError as error:
                print(f"\nCollector failed: {error}", file=sys.stderr)
                return 2
        print("Using CUDA backend with independent CPU result verification.")
        try:
            result = search_cuda(prefix, suffix, contains, watch_path=args.watch_output,
                                 device=args.device, engine=args.cuda_engine,
                                 ruleset=ruleset)
        except KeyboardInterrupt:
            print("\nSearch cancelled.")
            return 130
        except SearchCancelled:
            print("\nSearch cancelled.")
            return 0
        except RuntimeError as error:
            print(f"\nSearch failed: {error}", file=sys.stderr)
            return 2
    else:
        print(f"Using verified CPU backend with {args.workers} workers.")
        try:
            result = search(
                prefix, suffix, contains, args.workers,
                lambda n, e: print(
                    f"\r{n:,} keys | {n / max(e, .001):,.0f} keys/s",
                    end="", flush=True,
                ),
            )
        except KeyboardInterrupt:
            print("\nSearch cancelled.")
            return 130
        except SearchCancelled:
            print("\nSearch cancelled.")
            return 0
        except (RuntimeError, SearchWorkerError) as error:
            print(f"\nSearch failed: {error}", file=sys.stderr)
            return 2
    backend_label = result.backend.upper() + (f"/{result.engine}" if result.engine else "")
    print(f"\nFound {result.public_key} after ≤{result.attempts:,} attempts ({result.elapsed_seconds:.3f}s, {backend_label}).")
    output_path = args.output or default_result_path(result.public_key)
    try:
        save_result(result, output_path, overwrite=args.force)
    except FileExistsError as error:
        print(f"Refusing to overwrite a private-key file: {error}", file=sys.stderr)
        print("Choose another --output path or repeat with --force.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Could not save identity: {error}", file=sys.stderr)
        return 2
    print(f"Saved identity: {output_path}")
    if args.show_private:
        print(f"Private key: {result.private_key}")
    if result.backend == "cuda":
        print(f"Interesting matches: {args.watch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
