#!/usr/bin/env python3
"""Local, MeshCore-compatible Ed25519 vanity key generator."""

from __future__ import annotations

import argparse
from collections import deque
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
import subprocess
import stat
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

APP_DIR = Path(__file__).resolve().parent
VERSION_PATH = APP_DIR / "VERSION"
APP_VERSION = VERSION_PATH.read_text(encoding="utf-8").strip() if VERSION_PATH.is_file() else "development"
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("MESHCORE_VANITY_RESULTS_DIR", str(APP_DIR / "results"))
).expanduser().resolve()
DEFAULT_WATCH_PATH = DEFAULT_RESULTS_DIR / "rare-keys.jsonl"
REFERENCE_CUDA_RATE = 880_000_000.0
ICON_PATH = APP_DIR / "assets" / "meshcore-vanity-keygen.png"
RARE_LOG_SCHEMA = 2
RARE_BROWSER_LIMIT = 10_000
WATCH_WORDS = (
    "cafecafe00", "beefbeef00", "deadbeef00", "facebabe00",
    "babecafe00", "f00df00d00", "1337133713", "fadefade00",
)
PI_DIGITS = "3141592653589793238462643383279502884197169399375105820974944592"


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


SODIUM = Sodium()


class SearchCancelled(RuntimeError):
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


def verify_expanded_key(private: bytes, public: bytes) -> bool:
    if (len(private) != 64 or len(public) != 32 or private[0] & 7
            or private[31] & 128 or not private[31] & 64):
        return False
    message = b"MeshCore vanity key compatibility test"
    try:
        return SODIUM.derive_public(private[:32]) == public and SODIUM.verify(
            sign_expanded(private, public, message), message, public
        )
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


@dataclass(frozen=True)
class RareMatch:
    reason: str
    kind: str
    length: int
    rarity_bits: float
    mean_attempts: str


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
    stop = threading.Event()
    found: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
    counts = [0] * workers
    start = time.monotonic()

    def worker(index: int) -> None:
        local_count = 0
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
        counts[index] = local_count

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for thread in threads:
        thread.start()
    last = 0.0
    while not stop.wait(0.1):
        if cancel and cancel.is_set():
            stop.set()
            break
        elapsed = time.monotonic() - start
        if update and elapsed - last >= 0.25:
            update(sum(counts), elapsed)
            last = elapsed
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    attempts = sum(counts)
    if found.empty():
        raise SearchCancelled("Search cancelled")
    public_hex, private_hex = found.get_nowait()
    needle = ", ".join(part for part in (prefix and f"prefix {prefix}", suffix and f"suffix {suffix}", contains and f"contains {contains}") if part) or "any valid key"
    return Result(public_hex, private_hex, attempts, elapsed, needle, "cpu")


def cuda_executable() -> Path:
    return Path(__file__).resolve().with_name("meshcore_cuda_vanity")


def cuda_available() -> bool:
    return cuda_device_count() > 0 and cuda_executable().is_file()


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


def diagnostics() -> dict[str, object]:
    devices = cuda_device_count()
    names = cuda_device_names()
    firmware_vector = SODIUM.derive_public(TEST_PRIVATE[:32]).hex() == TEST_PUBLIC
    return {
        "version": APP_VERSION,
        "firmware_vector": firmware_vector,
        "cuda_devices": devices,
        "cuda_names": names,
        "cuda_engine_built": cuda_executable().is_file(),
        "cuda_ready": devices > 0 and cuda_executable().is_file(),
        "results_directory": str(DEFAULT_RESULTS_DIR),
    }


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
                process_update: Optional[Callable[[Optional[subprocess.Popen[str]]], None]] = None) -> Result:
    executable = cuda_executable()
    if not executable.is_file():
        raise RuntimeError("CUDA engine is not built; run 'make'")
    if engine == "incremental":
        engine = "optimized"
    if engine not in ("optimized", "baseline"):
        raise ValueError("CUDA engine must be optimized or baseline")
    command = [str(executable), "--device", str(device), "--engine", engine]
    for option, value in (("--prefix", prefix), ("--suffix", suffix), ("--contains", contains)):
        if value:
            command.extend((option, value))
    if update:
        update(0, 0.0)
    watch_path = watch_path or DEFAULT_WATCH_PATH
    initialize_watch_file(watch_path)
    process: Optional[subprocess.Popen[str]] = None
    errors: list[str] = []
    stdout = ""
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=1)
        if process_update:
            process_update(process)
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
                        watch_path, rule_number, public_hex, private_hex, engine
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
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
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
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        public_hex = payload["public_key"]
        private_hex = payload["private_key"]
        public = bytes.fromhex(public_hex)
        private = bytes.fromhex(private_hex)
    except (KeyError, ValueError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("CUDA engine returned an invalid result") from error
    if (payload.get("engine") != engine or not verify_expanded_key(private, public)
            or not matches(public_hex, prefix, suffix, contains)):
        raise RuntimeError("CUDA result failed independent CPU verification")
    needle = ", ".join(part for part in (prefix and f"prefix {prefix}", suffix and f"suffix {suffix}", contains and f"contains {contains}") if part)
    return Result(public_hex, private_hex, int(payload["attempts"]),
                  float(payload["elapsed_seconds"]), needle, "cuda", engine)


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


def count_interesting(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())
    except FileNotFoundError:
        return 0


def initialize_watch_file(path: Path) -> None:
    """Create the watch file up front so an empty file clearly means zero finds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.close(descriptor)


WATCH_REASONS = (
    "bookend-10", "mirror-10", "repeat-prefix-10",
    *(f"prefix-{word}" for word in WATCH_WORDS),
    "prefix-pi-3141592653",
)


def interesting_rule(public_hex: str) -> int:
    """Classify a rare public key independently of the CUDA implementation."""
    if len(public_hex) != 64 or not re.fullmatch(r"[0-9a-f]{64}", public_hex):
        return -1
    if public_hex[:10] == public_hex[-10:]:
        return 0
    if public_hex[:10] == public_hex[-10:][::-1]:
        return 1
    if public_hex[:10] == public_hex[0] * 10:
        return 2
    for index, word in enumerate(WATCH_WORDS):
        if public_hex.startswith(word):
            return 3 + index
    if public_hex.startswith("3141592653"):
        return 11
    return -1


def make_rare_match(reason: str, kind: str, length: int,
                    alternatives: int = 1) -> RareMatch:
    attempts = (16 ** length + alternatives - 1) // alternatives
    rarity_bits = length * 4.0 - math.log2(alternatives)
    return RareMatch(reason, kind, length, round(rarity_bits, 3), str(attempts))


def interesting_matches(public_hex: str) -> list[RareMatch]:
    """Describe every recognized rare property, strongest property first."""
    if len(public_hex) != 64 or not re.fullmatch(r"[0-9a-f]{64}", public_hex):
        return []
    matches_found: list[RareMatch] = []

    # Same-order bookends are not nested as their width changes, so inspect all
    # meaningful widths and retain the strongest one that this key satisfies.
    bookend_lengths = [
        length for length in range(10, 33)
        if public_hex[:length] == public_hex[-length:]
    ]
    if bookend_lengths:
        length = max(bookend_lengths)
        matches_found.append(make_rare_match(f"bookend-{length}", "bookend", length))

    mirror_length = 0
    for index in range(32):
        if public_hex[index] != public_hex[63 - index]:
            break
        mirror_length += 1
    if mirror_length >= 10:
        matches_found.append(make_rare_match(
            f"mirror-{mirror_length}", "mirror", mirror_length
        ))

    repeat_length = 1
    while repeat_length < len(public_hex) and public_hex[repeat_length] == public_hex[0]:
        repeat_length += 1
    if repeat_length >= 10 and public_hex[0] not in "0f":
        matches_found.append(make_rare_match(
            f"repeat-prefix-{repeat_length}", "repeat-prefix", repeat_length, 14
        ))

    for word in WATCH_WORDS:
        if public_hex.startswith(word):
            matches_found.append(make_rare_match(f"prefix-{word}", "phrase-prefix", 10))

    pi_length = 0
    while (pi_length < len(public_hex) and pi_length < len(PI_DIGITS)
           and public_hex[pi_length] == PI_DIGITS[pi_length]):
        pi_length += 1
    if pi_length >= 10:
        matches_found.append(make_rare_match(
            f"prefix-pi-{PI_DIGITS[:pi_length]}", "pi-prefix", pi_length
        ))

    return sorted(matches_found, key=lambda match: match.rarity_bits, reverse=True)


def infer_legacy_rarity(record: dict[str, object]) -> tuple[int, float]:
    """Infer basic sortable metadata for records written by older releases."""
    reason = str(record.get("reason", ""))
    numbered = re.fullmatch(r"(?:bookend|mirror|repeat-prefix)-(\d+)", reason)
    if numbered:
        length = int(numbered.group(1))
        alternatives = 14 if reason.startswith("repeat-prefix-") else 1
        return length, round(length * 4.0 - math.log2(alternatives), 3)
    for marker in ("prefix-pi-", "prefix-", "suffix-"):
        if reason.startswith(marker):
            length = len(reason.removeprefix(marker))
            return length, float(length * 4)
    return 0, 0.0


def normalize_interesting_record(record: object) -> Optional[dict[str, object]]:
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
    analyzed = interesting_matches(public_hex)
    if analyzed:
        primary = analyzed[0]
        normalized["reason"] = primary.reason
        normalized["match_length"] = primary.length
        normalized["rarity_bits"] = primary.rarity_bits
        normalized["matches"] = [asdict(match) for match in analyzed]
    else:
        length, rarity_bits = infer_legacy_rarity(normalized)
        normalized.setdefault("match_length", length)
        normalized.setdefault("rarity_bits", rarity_bits)
    return normalized


def load_interesting_records(path: Path, limit: int = RARE_BROWSER_LIMIT
                             ) -> tuple[list[dict[str, object]], int]:
    """Load the most recent valid records with bounded memory use."""
    records: deque[dict[str, object]] = deque(maxlen=max(1, limit))
    skipped = 0
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    normalized = normalize_interesting_record(json.loads(line))
                except json.JSONDecodeError:
                    normalized = None
                if normalized is None:
                    skipped += 1
                else:
                    records.append(normalized)
    except FileNotFoundError:
        pass
    return list(records), skipped


def append_interesting(path: Path, rule: int, public_hex: str, private_hex: str,
                       engine: str = "optimized") -> dict[str, object]:
    private = bytes.fromhex(private_hex)
    public = bytes.fromhex(public_hex)
    if (rule < 0 or rule >= len(WATCH_REASONS) or interesting_rule(public_hex) != rule
            or len(private) != 64 or len(public) != 32
            or public[0] in (0, 255)
            or not verify_expanded_key(private, public)):
        raise RuntimeError("An incidental CUDA result failed CPU verification")
    matches_found = interesting_matches(public_hex)
    if not matches_found:
        raise RuntimeError("An incidental CUDA result failed rarity analysis")
    primary = matches_found[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
    record = {
        "schema_version": RARE_LOG_SCHEMA,
        "found_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger": WATCH_REASONS[rule],
        "reason": primary.reason,
        "match_length": primary.length,
        "rarity_bits": primary.rarity_bits,
        "mean_attempts": primary.mean_attempts,
        "matches": [asdict(match) for match in matches_found],
        "public_key": public_hex,
        "private_key": private_hex,
        "backend": "cuda",
        "engine": engine,
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


def self_test() -> bool:
    valid = verify_expanded_key(TEST_PRIVATE, bytes.fromhex(TEST_PUBLIC))
    print("PASS" if valid else "FAIL", "MeshCore firmware derivation and signature vector")
    return valid


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError:
        print("Tk is unavailable. Install python3-tk or use the command line.", file=sys.stderr)
        return 2

    root = tk.Tk(className="MeshCoreVanityKeygen")
    root.title(f"MeshCore Vanity Key Generator {APP_VERSION}")
    if ICON_PATH.is_file():
        try:
            window_icon = tk.PhotoImage(file=str(ICON_PATH))
            root.iconphoto(True, window_icon)
        except tk.TclError:
            window_icon = None
    root.minsize(780, 620)
    frame = ttk.Frame(root, padding=16)
    frame.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    details = diagnostics()
    gpu_count = int(details["cuda_devices"])
    gpu_names = list(details["cuda_names"])
    fields: dict[str, tk.StringVar] = {
        name: tk.StringVar() for name in ("prefix", "suffix", "contains")
    }
    for row, (name, value) in enumerate(fields.items()):
        ttk.Label(frame, text=f"{name.title()} (hex)").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, width=48, textvariable=value).grid(row=row, column=1, columnspan=2,
                                                            sticky="ew", pady=3)

    estimate = tk.StringVar(value="Enter a hexadecimal pattern to see estimated difficulty.")
    ttk.Label(frame, textvariable=estimate).grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 8))

    workers = tk.IntVar(value=max(1, (os.cpu_count() or 2) - 1))
    ttk.Label(frame, text="CPU workers").grid(row=4, column=0, sticky="w", pady=3)
    ttk.Spinbox(frame, from_=1, to=max(1, os.cpu_count() or 1), width=8,
                textvariable=workers).grid(row=4, column=1, sticky="w")

    device = tk.StringVar()
    device_values = tuple(
        f"{index} — {gpu_names[index] if index < len(gpu_names) else 'NVIDIA GPU'}"
        for index in range(gpu_count)
    ) or ("None detected",)
    device.set(device_values[0])
    ttk.Label(frame, text="CUDA device").grid(row=5, column=0, sticky="w", pady=3)
    ttk.Combobox(frame, state="readonly", textvariable=device,
                 values=device_values).grid(row=5, column=1, columnspan=2, sticky="ew")

    cuda_engine = tk.StringVar(value="optimized")
    ttk.Label(frame, text="CUDA engine").grid(row=6, column=0, sticky="w", pady=3)
    ttk.Combobox(frame, width=14, state="readonly", textvariable=cuda_engine,
                 values=("optimized", "baseline")).grid(row=6, column=1, sticky="w")

    output_directory = tk.StringVar(value=str(DEFAULT_RESULTS_DIR))
    ttk.Label(frame, text="Results folder").grid(row=7, column=0, sticky="w", pady=3)
    ttk.Entry(frame, textvariable=output_directory).grid(row=7, column=1, sticky="ew", pady=3)

    def choose_output_directory() -> None:
        selected = filedialog.askdirectory(initialdir=output_directory.get() or str(DEFAULT_RESULTS_DIR))
        if selected:
            output_directory.set(selected)

    ttk.Button(frame, text="Choose…", command=choose_output_directory).grid(row=7, column=2, padx=(7, 0))

    vector_label = "PASS" if details["firmware_vector"] else "FAIL"
    cuda_label = "ready" if details["cuda_ready"] else "CPU fallback"
    diagnostic_text = (
        f"Self-test: {vector_label}  •  CUDA: {cuda_label}  •  "
        f"Devices: {gpu_count}  •  Version: {APP_VERSION}"
    )
    ttk.Label(frame, text=diagnostic_text).grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 3))
    status = tk.StringVar(value="Ready")
    ttk.Label(frame, textvariable=status).grid(row=9, column=0, columnspan=3, sticky="w", pady=3)
    initial_rare_count = count_interesting(DEFAULT_WATCH_PATH)
    incidental = tk.StringVar(value=f"Saved rare incidental keys: {initial_rare_count}")
    ttk.Label(frame, textvariable=incidental).grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 3))
    output = tk.Text(frame, width=86, height=10, state="disabled", wrap="word")
    output.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=5)
    frame.rowconfigure(11, weight=1)
    state: dict[str, object] = {
        "result": None, "cancel": threading.Event(), "searching": False,
        "process": None, "closing": False, "saved_path": None,
        "reveal_private": False, "observed_rate": None,
        "rare_count": initial_rare_count,
    }

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
            values = {name: valid_pattern(var.get(), name) for name, var in fields.items()}
            if not any(values.values()):
                estimate.set("Enter a hexadecimal pattern to see estimated difficulty.")
                return
            validate_constraints(**values)
            attempts = estimate_attempts(**values)
            rate = state.get("observed_rate")
            if not isinstance(rate, (int, float)) or rate <= 0:
                rate = REFERENCE_CUDA_RATE if details["cuda_ready"] else max(1, workers.get()) * 20_000
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
        try:
            values = {name: valid_pattern(var.get(), name) for name, var in fields.items()}
            if not any(values.values()):
                raise ValueError("Enter a prefix, suffix, or substring to search for")
            validate_constraints(**values)
        except ValueError as error:
            messagebox.showerror("Invalid pattern", str(error))
            return
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.clear()
        state["searching"] = True
        state["result"] = None
        state["saved_path"] = None
        state["reveal_private"] = False
        reveal_button.configure(text="Reveal private key")
        set_result_controls(False)
        worker_count = workers.get()
        selected_device = int(device.get().split()[0]) if gpu_count else 0
        selected_engine = cuda_engine.get()
        using_cuda = cuda_available()
        try:
            result_directory = selected_results_directory()
            result_directory.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as error:
            state["searching"] = False
            messagebox.showerror("Invalid results folder", str(error))
            return
        watch_path = result_directory / "rare-keys.jsonl"
        state["rare_count"] = count_interesting(watch_path)
        incidental.set(f"Saved rare incidental keys: {state['rare_count']}")
        button.configure(state="disabled")
        cancel_button.configure(state="normal")
        activity.start(12)
        status.set("Starting CUDA search…" if using_cuda else "Starting CPU search…")
        if using_cuda:
            show("CUDA search is active. Longer patterns may take minutes or hours.\nRare incidental keys are being saved while you wait.")
        else:
            show("CPU fallback search is active. Automatic rare-key collection requires CUDA.")

        def progress(attempts: int, elapsed: float) -> None:
            if not state["closing"]:
                rate = attempts / max(elapsed, .001)
                if attempts:
                    state["observed_rate"] = rate
                root.after(0, status.set, f"Searching: {attempts:,} keys, {rate:,.0f} keys/s")
                root.after(0, refresh_estimate)

        def watch_progress(rule: int, reason: str, public_key: str, path: Path) -> None:
            def display() -> None:
                state["rare_count"] = int(state.get("rare_count", 0)) + 1
                incidental.set(
                    f"Saved rare incidental keys: {state['rare_count']} | latest: {reason} | "
                    f"{public_key[:16]}… | saved: {path}"
                )
            if not state["closing"]:
                root.after(0, display)

        def process_progress(process: Optional[subprocess.Popen[str]]) -> None:
            state["process"] = process

        def job() -> None:
            try:
                if using_cuda:
                    result = search_cuda(**values, update=progress, cancel=cancel_event,
                                         watch_path=watch_path, watch_update=watch_progress,
                                         device=selected_device,
                                         engine=selected_engine,
                                         process_update=process_progress)
                else:
                    result = search(**values, workers=worker_count, update=progress,
                                    cancel=cancel_event)
                state["result"] = result
                if not state["closing"]:
                    root.after(0, complete, result, result_directory)
            except SearchCancelled:
                if not state["closing"]:
                    root.after(0, stopped)
            except Exception as error:
                error_text = str(error)
                if not state["closing"]:
                    root.after(0, failed, error_text)

        threading.Thread(target=job, daemon=True).start()

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
        button.configure(state="normal")
        cancel_button.configure(state="disabled")

    def stopped() -> None:
        state["searching"] = False
        activity.stop()
        status.set("Search cancelled")
        show("Search cancelled. Interesting keys found before cancellation remain saved.")
        button.configure(state="normal")
        cancel_button.configure(state="disabled")

    def failed(error_text: str) -> None:
        state["searching"] = False
        activity.stop()
        button.configure(state="normal")
        cancel_button.configure(state="disabled")
        messagebox.showerror("Search failed", error_text)

    def cancel_search() -> None:
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.set()
        process = state.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            process.terminate()
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
        browser.geometry("1080x650")
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
        ttk.Entry(toolbar, textvariable=filter_value).grid(row=0, column=1, sticky="ew")
        ttk.Label(toolbar, textvariable=browser_status).grid(row=0, column=2, padx=(12, 0))

        table_frame = ttk.Frame(browser, padding=(10, 5))
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("found", "reason", "length", "rarity", "public")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        headings = {
            "found": "Found (UTC)", "reason": "Strongest match", "length": "Length",
            "rarity": "Rarity", "public": "Public key",
        }
        widths = {"found": 165, "reason": 220, "length": 65, "rarity": 90, "public": 440}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], minwidth=55, stretch=column == "public")

        detail = tk.Text(browser, height=7, state="disabled", wrap="char")
        detail.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 5))
        browser_state: dict[str, object] = {
            "records": [], "visible": {}, "sort": "found", "reverse": True,
            "private_visible": False, "skipped": 0,
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
                detail.insert(
                    "1.0",
                    f"Public key: {record['public_key']}\n"
                    f"Private key: {private}\n"
                    f"Rarity: {float(record.get('rarity_bits', 0)):.1f} bits"
                    f"  •  Match length: {record.get('match_length', 0)}"
                    f"  •  All matches: {', '.join(match_names) or record.get('reason', 'unknown')}",
                )
            detail.configure(state="disabled")

        def refresh_view(*_args: object) -> None:
            query = filter_value.get().strip().lower()
            records = browser_state["records"]
            assert isinstance(records, list)
            filtered = [
                record for record in records
                if not query or query in " ".join((
                    str(record.get("found_at", "")), str(record.get("reason", "")),
                    str(record.get("public_key", "")),
                )).lower()
            ]
            sort_column = str(browser_state["sort"])
            if sort_column == "length":
                key = lambda record: int(record.get("match_length", 0))
            elif sort_column == "rarity":
                key = lambda record: float(record.get("rarity_bits", 0))
            elif sort_column == "public":
                key = lambda record: str(record.get("public_key", ""))
            elif sort_column == "reason":
                key = lambda record: str(record.get("reason", ""))
            else:
                key = lambda record: str(record.get("found_at", ""))
            filtered.sort(key=key, reverse=bool(browser_state["reverse"]))
            tree.delete(*tree.get_children())
            visible: dict[str, dict[str, object]] = {}
            for index, record in enumerate(filtered):
                item = f"record-{index}"
                visible[item] = record
                found_at = str(record.get("found_at", "")).replace("T", " ").replace("Z", "")
                rarity = f"{float(record.get('rarity_bits', 0)):.1f} bits"
                tree.insert("", "end", iid=item, values=(
                    found_at, record.get("reason", "unknown"),
                    record.get("match_length", 0), rarity, record.get("public_key", ""),
                ))
            browser_state["visible"] = visible
            browser_state["private_visible"] = False
            skipped = int(browser_state["skipped"])
            note = f" • {skipped} malformed skipped" if skipped else ""
            limited = " • showing newest 10,000" if len(records) == RARE_BROWSER_LIMIT else ""
            browser_status.set(f"{len(filtered):,} shown / {len(records):,} loaded{limited}{note}")
            render_selected()

        def sort_by(column: str) -> None:
            if browser_state["sort"] == column:
                browser_state["reverse"] = not bool(browser_state["reverse"])
            else:
                browser_state["sort"] = column
                browser_state["reverse"] = column in ("found", "length", "rarity")
            refresh_view()

        for column in columns:
            tree.heading(column, text=headings[column], command=lambda value=column: sort_by(value))

        def reload_records() -> None:
            records, skipped = load_interesting_records(path)
            browser_state["records"] = records
            browser_state["skipped"] = skipped
            refresh_view()

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
        filter_value.trace_add("write", refresh_view)
        browser_controls = ttk.Frame(browser, padding=(10, 5, 10, 10))
        browser_controls.grid(row=3, column=0, sticky="ew")
        ttk.Button(browser_controls, text="Refresh", command=reload_records).pack(side="left")
        ttk.Button(browser_controls, text="Copy public",
                   command=lambda: copy_selected(False)).pack(side="left", padx=6)
        ttk.Button(browser_controls, text="Copy private",
                   command=lambda: copy_selected(True)).pack(side="left")
        rare_reveal = ttk.Button(browser_controls, text="Reveal private", command=toggle_rare_private)
        rare_reveal.pack(side="left", padx=6)
        ttk.Button(browser_controls, text="Export selected…",
                   command=export_selected).pack(side="left")
        ttk.Button(browser_controls, text="Close", command=browser.destroy).pack(side="right")
        reload_records()

    activity = ttk.Progressbar(frame, mode="indeterminate", length=620)
    activity.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(7, 3))
    controls = ttk.Frame(frame)
    controls.grid(row=13, column=0, columnspan=3, sticky="ew")
    button = ttk.Button(controls, text="Find vanity key", command=start_search)
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
        cancel_event = state["cancel"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.set()
        process = state.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    refresh_estimate()
    root.mainloop()
    return 0


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
    if args.self_test:
        return 0 if self_test() else 1
    if args.diagnostics:
        print(json.dumps(diagnostics(), indent=2))
        return 0
    if args.gui:
        return run_gui()
    try:
        prefix, suffix, contains = (valid_pattern(getattr(args, name), name) for name in ("prefix", "suffix", "contains"))
        if not any((prefix, suffix, contains)):
            raise ValueError("Specify --prefix, --suffix, or --contains (or use --gui)")
        validate_constraints(prefix, suffix, contains)
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        if args.device < 0:
            raise ValueError("--device must be zero or greater")
        if args.output and (args.output.exists() or args.output.is_symlink()) and not args.force:
            raise ValueError(f"output already exists; choose another path or add --force: {args.output}")
    except ValueError as error:
        parser.error(str(error))
    use_cuda = args.backend == "cuda" or (args.backend == "auto" and cuda_available())
    if use_cuda:
        if not cuda_available():
            print("CUDA device or built engine unavailable; run 'make' and check nvidia-smi.", file=sys.stderr)
            return 2
        print("Using CUDA backend with independent CPU result verification.")
        result = search_cuda(prefix, suffix, contains, watch_path=args.watch_output,
                             device=args.device, engine=args.cuda_engine)
    else:
        print(f"Using verified CPU backend with {args.workers} workers.")
        result = search(prefix, suffix, contains, args.workers, lambda n, e: print(f"\r{n:,} keys | {n / max(e, .001):,.0f} keys/s", end="", flush=True))
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
