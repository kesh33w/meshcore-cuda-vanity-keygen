#!/usr/bin/env python3
"""Local, MeshCore-compatible Ed25519 vanity key generator."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import hashlib
import json
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
DEFAULT_WATCH_PATH = APP_DIR / "results" / "rare-keys.jsonl"
DEFAULT_RESULTS_DIR = APP_DIR / "results"


class Sodium:
    def __init__(self) -> None:
        library = ctypes.util.find_library("sodium") or "libsodium.so.23"
        self.lib = ctypes.CDLL(library)
        self.lib.sodium_init.restype = ctypes.c_int
        self.lib.crypto_scalarmult_ed25519_base_noclamp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.crypto_scalarmult_ed25519_base_noclamp.restype = ctypes.c_int
        if self.lib.sodium_init() < 0:
            raise RuntimeError("libsodium initialization failed")

    def derive_public(self, scalar: bytes) -> bytes:
        public = (ctypes.c_ubyte * 32)()
        secret = (ctypes.c_ubyte * 32).from_buffer_copy(scalar)
        if self.lib.crypto_scalarmult_ed25519_base_noclamp(public, secret) != 0:
            raise RuntimeError("libsodium rejected the Ed25519 scalar")
        return bytes(public)


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


@dataclass(frozen=True)
class Result:
    public_key: str
    private_key: str
    attempts: int
    elapsed_seconds: float
    match: str
    backend: str = "cpu"


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


def search_cuda(prefix: str, suffix: str, contains: str,
                update: Optional[Callable[[int, float], None]] = None,
                watch_path: Optional[Path] = None,
                cancel: Optional[threading.Event] = None,
                watch_update: Optional[Callable[[int, str, str, Path], None]] = None,
                device: int = 0,
                process_update: Optional[Callable[[Optional[subprocess.Popen[str]]], None]] = None) -> Result:
    executable = cuda_executable()
    if not executable.is_file():
        raise RuntimeError("CUDA engine is not built; run 'make'")
    command = [str(executable), "--device", str(device)]
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
                    appended = append_interesting(watch_path, rule_number, public_hex, private_hex)
                    if appended and watch_update:
                        watch_update(rule_number, WATCH_REASONS[rule_number], public_hex, watch_path)
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
    if return_code:
        detail = errors[-1] if errors else f"status {return_code}"
        raise RuntimeError(f"CUDA engine failed: {detail}")
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        public_hex = payload["public_key"]
        private_hex = payload["private_key"]
        seed = bytes.fromhex(payload["seed"])
        private = bytes.fromhex(private_hex)
    except (KeyError, ValueError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("CUDA engine returned an invalid result") from error
    digest = bytearray(hashlib.sha512(seed).digest())
    digest[0] &= 248
    digest[31] &= 63
    digest[31] |= 64
    if (bytes(digest) != private or SODIUM.derive_public(private[:32]).hex() != public_hex
            or not matches(public_hex, prefix, suffix, contains)):
        raise RuntimeError("CUDA result failed independent CPU verification")
    needle = ", ".join(part for part in (prefix and f"prefix {prefix}", suffix and f"suffix {suffix}", contains and f"contains {contains}") if part)
    return Result(public_hex, private_hex, int(payload["attempts"]),
                  float(payload["elapsed_seconds"]), needle, "cuda")


def secure_open(path: Path, flags: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"refusing to write non-regular file: {path}")
    return descriptor


def save_result(result: Result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = secure_open(path, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
        file.write("\n")


def initialize_watch_file(path: Path) -> None:
    """Create the watch file up front so an empty file clearly means zero finds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.close(descriptor)


WATCH_REASONS = (
    "bookend-10", "mirror-10",
    *(f"prefix-{word}" for word in ("cafecafe00", "beefbeef00", "deadbeef00", "facebabe00", "babecafe00", "f00df00d00", "1337133713", "fadefade00")),
    *(f"suffix-{word}" for word in ("cafecafe00", "beefbeef00", "deadbeef00", "facebabe00", "babecafe00", "f00df00d00", "1337133713", "fadefade00")),
)


def interesting_rule(public_hex: str) -> int:
    """Classify a rare public key independently of the CUDA implementation."""
    if len(public_hex) != 64 or not re.fullmatch(r"[0-9a-f]{64}", public_hex):
        return -1
    if public_hex[:10] == public_hex[-10:]:
        return 0
    if public_hex[:10] == public_hex[-10:][::-1]:
        return 1
    words = (
        "cafecafe00", "beefbeef00", "deadbeef00", "facebabe00",
        "babecafe00", "f00df00d00", "1337133713", "fadefade00",
    )
    for index, word in enumerate(words):
        if public_hex.startswith(word):
            return 2 + index
        if public_hex.endswith(word):
            return 10 + index
    return -1


def append_interesting(path: Path, rule: int, public_hex: str, private_hex: str) -> bool:
    private = bytes.fromhex(private_hex)
    public = bytes.fromhex(public_hex)
    if (rule < 0 or rule >= len(WATCH_REASONS) or interesting_rule(public_hex) != rule
            or len(private) != 64 or len(public) != 32
            or public[0] in (0, 255)
            or SODIUM.derive_public(private[:32]).hex() != public_hex):
        raise RuntimeError("An incidental CUDA result failed CPU verification")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = secure_open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
    record = {
        "found_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reason": WATCH_REASONS[rule],
        "public_key": public_hex,
        "private_key": private_hex,
        "backend": "cuda",
    }
    with os.fdopen(descriptor, "a+", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.seek(0)
        for line in file:
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if existing.get("reason") == WATCH_REASONS[rule] or existing.get("public_key") == public_hex:
                return False
        file.seek(0, os.SEEK_END)
        file.write(json.dumps(record, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return True


TEST_PRIVATE = bytes.fromhex(
    "7065e18fd9fabb70c1ed90dca19907de698c88b709ea146eafd93d9b830c7b60"
    "c4681193c7b9bc39945ba8064104bb618f8fd7a84a0af6f57033d6e8ddcd6471")
TEST_PUBLIC = "1ec77175b0918ed206f9ae04ec136d6d5d4315bb26305427f645b492e9350c10"


def self_test() -> bool:
    derived = SODIUM.derive_public(TEST_PRIVATE[:32]).hex()
    print("PASS" if derived == TEST_PUBLIC else "FAIL", "MeshCore firmware key vector")
    return derived == TEST_PUBLIC


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError:
        print("Tk is unavailable. Install python3-tk or use the command line.", file=sys.stderr)
        return 2

    root = tk.Tk()
    root.title("MeshCore Vanity Key Generator")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=16)
    frame.grid()
    fields: dict[str, tk.StringVar] = {name: tk.StringVar() for name in ("prefix", "suffix", "contains")}
    for row, (name, value) in enumerate(fields.items()):
        ttk.Label(frame, text=f"{name.title()} (hex)").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(frame, width=42, textvariable=value).grid(row=row, column=1, pady=3)
    workers = tk.IntVar(value=max(1, (os.cpu_count() or 2) - 1))
    ttk.Label(frame, text="CPU workers").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Spinbox(frame, from_=1, to=max(1, os.cpu_count() or 1), width=8, textvariable=workers).grid(row=3, column=1, sticky="w")
    gpu_count = cuda_device_count()
    device = tk.IntVar(value=0)
    ttk.Label(frame, text="CUDA device").grid(row=4, column=0, sticky="w", pady=3)
    device_box = ttk.Combobox(frame, width=8, state="readonly", textvariable=device,
                              values=tuple(range(gpu_count)) if gpu_count else ("None",))
    device_box.grid(row=4, column=1, sticky="w")
    if not gpu_count:
        device_box.current(0)
    status = tk.StringVar(value=f"CUDA devices visible: {gpu_count} ({'CUDA' if cuda_available() else 'CPU'} backend)")
    ttk.Label(frame, textvariable=status).grid(row=5, column=0, columnspan=2, sticky="w", pady=(9, 3))
    incidental = tk.StringVar(value="Rare incidental keys found: 0")
    ttk.Label(frame, textvariable=incidental).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 3))
    output = tk.Text(frame, width=80, height=8, state="disabled", wrap="word")
    output.grid(row=7, column=0, columnspan=2, pady=5)
    state: dict[str, object] = {
        "result": None, "cancel": threading.Event(), "searching": False,
        "incidental": 0, "process": None, "closing": False,
    }

    def show(text: str) -> None:
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("1.0", text)
        output.configure(state="disabled")

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
        state["incidental"] = 0
        incidental.set("Rare incidental keys found: 0")
        worker_count = workers.get()
        selected_device = device.get() if gpu_count else 0
        using_cuda = cuda_available()
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
                root.after(0, status.set, f"Searching: {attempts:,} keys, {attempts / max(elapsed, .001):,.0f} keys/s")

        def watch_progress(rule: int, reason: str, public_key: str, path: Path) -> None:
            def display() -> None:
                state["incidental"] = int(state["incidental"]) + 1
                count = state["incidental"]
                incidental.set(
                    f"Rare incidental keys found: {count} | latest: {reason} | "
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
                                         watch_update=watch_progress, device=selected_device,
                                         process_update=process_progress)
                else:
                    result = search(**values, workers=worker_count, update=progress,
                                    cancel=cancel_event)
                state["result"] = result
                if not state["closing"]:
                    root.after(0, complete, result)
            except SearchCancelled:
                if not state["closing"]:
                    root.after(0, stopped)
            except Exception as error:
                error_text = str(error)
                if not state["closing"]:
                    root.after(0, failed, error_text)

        threading.Thread(target=job, daemon=True).start()

    def complete(result: Result) -> None:
        state["searching"] = False
        activity.stop()
        status.set(f"Found with {result.backend.upper()} after ≤{result.attempts:,} attempts in {result.elapsed_seconds:.3f}s")
        show(f"PUBLIC KEY (64 hex characters):\n{result.public_key}\n\nPRIVATE KEY — keep secret (128 hex characters):\n{result.private_key}\n\nImport the private key as MeshCore prv.key, then reboot.\n\nRare incidental matches are saved to results/rare-keys.jsonl.")
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

    def save() -> None:
        result = state["result"]
        if not result:
            messagebox.showinfo("Nothing to save", "Find a key first.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".json", initialfile="meshcore-identity.json")
        if filename:
            save_result(result, Path(filename))
            messagebox.showinfo("Saved", "Saved with owner-only permissions where supported.")

    activity = ttk.Progressbar(frame, mode="indeterminate", length=620)
    activity.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(7, 3))
    controls = ttk.Frame(frame)
    controls.grid(row=9, column=0, columnspan=2, sticky="ew")
    button = ttk.Button(controls, text="Find vanity key", command=start_search)
    button.pack(side="left")
    cancel_button = ttk.Button(controls, text="Cancel", command=cancel_search, state="disabled")
    cancel_button.pack(side="left", padx=7)
    ttk.Button(controls, text="Save result", command=save).pack(side="right")
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
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="hexadecimal public-key prefix")
    parser.add_argument("--suffix", default="", help="hexadecimal public-key suffix")
    parser.add_argument("--contains", default="", help="hexadecimal substring anywhere in the public key")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, help="write result JSON with mode 0600")
    parser.add_argument("--watch-output", type=Path, default=DEFAULT_WATCH_PATH,
                        help="append incidental interesting keys here")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--show-private", action="store_true",
                        help="print the private key to the terminal")
    parser.add_argument("--gui", action="store_true", help="open the small desktop GUI")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
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
    except ValueError as error:
        parser.error(str(error))
    use_cuda = args.backend == "cuda" or (args.backend == "auto" and cuda_available())
    if use_cuda:
        if not cuda_available():
            print("CUDA device or built engine unavailable; run 'make' and check nvidia-smi.", file=sys.stderr)
            return 2
        print("Using CUDA backend with independent CPU result verification.")
        result = search_cuda(prefix, suffix, contains, watch_path=args.watch_output,
                             device=args.device)
    else:
        print(f"Using verified CPU backend with {args.workers} workers.")
        result = search(prefix, suffix, contains, args.workers, lambda n, e: print(f"\r{n:,} keys | {n / max(e, .001):,.0f} keys/s", end="", flush=True))
    print(f"\nFound {result.public_key} after ≤{result.attempts:,} attempts ({result.elapsed_seconds:.3f}s, {result.backend.upper()}).")
    output_path = args.output or (DEFAULT_RESULTS_DIR / f"meshcore-identity-{result.public_key[:8]}.json")
    save_result(result, output_path)
    print(f"Saved identity: {output_path}")
    if args.show_private:
        print(f"Private key: {result.private_key}")
    if result.backend == "cuda":
        print(f"Interesting matches: {args.watch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
