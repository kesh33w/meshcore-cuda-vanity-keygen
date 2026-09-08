# MeshCore Vanity Key Generator

Small, local-only generator for MeshCore-compatible Ed25519 vanity identities.
It searches a public-key prefix, suffix, or substring and writes the matching
128-hex-character private key required by MeshCore (`prv.key`).

## Run

```bash
make
python3 meshcore_vanity.py --prefix cafe
python3 meshcore_vanity.py --gui
```

No Python packages are required on Linux systems with `libsodium`; the program
uses its audited Ed25519 base-point operation through `ctypes`.  The GUI is a
small Tk window, so install your distribution's `python3-tk` package if it is
not already installed.

The private key is the 64-byte expanded key that MeshCore expects, not a
32-byte seed.  It is saved as JSON with owner-only file permissions.  Do not
share it.  To import it, use your MeshCore client's key import / `prv.key`
setting and then reboot the node.

## GPU acceleration

Build the CUDA engine with `make`. On an NVIDIA system, `--backend auto` then
selects the GPU automatically; use `--backend cuda` or `--backend cpu` to force
a backend. The engine seeds each search batch with 256 bits from `/dev/urandom`.
Every GPU result is re-derived and pattern-checked independently on the CPU
before it is displayed or saved.

## Automatic rare-key collection

Every CUDA search performs two checks on each generated public key:

1. Does it match the prefix, suffix, or substring you requested?
2. Does it match one of the built-in rare-key rules?

The second check is automatic; no extra option is required. It does not replace
or interrupt the requested search. When an incidental rare match appears, its
public and private key are independently verified on the CPU and immediately
appended to `results/rare-keys.jsonl`. The GUI shows a live count, the latest
rule matched, and a public-key preview.

The built-in rules are deliberately much harder than four-character vanity
patterns. Each has ten hex characters of effective difficulty, or about 1.1
trillion possibilities:

- First ten characters equal the last ten (`bookend-10`).
- First ten characters equal the reverse of the last ten (`mirror-10`).
- One of eight exact phrases appears at either end: `cafecafe00`, `beefbeef00`,
  `deadbeef00`, `facebabe00`, `babecafe00`, `f00df00d00`, `1337133713`, or
  `fadefade00`.

On the RTX 4070 Ti used during development, one specific rule averages roughly
ten hours. Because all 18 rules are checked together, some incidental match is
expected approximately every 30–40 minutes. Random search times vary widely,
and a short run may find none.

Each JSONL record keeps the matching pair together:

```json
{"found_at":"2026-09-08T12:34:56Z","reason":"prefix-deadbeef00","public_key":"...","private_key":"...","backend":"cuda"}
```

The file is created with owner-only permissions (`0600`) when the search starts;
an empty file means no rare key has been found. Its default location is anchored
to the application directory regardless of where the GUI is launched. Use
`--watch-output PATH` to choose another location.

Only the first key for each of the 18 rules is retained during one run, preventing
duplicates or unbounded growth. Collection stops when the requested key is
found or the search is cancelled; rare keys already written remain saved.

The vendored CUDA Ed25519 implementation is GPL-3.0; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Compatibility check

```bash
python3 meshcore_vanity.py --self-test
```

The test derives the public key from the MeshCore firmware's known-good private
key test vector.

## Publish your fork

`./publish_to_github.sh` installs GitHub CLI when needed, opens browser login,
checks that generated secrets are not tracked, and creates/pushes a public
repository.
