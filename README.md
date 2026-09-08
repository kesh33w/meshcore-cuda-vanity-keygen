# MeshCore Vanity Key Generator

Small, local-only generator for MeshCore-compatible Ed25519 vanity identities.
It searches a public-key prefix, suffix, or substring and saves the matching
128-hex-character private key required by MeshCore (`prv.key`).

## Install and run (Ubuntu)

```bash
sudo apt update
sudo apt install build-essential libsodium23 python3 python3-tk nvidia-cuda-toolkit
make
python3 meshcore_vanity.py --prefix cafe
python3 meshcore_vanity.py --gui
```

You also need a working NVIDIA driver (`nvidia-smi` should list your GPU). No
third-party Python packages are required. The program calls the system
`libsodium` Ed25519 base-point operation through `ctypes`; `python3-tk` supplies
the desktop GUI.

The private key is the 64-byte expanded key that MeshCore expects, not a
32-byte seed. CLI results are automatically saved under `results/` unless
`--output PATH` is supplied. Private keys are not printed in the terminal by
default; add `--show-private` if you explicitly want that. Files are forced to
owner-only permissions (`0600`). Do not share them. To import one, use your
MeshCore client's key import / `prv.key` setting and then reboot the node.

## GPU acceleration

Build the CUDA engine with `make`. The Makefile detects the first GPU's compute
capability and an available GNU C++ compiler. Override either when cross-building:

```bash
make CUDA_ARCH=sm_89 HOST_CXX=/usr/bin/g++-13 CUDA_MAX_REGISTERS=128
```

On an NVIDIA system, `--backend auto` selects CUDA automatically. Use
`--backend cuda` or `--backend cpu` to force a backend, and `--device N` to
select a GPU when more than one is installed:

```bash
python3 meshcore_vanity.py --backend cuda --device 1 --suffix 1337
```

The default incremental engine starts every batch from a securely random,
clamped scalar. Each GPU thread calculates one full public point and then walks
forward with the much cheaper `scalar += 8` and `point += 8B` operations. The
second half of the expanded private key (the Ed25519 nonce prefix) is generated
independently from `/dev/urandom` only when a key is retained, so it is never
reused between saved identities. The older full-SHA-512/full-multiplication
implementation remains available for comparison with `--cuda-engine baseline`
or from the GUI's CUDA engine selector.

Every retained GPU key is re-derived, pattern-checked, used to create a test
signature, and signature-verified independently on the CPU before it is
displayed or saved. Requested patterns that are impossible—including keys
beginning with the MeshCore-rejected bytes `00` or `ff`—are rejected up front.
The Cancel button, window close action, and `Ctrl+C` stop the CUDA process.

### Measured performance

On the development RTX 4070 Ti, the tuned incremental engine processes about
110 million keys per second, compared with about 29 million for the retained
baseline: roughly a 3.8× speedup. Approximate average search times at 110M/s are:

| Hex characters | Possibilities | Average time |
| ---: | ---: | ---: |
| 7 | 268 million | 2.4 seconds |
| 8 | 4.3 billion | 39 seconds |
| 9 | 68.7 billion | 10.4 minutes |
| 10 | 1.1 trillion | 2.8 hours |
| 11 | 17.6 trillion | 44 hours |

These are probabilistic averages, not maximums. GPU model, cooling, power
limits, and other workloads affect actual throughput.

## Automatic rare-key collection

Every CUDA search checks each generated public key both for the requested
pattern and for the built-in rare-key rules. The second check is automatic; no
extra option is required and it does not interrupt the requested search. When
an incidental rare match appears, its public/private relationship, rare rule,
and MeshCore validity are independently verified on the CPU and it is
immediately appended to `results/rare-keys.jsonl`. The GUI shows a live count,
the latest rule matched, and a public-key preview. Incidental collection is a
CUDA feature; the slower CPU fallback only searches for the requested pattern.

The built-in rules are deliberately much harder than four-character vanity
patterns. Each has ten hex characters of effective difficulty, or about 1.1
trillion possibilities:

- First ten characters equal the last ten (`bookend-10`).
- First ten characters equal the reverse of the last ten (`mirror-10`).
- One of eight exact phrases appears at either end: `cafecafe00`, `beefbeef00`,
  `deadbeef00`, `facebabe00`, `babecafe00`, `f00df00d00`, `1337133713`, or
  `fadefade00`.

At the measured 110M keys/s, one specific ten-character rule averages roughly
2.8 hours. Because all 18 rules are checked together, some incidental match is
expected approximately every nine minutes. Random search times vary widely,
and a short run may still find none.

Each JSONL record keeps the matching pair together:

```json
{"found_at":"2026-09-08T12:34:56Z","reason":"prefix-deadbeef00","public_key":"...","private_key":"...","backend":"cuda","engine":"incremental"}
```

The file is created with owner-only permissions (`0600`) when the search starts;
an empty file means no rare key has been found. Its default location is anchored
to the application directory regardless of where the GUI is launched. Use
`--watch-output PATH` to choose another location.

Only the first key for each of the 18 rules is retained in the output file,
including across later runs. File locking prevents two searches from appending
the same rule concurrently. Collection stops when the requested key is found or
the search is cancelled; rare keys already written remain saved.

The vendored CUDA Ed25519 implementation is GPL-3.0; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Tests

```bash
python3 meshcore_vanity.py --self-test
make test
make test-gpu
```

The compatibility check and CPU suite derive the public key from the MeshCore
firmware's known-good private-key test vector and verify a signature made with
its expanded key. The opt-in GPU suite exercises repeated incremental point
addition, tests both CUDA engines, verifies generated signatures, and confirms
cancellation leaves no child process behind.

## Security notes

- Everything runs locally; generated keys are not sent over the network.
- `results/`, the compiled binary, and Python caches are excluded from Git.
- Treat both identity JSON and rare-key JSONL files as secrets despite their
  restrictive file permissions. Back them up only to storage you trust.
- Search is probabilistic. A difficulty estimate is an average, not a deadline.

## Publish your fork

`./publish_to_github.sh` installs GitHub CLI when needed, opens browser login,
checks that generated secrets are not tracked, and creates/pushes a public
repository.
