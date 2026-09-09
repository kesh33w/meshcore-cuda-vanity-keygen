# MeshCore Vanity Key Generator

Small, local-only generator for MeshCore-compatible Ed25519 vanity identities.
It searches a public-key prefix, suffix, or substring and saves the matching
128-hex-character private key required by MeshCore (`prv.key`).

The current release is **v1.3.0**. Generated identities have been validated
against MeshCore firmware vectors and on physical RAK4631 hardware.

## Install and run (Ubuntu)

For a normal desktop installation, clone the repository and run:

```bash
./install.sh
```

This installs missing Ubuntu dependencies, builds the CUDA engine, installs a
`meshcore-vanity-keygen` command under `~/.local/bin`, and adds **MeshCore
Vanity Key Generator** to the desktop application menu. It does not install or
replace the NVIDIA display driver.

Launch it from the application menu or run:

```bash
meshcore-vanity-keygen --gui
```

The manual development setup remains:

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
32-byte seed. CLI and GUI results are saved automatically. A source checkout
uses its local `results/` folder; an installed copy uses
`~/.local/share/meshcore-vanity-keygen/results/`. The GUI lets you select
another folder.

Private keys are hidden in the GUI and are not printed in the terminal by
default. Use the explicit reveal/copy controls or add `--show-private` when you
intend to expose one. Identity files are written atomically with owner-only
permissions (`0600`), and existing files are never silently overwritten. Use
`--force` with an explicit CLI output path to authorize replacement. Do not
share private-key files. To import one, use your MeshCore client's key import /
`prv.key` setting and then reboot the node.

To see local readiness information without starting a search:

```bash
meshcore-vanity-keygen --diagnostics
```

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

The default optimized engine starts every batch from a securely random, clamped
scalar. Each GPU thread calculates one full public point and then walks forward
with the much cheaper `scalar += 8` and `point += 8B` operations. It compresses
32 projective public points together using Montgomery's batch-inversion trick,
sharing one expensive field inversion across the entire group. The second half
of the expanded private key (the Ed25519 nonce prefix) is generated independently
from `/dev/urandom` only when a key is retained, so it is never reused between
saved identities. The older full-SHA-512/full-multiplication implementation
remains available for comparison with `--cuda-engine baseline` or from the GUI's
CUDA engine selector. The former `incremental` CLI name remains an alias for
`optimized`.

Every retained GPU key is re-derived, pattern-checked, used to create a test
signature, and signature-verified independently on the CPU before it is
displayed or saved. Requested patterns that are impossible—including keys
beginning with the MeshCore-rejected bytes `00` or `ff`—are rejected up front.
The Cancel button, window close action, and `Ctrl+C` stop the CUDA process.
The GUI displays mean-work and average-time estimates, updates them using the
observed search rate, and shows GPU/self-test readiness before a search starts.

### Measured performance

On the development RTX 4070 Ti, the optimized engine processes about 880 million
keys per second, compared with about 29 million for the retained baseline:
roughly a 30× speedup. Approximate average search times at 880M/s are:

| Hex characters | Possibilities | Average time |
| ---: | ---: | ---: |
| 7 | 268 million | 0.3 seconds |
| 8 | 4.3 billion | 4.9 seconds |
| 9 | 68.7 billion | 1.3 minutes |
| 10 | 1.1 trillion | 20.8 minutes |
| 11 | 17.6 trillion | 5.6 hours |

These are probabilistic averages, not maximums. GPU model, cooling, power
limits, and other workloads affect actual throughput.

## Automatic rare-key collection

Every CUDA search checks each generated public key both for the requested
pattern and for the built-in rare-key rules. The second check is automatic; no
extra option is required and it does not interrupt the requested search. When
an incidental rare match appears, its public/private relationship, rare rule,
and MeshCore validity are independently verified on the CPU and it is
immediately appended to `results/rare-keys.jsonl`. The GUI shows a live count,
the latest rule matched, and a public-key preview. Its **Rare keys…** browser can
filter and sort the saved history by date, match, length, rarity, or public key.
Private keys remain masked and are independently verified before reveal, copy,
or owner-only export. Incidental collection is a CUDA feature; the slower CPU
fallback only searches for the requested pattern.

The built-in rules are deliberately much harder than four-character vanity
patterns. Every individual pattern constrains ten hexadecimal characters:

- First ten characters equal the last ten (`bookend-10`).
- First ten characters equal the reverse of the last ten (`mirror-10`).
- The first ten characters are identical (`repeat-prefix-10`), such as
  `aaaaaaaaaa`. Repeated `0` and `f` prefixes are excluded because MeshCore
  rejects identities beginning with bytes `00` and `ff`.
- One of eight exact phrases appears at the beginning: `cafecafe00`, `beefbeef00`,
  `deadbeef00`, `facebabe00`, `babecafe00`, `f00df00d00`, `1337133713`, or
  `fadefade00`.
- The first ten decimal digits of pi appear at the beginning:
  `3141592653` (`prefix-pi-3141592653`).

At the measured 880M keys/s, one specific ten-character rule averages roughly
20.8 minutes. The repeated-prefix rule accepts 14 valid repeated digits, so it
averages about 89 seconds. Across the 12 rule categories—25 effective
ten-character possibilities—some incidental match is expected approximately
every 50 seconds. Random search times vary widely, and a short run may still
find none.

Each JSONL record keeps the matching pair together:

```json
{"schema_version":2,"found_at":"2026-09-08T12:34:56Z","trigger":"repeat-prefix-10","reason":"repeat-prefix-13","match_length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322","matches":[{"reason":"repeat-prefix-13","kind":"repeat-prefix","length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322"}],"public_key":"...","private_key":"...","backend":"cuda","engine":"optimized"}
```

The ten-character GPU rules remain the collection threshold, but CPU analysis
preserves stronger properties of a retained key. For example, a key beginning
with 13 identical characters is labeled `repeat-prefix-13`, and a pi prefix
continuing for 15 digits records all 15 rather than being flattened to ten.
When one key has multiple interesting traits, the strongest becomes `reason`
and every trait is retained under `matches`. Rarity is expressed as equivalent
random-search bits, making unlike patterns sortable on one scale.

The file is created with owner-only permissions (`0600`) when the search starts;
an empty file means no rare key has been found. A source checkout stores it
under local `results/`; an installed copy uses its private application-data
directory. Both remain independent of the shell's working directory. Use
`--watch-output PATH` to choose another CLI location.

Every discovery is retained, even when earlier keys matched the same category.
File locking prevents concurrent searches from corrupting or interleaving
records. Collection stops when the requested key is found or the search is
cancelled; rare keys already written remain saved. At the measured rate and
current criteria, continuous searching averages about 1,700 new records per
day. The GUI keeps its live count in memory and the browser bounds memory use to
the newest 10,000 valid records. Upgrading does not rewrite or delete legacy
records; their basic rarity metadata is inferred when displayed. Legacy suffix
records created by v1.0.x remain visible, but new suffix matches are no longer
collected. Version-2 records are larger than the original minimal records, so
storage growth depends on how many traits each key matches.

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

For v1.0.0, a CUDA-generated `c0dec0…` identity was also imported into a
RAK4631 running MeshCore v1.17.1. The device exported the exact private key,
produced a signature verified against the generated public key, and was then
restored to its original identity. A device is not required for normal use;
this was an independent physical compatibility test. See
[`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md) for the secret-free record.

## Security notes

- Everything runs locally; generated keys are not sent over the network.
- `results/`, the compiled binary, and Python caches are excluded from Git.
- Treat both identity JSON and rare-key JSONL files as secrets despite their
  restrictive file permissions. Back them up only to storage you trust.
- Search is probabilistic. A difficulty estimate is an average, not a deadline.
- The project has extensive compatibility checks but no independent professional
  cryptographic audit; see `SECURITY.md` for reporting and assurance details.

## Uninstall

```bash
./uninstall.sh
```

The uninstaller removes the application and desktop launcher but deliberately
preserves generated keys under `~/.local/share/meshcore-vanity-keygen/`.
Review and delete that directory yourself only when its private keys are no
longer needed.

## Publish your fork

`./publish_to_github.sh` installs GitHub CLI when needed, opens browser login,
checks that generated secrets are not tracked, and creates/pushes a public
repository.
