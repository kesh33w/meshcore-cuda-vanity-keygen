# MeshCore Vanity Key Generator

Small, local-only generator for MeshCore-compatible Ed25519 vanity identities.
It searches a public-key prefix, suffix, or substring and saves the matching
128-hex-character private key required by MeshCore (`prv.key`).

The current release is **v1.8.0**. Generated identities have been validated
against MeshCore firmware vectors and on physical RAK4631 hardware.

## Install and run (Ubuntu)

For a normal desktop installation, clone the repository and run:

```bash
./install.sh
```

This installs missing Ubuntu dependencies, builds the CUDA engine, installs
`meshcore-vanity-keygen` and `meshcore-key-audit` commands under
`~/.local/bin`, and adds **MeshCore Vanity Key Generator** to the desktop
application menu. It does not install or replace the NVIDIA display driver.

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

You also need a working NVIDIA driver (`nvidia-smi` should list your GPU) and
CUDA toolkit 11.8 or newer. Some older Ubuntu releases package an earlier
`nvidia-cuda-toolkit`; in that case, install a current NVIDIA CUDA toolkit and
rerun `./install.sh --skip-packages`. The build checks the compiler version and
stops with a clear message before compilation when it is too old. No third-party
Python packages are required. The program calls the system
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

Diagnostics now allocate the optimized engine's production-sized lane state and
exercise one full configured block through the real optimized scan engine on
every visible GPU, rather than treating a listed device and an existing binary
as proof that CUDA can actually run. During background discovery, the GUI also
checks the baseline engine on every visible GPU so switching engines remains
instant. Probe results contain no key material.

## GPU acceleration

Build the CUDA engine with `make`. The Makefile detects every visible GPU
compute capability, embeds native code for each one, retains PTX for the newest,
and selects an available GNU C++ compiler. Override the architecture when
cross-building:

```bash
make CUDA_ARCH=sm_89 HOST_CXX=/usr/bin/g++-13 CUDA_MAX_REGISTERS=128
```

Header dependencies and a source/configuration fingerprint prevent stale CUDA
binaries when architecture or tuning values change—even when switching back to
an earlier configuration.

On an NVIDIA system, `--backend auto` selects CUDA automatically. Use
`--backend cuda` or `--backend cpu` to force a backend, and `--device N` to
select a GPU when more than one is installed:

```bash
python3 meshcore_vanity.py --backend cuda --device 1 --suffix 1337
```

The default optimized engine begins each batch with a securely random seed and
uses domain-separated SHA-512 to derive an independent expanded Ed25519 private
key for every GPU lane. Each lane calculates one full public point and then
walks forward with the much cheaper `scalar += 8` and `point += 8B` operations.
It compresses 32 projective public points together using Montgomery's
batch-inversion trick, sharing one expensive field inversion across the entire
group. SHA-512 also
supplies a separate Ed25519 nonce prefix for each lane. At most one identity is
retained from a lane, so two saved identities never come from the same short
`+8` walk. The older full-SHA-512/full-multiplication implementation
remains available for comparison with `--cuda-engine baseline` or from the GUI's
CUDA engine selector. The former `incremental` CLI name remains an alias for
`optimized`.

Every retained GPU key is independently re-derived and pattern-checked on the
CPU. It must also create a valid test signature and produce matching, nonzero
shared secrets through MeshCore's Ed25519 key-exchange path before it is
displayed or saved. Requested patterns that are impossible—including keys
beginning with the MeshCore-rejected bytes `00` or `ff`—are rejected up front.
The Cancel button, window close action, and `Ctrl+C` stop the CUDA process.
The GUI displays mean-work and average-time estimates, updates them using the
observed search rate, and shows GPU/self-test readiness before a search starts.
It also shows the CPU package and selected GPU temperatures in degrees Celsius,
both while idle and during a search. Sensor readings are best-effort monitoring,
not thermal protection or fan/power control; unavailable readings appear as
`—`. CPU sensors are read directly from Linux sysfs, while one bounded
`nvidia-smi` query covers all NVIDIA GPUs. GPU discovery, real-kernel readiness
checks, temperature sampling, history loading, and key searching all run
outside Tk's event thread, so the window remains responsive.

### Measured performance

On the development RTX 4070 Ti, the secured optimized engine processes about
866–870 million keys per second, compared with about 29 million for the
retained baseline:
roughly a 30× speedup. Approximate average search times at 870M/s are:

| Hex characters | Possibilities | Average time |
| ---: | ---: | ---: |
| 7 | 268 million | 0.3 seconds |
| 8 | 4.3 billion | 4.9 seconds |
| 9 | 68.7 billion | 1.3 minutes |
| 10 | 1.1 trillion | 21.1 minutes |
| 11 | 17.6 trillion | 5.6 hours |

These are probabilistic averages, not maximums. GPU model, cooling, power
limits, and other workloads affect actual throughput. Controlled trials of
alternate register limits, batch sizes, launch sizes, comparison packing,
larger per-lane walks, and fast-math flags found no repeatable improvement over
the shipped 128-register, 128-thread, 16-blocks-per-SM, 4096-attempt,
32-point-batch configuration.

## Continuous rare-key collector

Select **Continuous rare collector** in the GUI and click **Start rare
collector** to collect built-in rare keys indefinitely without inventing an
astronomically difficult vanity target. Pattern fields are disabled in this
mode. Use **Rare keys to keep → Choose…** to select the categories collected.
The live display reports candidates tested, runtime, keys per second,
discoveries during the current session, and the best session rarity. Press
**Cancel** whenever you want to stop; every completed discovery has already
been verified and saved.

The equivalent command-line mode is:

```bash
meshcore-vanity-keygen --collect-rare
```

Use `Ctrl+C` for a clean stop. `--watch-output PATH`, `--device N`, and
`--cuda-engine optimized|baseline` remain available. Collector mode requires
CUDA and cannot be combined with a vanity prefix, suffix, or substring.
Progress statistics and rare-match names appear in the terminal, but key
material does not.

## Automatic rare-key collection

Every CUDA search checks each generated public key both for the requested
pattern and for the selected rare-key rules. The second check is automatic; no
extra option is required and it does not interrupt the requested search. In the
GUI, **Rare keys to keep → Choose…** opens a checkbox for every configured
category. At least one category must remain selected. The selection is frozen
when a search starts and applies to new incidental discoveries as well as the
continuous collector. It does not delete or hide anything already saved.

Built-in GUI choices are remembered in the non-secret preferences file
`~/.config/meshcore-vanity-keygen/settings.json`. If the rule configuration
changes or that file is unreadable, the app safely restores the configured
defaults. Choices made while using a custom `--rare-rules` file last only for
that GUI session, so they cannot silently rewrite the custom policy. When
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
- The exact phrase `1337133713` appears at the beginning.
- The first ten decimal digits of pi appear at the beginning:
  `3141592653` (`prefix-pi-3141592653`).

These defaults live in [`rare_rules.json`](rare_rules.json), which is the single
source used by both Python verification and the generated fast CUDA classifier.
To experiment without modifying the installed defaults, copy that file, edit
the copy, and launch with `--rare-rules PATH` (this also works with `--gui`).
Supported rule kinds are `bookend`, `mirror`, `repeat-prefix`,
`literal-prefix`, and `sequence-prefix`. Rule order is significant: the first
matching enabled rule supplies the numeric CUDA trigger, while CPU analysis
still records every matching trait.

Custom files are strictly validated before GPU work starts. They are limited to
32 rules and 256 KiB, must use canonical lowercase hexadecimal values and valid
MeshCore prefixes, and cannot configure an individual or combined hit rate high
enough to overwhelm incidental-result handling. Each search freezes one parsed
ruleset; its semantic ID and SHA-256 fingerprint are written into every new
rare-key record. The GUI chooser revalidates every selected subset with the same
safety limits, including rules that were initially disabled in a custom file.

With all five defaults selected, at the measured 870M keys/s, one specific
ten-character rule averages roughly
21.1 minutes. The repeated-prefix rule accepts 14 valid repeated digits, so it
averages about 90 seconds. Across the 5 rule categories—18 effective
ten-character possibilities—some incidental match is expected approximately
every 70 seconds. Random search times vary widely, and a short run may still
find none.

Each JSONL record keeps the matching pair together:

```json
{"schema_version":4,"found_at":"2026-09-08T12:34:56Z","trigger":"repeat-prefix-10","reason":"repeat-prefix-13","match_length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322","matches":[{"reason":"repeat-prefix-13","kind":"repeat-prefix","length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322"}],"public_key":"...","private_key":"...","backend":"cuda","engine":"optimized","ruleset_id":"meshcore-default","ruleset_fingerprint":"...","active_rule_ids":["bookend","mirror","repeat-prefix","prefix-1337133713","pi"]}
```

The ten-character GPU rules remain the collection threshold, but CPU analysis
preserves stronger properties of a retained key. For example, a key beginning
with 13 identical characters is labeled `repeat-prefix-13`, and a pi prefix
continuing for 15 digits records all 15 rather than being flattened to ten.
When one key has multiple interesting traits, the strongest becomes `reason`
and every trait is retained under `matches`. Rarity is expressed as equivalent
random-search bits, making unlike patterns sortable on one scale. Schema-4
records also retain the active rule IDs so the history browser can show the
policy that was in force when each key was found. Older records remain fully
readable and display that policy as not recorded.

The file is created with owner-only permissions (`0600`) when the search starts;
an empty file means no rare key has been found. A source checkout stores it
under local `results/`; an installed copy uses its private application-data
directory. Both remain independent of the shell's working directory. Use
`--watch-output PATH` to choose another CLI location.

Every discovery is retained, even when earlier keys matched the same category.
File locking prevents concurrent searches from corrupting or interleaving
records. Collection stops when the requested key is found or the search is
cancelled; rare keys already written remain saved. At the measured rate and
current criteria, continuous searching averages about 1,250 new records per
day. The GUI keeps its live count in memory, reads history backward from the
file tail, and presents at most 500 rows per page from the newest 10,000 valid
records. Upgrading does not rewrite or delete legacy
records; their basic rarity metadata is inferred when displayed. Legacy suffix
records created by v1.0.x and records from later-removed phrase categories
remain visible, but those patterns are no longer collected. Version-2 and newer
records are larger than the original minimal records, so storage growth depends
on how many traits each key matches.

## Audit saved identities

Release v1.5.1 and earlier used one bounded scalar walk across all optimized
CUDA lanes in a batch. A risk exists when two or more retained identities came
from the same batch: their private scalars have a small, detectable
relationship. Disclosure of one related scalar can therefore allow recovery of
another with a bounded search, even though the saved nonce prefixes differ.
v1.5.2 isolates lanes and retains at most one identity from each bounded walk.

After upgrading, audit existing identity JSON and rare-key JSONL files locally:

```bash
meshcore-key-audit ~/.local/share/meshcore-vanity-keygen/results
```

The auditor reads files without modifying them, refuses symbolic links, and
reports only aggregate counts plus opaque file/line identifiers by default. It
never includes public or private key contents in text or JSON output. After a
finding, rerun locally with `--show-paths` to map opaque IDs to the records that
should be reviewed and replaced; note that generated filenames may themselves
contain a public-key prefix. Run `meshcore-key-audit --help` for JSON output and
threshold options. A `CLEAN` result applies to the files scanned and the
configured correlation span. The conservative default covers `2^40`
candidates—far above the shipped pre-v1.5.2 launch span. Increase
`--max-candidate-span` only when auditing keys made by a custom build whose
launch span exceeded that bound.

The vendored CUDA Ed25519 implementation is GPL-3.0; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Tests

```bash
python3 meshcore_vanity.py --self-test
make test
make test-gpu
```

The compatibility check and CPU suite derive the public key from the MeshCore
firmware's known-good private-key test vector, verify a signature made with its
expanded key, and exercise the firmware-compatible shared-secret path. The
shared-secret test includes an independent Python implementation of the
conversion and Montgomery ladder used by MeshCore at upstream commit
`d92964352441e53b93e8667b802e04f6e072b39e`. The
opt-in GPU suite exercises repeated incremental point addition, tests both CUDA
engines, compares CUDA rare-rule classification with Python, exercises the
real-kernel readiness probe, verifies generated identities, and confirms
cancellation leaves no child process behind. The CPU suite also checks strict
rule parsing, generated-header freshness, build configuration changes, GUI
rule selection and preference validation, worker failure paths, bounded history
loading, and pagination.
Temperature tests additionally cover CPU sensor selection, multi-GPU PCI
mapping, stale and malformed readings, subprocess time/output limits, and
monitor shutdown. The live GPU suite checks both versions of the key-free
readiness contract used by the current front end and native engine.

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
