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

During a CUDA search, the first match for each built-in rare-key rule is also
CPU-verified and immediately appended to `results/rare-keys.jsonl`. Every rule
has ten hex characters of effective difficulty (about 1.1 trillion possibilities):
ten-character bookends, mirrored endcaps, and ten-character phrases such
as `cafecafe00`, `deadbeef00`, and `f00df00d00` at either end. Use
`--watch-output PATH` to choose another file. At most 18 incidental records are
retained per run.

The default path is anchored to the application directory, regardless of the
directory from which the GUI is launched. The file is created when a CUDA
search starts; an empty file means no rare match has been found yet.

The vendored CUDA Ed25519 implementation is GPL-3.0; see `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

## Compatibility check

```bash
python3 meshcore_vanity.py --self-test
```

The test derives the public key from the MeshCore firmware's known-good private
key test vector.
