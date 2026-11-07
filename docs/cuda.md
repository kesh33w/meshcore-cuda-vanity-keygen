# CUDA engine and benchmarks

Build the CUDA engine with `make`. The Makefile detects every visible GPU
compute capability, embeds native code for each one, retains PTX for the newest,
and selects an available GNU C++ compiler. Override the architecture when
cross-building:

```bash
make CUDA_ARCH=sm_89 HOST_CXX=/usr/bin/g++-13 CUDA_MAX_REGISTERS=128
```

Header dependencies and a source/configuration fingerprint prevent stale CUDA
binaries when architecture or tuning values change. even when switching back to
an earlier configuration.

On an NVIDIA system, `--backend auto` selects CUDA automatically. Use
`--backend cuda` or `--backend cpu` to force a backend, and `--device N` to
select a GPU when more than one is installed:

```bash
python3 meshcore_vanity.py --backend cuda --device 1 --suffix 1337
```

The default optimized engine begins each batch with a securely random seed and
uses domain-separated SHA-512 to derive an independent expanded Ed25519 private
key for every GPU lane. Each lane covers the scalars `s, s + 8, s + 16, ...`
without a full scalar multiplication per key. It keeps an affine "centre"
point P and derives its neighbours P ± k·8B (k = 1...256) from a precomputed
table. Each ± pair shares one pair of denominators, and Montgomery's
batch-inversion trick shares a single field inversion across all 513 keys of
a span. A MeshCore public key is the point's y coordinate plus one sign bit
from x (bit 255, inside hex character 62), so the walk computes only y, at
about 5.6 field multiplications per key. x is computed only for keys that
could match. When no active check can read hex character 62 (a prefix search
with the bookend and mirror rare-key rules switched off), every check runs
once on y alone. Otherwise each key is checked with that bit as 0 and as 1,
and x is computed only if either could match, so no match is ever missed.
Field arithmetic uses eight 32-bit limbs with PTX carry chains (`fe32.cuh`),
about twice as fast on NVIDIA GPUs as the ref10 representation. Private keys
are reconstructed only for retained matches. `--internal-test-affine-walk`
compares every key of the walk with an independent reference scalar
multiplication, checks that each walk position is visited exactly once, and
confirms the y-only filter admits every true match for prefix and suffix
patterns that do and do not depend on bit 255.
SHA-512 also
supplies a separate Ed25519 nonce prefix for each lane. At most one identity is
retained from a lane, so two saved identities never come from the same short
`+8` walk. The older full-SHA-512/full-multiplication implementation
remains available for comparison with `--cuda-engine baseline` or from the GUI's
CUDA engine selector. The former `incremental` CLI name remains an alias for
`optimized`.

Every retained GPU key is independently re-derived and pattern-checked on the
CPU. It must also create a valid test signature and produce matching, nonzero
shared secrets through MeshCore's Ed25519 key-exchange path before it is
displayed or saved. Requested patterns that are impossible. including keys
beginning with the MeshCore-rejected bytes `00` or `ff`. are rejected up front.
The Cancel button, window close action, and `Ctrl+C` stop the CUDA process.
The GUI displays mean-work and average-time estimates, updates them using the
observed search rate, and shows GPU/self-test readiness before a search starts.
It also shows the CPU package and selected GPU temperatures in degrees Celsius,
both while idle and during a search. Sensor readings are best-effort monitoring,
not thermal protection or fan/power control; unavailable readings appear as
`. `. CPU sensors are read directly from Linux sysfs, while one bounded
`nvidia-smi` query covers all NVIDIA GPUs. GPU discovery, real-kernel readiness
checks, temperature sampling, history loading, and key searching all run
outside Tk's event thread. GUI searches also use shorter CUDA batches to keep
progress and cancellation responsive; direct command-line searches retain the
maximum-throughput launch profile.

### Measured performance

On the development RTX 4070 Ti, the v1.11.1 optimized engine's
maximum-throughput command-line profile processes about 3.45 billion keys per
second with the default rare-key rules, and about 3.99 billion for a prefix
search with the bookend and mirror rules switched off. That compares with
about 2.53 billion for v1.10, 970 million for v1.9, and 29 million for the
retained baseline. Approximate average search times at 3.45G/s are:

| Hex characters | Possibilities | Average time |
| ---: | ---: | ---: |
| 7 | 268 million | 0.1 seconds |
| 8 | 4.3 billion | 1.2 seconds |
| 9 | 68.7 billion | 20 seconds |
| 10 | 1.1 trillion | 5.3 minutes |
| 11 | 17.6 trillion | 1.4 hours |

These are probabilistic averages, not maximums. GPU model, cooling, power
limits, and other workloads affect actual throughput. Sweeps of the table size
(M = 16...500), register limits (96-255), threads per block (64-256), and blocks
per SM (2-16) settled on the shipped 128-register, 128-thread, 4-blocks-per-SM,
4096-attempt, M = 256 configuration. M = 256 uses about 1.6 GB of VRAM on this
60-SM GPU. Build with `make CUDA_AFFINE_HALF_SPAN=128` to use about 0.9 GB at
roughly 6% lower throughput. The GUI's responsive profile measured 3.31 billion
keys per second with default rules (v1.10: 2.54 billion; v1.9: 943 million) on
the same GPU.

