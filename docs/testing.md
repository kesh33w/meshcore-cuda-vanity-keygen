# Test the generator

```bash
python3 meshcore_vanity.py --self-test
make check
make test
```

Run the GPU suite separately on a machine with a working NVIDIA driver and
CUDA toolkit. It builds and executes the CUDA engine:

```sh
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
cancellation leaves no child process behind. It validates both responsive-GUI
and maximum-throughput CUDA launch profiles. The CPU suite also checks strict
rule parsing, generated-header freshness, build configuration changes, GUI
rule selection and preference validation, callback-queue recovery, worker
failure paths, bounded history loading, and pagination. Threshold tests cover
all three GUI minimums, settings migration, fixed-rule eligibility, and
non-nested longer bookends on both Python and CUDA.
Temperature tests additionally cover CPU sensor selection, multi-GPU PCI
mapping, stale and malformed readings, subprocess time/output limits, and
monitor shutdown. The live GPU suite checks both versions of the key-free
readiness contract used by the current front end and native engine.

For v1.0.0, a CUDA-generated `c0dec0...` identity was also imported into a
RAK4631 running MeshCore v1.17.1. The device exported the exact private key,
produced a signature verified against the generated public key, and was then
restored to its original identity. A device is not required for normal use;
this was an independent physical compatibility test. See
[`HARDWARE_VALIDATION.md`](../HARDWARE_VALIDATION.md) for the secret-free record.


## Publication helper

`publish_to_github.sh` installs GitHub CLI when needed, opens a login flow,
checks for tracked generated secrets, and creates and pushes a public repository.
Review its destination before running it. It is not part of installation or testing.
