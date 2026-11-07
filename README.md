# MeshCore vanity key generator

A local Ed25519 identity generator for MeshCore. Search for a public-key prefix,
suffix, or substring from the command line or desktop GUI. NVIDIA CUDA
accelerates the search. A CPU backend is also available.

The current release is **v1.11.1**. The program saves the 64-byte expanded
private key that MeshCore expects, encoded as 128 hexadecimal characters.
It does not save a 32-byte seed in its place.

If you generated keys with optimized CUDA in **v1.5.1 or earlier**, run the
[saved-key audit](docs/key-audit.md). Some identities from the same old batch
can have related private scalars. Rotate every identity named in an
`optimized_scalar_relationship` finding before sensitive use.

## Install on Ubuntu

Clone this repository and run the installer:

```sh
git clone https://github.com/kesh33w/meshcore-cuda-vanity-keygen.git
cd meshcore-cuda-vanity-keygen
./install.sh
```

The installer uses `sudo` for missing Ubuntu packages, builds the CUDA engine,
and installs the commands under `~/.local/bin`. It also adds the desktop
launcher. It does not replace the NVIDIA display driver.

You need a working NVIDIA driver and CUDA toolkit 11.8 or newer.
`nvidia-smi` must list the GPU. If Ubuntu's toolkit is too old, install a
current NVIDIA toolkit and rerun `./install.sh --skip-packages`.
The installer requires a CUDA build even though a source checkout can run
CPU searches.

Start the GUI from the application menu or the terminal:

```sh
meshcore-vanity-keygen --gui
```

Check the installed program and GPU readiness without starting a search:

```sh
meshcore-vanity-keygen --diagnostics
```

Diagnostics run a bounded kernel check on each visible GPU. They contain no
key material.

## Search for a key

```sh
meshcore-vanity-keygen --prefix cafe
meshcore-vanity-keygen --backend cuda --device 1 --suffix 1337
meshcore-vanity-keygen --backend cpu --prefix cafe
```

Use `--help` for the complete options. `--backend auto` selects CUDA on a
ready NVIDIA system. Impossible patterns, including prefixes beginning with
the MeshCore-rejected bytes `00` or `ff`, are rejected before the search.

The GUI's Cancel button, closing the window, and Ctrl+C stop a running search.
Time estimates are averages. They do not set a maximum wait.

For continuous collection of built-in rare patterns:

```sh
meshcore-vanity-keygen --collect-rare
```

This mode needs CUDA. Use **Rare keys to keep** in the GUI to choose categories
and a minimum match length. See [rare-key collection](docs/rare-keys.md) for
the rule settings, saved history, and record format.

## Saved identities

A source checkout saves results in its own `results/` directory. The installed
application uses `~/.local/share/meshcore-vanity-keygen/results/`. The GUI can
select another folder.

Private keys are masked in the GUI and omitted from terminal output by default.
Use the reveal and copy controls, or `--show-private`, only when you intend to
expose a key. The program writes identity files atomically with owner-only
`0600` permissions. Existing files require an explicit `--force` override
with the CLI output path.

Treat identity JSON and rare-key JSONL files as secrets. Keep them out of Git
and back them up only to trusted storage. To use an identity, import its private
key through the MeshCore client's key-import or `prv.key` setting, then reboot
the node.

Audit saved files locally:

```sh
meshcore-key-audit ~/.local/share/meshcore-vanity-keygen/results
```

The auditor leaves the files unchanged and reports opaque file and line IDs
without key contents. See [audit limits and result handling](docs/key-audit.md).

## Develop and test

The Python code uses the standard library and system `libsodium` through
`ctypes`. Tk provides the desktop GUI. No third-party Python packages are
required.

For a source checkout on Ubuntu:

```sh
sudo apt update
sudo apt install build-essential libsodium23 python3 python3-tk nvidia-cuda-toolkit
make
python3 meshcore_vanity.py --self-test
make check
make test
```

The toolkit still needs to meet the 11.8 minimum. Run `make test-gpu`
separately on a CUDA machine. See the [test guide](docs/testing.md) for coverage
and hardware requirements.

The [CUDA reference](docs/cuda.md) describes architecture selection, engine
internals, VRAM use, and the recorded RTX 4070 Ti measurements. Those
measurements describe that test machine and workload.

## Security and compatibility

Every retained GPU result is checked on the CPU for its public/private-key
relationship, requested pattern, signature, and MeshCore key exchange.
The test suite includes MeshCore firmware test vectors. A v1.0.0 identity was
also tested on physical RAK4631 hardware. See the
[hardware record](HARDWARE_VALIDATION.md) for that test's scope.

Key generation runs locally. Generated keys are not sent over the network.
The project has not had an independent professional cryptographic audit.
See [SECURITY.md](SECURITY.md) for the legacy-key issue, reporting instructions,
and assurance limits.

## Remove the application

```sh
./uninstall.sh
```

The uninstaller keeps generated keys in
`~/.local/share/meshcore-vanity-keygen/`. Remove that directory yourself only
when you no longer need those identities.

## License and history

The vendored CUDA Ed25519 implementation is GPL-3.0. Read
[LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for
attribution and license terms. Release changes are in
[CHANGELOG.md](CHANGELOG.md).
