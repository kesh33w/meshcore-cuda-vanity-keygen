# Changelog

All notable changes to this project are documented here.

## 1.0.0 — 2026-09-08

- Add a one-command Ubuntu installer, desktop launcher, and uninstaller.
- Automatically save GUI results using atomic owner-only files.
- Hide private keys by default and add separate reveal/copy controls.
- Add configurable result folders and an Open Results action.
- Add live difficulty and average-time estimates.
- Add startup GPU, CUDA, firmware-vector, and version diagnostics.
- Preserve explicit overwrite protection in both CLI and GUI workflows.
- Add CLI `--version`, `--diagnostics`, and `--force` options.
- Expand automated security, error-path, installer, and CUDA build coverage.
- Validate a generated vanity identity on physical RAK4631 MeshCore firmware.

## 0.9.0 — 2026-09-08

- Add the optimized CUDA engine with incremental point addition and batched
  Ed25519 point compression.
- Add independent CPU derivation and signature verification for every retained
  GPU result.
- Add automatic collection and validation of rare incidental identities.
- Add cancellation, GPU selection, baseline comparison, and a Tk GUI.
