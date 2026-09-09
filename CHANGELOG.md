# Changelog

All notable changes to this project are documented here.

## 1.2.0 — 2026-09-08

- Log every independently verified rare-key discovery, including later matches
  in a category that has already produced a result.
- Remove the persistent per-rule CUDA suppression mask and Python deduplication.
- Add a larger per-batch result buffer and fail explicitly on its practically
  unreachable overflow instead of silently dropping matches.

## 1.1.0 — 2026-09-08

- Add a rare rule for public keys whose first ten characters are identical.
- Add `3141592653`, the first ten decimal digits of pi, as a rare prefix.
- Retain the structural bookend and mirror rules and all eight phrase prefixes.
- Stop collecting phrase suffixes; existing saved records remain untouched.

## 1.0.1 — 2026-09-08

- Replace the generic terminal icon with a purpose-built mesh-and-key icon.
- Use the custom icon in the application menu, desktop shell, and GUI window.
- Install scalable SVG and 256-pixel PNG icon assets with the application.

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
