# Changelog

All notable changes to this project are documented here.

## 1.5.0 — 2026-09-08

- Stop collecting the `cafecafe00`, `beefbeef00`, `deadbeef00`,
  `facebabe00`, and `babecafe00` canned phrase prefixes.
- Preserve all historical records for the removed categories in the rare-key
  browser and JSONL history.
- Keep `f00df00d00`, `1337133713`, `fadefade00`, pi, repeated-prefix,
  mirror, and bookend collection active.

## 1.4.0 — 2026-09-08

- Add a native CUDA continuous collector that searches only for built-in rare
  keys and runs until cancelled without requiring a fake vanity target.
- Add a **Continuous rare collector** GUI mode with live attempts, runtime,
  throughput, session discovery count, and best-session rarity.
- Add the `--collect-rare` CLI mode with clean `Ctrl+C` shutdown and non-secret
  progress output.
- Disable incompatible vanity inputs while GUI collection is selected and
  reject conflicting collector/vanity CLI options.
- Add live-GPU coverage for collector operation, cancellation, and child-process
  cleanup.

## 1.3.1 — 2026-09-08

- Make the rare-key table row height follow the active desktop font metrics so
  records do not overlap on high-DPI or large-text desktops.
- Improve column widths, alignment, headings, and timestamp formatting.
- Automatically select the first visible record so its details are available
  immediately when the browser opens or its filter changes.

## 1.3.0 — 2026-09-08

- Preserve the strongest observed length for repeated prefixes, mirrors, pi
  prefixes, and other overlapping rare traits instead of flattening every find
  to its ten-character trigger category.
- Add versioned rare-log records with all matching traits, match length, rarity
  in bits, and estimated mean work while retaining legacy JSONL compatibility.
- Add a GUI rare-key browser with filtering, sortable rarity columns, masked
  private keys, verification before sensitive actions, and secure export.
- Keep the live rare-key count in memory during a search instead of rescanning
  the complete JSONL history after every discovery.
- Bound the browser to the newest 10,000 valid records and report malformed
  lines without allowing them to break the rest of the history.

## 1.2.1 — 2026-09-08

- Fix a GUI cancellation race that could misreport the CUDA startup banner as
  an engine failure when a search was stopped between progress updates.

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
