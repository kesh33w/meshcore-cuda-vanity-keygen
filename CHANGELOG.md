# Changelog

All notable changes to this project are documented here.

## 1.9.0 — 2026-09-12

- Add a saved **Minimum matching hex characters** selector to the rare-key GUI,
  with 10-, 11-, and 12-character choices shared by incidental searches and
  continuous collection.
- Raise structural, repeated-prefix, and pi rules before they reach CUDA;
  visibly deactivate fixed literals that cannot meet the selected minimum while
  preserving their checkbox preference for a later return to 10.
- Correct bookends to treat the configured length as a true minimum and retain
  any matching prefix/suffix width through 32 characters. Version the ruleset
  and CUDA protocol so an older native binary cannot silently apply the former
  exact-width behavior.
- Migrate existing GUI category preferences to the 10-character minimum and
  save both desired categories and minimum in schema-2 settings without doing
  filesystem synchronization on Tk's event thread.
- Record the frozen minimum alongside the active rules and semantic fingerprint
  in schema-5 rare-key history, with older records remaining readable.
- Add CPU, native-parser, GPU-parity, settings-migration, history-sanitization,
  and generated-classifier coverage for all three minimums.

## 1.8.1 — 2026-09-10

- Fix the saved-rare-key browser's page indicator so opening or refreshing the
  browser cannot stop delivery of GUI progress, completion, temperature, and
  cancellation updates.
- Isolate background interface callbacks and unconditionally reschedule the
  bounded event pump after a callback error, keeping later terminal updates
  deliverable even if one update fails.
- Add an automatically selected interactive CUDA launch profile for GUI work,
  reducing optimized progress batches from about 575 ms to 74 ms and retaining
  about 97% of maximum throughput in an A/B/A test on the RTX 4070 Ti. Direct
  CLI searches retain the maximum-throughput profile.
- Add live-GPU coverage for both launch profiles and both engines, along with a
  regression test that delivers a terminal update after a failed GUI callback.

## 1.8.0 — 2026-09-10

- Add live CPU-package and selected-GPU temperature readings to the GUI in
  degrees Celsius, both while idle and during CUDA searches.
- Sample sensors on a background thread so the window stays responsive and
  temperature monitoring does not interrupt or materially slow key generation;
  an A/B/A test on the RTX 4070 Ti showed no measurable throughput loss.
- Read CPU temperatures directly from Linux hardware sensors and query all
  NVIDIA devices with a time- and memory-bounded `nvidia-smi` call. Missing,
  unsupported, or temporarily unavailable sensors degrade cleanly to `—`.
- Map CUDA and NVIDIA devices by PCI bus identity when available, with a safe
  single-device fallback, and extend the key-free CUDA probe protocol to v2
  while retaining compatibility with v1 native binaries.
- Add regression coverage for sensor selection, malformed telemetry, stale
  readings, device replacement and reordering, subprocess bounds, monitor
  shutdown, and both live CUDA probe engines.

## 1.7.0 — 2026-09-10

- Add a scrollable **Rare keys to keep** chooser to the GUI, with a checkbox
  and plain-language description for every configured rare-key rule.
- Remember built-in rule choices across GUI launches while keeping custom
  `--rare-rules` choices session-only and treating stale or malformed settings
  as a safe request to restore the configured defaults.
- Freeze the selected, revalidated ruleset when each search starts so CUDA
  trigger indices, CPU verification, saved metadata, and live reporting remain
  consistent for the entire run.
- Record the active rule IDs in new schema-4 rare-key records and show that
  saved policy in the history browser without altering or hiding older records.
- Verify subset classification and compact CUDA trigger mapping on the CPU and
  GPU; selecting all defaults retains the generated fast path, while measured
  subsets retain the same approximately 870-million-keys/second throughput.

## 1.6.0 — 2026-09-10

- Keep all Tk updates on the GUI event thread while moving GPU
  discovery/readiness work, searches, and rare-history loading to background
  workers; propagate CPU worker failures and perform bounded CUDA child cleanup
  during cancel and window close.
- Read large JSONL histories backward from the tail and add debounced filtering
  plus 500-row pagination for the newest 10,000 valid records.
- Add a canonical, strictly validated `rare_rules.json` configuration shared by
  Python verification and CUDA, with `--rare-rules PATH` for frozen per-search
  custom policies and semantic ruleset fingerprints in new schema-3 records.
- Preserve the original stored analysis for versioned historical records even
  after rules are retired or customized.
- Add a native, key-free CUDA readiness protocol that initializes the selected
  device and executes a bounded synchronized smoke launch through the selected
  real scan engine; Python validates and caches the response instead of assuming
  that a listed GPU can run the engine.
- Build native code for every visible GPU architecture plus forward-compatible
  PTX for the newest, track included CUDA headers, and fingerprint all source
  and tuning inputs so stale binaries are rebuilt across configuration changes.
- Add strict CUDA rule-protocol parsing, generated-default and generic runtime
  classifiers, and CPU/GPU parity coverage without exposing key material.
- Benchmark register limits, batch sizes, launch sizes, packed comparisons,
  longer walks, and fast-math on the RTX 4070 Ti; retain the existing secure
  configuration because no alternative produced a repeatable improvement over
  its approximately 866–870 million keys/second.

## 1.5.2 — 2026-09-10

- Derive a separate pseudorandom expanded Ed25519 private key for every
  optimized CUDA lane, while retaining the fast bounded `+8` walk inside each
  lane.
- Retain at most one identity from any optimized lane and suppress an earlier
  incidental identity when that lane later wins the requested search. This
  prevents saved identities from exposing a small, known scalar relationship.
- Add an independent final rejection for MeshCore-reserved `00` and `ff`
  public-key prefixes.
- Extend retained-key validation with the same Ed25519-to-X25519 shared-secret
  operation used by MeshCore, in addition to public derivation and signatures.
- Add `meshcore-key-audit`, a read-only, key-redacting validator and bounded
  scalar-correlation detector for existing identity JSON and rare-key JSONL
  files.
- Document the v1.5.1-and-earlier optimized-output advisory and local audit and
  rotation guidance.
- Add regression coverage for key exchange, reserved prefixes, the auditor,
  and both CUDA engines.

## 1.5.1 — 2026-09-08

- Stop collecting the `f00df00d00` and `fadefade00` phrase prefixes.
- Keep historical records for both retired categories visible and unchanged.
- Retain `1337133713` as the only canned phrase-prefix rule.

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
