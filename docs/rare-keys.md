# Rare-key collection

## Collect continuously

Select **Continuous rare collector** in the GUI and click **Start rare
collector** to collect built-in rare keys indefinitely without inventing an
astronomically difficult vanity target. Pattern fields are disabled in this
mode. Use **Rare keys to keep → Choose...** to select the categories collected.
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
material does not. For an unattended collector, this command-line mode also
retains the maximum-throughput CUDA launch profile.

## Collect during a vanity search

Every CUDA search checks each generated public key both for the requested
pattern and for the selected rare-key rules. The second check is automatic; no
extra option is required and it does not interrupt the requested search. In the
GUI, **Rare keys to keep → Choose...** opens a checkbox for every configured
category and a **Minimum matching hex characters** selector with choices 10,
11, and 12. At least one eligible category must remain selected. The complete
policy is frozen when a search starts and applies to new incidental discoveries
as well as the continuous collector. It does not delete or hide anything
already saved.

Built-in GUI choices and the minimum are remembered in the non-secret
preferences file `~/.config/meshcore-vanity-keygen/settings.json`. Existing
preference files are migrated with their original 10-character minimum. If the
rule configuration changes or that file is unreadable, the app safely restores
the configured defaults. Choices made while using a custom `--rare-rules` file
last only for that GUI session, so they cannot silently rewrite the custom
policy. When an incidental rare match appears, its public/private relationship,
rare rule, and MeshCore validity are independently verified on the CPU and it is
immediately appended to `results/rare-keys.jsonl`. The GUI shows a live count,
the latest rule matched, and a public-key preview. Its **Rare keys...** browser can
filter and sort the saved history by date, match, length, rarity, or public key.
Private keys remain masked and are independently verified before reveal, copy,
or owner-only export. Incidental collection is a CUDA feature; the slower CPU
fallback only searches for the requested pattern.

The built-in rules are deliberately much harder than four-character vanity
patterns. Their default minimum is ten hexadecimal characters:

- At least ten leading characters equal the same number of trailing characters
  (`bookend-10`). The longest matching bookend is recorded.
- At least ten leading characters equal the reverse of the final characters
  (`mirror-10`).
- At least ten starting characters are identical (`repeat-prefix-10`), such as
  `aaaaaaaaaa`. Repeated `0` and `f` prefixes are excluded because MeshCore
  rejects identities beginning with bytes `00` and `ff`.
- The exact phrase `1337133713` appears at the beginning.
- The first ten decimal digits of pi appear at the beginning:
  `3141592653` (`prefix-pi-3141592653`).

Selecting 11 or 12 raises the structural, repeated-prefix, and pi rules to that
minimum before work reaches the GPU. The fixed `1337133713` phrase is visibly
inactive above 10 because adding arbitrary characters would not make that
10-character phrase rarer. Its checkbox preference is retained, so it becomes
active again if the minimum returns to 10. Rules in a custom file that already
require more characters are never weakened.

These defaults live in [`rare_rules.json`](../rare_rules.json), which is the single
source used by both Python verification and the generated fast CUDA classifier.
To experiment without modifying the installed defaults, copy that file, edit
the copy, and launch with `--rare-rules PATH` (this also works with `--gui`).
Supported rule kinds are `bookend`, `mirror`, `repeat-prefix`,
`literal-prefix`, and `sequence-prefix`. Rule order is significant: the first
matching enabled rule supplies the numeric CUDA trigger, while CPU analysis
still records every matching trait.

New custom files use ruleset schema 2. Schema-1 files remain accepted and are
migrated in memory to schema-2 semantics and fingerprints; the source file is
not rewritten.

Custom files are strictly validated before GPU work starts. They are limited to
32 rules and 256 KiB, must use canonical lowercase hexadecimal values and valid
MeshCore prefixes, and cannot configure an individual or combined hit rate high
enough to overwhelm incidental-result handling. Each search freezes one parsed
ruleset; its semantic ID and SHA-256 fingerprint are written into every new
rare-key record. The GUI chooser revalidates every selected subset with the same
safety limits, including rules that were initially disabled in a custom file.

With all defaults selected, the approximate average discovery cadence at the
measured 3.45G keys/s is:

| Minimum | Active categories | Average time between finds |
| ---: | ---: | ---: |
| 10 | 5 | 18 seconds |
| 11 | 4 | 5.0 minutes |
| 12 | 4 | 1.3 hours |

The fixed phrase accounts for the fifth category at 10. Bookend probability is
calculated across every possible matching width from the selected minimum
through 32. Random search times vary widely, and a run can be much shorter or
longer than the average.

Each JSONL record keeps the matching pair together:

```json
{"schema_version":5,"found_at":"2026-09-12T12:34:56Z","trigger":"repeat-prefix-10","reason":"repeat-prefix-13","match_length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322","matches":[{"reason":"repeat-prefix-13","kind":"repeat-prefix","length":13,"rarity_bits":48.193,"mean_attempts":"321685687669322"}],"public_key":"...","private_key":"...","backend":"cuda","engine":"optimized","ruleset_id":"meshcore-default","ruleset_fingerprint":"...","active_rule_ids":["bookend","mirror","repeat-prefix","prefix-1337133713","pi"],"minimum_match_nibbles":10}
```

The selected GPU minimum remains the collection threshold, but CPU analysis
preserves stronger properties of a retained key. For example, a key beginning
with 13 identical characters is labeled `repeat-prefix-13`, and a pi prefix
continuing for 15 digits records all 15 rather than being flattened to ten.
When one key has multiple interesting traits, the strongest becomes `reason`
and every trait is retained under `matches`. Rarity is expressed as equivalent
random-search bits, making unlike patterns sortable on one scale. Schema-5
records retain the active rule IDs and minimum so the history browser can show
the complete policy that was in force when each key was found. Older records
remain fully readable and display unavailable policy details as not recorded.

The file is created with owner-only permissions (`0600`) when the search starts;
an empty file means no rare key has been found. A source checkout stores it
under local `results/`; an installed copy uses its private application-data
directory. Both remain independent of the shell's working directory. Use
`--watch-output PATH` to choose another CLI location.

Every discovery is retained, even when earlier keys matched the same category.
File locking prevents concurrent searches from corrupting or interleaving
records. Collection stops when the requested key is found or the search is
cancelled; rare keys already written remain saved. At the measured rate and
current criteria, continuous searching averages about 4,900 new records per
day. The GUI keeps its live count in memory, reads history backward from the
file tail, and presents at most 500 rows per page from the newest 10,000 valid
records. Upgrading does not rewrite or delete legacy
records; their basic rarity metadata is inferred when displayed. Legacy suffix
records created by v1.0.x and records from later-removed phrase categories
remain visible, but those patterns are no longer collected. Version-2 and newer
records are larger than the original minimal records, so storage growth depends
on how many traits each key matches.

