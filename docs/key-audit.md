# Audit saved identities

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
candidates. far above the shipped pre-v1.5.2 launch span. Increase
`--max-candidate-span` only when auditing keys made by a custom build whose
launch span exceeded that bound.

For licensing, see [LICENSE](../LICENSE) and
[third-party notices](../THIRD_PARTY_NOTICES.md).

