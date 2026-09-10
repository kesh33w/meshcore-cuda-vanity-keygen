# Security policy

## Supported version

Security fixes are applied to the latest release.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting feature instead of opening
a public issue when a report could expose private keys or a reproducible key
generation weakness. Do not include real MeshCore private keys in any report.

## Scope and assurances

- The generator is local-only and does not intentionally make network requests.
- Saved private-key files use owner-only permissions on supported filesystems.
- Retained GPU results are independently derived, signature-verified, and
  key-exchange-validated on CPU.
- Optimized CUDA lanes are derived independently and at most one identity is
  retained from each lane's bounded scalar walk.
- Rare-key policies are strictly parsed, frozen for each search, and recorded by
  semantic fingerprint; every CUDA WATCH result is reclassified and fully
  key-validated by Python against that same policy before it is persisted.
- CUDA readiness is based on a versioned, key-free bounded smoke launch through
  the selected real scan engine rather than device enumeration alone.
- v1.0.0 was interoperability-tested on physical RAK4631 MeshCore firmware.

## v1.5.1-and-earlier saved-key audit

Optimized CUDA releases through v1.5.1 assigned every lane a different offset
inside one bounded scalar sequence. If a single batch retained two or more
identities, those private scalars have a detectable small relationship. This
does not make an individual key invalid, and identities from unrelated batches
remain independent. However, disclosure of one related scalar can allow
recovery of another with a bounded search, so related saved identities should
not be used together.

v1.5.2 derives an independent private key per lane and retains no more than one
identity from each lane. Users with older saved files can identify duplicates,
invalid records, and bounded relationships without printing key material:

```bash
meshcore-key-audit ~/.local/share/meshcore-vanity-keygen/results
```

If `optimized_scalar_relationship` is reported, rotate every identity named in
that finding before using any of them for sensitive deployments. Findings use
opaque file IDs by default; rerun locally with `--show-paths` when you need the
path mapping, keeping in mind that generated filenames include a public-key
prefix. The auditor is a focused local check, not a substitute for an
independent cryptographic audit.

This project has not received an independent professional cryptographic audit.
Users protecting high-value identities should review the source, secure their
host, and keep private-key backups encrypted and offline.
