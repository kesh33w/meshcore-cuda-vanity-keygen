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
- Retained GPU results are independently derived and signature-verified on CPU.
- v1.0.0 was interoperability-tested on physical RAK4631 MeshCore firmware.

This project has not received an independent professional cryptographic audit.
Users protecting high-value identities should review the source, secure their
host, and keep private-key backups encrypted and offline.
