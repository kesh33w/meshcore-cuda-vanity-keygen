#!/usr/bin/env python3
"""Read-only integrity and correlation audit for saved MeshCore identities.

The report deliberately contains opaque file/line identifiers but no paths or
public/private key material unless real paths are explicitly requested. Input
files are opened without following symbolic links on platforms that provide
``O_NOFOLLOW``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
from typing import BinaryIO, Iterable, Iterator, Optional

import meshcore_vanity as vanity


REPORT_SCHEMA = 1
# v1.5.1 used SM_count * 16 blocks * 128 threads * 4096 attempts.
# 2**40 conservatively covers practical GPUs and substantial custom launch
# configurations while leaving a negligible false-positive probability for
# independently generated clamped scalars.
DEFAULT_MAX_CANDIDATE_SPAN = 2**40
DEFAULT_MAX_FINDINGS = 100
MAX_RECORD_BYTES = 1_048_576
AUDIT_MESSAGE = b"MeshCore saved-key audit compatibility test"


@dataclass(frozen=True)
class _Record:
    identifier: str
    public: Optional[bytes]
    scalar: Optional[bytes]


class _ReportBuilder:
    def __init__(self, max_candidate_span: int, max_findings: int,
                 show_paths: bool) -> None:
        self.max_candidate_span = max_candidate_span
        self.max_findings = max_findings
        self.show_paths = show_paths
        self.summary: Counter[str] = Counter()
        self.finding_counts: Counter[str] = Counter()
        self.findings: list[dict[str, object]] = []
        self._seen_files: set[tuple[int, int]] = set()
        self._path_ids: dict[str, str] = {}

    def identifier(self, path: Path, line: Optional[int] = None) -> str:
        """Return an opaque ID by default so key-derived filenames stay private."""
        absolute = os.path.abspath(os.fspath(path))
        if self.show_paths:
            identifier = absolute
        else:
            identifier = self._path_ids.setdefault(
                absolute, f"file-{len(self._path_ids) + 1}"
            )
        return f"{identifier}:line-{line}" if line is not None else identifier

    def finding(self, kind: str, identifiers: Iterable[str], count: int = 1) -> None:
        """Record a non-secret finding and retain only a bounded sample."""
        self.finding_counts[kind] += count
        if len(self.findings) >= self.max_findings:
            return
        identifiers = list(identifiers)
        displayed = identifiers[:20]
        item: dict[str, object] = {"type": kind, "records": displayed}
        if len(identifiers) > len(displayed):
            item["record_identifiers_omitted"] = len(identifiers) - len(displayed)
        self.findings.append(item)

    def finish(self) -> dict[str, object]:
        finding_total = sum(self.finding_counts.values())
        return {
            "schema_version": REPORT_SCHEMA,
            "parameters": {
                "max_candidate_span": self.max_candidate_span,
                "max_scalar_difference": 8 * self.max_candidate_span,
                "paths_included": self.show_paths,
            },
            "summary": {
                key: self.summary[key]
                for key in (
                    "input_paths",
                    "files_discovered",
                    "files_scanned",
                    "unsupported_files_skipped",
                    "duplicate_files_skipped",
                    "symlinks_skipped",
                    "records_seen",
                    "records_compatible",
                    "records_invalid",
                    "duplicate_public_groups",
                    "duplicate_public_records",
                    "duplicate_scalar_groups",
                    "duplicate_scalar_records",
                    "optimized_scalar_related_pairs",
                )
            },
            "finding_counts": dict(sorted(self.finding_counts.items())),
            "findings": self.findings,
            "findings_omitted": max(0, finding_total - len(self.findings)),
            "clean": finding_total == 0,
        }


def _error_kind(error: OSError) -> str:
    if error.errno in (errno.EACCES, errno.EPERM):
        return "permission_error"
    if error.errno == errno.ENOENT:
        return "path_not_found"
    if error.errno == errno.ELOOP:
        return "symlink_refused"
    return "io_error"


def _discover_directory(path: Path, report: _ReportBuilder) -> Iterator[Path]:
    """Recursively yield regular JSON/JSONL files without following links."""
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name, reverse=True)
        except OSError as error:
            report.finding(_error_kind(error), (report.identifier(directory),))
            continue
        for entry in ordered:
            entry_path = Path(entry.path)
            try:
                if entry.is_symlink():
                    report.summary["symlinks_skipped"] += 1
                    report.finding("symlink_refused", (report.identifier(entry_path),))
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(entry_path)
                elif entry.is_file(follow_symlinks=False):
                    if entry_path.suffix.lower() in (".json", ".jsonl"):
                        report.summary["files_discovered"] += 1
                        yield entry_path
                    else:
                        report.summary["unsupported_files_skipped"] += 1
            except OSError as error:
                report.finding(_error_kind(error), (report.identifier(entry_path),))


def _discover(inputs: Iterable[Path], report: _ReportBuilder) -> Iterator[Path]:
    for path in inputs:
        report.summary["input_paths"] += 1
        identifier = report.identifier(path)
        try:
            metadata = os.stat(path, follow_symlinks=False)
        except OSError as error:
            report.finding(_error_kind(error), (identifier,))
            continue
        if stat.S_ISLNK(metadata.st_mode):
            report.summary["symlinks_skipped"] += 1
            report.finding("symlink_refused", (identifier,))
        elif stat.S_ISDIR(metadata.st_mode):
            yield from _discover_directory(path, report)
        elif stat.S_ISREG(metadata.st_mode):
            report.summary["files_discovered"] += 1
            yield path
        else:
            report.finding("non_regular_file", (identifier,))


def _open_regular(path: Path) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        # The generator takes an exclusive lock around each JSONL append.  A
        # shared lock makes the audit wait for that append to be flushed and
        # prevents a cooperating live writer from exposing a partial record.
        # Closing the returned file releases the lock on every exit path.
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return os.fdopen(descriptor, "rb"), metadata
    except BaseException:
        os.close(descriptor)
        raise


def _parse_json_bytes(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"))


def _records_from_file(path: Path, report: _ReportBuilder) -> Iterator[tuple[str, object]]:
    identifier = report.identifier(path)
    try:
        file, metadata = _open_regular(path)
    except OSError as error:
        report.finding(_error_kind(error), (identifier,))
        return
    with file:
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in report._seen_files:
            report.summary["duplicate_files_skipped"] += 1
            return
        report._seen_files.add(identity)
        report.summary["files_scanned"] += 1

        if path.suffix.lower() != ".jsonl":
            try:
                raw = file.read(MAX_RECORD_BYTES + 1)
            except OSError as error:
                report.finding(_error_kind(error), (identifier,))
                return
            if len(raw) > MAX_RECORD_BYTES:
                report.summary["records_seen"] += 1
                report.summary["records_invalid"] += 1
                report.finding("record_too_large", (identifier,))
                return
            try:
                yield identifier, _parse_json_bytes(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                report.summary["records_seen"] += 1
                report.summary["records_invalid"] += 1
                report.finding("malformed_json", (identifier,))
            return

        line_number = 0
        while True:
            try:
                raw = file.readline(MAX_RECORD_BYTES + 1)
            except OSError as error:
                report.finding(_error_kind(error), (identifier,))
                return
            if not raw:
                return
            line_number += 1
            record_identifier = report.identifier(path, line_number)
            if len(raw) > MAX_RECORD_BYTES:
                # Drain the remainder of this overlong logical line in bounded chunks.
                while raw and not raw.endswith(b"\n"):
                    try:
                        raw = file.readline(MAX_RECORD_BYTES + 1)
                    except OSError as error:
                        report.finding(_error_kind(error), (identifier,))
                        return
                report.summary["records_seen"] += 1
                report.summary["records_invalid"] += 1
                report.finding("record_too_large", (record_identifier,))
                continue
            if not raw.strip():
                continue
            try:
                yield record_identifier, _parse_json_bytes(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                report.summary["records_seen"] += 1
                report.summary["records_invalid"] += 1
                report.finding("malformed_json", (record_identifier,))


def _decode_hex_field(record: dict[str, object], name: str, length: int) -> Optional[bytes]:
    value = record.get(name)
    if not isinstance(value, str) or len(value) != length * 2:
        return None
    # Saved generator records use canonical lowercase hexadecimal.
    if any(character not in "0123456789abcdef" for character in value):
        return None
    try:
        return bytes.fromhex(value)
    except ValueError:
        return None


def _inspect_record(identifier: str, value: object,
                    report: _ReportBuilder) -> _Record:
    report.summary["records_seen"] += 1
    if not isinstance(value, dict):
        report.summary["records_invalid"] += 1
        report.finding("invalid_record_format", (identifier,))
        return _Record(identifier, None, None)

    public = _decode_hex_field(value, "public_key", 32)
    private = _decode_hex_field(value, "private_key", 64)
    if public is None or private is None:
        report.summary["records_invalid"] += 1
        report.finding("invalid_record_format", (identifier,))
        return _Record(identifier, public, private[:32] if private is not None else None)

    scalar = private[:32]
    compatible = True
    if private[0] & 7 or private[31] & 128 or not private[31] & 64:
        compatible = False
        report.finding("invalid_expanded_scalar", (identifier,))
    if public[0] in (0, 255):
        compatible = False
        report.finding("meshcore_public_prefix_rejected", (identifier,))

    derivation_valid = False
    try:
        derivation_valid = vanity.SODIUM.derive_public(scalar) == public
    except (ValueError, RuntimeError):
        pass
    if not derivation_valid:
        compatible = False
        report.finding("public_derivation_mismatch", (identifier,))

    signature_valid = False
    try:
        signature = vanity.sign_expanded(private, public, AUDIT_MESSAGE)
        signature_valid = vanity.SODIUM.verify(signature, AUDIT_MESSAGE, public)
    except (ValueError, RuntimeError):
        pass
    if not signature_valid:
        compatible = False
        report.finding("signature_compatibility_failure", (identifier,))

    # Keep this project-level compatibility check in addition to the two
    # component checks so changes to the generator's acceptance policy are
    # visible to the auditor.
    if compatible and not vanity.verify_expanded_key(private, public):
        compatible = False
        report.finding("project_compatibility_failure", (identifier,))

    if compatible:
        report.summary["records_compatible"] += 1
    else:
        report.summary["records_invalid"] += 1
    return _Record(identifier, public, scalar)


def _duplicate_findings(records: list[_Record], report: _ReportBuilder) -> None:
    public_records: dict[bytes, list[str]] = defaultdict(list)
    scalar_records: dict[bytes, list[str]] = defaultdict(list)
    for record in records:
        if record.public is not None:
            public_records[record.public].append(record.identifier)
        if record.scalar is not None:
            scalar_records[record.scalar].append(record.identifier)

    for identifiers in public_records.values():
        if len(identifiers) > 1:
            report.summary["duplicate_public_groups"] += 1
            report.summary["duplicate_public_records"] += len(identifiers)
            report.finding("duplicate_public_key", identifiers)
    for identifiers in scalar_records.values():
        if len(identifiers) > 1:
            report.summary["duplicate_scalar_groups"] += 1
            report.summary["duplicate_scalar_records"] += len(identifiers)
            report.finding("duplicate_private_scalar", identifiers)


def _relationship_findings(records: list[_Record], report: _ReportBuilder) -> None:
    scalar_records: dict[int, list[str]] = defaultdict(list)
    for record in records:
        # Engine metadata is optional and not a trustworthy security boundary.
        # The pattern is vanishingly unlikely for independent keys, so check all
        # well-formed scalars, including records from older releases.
        if record.scalar is not None:
            scalar_records[int.from_bytes(record.scalar, "little")].append(record.identifier)

    # A difference is divisible by eight exactly when the two scalars have the
    # same residue modulo eight.  Maintain one sorted sliding window per
    # residue, along with its record count.  This counts all related record
    # pairs in O(n log n + retained_samples) time after sorting, even if every
    # scalar falls inside one large clustered window.  Equal scalar values are
    # grouped above and remain the responsibility of the duplicate checks.
    windows: list[deque[tuple[int, list[str]]]] = [deque() for _ in range(8)]
    window_record_counts = [0] * 8
    maximum_difference = 8 * report.max_candidate_span
    finding_kind = "optimized_scalar_relationship"

    for right_scalar, right_identifiers in sorted(scalar_records.items()):
        residue = right_scalar % 8
        window = windows[residue]
        while window and right_scalar - window[0][0] > maximum_difference:
            _, expired_identifiers = window.popleft()
            window_record_counts[residue] -= len(expired_identifiers)

        pair_count = len(right_identifiers) * window_record_counts[residue]
        if pair_count:
            report.summary["optimized_scalar_related_pairs"] += pair_count

            # Retain representative record pairs while counting every pair.
            available_slots = max(0, report.max_findings - len(report.findings))
            emitted = 0
            if available_slots:
                for _, left_identifiers in window:
                    for left_identifier in left_identifiers:
                        for right_identifier in right_identifiers:
                            report.finding(
                                finding_kind,
                                (left_identifier, right_identifier),
                            )
                            emitted += 1
                            if emitted >= available_slots:
                                break
                        if emitted >= available_slots:
                            break
                    if emitted >= available_slots:
                        break
            if pair_count > emitted:
                report.finding_counts[finding_kind] += pair_count - emitted

        window.append((right_scalar, right_identifiers))
        window_record_counts[residue] += len(right_identifiers)


def audit_paths(paths: Iterable[os.PathLike[str] | str], *,
                max_candidate_span: int = DEFAULT_MAX_CANDIDATE_SPAN,
                max_findings: int = DEFAULT_MAX_FINDINGS,
                show_paths: bool = False) -> dict[str, object]:
    """Audit JSON identities and JSONL logs without returning key material."""
    if max_candidate_span < 0:
        raise ValueError("max_candidate_span must not be negative")
    if max_findings < 0:
        raise ValueError("max_findings must not be negative")
    report = _ReportBuilder(max_candidate_span, max_findings, show_paths)
    records: list[_Record] = []
    for path in _discover((Path(path) for path in paths), report):
        for identifier, value in _records_from_file(path, report):
            records.append(_inspect_record(identifier, value, report))
    _duplicate_findings(records, report)
    _relationship_findings(records, report)
    return report.finish()


def format_text(report: dict[str, object]) -> str:
    """Render a concise report while keeping all key material out of output."""
    parameters = report["parameters"]
    summary = report["summary"]
    finding_counts = report["finding_counts"]
    findings = report["findings"]
    assert isinstance(parameters, dict) and isinstance(summary, dict)
    assert isinstance(finding_counts, dict) and isinstance(findings, list)
    lines = [
        "MeshCore saved-key audit",
        (f"Files: {summary['files_scanned']} scanned; "
         f"{summary['symlinks_skipped']} symlinks refused; "
         f"{summary['duplicate_files_skipped']} duplicate files skipped"),
        (f"Records: {summary['records_seen']} seen; "
         f"{summary['records_compatible']} compatible; "
         f"{summary['records_invalid']} invalid"),
        (f"Duplicates: {summary['duplicate_public_groups']} public-key groups; "
         f"{summary['duplicate_scalar_groups']} private-scalar groups"),
        (f"Optimized scalar relationships: "
         f"{summary['optimized_scalar_related_pairs']} record pairs "
         f"within {parameters['max_candidate_span']} candidates"),
    ]
    if finding_counts:
        lines.append("Finding counts:")
        for kind, count in sorted(finding_counts.items()):
            lines.append(f"  {kind}: {count}")
    if findings:
        lines.append("Finding samples (record identifiers only):")
        for finding in findings:
            identifiers = ", ".join(json.dumps(item) for item in finding["records"])
            lines.append(f"  {finding['type']}: {identifiers}")
    omitted = int(report["findings_omitted"])
    if omitted:
        lines.append(f"Finding samples omitted: {omitted}")
    lines.append("Result: CLEAN" if report["clean"] else "Result: FINDINGS PRESENT")
    if parameters.get("paths_included"):
        lines.append("Key contents are omitted; explicitly requested paths may contain key-derived text.")
    else:
        lines.append("No paths, public keys, or private keys are included in this report.")
    return "\n".join(lines)


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Read-only validation and correlation audit for MeshCore "
                     "identity JSON files and rare-key JSONL logs."),
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="identity JSON file, rare-key JSONL file, or directory")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit machine-readable JSON")
    parser.add_argument(
        "--show-paths", action="store_true",
        help="include real paths for local remediation (filenames may reveal key fragments)",
    )
    parser.add_argument(
        "--max-candidate-span", type=_nonnegative_integer,
        default=DEFAULT_MAX_CANDIDATE_SPAN,
        help=("largest optimized-engine candidate separation to flag "
              f"(default: {DEFAULT_MAX_CANDIDATE_SPAN})"),
    )
    parser.add_argument(
        "--max-findings", type=_nonnegative_integer, default=DEFAULT_MAX_FINDINGS,
        help=f"maximum finding samples to print (default: {DEFAULT_MAX_FINDINGS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = audit_paths(
        arguments.paths,
        max_candidate_span=arguments.max_candidate_span,
        max_findings=arguments.max_findings,
        show_paths=arguments.show_paths,
    )
    if arguments.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
