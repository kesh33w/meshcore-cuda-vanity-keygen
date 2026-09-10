import contextlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import fcntl
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest

import meshcore_key_audit as audit
import meshcore_vanity as vanity


def related_identity(candidate_offset: int, *, engine: str = "optimized") -> dict[str, object]:
    scalar_value = int.from_bytes(vanity.TEST_PRIVATE[:32], "little") + 8 * candidate_offset
    scalar = scalar_value.to_bytes(32, "little")
    private = scalar + vanity.TEST_PRIVATE[32:]
    public = vanity.SODIUM.derive_public(scalar)
    return {
        "public_key": public.hex(),
        "private_key": private.hex(),
        "backend": "cuda",
        "engine": engine,
    }


class KeyAuditTests(unittest.TestCase):
    def write_json(self, path: Path, record: object) -> None:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_valid_json_and_jsonl_are_verified_without_key_output(self):
        first = related_identity(0)
        second = related_identity(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "identity.json", first)
            (root / "rare-keys.jsonl").write_text(
                json.dumps(second) + "\n\n", encoding="utf-8"
            )

            report = audit.audit_paths([root], max_candidate_span=17)
            rendered = audit.format_text(report)
            encoded = json.dumps(report)

        self.assertEqual(report["summary"]["records_seen"], 2)
        self.assertEqual(report["summary"]["records_compatible"], 2)
        self.assertEqual(report["summary"]["optimized_scalar_related_pairs"], 1)
        self.assertNotIn(first["private_key"], rendered)
        self.assertNotIn(second["private_key"], rendered)
        self.assertNotIn(first["private_key"], encoded)
        self.assertNotIn(second["private_key"], encoded)
        self.assertNotIn(first["public_key"], rendered)
        self.assertNotIn(first["public_key"], encoded)

    def test_live_jsonl_append_is_locked_until_the_record_is_complete(self):
        record_bytes = (json.dumps(related_identity(0)) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare-keys.jsonl"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            executor = ThreadPoolExecutor(max_workers=1)
            future = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                midpoint = len(record_bytes) // 2
                os.write(descriptor, record_bytes[:midpoint])
                future = executor.submit(audit.audit_paths, [path])

                # The reader must not parse the partial line while the writer
                # holds the same exclusive lock used by the generator.
                with self.assertRaises(FutureTimeout):
                    future.result(timeout=0.1)

                os.write(descriptor, record_bytes[midpoint:])
                os.fsync(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
                descriptor = -1
                report = future.result(timeout=2)
            finally:
                if descriptor >= 0:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                executor.shutdown(wait=True)

        self.assertEqual(report["summary"]["records_seen"], 1)
        self.assertEqual(report["summary"]["records_compatible"], 1)
        self.assertNotIn("malformed_json", report["finding_counts"])

    def test_relationship_limit_is_in_candidate_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "one.json", related_identity(0))
            self.write_json(root / "two.json", related_identity(11))
            below = audit.audit_paths([root], max_candidate_span=10)
            boundary = audit.audit_paths([root], max_candidate_span=11)

        self.assertEqual(below["summary"]["optimized_scalar_related_pairs"], 0)
        self.assertEqual(boundary["summary"]["optimized_scalar_related_pairs"], 1)
        self.assertEqual(boundary["parameters"]["max_scalar_difference"], 88)

    def test_default_span_covers_more_than_the_development_gpu(self):
        old_development_span = 503_316_480
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "one.json", related_identity(0))
            self.write_json(
                root / "two.json", related_identity(old_development_span + 1)
            )
            report = audit.audit_paths([root])

        self.assertGreater(audit.DEFAULT_MAX_CANDIDATE_SPAN, old_development_span)
        self.assertEqual(report["summary"]["optimized_scalar_related_pairs"], 1)

    def test_relationship_check_does_not_trust_engine_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "one.json", related_identity(0, engine="cpu"))
            self.write_json(root / "two.json", related_identity(1, engine="optimized"))
            report = audit.audit_paths([root], max_candidate_span=10)

        self.assertEqual(report["summary"]["records_compatible"], 2)
        self.assertEqual(report["summary"]["optimized_scalar_related_pairs"], 1)

    def test_duplicate_public_and_scalar_records_are_detected(self):
        record = related_identity(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "one.json", record)
            self.write_json(root / "two.json", record)
            report = audit.audit_paths([root])

        self.assertEqual(report["summary"]["duplicate_public_groups"], 1)
        self.assertEqual(report["summary"]["duplicate_public_records"], 2)
        self.assertEqual(report["summary"]["duplicate_scalar_groups"], 1)
        self.assertEqual(report["summary"]["duplicate_scalar_records"], 2)
        self.assertEqual(report["summary"]["optimized_scalar_related_pairs"], 0)

    def test_clustered_relationship_scan_counts_all_pairs_with_bounded_samples(self):
        scalar_base = int.from_bytes(vanity.TEST_PRIVATE[:32], "little")
        scalar_count = 10_000
        records = [
            audit._Record(
                f"record-{index}",
                None,
                (scalar_base + 8 * index).to_bytes(32, "little"),
            )
            for index in range(scalar_count)
        ]
        report = audit._ReportBuilder(
            max_candidate_span=scalar_count,
            max_findings=7,
            show_paths=False,
        )

        audit._relationship_findings(records, report)

        expected_pairs = scalar_count * (scalar_count - 1) // 2
        self.assertEqual(
            report.summary["optimized_scalar_related_pairs"], expected_pairs
        )
        self.assertEqual(
            report.finding_counts["optimized_scalar_relationship"], expected_pairs
        )
        self.assertEqual(len(report.findings), 7)

    def test_malformed_records_and_key_mismatches_are_reported_safely(self):
        valid = related_identity(0)
        mismatch = dict(valid)
        mismatch["public_key"] = related_identity(1)["public_key"]
        secret = valid["private_key"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{not-json " + secret, encoding="utf-8")
            self.write_json(root / "wrong.json", mismatch)
            (root / "rare.jsonl").write_text(
                json.dumps({"public_key": "nope", "private_key": secret}) + "\n",
                encoding="utf-8",
            )
            report = audit.audit_paths([root])
            rendered = audit.format_text(report)
            encoded = json.dumps(report)

        self.assertEqual(report["summary"]["records_seen"], 3)
        self.assertEqual(report["summary"]["records_invalid"], 3)
        self.assertEqual(report["finding_counts"]["malformed_json"], 1)
        self.assertEqual(report["finding_counts"]["invalid_record_format"], 1)
        self.assertEqual(report["finding_counts"]["public_derivation_mismatch"], 1)
        self.assertNotIn(secret, rendered)
        self.assertNotIn(secret, encoded)

    def test_directory_walk_refuses_symlinks_and_skips_other_files(self):
        record = related_identity(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.json"
            self.write_json(outside, record)
            scanned = root / "scan"
            scanned.mkdir()
            (scanned / "linked.json").symlink_to(outside)
            (scanned / "notes.txt").write_text("not an identity", encoding="utf-8")
            report = audit.audit_paths([scanned])

        self.assertEqual(report["summary"]["files_scanned"], 0)
        self.assertEqual(report["summary"]["symlinks_skipped"], 1)
        self.assertEqual(report["summary"]["unsupported_files_skipped"], 1)
        self.assertEqual(report["finding_counts"]["symlink_refused"], 1)

    def test_overlapping_inputs_do_not_audit_hard_link_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one.json"
            second = root / "two.json"
            self.write_json(first, related_identity(0))
            os.link(first, second)
            report = audit.audit_paths([first, root])

        self.assertEqual(report["summary"]["files_scanned"], 1)
        self.assertEqual(report["summary"]["duplicate_files_skipped"], 2)
        self.assertEqual(report["summary"]["records_seen"], 1)

    def test_cli_json_output_contains_no_keys_and_returns_finding_status(self):
        record = related_identity(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"identity-{record['public_key'][:12]}.json"
            self.write_json(path, record)
            stream = StringIO()
            with contextlib.redirect_stdout(stream):
                clean_status = audit.main(["--json", str(path)])
            clean_output = stream.getvalue()

            duplicate = Path(directory) / "duplicate.json"
            self.write_json(duplicate, record)
            stream = StringIO()
            with contextlib.redirect_stdout(stream):
                finding_status = audit.main(["--json", str(Path(directory))])

        self.assertEqual(clean_status, 0)
        self.assertEqual(finding_status, 1)
        self.assertNotIn(record["private_key"], clean_output)
        self.assertNotIn(record["public_key"], clean_output)
        self.assertNotIn(record["public_key"][:12], clean_output)

    def test_default_finding_identifiers_are_opaque(self):
        record = related_identity(0)
        public_fragment = str(record["public_key"])[:12]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / f"first-{public_fragment}.json", record)
            self.write_json(root / f"second-{public_fragment}.json", record)
            report = audit.audit_paths([root])
            encoded = json.dumps(report)

        self.assertNotIn(public_fragment, encoded)
        identifiers = report["findings"][0]["records"]
        self.assertEqual(len(identifiers), 2)
        self.assertTrue(all(str(item).startswith("file-") for item in identifiers))
        self.assertFalse(report["parameters"]["paths_included"])

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            audit.audit_paths([], max_candidate_span=-1)
        with self.assertRaises(ValueError):
            audit.audit_paths([], max_findings=-1)


if __name__ == "__main__":
    unittest.main()
