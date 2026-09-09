import json
import os
import stat
import struct
import tempfile
import threading
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import meshcore_vanity as vanity


class KeygenTests(unittest.TestCase):
    def test_release_icon_assets_and_desktop_metadata(self):
        root = Path(vanity.__file__).resolve().parent
        png = (root / "assets" / "meshcore-vanity-keygen.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png[16:24], struct.pack(">II", 256, 256))
        self.assertIn("<svg", (root / "assets" / "meshcore-vanity-keygen.svg").read_text())
        desktop = (root / "meshcore-vanity-keygen.desktop.in").read_text()
        self.assertIn("Icon=meshcore-vanity-keygen", desktop)
        self.assertIn("StartupWMClass=Meshcorevanitykeygen", desktop)
        self.assertNotIn("utilities-terminal", desktop)

    def test_meshcore_firmware_vector(self):
        self.assertEqual(
            vanity.SODIUM.derive_public(vanity.TEST_PRIVATE[:32]).hex(),
            vanity.TEST_PUBLIC,
        )
        self.assertTrue(vanity.verify_expanded_key(
            vanity.TEST_PRIVATE, bytes.fromhex(vanity.TEST_PUBLIC)
        ))
        self.assertFalse(vanity.verify_expanded_key(b"short", bytes.fromhex(vanity.TEST_PUBLIC)))
        malformed = bytearray(vanity.TEST_PRIVATE)
        malformed[0] |= 1
        self.assertFalse(vanity.verify_expanded_key(bytes(malformed), bytes.fromhex(vanity.TEST_PUBLIC)))

    def test_constraint_validation(self):
        for prefix in ("00", "ff1234"):
            with self.assertRaises(ValueError):
                vanity.validate_constraints(prefix, "", "")
        with self.assertRaises(ValueError):
            vanity.validate_constraints("a" * 60, "b" * 8, "")
        with self.assertRaises(ValueError):
            vanity.validate_constraints("a" * 60, "aaaa", "b" * 60)
        vanity.validate_constraints("cafe", "beef", "face")

    def test_cpu_search_can_be_cancelled(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(vanity.SearchCancelled):
            vanity.search("123456789abc", "", "", 1, cancel=cancel)

    def test_rare_rule_classifier(self):
        middle = "1" * 44
        self.assertEqual(vanity.interesting_rule("abcde12345" + middle + "abcde12345"), 0)
        self.assertEqual(vanity.interesting_rule("abcde12345" + middle + "54321edcba"), 1)
        self.assertEqual(vanity.interesting_rule("b" * 10 + "1234567890" * 5 + "1234"), 2)
        self.assertEqual(vanity.interesting_rule("deadbeef00" + "1" * 54), 5)
        self.assertEqual(vanity.interesting_rule("3141592653" + "1" * 54), 11)
        self.assertEqual(vanity.interesting_rule("abcdef0123" + "1" * 44 + "fadefade00"), -1)
        self.assertEqual(vanity.interesting_rule("1" * 64), 0)
        self.assertEqual(vanity.interesting_rule("not hex"), -1)
        self.assertEqual(len(vanity.WATCH_REASONS), 12)
        self.assertFalse(any(reason.startswith("suffix-") for reason in vanity.WATCH_REASONS))

    def test_rare_analysis_preserves_stronger_matches(self):
        repeat_key = "a" * 13 + "1234567890abcdef" * 3 + "123"
        repeat = vanity.interesting_matches(repeat_key)[0]
        self.assertEqual((repeat.reason, repeat.length), ("repeat-prefix-13", 13))
        self.assertAlmostEqual(repeat.rarity_bits, 13 * 4 - 3.807, places=3)

        mirror_key = "123456789abcd" + "1" + "0" * 37 + "dcba987654321"
        mirror = vanity.interesting_matches(mirror_key)[0]
        self.assertEqual((mirror.reason, mirror.length), ("mirror-13", 13))

        pi_key = vanity.PI_DIGITS[:15] + "a" * 49
        pi = vanity.interesting_matches(pi_key)[0]
        self.assertEqual((pi.reason, pi.length, pi.rarity_bits),
                         ("prefix-pi-314159265358979", 15, 60.0))

    def test_rare_analysis_keeps_all_matching_traits(self):
        matches = vanity.interesting_matches("1" * 64)
        self.assertEqual(matches[0].reason, "repeat-prefix-64")
        self.assertEqual(
            {match.kind for match in matches}, {"bookend", "mirror", "repeat-prefix"}
        )

    def test_result_file_is_atomic_and_private(self):
        result = vanity.Result("11" * 32, "22" * 64, 1, 0.1, "test", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            vanity.save_result(result, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text())["public_key"], "11" * 32)
            self.assertFalse(list(Path(directory).glob(".*.tmp-*")))

    def test_result_file_refuses_overwrite_without_explicit_permission(self):
        result = vanity.Result("11" * 32, "22" * 64, 1, 0.1, "test", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                vanity.save_result(result, path)
            self.assertEqual(path.read_text(), "original")
            vanity.save_result(result, path, overwrite=True)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text())["private_key"], "22" * 64)

    def test_result_file_refuses_symlink(self):
        result = vanity.Result("11" * 32, "22" * 64, 1, 0.1, "test", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("do not replace", encoding="utf-8")
            link = root / "identity.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                vanity.save_result(result, link, overwrite=True)
            self.assertEqual(target.read_text(), "do not replace")

    def test_difficulty_estimates(self):
        self.assertEqual(vanity.estimate_attempts("cafe", "", ""), 16 ** 4)
        self.assertEqual(vanity.estimate_attempts("cafe", "beef", ""), 16 ** 8)
        # A four-nibble substring has 61 possible placements; this is an
        # intentionally documented approximation because placements overlap.
        self.assertAlmostEqual(vanity.estimate_attempts("", "", "cafe"), 16 ** 4 / 61)
        self.assertEqual(vanity.format_duration(60), "1.0 minutes")

    def test_default_result_path_avoids_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(vanity, "DEFAULT_RESULTS_DIR", Path(directory)):
                first = vanity.default_result_path("a" * 64)
                first.touch()
                second = vanity.default_result_path("a" * 64)
                self.assertNotEqual(first, second)
                self.assertFalse(second.exists())

    def test_cuda_malformed_output_is_rejected(self):
        class FakeProcess:
            def __init__(self):
                self.stderr = StringIO("")
                self.stdout = StringIO("not-json\n")
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
            def terminate(self):
                pass
            def kill(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "Popen", return_value=FakeProcess()):
                with self.assertRaisesRegex(RuntimeError, "invalid result"):
                    vanity.search_cuda("cafe", "", "", watch_path=Path(directory) / "rare.jsonl")

    def test_cuda_gui_termination_is_reported_as_cancellation(self):
        cancel = threading.Event()

        class TerminatedProcess:
            def __init__(self):
                self.stderr = StringIO("GPU 0: NVIDIA GPU, engine optimized\n")
                self.stdout = StringIO("")

            def wait(self, timeout=None):
                cancel.set()
                return -15

            def poll(self):
                return -15

            def terminate(self):
                pass

            def kill(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "Popen", return_value=TerminatedProcess()):
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda(
                        "cafe", "", "", cancel=cancel,
                        watch_path=Path(directory) / "rare.jsonl",
                    )

    def test_count_interesting_handles_missing_and_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            self.assertEqual(vanity.count_interesting(path), 0)
            path.write_text("{}\n\n{}\n", encoding="utf-8")
            self.assertEqual(vanity.count_interesting(path), 2)

    def test_every_interesting_discovery_is_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            public = "123456789a" + "b" * 44 + "123456789a"
            private = "22" * 64
            with mock.patch.object(vanity, "verify_expanded_key", return_value=True):
                first = vanity.append_interesting(path, 0, public, private)
                second = vanity.append_interesting(path, 0, public, private)
                self.assertEqual(first["schema_version"], 2)
                self.assertEqual(first["trigger"], "bookend-10")
                self.assertEqual(second["match_length"], 10)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([record["reason"] for record in records], ["bookend-10"] * 2)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_rare_browser_loads_legacy_records_with_bounded_memory(self):
        legacy_public = vanity.TEST_PUBLIC
        records = [
            {"found_at": f"2026-09-08T00:00:0{index}Z",
             "reason": "suffix-deadbeef00", "public_key": legacy_public,
             "private_key": f"{index + 1:x}" * 128}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            path.write_text(
                "not json\n" + "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded, skipped = vanity.load_interesting_records(path, limit=2)
        self.assertEqual(skipped, 1)
        self.assertEqual([record["found_at"] for record in loaded],
                         ["2026-09-08T00:00:01Z", "2026-09-08T00:00:02Z"])
        self.assertEqual((loaded[0]["match_length"], loaded[0]["rarity_bits"]), (10, 40.0))

    @unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA smoke test is opt-in")
    def test_cuda_result_is_independently_verified(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        with tempfile.TemporaryDirectory() as directory:
            # Seven nibbles makes an attempt-zero hit extraordinarily unlikely,
            # exercising repeated point addition rather than only initialization.
            result = vanity.search_cuda("abc1234", "", "",
                                        watch_path=Path(directory) / "rare.jsonl",
                                        engine="optimized")
        self.assertTrue(result.public_key.startswith("abc1234"))
        self.assertEqual(result.engine, "optimized")
        self.assertEqual(vanity.SODIUM.derive_public(bytes.fromhex(result.private_key)[:32]).hex(),
                         result.public_key)
        self.assertTrue(vanity.verify_expanded_key(
            bytes.fromhex(result.private_key), bytes.fromhex(result.public_key)
        ))

    @unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA smoke test is opt-in")
    def test_baseline_cuda_engine_remains_available(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = vanity.search_cuda("a", "", "",
                                        watch_path=Path(directory) / "rare.jsonl",
                                        engine="baseline")
        self.assertTrue(result.public_key.startswith("a"))
        self.assertEqual(result.engine, "baseline")
        self.assertTrue(vanity.verify_expanded_key(
            bytes.fromhex(result.private_key), bytes.fromhex(result.public_key)
        ))

    @unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA smoke test is opt-in")
    def test_optimized_cuda_suffix_and_substring_modes(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        cases = (("", "abcde", ""), ("", "", "abcde"))
        with tempfile.TemporaryDirectory() as directory:
            for index, (prefix, suffix, contains) in enumerate(cases):
                result = vanity.search_cuda(
                    prefix, suffix, contains,
                    watch_path=Path(directory) / f"rare-{index}.jsonl",
                    engine="optimized",
                )
                self.assertTrue(vanity.matches(result.public_key, prefix, suffix, contains))
                self.assertTrue(vanity.verify_expanded_key(
                    bytes.fromhex(result.private_key), bytes.fromhex(result.public_key)
                ))

    @unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA smoke test is opt-in")
    def test_cuda_search_can_be_cancelled_without_orphaning_process(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        cancel = threading.Event()
        child = []
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda("123456789abc", "", "", cancel=cancel,
                                       watch_path=Path(directory) / "rare.jsonl",
                                       process_update=lambda process: child.append(process))
        finally:
            timer.cancel()
        processes = [process for process in child if process is not None]
        self.assertTrue(processes)
        self.assertIsNotNone(processes[0].poll())


if __name__ == "__main__":
    unittest.main()
