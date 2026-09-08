import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path

import meshcore_vanity as vanity


class KeygenTests(unittest.TestCase):
    def test_meshcore_firmware_vector(self):
        self.assertEqual(
            vanity.SODIUM.derive_public(vanity.TEST_PRIVATE[:32]).hex(),
            vanity.TEST_PUBLIC,
        )

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
        self.assertEqual(vanity.interesting_rule("deadbeef00" + "1" * 54), 4)
        self.assertEqual(vanity.interesting_rule("1" * 54 + "fadefade00"), 17)
        self.assertEqual(vanity.interesting_rule("1" * 64), 0)
        self.assertEqual(vanity.interesting_rule("not hex"), -1)

    def test_result_file_is_private_even_if_it_already_exists(self):
        result = vanity.Result("11" * 32, "22" * 64, 1, 0.1, "test", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text("old", encoding="utf-8")
            path.chmod(0o644)
            vanity.save_result(result, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text())["public_key"], "11" * 32)

    @unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA smoke test is opt-in")
    def test_cuda_result_is_independently_verified(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = vanity.search_cuda("a", "", "", watch_path=Path(directory) / "rare.jsonl")
        self.assertTrue(result.public_key.startswith("a"))
        self.assertEqual(vanity.SODIUM.derive_public(bytes.fromhex(result.private_key)[:32]).hex(),
                         result.public_key)

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
