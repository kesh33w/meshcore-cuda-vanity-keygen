import json
import os
import re
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import meshcore_vanity as vanity


class SecurityValidationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_CUDA_TESTS") == "1", "CUDA isolation test is opt-in"
    )
    def test_gpu_lane_derivation_matches_fixed_isolated_vectors(self):
        executable = vanity.cuda_executable()
        self.assertTrue(
            vanity.cuda_available(),
            "RUN_CUDA_TESTS=1 requires a built CUDA engine and a visible CUDA device",
        )
        completed = subprocess.run(
            [str(executable), "--internal-test-lane-isolation"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout + completed.stderr
        self.assertIn("CUDA lane-isolation self-test: PASS (4 fixed lanes)", output)
        self.assertIsNone(
            re.search(r"(?i)(?<![0-9a-f])[0-9a-f]{64,}(?![0-9a-f])", output),
            "the CUDA isolation self-test must not print key material",
        )

    def test_meshcore_shared_secret_matches_firmware_reference(self):
        peer_public = vanity.ECDH_TEST_PEER_PUBLIC
        self.assertEqual(
            vanity.SODIUM.derive_public(vanity.ECDH_TEST_PEER_PRIVATE[:32]),
            peer_public,
        )

        expected = bytes.fromhex(
            "86d47e289ad85d9b272a0fd1a739f6931d47ce0be5c5ea5ba6206644b73be228"
        )
        forward = vanity.meshcore_shared_secret(vanity.TEST_PRIVATE, peer_public)
        reverse = vanity.meshcore_shared_secret(
            vanity.ECDH_TEST_PEER_PRIVATE, bytes.fromhex(vanity.TEST_PUBLIC)
        )
        self.assertEqual(forward, expected)
        self.assertEqual(reverse, expected)

    def test_retained_key_validation_requires_nonzero_ecdh_parity(self):
        public = bytes.fromhex(vanity.TEST_PUBLIC)
        with mock.patch.object(
                vanity, "meshcore_shared_secret",
                side_effect=(b"\x01" * 32, b"\x02" * 32)):
            self.assertFalse(vanity.verify_expanded_key(vanity.TEST_PRIVATE, public))
        with mock.patch.object(
                vanity, "meshcore_shared_secret",
                side_effect=(b"\x00" * 32, b"\x00" * 32)):
            self.assertFalse(vanity.verify_expanded_key(vanity.TEST_PRIVATE, public))

    def test_reserved_public_bytes_are_rejected_before_crypto(self):
        with mock.patch.object(
                vanity.SODIUM, "derive_public",
                side_effect=AssertionError("reserved key reached crypto validation")):
            for first_byte in (0, 255):
                public = bytes((first_byte,)) + bytes.fromhex(vanity.TEST_PUBLIC)[1:]
                self.assertFalse(vanity.verify_expanded_key(vanity.TEST_PRIVATE, public))

    def test_final_cuda_result_rejects_reserved_public_byte(self):
        class FakeProcess:
            def __init__(self, payload):
                self.stderr = StringIO("")
                self.stdout = StringIO(json.dumps(payload) + "\n")

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        for first_byte in ("00", "ff"):
            payload = {
                "public_key": first_byte + "11" * 31,
                "private_key": "22" * 64,
                "attempts": 1,
                "elapsed_seconds": 0.01,
                "engine": "optimized",
            }
            with self.subTest(first_byte=first_byte), tempfile.TemporaryDirectory() as directory:
                executable = Path(directory) / "engine"
                executable.touch()
                process = FakeProcess(payload)
                with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(vanity.subprocess, "Popen", return_value=process), \
                        mock.patch.object(
                            vanity, "verify_expanded_key",
                            side_effect=AssertionError("reserved key reached pair validation"),
                        ):
                    with self.assertRaisesRegex(
                            RuntimeError, "failed independent CPU verification"):
                        vanity.search_cuda(
                            "", "", "",
                            watch_path=Path(directory) / "rare.jsonl",
                        )


if __name__ == "__main__":
    unittest.main()
