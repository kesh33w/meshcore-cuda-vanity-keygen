import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import meshcore_vanity as vanity


def successful_probe(device: int = 0, engine: str = "optimized", *,
                     interactive: bool = False) -> dict[str, object]:
    return {
        "schema": 1,
        "protocol": vanity.CUDA_PROBE_PROTOCOL,
        "ready": True,
        "device": device,
        "device_name": "Test GPU",
        "pci_bus_id": "00000000:01:00.0",
        "compute_capability": "8.9",
        "engine": engine,
        "build_fingerprint": "0123456789abcdef",
        "build_arches": "sm_89",
        "threads": 128,
        "blocks_per_sm": 4 if interactive else 16,
        "attempts_per_thread": (
            vanity.INTERACTIVE_CUDA_ATTEMPTS[engine]
            if interactive else 4096
        ),
        "max_registers": 128,
        "rare_rule_protocol": vanity.CUDA_RULE_PROTOCOL_VERSION,
        "default_ruleset_fingerprint": "a" * 64,
    }


class PythonCudaProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        with vanity._CUDA_PROBE_LOCK:
            vanity._CUDA_PROBE_CACHE.clear()

    def tearDown(self) -> None:
        with vanity._CUDA_PROBE_LOCK:
            vanity._CUDA_PROBE_CACHE.clear()

    def test_probe_validates_response_and_caches_by_binary_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(successful_probe()), stderr="",
            )
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", return_value=completed) as run:
                first = vanity.cuda_probe()
                second = vanity.cuda_probe()

        self.assertTrue(first["ready"])
        self.assertEqual(first, second)
        run.assert_called_once()

    def test_interactive_probe_is_negotiated_and_cached_separately(self) -> None:
        def run(command, **_kwargs):
            interactive = "--interactive" in command
            return subprocess.CompletedProcess(
                command, 0,
                stdout=json.dumps(successful_probe(interactive=interactive)),
                stderr="",
            )

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", side_effect=run) as invoked:
                throughput = vanity.cuda_probe()
                interactive = vanity.cuda_probe(interactive=True)
                self.assertEqual(throughput, vanity.cuda_probe())
                self.assertEqual(
                    interactive, vanity.cuda_probe(interactive=True),
                )

        self.assertTrue(throughput["ready"])
        self.assertTrue(interactive["ready"])
        self.assertEqual(invoked.call_count, 2)
        self.assertNotIn("--interactive", invoked.call_args_list[0].args[0])
        self.assertIn("--interactive", invoked.call_args_list[1].args[0])

    def test_interactive_probe_rejects_legacy_or_unsafe_geometry(self) -> None:
        invalid = (
            {**successful_probe(interactive=True),
             "protocol": "meshcore-cuda-probe-v1"},
            {**successful_probe(interactive=True), "blocks_per_sm": 16},
            {**successful_probe(interactive=True), "attempts_per_thread": 4096},
        )
        for response in invalid:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                executable = Path(directory) / "engine"
                executable.touch(mode=0o700)
                completed = subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(response), stderr="",
                )
                with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(vanity.subprocess, "run", return_value=completed):
                    result = vanity.cuda_probe(interactive=True)
            self.assertFalse(result["ready"])

    def test_probe_rejects_unexpected_or_key_bearing_fields(self) -> None:
        response = successful_probe()
        response["private_key"] = "not accepted"
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(response), stderr="",
            )
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", return_value=completed):
                result = vanity.cuda_probe()

        self.assertFalse(result["ready"])
        self.assertNotIn("private_key", result)
        self.assertIn("unsupported fields", str(result["error"]))

    def test_probe_requires_valid_rare_rule_compatibility_metadata(self) -> None:
        invalid_values = (
            ("rare_rule_protocol", None),
            ("rare_rule_protocol", 1),
            ("rare_rule_protocol", 3),
            ("rare_rule_protocol", True),
            ("default_ruleset_fingerprint", None),
            ("default_ruleset_fingerprint", "A" * 64),
            ("default_ruleset_fingerprint", "a" * 63),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                executable = Path(directory) / "engine"
                executable.touch(mode=0o700)
                response = successful_probe()
                response[field] = value
                completed = subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(response), stderr="",
                )
                with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(vanity.subprocess, "run", return_value=completed):
                    result = vanity.cuda_probe()
            self.assertFalse(result["ready"])
            self.assertIn("rare-rule compatibility", str(result["error"]))

    def test_probe_accepts_optional_and_normalizes_present_pci_bus_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            valid = successful_probe()
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(valid), stderr="",
            )
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", return_value=completed):
                result = vanity.cuda_probe()
        self.assertEqual(result["pci_bus_id"], "0000:01:00.0")

        without_bus = successful_probe()
        del without_bus["pci_bus_id"]
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(without_bus), stderr="",
            )
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", return_value=completed):
                ready_without_bus = vanity.cuda_probe()
        self.assertTrue(ready_without_bus["ready"])
        self.assertNotIn("pci_bus_id", ready_without_bus)

        for malformed in (None, "01:00.0", "not-a-bus", "00010000:01:00.0"):
            with self.subTest(pci_bus_id=malformed), tempfile.TemporaryDirectory() as directory:
                executable = Path(directory) / "engine"
                executable.touch(mode=0o700)
                response = successful_probe()
                response["pci_bus_id"] = malformed
                completed = subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(response), stderr="",
                )
                with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(vanity.subprocess, "run", return_value=completed):
                    rejected = vanity.cuda_probe()
            self.assertFalse(rejected["ready"])
            self.assertIn("readiness details", str(rejected["error"]))

    def test_probe_accepts_previous_protocol_without_optional_bus_id(self) -> None:
        response = successful_probe()
        response["protocol"] = "meshcore-cuda-probe-v1"
        del response["pci_bus_id"]
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            completed = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(response), stderr="",
            )
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "run", return_value=completed):
                result = vanity.cuda_probe()
        self.assertTrue(result["ready"])
        self.assertEqual(result["protocol"], "meshcore-cuda-probe-v1")

    def test_probe_requires_strict_integer_schema_and_device(self) -> None:
        invalid_values = (
            (0, "schema", True),
            (0, "schema", 1.0),
            (0, "device", False),
            (1, "device", True),
            (1, "device", 1.0),
        )
        for requested_device, field, value in invalid_values:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                executable = Path(directory) / "engine"
                executable.touch(mode=0o700)
                response = successful_probe(device=requested_device)
                response[field] = value
                completed = subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(response), stderr="",
                )
                with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(vanity.subprocess, "run", return_value=completed):
                    result = vanity.cuda_probe(requested_device)
            self.assertFalse(result["ready"])
            self.assertIn("requested protocol", str(result["error"]))

    def test_transient_probe_failure_is_not_cached(self) -> None:
        failure = successful_probe()
        failure["ready"] = False
        failure["error"] = "driver is temporarily unavailable"
        first = subprocess.CompletedProcess(
            [], 2, stdout=json.dumps(failure), stderr="",
        )
        second = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(successful_probe()), stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch(mode=0o700)
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(
                        vanity.subprocess, "run", side_effect=(first, second),
                    ) as run:
                failed = vanity.cuda_probe()
                recovered = vanity.cuda_probe()

        self.assertFalse(failed["ready"])
        self.assertTrue(recovered["ready"])
        self.assertEqual(run.call_count, 2)

    def test_cuda_available_requires_device_and_successful_probe(self) -> None:
        with mock.patch.object(vanity, "cuda_device_count", return_value=1), \
                mock.patch.object(vanity, "cuda_probe", return_value={"ready": False}) as probe:
            self.assertFalse(vanity.cuda_available(0, "optimized"))
            self.assertFalse(vanity.cuda_available(1, "optimized"))
        probe.assert_called_once_with(
            0, "optimized", interactive=False, refresh=False,
        )

    def test_diagnostics_reports_native_probe_result(self) -> None:
        response = successful_probe()
        with mock.patch.object(vanity, "cuda_device_count", return_value=1), \
                mock.patch.object(vanity, "cuda_device_names", return_value=["Test GPU"]), \
                mock.patch.object(vanity, "cuda_probe", return_value=response):
            details = vanity.diagnostics()
        self.assertTrue(details["cuda_ready"])
        self.assertEqual(details["cuda_probes"], [response])


if __name__ == "__main__":
    unittest.main()
