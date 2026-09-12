import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "meshcore_cuda_vanity"
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
CUDA_SOURCE = (ROOT / "cuda_vanity.cu").read_text(encoding="utf-8")
CI_WORKFLOW = (ROOT / ".github" / "workflows" / "test.yml").read_text(
    encoding="utf-8"
)


def make_dry_run(*assignments: str, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        ["make", "-nB", "meshcore_cuda_vanity", *assignments],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


class BuildHardeningTests(unittest.TestCase):
    def test_release_version_is_consistent_across_user_facing_files(self) -> None:
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        current_release = re.search(
            r"^The current release is \*\*v([^*]+)\*\*\.",
            readme,
            re.MULTILINE,
        )
        newest_changelog = re.search(
            r"^## ([0-9]+\.[0-9]+\.[0-9]+) — ",
            changelog,
            re.MULTILINE,
        )
        self.assertRegex(version, r"^[0-9]+\.[0-9]+\.[0-9]+$")
        self.assertIsNotNone(current_release)
        self.assertIsNotNone(newest_changelog)
        self.assertEqual(current_release.group(1), version)
        self.assertEqual(newest_changelog.group(1), version)

    def test_explicit_arch_remains_cross_buildable_without_a_gpu(self) -> None:
        output = make_dry_run("CUDA_ARCH=sm_75")
        self.assertIn("-arch=sm_75", output)
        self.assertIn('-DMC_BUILD_ARCHES=\\"sm_75\\"', output)

    def test_minimum_nvcc_version_is_checked_before_build(self) -> None:
        self.assertIn(".DELETE_ON_ERROR:", MAKEFILE)
        self.assertIn("| check-nvcc", MAKEFILE)
        with tempfile.TemporaryDirectory() as temporary:
            fake_nvcc = Path(temporary) / "nvcc"
            fake_nvcc.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Cuda compilation tools, release 11.7, V11.7.0'\n",
                encoding="utf-8",
            )
            fake_nvcc.chmod(0o755)
            rejected = subprocess.run(
                ["make", "check-nvcc", f"NVCC={fake_nvcc}"], cwd=ROOT,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            fake_nvcc.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'Cuda compilation tools, release 11.8, V11.8.0'\n",
                encoding="utf-8",
            )
            accepted = subprocess.run(
                ["make", "check-nvcc", f"NVCC={fake_nvcc}"], cwd=ROOT,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("CUDA toolkit 11.8 or newer is required", rejected.stderr)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_ci_install_smoke_preserves_cross_build_configuration(self) -> None:
        self.assertIn(
            "make install-smoke CUDA_ARCH=sm_75 HOST_CXX=/usr/bin/g++",
            CI_WORKFLOW,
        )

    def test_auto_detection_builds_every_unique_sass_and_highest_ptx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake_smi = Path(temporary) / "nvidia-smi"
            fake_smi.write_text(
                "#!/bin/sh\nprintf '%s\\n' 8.9 7.5 8.9\n",
                encoding="utf-8",
            )
            fake_smi.chmod(0o755)
            environment = os.environ.copy()
            environment.pop("CUDA_ARCH", None)
            environment["PATH"] = temporary + os.pathsep + environment["PATH"]
            output = make_dry_run(env=environment)

        self.assertIn("arch=compute_75,code=sm_75", output)
        self.assertIn("arch=compute_89,code=sm_89", output)
        self.assertIn("arch=compute_89,code=compute_89", output)
        self.assertEqual(output.count("arch=compute_75,code=sm_75"), 1)
        self.assertEqual(output.count("arch=compute_89,code=sm_89"), 1)

    def test_tuning_change_changes_build_stamp_and_fingerprint(self) -> None:
        first = make_dry_run("CUDA_ARCH=sm_75", "CUDA_THREADS=128")
        second = make_dry_run("CUDA_ARCH=sm_75", "CUDA_THREADS=256")
        pattern = re.compile(r"MC_BUILD_FINGERPRINT=\\\"([0-9a-f]{16})\\\"")
        first_id = pattern.search(first)
        second_id = pattern.search(second)
        self.assertIsNotNone(first_id)
        self.assertIsNotNone(second_id)
        self.assertNotEqual(first_id.group(1), second_id.group(1))
        self.assertIn("touch .meshcore_cuda_vanity.build-", first)
        self.assertIn(
            "$(filter-out $@,$(wildcard $(CUDA_CONFIG_STAMP_GLOB)))",
            MAKEFILE,
        )

    def test_nvcc_dependency_file_and_direct_dependencies_are_present(self) -> None:
        output = make_dry_run("CUDA_ARCH=sm_75")
        self.assertIn("-MMD -MP -MF meshcore_cuda_vanity.d", output)
        self.assertIn("-MT meshcore_cuda_vanity", output)
        self.assertIn("vendor/cuda-ed25519/*.cu", MAKEFILE)
        self.assertIn("vendor/cuda-ed25519/*.h", MAKEFILE)
        self.assertIn("generated/rare_rules_default.cuh", MAKEFILE)
        self.assertIn("-include $(CUDA_DEPFILE)", MAKEFILE)
        self.assertIn("sha256sum Makefile cuda_vanity.cu", MAKEFILE)
        self.assertIn("meshcore_cuda_vanity: Makefile cuda_vanity.cu", MAKEFILE)

    def test_generated_rare_header_has_inputs_and_a_staleness_check(self) -> None:
        self.assertIn(
            "CUDA_RARE_INPUTS := rare_rules.json rare_rules.py "
            "tools/generate_rare_rules.py",
            MAKEFILE,
        )
        self.assertIn("$(CUDA_RARE_HEADER): $(CUDA_RARE_INPUTS)", MAKEFILE)
        self.assertIn("python3 tools/generate_rare_rules.py --check", MAKEFILE)
        self.assertIn("\t@touch $@", MAKEFILE)
        output = make_dry_run("CUDA_ARCH=sm_75")
        self.assertIn("python3 tools/generate_rare_rules.py", output)

    def test_clean_does_not_remove_rare_rule_sources(self) -> None:
        completed = subprocess.run(
            ["make", "-n", "clean", "CUDA_ARCH=sm_75"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        self.assertIn("meshcore_cuda_vanity.d", completed.stdout)
        self.assertIn(".meshcore_cuda_vanity.build-*", completed.stdout)
        self.assertNotIn("rare_rules", completed.stdout)


class ProbeInterfaceTests(unittest.TestCase):
    def test_probe_is_a_real_synchronized_kernel_and_key_free(self) -> None:
        self.assertIn("readiness_probe_kernel<<<1, 1>>>", CUDA_SOURCE)
        self.assertIn("derive_lane_private_keys<<<1, kThreads>>>", CUDA_SOURCE)
        self.assertIn(
            "scan_kernel_optimized<kAttemptsPerThread><<<1, kThreads>>>",
            CUDA_SOURCE,
        )
        self.assertIn(
            "scan_kernel_baseline<kAttemptsPerThread><<<1, kThreads>>>",
            CUDA_SOURCE,
        )
        self.assertIn("production_lane_count * 64", CUDA_SOURCE)
        self.assertIn(
            "properties.multiProcessorCount * profile.blocks_per_sm", CUDA_SOURCE
        )
        self.assertIn(
            "scan_kernel_optimized<kInteractiveOptimizedAttemptsPerThread>",
            CUDA_SOURCE,
        )
        self.assertIn(
            "scan_kernel_baseline<kInteractiveBaselineAttemptsPerThread>",
            CUDA_SOURCE,
        )
        self.assertIn("increment_seed(seed, lane * AttemptCount)", CUDA_SOURCE)
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", CUDA_SOURCE)
        self.assertIn("constexpr int kInteractiveBlocksPerSm = 4", CUDA_SOURCE)
        self.assertIn(
            "resident_blocks < interactive_limit", CUDA_SOURCE,
        )
        self.assertIn("cudaDeviceSynchronize()", CUDA_SOURCE)
        self.assertIn("meshcore-cuda-probe-v2", CUDA_SOURCE)
        self.assertIn("constexpr int kProbeSchemaVersion = 1;", CUDA_SOURCE)
        self.assertIn("constexpr int kRareRulesetSchemaVersion = 2;", CUDA_SOURCE)
        self.assertIn("constexpr int kRareRuleProtocolVersion = 2;", CUDA_SOURCE)
        self.assertIn("--rare-rules-v2", CUDA_SOURCE)
        self.assertIn("--rare-rule-v2", CUDA_SOURCE)
        self.assertIn("protocol v1 is unsupported", CUDA_SOURCE)
        self.assertEqual(
            CUDA_SOURCE.count("kProbeSchemaVersion, kRareRuleProtocolVersion"),
            2,
        )
        self.assertIn("(void)cudaGetLastError();", CUDA_SOURCE)
        probe_section = CUDA_SOURCE[
            CUDA_SOURCE.index("int emit_probe_failure"):
            CUDA_SOURCE.index("int run_lane_isolation_self_test")
        ]
        self.assertNotIn('\\"public_key\\"', probe_section)
        self.assertNotIn('\\"private_key\\"', probe_section)

    @unittest.skipUnless(BINARY.is_file(), "CUDA executable has not been built")
    def test_probe_rejects_search_and_compatibility_options_before_cuda(self) -> None:
        invalid_commands = (
            ("--probe", "--prefix", "aa"),
            ("--probe", "--collect-only"),
            ("--probe", "--engine", "incremental"),
            ("--probe", "--device", "0", "--device", "1"),
            ("--probe", "--probe"),
            ("--probe", "--interactive", "--interactive"),
        )
        for command in invalid_commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    [str(BINARY), *command],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertIn("Usage:", completed.stderr)

    @unittest.skipUnless(
        os.environ.get("RUN_CUDA_TESTS") == "1",
        "set RUN_CUDA_TESTS=1 to exercise a live CUDA device",
    )
    def test_live_probe_reports_key_free_json_for_both_engines(self) -> None:
        self.assertTrue(BINARY.is_file(), "build meshcore_cuda_vanity first")
        expected_interactive_attempts = {"optimized": 2048, "baseline": 32}
        for engine in ("optimized", "baseline"):
            for interactive in (False, True):
                with self.subTest(engine=engine, interactive=interactive):
                    command = [
                        str(BINARY), "--probe", "--device", "0",
                        "--engine", engine,
                    ]
                    if interactive:
                        command.append("--interactive")
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5,
                        check=True,
                    )
                    payload = json.loads(completed.stdout)
                    self.assertTrue(payload["ready"])
                    self.assertEqual(payload["schema"], 1)
                    self.assertEqual(
                        payload["protocol"], "meshcore-cuda-probe-v2",
                    )
                    self.assertEqual(payload["engine"], engine)
                    if interactive:
                        self.assertLessEqual(payload["blocks_per_sm"], 4)
                        self.assertEqual(
                            payload["attempts_per_thread"],
                            expected_interactive_attempts[engine],
                        )
                    else:
                        self.assertEqual(payload["blocks_per_sm"], 16)
                        self.assertEqual(payload["attempts_per_thread"], 4096)
                    self.assertRegex(
                        payload["compute_capability"], r"^\d+\.\d+$",
                    )
                    if "pci_bus_id" in payload:
                        self.assertRegex(
                            payload["pci_bus_id"],
                            r"^(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
                            r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$",
                        )
                    self.assertRegex(
                        payload["build_fingerprint"], r"^[0-9a-f]{16}$",
                    )
                    self.assertNotIn("public_key", payload)
                    self.assertNotIn("private_key", payload)
                    self.assertNotIn("seed", payload)


if __name__ == "__main__":
    unittest.main()
