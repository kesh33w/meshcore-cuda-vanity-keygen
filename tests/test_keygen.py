import contextlib
import inspect
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
import rare_rules


class KeygenTests(unittest.TestCase):
    @staticmethod
    def custom_ruleset() -> rare_rules.RareRuleset:
        return rare_rules.parse_ruleset({
            "schema_version": 1,
            "ruleset_id": "test-custom",
            "rules": [{
                "id": "custom-prefix",
                "kind": "literal-prefix",
                "enabled": True,
                "value": "abcdef1234",
            }],
        })

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
        self.assertEqual(vanity.interesting_rule("1337133713" + "1" * 54), 3)
        self.assertEqual(vanity.interesting_rule("3141592653" + "1" * 54), 4)
        for removed in ("cafecafe00", "beefbeef00", "deadbeef00",
                        "facebabe00", "babecafe00", "f00df00d00",
                        "fadefade00"):
            self.assertEqual(vanity.interesting_rule(removed + "1" * 54), -1)
        self.assertEqual(vanity.interesting_rule("abcdef0123" + "1" * 44 + "1337133713"), -1)
        self.assertEqual(vanity.interesting_rule("1" * 64), 0)
        self.assertEqual(vanity.interesting_rule("not hex"), -1)
        self.assertEqual(len(vanity.WATCH_REASONS), 5)
        self.assertFalse(any(reason.startswith("suffix-") for reason in vanity.WATCH_REASONS))

    def test_rare_rule_compatibility_api_uses_canonical_types_and_defaults(self):
        self.assertIs(vanity.RareMatch, rare_rules.RareMatch)
        self.assertIs(vanity.DEFAULT_RARE_RULESET, rare_rules.DEFAULT_RULESET)
        self.assertEqual(vanity.WATCH_REASONS, rare_rules.DEFAULT_RULESET.watch_reasons)
        self.assertEqual(
            vanity.diagnostics()["rare_ruleset_fingerprint"],
            rare_rules.DEFAULT_RULESET.fingerprint,
        )

    def test_gui_rule_selection_settings_roundtrip_and_fail_safe(self):
        base = rare_rules.DEFAULT_RULESET
        selected = rare_rules.select_rules(base, ("mirror", "pi"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            vanity.save_gui_rule_selection(base, selected, path)
            loaded = vanity.load_gui_rule_selection(base, path)
            self.assertEqual(loaded.fingerprint, selected.fingerprint)
            self.assertEqual(vanity.active_rule_ids(loaded), ("mirror", "pi"))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            document = json.loads(path.read_text(encoding="utf-8"))
            document["base_ruleset_fingerprint"] = "0" * 64
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertIs(vanity.load_gui_rule_selection(base, path), base)

            path.write_text("not json", encoding="utf-8")
            self.assertIs(vanity.load_gui_rule_selection(base, path), base)

            path.write_bytes(b" " * (vanity.GUI_SETTINGS_MAX_BYTES + 1))
            self.assertIs(vanity.load_gui_rule_selection(base, path), base)

            path.write_text(
                '{"schema_version":1,"schema_version":1,\n'
                '"base_ruleset_fingerprint":"unused","enabled_rule_ids":["pi"]}',
                encoding="utf-8",
            )
            self.assertIs(vanity.load_gui_rule_selection(base, path), base)

            target = Path(directory) / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = Path(directory) / "settings-link.json"
            link.symlink_to(target)
            self.assertIs(vanity.load_gui_rule_selection(base, link), base)
            with self.assertRaises(ValueError):
                vanity.save_gui_rule_selection(base, selected, link)

            fifo = Path(directory) / "settings-fifo"
            os.mkfifo(fifo)
            loaded_fifo: list[rare_rules.RareRuleset] = []
            reader = threading.Thread(
                target=lambda: loaded_fifo.append(
                    vanity.load_gui_rule_selection(base, fifo)
                ),
                daemon=True,
            )
            reader.start()
            reader.join(1.0)
            self.assertFalse(reader.is_alive(), "FIFO preference path blocked GUI startup")
            self.assertEqual(loaded_fifo, [base])

            with mock.patch.object(vanity.json, "loads", side_effect=RecursionError):
                path.write_text("{}", encoding="utf-8")
                self.assertIs(vanity.load_gui_rule_selection(base, path), base)

    def test_gui_rule_labels_and_summary_are_human_readable(self):
        base = rare_rules.DEFAULT_RULESET
        labels = [vanity.rare_rule_choice_label(rule) for rule in base.rules]
        self.assertTrue(any("Bookends" in label for label in labels))
        self.assertTrue(any("Mirrors" in label for label in labels))
        self.assertTrue(any("1337133713" in label for label in labels))
        self.assertTrue(any("Pi" in label for label in labels))
        custom_literal = rare_rules.RareRule(
            "bookend", "literal-prefix", True, 0, "abcdef1234",
        )
        self.assertTrue(
            vanity.rare_rule_choice_label(custom_literal).startswith("bookend —"),
        )
        self.assertEqual(vanity.rare_rule_selection_summary(base), "All 5 selected")
        subset = rare_rules.select_rules(base, ("mirror", "pi"))
        self.assertEqual(
            vanity.rare_rule_selection_summary(subset),
            "2 of 5 selected: Mirrors, Pi",
        )

    def test_cuda_rare_rules_match_python_rules(self):
        root = Path(vanity.__file__).resolve().parent
        cuda_source = (root / "cuda_vanity.cu").read_text(encoding="utf-8")
        self.assertIn('#include "generated/rare_rules_default.cuh"', cuda_source)
        self.assertIn("classify_generated_default", cuda_source)
        self.assertIn("classify_generic_rules", cuda_source)
        self.assertNotIn("gpu_watch_words", cuda_source)
        for removed in ("cafecafe00", "beefbeef00", "deadbeef00",
                        "facebabe00", "babecafe00", "f00df00d00",
                        "fadefade00"):
            self.assertNotIn(f'"{removed}"', cuda_source)

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

    def test_gui_cuda_search_uses_explicit_interactive_profile(self):
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
                    mock.patch.object(
                        vanity.subprocess, "Popen", side_effect=lambda *_args, **_kwargs: FakeProcess(),
                    ) as popen:
                for interactive in (False, True):
                    with self.subTest(interactive=interactive), self.assertRaisesRegex(
                            RuntimeError, "invalid result"):
                        vanity.search_cuda(
                            "cafe", "", "", interactive=interactive,
                            watch_path=Path(directory) / f"rare-{interactive}.jsonl",
                        )
                    command = popen.call_args.args[0]
                    self.assertEqual("--interactive" in command, interactive)

        gui_source = inspect.getsource(vanity.run_gui)
        self.assertIn("interactive=True", gui_source)
        self.assertIn('page_status = tk.StringVar(value="Page 1 of 1")', gui_source)

    def test_cuda_collector_termination_is_reported_as_cancellation(self):
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
                    mock.patch.object(vanity.subprocess, "Popen",
                                      return_value=TerminatedProcess()) as popen:
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda(
                        "", "", "", cancel=cancel, collect_only=True,
                        watch_path=Path(directory) / "rare.jsonl",
                    )
            self.assertIn("--collect-only", popen.call_args.args[0])

    def test_cuda_collector_rejects_vanity_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            with mock.patch.object(vanity, "cuda_executable", return_value=executable):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    vanity.search_cuda(
                        "cafe", "", "", collect_only=True,
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
                self.assertEqual(first["schema_version"], vanity.RARE_LOG_SCHEMA)
                self.assertEqual(first["trigger"], "bookend-10")
                self.assertEqual(
                    first["active_rule_ids"],
                    [rule.id for rule in vanity.DEFAULT_RARE_RULESET.active_rules],
                )
                self.assertEqual(second["match_length"], 10)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([record["reason"] for record in records], ["bookend-10"] * 2)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_custom_rule_is_carried_into_cuda_argv_and_saved_metadata(self):
        class FinishedCollector:
            def __init__(self):
                self.stderr = StringIO("")
                self.stdout = StringIO("")

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        ruleset = self.custom_ruleset()
        public = "abcdef1234" + "1" * 54
        private = "22" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "engine"
            executable.touch()
            watch_path = root / "rare.jsonl"
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(
                        vanity.subprocess, "Popen", return_value=FinishedCollector()
                    ) as popen:
                with self.assertRaisesRegex(RuntimeError, "stopped unexpectedly"):
                    vanity.search_cuda(
                        "", "", "", collect_only=True, watch_path=watch_path,
                        ruleset=ruleset,
                    )
            command = popen.call_args.args[0]
            offset = command.index("--rare-rules-v1")
            self.assertEqual(
                tuple(command[offset:offset + len(ruleset.cuda_arguments())]),
                ruleset.cuda_arguments(),
            )

            with mock.patch.object(vanity, "verify_expanded_key", return_value=True):
                record = vanity.append_interesting(
                    watch_path, 0, public, private, ruleset=ruleset,
                )
            self.assertEqual(record["trigger"], "prefix-abcdef1234")
            self.assertEqual(record["ruleset_id"], ruleset.ruleset_id)
            self.assertEqual(record["ruleset_fingerprint"], ruleset.fingerprint)
            self.assertEqual(record["active_rule_ids"], ["custom-prefix"])

    def test_selected_rule_watch_index_is_compact_and_saved(self):
        ruleset = rare_rules.select_rules(
            rare_rules.DEFAULT_RULESET, ("pi",),
        )
        public = "3141592653" + "1" * 54
        private = "22" * 64
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(vanity, "verify_expanded_key", return_value=True):
            record = vanity.append_interesting(
                Path(directory) / "rare.jsonl", 0, public, private,
                ruleset=ruleset,
            )
        self.assertEqual(record["trigger"], "prefix-pi-3141592653")
        self.assertEqual(record["active_rule_ids"], ["pi"])
        self.assertEqual(record["ruleset_fingerprint"], ruleset.fingerprint)

    def test_schema_v2_and_newer_history_preserves_stored_analysis(self):
        stored_matches = [{
            "reason": "prefix-fadefade00",
            "kind": "phrase-prefix",
            "length": 10,
            "rarity_bits": 40.0,
            "mean_attempts": str(16 ** 10),
        }]
        for schema_version in (2, 3, 4):
            record = {
                "schema_version": schema_version,
                "reason": "prefix-fadefade00",
                "match_length": 10,
                "rarity_bits": 40.0,
                "mean_attempts": str(16 ** 10),
                "matches": stored_matches,
                # This key matches stronger current structural rules, proving
                # normalization does not overwrite persisted v2+ analysis.
                "public_key": "1" * 64,
                "private_key": "2" * 128,
            }
            with self.subTest(schema_version=schema_version):
                normalized = vanity.normalize_interesting_record(record)
                self.assertIsNotNone(normalized)
                assert normalized is not None
                self.assertEqual(normalized["reason"], "prefix-fadefade00")
                self.assertEqual(normalized["matches"], stored_matches)

            damaged_summary = {
                **record,
                "match_length": -1,
                "rarity_bits": -2.0,
                "mean_attempts": None,
            }
            normalized = vanity.normalize_interesting_record(damaged_summary)
            self.assertIsNotNone(normalized)
            assert normalized is not None
            self.assertEqual(normalized["match_length"], 10)
            self.assertEqual(normalized["rarity_bits"], 40.0)
            self.assertEqual(normalized["mean_attempts"], str(16 ** 10))

    def test_cli_rejects_invalid_rare_rules_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-rules.json"
            path.write_text(
                json.dumps({
                    "schema_version": 99,
                    "ruleset_id": "invalid",
                    "rules": [],
                }),
                encoding="utf-8",
            )
            error_output = StringIO()
            with mock.patch("sys.argv", [
                    "meshcore_vanity.py", "--diagnostics", "--rare-rules", str(path),
            ]), contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as raised:
                vanity.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unsupported ruleset schema", error_output.getvalue())
        self.assertNotIn("Traceback", error_output.getvalue())

    def test_cli_loads_custom_rules_once_and_passes_frozen_instance(self):
        ruleset = self.custom_ruleset()
        output = StringIO()
        with mock.patch("sys.argv", [
                "meshcore_vanity.py", "--collect-rare", "--backend", "cuda",
                "--rare-rules", "/tmp/test-rules.json",
        ]), mock.patch.object(vanity, "load_ruleset", return_value=ruleset) as loader, \
                mock.patch.object(vanity, "cuda_available", return_value=True), \
                mock.patch.object(vanity, "search_cuda", side_effect=KeyboardInterrupt) as search, \
                contextlib.redirect_stdout(output):
            status = vanity.main()
        self.assertEqual(status, 0)
        loader.assert_called_once_with(Path("/tmp/test-rules.json"))
        self.assertIs(search.call_args.kwargs["ruleset"], ruleset)

    def test_gui_custom_rules_are_session_only(self):
        ruleset = self.custom_ruleset()
        with mock.patch("sys.argv", [
                "meshcore_vanity.py", "--gui",
                "--rare-rules", "/tmp/test-rules.json",
        ]), mock.patch.object(vanity, "load_ruleset", return_value=ruleset), \
                mock.patch.object(vanity, "run_gui", return_value=0) as gui:
            status = vanity.main()
        self.assertEqual(status, 0)
        gui.assert_called_once_with(ruleset, persist_rule_selection=False)

    def test_installer_includes_canonical_rule_sources(self):
        install_script = (Path(vanity.__file__).resolve().parent / "install.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$PROJECT_DIR/rare_rules.py" "$APP_HOME/rare_rules.py"', install_script)
        self.assertIn('"$PROJECT_DIR/rare_rules.json" "$APP_HOME/rare_rules.json"', install_script)

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
                                        engine="optimized", interactive=True)
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
                                        engine="baseline", interactive=True)
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
    def test_cuda_collector_can_be_cancelled_without_orphaning_process(self):
        if not vanity.cuda_available():
            self.skipTest("CUDA engine/device unavailable")
        cancel = threading.Event()
        child = []
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda("", "", "", cancel=cancel, collect_only=True,
                                       watch_path=Path(directory) / "rare.jsonl",
                                       interactive=True,
                                       process_update=lambda process: child.append(process))
        finally:
            timer.cancel()
        processes = [process for process in child if process is not None]
        self.assertTrue(processes)
        self.assertIsNotNone(processes[0].poll())


if __name__ == "__main__":
    unittest.main()
