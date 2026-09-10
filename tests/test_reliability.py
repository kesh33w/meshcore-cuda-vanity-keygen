import contextlib
import fcntl
import json
import subprocess
import tempfile
import threading
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import meshcore_vanity as vanity


def rare_record(index: int) -> dict[str, object]:
    return {
        "found_at": f"2026-09-08T00:{index // 60:02d}:{index % 60:02d}Z",
        "reason": "bookend-10",
        "public_key": vanity.TEST_PUBLIC,
        "private_key": "11" * 64,
    }


class ReliabilityTests(unittest.TestCase):
    def test_gui_mainloop_keyboard_interrupt_uses_cleanup_path(self):
        closed: list[bool] = []

        def interrupted_mainloop():
            raise KeyboardInterrupt

        status = vanity._run_gui_mainloop(
            interrupted_mainloop, lambda: closed.append(True),
        )
        self.assertEqual(status, 130)
        self.assertEqual(closed, [True])

    def test_gui_diagnostics_launcher_does_not_block_caller(self):
        entered = threading.Event()
        release = threading.Event()
        delivered: list[tuple[object, object, int]] = []

        def slow_diagnostics(_ruleset):
            entered.set()
            self.assertTrue(release.wait(2.0))
            return {"cuda_devices": 0, "cuda_engine_probes": {}}

        def receive(result, error):
            delivered.append((result, error, threading.get_ident()))

        caller_thread = threading.get_ident()
        with mock.patch.object(vanity, "gui_diagnostics", side_effect=slow_diagnostics):
            started = time.monotonic()
            worker = vanity.start_gui_diagnostics(receive)
            elapsed = time.monotonic() - started
            self.assertTrue(entered.wait(1.0))
            self.assertLess(elapsed, 0.25)
            self.assertTrue(worker.is_alive())
            release.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(delivered[0][1], None)
        self.assertNotEqual(delivered[0][2], caller_thread)

    def test_gui_diagnostics_preloads_both_engines_for_every_device(self):
        optimized = [
            {"device": index, "engine": "optimized", "ready": True}
            for index in range(2)
        ]
        base = {
            "cuda_devices": 2, "cuda_names": ["first", "second"],
            "cuda_probes": optimized, "cuda_ready": True,
            "firmware_vector": True,
        }

        def probe(device, engine):
            return {"device": device, "engine": engine, "ready": True}

        with mock.patch.object(vanity, "diagnostics", return_value=base), \
                mock.patch.object(vanity, "cuda_probe", side_effect=probe) as cuda_probe:
            discovered = vanity.gui_diagnostics()

        self.assertEqual(
            set(discovered["cuda_engine_probes"]),
            {(0, "optimized"), (0, "baseline"),
             (1, "optimized"), (1, "baseline")},
        )
        self.assertEqual(
            cuda_probe.call_args_list,
            [mock.call(0, "baseline"), mock.call(1, "baseline")],
        )

    def test_cpu_worker_failure_wakes_coordinator_and_propagates(self):
        started = time.monotonic()
        with mock.patch.object(
                vanity, "meshcore_keypair", side_effect=RuntimeError("worker exploded")):
            with self.assertRaisesRegex(vanity.SearchWorkerError, "worker exploded"):
                vanity.search("abcdef012345", "", "", workers=2)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_tail_loader_returns_newest_without_scanning_large_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            old_oversized_line = b"x" * (32 * 1024) + b"\n"
            recent = b"".join(
                json.dumps(rare_record(index)).encode() + b"\n"
                for index in range(8)
            )
            path.write_bytes(old_oversized_line + recent + b'{"incomplete":')
            progress: list[tuple[int, int]] = []
            loaded, skipped = vanity.load_interesting_records(
                path, limit=3, block_size=512,
                progress=lambda scanned, total: progress.append((scanned, total)),
            )

        self.assertEqual(
            [record["found_at"] for record in loaded],
            [rare_record(index)["found_at"] for index in range(5, 8)],
        )
        self.assertEqual(skipped, 1)
        self.assertTrue(progress)
        self.assertLess(progress[-1][0], progress[-1][1])

    def test_tail_loader_skips_oversized_crash_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            path.write_bytes(
                json.dumps(rare_record(1)).encode() + b"\n" + b"z" * 4096
            )
            loaded, skipped = vanity.load_interesting_records(
                path, limit=1, block_size=128, max_record_bytes=512,
            )
        self.assertEqual([record["found_at"] for record in loaded],
                         [rare_record(1)["found_at"]])
        self.assertEqual(skipped, 1)

    def test_tail_loader_skips_pathological_json_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            valid = json.dumps(rare_record(1)).encode() + b"\n"
            pathological = b'{"value":' + b"9" * 5_000 + b"}\n"
            path.write_bytes(valid + pathological)
            loaded, skipped = vanity.load_interesting_records(path, limit=2)
        self.assertEqual([record["found_at"] for record in loaded],
                         [rare_record(1)["found_at"]])
        self.assertEqual(skipped, 1)

    def test_tail_loader_cancellation_stops_before_full_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            path.write_bytes(b"not-json\n" * 10_000)
            cancel = threading.Event()
            progress: list[tuple[int, int]] = []

            def stop_after_first_block(scanned, total):
                progress.append((scanned, total))
                cancel.set()

            loaded, _skipped = vanity.load_interesting_records(
                path, limit=10_000, block_size=128,
                progress=stop_after_first_block, cancel=cancel,
            )
        self.assertEqual(loaded, [])
        self.assertTrue(progress)
        self.assertLess(progress[-1][0], progress[-1][1])

    def test_count_uses_unlocked_stable_size_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rare.jsonl"
            original_count = 10_000
            path.write_bytes(b"record\n" * original_count)
            entered = threading.Event()
            release = threading.Event()
            result: list[int] = []

            def pause_after_first_block(_scanned, _total):
                if not entered.is_set():
                    entered.set()
                    self.assertTrue(release.wait(2.0))

            worker = threading.Thread(target=lambda: result.append(
                vanity.count_interesting(
                    path, progress=pause_after_first_block, block_size=128,
                )
            ))
            worker.start()
            self.assertTrue(entered.wait(1.0))
            try:
                with path.open("ab") as writer:
                    fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    writer.write(b"late-record\n")
                    writer.flush()
                    fcntl.flock(writer.fileno(), fcntl.LOCK_UN)
            finally:
                release.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [original_count])

    def test_stale_initial_count_cannot_overwrite_completed_search_state(self):
        state: dict[str, object] = {
            "rare_count": None, "rare_count_generation": 0,
        }
        startup_generation = 0
        state["rare_count_generation"] = 1
        state["rare_count"] = 7

        accepted = vanity.apply_rare_count_snapshot(
            state, startup_generation, 6,
        )

        self.assertFalse(accepted)
        self.assertEqual(state["rare_count"], 7)

    def test_page_selection_caps_rendering_but_keeps_all_matches_reachable(self):
        records = [
            {**rare_record(index), "public_key": f"{index:064x}"}
            for index in range(1_201)
        ]
        first, matching, page, pages = vanity.select_interesting_page(
            records, "", "found", False, 0,
        )
        last, last_matching, last_page, last_pages = vanity.select_interesting_page(
            records, "", "found", False, 2,
        )
        filtered, filtered_count, _, _ = vanity.select_interesting_page(
            records, "00000000000000000000000000000000000000000000000000000000000004b0",
            "public", False, 0,
        )
        self.assertEqual((len(first), matching, page, pages), (500, 1_201, 0, 3))
        self.assertEqual((len(last), last_matching, last_page, last_pages), (201, 1_201, 2, 3))
        self.assertEqual((len(filtered), filtered_count), (1, 1))

    def test_malformed_legacy_display_metadata_is_sanitized(self):
        record = {
            **rare_record(1),
            "reason": "suffix-deadbeef00",
            "match_length": "not-a-number",
            "rarity_bits": {"bad": True},
            "mean_attempts": ["bad"],
            "matches": "not-a-list",
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["match_length"], 10)
        self.assertEqual(normalized["rarity_bits"], 40.0)
        self.assertEqual(normalized["mean_attempts"], str(16 ** 10))
        self.assertEqual(normalized["matches"], [])

    def test_current_rule_reclassification_sets_mean_attempts(self):
        record = {
            **rare_record(1),
            "public_key": "1" * 64,
            "reason": "legacy-rule",
            "mean_attempts": "1",
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        primary = vanity.interesting_matches("1" * 64)[0]
        self.assertEqual(normalized["reason"], primary.reason)
        self.assertEqual(normalized["mean_attempts"], primary.mean_attempts)

    def test_history_filters_malformed_matches_but_preserves_valid_analysis(self):
        valid_match = {
            "reason": "retired-rule", "kind": "literal-prefix", "length": 10,
            "rarity_bits": 40.0, "mean_attempts": str(16 ** 10),
        }
        record = {
            **rare_record(1), "schema_version": 3,
            "reason": "damaged", "match_length": None,
            "rarity_bits": float("nan"), "mean_attempts": object(),
            "matches": [{"reason": "broken"}, valid_match],
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["reason"], "retired-rule")
        self.assertEqual(normalized["match_length"], 10)
        self.assertEqual(normalized["rarity_bits"], 40.0)
        self.assertEqual(normalized["mean_attempts"], str(16 ** 10))
        self.assertEqual(normalized["matches"], [valid_match])

    def test_stored_primary_overrides_valid_but_contradictory_summary(self):
        valid_match = {
            "reason": "retired-rule", "kind": "literal-prefix", "length": 10,
            "rarity_bits": 40.0, "mean_attempts": str(16 ** 10),
        }
        record = {
            **rare_record(1), "schema_version": 3,
            "reason": "different-rule", "match_length": 12,
            "rarity_bits": 48.0, "mean_attempts": str(16 ** 12),
            "matches": [valid_match],
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["reason"], "retired-rule")
        self.assertEqual(normalized["match_length"], 10)
        self.assertEqual(normalized["rarity_bits"], 40.0)
        self.assertEqual(normalized["mean_attempts"], str(16 ** 10))

    def test_history_sanitizes_integer_too_large_for_float(self):
        valid_match = {
            "reason": "retired-rule", "kind": "literal-prefix", "length": 10,
            "rarity_bits": 40.0, "mean_attempts": str(16 ** 10),
        }
        record = {
            **rare_record(1), "schema_version": 3, "matches": [valid_match],
            "rarity_bits": 10 ** 400,
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["rarity_bits"], 40.0)

    def test_history_sanitizes_and_filters_active_rule_ids(self):
        valid_match = {
            "reason": "retired-rule", "kind": "literal-prefix", "length": 10,
            "rarity_bits": 40.0, "mean_attempts": str(16 ** 10),
        }
        record = {
            **rare_record(1), "schema_version": 4, "matches": [valid_match],
            "active_rule_ids": ["mirror", "pi"],
        }
        normalized = vanity.normalize_interesting_record(record)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["active_rule_ids"], ["mirror", "pi"])
        selected, matching, _, _ = vanity.select_interesting_page(
            [normalized], "pi", "found", False, 0,
        )
        self.assertEqual((selected, matching), ([normalized], 1))

        malformed = vanity.normalize_interesting_record({
            **record, "active_rule_ids": ["pi", "pi"],
        })
        self.assertIsNotNone(malformed)
        assert malformed is not None
        self.assertEqual(malformed["active_rule_ids"], [])

    def test_numeric_history_sort_is_defensive(self):
        records = [
            {"public_key": "a", "match_length": object(), "rarity_bits": None},
            {"public_key": "b", "match_length": 12, "rarity_bits": 48.0},
            {"public_key": "c", "match_length": "huge", "rarity_bits": float("inf")},
        ]
        by_length, _, _, _ = vanity.select_interesting_page(
            records, "", "length", False, 0,
        )
        by_rarity, _, _, _ = vanity.select_interesting_page(
            records, "", "rarity", False, 0,
        )
        self.assertEqual([item["public_key"] for item in by_length], ["a", "c", "b"])
        self.assertEqual([item["public_key"] for item in by_rarity], ["a", "c", "b"])

    def test_process_reaper_escalates_and_waits(self):
        class StubbornProcess:
            def __init__(self):
                self.terminated = False
                self.killed = False
                self.waits = 0

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.waits += 1
                if timeout is not None:
                    raise subprocess.TimeoutExpired("fake", timeout)
                return -9

        process = StubbornProcess()
        vanity.terminate_process(process, timeout=0.01)  # type: ignore[arg-type]
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.waits, 2)

    def test_nonblocking_termination_ignores_concurrent_process_exit(self):
        class VanishedProcess:
            def poll(self):
                return None

            def terminate(self):
                raise ProcessLookupError

        process = VanishedProcess()
        self.assertFalse(
            vanity.signal_process_termination(process)  # type: ignore[arg-type]
        )

    def test_cuda_cancelled_before_spawn_does_not_launch_child(self):
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "Popen") as popen:
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda(
                        "abcd", "", "", cancel=cancel,
                        watch_path=Path(directory) / "rare.jsonl",
                    )
        popen.assert_not_called()

    def test_cuda_cancel_during_spawn_registration_reaps_child(self):
        class SpawnedProcess:
            def __init__(self):
                self.stdout = StringIO("")
                self.stderr = StringIO("")
                self.return_code = None
                self.terminated = False

            def poll(self):
                return self.return_code

            def terminate(self):
                self.terminated = True
                self.return_code = -15

            def wait(self, timeout=None):
                return self.return_code

        cancel = threading.Event()
        process = SpawnedProcess()
        updates: list[object] = []

        def register(value):
            updates.append(value)
            if value is process:
                cancel.set()

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            with mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                    mock.patch.object(vanity.subprocess, "Popen", return_value=process):
                with self.assertRaises(vanity.SearchCancelled):
                    vanity.search_cuda(
                        "abcd", "", "", cancel=cancel,
                        watch_path=Path(directory) / "rare.jsonl",
                        process_update=register,
                    )

        self.assertTrue(process.terminated)
        self.assertEqual(updates, [process, None])

    def test_cuda_result_protocol_rejects_malformed_typed_payloads(self):
        valid = {
            "public_key": vanity.TEST_PUBLIC,
            "private_key": vanity.TEST_PRIVATE.hex(),
            "engine": "optimized",
            "attempts": 1,
            "elapsed_seconds": 0.1,
        }
        malformed = (
            None,
            {**valid, "extra": "field"},
            {**valid, "attempts": True},
            {**valid, "attempts": -1},
            {**valid, "attempts": 1.0},
            {**valid, "elapsed_seconds": True},
            {**valid, "elapsed_seconds": -1.0},
            {**valid, "elapsed_seconds": float("nan")},
            {**valid, "elapsed_seconds": 10 ** 400},
            {**valid, "engine": "baseline"},
            {**valid, "public_key": vanity.TEST_PUBLIC.upper()},
        )

        class FinishedProcess:
            def __init__(self, payload):
                self.stderr = StringIO("")
                self.stdout = StringIO(json.dumps(payload) + "\n")

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "engine"
            executable.touch()
            for index, payload in enumerate(malformed):
                with self.subTest(index=index), \
                        mock.patch.object(vanity, "cuda_executable", return_value=executable), \
                        mock.patch.object(
                            vanity.subprocess, "Popen",
                            return_value=FinishedProcess(payload),
                        ):
                    with self.assertRaisesRegex(RuntimeError, "invalid result"):
                        vanity.search_cuda(
                            vanity.TEST_PUBLIC[:4], "", "",
                            watch_path=Path(directory) / "rare.jsonl",
                        )

    def test_cpu_cli_keyboard_interrupt_is_clean(self):
        output = StringIO()
        error_output = StringIO()
        argv = ["meshcore_vanity.py", "--prefix", "abcd", "--backend", "cpu"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(vanity, "search", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            status = vanity.main()
        self.assertEqual(status, 130)
        self.assertIn("Search cancelled.", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())

    def test_diagnostics_cli_keyboard_interrupt_is_clean(self):
        output = StringIO()
        error_output = StringIO()
        argv = ["meshcore_vanity.py", "--diagnostics"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(vanity, "diagnostics", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            status = vanity.main()
        self.assertEqual(status, 130)
        self.assertIn("Diagnostics cancelled.", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())

    def test_cuda_cli_keyboard_interrupt_is_clean(self):
        output = StringIO()
        error_output = StringIO()
        argv = ["meshcore_vanity.py", "--prefix", "abcd", "--backend", "cuda"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(vanity, "cuda_available", return_value=True), \
                mock.patch.object(vanity, "search_cuda", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            status = vanity.main()
        self.assertEqual(status, 130)
        self.assertIn("Search cancelled.", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())

    def test_cuda_cli_preflight_keyboard_interrupt_is_clean(self):
        for argv in (
                ["meshcore_vanity.py", "--prefix", "abcd", "--backend", "cuda"],
                ["meshcore_vanity.py", "--collect-rare", "--backend", "cuda"],
        ):
            output = StringIO()
            error_output = StringIO()
            with self.subTest(argv=argv), mock.patch("sys.argv", argv), \
                    mock.patch.object(
                        vanity, "cuda_available", side_effect=KeyboardInterrupt,
                    ), contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(error_output):
                status = vanity.main()
            self.assertEqual(status, 130)
            self.assertIn("Search cancelled.", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())

        output = StringIO()
        error_output = StringIO()
        argv = ["meshcore_vanity.py", "--prefix", "abcd", "--backend", "cuda"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(vanity, "cuda_available", return_value=False), \
                mock.patch.object(vanity, "cuda_probe", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            status = vanity.main()
        self.assertEqual(status, 130)
        self.assertIn("Search cancelled.", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())

    def test_collector_cli_keyboard_interrupt_is_clean(self):
        output = StringIO()
        error_output = StringIO()
        argv = ["meshcore_vanity.py", "--collect-rare", "--backend", "cuda"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(vanity, "cuda_available", return_value=True), \
                mock.patch.object(vanity, "search_cuda", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            status = vanity.main()
        self.assertEqual(status, 0)
        self.assertIn("Collector stopped.", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue() + error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
