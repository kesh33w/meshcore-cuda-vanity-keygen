import copy
from contextlib import redirect_stderr
from io import StringIO
import json
import os
import stat
from pathlib import Path
import subprocess
import tempfile
import unittest

import rare_rules
from tools import generate_rare_rules


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "meshcore_cuda_vanity"


REMOVED_PREFIXES = (
    "cafecafe00",
    "beefbeef00",
    "deadbeef00",
    "facebabe00",
    "babecafe00",
    "f00df00d00",
    "fadefade00",
)


def literal_rule(rule_id: str, value: str, enabled: bool = True) -> dict[str, object]:
    return {
        "id": rule_id,
        "kind": "literal-prefix",
        "enabled": enabled,
        "value": value,
    }


def document_with(*rules: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ruleset_id": "test-rules",
        "rules": list(rules),
    }


class RareRulesTests(unittest.TestCase):
    def test_default_indices_triggers_and_removed_categories(self):
        ruleset = rare_rules.DEFAULT_RULESET
        self.assertEqual(
            ruleset.watch_reasons,
            (
                "bookend-10",
                "mirror-10",
                "repeat-prefix-10",
                "prefix-1337133713",
                "prefix-pi-3141592653",
            ),
        )
        middle = "1" * 44
        examples = (
            "abcde12345" + middle + "abcde12345",
            "abcde12345" + middle + "54321edcba",
            "b" * 10 + "1234567890" * 5 + "1234",
            "1337133713" + "1" * 54,
            "3141592653" + "1" * 54,
        )
        for expected, public_hex in enumerate(examples):
            self.assertEqual(ruleset.classify(public_hex), expected)
        for removed in REMOVED_PREFIXES:
            self.assertEqual(ruleset.classify(removed + "1" * 54), -1)
            self.assertNotIn(removed, json.dumps(ruleset.normalized()))

    def test_default_analysis_preserves_stronger_and_overlapping_matches(self):
        repeat_key = "a" * 13 + "1234567890abcdef" * 3 + "123"
        repeat = rare_rules.analyze(repeat_key)[0]
        self.assertEqual((repeat.reason, repeat.kind, repeat.length),
                         ("repeat-prefix-13", "repeat-prefix", 13))
        self.assertAlmostEqual(repeat.rarity_bits, 13 * 4 - 3.807, places=3)

        pi_key = "314159265358979" + "a" * 49
        pi = rare_rules.analyze(pi_key)[0]
        self.assertEqual(
            (pi.reason, pi.kind, pi.length, pi.rarity_bits),
            ("prefix-pi-314159265358979", "pi-prefix", 15, 60.0),
        )

        matches = rare_rules.analyze("1" * 64)
        self.assertEqual(matches[0].reason, "repeat-prefix-64")
        self.assertEqual(
            {match.kind for match in matches}, {"bookend", "mirror", "repeat-prefix"}
        )

    def test_default_rarity_limits_are_conservative(self):
        ruleset = rare_rules.DEFAULT_RULESET
        self.assertGreaterEqual(
            min(rule.rarity_bits for rule in ruleset.rules),
            rare_rules.MIN_INDIVIDUAL_RARITY_BITS,
        )
        self.assertGreaterEqual(
            ruleset.rarity_bits_lower_bound, rare_rules.MIN_RULESET_RARITY_BITS
        )
        self.assertAlmostEqual(ruleset.rarity_bits_lower_bound, 35.82, places=2)

    def test_custom_rules_classify_and_serialize_in_active_order(self):
        ruleset = rare_rules.parse_ruleset(document_with(
            literal_rule("disabled-rule", "abcdef1234", enabled=False),
            literal_rule("first", "123456789a"),
            {
                "id": "repeat-prefix",
                "kind": "repeat-prefix",
                "enabled": True,
                "minimum_nibbles": 10,
                "excluded_nibbles": "0f",
            },
        ))
        self.assertEqual(ruleset.classify("123456789a" + "0" * 54), 0)
        self.assertEqual(ruleset.classify("c" * 10 + "1" * 54), 1)
        self.assertEqual(
            ruleset.cuda_arguments(),
            (
                "--rare-rules-v2",
                ruleset.fingerprint,
                "--rare-rule-v2",
                "3:10:0000:123456789a",
                "--rare-rule-v2",
                "2:10:8001:",
            ),
        )
        self.assertLessEqual(max(map(len, ruleset.cuda_arguments())), 80)

    def test_selection_preserves_order_indices_and_default_fingerprint(self):
        default = rare_rules.DEFAULT_RULESET
        selected = rare_rules.select_rules(default, ("pi", "mirror"))
        self.assertEqual(
            tuple(rule.id for rule in selected.active_rules), ("mirror", "pi"),
        )
        self.assertEqual(
            selected.watch_reasons,
            ("mirror-10", "prefix-pi-3141592653"),
        )
        self.assertEqual(
            selected.classify("abcde12345" + "2" * 44 + "54321edcba"), 0,
        )
        self.assertEqual(selected.classify("3141592653" + "2" * 54), 1)

        all_ids = tuple(rule.id for rule in default.rules)
        all_selected = rare_rules.select_rules(default, reversed(all_ids))
        restored = rare_rules.select_rules(selected, all_ids)
        self.assertEqual(all_selected.fingerprint, default.fingerprint)
        self.assertEqual(restored.fingerprint, default.fingerprint)
        self.assertEqual(all_selected.cuda_arguments(), default.cuda_arguments())

    def test_selection_rejects_empty_duplicate_unknown_and_unsafe_sets(self):
        default = rare_rules.DEFAULT_RULESET
        for selected in ((), ("pi", "pi"), ("not-configured",)):
            with self.subTest(selected=selected), self.assertRaises(
                    rare_rules.RuleConfigError):
                rare_rules.select_rules(default, selected)

        rules = tuple(
            literal_rule(f"rule-{index}", f"a{index:07x}", enabled=index == 0)
            for index in range(17)
        )
        base = rare_rules.parse_ruleset(document_with(*rules))
        enabled_two = rare_rules.select_rules(base, ("rule-0", "rule-1"))
        self.assertEqual(len(enabled_two.active_rules), 2)
        with self.assertRaisesRegex(rare_rules.RuleConfigError, "combined rules"):
            rare_rules.select_rules(base, (rule.id for rule in base.rules))

    def test_minimum_configuration_preserves_default_and_raises_thresholds(self):
        default = rare_rules.DEFAULT_RULESET
        all_ids = tuple(rule.id for rule in default.rules)

        unchanged = rare_rules.configure_rules(default, all_ids, 10)
        self.assertEqual(unchanged.fingerprint, default.fingerprint)
        self.assertEqual(unchanged.cuda_arguments(), default.cuda_arguments())

        eleven = rare_rules.configure_rules(default, all_ids, 11)
        twelve = rare_rules.configure_rules(default, all_ids, 12)
        for configured, minimum in ((eleven, 11), (twelve, 12)):
            with self.subTest(minimum=minimum):
                self.assertEqual(
                    tuple(rule.id for rule in configured.active_rules),
                    ("bookend", "mirror", "repeat-prefix", "pi"),
                )
                self.assertEqual(
                    tuple(rule.threshold_length for rule in configured.active_rules),
                    (minimum,) * 4,
                )
                self.assertNotEqual(configured.fingerprint, default.fingerprint)
                self.assertNotIn("prefix-1337133713", configured.watch_reasons)

        # Configuring a derivative always starts from its own immutable base.
        self.assertEqual(
            tuple(rule.threshold_length for rule in default.active_rules),
            (10, 10, 10, 10, 10),
        )
        self.assertEqual(default.fingerprint, unchanged.fingerprint)

        # Bookend widths are not nested: a 12-character match need not match at
        # width 10 or 11, but every configured minimum through 12 must find it.
        non_nested_bookend = "abcdef123456" + "2" * 40 + "abcdef123456"
        self.assertNotEqual(non_nested_bookend[:10], non_nested_bookend[-10:])
        self.assertNotEqual(non_nested_bookend[:11], non_nested_bookend[-11:])
        for minimum in (10, 11, 12):
            configured = rare_rules.configure_rules(default, all_ids, minimum)
            with self.subTest(bookend_minimum=minimum):
                self.assertEqual(configured.classify(non_nested_bookend), 0)
                self.assertEqual(
                    configured.analyze(non_nested_bookend)[0].reason, "bookend-12"
                )

    def test_minimum_configuration_classifies_only_at_effective_threshold(self):
        configured = rare_rules.configure_rules(
            rare_rules.DEFAULT_RULESET,
            tuple(rule.id for rule in rare_rules.DEFAULT_RULESET.rules),
            12,
        )
        misses = (
            "abcdef12345" + "2" * 42 + "abcdef12345",
            "abcdef12345" + "1" + "2" * 40 + "3" + "54321fedcba",
            "b" * 11 + "2" * 53,
            "1337133713" + "2" * 54,
            "31415926535" + "2" * 53,
        )
        for public_hex in misses:
            with self.subTest(public_hex=public_hex):
                self.assertEqual(configured.classify(public_hex), -1)

        hits = (
            ("abcdef123456" + "2" * 40 + "abcdef123456", "bookend-12"),
            ("abcdef123456" + "1" + "2" * 38 + "3" + "654321fedcba",
             "mirror-12"),
            ("b" * 12 + "2" * 52, "repeat-prefix-12"),
            ("314159265358" + "2" * 52, "prefix-pi-314159265358"),
        )
        for expected_index, (public_hex, reason) in enumerate(hits):
            with self.subTest(reason=reason):
                self.assertEqual(configured.classify(public_hex), expected_index)
                self.assertEqual(configured.analyze(public_hex)[0].reason, reason)

    def test_minimum_configuration_keeps_stricter_custom_rule_thresholds(self):
        base = rare_rules.parse_ruleset(document_with(
            {
                "id": "bookend", "kind": "bookend", "enabled": True,
                "minimum_nibbles": 14,
            },
            {
                "id": "sequence", "kind": "sequence-prefix", "enabled": True,
                "minimum_nibbles": 15,
                "value": "271828182845904523536028747135266249775724709369995",
            },
            literal_rule("fixed", "abcdef123456"),
        ))
        original = copy.deepcopy(base.normalized())

        configured = rare_rules.configure_rules(
            base, ("bookend", "sequence", "fixed"), 11,
        )
        self.assertEqual(
            tuple(rule.threshold_length for rule in configured.active_rules),
            (14, 15, 12),
        )
        self.assertEqual(base.normalized(), original)

        raised = rare_rules.configure_rules(
            base, ("bookend", "sequence", "fixed"), 13,
        )
        self.assertEqual(
            tuple(rule.id for rule in raised.active_rules),
            ("bookend", "sequence"),
        )
        self.assertEqual(
            tuple(rule.threshold_length for rule in raised.active_rules),
            (14, 15),
        )

    def test_minimum_eligibility_and_configuration_validation(self):
        rules = {rule.id: rule for rule in rare_rules.DEFAULT_RULESET.rules}
        for minimum in (10, 11, 12):
            with self.subTest(minimum=minimum):
                self.assertTrue(
                    rare_rules.rule_can_meet_minimum(rules["bookend"], minimum)
                )
                self.assertTrue(
                    rare_rules.rule_can_meet_minimum(rules["mirror"], minimum)
                )
                self.assertTrue(
                    rare_rules.rule_can_meet_minimum(rules["repeat-prefix"], minimum)
                )
                self.assertEqual(
                    rare_rules.rule_can_meet_minimum(
                        rules["prefix-1337133713"], minimum,
                    ),
                    minimum == 10,
                )
                self.assertTrue(rare_rules.rule_can_meet_minimum(rules["pi"], minimum))

        default = rare_rules.DEFAULT_RULESET
        invalid_selections = (
            (),
            ("pi", "pi"),
            ("not-configured",),
            ("",),
        )
        for selected in invalid_selections:
            with self.subTest(selected=selected), self.assertRaises(
                    rare_rules.RuleConfigError):
                rare_rules.configure_rules(default, selected, 10)

        with self.assertRaisesRegex(rare_rules.RuleConfigError, "none of the selected"):
            rare_rules.configure_rules(default, ("prefix-1337133713",), 11)
        for invalid_minimum in (True, 0, 65, 10.0):
            with self.subTest(minimum=invalid_minimum), self.assertRaises(
                    rare_rules.RuleConfigError):
                rare_rules.configure_rules(default, ("pi",), invalid_minimum)

    def test_fingerprint_is_semantic_deterministic_and_order_sensitive(self):
        first = literal_rule("first", "123456789a")
        second = literal_rule("second", "abcdef1234")
        document = document_with(first, second)
        reordered_keys = {
            "rules": [dict(reversed(tuple(item.items()))) for item in document["rules"]],
            "ruleset_id": document["ruleset_id"],
            "schema_version": document["schema_version"],
        }
        a = rare_rules.parse_ruleset(document)
        b = rare_rules.parse_ruleset(reordered_keys)
        c = rare_rules.parse_ruleset(document_with(second, first))
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.fingerprint, c.fingerprint)
        self.assertRegex(a.fingerprint, r"[0-9a-f]{64}\Z")

    def test_schema_one_is_migrated_to_semantic_schema_two(self):
        legacy = document_with(literal_rule("legacy", "12345678"))
        migrated = rare_rules.parse_ruleset(legacy)
        semantic = copy.deepcopy(legacy)
        semantic["schema_version"] = rare_rules.RULESET_SCHEMA_VERSION
        current = rare_rules.parse_ruleset(semantic)
        self.assertEqual(migrated.schema_version, 2)
        self.assertEqual(migrated.normalized(), current.normalized())
        self.assertEqual(migrated.fingerprint, current.fingerprint)

    def test_strict_schema_rejects_invalid_documents(self):
        valid = document_with(literal_rule("valid", "12345678"))
        invalid_documents: list[object] = []

        unknown_root = copy.deepcopy(valid)
        unknown_root["extra"] = True
        invalid_documents.append(unknown_root)

        wrong_schema = copy.deepcopy(valid)
        wrong_schema["schema_version"] = 3
        invalid_documents.append(wrong_schema)

        invalid_documents.extend((
            document_with(
                literal_rule("duplicate", "12345678"),
                literal_rule("duplicate", "abcdef12"),
            ),
            document_with({"id": "bad", "kind": "unknown", "enabled": True}),
            document_with(literal_rule("uppercase", "ABCDEF12")),
            document_with(literal_rule("too-short", "1234567")),
            document_with({
                "id": "repeat-prefix", "kind": "repeat-prefix", "enabled": True,
                "minimum_nibbles": 10, "excluded_nibbles": "0",
            }),
            document_with({
                "id": "sequence", "kind": "sequence-prefix", "enabled": True,
                "minimum_nibbles": 11, "value": "123456789a",
            }),
            document_with({
                "id": "bookend", "kind": "bookend", "enabled": True,
                "minimum_nibbles": 33,
            }),
        ))
        too_many = document_with(*(
            literal_rule(f"rule-{index}", f"a{index:07x}") for index in range(33)
        ))
        invalid_documents.append(too_many)
        too_frequent_together = document_with(*(
            literal_rule(f"rule-{index}", f"a{index:07x}") for index in range(17)
        ))
        invalid_documents.append(too_frequent_together)

        for index, document in enumerate(invalid_documents):
            with self.subTest(index=index), self.assertRaises(rare_rules.RuleConfigError):
                rare_rules.parse_ruleset(document)

    def test_generated_header_is_current_and_contains_same_fingerprint(self):
        root = Path(rare_rules.__file__).resolve().parent
        header_path = root / "generated" / "rare_rules_default.cuh"
        expected = generate_rare_rules.render_header(rare_rules.DEFAULT_RULESET)
        self.assertEqual(header_path.read_text(encoding="utf-8"), expected)
        self.assertIn(rare_rules.DEFAULT_RULESET.fingerprint, expected)
        self.assertIn("constexpr unsigned int kRuleCount = 5U;", expected)
        for trigger in rare_rules.DEFAULT_RULESET.watch_reasons:
            self.assertIn(f'"{trigger}"', expected)
        self.assertIn("__device__ __forceinline__ int classify_generated_default", expected)
        cuda_source = (root / "cuda_vanity.cu").read_text(encoding="utf-8")
        self.assertIn("meshcore_rare_generated::classify_generated_default", cuda_source)
        self.assertNotIn("gpu_watch_words", cuda_source)
        self.assertNotIn("gpu_pi_prefix", cuda_source)

    def test_generator_check_and_write_if_changed(self):
        expected = generate_rare_rules.render_header(rare_rules.DEFAULT_RULESET)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rules.cuh"
            self.assertTrue(generate_rare_rules.write_if_changed(output, expected))
            first_stat = output.stat()
            self.assertEqual(stat.S_IMODE(first_stat.st_mode), 0o644)
            self.assertFalse(generate_rare_rules.write_if_changed(output, expected))
            self.assertEqual(output.stat().st_mtime_ns, first_stat.st_mtime_ns)
            self.assertEqual(
                generate_rare_rules.main(["--output", str(output), "--check"]), 0
            )
            output.write_text("stale\n", encoding="utf-8")
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    generate_rare_rules.main(["--output", str(output), "--check"]), 1
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "stale\n")

    def test_load_ruleset_rejects_oversized_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.json"
            malformed.write_text("not JSON", encoding="utf-8")
            with self.assertRaises(rare_rules.RuleConfigError):
                rare_rules.load_ruleset(malformed)
            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * (rare_rules.MAX_CONFIG_BYTES + 1))
            with self.assertRaises(rare_rules.RuleConfigError):
                rare_rules.load_ruleset(oversized)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":1,"schema_version":1,'
                '"ruleset_id":"duplicate","rules":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(rare_rules.RuleConfigError, "duplicate field"):
                rare_rules.load_ruleset(duplicate)


@unittest.skipUnless(BINARY.is_file(), "CUDA executable has not been built")
class NativeCudaRareProtocolTests(unittest.TestCase):
    def run_parser(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BINARY), "--internal-test-rare-rules", *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )

    def test_exact_generated_default_is_the_only_fast_path(self):
        default_arguments = rare_rules.DEFAULT_RULESET.cuda_arguments()
        exact = self.run_parser(*default_arguments)
        self.assertEqual(exact.returncode, 0, exact.stderr)
        exact_payload = json.loads(exact.stdout)
        self.assertEqual(exact_payload["rules"], 5)
        self.assertTrue(exact_payload["default_fast_path"])

        altered = list(default_arguments)
        altered[altered.index("3:10:0000:1337133713")] = "3:10:0000:123456789a"
        mismatch = self.run_parser(*altered)
        self.assertEqual(mismatch.returncode, 0, mismatch.stderr)
        self.assertFalse(json.loads(mismatch.stdout)["default_fast_path"])

    def test_host_parser_accepts_32_ordered_rules(self):
        arguments = ["--rare-rules-v2", "1" * 64]
        for index in range(32):
            arguments.extend((
                "--rare-rule-v2", f"3:16:0000:1{index:015x}",
            ))
        completed = self.run_parser(*arguments)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["rules"], 32)
        self.assertFalse(payload["default_fast_path"])

    def test_protocol_v1_is_explicitly_rejected(self):
        completed = self.run_parser(
            "--rare-rules-v1", "1" * 64,
            "--rare-rule-v1", "3:10:0000:123456789a",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("protocol v1 is unsupported", completed.stderr)

    def test_malformed_protocol_is_rejected_before_cuda(self):
        fingerprint = "a" * 64
        valid = "3:10:0000:123456789a"
        invalid_commands = (
            ("--rare-rules-v2", fingerprint),
            ("--rare-rules-v2", fingerprint.upper(), "--rare-rule-v2", valid),
            ("--rare-rule-v2", valid),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "03:10:0000:123456789a"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:010:0000:123456789a"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:10:0000:123456789A"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:10:0000:123456789a:extra"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "0:33:0000:"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "2:10:0001:"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:9:0000:123456789a"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:10:0000:003456789a"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", "3:7:0000:1234567"),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", valid,
             "--collect-only", "--rare-rule-v2", valid),
            ("--rare-rules-v2", fingerprint, "--rare-rule-v2", valid,
             "--rare-rules-v2", fingerprint),
        )
        too_many = ["--rare-rules-v2", fingerprint]
        for index in range(33):
            too_many.extend(("--rare-rule-v2", f"3:16:0000:1{index:015x}"))

        for arguments in (*invalid_commands, tuple(too_many)):
            with self.subTest(arguments=arguments):
                completed = self.run_parser(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertIn("Invalid rare-rule protocol:", completed.stderr)

    def test_probe_source_reports_rule_protocol_and_default_fingerprint(self):
        source = (ROOT / "cuda_vanity.cu").read_text(encoding="utf-8")
        self.assertIn('"\\\"rare_rule_protocol\\\":%d', source)
        self.assertIn('\\\"default_ruleset_fingerprint\\\":\\\"%s', source)

    @unittest.skipUnless(
        os.environ.get("RUN_CUDA_TESTS") == "1",
        "CUDA classifier parity test is opt-in",
    )
    def test_gpu_default_and_custom_classifier_parity(self):
        default_samples = (
            "abcde12345" + "1" * 44 + "abcde12345",
            "abcde12345" + "2" * 44 + "54321edcba",
            "b" * 10 + "23456789abcdef" * 3 + "23456789abcd",
            "1337133713" + "2" * 54,
            "3141592653" + "2" * 54,
            "123456789a" + "2" * 54,
            "1" * 64,
        )
        custom = rare_rules.parse_ruleset(document_with(
            {"id": "bookend", "kind": "bookend", "enabled": True,
             "minimum_nibbles": 12},
            {"id": "mirror", "kind": "mirror", "enabled": True,
             "minimum_nibbles": 11},
            {"id": "repeat", "kind": "repeat-prefix", "enabled": True,
             "minimum_nibbles": 11, "excluded_nibbles": "0f"},
            literal_rule("literal", "abcdef1234"),
            {"id": "sequence", "kind": "sequence-prefix", "enabled": True,
             "minimum_nibbles": 10,
             "value": "271828182845904523536028747135266249775724709369995"},
        ))
        custom_samples = (
            "123456789abc" + "2" * 40 + "123456789abc",
            "abcdef12345" + "2" * 42 + "54321fedcba",
            "b" * 11 + "2" * 53,
            "abcdef1234" + "2" * 54,
            "2718281828" + "2" * 54,
            "123456789a" + "2" * 54,
            "1" * 64,
        )

        pi_only = rare_rules.select_rules(rare_rules.DEFAULT_RULESET, ("pi",))
        mirror_and_pi = rare_rules.select_rules(
            rare_rules.DEFAULT_RULESET, ("pi", "mirror"),
        )
        all_default_ids = tuple(
            rule.id for rule in rare_rules.DEFAULT_RULESET.rules
        )
        minimum_eleven = rare_rules.configure_rules(
            rare_rules.DEFAULT_RULESET, all_default_ids, 11,
        )
        minimum_twelve = rare_rules.configure_rules(
            rare_rules.DEFAULT_RULESET, all_default_ids, 12,
        )
        threshold_samples = (
            # Exercise the rolling matcher's odd-offset branch at width 11.
            "abcdef12345" + "2" * 42 + "abcdef12345",
            # A 12-wide bookend that deliberately fails at widths 10 and 11.
            "abcdef123456" + "2" * 40 + "abcdef123456",
            "abcdef123456" + "1" + "2" * 38 + "3" + "654321fedcba",
            "b" * 12 + "2" * 52,
            "314159265358" + "2" * 52,
            "1337133713" + "2" * 54,
            "31415926535" + "2" * 53,
            "123456789a" + "2" * 54,
        )
        for ruleset, samples, expect_fast in (
                (rare_rules.DEFAULT_RULESET, default_samples, True),
                (pi_only, default_samples, False),
                (mirror_and_pi, default_samples, False),
                (custom, custom_samples, False),
                (minimum_eleven, threshold_samples, False),
                (minimum_twelve, threshold_samples, False)):
            arguments: list[str] = []
            for sample in samples:
                arguments.extend(("--internal-test-rare-classifier", sample))
            arguments.extend(ruleset.cuda_arguments())
            completed = subprocess.run(
                [str(BINARY), *arguments], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["indices"], [ruleset.classify(sample) for sample in samples]
            )
            self.assertEqual(payload["default_fast_path"], expect_fast)
            self.assertTrue(payload["default_generic_parity"])
            self.assertNotIn("private", completed.stdout)
            self.assertNotIn("public", completed.stdout)


if __name__ == "__main__":
    unittest.main()
