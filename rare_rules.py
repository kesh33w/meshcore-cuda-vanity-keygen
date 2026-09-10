"""Canonical, strictly validated rare-key rule definitions.

This module contains no key-generation or file-writing behavior.  It turns the
public JSON ruleset into one immutable representation that Python analysis and
the CUDA command protocol can share.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional


RULESET_SCHEMA_VERSION = 1
CUDA_RULE_PROTOCOL_VERSION = 1
MAX_RULES = 32
MAX_RULE_VALUE_NIBBLES = 64
MAX_CONFIG_BYTES = 256 * 1024
MIN_INDIVIDUAL_RARITY_BITS = 32.0
MIN_RULESET_RARITY_BITS = 28.0
DEFAULT_RULES_PATH = Path(__file__).resolve().with_name("rare_rules.json")

KIND_CODES = {
    "bookend": 0,
    "mirror": 1,
    "repeat-prefix": 2,
    "literal-prefix": 3,
    "sequence-prefix": 4,
}
STRUCTURAL_KINDS = frozenset(("bookend", "mirror", "repeat-prefix"))
HEX_RE = re.compile(r"[0-9a-f]+")
PUBLIC_HEX_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[a-z][a-z0-9-]{0,47}")
RULESET_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


class RuleConfigError(ValueError):
    """Raised when a ruleset is malformed, ambiguous, or unsafe."""


@dataclass(frozen=True)
class RareMatch:
    reason: str
    kind: str
    length: int
    rarity_bits: float
    mean_attempts: str


@dataclass(frozen=True)
class RareRule:
    id: str
    kind: str
    enabled: bool
    minimum_nibbles: int
    value: str = ""
    excluded_nibbles: str = ""

    @property
    def kind_code(self) -> int:
        return KIND_CODES[self.kind]

    @property
    def excluded_mask(self) -> int:
        mask = 0
        for character in self.excluded_nibbles:
            mask |= 1 << int(character, 16)
        return mask

    @property
    def alternatives(self) -> int:
        if self.kind == "repeat-prefix":
            return 16 - len(self.excluded_nibbles)
        return 1

    @property
    def threshold_length(self) -> int:
        if self.kind == "literal-prefix":
            return len(self.value)
        return self.minimum_nibbles

    @property
    def rarity_bits(self) -> float:
        return self.threshold_length * 4.0 - math.log2(self.alternatives)

    @property
    def hit_probability(self) -> float:
        return self.alternatives / (16 ** self.threshold_length)

    @property
    def trigger(self) -> str:
        if self.kind in STRUCTURAL_KINDS:
            return f"{self.kind}-{self.minimum_nibbles}"
        if self.kind == "literal-prefix":
            return f"prefix-{self.value}"
        return f"prefix-{self.id}-{self.value[:self.minimum_nibbles]}"

    def normalized(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "enabled": self.enabled,
        }
        if self.kind != "literal-prefix":
            result["minimum_nibbles"] = self.minimum_nibbles
        if self.kind in ("literal-prefix", "sequence-prefix"):
            result["value"] = self.value
        if self.kind == "repeat-prefix":
            result["excluded_nibbles"] = self.excluded_nibbles
        return result

    def cuda_specification(self) -> str:
        """Return a compact, bounded v1 argument understood by the CUDA host."""
        return (
            f"{self.kind_code}:{self.threshold_length}:"
            f"{self.excluded_mask:04x}:{self.value}"
        )


@dataclass(frozen=True)
class RareRuleset:
    schema_version: int
    ruleset_id: str
    rules: tuple[RareRule, ...]
    fingerprint: str

    @property
    def active_rules(self) -> tuple[RareRule, ...]:
        return tuple(rule for rule in self.rules if rule.enabled)

    @property
    def watch_reasons(self) -> tuple[str, ...]:
        return tuple(rule.trigger for rule in self.active_rules)

    @property
    def hit_probability_upper_bound(self) -> float:
        return sum(rule.hit_probability for rule in self.active_rules)

    @property
    def rarity_bits_lower_bound(self) -> float:
        probability = self.hit_probability_upper_bound
        return math.inf if probability == 0 else -math.log2(probability)

    def cuda_arguments(self) -> tuple[str, ...]:
        """Serialize active rules without shell quoting or unbounded fields.

        The first option carries the normalized semantic fingerprint.  Each
        following specification is a separate argv value, in rule-index order.
        """
        arguments = [f"--rare-rules-v{CUDA_RULE_PROTOCOL_VERSION}", self.fingerprint]
        for rule in self.active_rules:
            arguments.extend(("--rare-rule-v1", rule.cuda_specification()))
        return tuple(arguments)

    def classify(self, public_hex: str) -> int:
        if not isinstance(public_hex, str) or PUBLIC_HEX_RE.fullmatch(public_hex) is None:
            return -1
        for index, rule in enumerate(self.active_rules):
            if _rule_matches_threshold(rule, public_hex):
                return index
        return -1

    def analyze(self, public_hex: str) -> tuple[RareMatch, ...]:
        if not isinstance(public_hex, str) or PUBLIC_HEX_RE.fullmatch(public_hex) is None:
            return ()
        matches: list[RareMatch] = []
        for rule in self.active_rules:
            match = _analyze_rule(rule, public_hex)
            if match is not None:
                matches.append(match)
        return tuple(sorted(matches, key=lambda match: match.rarity_bits, reverse=True))

    def normalized(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ruleset_id": self.ruleset_id,
            "rules": [rule.normalized() for rule in self.rules],
        }


def _exact_keys(value: Mapping[str, object], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise RuleConfigError(f"{context} contains unsupported field(s): {', '.join(sorted(unknown))}")


def _required_string(value: Mapping[str, object], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result:
        raise RuleConfigError(f"{context}.{name} must be a non-empty string")
    return result


def _required_integer(value: Mapping[str, object], name: str, context: str) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int):
        raise RuleConfigError(f"{context}.{name} must be an integer")
    return result


def _hex_value(value: Mapping[str, object], name: str, context: str) -> str:
    result = _required_string(value, name, context)
    if HEX_RE.fullmatch(result) is None:
        raise RuleConfigError(f"{context}.{name} must use lowercase hexadecimal characters")
    if len(result) > MAX_RULE_VALUE_NIBBLES:
        raise RuleConfigError(
            f"{context}.{name} exceeds {MAX_RULE_VALUE_NIBBLES} hexadecimal characters"
        )
    return result


def _parse_rule(value: object, index: int) -> RareRule:
    context = f"rules[{index}]"
    if not isinstance(value, Mapping):
        raise RuleConfigError(f"{context} must be an object")
    common = {"id", "kind", "enabled"}
    rule_id = _required_string(value, "id", context)
    if ID_RE.fullmatch(rule_id) is None:
        raise RuleConfigError(f"{context}.id is not a valid stable rule identifier")
    kind = _required_string(value, "kind", context)
    if kind not in KIND_CODES:
        raise RuleConfigError(f"{context}.kind is unsupported")
    enabled_value = value.get("enabled", True)
    if not isinstance(enabled_value, bool):
        raise RuleConfigError(f"{context}.enabled must be boolean")

    if kind in ("bookend", "mirror"):
        _exact_keys(value, common | {"minimum_nibbles"}, context)
        minimum = _required_integer(value, "minimum_nibbles", context)
        if not 1 <= minimum <= 32:
            raise RuleConfigError(f"{context}.minimum_nibbles must be between 1 and 32")
        rule = RareRule(rule_id, kind, enabled_value, minimum)
    elif kind == "repeat-prefix":
        _exact_keys(value, common | {"minimum_nibbles", "excluded_nibbles"}, context)
        minimum = _required_integer(value, "minimum_nibbles", context)
        if not 1 <= minimum <= 64:
            raise RuleConfigError(f"{context}.minimum_nibbles must be between 1 and 64")
        excluded = _hex_value(value, "excluded_nibbles", context)
        if len(set(excluded)) != len(excluded):
            raise RuleConfigError(f"{context}.excluded_nibbles contains duplicates")
        if excluded != "".join(sorted(excluded, key=lambda item: int(item, 16))):
            raise RuleConfigError(f"{context}.excluded_nibbles must be numerically sorted")
        if not {"0", "f"}.issubset(excluded):
            raise RuleConfigError(
                f"{context}.excluded_nibbles must include MeshCore-reserved 0 and f"
            )
        if len(excluded) >= 16:
            raise RuleConfigError(f"{context}.excluded_nibbles excludes every possible nibble")
        rule = RareRule(rule_id, kind, enabled_value, minimum, excluded_nibbles=excluded)
    elif kind == "literal-prefix":
        _exact_keys(value, common | {"value"}, context)
        pattern = _hex_value(value, "value", context)
        if len(pattern) >= 2 and pattern[:2] in ("00", "ff"):
            raise RuleConfigError(f"{context}.value begins with a MeshCore-reserved byte")
        rule = RareRule(rule_id, kind, enabled_value, len(pattern), value=pattern)
    else:
        _exact_keys(value, common | {"minimum_nibbles", "value"}, context)
        minimum = _required_integer(value, "minimum_nibbles", context)
        pattern = _hex_value(value, "value", context)
        if not 1 <= minimum <= len(pattern):
            raise RuleConfigError(
                f"{context}.minimum_nibbles must be between 1 and the sequence length"
            )
        if minimum >= 2 and pattern[:2] in ("00", "ff"):
            raise RuleConfigError(f"{context}.value begins with a MeshCore-reserved byte")
        rule = RareRule(rule_id, kind, enabled_value, minimum, value=pattern)

    if rule.rarity_bits < MIN_INDIVIDUAL_RARITY_BITS:
        raise RuleConfigError(
            f"{context} is too frequent ({rule.rarity_bits:.3f} rarity bits; "
            f"minimum is {MIN_INDIVIDUAL_RARITY_BITS:.0f})"
        )
    return rule


def _canonical_bytes(schema_version: int, ruleset_id: str,
                     rules: tuple[RareRule, ...]) -> bytes:
    normalized = {
        "schema_version": schema_version,
        "ruleset_id": ruleset_id,
        "rules": [rule.normalized() for rule in rules],
    }
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def parse_ruleset(value: object) -> RareRuleset:
    if not isinstance(value, Mapping):
        raise RuleConfigError("ruleset must be a JSON object")
    _exact_keys(value, {"schema_version", "ruleset_id", "rules"}, "ruleset")
    schema_version = _required_integer(value, "schema_version", "ruleset")
    if schema_version != RULESET_SCHEMA_VERSION:
        raise RuleConfigError(
            f"unsupported ruleset schema {schema_version}; expected {RULESET_SCHEMA_VERSION}"
        )
    ruleset_id = _required_string(value, "ruleset_id", "ruleset")
    if RULESET_ID_RE.fullmatch(ruleset_id) is None:
        raise RuleConfigError("ruleset.ruleset_id is not a valid identifier")
    raw_rules = value.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuleConfigError("ruleset.rules must be a non-empty array")
    if len(raw_rules) > MAX_RULES:
        raise RuleConfigError(f"ruleset contains more than {MAX_RULES} rules")
    rules = tuple(_parse_rule(raw_rule, index) for index, raw_rule in enumerate(raw_rules))

    ids = [rule.id for rule in rules]
    if len(set(ids)) != len(ids):
        raise RuleConfigError("rule identifiers must be unique")
    structural = [rule.kind for rule in rules if rule.kind in STRUCTURAL_KINDS]
    if len(set(structural)) != len(structural):
        raise RuleConfigError("bookend, mirror, and repeat-prefix may each appear only once")
    active = tuple(rule for rule in rules if rule.enabled)
    if not active:
        raise RuleConfigError("ruleset must contain at least one enabled rule")
    triggers = [rule.trigger for rule in active]
    if len(set(triggers)) != len(triggers):
        raise RuleConfigError("enabled rules must have unique trigger names")

    probability = sum(rule.hit_probability for rule in active)
    rarity_lower_bound = -math.log2(probability)
    if rarity_lower_bound < MIN_RULESET_RARITY_BITS:
        raise RuleConfigError(
            f"combined rules are too frequent ({rarity_lower_bound:.3f} rarity bits; "
            f"minimum is {MIN_RULESET_RARITY_BITS:.0f})"
        )

    fingerprint = hashlib.sha256(
        _canonical_bytes(schema_version, ruleset_id, rules)
    ).hexdigest()
    return RareRuleset(schema_version, ruleset_id, rules, fingerprint)


def load_ruleset(path: Optional[Path] = None) -> RareRuleset:
    source = DEFAULT_RULES_PATH if path is None else Path(path)
    try:
        with source.open("rb") as file:
            raw = file.read(MAX_CONFIG_BYTES + 1)
    except OSError as error:
        raise RuleConfigError(f"could not read rare rules file: {source}") from error
    if len(raw) > MAX_CONFIG_BYTES:
        raise RuleConfigError(f"rare rules file exceeds {MAX_CONFIG_BYTES} bytes")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, item in pairs:
            if name in result:
                raise RuleConfigError(f"rare rules JSON contains duplicate field: {name}")
            result[name] = item
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuleConfigError("rare rules file is not valid UTF-8 JSON") from error
    return parse_ruleset(value)


def select_rules(ruleset: RareRuleset, enabled_ids: Iterable[str]) -> RareRuleset:
    """Return a revalidated immutable ruleset with exactly these rules enabled."""
    requested = tuple(enabled_ids)
    if not requested:
        raise RuleConfigError("select at least one rare-key rule")
    if any(not isinstance(rule_id, str) or not rule_id for rule_id in requested):
        raise RuleConfigError("selected rule identifiers must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise RuleConfigError("selected rule identifiers must be unique")
    configured = {rule.id for rule in ruleset.rules}
    unknown = set(requested) - configured
    if unknown:
        raise RuleConfigError(
            f"unknown selected rule(s): {', '.join(sorted(unknown))}"
        )
    selected = set(requested)
    document = ruleset.normalized()
    raw_rules = document["rules"]
    assert isinstance(raw_rules, list)
    for raw_rule in raw_rules:
        assert isinstance(raw_rule, dict)
        raw_rule["enabled"] = raw_rule["id"] in selected
    return parse_ruleset(document)


def _rule_matches_threshold(rule: RareRule, public_hex: str) -> bool:
    length = rule.threshold_length
    if rule.kind == "bookend":
        return public_hex[:length] == public_hex[-length:]
    if rule.kind == "mirror":
        return public_hex[:length] == public_hex[-length:][::-1]
    if rule.kind == "repeat-prefix":
        return public_hex[0] not in rule.excluded_nibbles and public_hex.startswith(
            public_hex[0] * length
        )
    if rule.kind == "literal-prefix":
        return public_hex.startswith(rule.value)
    return public_hex.startswith(rule.value[:length])


def _make_match(reason: str, kind: str, length: int, alternatives: int) -> RareMatch:
    attempts = (16 ** length + alternatives - 1) // alternatives
    bits = length * 4.0 - math.log2(alternatives)
    return RareMatch(reason, kind, length, round(bits, 3), str(attempts))


def _analyze_rule(rule: RareRule, public_hex: str) -> Optional[RareMatch]:
    if not _rule_matches_threshold(rule, public_hex):
        return None
    if rule.kind == "bookend":
        lengths = [
            length for length in range(rule.minimum_nibbles, 33)
            if public_hex[:length] == public_hex[-length:]
        ]
        length = max(lengths)
        return _make_match(f"bookend-{length}", "bookend", length, 1)
    if rule.kind == "mirror":
        length = 0
        for index in range(32):
            if public_hex[index] != public_hex[63 - index]:
                break
            length += 1
        return _make_match(f"mirror-{length}", "mirror", length, 1)
    if rule.kind == "repeat-prefix":
        length = 1
        while length < 64 and public_hex[length] == public_hex[0]:
            length += 1
        return _make_match(
            f"repeat-prefix-{length}", "repeat-prefix", length, rule.alternatives
        )
    if rule.kind == "literal-prefix":
        return _make_match(
            f"prefix-{rule.value}", "phrase-prefix", len(rule.value), 1
        )

    length = 0
    while (length < len(rule.value) and length < 64
           and public_hex[length] == rule.value[length]):
        length += 1
    match_kind = "pi-prefix" if rule.id == "pi" else "sequence-prefix"
    return _make_match(
        f"prefix-{rule.id}-{rule.value[:length]}", match_kind, length, 1
    )


def classify(public_hex: str, ruleset: Optional[RareRuleset] = None) -> int:
    return (DEFAULT_RULESET if ruleset is None else ruleset).classify(public_hex)


def analyze(public_hex: str, ruleset: Optional[RareRuleset] = None
            ) -> tuple[RareMatch, ...]:
    return (DEFAULT_RULESET if ruleset is None else ruleset).analyze(public_hex)


DEFAULT_RULESET = load_ruleset()
WATCH_REASONS = DEFAULT_RULESET.watch_reasons


__all__ = (
    "CUDA_RULE_PROTOCOL_VERSION",
    "DEFAULT_RULESET",
    "DEFAULT_RULES_PATH",
    "KIND_CODES",
    "MAX_RULES",
    "MIN_INDIVIDUAL_RARITY_BITS",
    "MIN_RULESET_RARITY_BITS",
    "RareMatch",
    "RareRule",
    "RareRuleset",
    "RuleConfigError",
    "WATCH_REASONS",
    "analyze",
    "classify",
    "load_ruleset",
    "parse_ruleset",
    "select_rules",
)
