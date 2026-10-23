"""The ${...} expression language."""

from __future__ import annotations

import pytest

from reflex.expr import ExprError, evaluate, resolve


def test_arithmetic_logic_and_fields():
    env = {"hp": 42, "boss": None, "loot": {"count": 3, "xy": [5, 6]}, "name": "kuro"}
    assert evaluate("hp + 8", env) == 50
    assert evaluate("hp < 50 and boss is None", env) is True
    assert evaluate("boss is not None", env) is False
    assert evaluate("loot.count * 2", env) == 6
    assert evaluate("loot.xy[1]", env) == 6
    assert evaluate("min(hp, 10) if loot.count > 1 else 0", env) == 10
    assert evaluate("'ku' in name", env) is True
    assert evaluate("null", env) is None and evaluate("true", env) is True


def test_resolve_forms():
    env = {"n": 3, "pt": {"x": 1}}
    assert resolve("${n * 2}", env) == 6
    assert resolve("$pt", env) == {"x": 1}
    assert resolve("run ${n} of ${n + 1}", env) == "run 3 of 4"
    assert resolve(["${n}", {"a": "$n"}], env) == [3, {"a": 3}]
    assert resolve("plain $ text", env) == "plain $ text"


@pytest.mark.parametrize("src", [
    "__import__('os')",
    "open('x')",
    "(1).__class__",
    "pt._secret",
    "[x for x in range(3)]",
    "lambda: 1",
])
def test_sandbox_rejects_code(src):
    with pytest.raises(ExprError):
        evaluate(src, {"pt": {"_secret": 1}})


def test_unknown_variable_lists_known_ones():
    with pytest.raises(ExprError, match="unknown variable 'hpp'.*known: hp"):
        evaluate("hpp + 1", {"hp": 1})
