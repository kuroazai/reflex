"""A small, safe expression language for script variables.

    ${runs + 1}            arithmetic
    ${hp < 30 and not boss} comparisons and logic
    ${loot.count}          fields of a stored match (dicts read with dots)
    ${pos[0] + 10}         indexing
    ${min(a, b)}           a few pure functions: min max abs round int float str len

Evaluated by walking Python's AST with a whitelist: no attribute access on
real objects, no calls except the listed functions, no imports, no dunder
names. A macro file can never run arbitrary code.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Mapping
from typing import Any


class ExprError(ValueError):
    pass


_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float, "str": str,
          "len": len, "bool": bool}



def evaluate(src: str, variables: Mapping[str, Any]) -> Any:
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"bad expression {src!r}: {exc.msg}") from None
    return _eval(tree.body, variables, src)


def _eval(node: ast.AST, env: Mapping[str, Any], src: str) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("true", "True"):
            return True
        if node.id in ("false", "False"):
            return False
        if node.id in ("null", "None"):
            return None
        if node.id not in env:
            raise ExprError(f"unknown variable {node.id!r} in {src!r} (known: {', '.join(sorted(env)) or 'none'})")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ExprError(f"{node.attr!r} is not readable in {src!r}")
        base = _eval(node.value, env, src)
        if isinstance(base, Mapping):
            if node.attr not in base:
                raise ExprError(f"{node.attr!r} not found in {src!r} (has: {', '.join(map(str, base))})")
            return base[node.attr]
        raise ExprError(f"cannot read .{node.attr} of {type(base).__name__} in {src!r}")
    if isinstance(node, ast.Subscript):
        base = _eval(node.value, env, src)
        key = _eval(node.slice, env, src)
        try:
            return base[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExprError(f"{src!r}: {exc!r}") from None
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        try:
            return _BIN[type(node.op)](_eval(node.left, env, src), _eval(node.right, env, src))
        except (TypeError, ZeroDivisionError) as exc:
            raise ExprError(f"{src!r}: {exc}") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env, src))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            val: Any = True
            for v in node.values:
                val = _eval(v, env, src)
                if not val:
                    return val
            return val
        val = False
        for v in node.values:
            val = _eval(v, env, src)
            if val:
                return val
        return val
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env, src)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            right = _eval(comp, env, src)
            try:
                ok = _CMP[type(op)](left, right)
            except TypeError as exc:
                raise ExprError(f"{src!r}: {exc}") from None
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body if _eval(node.test, env, src) else node.orelse, env, src)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env, src) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        try:
            return _FUNCS[node.func.id](*[_eval(a, env, src) for a in node.args])
        except (TypeError, ValueError) as exc:
            raise ExprError(f"{src!r}: {exc}") from None
    raise ExprError(f"{type(node).__name__} is not allowed in expressions: {src!r}")


_INTERP = re.compile(r"\$\{([^}]*)\}")
_WHOLE = re.compile(r"^\$\{([^}]*)\}$")
_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_.]*)$")


def resolve(value: Any, variables: Mapping[str, Any]) -> Any:
    """Resolve a YAML value: "${expr}" -> its value (any type), "$name" -> the
    variable, "a ${x} b" -> string interpolation, lists/dicts recursively."""
    if isinstance(value, str):
        m = _WHOLE.match(value)
        if m:
            return evaluate(m.group(1), variables)
        m = _REF.match(value)
        if m:
            return evaluate(m.group(1), variables)
        if "${" in value:
            return _INTERP.sub(lambda mm: _fmt(evaluate(mm.group(1), variables)), value)
        return value
    if isinstance(value, list):
        return [resolve(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, variables) for k, v in value.items()}
    return value


def has_refs(value: Any) -> bool:
    if isinstance(value, str):
        return "${" in value or bool(_REF.match(value))
    if isinstance(value, list):
        return any(has_refs(v) for v in value)
    if isinstance(value, dict):
        return any(has_refs(v) for v in value.values())
    return False


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
