#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FLOAT = "float"
BOOL = "bool"
UNIT = "unit"
TENSOR = "tensor1<f32>"


class FrontendError(Exception):
    pass


@dataclass(frozen=True)
class FunType:
    params: tuple[Any, ...]
    result: Any


@dataclass(frozen=True)
class TupleType:
    items: tuple[Any, ...]


def type_to_string(ty: Any) -> str:
    if isinstance(ty, FunType):
        parts = [type_to_string(item) for item in ty.params]
        parts.append(type_to_string(ty.result))
        return "(" + " -> ".join(parts) + ")"
    if isinstance(ty, TupleType):
        return "(" + " * ".join(type_to_string(item) for item in ty.items) + ")"
    return str(ty)


def fun_result_after_call(ty: Any, supplied: int) -> Any:
    if not isinstance(ty, FunType):
        return UNIT
    if supplied > len(ty.params):
        return UNIT
    if supplied == len(ty.params):
        return ty.result
    return FunType(ty.params[supplied:], ty.result)


def loc(node: ast.AST) -> str:
    line = getattr(node, "lineno", "?")
    col = getattr(node, "col_offset", "?")
    return f"line {line}, column {col}"


def fail(node: ast.AST, message: str) -> None:
    raise FrontendError(f"{loc(node)}: {message}")


def param_json(name: str, ty: Any) -> dict[str, Any]:
    return {"name": name, "type": type_to_string(ty)}


def var_expr(name: str, ty: Any) -> dict[str, Any]:
    return {"kind": "var", "name": name, "type": type_to_string(ty)}


def pattern_json(target: ast.expr, ty: Any) -> dict[str, Any]:
    if isinstance(target, ast.Name):
        return {"kind": "var", "name": target.id, "type": type_to_string(ty)}
    if isinstance(target, ast.Tuple):
        if not isinstance(ty, TupleType):
            fail(target, "tuple destructuring requires a tuple-producing expression")
        if len(target.elts) != len(ty.items):
            fail(target, "tuple pattern arity mismatch")
        return {
            "kind": "tuple",
            "items": [
                pattern_json(item, item_ty)
                for item, item_ty in zip(target.elts, ty.items, strict=True)
            ],
        }
    fail(target, "unsupported assignment target in staged region")


def parse_annotation(node: ast.AST | None) -> Any:
    if node is None:
        raise FrontendError("missing Python type annotation")
    if isinstance(node, ast.Name):
        if node.id == "float":
            return FLOAT
        if node.id == "bool":
            return BOOL
        if node.id in {"Tensor1", "Tensor"}:
            return TENSOR
    if isinstance(node, ast.Attribute):
        dotted = dotted_name(node)
        if dotted in {"loom.Tensor1", "loom.Tensor", "Tensor1.t", "loom.Tensor1.t"}:
            return TENSOR
    if isinstance(node, ast.Subscript):
        value = dotted_name(node.value)
        if value in {"tuple", "Tuple"}:
            items = []
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                items = [parse_annotation(item) for item in slice_node.elts]
            else:
                items = [parse_annotation(slice_node)]
            return TupleType(tuple(items))
    raise FrontendError(f"unsupported Python type annotation: {ast.unparse(node)}")


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted_name(node.value)}.{node.attr}"
    return ""


def is_entry_decorator(decorator: ast.expr) -> bool:
    name = dotted_name(decorator)
    if name in {"loom.entry", "entry"}:
        return True
    if isinstance(decorator, ast.Call):
        return is_entry_decorator(decorator.func)
    return False


def is_ignored_toplevel(stmt: ast.stmt) -> bool:
    return isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def scalar_prim(name: str, arity: int) -> str | None:
    table = {
        ("sqrt", 1): "fsqrt",
        ("math.sqrt", 1): "fsqrt",
        ("loom.sqrt", 1): "fsqrt",
        ("exp", 1): "fexp",
        ("math.exp", 1): "fexp",
        ("loom.exp", 1): "fexp",
        ("log", 1): "flog",
        ("math.log", 1): "flog",
        ("loom.log", 1): "flog",
        ("min", 2): "fmin",
        ("loom.min", 2): "fmin",
        ("max", 2): "fmax",
        ("loom.max", 2): "fmax",
        ("select", 3): "select",
        ("loom.select", 3): "select",
    }
    return table.get((name, arity))


class Lowerer:
    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.toplevel_entries = {
            stmt.name: stmt
            for stmt in tree.body
            if isinstance(stmt, ast.FunctionDef)
            and any(is_entry_decorator(item) for item in stmt.decorator_list)
        }

    def list_entries(self) -> list[dict[str, Any]]:
        entries = []
        for name in sorted(self.toplevel_entries):
            fn = self.toplevel_entries[name]
            params = []
            for arg in fn.args.args:
                ty = parse_annotation(arg.annotation)
                if ty == FLOAT:
                    kind = "scalar-f32"
                elif ty == TENSOR:
                    kind = "tensor1-f32"
                else:
                    fail(arg, f"unsupported entry parameter type {type_to_string(ty)}")
                params.append({"name": arg.arg, "kind": kind})
            entries.append({"name": fn.name, "params": params})
        return entries

    def entry(self, name: str) -> dict[str, Any]:
        fn = self.toplevel_entries.get(name)
        if fn is None:
            raise FrontendError(f'entry "{name}" not found')
        return self.lower_entry(fn)

    def lower_entry(self, fn: ast.FunctionDef) -> dict[str, Any]:
        self.reject_function_shape(fn)
        env: dict[str, Any] = {}
        params = []
        for arg in fn.args.args:
            ty = parse_annotation(arg.annotation)
            if ty not in {FLOAT, TENSOR}:
                fail(arg, f"unsupported entry parameter type {type_to_string(ty)}")
            env[arg.arg] = ty
            params.append(param_json(arg.arg, ty))
        return_ty = parse_annotation(fn.returns)
        body_expr, body_ty = self.lower_stmt_list(fn.body, env, fn.name)
        if type_to_string(return_ty) != type_to_string(body_ty):
            fail(fn, f"entry return annotation {type_to_string(return_ty)} does not match inferred {type_to_string(body_ty)}")
        return {
            "entry": fn.name,
            "params": params,
            "return_type": type_to_string(return_ty),
            "body": body_expr,
        }

    def reject_function_shape(self, fn: ast.FunctionDef) -> None:
        if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
            fail(fn, "only positional Python parameters are supported in Loom entries")
        if fn.args.defaults or fn.args.kw_defaults:
            fail(fn, "default Python parameters are not supported in Loom entries")
        if fn.returns is None:
            fail(fn, "Loom entrypoints require a return annotation")

    def lower_local_function(self, fn: ast.FunctionDef, env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        self.reject_function_shape(fn)
        local_env = dict(env)
        params = []
        param_types = []
        for arg in fn.args.args:
            ty = parse_annotation(arg.annotation)
            local_env[arg.arg] = ty
            param_types.append(ty)
            params.append(param_json(arg.arg, ty))
        return_ty = parse_annotation(fn.returns)
        body_expr, body_ty = self.lower_stmt_list(fn.body, local_env, fn.name)
        if type_to_string(return_ty) != type_to_string(body_ty):
            fail(fn, f"function return annotation {type_to_string(return_ty)} does not match inferred {type_to_string(body_ty)}")
        fn_ty = FunType(tuple(param_types), return_ty)
        return {"kind": "lambda", "params": params, "body": body_expr}, fn_ty

    def lower_stmt_list(self, stmts: list[ast.stmt], env: dict[str, Any], owner: str) -> tuple[dict[str, Any], Any]:
        while stmts and is_ignored_toplevel(stmts[0]):
            stmts = stmts[1:]
        if not stmts:
            raise FrontendError(f"{owner}: staged function body must return a value")
        first, rest = stmts[0], stmts[1:]
        if isinstance(first, ast.Return):
            if rest:
                fail(rest[0], "statements after return are not supported in staged regions")
            if first.value is None:
                return {"kind": "unit"}, UNIT
            return self.lower_expr(first.value, env)
        if isinstance(first, ast.FunctionDef):
            value, fn_ty = self.lower_local_function(first, env)
            next_env = dict(env)
            next_env[first.name] = fn_ty
            body, body_ty = self.lower_stmt_list(rest, next_env, owner)
            return {
                "kind": "let",
                "pattern": {"kind": "var", "name": first.name, "type": type_to_string(fn_ty)},
                "value": value,
                "body": body,
            }, body_ty
        if isinstance(first, ast.Assign):
            if len(first.targets) != 1:
                fail(first, "parallel assignment is not supported in staged regions")
            value, value_ty = self.lower_expr(first.value, env)
            target = first.targets[0]
            next_env = dict(env)
            self.bind_pattern_types(target, value_ty, next_env)
            body, body_ty = self.lower_stmt_list(rest, next_env, owner)
            return {
                "kind": "let",
                "pattern": pattern_json(target, value_ty),
                "value": value,
                "body": body,
            }, body_ty
        if isinstance(first, ast.AnnAssign):
            if first.value is None:
                fail(first, "annotation-only bindings are not supported in staged regions")
            value, value_ty = self.lower_expr(first.value, env)
            annotation_ty = parse_annotation(first.annotation)
            if type_to_string(annotation_ty) != type_to_string(value_ty):
                fail(first, f"annotation {type_to_string(annotation_ty)} does not match inferred {type_to_string(value_ty)}")
            next_env = dict(env)
            self.bind_pattern_types(first.target, value_ty, next_env)
            body, body_ty = self.lower_stmt_list(rest, next_env, owner)
            return {
                "kind": "let",
                "pattern": pattern_json(first.target, value_ty),
                "value": value,
                "body": body,
            }, body_ty
        if isinstance(first, ast.If):
            if rest:
                fail(first, "statement-level if is only supported as a tail expression")
            cond, cond_ty = self.lower_expr(first.test, env)
            if cond_ty != BOOL:
                fail(first.test, "if condition must be bool")
            then_expr, then_ty = self.lower_stmt_list(first.body, dict(env), owner)
            else_expr, else_ty = self.lower_stmt_list(first.orelse, dict(env), owner)
            if type_to_string(then_ty) != type_to_string(else_ty):
                fail(first, "if branches must have the same staged type")
            return {"kind": "if", "cond": cond, "then": then_expr, "else": else_expr}, then_ty
        if isinstance(first, (ast.For, ast.While)):
            fail(first, "loops are not supported in staged regions")
        if isinstance(first, (ast.AugAssign, ast.Delete, ast.With, ast.Try, ast.Raise)):
            fail(first, "mutation, exceptions, and resource scopes are not supported in staged regions")
        fail(first, "unsupported Python statement in staged region")

    def bind_pattern_types(self, target: ast.expr, ty: Any, env: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = ty
            return
        if isinstance(target, ast.Tuple):
            if not isinstance(ty, TupleType) or len(target.elts) != len(ty.items):
                fail(target, "tuple destructuring requires a matching tuple value")
            for item, item_ty in zip(target.elts, ty.items, strict=True):
                self.bind_pattern_types(item, item_ty, env)
            return
        fail(target, "unsupported assignment target in staged region")

    def lower_expr(self, expr: ast.expr, env: dict[str, Any], expected: Any | None = None) -> tuple[dict[str, Any], Any]:
        if isinstance(expr, ast.Name):
            if expr.id in env:
                ty = env[expr.id]
                return var_expr(expr.id, ty), ty
            fail(expr, f"unknown staged variable {expr.id}")
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return {"kind": "bool", "value": expr.value}, BOOL
            if isinstance(expr.value, (int, float)):
                return {"kind": "float", "value": float(expr.value)}, FLOAT
            if expr.value is None:
                return {"kind": "unit"}, UNIT
            fail(expr, "unsupported constant in staged region")
        if isinstance(expr, ast.Tuple):
            items = [self.lower_expr(item, env) for item in expr.elts]
            return {
                "kind": "tuple",
                "items": [item for item, _ in items],
            }, TupleType(tuple(ty for _, ty in items))
        if isinstance(expr, ast.IfExp):
            cond, cond_ty = self.lower_expr(expr.test, env)
            if cond_ty != BOOL:
                fail(expr.test, "conditional expression requires a bool condition")
            then_expr, then_ty = self.lower_expr(expr.body, env)
            else_expr, else_ty = self.lower_expr(expr.orelse, env)
            if type_to_string(then_ty) != type_to_string(else_ty):
                fail(expr, "conditional branches must have the same staged type")
            return {"kind": "if", "cond": cond, "then": then_expr, "else": else_expr}, then_ty
        if isinstance(expr, ast.Lambda):
            if not isinstance(expected, FunType):
                fail(expr, "lambda expressions require an expected Loom function type")
            if len(expr.args.args) != len(expected.params):
                fail(expr, "lambda arity does not match expected Loom function type")
            local_env = dict(env)
            params = []
            for arg, ty in zip(expr.args.args, expected.params, strict=True):
                local_env[arg.arg] = ty
                params.append(param_json(arg.arg, ty))
            body, body_ty = self.lower_expr(expr.body, local_env)
            if type_to_string(body_ty) != type_to_string(expected.result):
                fail(expr, "lambda result type does not match expected Loom function type")
            return {"kind": "lambda", "params": params, "body": body}, expected
        if isinstance(expr, ast.UnaryOp):
            value, value_ty = self.lower_expr(expr.operand, env)
            if value_ty != FLOAT:
                fail(expr, "unary numeric operators require float operands")
            if isinstance(expr.op, ast.USub):
                return {"kind": "prim", "op": "fneg", "args": [value]}, FLOAT
            if isinstance(expr.op, ast.UAdd):
                return value, FLOAT
            fail(expr, "unsupported unary operator in staged region")
        if isinstance(expr, ast.BinOp):
            left, left_ty = self.lower_expr(expr.left, env)
            right, right_ty = self.lower_expr(expr.right, env)
            if left_ty != FLOAT or right_ty != FLOAT:
                fail(expr, "binary numeric operators require float operands")
            op = {
                ast.Add: "fadd",
                ast.Sub: "fsub",
                ast.Mult: "fmul",
                ast.Div: "fdiv",
            }.get(type(expr.op))
            if op is None:
                fail(expr, "unsupported binary operator in staged region")
            return {"kind": "prim", "op": op, "args": [left, right]}, FLOAT
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                fail(expr, "chained comparisons are not supported in staged regions")
            left, left_ty = self.lower_expr(expr.left, env)
            right, right_ty = self.lower_expr(expr.comparators[0], env)
            if left_ty != FLOAT or right_ty != FLOAT:
                fail(expr, "scalar comparisons require float operands")
            op = {
                ast.Lt: "fcmplt",
                ast.LtE: "fcmple",
                ast.Gt: "fcmpgt",
                ast.GtE: "fcmpge",
                ast.Eq: "fcmpeq",
            }.get(type(expr.ops[0]))
            if op is None:
                fail(expr, "unsupported comparison operator in staged region")
            return {"kind": "prim", "op": op, "args": [left, right]}, BOOL
        if isinstance(expr, ast.Call):
            return self.lower_call(expr, env)
        if isinstance(expr, ast.Attribute):
            fail(expr, f"unsupported staged attribute access {dotted_name(expr)}")
        fail(expr, "unsupported Python expression in staged region")

    def lower_call(self, expr: ast.Call, env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        if expr.keywords:
            fail(expr, "keyword arguments are not supported in staged regions")
        name = dotted_name(expr.func)
        tensor_prim = {
            "loom.Tensor1.map": ("tensor-map", FunType((FLOAT,), FLOAT), TENSOR),
            "Tensor1.map": ("tensor-map", FunType((FLOAT,), FLOAT), TENSOR),
            "loom.Tensor1.map2": ("tensor-map2", FunType((FLOAT, FLOAT), FLOAT), TENSOR),
            "Tensor1.map2": ("tensor-map2", FunType((FLOAT, FLOAT), FLOAT), TENSOR),
            "loom.Tensor1.reduce_sum": ("tensor-reduce-sum", None, FLOAT),
            "Tensor1.reduce_sum": ("tensor-reduce-sum", None, FLOAT),
            "loom.Tensor1.reduce_max": ("tensor-reduce-max", None, FLOAT),
            "Tensor1.reduce_max": ("tensor-reduce-max", None, FLOAT),
        }.get(name)
        if tensor_prim is not None:
            op, fn_type, result_ty = tensor_prim
            if op == "tensor-map":
                if len(expr.args) != 2:
                    fail(expr, "Tensor1.map expects a function and one tensor")
                fn, _ = self.lower_expr(expr.args[0], env, fn_type)
                tensor, tensor_ty = self.lower_expr(expr.args[1], env)
                if tensor_ty != TENSOR:
                    fail(expr.args[1], "Tensor1.map expects a tensor argument")
                return {"kind": "tensor-prim", "op": op, "args": [fn, tensor]}, result_ty
            if op == "tensor-map2":
                if len(expr.args) != 3:
                    fail(expr, "Tensor1.map2 expects a function and two tensors")
                fn, _ = self.lower_expr(expr.args[0], env, fn_type)
                lhs, lhs_ty = self.lower_expr(expr.args[1], env)
                rhs, rhs_ty = self.lower_expr(expr.args[2], env)
                if lhs_ty != TENSOR or rhs_ty != TENSOR:
                    fail(expr, "Tensor1.map2 expects tensor arguments")
                return {"kind": "tensor-prim", "op": op, "args": [fn, lhs, rhs]}, result_ty
            if len(expr.args) != 1:
                fail(expr, f"{name} expects one tensor")
            tensor, tensor_ty = self.lower_expr(expr.args[0], env)
            if tensor_ty != TENSOR:
                fail(expr.args[0], f"{name} expects a tensor argument")
            return {"kind": "tensor-prim", "op": op, "args": [tensor]}, result_ty

        prim = scalar_prim(name, len(expr.args))
        if prim is not None:
            args = [self.lower_expr(arg, env)[0] for arg in expr.args]
            if prim == "select":
                return {"kind": "prim", "op": prim, "args": args}, FLOAT
            return {"kind": "prim", "op": prim, "args": args}, FLOAT

        fn, fn_ty = self.lower_expr(expr.func, env)
        args = [self.lower_expr(arg, env)[0] for arg in expr.args]
        return {"kind": "apply", "fn": fn, "args": args}, fun_result_after_call(fn_ty, len(args))


def parse_source(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise FrontendError(str(exc)) from exc


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="loom_frontend_python.py")
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list-entries-json")
    list_parser.add_argument("file")
    front_parser = sub.add_parser("front-ir-json")
    front_parser.add_argument("file")
    front_parser.add_argument("--entry", required=True)
    args = parser.parse_args(argv)

    tree = parse_source(Path(args.file))
    lowerer = Lowerer(tree)
    if args.command == "list-entries-json":
        print(json.dumps({"entries": lowerer.list_entries()}, indent=2, sort_keys=True))
        return 0
    if args.command == "front-ir-json":
        print(json.dumps(lowerer.entry(args.entry), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def main() -> None:
    try:
        raise SystemExit(run(sys.argv[1:]))
    except FrontendError as exc:
        print(f"Python frontend error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
