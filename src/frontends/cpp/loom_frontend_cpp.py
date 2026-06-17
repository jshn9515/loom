#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FLOAT = "float"
BOOL = "bool"
UNIT = "unit"
TENSOR = "tensor1<f32>"

LOOM_HEADER = r"""
#pragma once
#define LOOM_ENTRY [[clang::annotate("loom.entry")]]
namespace loom {
struct Tensor1 {
  template <typename F> static Tensor1 map(F, Tensor1) { return {}; }
  template <typename F> static Tensor1 map2(F, Tensor1, Tensor1) { return {}; }
  static float reduce_sum(Tensor1) { return 0.0f; }
  static float reduce_max(Tensor1) { return 0.0f; }
};
template <typename F> Tensor1 map(F, Tensor1) { return {}; }
template <typename F> Tensor1 map2(F, Tensor1, Tensor1) { return {}; }
inline float reduce_sum(Tensor1) { return 0.0f; }
inline float reduce_max(Tensor1) { return 0.0f; }
inline float sqrt(float value) { return value; }
inline float exp(float value) { return value; }
inline float log(float value) { return value; }
inline float min(float lhs, float rhs) { return lhs < rhs ? lhs : rhs; }
inline float max(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }
inline float select(bool cond, float lhs, float rhs) { return cond ? lhs : rhs; }
template <typename A, typename B> struct Tuple2 { A first; B second; };
template <typename A, typename B> Tuple2<A, B> tuple(A a, B b) { return {a, b}; }
template <unsigned long I, typename A, typename B>
auto get(Tuple2<A, B>& t) -> decltype((I == 0 ? t.first : t.second)) {
  if constexpr (I == 0) return t.first; else return t.second;
}
template <unsigned long I, typename A, typename B>
auto get(const Tuple2<A, B>& t) -> decltype((I == 0 ? t.first : t.second)) {
  if constexpr (I == 0) return t.first; else return t.second;
}
template <unsigned long I, typename A, typename B>
auto get(Tuple2<A, B>&& t) -> decltype((I == 0 ? t.first : t.second)) {
  if constexpr (I == 0) return t.first; else return t.second;
}
template <typename F, typename... Args> struct Partial {};
template <typename F, typename... Args> Partial<F, Args...> partial(F, Args...) { return {}; }
}
namespace std {
template <typename T> struct tuple_size;
template <unsigned long I, typename T> struct tuple_element;
template <typename A, typename B> struct tuple_size<loom::Tuple2<A, B>> {
  static constexpr unsigned long value = 2;
};
template <typename A, typename B> struct tuple_element<0, loom::Tuple2<A, B>> {
  using type = A;
};
template <typename A, typename B> struct tuple_element<1, loom::Tuple2<A, B>> {
  using type = B;
};
}
"""


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


def node_kind(node: dict[str, Any] | None) -> str:
    return "" if node is None else str(node.get("kind", ""))


def node_children(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in node.get("inner", []) if isinstance(item, dict)]


def qual_type(node: dict[str, Any]) -> str:
    ty = node.get("type", {})
    if not isinstance(ty, dict):
        return ""
    return str(ty.get("desugaredQualType") or ty.get("qualType") or "")


def loc(node: dict[str, Any]) -> str:
    item = node.get("loc")
    if not isinstance(item, dict):
        item = node.get("range", {}).get("begin") if isinstance(node.get("range"), dict) else None
    if not isinstance(item, dict):
        return "unknown location"
    line = item.get("line", "?")
    col = item.get("col", "?")
    return f"line {line}, column {col}"


def fail(node: dict[str, Any], message: str) -> None:
    raise FrontendError(f"{loc(node)}: {message}")


def param_json(name: str, ty: Any) -> dict[str, Any]:
    return {"name": name, "type": type_to_string(ty)}


def var_expr(name: str, ty: Any) -> dict[str, Any]:
    return {"kind": "var", "name": name, "type": type_to_string(ty)}


def parse_cpp_type(node: dict[str, Any]) -> Any:
    text = qual_type(node)
    cleaned = (
        text.replace("const ", "")
        .replace("volatile ", "")
        .replace("struct ", "")
        .replace("class ", "")
        .strip()
    )
    cleaned = cleaned.rstrip("&").rstrip("*").strip()
    if cleaned in {"float", "double"}:
        return FLOAT
    if cleaned == "bool":
        return BOOL
    if cleaned in {"loom::Tensor1", "Tensor1"} or cleaned.endswith("::Tensor1"):
        return TENSOR
    raise FrontendError(f"unsupported C++ type annotation: {text}")


def return_type_of_function(fn: dict[str, Any]) -> Any:
    text = qual_type(fn)
    if " (" not in text:
        raise FrontendError(f"unsupported C++ function type: {text}")
    fake = {"type": {"qualType": text.split(" (", 1)[0]}}
    return parse_cpp_type(fake)


def entry_param_kind(ty: Any) -> str:
    if ty == FLOAT:
        return "scalar-f32"
    if ty == TENSOR:
        return "tensor1-f32"
    raise FrontendError(f"unsupported entry parameter type {type_to_string(ty)}")


def unwrap(node: dict[str, Any]) -> dict[str, Any]:
    wrappers = {
        "ConstantExpr",
        "CXXBindTemporaryExpr",
        "CXXConstructExpr",
        "CXXFunctionalCastExpr",
        "ExprWithCleanups",
        "ImplicitCastExpr",
        "MaterializeTemporaryExpr",
        "ParenExpr",
    }
    current = node
    while node_kind(current) in wrappers:
        children = node_children(current)
        if len(children) != 1:
            break
        current = children[0]
    return current


def referenced_name(node: dict[str, Any]) -> str:
    node = unwrap(node)
    if node_kind(node) == "DeclRefExpr":
        ref = node.get("referencedDecl", {})
        if isinstance(ref, dict):
            return str(ref.get("name") or node.get("name") or "")
    if node_kind(node) == "MemberExpr":
        return str(node.get("name", ""))
    return str(node.get("name", ""))


def has_entry_attr(fn: dict[str, Any]) -> bool:
    return any(node_kind(child) == "AnnotateAttr" for child in node_children(fn))


def find_entries(tree: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node_kind(node) == "FunctionDecl" and has_entry_attr(node):
            if node.get("isThisDeclarationADefinition", True):
                entries.append(node)
            return
        for child in node_children(node):
            visit(child)

    visit(tree)
    return sorted(entries, key=lambda item: str(item.get("name", "")))


def lambda_method(lambda_expr: dict[str, Any]) -> dict[str, Any]:
    for child in node_children(lambda_expr):
        if node_kind(child) != "CXXRecordDecl":
            continue
        for member in node_children(child):
            if node_kind(member) == "CXXMethodDecl" and member.get("name") == "operator()":
                return member
    fail(lambda_expr, "could not find lambda call operator")


def lambda_body(lambda_expr: dict[str, Any]) -> dict[str, Any]:
    method = lambda_method(lambda_expr)
    for child in node_children(method):
        if node_kind(child) == "CompoundStmt":
            return child
    fail(lambda_expr, "lambda body is missing")


def function_params(fn: dict[str, Any]) -> list[dict[str, Any]]:
    return [child for child in node_children(fn) if node_kind(child) == "ParmVarDecl"]


def function_body(fn: dict[str, Any]) -> dict[str, Any]:
    for child in node_children(fn):
        if node_kind(child) == "CompoundStmt":
            return child
    fail(fn, "entry body is missing")


class Lowerer:
    def __init__(self, tree: dict[str, Any]):
        self.tree = tree
        self.entries = {str(entry.get("name")): entry for entry in find_entries(tree)}

    def list_entries(self) -> list[dict[str, Any]]:
        rows = []
        for name in sorted(self.entries):
            fn = self.entries[name]
            params = []
            for param in function_params(fn):
                ty = parse_cpp_type(param)
                params.append({"name": str(param.get("name", "")), "kind": entry_param_kind(ty)})
            rows.append({"name": name, "params": params})
        return rows

    def entry(self, name: str) -> dict[str, Any]:
        fn = self.entries.get(name)
        if fn is None:
            raise FrontendError(f'entry "{name}" not found')
        return self.lower_entry(fn)

    def lower_entry(self, fn: dict[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = {}
        params = []
        for param in function_params(fn):
            ty = parse_cpp_type(param)
            if ty not in {FLOAT, TENSOR}:
                fail(param, f"unsupported entry parameter type {type_to_string(ty)}")
            name = str(param.get("name", ""))
            env[name] = ty
            params.append(param_json(name, ty))
        return_ty = return_type_of_function(fn)
        body_expr, body_ty = self.lower_stmt_list(node_children(function_body(fn)), env, str(fn.get("name", "")))
        if type_to_string(return_ty) != type_to_string(body_ty):
            fail(fn, f"entry return type {type_to_string(return_ty)} does not match inferred {type_to_string(body_ty)}")
        return {
            "entry": str(fn.get("name", "")),
            "params": params,
            "return_type": type_to_string(return_ty),
            "body": body_expr,
        }

    def lower_stmt_list(self, stmts: list[dict[str, Any]], env: dict[str, Any], owner: str) -> tuple[dict[str, Any], Any]:
        stmts = [stmt for stmt in stmts if node_kind(stmt) != "NullStmt"]
        if not stmts:
            raise FrontendError(f"{owner}: staged function body must return a value")
        first, rest = stmts[0], stmts[1:]
        kind = node_kind(first)
        if kind == "ReturnStmt":
            if rest:
                fail(rest[0], "statements after return are not supported in staged regions")
            children = node_children(first)
            if not children:
                return {"kind": "unit"}, UNIT
            return self.lower_expr(children[0], env)
        if kind == "DeclStmt":
            return self.lower_decl_stmt(first, rest, env, owner)
        if kind == "IfStmt":
            if rest:
                fail(first, "statement-level if is only supported as a tail expression")
            return self.lower_if_stmt(first, env, owner)
        if kind in {"ForStmt", "WhileStmt", "DoStmt"}:
            fail(first, "loops are not supported in staged regions")
        if kind in {"CXXTryStmt", "CXXThrowExpr", "GotoStmt"}:
            fail(first, "exceptions and nonlocal control flow are not supported in staged regions")
        fail(first, "unsupported C++ statement in staged region")

    def lower_decl_stmt(
        self,
        stmt: dict[str, Any],
        rest: list[dict[str, Any]],
        env: dict[str, Any],
        owner: str,
    ) -> tuple[dict[str, Any], Any]:
        decls = node_children(stmt)
        if not decls:
            fail(stmt, "empty declaration statement is not supported")
        return self.lower_decl_bindings(decls, rest, env, owner)

    def lower_decl_bindings(
        self,
        decls: list[dict[str, Any]],
        rest: list[dict[str, Any]],
        env: dict[str, Any],
        owner: str,
    ) -> tuple[dict[str, Any], Any]:
        if not decls:
            return self.lower_stmt_list(rest, env, owner)
        decl, remaining = decls[0], decls[1:]
        kind = node_kind(decl)
        if kind == "VarDecl":
            name = str(decl.get("name", ""))
            children = node_children(decl)
            if not children:
                fail(decl, "uninitialized staged bindings are not supported")
            value_expr, value_ty = self.lower_expr(children[0], env)
            next_env = dict(env)
            next_env[name] = value_ty
            body_expr, body_ty = self.lower_decl_bindings(remaining, rest, next_env, owner)
            return {
                "kind": "let",
                "pattern": {"kind": "var", "name": name, "type": type_to_string(value_ty)},
                "value": value_expr,
                "body": body_expr,
            }, body_ty
        if kind == "DecompositionDecl":
            value_child = self.decomposition_value(decl)
            value_expr, value_ty = self.lower_expr(value_child, env)
            if not isinstance(value_ty, TupleType):
                fail(decl, "structured binding requires a tuple-producing expression")
            bindings = [child for child in node_children(decl) if node_kind(child) == "BindingDecl"]
            if len(bindings) != len(value_ty.items):
                fail(decl, "structured binding arity does not match tuple value")
            next_env = dict(env)
            for binding, item_ty in zip(bindings, value_ty.items, strict=True):
                next_env[str(binding.get("name", ""))] = item_ty
            body_expr, body_ty = self.lower_decl_bindings(remaining, rest, next_env, owner)
            return {
                "kind": "let",
                "pattern": {
                    "kind": "tuple",
                    "items": [
                        {"kind": "var", "name": str(binding.get("name", "")), "type": type_to_string(item_ty)}
                        for binding, item_ty in zip(bindings, value_ty.items, strict=True)
                    ],
                },
                "value": value_expr,
                "body": body_expr,
            }, body_ty
        fail(decl, "unsupported declaration in staged region")

    def decomposition_value(self, decl: dict[str, Any]) -> dict[str, Any]:
        for child in node_children(decl):
            if node_kind(child) != "BindingDecl":
                return child
        fail(decl, "structured binding initializer is missing")

    def lower_if_stmt(self, stmt: dict[str, Any], env: dict[str, Any], owner: str) -> tuple[dict[str, Any], Any]:
        children = node_children(stmt)
        if len(children) < 2:
            fail(stmt, "if statement is missing branches")
        cond, cond_ty = self.lower_expr(children[0], env)
        if cond_ty != BOOL:
            fail(children[0], "if condition must be bool")
        then_expr, then_ty = self.lower_branch_stmt(children[1], dict(env), owner)
        if len(children) < 3:
            fail(stmt, "if statements in staged regions require an else branch")
        else_expr, else_ty = self.lower_branch_stmt(children[2], dict(env), owner)
        if type_to_string(then_ty) != type_to_string(else_ty):
            fail(stmt, "if branches must have the same staged type")
        return {"kind": "if", "cond": cond, "then": then_expr, "else": else_expr}, then_ty

    def lower_branch_stmt(self, stmt: dict[str, Any], env: dict[str, Any], owner: str) -> tuple[dict[str, Any], Any]:
        if node_kind(stmt) == "CompoundStmt":
            return self.lower_stmt_list(node_children(stmt), env, owner)
        if node_kind(stmt) == "ReturnStmt":
            return self.lower_stmt_list([stmt], env, owner)
        fail(stmt, "if branches must return staged values")

    def lower_expr(
        self,
        expr: dict[str, Any],
        env: dict[str, Any],
        expected: Any | None = None,
    ) -> tuple[dict[str, Any], Any]:
        expr = unwrap(expr)
        kind = node_kind(expr)
        if kind == "DeclRefExpr":
            name = referenced_name(expr)
            if name in env:
                ty = env[name]
                return var_expr(name, ty), ty
            fail(expr, f"unknown staged variable {name}")
        if kind == "FloatingLiteral":
            return {"kind": "float", "value": float(str(expr.get("value", "0")))}, FLOAT
        if kind == "IntegerLiteral":
            return {"kind": "float", "value": float(str(expr.get("value", "0")))}, FLOAT
        if kind == "CXXBoolLiteralExpr":
            return {"kind": "bool", "value": bool(expr.get("value", False))}, BOOL
        if kind == "LambdaExpr":
            return self.lower_lambda(expr, env, expected)
        if kind == "UnaryOperator":
            return self.lower_unary(expr, env)
        if kind == "BinaryOperator":
            return self.lower_binary(expr, env)
        if kind == "ConditionalOperator":
            return self.lower_conditional(expr, env)
        if kind == "CallExpr":
            return self.lower_call(expr, env)
        if kind == "CXXOperatorCallExpr":
            return self.lower_operator_call(expr, env)
        if kind == "InitListExpr":
            items = [self.lower_expr(item, env) for item in node_children(expr)]
            return {
                "kind": "tuple",
                "items": [item for item, _ in items],
            }, TupleType(tuple(ty for _, ty in items))
        fail(expr, "unsupported C++ expression in staged region")

    def lower_lambda(
        self,
        expr: dict[str, Any],
        env: dict[str, Any],
        expected: Any | None,
    ) -> tuple[dict[str, Any], Any]:
        method = lambda_method(expr)
        local_env = dict(env)
        params = []
        param_types = []
        for param in function_params(method):
            ty = parse_cpp_type(param)
            name = str(param.get("name", ""))
            local_env[name] = ty
            param_types.append(ty)
            params.append(param_json(name, ty))
        body, body_ty = self.lower_stmt_list(node_children(lambda_body(expr)), local_env, "lambda")
        fn_ty = FunType(tuple(param_types), body_ty)
        if expected is not None and type_to_string(expected) != type_to_string(fn_ty):
            fail(expr, f"lambda type {type_to_string(fn_ty)} does not match expected {type_to_string(expected)}")
        return {"kind": "lambda", "params": params, "body": body}, fn_ty

    def lower_unary(self, expr: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        children = node_children(expr)
        if len(children) != 1:
            fail(expr, "unsupported unary operator shape")
        value, value_ty = self.lower_expr(children[0], env)
        if value_ty != FLOAT:
            fail(expr, "unary numeric operators require float operands")
        opcode = str(expr.get("opcode", ""))
        if opcode == "-":
            return {"kind": "prim", "op": "fneg", "args": [value]}, FLOAT
        if opcode == "+":
            return value, FLOAT
        fail(expr, "unsupported unary operator in staged region")

    def lower_binary(self, expr: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        children = node_children(expr)
        if len(children) != 2:
            fail(expr, "unsupported binary operator shape")
        left, left_ty = self.lower_expr(children[0], env)
        right, right_ty = self.lower_expr(children[1], env)
        opcode = str(expr.get("opcode", ""))
        numeric = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}
        compare = {"<": "fcmplt", "<=": "fcmple", ">": "fcmpgt", ">=": "fcmpge", "==": "fcmpeq"}
        if opcode in numeric:
            if left_ty != FLOAT or right_ty != FLOAT:
                fail(expr, "binary numeric operators require float operands")
            return {"kind": "prim", "op": numeric[opcode], "args": [left, right]}, FLOAT
        if opcode in compare:
            if left_ty != FLOAT or right_ty != FLOAT:
                fail(expr, "scalar comparisons require float operands")
            return {"kind": "prim", "op": compare[opcode], "args": [left, right]}, BOOL
        if opcode in {"=", "+=", "-=", "*=", "/="}:
            fail(expr, "mutation is not supported in staged regions")
        fail(expr, "unsupported binary operator in staged region")

    def lower_conditional(self, expr: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        children = node_children(expr)
        if len(children) != 3:
            fail(expr, "unsupported conditional expression shape")
        cond, cond_ty = self.lower_expr(children[0], env)
        if cond_ty != BOOL:
            fail(expr, "conditional expression requires a bool condition")
        then_expr, then_ty = self.lower_expr(children[1], env)
        else_expr, else_ty = self.lower_expr(children[2], env)
        if type_to_string(then_ty) != type_to_string(else_ty):
            fail(expr, "conditional branches must have the same staged type")
        return {"kind": "if", "cond": cond, "then": then_expr, "else": else_expr}, then_ty

    def lower_call(self, expr: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        children = node_children(expr)
        if not children:
            fail(expr, "unsupported empty call expression")
        callee = children[0]
        args = children[1:]
        name = referenced_name(callee)
        tensor_prim = {
            "map": ("tensor-map", FunType((FLOAT,), FLOAT), TENSOR),
            "map2": ("tensor-map2", FunType((FLOAT, FLOAT), FLOAT), TENSOR),
            "reduce_sum": ("tensor-reduce-sum", None, FLOAT),
            "reduce_max": ("tensor-reduce-max", None, FLOAT),
        }.get(name)
        if tensor_prim is not None:
            return self.lower_tensor_prim(expr, args, *tensor_prim, env=env)
        prim = {
            ("sqrt", 1): "fsqrt",
            ("exp", 1): "fexp",
            ("log", 1): "flog",
            ("min", 2): "fmin",
            ("max", 2): "fmax",
            ("select", 3): "select",
        }.get((name, len(args)))
        if prim is not None:
            lowered = [self.lower_expr(arg, env)[0] for arg in args]
            return {"kind": "prim", "op": prim, "args": lowered}, FLOAT
        if name == "tuple":
            items = [self.lower_expr(arg, env) for arg in args]
            if len(items) < 2:
                fail(expr, "loom::tuple expects at least two items")
            return {
                "kind": "tuple",
                "items": [item for item, _ in items],
            }, TupleType(tuple(ty for _, ty in items))
        if name == "partial":
            if len(args) < 1:
                fail(expr, "loom::partial expects a function argument")
            fn, fn_ty = self.lower_expr(args[0], env)
            lowered_args = [self.lower_expr(arg, env)[0] for arg in args[1:]]
            return {"kind": "apply", "fn": fn, "args": lowered_args}, fun_result_after_call(fn_ty, len(lowered_args))
        fail(expr, f"unsupported C++ function call {name}")

    def lower_tensor_prim(
        self,
        expr: dict[str, Any],
        args: list[dict[str, Any]],
        op: str,
        fn_type: Any | None,
        result_ty: Any,
        env: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        if op == "tensor-map":
            if len(args) != 2:
                fail(expr, "Tensor1::map expects a function and one tensor")
            fn, _ = self.lower_expr(args[0], env, fn_type)
            tensor, tensor_ty = self.lower_expr(args[1], env)
            if tensor_ty != TENSOR:
                fail(args[1], "Tensor1::map expects a tensor argument")
            return {"kind": "tensor-prim", "op": op, "args": [fn, tensor]}, result_ty
        if op == "tensor-map2":
            if len(args) != 3:
                fail(expr, "Tensor1::map2 expects a function and two tensors")
            fn, _ = self.lower_expr(args[0], env, fn_type)
            lhs, lhs_ty = self.lower_expr(args[1], env)
            rhs, rhs_ty = self.lower_expr(args[2], env)
            if lhs_ty != TENSOR or rhs_ty != TENSOR:
                fail(expr, "Tensor1::map2 expects tensor arguments")
            return {"kind": "tensor-prim", "op": op, "args": [fn, lhs, rhs]}, result_ty
        if len(args) != 1:
            fail(expr, f"Tensor1::{op} expects one tensor")
        tensor, tensor_ty = self.lower_expr(args[0], env)
        if tensor_ty != TENSOR:
            fail(args[0], f"Tensor1::{op} expects a tensor argument")
        return {"kind": "tensor-prim", "op": op, "args": [tensor]}, result_ty

    def lower_operator_call(self, expr: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        children = node_children(expr)
        if len(children) < 2:
            fail(expr, "unsupported C++ operator call")
        callee_name = referenced_name(children[0])
        if callee_name != "operator()":
            fail(expr, "only lambda/function call operators are supported")
        fn, fn_ty = self.lower_expr(children[1], env)
        args = [self.lower_expr(arg, env)[0] for arg in children[2:]]
        return {"kind": "apply", "fn": fn, "args": args}, fun_result_after_call(fn_ty, len(args))


def parse_source(path: Path) -> dict[str, Any]:
    clangxx = os.environ.get("LOOM_CLANGXX", "clang++")
    with tempfile.TemporaryDirectory(prefix="loom_cpp_frontend_") as tmp:
        include_dir = Path(tmp) / "include" / "loom"
        include_dir.mkdir(parents=True)
        (include_dir / "loom.hpp").write_text(LOOM_HEADER, encoding="utf-8")
        result = subprocess.run(
            [
                clangxx,
                "-std=c++20",
                "-I",
                str(Path(tmp) / "include"),
                "-fsyntax-only",
                "-Xclang",
                "-ast-dump=json",
                str(path),
            ],
            cwd=path.parent,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise FrontendError(message)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FrontendError(f"failed to parse clang AST JSON: {exc}") from exc


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="loom_frontend_cpp.py")
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list-entries-json")
    list_parser.add_argument("file")
    front_parser = sub.add_parser("front-ir-json")
    front_parser.add_argument("file")
    front_parser.add_argument("--entry", required=True)
    args = parser.parse_args(argv)

    tree = parse_source(Path(args.file).resolve())
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
        print(f"C++ frontend error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
