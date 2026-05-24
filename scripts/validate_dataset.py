#!/usr/bin/env python3
"""Validate JBUJB tool-calling JSONL datasets against the registry."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MUTATING_TOOLS = {"create_order", "add_to_order", "remove_from_order"}
ID_KEYS = {"product_id", "merchant_id", "business_id", "draft_id", "user_id", "city_id"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_language(value: Any) -> Any:
    if value == "ar_darija":
        return "ar"
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", data) if isinstance(data, dict) else data
    registry: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if name:
            registry[name] = function
    return registry


def parse_args(args: Any) -> Any:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args
    if isinstance(args, dict):
        out = {}
        for key, value in args.items():
            if key == "language":
                out[key] = canonical_language(value)
            else:
                out[key] = parse_args(value)
        return out
    if isinstance(args, list):
        return [parse_args(value) for value in args]
    return args


def validate_against_schema(value: Any, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        for key, item_value in value.items():
            if key not in properties:
                errors.append(f"{path}: unknown field {key}")
                continue
            errors.extend(validate_against_schema(item_value, properties[key], f"{path}.{key}"))
    elif schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        for idx, item in enumerate(value):
            errors.extend(validate_against_schema(item, item_schema, f"{path}[{idx}]"))
    return errors


def validate_file(path: Path, registry: dict[str, dict[str, Any]]) -> tuple[list[str], Counter]:
    errors: list[str] = []
    counts: Counter = Counter()
    rows = load_jsonl(path)
    counts["rows"] = len(rows)

    for row_idx, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            errors.append(f"{path}:{row_idx}: missing messages list")
            continue

        roles = [m.get("role") for m in messages if isinstance(m, dict)]
        if "system" not in roles:
            errors.append(f"{path}:{row_idx}: missing system message")
        if "user" not in roles:
            errors.append(f"{path}:{row_idx}: missing user message")

        metadata = row.get("metadata", {})
        if isinstance(metadata, dict) and "language" in metadata:
            metadata_language = canonical_language(metadata.get("language"))
            if metadata_language not in {"fr", "en", "ar"}:
                errors.append(f"{path}:{row_idx}: invalid metadata language {metadata.get('language')!r}")

        has_confirmation = bool(row.get("metadata", {}).get("has_confirmation"))
        pending_tool_calls = 0

        for msg_idx, message in enumerate(messages, 1):
            if not isinstance(message, dict):
                errors.append(f"{path}:{row_idx}:{msg_idx}: message is not an object")
                continue

            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                if message.get("content") not in {None, ""}:
                    errors.append(f"{path}:{row_idx}:{msg_idx}: assistant tool-call message must not contain content")
                for call_idx, tool_call in enumerate(message.get("tool_calls") or [], 1):
                    if not isinstance(tool_call, dict):
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: tool call is not an object")
                        continue
                    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    name = function.get("name")
                    if not name:
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: missing tool name")
                        continue
                    if name not in registry:
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: unknown tool {name}")
                        continue

                    schema = registry[name].get("parameters", {})
                    required = set(schema.get("required", []))
                    properties = schema.get("properties", {})
                    args = parse_args(function.get("arguments", {}))
                    counts["tool_calls"] += 1
                    if not isinstance(args, dict):
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: tool arguments are not an object")
                        continue

                    missing = sorted(required - set(args.keys()))
                    if missing:
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: missing required args {missing} for {name}")

                    unknown = sorted(k for k in args.keys() if k not in properties)
                    if unknown:
                        counts["unknown_args"] += len(unknown)

                    schema_errors = validate_against_schema(args, schema, f"{path}:{row_idx}:{msg_idx}:{call_idx}")
                    if schema_errors:
                        errors.extend(schema_errors)

                    def walk(obj: Any, current_key: str = "") -> None:
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if key in ID_KEYS and (value is None or value == ""):
                                    errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: empty {key} for {name}")
                                if key == "language" and value not in {"fr", "en", "ar"}:
                                    errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: invalid language value {value!r} for {name}")
                                walk(value, key)
                        elif isinstance(obj, list):
                            for item in obj:
                                walk(item, current_key)

                    walk(args)

                    if name in MUTATING_TOOLS and not has_confirmation:
                        errors.append(f"{path}:{row_idx}:{msg_idx}:{call_idx}: mutating tool without confirmation metadata")
                pending_tool_calls += len(message.get("tool_calls") or [])

            if role == "tool":
                if pending_tool_calls <= 0:
                    errors.append(f"{path}:{row_idx}:{msg_idx}: tool message without preceding assistant tool call")
                else:
                    pending_tool_calls -= 1
                if not message.get("tool_call_id"):
                    errors.append(f"{path}:{row_idx}:{msg_idx}: tool message missing tool_call_id")
                if not message.get("name"):
                    errors.append(f"{path}:{row_idx}:{msg_idx}: tool message missing tool name")
                if not isinstance(message.get("content"), str):
                    errors.append(f"{path}:{row_idx}:{msg_idx}: tool message content must be a string")
            elif role in {"user", "assistant"} and not (role == "assistant" and message.get("tool_calls")) and pending_tool_calls > 0:
                errors.append(f"{path}:{row_idx}:{msg_idx}: conversation continues before all tool calls are answered")

        if pending_tool_calls > 0:
            errors.append(f"{path}:{row_idx}: unresolved tool calls at end of conversation")

    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data/tool_registry.json")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[
            "data/train.jsonl",
            "data/validation.jsonl",
            "data/evaluation.jsonl",
            "data/5k/train.jsonl",
            "data/5k/validation.jsonl",
            "data/5k/evaluation.jsonl",
        ],
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    if not registry_path.exists():
        print(f"Missing registry: {registry_path}", file=sys.stderr)
        return 2

    registry = load_registry(registry_path)
    total_errors: list[str] = []
    total_counts: Counter = Counter()

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        errors, counts = validate_file(path, registry)
        total_errors.extend(errors)
        total_counts.update(counts)
        print(f"{path}: rows={counts['rows']} tool_calls={counts['tool_calls']} unknown_args={counts['unknown_args']} errors={len(errors)}")

    if total_errors:
        print("\nValidation errors:", file=sys.stderr)
        for err in total_errors[:200]:
            print(f"- {err}", file=sys.stderr)
        if len(total_errors) > 200:
            print(f"- ... {len(total_errors) - 200} more", file=sys.stderr)
        return 1

    print(
        f"\nOK: validated {total_counts['rows']} rows and {total_counts['tool_calls']} tool calls "
        f"with {total_counts['unknown_args']} unknown arguments."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
