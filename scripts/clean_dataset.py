#!/usr/bin/env python3
"""Canonicalize JBUJB dataset tool-call arguments against the tool registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", data) if isinstance(data, dict) else data
    registry: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if name:
            registry[name] = fn.get("parameters") or {"type": "object", "properties": {}}
    return registry


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def canonical_language(value: Any) -> Any:
    if value == "ar_darija":
        return "ar"
    return value


def clean_value(value: Any, schema: dict | None) -> Any:
    if not isinstance(schema, dict):
        return value
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        out: dict[str, Any] = {}
        for key, sub_value in value.items():
            if key == "qty" and "quantity" in properties and "quantity" not in value:
                out["quantity"] = clean_value(sub_value, properties.get("quantity"))
                continue
            if key == "business_id" and "merchant_id" in properties and "merchant_id" not in value:
                out["merchant_id"] = clean_value(sub_value, properties.get("merchant_id"))
                continue
            if key == "restaurant_name" and "name" in properties and "name" not in value:
                out["name"] = clean_value(sub_value, properties.get("name"))
                continue
            if key == "merchant" and "merchant_id" in properties and "merchant_id" not in value:
                out["merchant_id"] = clean_value(sub_value, properties.get("merchant_id"))
                continue
            if key not in properties:
                continue
            out[key] = clean_value(sub_value, properties.get(key))
        if "language" in out:
            out["language"] = canonical_language(out["language"])
        return out
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [clean_value(item, item_schema) for item in value]
    if schema_type == "string":
        return canonical_language(value)
    return value


def clean_message(message: dict, registry: dict[str, dict[str, Any]]) -> dict:
    message = dict(message)
    role = message.get("role")
    content = message.get("content")
    if role in {"user", "assistant"} and isinstance(content, list):
        if len(content) == 1:
            message["content"] = content[0]
        elif content and isinstance(content[0], str):
            message["content"] = content[0]
        else:
            message["content"] = json.dumps(content, ensure_ascii=False)
    if role in {"user", "assistant"} and isinstance(message.get("content"), dict):
        message["content"] = json.dumps(message["content"], ensure_ascii=False)
    return message


def clean_row(row: dict, registry: dict[str, dict[str, Any]]) -> dict:
    row = dict(row)
    if "language" in row:
        row["language"] = canonical_language(row["language"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        if "language" in metadata:
            metadata["language"] = canonical_language(metadata["language"])
        row["metadata"] = metadata

    cleaned_messages = []
    pending_calls: list[tuple[str, str | None]] = []
    call_index = 1

    for raw_message in row.get("messages", []):
        if not isinstance(raw_message, dict):
            continue
        message = clean_message(raw_message, registry)
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            new_calls = []
            seen_signatures = set()
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = function.get("name")
                schema = registry.get(name, {})
                args = clean_value(parse_json(function.get("arguments", {})), schema)
                signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                call_id = f"call_{call_index}_{name or 'tool'}"
                call_index += 1
                new_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args,
                        },
                    }
                )
                pending_calls.append((call_id, name))
            message["tool_calls"] = new_calls
            message["content"] = None
            message.pop("tool_call_id", None)
            message.pop("name", None)
        elif role == "tool":
            if pending_calls:
                call_id, name = pending_calls.pop(0)
                message["tool_call_id"] = call_id
                if not message.get("name"):
                    message["name"] = name
            if not isinstance(message.get("content"), str):
                message["content"] = json.dumps(message.get("content", {}), ensure_ascii=False)
            message.pop("tool_calls", None)
        else:
            message.pop("tool_calls", None)
            message.pop("tool_call_id", None)
            message.pop("name", None)
            if role == "assistant" and message.get("content") is None:
                message["content"] = ""

        cleaned_messages.append(message)

    row["messages"] = cleaned_messages
    return row


def process_file(path: Path, registry: dict[str, dict[str, Any]], inplace: bool = True, output: Path | None = None) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    out_lines = []
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        row = json.loads(line)
        cleaned = clean_row(row, registry)
        cleaned_line = json.dumps(cleaned, ensure_ascii=False)
        if cleaned_line != line:
            changed += 1
        out_lines.append(cleaned_line)
    if inplace:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    elif output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len([l for l in lines if l.strip()]), changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data/tool_registry.json")
    parser.add_argument("--paths", nargs="*", default=["data/train.jsonl", "data/validation.jsonl", "data/evaluation.jsonl", "data/5k/train.jsonl", "data/5k/validation.jsonl", "data/5k/evaluation.jsonl", "data/5k/val.jsonl", "data/5k/eval.jsonl", "data/5k/all.jsonl"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    registry = load_registry(registry_path)

    total_rows = 0
    total_changed = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            continue
        rows, changed = process_file(path, registry, inplace=not args.dry_run)
        total_rows += rows
        total_changed += changed
        print(f"{path}: rows={rows} changed={changed}")

    print(f"total_rows={total_rows} changed_rows={total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
