#!/usr/bin/env python3
"""Generate ToolWeave-style tool-calling dialogues with OpenRouter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are JBUJB assistant, a food ordering and restaurant discovery agent. Use only available tools. "
    "Never invent IDs — always resolve them through search or resolution tools. Ask for clarification when "
    "required information is missing (location, restaurant name, ambiguous results). Mutating order actions "
    "(create, add, remove, update, clear) require explicit user confirmation. Respond in the same language "
    "as the user (French, English, or Moroccan Arabic). Be concise, friendly, and helpful."
)

WORKFLOW_PATTERNS = {
    "clarification_required": "The user request is underspecified. Ask one concise clarification question and do not call tools.",
    "dish_details": "The user wants details about a dish. Search, then resolve dish details, then summarize succinctly.",
    "restaurant_browse_by_criteria": "The user wants restaurant recommendations by cuisine or criteria. Search restaurants, inspect one result, and summarize.",
    "discovery_only": "The user is browsing and should receive a discovery answer without mutation.",
    "ordering_from_search": "The user wants to order a dish found through search. Search, validate availability, ask for confirmation, then create a draft order.",
    "ordering_from_menu": "The user wants to order from a restaurant menu. Resolve restaurant, fetch menu, validate product, ask for confirmation, then create a draft order.",
    "cart_management": "The user wants to modify an existing cart. Use add_to_order or remove_from_order only after confirming context from prior tool output.",
    "full_purchase": "The user wants a complete purchase flow from restaurant resolution to draft order creation with confirmation.",
    "refusal_or_blocked": "The request must be refused because it is unsafe, unsupported, or outside the assistant's capabilities.",
    "nearby_discovery": "The user asks for nearby places. Use geolocation-based search and summarize results.",
    "menu_browsing": "The user wants to browse a menu. Resolve the restaurant first, fetch the menu, and summarize the relevant items.",
    "multi_intent": "The user asks for two related tasks in one message. Handle the first with tools, then address the second clearly.",
    "pre_order_validation": "The user wants to know if a dish is available and how much it costs. Search and validate before answering.",
    "restaurant_details_by_name": "The user asks about a named restaurant. Resolve the restaurant and return details.",
    "promo_hunting": "The user asks about promotions or discounts. Search active promotions and summarize applicable offers.",
    "location_discovery": "The user asks for restaurants in a location. Search with geo context and summarize the top choices.",
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_registry(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", data) if isinstance(data, dict) else data
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def load_seed_examples(path: Path, limit: int = 32) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pattern = row.get("metadata", {}).get("workflow_pattern")
        if pattern and pattern not in examples:
            examples[pattern] = row
        if len(examples) >= limit:
            break
    return examples


def build_schema() -> dict[str, Any]:
    message_schema = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
            "content": {"type": ["string", "null"]},
            "tool_calls": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "string"},
                            },
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "type", "function"],
                    "additionalProperties": False,
                },
            },
            "tool_call_id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
        },
        "required": ["role", "content", "tool_calls", "tool_call_id", "name"],
        "additionalProperties": False,
    }

    return {
        "name": "toolweave_dialogue",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "workflow_pattern": {"type": "string"},
                "language": {"type": "string", "enum": ["fr", "en", "ar"]},
                "user_goal": {"type": "string"},
                "has_confirmation": {"type": "boolean"},
                "has_parameter_provenance": {"type": "boolean"},
                "simulated_outputs": {"type": "boolean"},
                "tools_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "messages": {
                    "type": "array",
                    "items": message_schema,
                    "minItems": 3,
                },
            },
            "required": [
                "workflow_pattern",
                "language",
                "user_goal",
                "has_confirmation",
                "has_parameter_provenance",
                "simulated_outputs",
                "tools_used",
                "messages",
            ],
            "additionalProperties": False,
        },
    }


def extract_json(response: dict) -> dict:
    message = response["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("OpenRouter response did not contain JSON content")


def canonicalize_messages(messages: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        item = {"role": role, "content": content}
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                call_item = {
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    },
                }
                calls.append(call_item)
            if calls:
                item["tool_calls"] = calls
        if role == "tool":
            item["tool_call_id"] = message.get("tool_call_id")
            item["name"] = message.get("name")
        cleaned.append(item)
    if not cleaned or cleaned[0].get("role") != "system":
        cleaned.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return cleaned


def canonical_language(value: Any) -> Any:
    if value == "ar_darija":
        return "ar"
    return value


def validate_dialogue(item: dict, tool_names: set[str]) -> list[str]:
    errors: list[str] = []
    messages = item.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ["missing messages"]
    if messages[0].get("role") != "system":
        errors.append("first message must be system")

    metadata = item.get("metadata", {})
    metadata_language = canonical_language(metadata.get("language")) if isinstance(metadata, dict) else None
    if metadata_language not in {"fr", "en", "ar"}:
        errors.append("invalid metadata language")

    pending_tool_calls = 0
    saw_user = False
    for idx, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            errors.append(f"message {idx} is not an object")
            continue
        role = message.get("role")
        if role == "user":
            saw_user = True
            if pending_tool_calls > 0:
                errors.append(f"user turn {idx} appears before all tool calls are answered")
        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if message.get("content") not in {None, ""}:
                    errors.append(f"assistant tool-call turn {idx} should not include content")
                for call in tool_calls:
                    if not isinstance(call, dict):
                        errors.append(f"assistant tool-call turn {idx} contains a non-object call")
                        continue
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    name = fn.get("name")
                    args = fn.get("arguments")
                    if not name:
                        errors.append(f"assistant tool-call turn {idx} is missing a function name")
                    elif tool_names and name not in tool_names:
                        errors.append(f"assistant tool-call turn {idx} uses unknown tool {name}")
                    if not isinstance(args, dict):
                        errors.append(f"assistant tool-call turn {idx} arguments must be an object")
                pending_tool_calls += len(tool_calls)
            elif pending_tool_calls > 0:
                errors.append(f"assistant turn {idx} appears before all tool calls are answered")
        elif role == "tool":
            if pending_tool_calls <= 0:
                errors.append(f"tool turn {idx} has no preceding tool call")
            else:
                pending_tool_calls -= 1
            if not message.get("tool_call_id"):
                errors.append(f"tool turn {idx} missing tool_call_id")
            if not message.get("name"):
                errors.append(f"tool turn {idx} missing tool name")
            if not isinstance(message.get("content"), str):
                errors.append(f"tool turn {idx} content must be a string")
        else:
            errors.append(f"message {idx} has invalid role {role!r}")

    if not saw_user:
        errors.append("missing user turn")
    if pending_tool_calls > 0:
        errors.append("unresolved tool calls at end of dialogue")
    return errors


def call_openrouter(api_key: str, model: str, messages: list[dict], response_format: dict[str, Any], temperature: float, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": response_format,
        },
        "stream": False,
    }
    req = Request(
        OPENROUTER_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/codex",
            "X-Title": "project-5-toolweave-generator",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc


def pick_pattern(patterns: list[str], index: int) -> str:
    return patterns[index % len(patterns)]


def build_prompt(pattern: str, tools: list[dict], seed_example: dict[str, Any] | None, language: str | None) -> list[dict]:
    registry_summary = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in tools)
    seed_block = ""
    if seed_example:
        messages = seed_example.get("messages", [])
        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        final_assistant = next((m.get("content") for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")), None)
        seed_block = json.dumps(
            {
                "workflow_pattern": seed_example.get("metadata", {}).get("workflow_pattern"),
                "language": seed_example.get("metadata", {}).get("language"),
                "tools_used": seed_example.get("metadata", {}).get("tools_used", []),
                "first_user": first_user,
                "final_assistant": final_assistant,
            },
            ensure_ascii=False,
        )
    system = (
        "You generate tool-calling training dialogues for a food-ordering agent. "
        "Follow a ToolWeave-style pipeline: choose a user goal, preserve parameter provenance, "
        "use only tools from the registry, ask clarification when information is missing, "
        "require confirmation before mutating actions, and keep the dialogue executable. "
        "Output a single JSON object that matches the schema exactly."
    )
    user = {
        "pattern": pattern,
        "pattern_instruction": WORKFLOW_PATTERNS[pattern],
        "registry": registry_summary,
        "preferred_language": language,
        "seed_example": seed_block or None,
        "constraints": [
            "Use tool calls only for tools present in the registry.",
            "Tool arguments must only contain fields allowed by the schema.",
            "Do not invent IDs; they must come from previous tool outputs.",
            "Keep messages concise and natural.",
            "Keep mock tool output JSON compact and realistic.",
            "If the workflow is mutating, include a confirmation turn before the final mutation tool call.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--output", default="data/openrouter_toolweave_generated.jsonl")
    parser.add_argument("--registry", default="data/tool_registry.json")
    parser.add_argument("--seed-data", default="data/5k/all.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default=None, choices=["fr", "en", "ar"], nargs="?")
    args = parser.parse_args()

    load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    registry_path = PROJECT_ROOT / args.registry if not Path(args.registry).is_absolute() else Path(args.registry)
    seed_path = PROJECT_ROOT / args.seed_data if not Path(args.seed_data).is_absolute() else Path(args.seed_data)
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    tools = load_registry(registry_path)
    if not tools:
        raise SystemExit(f"No tools loaded from {registry_path}")
    tool_names = {tool.get("name", "") for tool in tools if isinstance(tool, dict)}
    seed_examples = load_seed_examples(seed_path)

    rnd = random.Random(args.seed)
    patterns = list(WORKFLOW_PATTERNS.keys())
    rnd.shuffle(patterns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    generated = []
    seen_hashes: set[str] = set()
    candidate_idx = 0
    max_candidates = max(args.count * 20, args.count + 10)
    while len(generated) < args.count and candidate_idx < max_candidates:
        pattern = pick_pattern(patterns, candidate_idx)
        candidate_idx += 1
        seed_example = seed_examples.get(pattern)
        messages = build_prompt(pattern, tools, seed_example, args.language)
        response = call_openrouter(
            api_key=api_key,
            model=args.model,
            messages=messages,
            response_format=build_schema(),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        item = extract_json(response)
        item.setdefault("workflow_pattern", pattern)
        item["language"] = canonical_language(item.get("language", args.language or "fr"))
        if item["language"] not in {"fr", "en", "ar"}:
            item["language"] = args.language or "fr"
        item.setdefault("has_confirmation", False)
        item.setdefault("has_parameter_provenance", True)
        item.setdefault("simulated_outputs", True)
        item.setdefault("tools_used", [])
        if "messages" not in item or not isinstance(item["messages"], list):
            raise ValueError(f"Invalid OpenRouter response for pattern {pattern}")
        item["messages"] = canonicalize_messages(item["messages"])
        item["simulated_outputs"] = True
        item["metadata"] = {
            "workflow_pattern": item["workflow_pattern"],
            "language": item["language"],
            "tools_used": item.get("tools_used", []),
            "has_confirmation": item.get("has_confirmation", False),
            "has_parameter_provenance": item.get("has_parameter_provenance", True),
            "simulated_outputs": item.get("simulated_outputs", True),
        }
        dialogue_hash = hashlib.sha256(
            json.dumps(item["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        validation_errors = validate_dialogue(item, tool_names)
        if dialogue_hash in seen_hashes:
            validation_errors.append("duplicate dialogue")
        if validation_errors:
            print(f"Skipping invalid dialogue for pattern {pattern}: {'; '.join(validation_errors[:3])}")
            continue

        item["id"] = f"openrouter_{pattern}_{len(generated):04d}"
        item["metadata"]["plan_id"] = item["id"].replace("openrouter_", "plan_")
        item["metadata"]["goal_id"] = item["id"].replace("openrouter_", "")
        generated.append(item)
        seen_hashes.add(dialogue_hash)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        time.sleep(0.2)

    if len(generated) < args.count:
        raise RuntimeError(f"Only generated {len(generated)} valid dialogues out of requested {args.count}")

    print(f"generated {len(generated)} dialogues -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
