#!/usr/bin/env python3
"""Rewrite natural-language turns in a JSONL dataset with OpenRouter.

The script keeps tool calls, tool responses, and message ordering intact.
It uses a local memory of previously generated phrasing so the OpenRouter
model stays stateless between requests while the run as a whole remains
diverse and avoids repeated formulations.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:nitro")


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize_text(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'`“”‘’।,;:!?()\[\]{}]", "", value)
    return value


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def local_fallback_rewrite(source_text: str, language: str | None) -> str | None:
    text = source_text.strip()
    if not text:
        return None

    lang = (language or "").lower()
    rewrites: list[tuple[str, str]] = []
    if lang.startswith("fr"):
        rewrites = [
            (r"\bJe veux\b", "Je voudrais"),
            (r"\bJe désire\b", "Je souhaiterais"),
            (r"\bJe cherche\b", "Je voudrais trouver"),
            (r"\bVoulez-vous\b", "Souhaitez-vous"),
            (r"\bConfirmez-vous\b", "Pouvez-vous confirmer"),
            (r"\bTerminé\b", "C'est prêt"),
            (r"\bD’accord\b", "Très bien"),
            (r"\bD'accord\b", "Très bien"),
            (r"\bCommande enregistrée\b", "Commande validée"),
        ]
    elif lang.startswith("en"):
        rewrites = [
            (r"\bFind me\b", "Could you show me"),
            (r"\bI want\b", "I'd like"),
            (r"\bI need\b", "I'd like"),
            (r"\bCan I have\b", "Could I get"),
            (r"\bCan you\b", "Could you"),
            (r"\bDelivery: oui\b", "Delivery available"),
            (r"\bDelivery: no\b", "Delivery unavailable"),
            (r"\bPhone:\b", "Call:"),
            (r"\bTerminé:\b", "Done:"),
        ]
    else:
        rewrites = [
            (r"\bبغيت\b", "كنبغي"),
            (r"\bبغيتي\b", "كتفضلي"),
            (r"\bواش\b", "شحال"),
        ]

    rewritten = text
    for pattern, replacement in rewrites:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

    rewritten = rewritten.replace("  ", " ").strip()
    if normalize_text(rewritten) == normalize_text(text):
        if lang.startswith("en"):
            rewritten = f"Could you please {text[0].lower() + text[1:]}" if text else text
        elif lang.startswith("fr"):
            rewritten = f"Pouvez-vous {text[0].lower() + text[1:]}" if text else text
        else:
            rewritten = f"واش تقدر {text}"

    rewritten = rewritten.strip()
    return rewritten if normalize_text(rewritten) != normalize_text(text) else None


def load_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen": [], "recent": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"seen": [], "recent": []}
    seen = data.get("seen", [])
    recent = data.get("recent", [])
    if not isinstance(seen, list):
        seen = []
    if not isinstance(recent, list):
        recent = []
    return {"seen": [str(item) for item in seen], "recent": [str(item) for item in recent]}


def save_memory(path: Path, seen: set[str], recent: deque[str], max_seen: int = 10000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seen": sorted(seen)[-max_seen:],
        "recent": list(recent),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_response_schema() -> dict[str, Any]:
    return {
        "name": "rewrite_phrase",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewritten_text": {"type": "string"},
            },
            "required": ["rewritten_text"],
            "additionalProperties": False,
        },
    }


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": response_format,
        }
    req = Request(
        OPENROUTER_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/codex",
            "X-Title": "project-5-dialogue-rewriter",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


def extract_json(response: dict[str, Any]) -> dict[str, Any]:
    message = response["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return json.loads("".join(text_parts))
    raise ValueError("OpenRouter response did not contain JSON content")


def recent_examples(memory: deque[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return list(memory)[-limit:]


def build_prompt(
    source_text: str,
    role: str,
    language: str | None,
    recent_phrases: list[str],
    style: str,
) -> list[dict[str, Any]]:
    system = (
        "You rewrite one natural-language utterance for a tool-calling dataset. "
        "Preserve the exact meaning, intent, named entities, numbers, dates, IDs, tool names, and language. "
        "Do not add new facts, do not remove constraints, and do not explain what you are doing. "
        "Return a single JSON object only."
    )
    user_payload = {
        "role_to_rewrite": role,
        "source_text": source_text,
        "language": language,
        "style": style,
        "constraints": [
            "Keep the same meaning and task intent.",
            "Keep the same language as the source text.",
            "Keep numbers, dates, IDs, names, URLs, and tool names unchanged.",
            "Do not echo any of the recent phrases.",
            "Do not reuse the same opening or closing words if a safe alternative exists.",
            "Keep the rewrite concise and natural.",
        ],
        "recent_phrases_to_avoid": recent_phrases,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_fallback_prompt(
    source_text: str,
    role: str,
    language: str | None,
    recent_phrases: list[str],
    style: str,
) -> list[dict[str, Any]]:
    system = (
        "You rewrite one natural-language utterance for a tool-calling dataset. "
        "Preserve the exact meaning, intent, named entities, numbers, dates, IDs, tool names, and language. "
        "Do not add new facts, do not remove constraints, and do not explain what you are doing. "
        "Return only the rewritten text, with no markdown and no JSON."
    )
    user_payload = {
        "role_to_rewrite": role,
        "source_text": source_text,
        "language": language,
        "style": style,
        "recent_phrases_to_avoid": recent_phrases,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def rewrite_text(
    api_key: str,
    model: str,
    source_text: str,
    role: str,
    language: str | None,
    recent_memory: deque[str],
    seen_texts: set[str],
    style: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    max_source_similarity: float,
    retry_sleep: float,
) -> str:
    if not source_text.strip():
        return source_text

    source_norm = normalize_text(source_text)
    source_len = len(source_norm)
    if source_len < 18:
        effective_max_source_similarity = 0.995
    elif source_len < 40:
        effective_max_source_similarity = 0.985
    else:
        effective_max_source_similarity = max_source_similarity
    forbidden = {normalize_text(item) for item in recent_memory}
    last_error: str | None = None
    last_candidate: str | None = None

    for attempt in range(1, max_retries + 1):
        recent_phrases = recent_examples(recent_memory, min(len(recent_memory), 12))
        messages = build_prompt(source_text, role, language, recent_phrases, style)
        response = call_openrouter(
            api_key=api_key,
            model=model,
            messages=messages,
            response_format=build_response_schema(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            item = extract_json(response)
        except Exception as exc:
            last_error = str(exc)
            fallback_messages = build_fallback_prompt(source_text, role, language, recent_phrases, style)
            fallback_response = call_openrouter(
                api_key=api_key,
                model=model,
                messages=fallback_messages,
                response_format=None,
                temperature=max(0.4, temperature - 0.2),
                max_tokens=max_tokens,
            )
            fallback_message = fallback_response["choices"][0]["message"]
            fallback_content = fallback_message.get("content")
            if isinstance(fallback_content, list):
                parts = []
                for part in fallback_content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                fallback_content = "".join(parts) if parts else None
            if isinstance(fallback_content, str):
                rewritten = fallback_content.strip()
                rewritten = re.sub(r"^rewritten\s*:\s*", "", rewritten, flags=re.IGNORECASE)
                rewritten = rewritten.strip(" \t\r\n\"'")
                if rewritten:
                    item = {"rewritten_text": rewritten}
                else:
                    item = {}
            else:
                item = {}
            if not item:
                if attempt < max_retries:
                    time.sleep(retry_sleep * attempt)
                continue
        rewritten = item.get("rewritten_text")
        if not isinstance(rewritten, str):
            last_error = "rewritten_text is not a string"
        else:
            rewritten = rewritten.strip()
            rewritten_norm = normalize_text(rewritten)
            if not rewritten:
                last_error = "empty rewritten text"
            elif rewritten_norm == source_norm:
                last_error = "rewrite identical to source"
            elif rewritten_norm in forbidden:
                last_error = "rewrite repeats recent memory"
            elif similarity(source_text, rewritten) > effective_max_source_similarity:
                last_error = "rewrite too close to source"
                last_candidate = rewritten
            else:
                return rewritten

        if attempt < max_retries:
            time.sleep(retry_sleep * attempt)

    if last_candidate and source_len < 40:
        return last_candidate
    local_candidate = local_fallback_rewrite(source_text, language)
    if local_candidate:
        return local_candidate
    raise RuntimeError(f"Could not generate a sufficiently distinct rewrite: {last_error or 'unknown error'}")


def clean_message(message: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(message)
    content = cleaned.get("content")
    if isinstance(content, list):
        if len(content) == 1:
            cleaned["content"] = content[0]
        elif content and isinstance(content[0], str):
            cleaned["content"] = content[0]
        else:
            cleaned["content"] = json.dumps(content, ensure_ascii=False)
    elif isinstance(content, dict):
        cleaned["content"] = json.dumps(content, ensure_ascii=False)
    return cleaned


def rewrite_row(
    row: dict[str, Any],
    api_key: str,
    model: str,
    roles: set[str],
    memory: deque[str],
    seen_texts: set[str],
    style: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    max_source_similarity: float,
    retry_sleep: float,
) -> tuple[dict[str, Any], int]:
    row = dict(row)
    metadata = row.get("metadata", {})
    language_hint: str | None = None
    if isinstance(metadata, dict):
        language_value = metadata.get("language")
        if isinstance(language_value, str):
            language_hint = language_value

    changed = 0
    messages = []
    for raw_message in row.get("messages", []):
        if not isinstance(raw_message, dict):
            continue
        message = clean_message(raw_message)
        role = message.get("role")
        content = message.get("content")
        if role in roles and isinstance(content, str) and content.strip() and not message.get("tool_calls") and role != "tool":
            rewritten = rewrite_text(
                api_key=api_key,
                model=model,
                source_text=content,
                role=role,
                language=language_hint,
                recent_memory=memory,
                seen_texts=seen_texts,
                style=style,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                max_source_similarity=max_source_similarity,
                retry_sleep=retry_sleep,
            )
            if rewritten != content:
                changed += 1
                message["content"] = rewritten
                normalized = normalize_text(rewritten)
                seen_texts.add(normalized)
                memory.append(rewritten)
        messages.append(message)
    row["messages"] = messages
    return row, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/5k/all.jsonl")
    parser.add_argument("--output", default="data/5k/all_rewritten_openrouter.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--roles", default="user,assistant")
    parser.add_argument("--style", default="natural, varied, concise")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--max-source-similarity",
        type=float,
        default=0.92,
        help="Reject rewrites that are too close to the source above this similarity threshold.",
    )
    parser.add_argument("--retry-sleep", type=float, default=0.4)
    parser.add_argument(
        "--state-file",
        default=".cache/openrouter_rewrite_state.json",
        help="Local cache of seen and recent rewrites to avoid repetition.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=32,
        help="How many recent rewrites to keep in prompt memory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output file if it already exists and continue from the state file.",
    )
    args = parser.parse_args()

    load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    state = load_memory(state_path) if args.resume or state_path.exists() else {"seen": [], "recent": []}
    seen_texts = {normalize_text(text) for text in state.get("seen", []) if isinstance(text, str)}
    memory = deque(
        [text for text in state.get("recent", []) if isinstance(text, str)],
        maxlen=max(args.memory_size, 1),
    )
    start_index = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            start_index = sum(1 for line in handle if line.strip())

    roles = {part.strip() for part in args.roles.split(",") if part.strip()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    changed_messages = 0
    for idx, row in enumerate(rows, 1):
        if idx <= start_index:
            continue
        rewritten, changed = rewrite_row(
            row=row,
            api_key=api_key,
            model=args.model,
            roles=roles,
            memory=memory,
            seen_texts=seen_texts,
            style=args.style,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            max_source_similarity=args.max_source_similarity,
            retry_sleep=args.retry_sleep,
        )
        changed_messages += changed
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
        if idx % 10 == 0 or idx == len(rows):
            save_memory(state_path, seen_texts, memory)
            print(f"processed={idx} changed_messages={changed_messages} output={output_path}")

    save_memory(state_path, seen_texts, memory)
    print(f"done rows={len(rows)} changed_messages={changed_messages} output={output_path} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
