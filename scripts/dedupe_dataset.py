#!/usr/bin/env python3
"""Deduplicate dialogue JSONL files by exact message sequence.

The script is intentionally opinionated for this repo:
- it can dedupe one or more independent groups of files;
- within each group, later files lose rows that already appeared earlier in the same group;
- alias files can be refreshed from their canonical source after deduplication.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dialogue_key(row: dict) -> str:
    return json.dumps(row.get("messages", []), ensure_ascii=False, sort_keys=True)


def dedupe_group(paths: list[Path]) -> tuple[int, int]:
    seen: set[str] = set()
    total_rows = 0
    total_removed = 0
    for path in paths:
        rows = load_rows(path)
        total_rows += len(rows)
        kept: list[dict] = []
        removed = 0
        for row in rows:
            key = dialogue_key(row)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(row)
        total_removed += removed
        write_rows(path, kept)
        print(f"{path}: kept={len(kept)} removed={removed}")
    return total_rows, total_removed


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Comma-separated list of JSONL paths that should share a dedupe set.",
    )
    parser.add_argument(
        "--refresh-alias",
        action="append",
        default=[],
        help="Copy canonical_path:alias_path after deduplication, preserving exact alias mirrors.",
    )
    args = parser.parse_args()

    for raw_group in args.group:
        paths = [resolve_path(item.strip()) for item in raw_group.split(",") if item.strip()]
        if not paths:
            continue
        total_rows, total_removed = dedupe_group(paths)
        print(f"group_summary: rows={total_rows} removed={total_removed}")

    for raw_pair in args.refresh_alias:
        if ":" not in raw_pair:
            raise SystemExit(f"Invalid --refresh-alias value: {raw_pair!r}")
        canonical_raw, alias_raw = raw_pair.split(":", 1)
        canonical = resolve_path(canonical_raw.strip())
        alias = resolve_path(alias_raw.strip())
        shutil.copyfile(canonical, alias)
        print(f"refreshed alias {alias} from {canonical}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
