#!/usr/bin/env python3
"""Check that a fine-tuning run produced the artifacts needed for inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def check_exists(path: Path, required: bool = True) -> bool:
    if path.exists():
        return True
    if required:
        print(f"missing: {path}", file=sys.stderr)
    return False


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs/qwen35_4b_tool_sft_v2")
    parser.add_argument("--require-test-report", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    adapter_dir = run_dir / "adapter"
    eval_dir = run_dir / "eval"
    checkpoints_dir = run_dir / "checkpoints"

    ok = True
    ok &= check_exists(run_dir)
    ok &= check_exists(run_dir / "run_config.json")
    ok &= check_exists(adapter_dir)
    ok &= check_exists(adapter_dir / "adapter_config.json")
    ok &= check_exists(adapter_dir / "tokenizer.json")
    ok &= check_exists(adapter_dir / "tokenizer_config.json")
    ok &= check_exists(adapter_dir / "chat_template.jinja")
    adapter_weight_found = any((adapter_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not adapter_weight_found:
        print(f"missing: adapter weights under {adapter_dir}", file=sys.stderr)
        ok = False

    ok &= check_exists(eval_dir / "baseline_validation_raw_report.json")
    ok &= check_exists(eval_dir / "sft_validation_raw_report.json")
    if args.require_test_report:
        ok &= check_exists(eval_dir / "sft_test_raw_report.json")

    if checkpoints_dir.exists():
        checkpoint_dirs = sorted(p for p in checkpoints_dir.iterdir() if p.is_dir())
        if not checkpoint_dirs:
            print(f"missing: no checkpoints found in {checkpoints_dir}", file=sys.stderr)
            ok = False

    if check_exists(run_dir / "run_config.json"):
        config = load_json(run_dir / "run_config.json")
        print(
            json.dumps(
                {
                    "output_dir": str(run_dir),
                    "model": config.get("model"),
                    "train": config.get("train"),
                    "validation": config.get("validation"),
                    "test": config.get("test"),
                    "tools_count": config.get("tools_count"),
                },
                ensure_ascii=False,
            )
        )

    if ok:
        print(f"OK: verified run artifacts under {run_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
