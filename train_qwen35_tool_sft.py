#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-4B tool-calling SFT with Unsloth + optional ClearML.

Pipeline:
1) Load train / validation / test JSONL conversations.
2) Run a raw baseline eval on validation BEFORE formatting/training.
3) Format conversations with tokenizer.apply_chat_template(..., tools=...).
4) Train LoRA / optional QLoRA adapter.
5) Save adapter and run post-training eval.

Expected JSONL row:
{
  "id": "case_001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "name": "...", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}

from __future__ import annotations

try:
    from unsloth import FastModel
except Exception:
    from unsloth import FastLanguageModel as FastModel  # older Unsloth fallback

import argparse
import copy
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer




# -----------------------------
# IO helpers
# -----------------------------


def read_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {e}") from e
    return rows

def get_text_tokenizer(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    
def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Tool registry loading
# -----------------------------


def _normalize_tool(t: dict) -> Optional[dict]:
    if not isinstance(t, dict):
        return None
    if "function" in t and isinstance(t["function"], dict):
        fn = t["function"]
        return {
            "type": t.get("type", "function"),
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    return {
        "type": t.get("type", "function"),
        "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def load_tool_registry(path: Optional[str]) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")

    # JSON file containing list or {tools:[...]}
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        tools = data.get("tools", data) if isinstance(data, dict) else data
        return [x for x in (_normalize_tool(t) for t in tools) if x]

    # YAML support if PyYAML is installed
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(text)
        tools = data.get("tools", data) if isinstance(data, dict) else data
        return [x for x in (_normalize_tool(t) for t in tools) if x]

    # Markdown fallback: extract JSON code blocks or raw JSON arrays.
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates = blocks + [text]
    extracted: List[dict] = []
    for c in candidates:
        try:
            data = json.loads(c.strip())
        except Exception:
            continue
        tools = data.get("tools", data) if isinstance(data, dict) else data
        if isinstance(tools, list):
            extracted.extend([x for x in (_normalize_tool(t) for t in tools) if x])
    return extracted


def tool_names(tools: List[dict]) -> set[str]:
    return {t.get("function", {}).get("name", "") for t in tools if t.get("function")}


# -----------------------------
# Message normalization & formatting
# -----------------------------


def parse_arguments(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return args
    return args


def normalize_messages(messages: List[dict]) -> List[dict]:
    out: List[dict] = []
    id_to_name: Dict[str, str] = {}

    for m in messages:
        m = dict(m)
        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        # assistant.content must often be string for templates
        if role == "assistant" and m.get("content") is None:
            m["content"] = ""

        # Normalize tool calls into HF/OpenAI format
        if role == "assistant" and m.get("tool_calls"):
            new_calls = []
            for idx, tc in enumerate(m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                if "function" in tc and isinstance(tc["function"], dict):
                    fn = dict(tc["function"])
                    name = fn.get("name", "")
                    args = parse_arguments(fn.get("arguments", {}))
                    call_id = tc.get("id") or tc.get("tool_call_id") or f"call_{idx}_{name}"
                    new_tc = {"id": call_id, "type": tc.get("type", "function"), "function": {"name": name, "arguments": args}}
                else:
                    name = tc.get("name", "")
                    args = parse_arguments(tc.get("arguments", {}))
                    call_id = tc.get("id") or tc.get("tool_call_id") or f"call_{idx}_{name}"
                    new_tc = {"id": call_id, "type": tc.get("type", "function"), "function": {"name": name, "arguments": args}}
                id_to_name[new_tc["id"]] = new_tc["function"]["name"]
                new_calls.append(new_tc)
            m["tool_calls"] = new_calls

        if role == "tool":
            if not m.get("name"):
                tcid = m.get("tool_call_id")
                m["name"] = id_to_name.get(tcid, "unknown_tool")
            if not isinstance(m.get("content"), str):
                m["content"] = json.dumps(m.get("content", {}), ensure_ascii=False)

        out.append(m)

    return out


def format_row(row: dict, tokenizer: Any, tools: List[dict], max_chars: Optional[int] = None) -> dict:
    messages = normalize_messages(row.get("messages", []))
    if not messages:
        return {"text": None, "id": row.get("id")}
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        # Some templates do not accept tools=None / tools=...
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(text, str):
        text = text.removeprefix("<bos>")
    if max_chars and len(text) > max_chars:
        return {"text": None, "id": row.get("id"), "skip_reason": "too_long_chars"}
    return {"text": text, "id": row.get("id"), "metadata": row.get("metadata", {})}


def build_text_dataset(rows: List[dict], tokenizer: Any, tools: List[dict], max_chars: Optional[int]) -> Dataset:
    formatted = [format_row(r, tokenizer, tools, max_chars=max_chars) for r in rows]
    formatted = [x for x in formatted if x.get("text")]
    return Dataset.from_list(formatted)


# -----------------------------
# Baseline / eval cases
# -----------------------------


def canonical_tool_call(msg: dict) -> Optional[dict]:
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    tc = normalize_messages([msg])[0]["tool_calls"][0]
    return {
        "type": "tool_call",
        "name": tc["function"].get("name"),
        "arguments": parse_arguments(tc["function"].get("arguments", {})) or {},
    }


def canonical_assistant(msg: dict) -> dict:
    tc = canonical_tool_call(msg)
    if tc:
        return tc
    return {"type": "assistant_message", "content": msg.get("content") or ""}


def build_eval_cases(rows: List[dict], require_user_before_target: bool = False) -> List[dict]:
    """Create next-assistant-turn cases from full dialogues."""
    cases = []
    for row in rows:
        messages = normalize_messages(row.get("messages", []))
        for i, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            context = messages[:i]
            if not context:
                continue
            if require_user_before_target and not any(x.get("role") == "user" for x in context):
                continue
            expected = canonical_assistant(m)
            # Useful for tool-call eval; also includes final answer eval as type-level only.
            cases.append({
                "id": f"{row.get('id', 'row')}::turn_{i}",
                "messages": context,
                "expected": expected,
                "metadata": row.get("metadata", {}),
            })
    return cases


def extract_first_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # fenced JSON
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # first object or array
    for start, end in [("{", "}"), ("[", "]")]:
        s = text.find(start)
        e = text.rfind(end)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return None


def normalize_prediction(raw: str) -> Optional[dict]:
    obj = extract_first_json(raw)
    if obj is None:
        return None
    # Qwen/tool templates often produce list of tool calls
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and ("name" in first or "function" in first):
            if "function" in first:
                return {"type": "tool_call", "name": first["function"].get("name"), "arguments": parse_arguments(first["function"].get("arguments", {})) or {}}
            return {"type": "tool_call", "name": first.get("name"), "arguments": parse_arguments(first.get("arguments", {})) or {}}
    if isinstance(obj, dict):
        if obj.get("type") == "tool_call":
            return {"type": "tool_call", "name": obj.get("name"), "arguments": parse_arguments(obj.get("arguments", {})) or {}}
        if "function" in obj:
            return {"type": "tool_call", "name": obj["function"].get("name"), "arguments": parse_arguments(obj["function"].get("arguments", {})) or {}}
        if "name" in obj and "arguments" in obj:
            return {"type": "tool_call", "name": obj.get("name"), "arguments": parse_arguments(obj.get("arguments", {})) or {}}
        return {"type": obj.get("type", "assistant_message"), "content": obj.get("content", raw)}
    return None


def score_prediction(pred: Optional[dict], expected: dict, allowed_tools: set[str]) -> dict:
    s = {
        "json_valid": pred is not None,
        "type_match": False,
        "tool_match": False,
        "required_args_match": False,
        "exact_args_match": False,
        "unknown_tool": False,
        "hallucinated_id": False,
        "exact_action_match": False,
    }
    if pred is None:
        return s
    s["type_match"] = pred.get("type") == expected.get("type")
    if pred.get("type") == "tool_call":
        s["unknown_tool"] = bool(allowed_tools and pred.get("name") not in allowed_tools)
    if expected.get("type") == "tool_call":
        s["tool_match"] = pred.get("type") == "tool_call" and pred.get("name") == expected.get("name")
        exp_args = expected.get("arguments") or {}
        pred_args = pred.get("arguments") or {}
        if not isinstance(pred_args, dict):
            pred_args = {}
        required = set(exp_args.keys())
        s["required_args_match"] = all(k in pred_args for k in required)
        s["exact_args_match"] = all(pred_args.get(k) == v for k, v in exp_args.items())
        # crude ID hallucination rule: ID-looking values in pred not in prompt context expected args.
        for v in pred_args.values():
            if isinstance(v, str) and re.search(r"\b(?:prod|biz|merchant|draft|user|order)_[A-Za-z0-9]+", v):
                if v not in json.dumps(exp_args, ensure_ascii=False):
                    s["hallucinated_id"] = True
        s["exact_action_match"] = s["type_match"] and s["tool_match"] and s["required_args_match"]
    else:
        s["exact_action_match"] = s["type_match"]
    return s


def generate_raw(model: Any, tokenizer: Any, messages: List[dict], tools: List[dict], max_new_tokens: int) -> str:
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, pad_token_id=tokenizer.eos_token_id)
    else:
        inputs = inputs.to(model.device)
        input_len = inputs.shape[-1]
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def log_to_clearml(report: dict, prefix: str = "", step: int = 0) -> None:
    """Log evaluation report as ClearML scalars."""
    try:
        from clearml import Task
        task = Task.current_task()
        if task is None:
            return
        logger = task.get_logger()
        scalar_keys = [
            "json_validity", "type_accuracy", "tool_accuracy",
            "required_args_accuracy", "exact_args_accuracy",
            "unknown_tool_rate", "hallucinated_id_rate", "exact_action_accuracy",
            "latency_sec", "total",
        ]
        for k in scalar_keys:
            if k in report:
                logger.report_scalar(
                    title=f"{prefix}_{k}" if prefix else k,
                    series=k,
                    value=report[k],
                    iteration=step,
                )
        # Per-pattern metrics
        for pattern, metrics in report.get("by_workflow_pattern", {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.report_scalar(
                        title=f"{prefix}_by_pattern" if prefix else "by_pattern",
                        series=f"{pattern}/{k}",
                        value=v,
                        iteration=step,
                    )
    except ImportError:
        pass
    except Exception as e:
        print(f"[clearml] Failed to log metrics: {e}")


def evaluate_model(model: Any, tokenizer: Any, rows: List[dict], tools: List[dict], out_dir: Path, name: str, limit: int, max_new_tokens: int) -> dict:
    cases = build_eval_cases(rows)
    if limit:
        cases = cases[:limit]
    allowed = tool_names(tools)
    preds = []
    counts = Counter()
    by_pattern = defaultdict(Counter)
    start = time.time()

    for i, case in enumerate(cases, 1):
        t0 = time.time()
        raw = generate_raw(model, tokenizer, case["messages"], tools, max_new_tokens=max_new_tokens)
        pred = normalize_prediction(raw)
        metrics = score_prediction(pred, case["expected"], allowed)
        elapsed = time.time() - t0
        for k, v in metrics.items():
            if v:
                counts[k] += 1
        pattern = case.get("metadata", {}).get("workflow_pattern", "unknown")
        for k, v in metrics.items():
            if v:
                by_pattern[pattern][k] += 1
        by_pattern[pattern]["total"] += 1
        preds.append({**case, "raw_output": raw, "prediction": pred, "metrics": metrics})
        if i % 25 == 0 or i == 1 or i == len(cases):
            avg_time = (time.time() - start) / i
            eta = avg_time * (len(cases) - i)
            running_json = counts["json_valid"] / i
            running_tool = counts["tool_match"] / i if counts["exact_action_match"] else 0
            print(f"[{name}] {i}/{len(cases)} | json={running_json:.0%} tool={running_tool:.0%} | {elapsed:.1f}s/step | ETA {eta:.0f}s")

    total = max(len(cases), 1)
    report = {
        "name": name,
        "total": len(cases),
        "latency_sec": round(time.time() - start, 3),
        "json_validity": counts["json_valid"] / total,
        "type_accuracy": counts["type_match"] / total,
        "tool_accuracy": counts["tool_match"] / total,
        "required_args_accuracy": counts["required_args_match"] / total,
        "exact_args_accuracy": counts["exact_args_match"] / total,
        "unknown_tool_rate": counts["unknown_tool"] / total,
        "hallucinated_id_rate": counts["hallucinated_id"] / total,
        "exact_action_accuracy": counts["exact_action_match"] / total,
        "by_workflow_pattern": {p: {k: (v / c["total"] if k != "total" else v) for k, v in c.items()} for p, c in by_pattern.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(preds, out_dir / f"{name}_predictions.jsonl")
    write_json(report, out_dir / f"{name}_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--tool-registry", default=None)
    ap.add_argument("--output-dir", default="runs/qwen35_tool_sft")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA mode. Not recommended by Unsloth for Qwen3.5; use only if VRAM forces it.")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--baseline-limit", type=int, default=100)
    ap.add_argument("--eval-max-new-tokens", type=int, default=512)
    ap.add_argument("--report-to", default="tensorboard", choices=["none", "tensorboard", "clearml", "wandb", "mlflow"])
    ap.add_argument("--clearml-project", default="JBUJB-Qwen35-ToolSFT")
    ap.add_argument("--clearml-task", default=None)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    eval_dir = out_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.load_in_4bit:
        print("WARNING: You enabled --load-in-4bit / QLoRA. Current Unsloth Qwen3.5 docs warn that QLoRA is not recommended for Qwen3.5 because quantization differences are higher than normal. Prefer bf16 LoRA when you have ~10GB+ VRAM for 4B.")

    if args.report_to == "clearml":
        os.environ.setdefault("CLEARML_PROJECT", args.clearml_project)
        os.environ.setdefault("CLEARML_TASK", args.clearml_task or f"qwen35_tool_sft_{int(time.time())}")
        os.environ.setdefault("CLEARML_LOG_MODEL", "True")

    tools = load_tool_registry(args.tool_registry)
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.validation)
    test_rows = read_jsonl(args.test) if args.test else []

    dtype = None
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16

    print(f"Loading model: {args.model}")
    model, processor_or_tokenizer  = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
    )
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    print("Running baseline eval BEFORE dataset text formatting/training...")
    baseline_report = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        rows=val_rows,
        tools=tools,
        out_dir=eval_dir,
        name="baseline_validation_raw",
        limit=args.baseline_limit,
        max_new_tokens=args.eval_max_new_tokens,
    )
    if args.report_to == "clearml":
        log_to_clearml(baseline_report, prefix="baseline")

    print("Preparing text datasets...")
    train_ds = build_text_dataset(train_rows, tokenizer, tools, max_chars=None)
    val_ds = build_text_dataset(val_rows, tokenizer, tools, max_chars=None)
    print(f"train examples: {len(train_ds)} | validation examples: {len(val_ds)} | tools: {len(tools)}")

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    report_to = [] if args.report_to == "none" else [args.report_to]
    train_args = SFTConfig(
        dataset_text_field="text",
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        bf16=(args.dtype == "bf16" and torch.cuda.is_available()),
        fp16=(args.dtype == "fp16" and torch.cuda.is_available()),
        report_to=report_to,
        run_name=Path(args.output_dir).name,
    )

    config_snapshot = vars(args) | {"tools_count": len(tools), "baseline_report": baseline_report}
    write_json(config_snapshot, out_dir / "run_config.json")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
    )

    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    print("Running post-training eval...")
    sft_report = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        rows=val_rows,
        tools=tools,
        out_dir=eval_dir,
        name="sft_validation_raw",
        limit=args.baseline_limit,
        max_new_tokens=args.eval_max_new_tokens,
    )
    if args.report_to == "clearml":
        log_to_clearml(sft_report, prefix="sft")
        # Log delta (improvement)
        from clearml import Task
        task = Task.current_task()
        if task:
            logger = task.get_logger()
            for k in ["json_validity", "tool_accuracy", "exact_action_accuracy", "hallucinated_id_rate"]:
                delta = sft_report[k] - baseline_report[k]
                logger.report_scalar(title="delta_sft_vs_baseline", series=k, value=delta)
    if test_rows:
        test_report = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            rows=test_rows,
            tools=tools,
            out_dir=eval_dir,
            name="sft_test_raw",
            limit=args.baseline_limit,
            max_new_tokens=args.eval_max_new_tokens,
        )
        if args.report_to == "clearml":
            log_to_clearml(test_report, prefix="sft_test")


if __name__ == "__main__":
    main()
