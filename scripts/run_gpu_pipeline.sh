#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${1:-runs/qwen35_4b_tool_sft_v2}"

cd "$PROJECT_ROOT"

python scripts/validate_dataset.py

python train_qwen35_tool_sft.py \
  --model unsloth/Qwen3.5-4B \
  --train data/5k/train.jsonl \
  --validation data/5k/validation.jsonl \
  --test data/5k/evaluation.jsonl \
  --tool-registry data/tool_registry.json \
  --output-dir "$RUN_DIR" \
  --max-seq-length 2048 \
  --dtype bf16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --batch-size 2 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 5e-5 \
  --baseline-limit 100 \
  --report-to clearml

python scripts/check_run_artifacts.py --run-dir "$RUN_DIR" --require-test-report

python inference.py \
  --checkpoint "$RUN_DIR/adapter" \
  --base-model unsloth/Qwen3.5-4B \
  --registry data/tool_registry.json \
  --max-seq-length 2048 \
  --prompt "I want to order 2 Raw Cakes from Veggie Delight." \
  --expect-tool create_order
