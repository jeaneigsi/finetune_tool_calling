#!/usr/bin/env bash
set -euo pipefail

export CLEARML_PROJECT="JBUJB-Qwen35-ToolSFT"
export CLEARML_TASK="qwen35_4b_tool_sft_v2"
export CLEARML_LOG_MODEL=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python train_qwen35_tool_sft.py \
  --model unsloth/Qwen3.5-4B \
  --train data/5k/train.jsonl \
  --validation data/5k/validation.jsonl \
  --test data/5k/evaluation.jsonl \
  --tool-registry data/tool_registry.json \
  --output-dir runs/qwen35_4b_tool_sft_v2 \
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
