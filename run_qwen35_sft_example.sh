#!/usr/bin/env bash
set -euo pipefail

export CLEARML_PROJECT="JBUJB-Qwen35-ToolSFT"
export CLEARML_TASK="qwen35_4b_tool_sft_v1"
export CLEARML_LOG_MODEL=True

python train_qwen35_tool_sft.py \
  --model unsloth/Qwen3.5-4B \
  --train outputs/splits/train.jsonl \
  --validation outputs/splits/validation.jsonl \
  --test outputs/splits/test.jsonl \
  --tool-registry data/tool_registry.json \
  --output-dir runs/qwen35_4b_tool_sft_v1 \
  --max-seq-length 4096 \
  --dtype bf16 \
  --lora-r 64 \
  --lora-alpha 128 \
  --batch-size 2 \
  --grad-accum 8 \
  --epochs 1 \
  --lr 2e-5 \
  --baseline-limit 100 \
  --report-to clearml
