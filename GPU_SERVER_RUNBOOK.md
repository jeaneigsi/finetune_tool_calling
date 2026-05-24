# Dedicated GPU Runbook

This repo is set up so the dedicated GPU server does the heavy lifting.
The VPS is only for dataset cleanup, script validation, and repository hygiene.

## 1. Environment

Use a fresh Python 3.11 or 3.12 environment with CUDA-enabled PyTorch.

Install the project dependencies:

```bash
pip install -r requirements_qwen35_sft.txt
```

If the machine has strong CUDA support, prefer `bf16` and `flash_attention_2`.
If flash-attn is unavailable, use `--attn-implementation sdpa`.
The repo defaults to non-thinking chat templates for function calling; use `--enable-thinking` only if you need the Qwen3.5 reasoning trace.

## 2. Validate the corpus

Run the dataset validator before training:

```bash
python scripts/validate_dataset.py
```

You can also run the full GPU pipeline wrapper after the environment is ready:

```bash
bash scripts/run_gpu_pipeline.sh runs/qwen35_4b_tool_sft_v2
```

Expected result:

- no schema errors
- no unknown args
- no unresolved tool calls

## 3. Train

Recommended command:

```bash
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
```

If you need a quicker smoke run:

```bash
python train_qwen35_tool_sft.py \
  --train data/5k/train.jsonl \
  --validation data/5k/validation.jsonl \
  --tool-registry data/tool_registry.json \
  --output-dir runs/qwen35_4b_tool_sft_v2 \
  --dtype bf16 \
  --batch-size 2 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 5e-5 \
  --skip-baseline \
  --report-to clearml
```

## 4. Verify artifacts

After training finishes, verify the run:

```bash
python scripts/check_run_artifacts.py --run-dir runs/qwen35_4b_tool_sft_v2
```

Expected artifacts:

- `runs/qwen35_4b_tool_sft_v2/run_config.json`
- `runs/qwen35_4b_tool_sft_v2/adapter/adapter_config.json`
- `runs/qwen35_4b_tool_sft_v2/adapter/adapter_model.safetensors` or `adapter_model.bin`
- `runs/qwen35_4b_tool_sft_v2/eval/baseline_validation_raw_report.json`
- `runs/qwen35_4b_tool_sft_v2/eval/sft_validation_raw_report.json`

## 5. Smoke-test inference

Use the adapter directory directly with the inference script:

```bash
python inference.py \
  --checkpoint runs/qwen35_4b_tool_sft_v2/adapter \
  --base-model unsloth/Qwen3.5-4B \
  --registry data/tool_registry.json \
  --max-seq-length 2048
```

The smoke test should show:

- the model loads successfully
- a prompt returns either a normal answer or a structured tool call
- tool-call output can be parsed by the inference helper

The noninteractive smoke test used by the pipeline wrapper is:

```bash
python inference.py \
  --checkpoint runs/qwen35_4b_tool_sft_v2/adapter \
  --base-model unsloth/Qwen3.5-4B \
  --registry data/tool_registry.json \
  --max-seq-length 2048 \
  --prompt "I want to order 2 Raw Cakes from Veggie Delight." \
  --expect-tool create_order
```

## 6. What counts as done

The run is only considered verified when:

1. the dataset validator passes,
2. training completes,
3. the run artifact checker passes, and
4. inference loads the saved adapter successfully.
