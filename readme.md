Voici un `README.md` complet que tu peux mettre directement à la racine de ton repo.

````md
# Tool Dataset Weaver — Synthetic Tool-Calling Dataset Generator for Reasoning Models

## Overview

**Tool Dataset Weaver** is a framework for generating, validating, evaluating, and fine-tuning high-quality multi-turn tool-calling datasets.

The project is inspired by the **ToolWeave** methodology: instead of generating tool-calling conversations directly from prompts, the dataset is built through a structured pipeline:

```text
Tool Registry
→ Tool Graph
→ User Goals
→ Workflows
→ Executable Plans
→ Simulated Tool Outputs
→ Multi-turn Dialogues
→ Validation
→ Baseline Evaluation
→ Fine-tuning
→ Post-training Evaluation
````

The objective is to train small language models, such as **Qwen3.5-4B**, to behave like reliable reasoning agents that can:

```text
understand the user request
identify missing information
select the correct tool
avoid inventing IDs
follow tool dependencies
ask clarification when needed
respect confirmation rules
recover from tool errors
produce valid tool calls
```

This repository is not just a dataset generator. It is a full experimental stack for building a domain-specific tool-calling model.

---

## Why this project exists

Most tool-calling datasets are generated in a naïve way:

```text
User request → tool call
```

This is not enough for real agentic behavior.

A model trained only on direct mappings may learn to call tools, but it often fails when the task requires reasoning across multiple steps:

```text
resolve restaurant
→ search product
→ verify availability
→ ask confirmation
→ create draft order
```

The main problems we want to avoid are:

```text
parameter hallucination
wrong tool order
missing clarification
unsafe mutation without confirmation
invalid JSON
tool calls without context
IDs invented by the model
datasets that look correct but are not executable
```

For example, a bad dataset may teach the model to do this:

```json
{
  "type": "tool_call",
  "name": "create_order",
  "arguments": {
    "items": [
      {
        "product_id": "prod_123",
        "quantity": 2
      }
    ]
  }
}
```

even if `prod_123` was never produced by a previous tool.

This repository fixes that by forcing every tool argument to have a clear provenance.

---

## Core idea

The core principle is simple:

> A tool-calling dataset should be compiled like a program, not improvised like a conversation.

Every tool call must be explainable from the current state:

```text
argument source ∈ {
  user_input,
  session_context,
  tool_default,
  system_default,
  previous_tool_output,
  confirmation_turn
}
```

If an argument has no valid source, the example must be rejected.

This is especially important for IDs:

```text
merchant_id
product_id
draft_id
user_id
order_id
```

The model must never learn to invent these values.

---

## Project goal

The main goal of this repo is to build a reliable fine-tuning pipeline for a small reasoning model specialized in tool use.

The current target model is:

```text
unsloth/Qwen3.5-4B
```

The training strategy is:

```text
1. Generate structured tool-calling conversations
2. Validate the dataset before training
3. Run baseline evaluation on the base model
4. Fine-tune with LoRA / QLoRA using Unsloth
5. Evaluate the fine-tuned model
6. Compare baseline vs fine-tuned performance
7. Analyze errors
8. Regenerate better data based on failures
```

The first objective is not to maximize benchmark scores.
The first objective is to build a model that behaves safely and predictably in a real tool-calling environment.

---

## Repository responsibilities

This repository handles five major responsibilities.

### 1. Dataset generation

The repo generates multi-turn conversations from structured tool workflows.

It does not generate conversations directly. It first creates intermediate representations:

```text
tool registry
tool graph
workflow pattern
user goal
executable plan
simulated outputs
dialogue
```

This allows the dataset to remain auditable.

### 1b. Dialogue rewriting with OpenRouter

When you want lexical diversity without changing the tool logic, the repo can rewrite the natural-language turns with OpenRouter.

The rewriter:

- keeps `tool_calls` and `tool` messages unchanged
- rewrites only natural-language content
- carries a local memory of prior outputs so the prompts stay stateless but the run stays diverse
- rejects outputs that are too close to the source or too similar to prior generations

Default usage:

```bash
python scripts/rewrite_dialogues_openrouter.py \
  --input data/5k/all.jsonl \
  --output data/5k/all_rewritten_openrouter.jsonl
```

It expects `OPENROUTER_API_KEY` in `.env`, and you can override the model with `OPENROUTER_MODEL`.

---

### 2. Dataset validation

The repo validates every generated dialogue before training.

The validator checks:

```text
JSON validity
message structure
role order
tool existence
required arguments
argument types
unknown arguments
parameter provenance
empty IDs
confirmation before mutation
dangerous tool blocking
workflow order
tool output consistency
```

A dialogue should not enter training if it violates the rules.

---

### 3. Baseline evaluation

Before fine-tuning, the base model is tested on the validation or test dataset.

This answers the question:

> How good is the base model before training?

The baseline evaluation measures:

```text
json_validity
action_type_accuracy
tool_accuracy
required_args_accuracy
exact_args_accuracy
exact_action_accuracy
hallucinated_id_rate
confirmation_policy_accuracy
```

Without baseline evaluation, the fine-tuning result cannot be trusted.

---

### 4. Fine-tuning

The repo includes a training script for Qwen3.5-4B using Unsloth.

The current training strategy is:

```text
bf16 LoRA when the GPU supports it
QLoRA only if memory is limited
ClearML logging for experiment tracking
baseline evaluation before training
post-training evaluation after training
```

For Qwen3.5-4B, bf16 LoRA is preferred when enough VRAM is available.

---

### 5. Experiment tracking

The repo supports experiment monitoring with ClearML.

ClearML is used to track:

```text
training loss
eval loss
learning rate
gradient norm
training runtime
baseline metrics
post-SFT metrics
prediction files
error analysis files
model artifacts
```

The goal is not only to know whether the loss decreased.
The goal is to know whether the model became better at tool reasoning.

---

## ToolWeave-inspired pipeline

The project follows a ToolWeave-style generation process.

### Step 1 — Tool Registry

The tool registry is the source of truth for all tools.

It defines:

```text
tool name
description
input schema
required parameters
output schema
tool family
side effects
confirmation rules
security rules
```

Example:

```json
{
  "type": "function",
  "function": {
    "name": "resolve_restaurant",
    "description": "Resolve a free-form restaurant name into an exact business identifier.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        },
        "language": {
          "type": "string"
        },
        "limit": {
          "type": "integer"
        }
      },
      "required": ["name"]
    }
  },
  "metadata": {
    "family": "DETAILS",
    "side_effect": false,
    "requires_confirmation": false,
    "produces": ["business_id", "business_name", "confidence"]
  }
}
```

The registry is used by both the generator and the evaluator.

---

### Step 2 — Tool Graph

The tool graph defines how tools can be chained.

It describes valid data flow between tool outputs and tool inputs.

Example:

```yaml
edges:
  - from: resolve_restaurant
    to: get_restaurant_menu
    dataflow:
      business_id: merchant_id
    reason: get_restaurant_menu requires a merchant_id produced by resolve_restaurant.

  - from: search_food
    to: get_food_details
    dataflow:
      product_id: product_id
    reason: get_food_details requires a product_id produced by search_food.
```

This prevents invalid chains such as:

```text
get_restaurant_menu before resolve_restaurant
create_order before product_id exists
check_food_available with empty IDs
```

---

### Step 3 — User Goals

A user goal is an abstract task.

Example:

```json
{
  "goal_id": "goal_0001",
  "workflow_pattern": "ordering_from_named_restaurant",
  "goal": "Create a draft order for two chicken tacos from Tacos de Lyon after explicit confirmation.",
  "entities": {
    "restaurant_name": "Tacos de Lyon",
    "item_name": "tacos poulet",
    "quantity": 2,
    "language": "fr"
  },
  "constraints": [
    "Do not invent merchant_id",
    "Do not invent product_id",
    "Ask confirmation before create_order"
  ]
}
```

The user goal is not yet a dialogue.
It is the intent that will drive the workflow.

---

### Step 4 — Workflow Sampling

A workflow is the logical chain of actions required to satisfy the goal.

Example:

```text
resolve_restaurant
→ search_food
→ ask_confirmation
→ create_order
```

Workflow patterns currently include:

```text
discovery_only
details_with_resolution
ordering_from_named_restaurant
order_modification
clarification_required
failure_recovery
refusal_or_blocked
```

The workflow must be semantically valid.

---

### Step 5 — Executable Plan

The executable plan is the most important intermediate representation.

It specifies:

```text
which tool is called
which arguments are used
where each argument comes from
which outputs are expected
which preconditions must hold
```

Example:

```json
{
  "step": 2,
  "type": "tool_call",
  "tool": "search_food",
  "arguments": {
    "query": {
      "value": "tacos poulet",
      "source": "user_input"
    },
    "business_id": {
      "value": "$step1.output.business_id",
      "source": "previous_tool_output"
    },
    "language": {
      "value": "fr",
      "source": "system_default"
    }
  }
}
```

The rule is strict:

> No argument can appear in a tool call if it does not exist in the executable plan with a valid source.

---

### Step 6 — Tool Simulation

The repo simulates realistic tool outputs.

Example:

```json
{
  "resolved": true,
  "business_id": "biz_001",
  "business_name": "Tacos de Lyon",
  "confidence": 0.96
}
```

The simulator can also generate failure cases:

```text
restaurant not found
ambiguous restaurant
empty search result
product unavailable
restaurant closed
invalid draft_id
missing location
```

Failure cases are important because they teach the model how to recover instead of blindly continuing.

---

### Step 7 — Dialogue Synthesis

Only after the plan and simulated outputs are ready, the repo generates the final conversation.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are JBUJB assistant. Use only available tools. Never invent IDs."
    },
    {
      "role": "user",
      "content": "Je veux commander deux tacos poulet chez Tacos de Lyon"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "resolve_restaurant",
            "arguments": "{\"name\":\"Tacos de Lyon\",\"language\":\"fr\",\"limit\":5}"
          }
        }
      ]
    }
  ]
}
```

The dialogue is the final training artifact, but it is not the source of truth.
The source of truth is the structured plan behind it.

---

## Dataset format

The dataset uses JSONL format.

Each line is one dialogue:

```json
{
  "id": "dialogue_0001",
  "messages": [],
  "metadata": {}
}
```

The `messages` field follows a chat format compatible with common SFT pipelines:

```json
[
  {
    "role": "system",
    "content": "..."
  },
  {
    "role": "user",
    "content": "..."
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": []
  },
  {
    "role": "tool",
    "tool_call_id": "call_1",
    "name": "search_food",
    "content": "{}"
  },
  {
    "role": "assistant",
    "content": "..."
  }
]
```

The `metadata` field may contain:

```json
{
  "plan_id": "plan_0001",
  "goal_id": "goal_0001",
  "workflow_pattern": "ordering_from_named_restaurant",
  "language": "fr",
  "tools_used": ["resolve_restaurant", "search_food", "create_order"],
  "has_confirmation": true,
  "has_parameter_provenance": true,
  "simulated_outputs": true
}
```

---

## Dataset splits

The dataset should be split into:

```text
train.jsonl
validation.jsonl
test.jsonl
test_hard.jsonl
```

The split should avoid leakage.

If multiple dialogues come from the same `goal_id`, they must remain in the same split.

The goal is to avoid this situation:

```text
train: "Je veux commander deux tacos chez Tacos de Lyon"
test:  "Mets-moi deux tacos chez Tacos de Lyon"
```

If both examples come from the same goal, the model is not really being tested on generalization.

---

## Baseline evaluation

Before fine-tuning, the repo evaluates the base model.

The baseline answers:

```text
How well does the base model perform before training?
```

Metrics include:

```text
json_validity
action_type_accuracy
tool_accuracy
required_args_accuracy
exact_args_accuracy
exact_action_accuracy
hallucinated_id_rate
confirmation_policy_accuracy
```

This baseline is essential because it gives a real comparison point.

Example:

```text
Base Qwen3.5-4B:
json_validity: 0.72
tool_accuracy: 0.51
required_args_accuracy: 0.44
confirmation_policy_accuracy: 0.36

Fine-tuned Qwen3.5-4B:
json_validity: 0.94
tool_accuracy: 0.83
required_args_accuracy: 0.78
confirmation_policy_accuracy: 0.91
```

---

## Fine-tuning strategy

The current fine-tuning script targets:

```text
unsloth/Qwen3.5-4B
```

The recommended training approach is:

```text
bf16 LoRA if GPU supports bf16
QLoRA only when VRAM is limited
```

Recommended GPU:

```text
RTX 4090 24 GB
RTX A6000 48 GB
RTX 6000 Ada 48 GB
RTX PRO 6000 Blackwell 96 GB  ← best option (96 GB, FP4/FP6/FP8 support)
L40S 48 GB
A100 80 GB
H100 80 GB
```

For a powerful GPU with enough VRAM, use bf16 LoRA.

Example command:

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
  --attn-implementation flash_attention_2 \
  --baseline-limit 100 \
  --report-to clearml \
  --clearml-project "JBUJB-Qwen35-ToolSFT"
```

For the dedicated GPU server workflow, see [GPU_SERVER_RUNBOOK.md](/home/jean/projects/doctorat/project-5/GPU_SERVER_RUNBOOK.md).
The end-to-end wrapper is [scripts/run_gpu_pipeline.sh](/home/jean/projects/doctorat/project-5/scripts/run_gpu_pipeline.sh).

**Fast iteration (skip baseline, focus on training):**

```bash
python train_qwen35_tool_sft.py \
  --train data/5k/train.jsonl \
  --validation data/5k/validation.jsonl \
  --tool-registry data/tool_registry.json \
  --output-dir runs/qwen35_4b_tool_sft_v2 \
  --dtype bf16 \
  --batch-size 2 --grad-accum 4 --epochs 3 --lr 5e-5 \
  --skip-baseline \
  --report-to clearml
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unsloth/Qwen3.5-4B` | Base model |
| `--train` | *(required)* | Training JSONL |
| `--validation` | *(required)* | Validation JSONL |
| `--test` | `None` | Test JSONL (optional) |
| `--tool-registry` | `None` | Tool registry JSON file |
| `--output-dir` | `runs/qwen35_tool_sft` | Output directory |
| `--max-seq-length` | `2048` | Max sequence length |
| `--load-in-4bit` | `False` | QLoRA mode (not recommended for Qwen3.5) |
| `--load-in-8bit` | `False` | 8-bit mode |
| `--dtype` | `bf16` | `bf16`, `fp16`, or `auto` |
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha |
| `--lora-dropout` | `0.0` | LoRA dropout |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--epochs` | `1.0` | Training epochs |
| `--max-steps` | `-1` | Max steps (-1 = auto) |
| `--lr` | `2e-5` | Learning rate |
| `--warmup-ratio` | `0.03` | Warmup ratio |
| `--weight-decay` | `0.01` | Weight decay |
| `--logging-steps` | `5` | Log every N steps |
| `--eval-steps` | `200` | Evaluate every N steps |
| `--save-steps` | `200` | Save checkpoint every N steps |
| `--baseline-limit` | `100` | Max eval examples for baseline |
| `--eval-max-new-tokens` | `512` | Max new tokens during eval |
| `--eval-workers` | `2` | Workers for eval tokenization |
| `--skip-baseline` | `False` | Skip baseline evaluation (faster iterations) |
| `--attn-implementation` | `flash_attention_2` | `flash_attention_2`, `sdpa`, `eager` |
| `--report-to` | `tensorboard` | `none`, `tensorboard`, `clearml`, `wandb`, `mlflow` |
| `--clearml-project` | `JBUJB-Qwen35-ToolSFT` | ClearML project name |
| `--clearml-task` | `None` | ClearML task name (auto-generated if None) |
| `--enable-thinking` | `False` | Enable Qwen3.5 thinking mode in chat templates |
| `--seed` | `3407` | Random seed |

### Speed optimizations (no quality impact)

| Flag | Effect | Speedup |
|------|--------|---------|
| `--attn-implementation flash_attention_2` | Fused attention kernel | 1.5-2x |
| `--lora-r 32 --lora-alpha 64` | Fewer trainable params | ~2x faster forward/backward |
| `--skip-baseline` | Skip pre-training eval | Saves ~10 min |
| `packing=True` (built-in) | Merge short examples | 20-40% less padding |
| `dataset_num_proc=4` (built-in) | Parallel tokenization | 4x faster prep |

---

## Training script responsibilities

The training script does more than fine-tuning.

It is responsible for:

```text
loading train / validation / test datasets
loading the tool registry
normalizing tool-call messages
formatting conversations with the Qwen chat template
running baseline evaluation before training
training the LoRA adapter
saving the adapter
running post-SFT evaluation
saving predictions and reports
logging metrics to ClearML
```

Expected outputs:

```text
runs/qwen35_4b_tool_sft_v1/
  adapter/
  eval/
    baseline_validation_raw_report.json
    baseline_validation_raw_predictions.jsonl
    sft_validation_raw_report.json
    sft_validation_raw_predictions.jsonl
    sft_test_raw_report.json
    sft_test_raw_predictions.jsonl
  run_config.json
```

---

## Model export & deployment

The training script supports 4 export modes. All are optional and can be combined.

### Export modes

```bash
# 1. LoRA adapter (always saved — use directly with vLLM)
# → runs/qwen35_tool_sft/adapter/

# 2. Float16 merged model (for vLLM / SGLang)
python train_qwen35_tool_sft.py ... --export-merged
# → runs/qwen35_tool_sft/merged/

# 3. GGUF quantized (for llama.cpp / Ollama)
python train_qwen35_tool_sft.py ... --export-gguf q4_k_m
# → runs/qwen35_tool_sft/merged/ + runs/qwen35_tool_sft/gguf/

# 4. HuggingFace Hub (1-click deploy — pushes both LoRA adapter + merged model)
python train_qwen35_tool_sft.py ... \
  --push-to-hub moncrolio/jbujb-qwen-tool-sft \
  --hf-token hf_xxx
# → https://huggingface.co/moncrolio/jbujb-qwen-tool-sft-lora   (LoRA adapter, <100MB)
# → https://huggingface.co/moncrolio/jbujb-qwen-tool-sft        (merged 16bit, ~9GB)
```

### Supported quant methods (for --export-gguf)

| Method | Description |
|--------|-------------|
| `q8_0` | Fast conversion, high resource use, generally acceptable |
| `q4_k_m` | **Recommended.** Q6_K for half of attn.wv/feed_forward.w2, else Q4_K |
| `q5_k_m` | Q6_K for half of attn.wv/feed_forward.w2, else Q5_K |
| `f16` | No quantization, full float16 |

### Deployment recipes

**vLLM with LoRA adapter (recommended — no merge needed):**
```bash
vllm serve unsloth/Qwen3.5-4B \
  --enable-lora \
  --lora-modules jbujb=runs/qwen35_tool_sft/adapter
```

**vLLM with merged model:**
```bash
vllm serve runs/qwen35_tool_sft/merged
```

**SGLang:**
```bash
python -m sglang.launch_server \
  --model unsloth/Qwen3.5-4B \
  --lora-paths runs/qwen35_tool_sft/adapter
```

**Ollama (via GGUF):**
```bash
echo "FROM ./runs/qwen35_tool_sft/gguf/model-Q4_K_M.gguf" > Modelfile
ollama create jbujb-qwen-tool -f Modelfile
ollama run jbujb-qwen-tool
```

**No custom parser needed.** vLLM, SGLang, and Ollama handle Qwen's native tool-calling
format (`<tool_call>...</tool_call>`) automatically. The XML parser in `normalize_prediction()`
is only for offline evaluation.

---

## ClearML observability

ClearML is used to track experiments.

**Default training metrics** (via SFTConfig):

```text
train/loss
eval/loss
train/learning_rate
train/grad_norm
train/epoch
train/global_step
train/runtime
train/samples_per_second
train/steps_per_second
```

**Custom tool-calling metrics** (logged automatically when `--report-to clearml`):

```text
baseline_json_validity
baseline_tool_accuracy
baseline_exact_action_accuracy
baseline_hallucinated_id_rate

sft_json_validity
sft_tool_accuracy
sft_exact_action_accuracy
sft_hallucinated_id_rate

delta_sft_vs_baseline/
├── json_validity
├── tool_accuracy
├── exact_action_accuracy
└── hallucinated_id_rate

by_pattern/                          # Per workflow pattern
├── full_purchase/tool_accuracy
├── ordering_from_search/json_validity
├── dish_details/exact_action_accuracy
└── ...
```

**Console progress during evaluation:**

```
[baseline_validation_raw] 1/50 | json=100% tool=100% | 2.3s/step | ETA 113s
[baseline_validation_raw] 25/50 | json=72% tool=48% | 2.1s/step | ETA 52s
[baseline_validation_raw] 50/50 | json=68% tool=44% | 2.0s/step | ETA 0s
```

---

## Error analysis

Every evaluation run should produce an error analysis file.

Example:

```json
{
  "id": "case_001",
  "workflow_pattern": "ordering_from_named_restaurant",
  "user": "Commande-moi deux tacos poulet",
  "expected": {
    "type": "assistant_message",
    "policy": "ask_confirmation"
  },
  "prediction": {
    "type": "tool_call",
    "name": "create_order"
  },
  "error_type": "mutation_without_confirmation",
  "severity": "critical"
}
```

This file is used to improve the dataset.

The loop is:

```text
evaluate model
→ inspect errors
→ classify failures
→ generate targeted examples
→ retrain
→ compare
```

---

## Evaluation philosophy

The model is not evaluated like a normal chatbot.

It is evaluated like an agent.

A good model must:

```text
choose the right tool
produce valid JSON
provide required arguments
avoid hallucinated IDs
respect tool dependencies
ask clarification when information is missing
ask confirmation before mutating actions
recover from tool errors
refuse dangerous actions
```

A bad model may still sound fluent, but if it violates tool rules, it is not production-ready.

---

## Current training stack

The current stack is:

```text
Model: unsloth/Qwen3.5-4B
Training: Unsloth + TRL SFTTrainer
Adapter: LoRA / optional QLoRA
Monitoring: ClearML
Evaluation: custom EvalOps
Inference testing: Transformers / Unsloth
Future serving: vLLM or SGLang
```

Recommended future stack:

```text
Training: Unsloth
Experiment tracking: ClearML or MLflow
Inference: vLLM
LLM tracing: Langfuse
System observability: OpenTelemetry + OpenObserve
```

---

## Important implementation notes

### Qwen3.5 tokenizer / processor issue

For Qwen3.5, `FastModel.from_pretrained()` may return a processor instead of a plain tokenizer.

The script must extract the inner text tokenizer:

```python
model, processor_or_tokenizer = FastModel.from_pretrained(...)

tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
```

This avoids errors where the processor expects multimodal content blocks:

```json
[
  {
    "type": "text",
    "text": "..."
  }
]
```

while the dataset uses normal text messages:

```json
{
  "role": "user",
  "content": "Je veux commander deux tacos"
}
```

---

## Flash Attention 2 installation

Flash Attention 2 provides 1.5-2x training speedup with zero quality impact. It fuses
the attention computation into a single CUDA kernel, reducing GPU memory I/O.

### Prerequisites

- **CUDA >= 12.0** (check: `nvcc --version` or `nvidia-smi`)
- **PyTorch >= 2.2** (check: `python -c "import torch; print(torch.__version__)"`)
- **GPU**: Ampere, Ada, Blackwell, or Hopper (RTX 3090, RTX 4090, RTX PRO 6000, A100, A10G, L40S, H100, B200)
  - ❌ NOT supported: T4, RTX 2080 (Turing) — falls back to `sdpa` or `eager`
- **RAM**: >= 32 GB recommended for compilation (set `MAX_JOBS=4` if less)

### Quick install (pre-compiled wheel)

```bash
# For CUDA 12.x (Ampere/Ada/Hopper: A100, RTX 4090, H100, etc.)
pip install flash-attn --no-build-isolation

# For CUDA 13.x (Blackwell: RTX PRO 6000, B200, etc.)
pip install "flash-attn-4[cu13]"
```

### Verify installation

```bash
# Flash Attention 2 (CUDA 12.x)
python -c "from flash_attn import flash_attn_func; print('Flash Attention 2 OK')"

# Flash Attention 4 (CUDA 13.x / Blackwell)
python -c "from flash_attn.cute import flash_attn_func; print('Flash Attention 4 OK')"
```

### Common issues

| Symptom | Fix |
|---------|-----|
| `pip install` hangs or takes >30 min | Install `ninja`: `pip install ninja` |
| `out of memory` during compilation | `MAX_JOBS=2 pip install flash-attn --no-build-isolation` |
| `CUDA version mismatch` | Match PyTorch CUDA to system CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `GPU not supported` (T4, RTX 2080) | Use `--attn-implementation sdpa` instead |
| `undefined symbol` at import | Rebuild: `pip uninstall flash-attn -y && MAX_JOBS=4 pip install flash-attn --no-build-isolation` |
| `ninja --version` exits with error | `pip uninstall ninja -y && pip install ninja` |

### Fallback options

If Flash Attention 2 cannot be installed, the training script falls back gracefully:

```bash
# Option 1: PyTorch's built-in scaled dot product attention (good, ~20% slower)
--attn-implementation sdpa

# Option 2: Standard eager attention (slowest, most compatible)
--attn-implementation eager
```

The script auto-detects flash-attn availability at startup and logs a warning
if it's not found, continuing with the fallback implementation.

---

## Validation rules

Before training, the dataset should pass these rules:

```text
each dialogue has at least system → user → assistant
assistant tool calls are valid
tool names exist in the registry
tool arguments match schema
required arguments are present
no unknown argument is added
tool outputs are valid JSON
IDs are not empty
IDs are not invented
mutating tools require confirmation
dangerous tools are excluded from normal training
workflow order is respected
```

Examples with the following issues must be rejected or repaired:

```text
missing user message
empty product_id
empty merchant_id
empty draft_id
create_order without confirmation
get_restaurant_menu without merchant_id
tool call before required resolver
invalid JSON arguments
```

---

## Recommended development workflow

Use this workflow for each experiment:

```text
1. Generate dataset
2. Validate dataset
3. Split train / validation / test / test_hard
4. Run baseline evaluation on base model
5. Fine-tune model
6. Run post-SFT evaluation
7. Compare baseline vs fine-tuned model
8. Analyze errors
9. Generate targeted correction data
10. Retrain
```

Do not skip baseline evaluation.

Do not trust loss alone.

Do not train on invalid dialogues.

---

## Suggested repository structure

```text
.
├── data/
│   ├── tool_registry.json
│   ├── openrouter_toolweave_seed.jsonl
│   ├── tool_graph.yaml
│   ├── workflow_patterns.yaml
│   └── seed_entities.yaml
│
├── outputs/
│   ├── clean_dialogues.jsonl
│   ├── rejected_dialogues.jsonl
│   └── splits/
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       └── test_hard.jsonl
│
├── scripts/
│   ├── clean_dataset.py
│   ├── generate_openrouter_dataset.py
│   ├── validate_dataset.py
│   ├── split_dataset.py
│   └── check_split_leakage.py
│
├── evalops/
│   ├── runner.py
│   ├── parser.py
│   ├── scorers.py
│   └── report.py
│
├── training/
│   └── train_qwen35_tool_sft.py
│
├── runs/
│   └── qwen35_4b_tool_sft_v1/
│
├── requirements.txt
└── README.md
```

---

## Roadmap

### Phase 1 — Dataset foundation

```text
tool registry
tool graph
workflow patterns
goal generation
executable plans
dialogue generation
validation
```

### Phase 2 — Baseline and SFT

```text
baseline evaluation
Qwen3.5-4B LoRA fine-tuning
post-SFT evaluation
ClearML tracking
error analysis
```

### Phase 3 — Dataset improvement

```text
negative examples
failure recovery examples
hard test set
workflow balancing
tool confusion analysis
```

### Phase 4 — Reasoning model improvement

```text
plan + action training format
preference pairs
DPO / ORPO
GRPO / ToolRL environment
reward functions
```

### Phase 5 — Production readiness

```text
vLLM serving
tool-call guardrails
runtime validation
OpenTelemetry tracing
OpenObserve monitoring
Langfuse traces
model regression tests
```

---

## Long-term vision

The long-term goal is to build a small domain-specific reasoning model capable of reliable tool use.

This model should be able to:

```text
understand user intent
plan the next action
call tools safely
read tool outputs
continue multi-step workflows
avoid hallucinating operational IDs
respect business constraints
recover from errors
produce production-ready structured outputs
```

The broader vision is to turn this repo into a reusable framework:

> A tool-calling dataset and training framework that can take a semantic tool registry and produce high-quality multi-turn reasoning trajectories for fine-tuning small language models.

---

## Status

Current focus:

```text
Qwen3.5-4B SFT
Tool-calling dataset validation
Baseline vs fine-tuned evaluation
ClearML experiment tracking
JBUJB food-ordering tool workflows
OpenRouter + ToolWeave seed corpus
```

Next priorities:

```text
improve dataset validation
log custom metrics to ClearML
fix or ignore Flash Attention depending on speed needs
run first full baseline
run first LoRA fine-tuning
analyze post-training errors
generate targeted correction dataset
```

---

## Important warning

This project should not train on dirty data.

Before any training run, check:

```text
no missing user turns
no empty IDs
no invalid tool calls
no unsafe mutation without confirmation
no impossible tool chain
no fake product_id / merchant_id / draft_id
```

A small clean dataset is better than a large corrupted dataset.

The goal is not to generate more data.

The goal is to generate data that teaches the model correct agentic behavior.

```
```
