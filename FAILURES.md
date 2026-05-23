# Run Failures & Fixes Log

Document every failure encountered, root cause, and the patch applied.
One entry per problem. Goal: never hit the same issue twice.

---

## Run v1 — 2026-05-23

### F1: SFT scores identical to baseline (4% json, 0% tool)

**Symptom:** After fine-tuning on 200 examples, `json_validity` stayed at 4-5%,
`tool_accuracy` at 0%. Baseline and SFT indistinguishable.

**Root cause (3 simultaneous issues):**

1. `add_generation_prompt=False` in `format_row()` — the Qwen chat template
   wasn't adding the `<|im_start|>assistant` trigger token, so the model
   learned conversations without knowing where to start generating.

2. `batch_size=8 + grad_accum=8 = batch 64` on 200 examples → only **4 training
   steps**. LoRA adapters couldn't converge with 4 weight updates.

3. `max_seq_length=8912` caused tokenizer truncation on long dialogues,
   silently dropping tool call messages.

**Fix (commit `d3a899b`):**
```python
# format_row()
add_generation_prompt=True   # was False
# argparse defaults
max_seq_length=2048          # was 4096 → was run with 8912
# CLI params changed to:
--batch-size 2 --grad-accum 4 --epochs 3 --lr 5e-5
```

---

### F2: `get_cities` and `get_districts` in dialogues but not in `tool_registry.json`

**Symptom:** Dialogues contained 16 calls to `get_cities` and 9 calls to
`get_districts`, but `tool_registry.json` had only 23 tools (didn't include these).
`apply_chat_template(messages, tools=registry)` couldn't format these tools
properly, causing inconsistent training signal.

**Root cause:** The dataset generator includes proposed LOCATION tools, but the
registry exported manually only included live DISCOVERY/DETAILS/ORDERING tools.

**Fix (commit `88fc3ee`):** Added `get_cities` and `get_districts` to
`data/tool_registry.json` → 23 → 25 tools.

---

### F3: FastModel import commented out (docstring accident)

**Symptom:** `NameError: name 'FastModel' is not defined` at `FastModel.from_pretrained()`.

**Root cause:** An edit accidentally wrapped the `try/except` import block inside a
triple-quoted docstring (lines 28-34 were inside `"""..."""`).

**Fix (commit `93c2753`):** Removed the wrapping `"""` — import is now executable.

---

### F4: No ClearML custom metrics — only default SFTConfig scalars

**Symptom:** ClearML showed `train/loss`, `eval/loss`, `learning_rate` but
NO tool-calling quality metrics. Impossible to know if `json_validity` or
`hallucinated_id_rate` improved.

**Root cause:** `evaluate_model()` computed metrics but only wrote them to
JSON files. Never called `clearml.Logger.report_scalar()`.

**Fix (commit `93c2753`):** Added `log_to_clearml()` function that reports
all evaluation scalars to ClearML:
- `baseline_json_validity`, `baseline_tool_accuracy`, ...
- `sft_json_validity`, `sft_tool_accuracy`, ...
- `delta_sft_vs_baseline/tool_accuracy`, ...
- `by_pattern/full_purchase/tool_accuracy`, ...

---

## Run v2 — 2026-05-23

### F5: `SyntaxError: unterminated triple-quoted string literal`

**Symptom:** `File "train_qwen35_tool_sft.py", line 691` — script wouldn't start.

**Root cause:** The module-level docstring (`"""Qwen3.5-4B tool-calling SFT..."`)
was opened at line 3 but never closed. An edit to fix F3 removed the closing `"""`.

**Fix (commit `f4ea86e`):** Added closing `"""` after the docstring block + fixed
`\s` invalid escape in regex line 138.

---

### F6: SFT eval scores still 8% despite perfect training loss (0.0002)

**Symptom:** Training loss converged to 0.00024 — model clearly learned the
format. But `json_validity` stayed at 8% in post-training eval. User messages
were correct (no empty entities).

**Root cause:** The Qwen chat template formats tool calls as **XML**:
```xml
<tool_call><function=search_food><parameter=query>tacos</parameter></function></tool_call>
```
But the evaluation parser `normalize_prediction()` only handled **JSON**:
```json
{"name": "search_food", "arguments": {"query": "tacos"}}
```
`json.loads("<tool_call>...")` failed → `json_valid: false` for every
correct XML output.

This is fundamentally different from F1 (v1): in v1, the model genuinely
didn't learn. In v2, the model is correct but the parser is wrong.

**Fix (commit `77478ef`):** Added Qwen XML format parsing in `normalize_prediction()`:
```python
tc_match = re.search(r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>', raw, ...)
if tc_match:
    name = tc_match.group(1)
    args = {}
    for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', ...):
        args[pm.group(1)] = pm.group(2).strip()
    return {"type": "tool_call", "name": name, "arguments": args}
```
Also strips `⟨think⟩...⟨/think⟩` prefix before parsing.

---

### F7: Flash Attention 4 installation failed on CUDA 12.8

**Symptom:** `pip install "flash-attn-4[cu13]"` → `No matching distribution found`.
`pip install flash-attn-4` (without extra) → same error.

**Root cause:** `flash-attn-4` wheel is `py3-none-any` but `[cu13]` extra
requires exact CUDA 13.x. PyTorch was compiled for CUDA 12.8.

**Fix:**
```bash
# Fallback 1: Standard flash-attn for CUDA 12.x
pip install flash-attn --no-build-isolation

# Fallback 2: Use built-in SDPA (zero install, ~20% slower)
--attn-implementation sdpa
```

---

### F8: Sample packing skipped (vision-language model detected)

**Symptom:** `Unsloth: Sample packing skipped (vision-language model detected).`

**Root cause:** Qwen 3.5 is a multimodal model (text + vision). Unsloth detects
the vision processor and disables packing. Packing combines multiple short
examples into one sequence, but can't merge vision inputs.

**Impact:** Minor — 200 examples at 2048 tokens means very little padding waste
anyway. Not worth fixing for this dataset size.

---

## Checklist before next run

- [ ] `add_generation_prompt=True` in `format_row()`
- [ ] `max_seq_length <= 2048`
- [ ] `batch_size 2, grad_accum 4, epochs >= 3` (≥75 steps)
- [ ] All tools in dialogues are in `tool_registry.json`
- [ ] `normalize_prediction()` handles XML format (Qwen native)
- [ ] ClearML custom metrics are enabled (`--report-to clearml`)
- [ ] Dataset validated (0 missing user msgs, 0 empty IDs, 0 provenance errors)
- [ ] FastModel import is outside docstring
- [ ] Syntax check: `python -c "import ast; ast.parse(open('train_qwen35_tool_sft.py').read())"`
- [ ] Flash Attention installed OR fallback `--attn-implementation sdpa` specified
