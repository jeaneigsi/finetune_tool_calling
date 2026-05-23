# Failures & Fixes

---

## F1 — SFT scores identical to baseline

**What:** `json_validity` stayed at 5%, `tool_accuracy` at 0%. Training appeared to do nothing.

**Why:** Three simultaneous issues:
- `add_generation_prompt=False` → model never learned where to start generating
- `batch_size=8 × grad_accum=8` on 200 examples → only 4 training steps
- `max_seq_length=8912` → tokenizer silently truncated long dialogues, dropping tool calls

**Fix:** `add_generation_prompt=True`, `--batch-size 2 --grad-accum 4 --epochs 3`, `max_seq_length=2048`

---

## F2 — `get_cities` / `get_districts` in dialogues but not in tool registry

**What:** 16 dialogues called tools that didn't exist in `tool_registry.json`. The chat template couldn't format them.

**Why:** Dataset generator included proposed LOCATION tools. The registry was exported with only live tools.

**Fix:** Added `get_cities` and `get_districts` to `data/tool_registry.json` (23 → 25 tools)

---

## F3 — FastModel import was inside a docstring

**What:** `NameError: FastModel not defined` at startup.

**Why:** The `try/except` import block got wrapped in `"""..."""` during an edit.

**Fix:** Removed the wrapping triple quotes — import is now executable code.

---

## F4 — No tool-calling metrics in ClearML

**What:** ClearML only showed `train/loss`. No `json_validity`, `tool_accuracy`, etc.

**Why:** `evaluate_model()` computed metrics but only wrote JSON files. Never called `clearml.Logger.report_scalar()`.

**Fix:** Added `log_to_clearml()` — reports baseline, SFT, delta, and per-pattern metrics as ClearML scalars.

---

## F5 — SyntaxError at line 691

**What:** Script refused to start. `unterminated triple-quoted string literal`.

**Why:** Module docstring opened at line 3 was never closed after fixing F3. Also `\s` in regex needed escaping to `\\s`.

**Fix:** Closed the docstring. Escaped `\s` → `\\s`.

---

## F6 — SFT eval scores 8% despite perfect training (loss 0.0002)

**What:** Model clearly learned (loss converged to 0.0002) but `json_validity` stayed at 8%.

**Why:** Qwen produces tool calls in XML format:
```xml
<tool_call><function=search_food><parameter=query>tacos</parameter></function></tool_call>
```
The parser `normalize_prediction()` only understood JSON. `json.loads("<tool_call>...")` failed for every correct output.

**Fix:** Added XML parsing to `normalize_prediction()` — extracts tool name and parameters from `<function=X><parameter=Y>V</parameter>` tags. Also strips `⟨think⟩...⟨/think⟩` prefix.

---

## F7 — Flash Attention 4 install failed

**What:** `pip install "flash-attn-4[cu13]"` → `No matching distribution found`.

**Why:** `[cu13]` extra requires CUDA 13.x. PyTorch was compiled for CUDA 12.8.

**Fix:** `--attn-implementation sdpa` (built-in, zero install, ~20% slower). Or `pip install flash-attn --no-build-isolation` for CUDA 12.x.

---

## F8 — Sample packing skipped

**What:** `Unsloth: Sample packing skipped (vision-language model detected)`.

**Why:** Qwen 3.5 is multimodal. Unsloth disables packing when it detects a vision processor.

**Impact:** None. 200 examples at 2048 tokens = negligible padding waste.

---

## Pre-flight checklist

```
[ ] add_generation_prompt=True
[ ] max_seq_length <= 2048
[ ] batch 2, grad_accum 4, epochs >= 3
[ ] All dialog tools in tool_registry.json
[ ] normalize_prediction handles XML
[ ] ClearML --report-to clearml
[ ] Dataset validated (0 missing user, 0 empty IDs, 0 provenance errors)
[ ] FastModel import is outside docstring
[ ] python -c "import ast; ast.parse(open('train_qwen35_tool_sft.py').read())"
[ ] Flash Attn installed OR --attn-implementation sdpa
```
