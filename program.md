# PetiteLLM: VLM

Bridge a frozen vision encoder to a frozen LLM and teach it visual QA. You train only (1) a projection module mapping SigLIP image embeddings into the LLM's token embedding space and (2) optional LoRA adapters on the LLM. Both backbones stay frozen; only your trainable artifact is budget-constrained.

- **Vision backbone (frozen)**: SigLIP-base (`google/siglip-base-patch16-224`).
- **Language backbone (frozen)**: Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`).
- **Metric**: `vqa_acc` = exact-match accuracy on a 500-sample held-out ScienceQA subset (multiple-choice and short-answer). Higher is better.

## Setup

1. **Read the in-scope files**:
   - `train.py` — projection (+ LoRA) training loop. You modify this.
   - `eval/eval.sh` — runs training + evaluation. Do not modify.
   - `eval/evaluate.py` — scoring (with `llm.generate(inputs_embeds=...)` probe and a KV-cache autoregressive fallback). Do not modify.
   - `prepare.sh` — downloads backbones, LLaVA-Pretrain (20k image-caption pairs), ScienceQA (5k instruct + 500 val + 500 test), and pre-caches SigLIP vision features. Do not modify.
   - `models/siglip`, `models/qwen` — frozen backbones. Do not modify.
   - `data/vision_cache/*.pt` — pre-extracted SigLIP features (shape `[num_patches, hidden]` per image, fp16). Do not modify.
2. **Run prepare**: `bash prepare.sh`. Takes ~5-10 minutes on first run (downloads + feature caching). Sets up `data/{pretrain,sqa_train,sqa_val,sqa_test}.jsonl` with fields `{image_id, prompt, answer}` and `data/vision_cache/<image_id>.pt`.
3. **Initialize results.tsv**: `echo -e "commit\tvqa_acc\tartifact_bytes\tstatus\tdescription" > results.tsv`.
4. **Run baseline**: `bash eval/eval.sh > run.log 2>&1`.

## The benchmark

- **Training budget**: 10 minutes wallclock on 1×A100 (`MAX_WALLCLOCK_SECONDS=600`).
- **Artifact cap**: `train.py` + **all `*.pt`, `*.ptz`, `*.safetensors` in the task root** ≤ 16,000,000 bytes. Backbones under `models/` are excluded.
- **Trainable pieces**:
  - **Projection** (required): a module mapping `[num_patches, siglip_hidden]` → `[num_patches, llm_hidden]`. Baseline is a single `nn.Linear`.
  - **LoRA on LLM** (optional): if enabled, `peft.LoraConfig` adapters on attention projections only (q, k, v, o). Rank and alpha are yours to tune.
- **Input format** (baseline): `<projected_image_tokens> <text_tokens>` prepended to a short system + instruction string; target is the answer string teacher-forced to the LLM.

## Experimentation

**What you CAN modify:**

- `train.py`: projection architecture (linear, MLP, cross-attn pooler), LoRA on/off and hyperparameters, prompt template, loss (next-token CE over the answer span, with/without masked-context), optimizer, learning rate schedule, data mixing (LLaVA-Pretrain + ScienceQA proportions), feature pooling (all patches vs pooled vs QFormer-like queries), quantization of saved artifact.

**What you CANNOT modify:**

- `eval/`, `prepare.sh`, `data/`, `models/`, the backbones or their weights.
- The eval prompt template is whatever eval/evaluate.py uses (see below); your `train.py` must be consistent with that template because eval reconstructs the same packing.

## Contract with the evaluator

`train.py` must at the end produce:

- `final_projection.ptz` — zlib-compressed projection state dict (fp16 preferred).
- `final_lora.safetensors` — peft-compatible LoRA weights (ONLY if you used LoRA). If you didn't, simply omit; the evaluator will skip loading it.
- `config.json` (optional) — key/value hyperparameters needed by the evaluator (see below).

Plus, as module-level Python entry points inside `train.py`:

```
def build_components() -> tuple[projection, llm_with_or_without_lora, tokenizer, siglip_hidden, llm_hidden, prompt_template]:
    ...
def score_batch(projected_features, text_input_ids, text_attention_mask, llm, tokenizer, max_new_tokens):
    # returns generated strings (batch of len B).
```

The evaluator probes `llm.generate(inputs_embeds=..., attention_mask=...)`. If it errors, the evaluator falls back to a KV-cache-aware manual autoregressive loop that concatenates projected image tokens with text embeddings itself. Your `build_components` must return a valid prompt template and a `score_batch` or rely on the default `score_batch` provided in the baseline `train.py`.

## Output format

```
---
vqa_acc:           0.3420
artifact_bytes:    12345678
line_count:        406
valid:             true
```

- `artifact_bytes` counts `train.py` + every `*.pt`/`*.ptz`/`*.safetensors` in the task root directory (NOT `models/`, NOT `data/`).
- `valid`: `true` iff `artifact_bytes ≤ 16_000_000`, eval produced a float accuracy, and the frozen backbone weights under `models/` were not modified (eval checks sha256 of key files).

Hive submit: `--score vqa_acc` (higher is better).

## Logging results

```
commit  vqa_acc  artifact_bytes  status  description
a1b2c3d 0.3000   7234567         keep    baseline linear proj, no LoRA
b2c3d4e 0.3520   11200000        keep    MLP proj dim=1024 + LoRA rank=8 on q,v
```

## Caveats

- **Vision features are pre-cached.** `prepare.sh` runs SigLIP on every train image once and stores fp16 patch features. Do not re-run SigLIP per step; just read `data/vision_cache/<image_id>.pt`.
- **Dtype discipline.** Qwen2.5-0.5B runs in bf16. Before concatenating projected vision embeddings with `llm.embed_tokens(text_ids)`, cast them with `.to(llm.dtype)`. Mismatched dtypes in `inputs_embeds` is a common silent failure.
- **`generate(inputs_embeds=...)`.** Some HF versions don't support this cleanly. The evaluator probes at runtime and falls back to a KV-cache manual loop on failure. Stick to the default prompt template unless you know what you're doing.
- **Answer matching.** Eval normalizes predictions: lowercases, strips punctuation, extracts the first option letter if present (e.g. `(A) bar` → `A`), then compares to the normalized gold. Avoid extra chatter in model outputs — the baseline uses short system prompt + `Answer:` suffix to encourage terse outputs.
- **ScienceQA image scarcity.** Some ScienceQA items have no image; `prepare.sh` keeps only items with images (≥5k train, 500 val, 500 test).
- **Honor system on frozen models**. Do not modify `models/`. Eval hashes a key file and refuses mismatched runs.

## The experiment loop

LOOP FOREVER:

1. THINK — read `results.tsv` and `train.py`.
2. Modify `train.py`.
3. `git commit -am "..."`.
4. `bash eval/eval.sh > run.log 2>&1`.
5. `grep "^vqa_acc:\|^valid:" run.log`.
6. If `valid=false`, debug with `tail -n 100 run.log`.
7. Record in `results.tsv`.
8. Keep if `vqa_acc` improved AND `valid=true`, else `git reset --hard HEAD~1`.

**Timeout**: 15 minutes per run.

**NEVER STOP**: fully autonomous loop once started.
