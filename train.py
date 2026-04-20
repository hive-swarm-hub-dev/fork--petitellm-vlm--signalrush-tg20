"""
PetiteLLM: VLM — trains a projection (+ optional LoRA) bridging frozen SigLIP
to frozen Qwen2.5-0.5B for visual QA.

Baseline:
  - Frozen SigLIP features are pre-cached by prepare.sh to data/vision_cache/.
  - Single-layer linear projection: SigLIP_hidden (768) -> Qwen_hidden (896).
  - LoRA on q/k/v/o of the LLM (optional; toggle USE_LORA env var).
  - Supervised fine-tuning on ScienceQA train split: teacher-forced next-token
    loss over the answer span (masked prompt loss).
  - Save final_projection.ptz (zlib fp16) and final_lora.safetensors if used.

Exposes the eval contract:
    build_components() -> (projection, llm, tokenizer, siglip_hidden, llm_hidden, prompt_template)

Agents may replace this baseline but must preserve the eval contract.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- config -----------------------------


class HP:
    seed = int(os.environ.get("SEED", 1337))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    batch_size = int(os.environ.get("BATCH_SIZE", 8))
    lr_proj = float(os.environ.get("LR_PROJ", 5e-4))
    lr_lora = float(os.environ.get("LR_LORA", 2e-4))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))
    max_text_len = int(os.environ.get("MAX_TEXT_LEN", 384))
    max_answer_len = int(os.environ.get("MAX_ANSWER_LEN", 16))
    use_lora = os.environ.get("USE_LORA", "1") not in ("0", "false", "False")
    lora_rank = int(os.environ.get("LORA_RANK", 32))
    lora_alpha = int(os.environ.get("LORA_ALPHA", 64))
    lora_target = os.environ.get("LORA_TARGET", "qkvo")  # qkvo|all|qkvo_mlp4
    lora_mlp_rank = int(os.environ.get("LORA_MLP_RANK", 4))
    projection_type = os.environ.get("PROJECTION_TYPE", "mlp")  # linear|mlp
    projection_hidden = int(os.environ.get("PROJECTION_HIDDEN", 1024))
    cosine_decay = os.environ.get("COSINE_DECAY", "1") not in ("0", "false", "False")


# ----------------------------- prompt -----------------------------


PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "qa")  # chat|qa
SYSTEM_PROMPT = "You are a helpful visual assistant. Answer concisely with the correct option letter only."

def prompt_template(question: str) -> str:
    if PROMPT_STYLE == "chat":
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    # Minimal QA format: base Qwen2.5-0.5B saw plenty of "Q: ... A: " style in
    # pretraining but has no instruction-tuning on the chat template.
    return f"Question: {question}\nAnswer:"


# ----------------------------- projection -----------------------------


def _init_proj_linear(m: nn.Linear):
    nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        _init_proj_linear(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        _init_proj_linear(self.fc1)
        _init_proj_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def make_projection(kind: str, in_dim: int, out_dim: int, hidden: int) -> nn.Module:
    if kind == "mlp":
        return MLPProjection(in_dim, hidden, out_dim)
    return LinearProjection(in_dim, out_dim)


# ----------------------------- backbone loading -----------------------------


def load_backbones(device: torch.device):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel

    siglip_cfg = AutoConfig.from_pretrained("models/siglip")
    # SigLIP top-level config has .vision_config.hidden_size.
    siglip_hidden = getattr(getattr(siglip_cfg, "vision_config", None), "hidden_size", None) \
        or getattr(siglip_cfg, "hidden_size", 768)

    tokenizer = AutoTokenizer.from_pretrained("models/qwen", trust_remote_code=True)
    dtype_name = os.environ.get("LLM_DTYPE", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    attn_impl = os.environ.get("ATTN_IMPL", "eager")
    llm = AutoModelForCausalLM.from_pretrained(
        "models/qwen", torch_dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    llm.config.use_cache = True
    llm_hidden = llm.config.hidden_size

    for p in llm.parameters():
        p.requires_grad = False
    llm.eval()
    llm.to(device)
    return tokenizer, llm, siglip_hidden, llm_hidden


def maybe_apply_lora(llm):
    if not HP.use_lora:
        return llm
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[train] peft not installed; skipping LoRA.", flush=True)
        return llm
    rank_pattern = {}
    alpha_pattern = {}
    if HP.lora_target == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    elif HP.lora_target == "qkvo_mlp4":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        mlp_alpha = max(1, HP.lora_alpha * HP.lora_mlp_rank // max(1, HP.lora_rank))
        for m in ("gate_proj", "up_proj", "down_proj"):
            rank_pattern[m] = HP.lora_mlp_rank
            alpha_pattern[m] = mlp_alpha
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=HP.lora_rank,
        lora_alpha=HP.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    llm = get_peft_model(llm, cfg)
    llm.print_trainable_parameters()
    return llm


# ----------------------------- data -----------------------------


class SqaDataset:
    def __init__(self, path: str):
        self.rows = []
        if not Path(path).exists():
            return
        with open(path) as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self): return len(self.rows)

    def sample(self, rng) -> dict:
        return self.rows[rng.integers(0, len(self.rows))]


# ----------------------------- save/load -----------------------------


def save_projection_compressed(proj: nn.Module, path: str) -> int:
    sd = {k: v.detach().to(torch.float16).cpu() for k, v in proj.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    with open(path, "wb") as f:
        f.write(blob)
    return os.path.getsize(path)


def load_projection_compressed(path: str, proj: nn.Module):
    with open(path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(zlib.decompress(blob))
    sd = torch.load(buf, map_location="cpu")
    out = {}
    for k, v in proj.state_dict().items():
        out[k] = sd[k].to(dtype=v.dtype) if k in sd else v
    proj.load_state_dict(out, strict=True)


def save_lora(llm, path: str):
    try:
        from peft import PeftModel
        if not isinstance(llm, PeftModel):
            return 0
        # Cast LoRA tensors to bf16 on save: peft stores them in fp32 by default,
        # which doubles the artifact. Loss in inference accuracy is negligible
        # and load_lora casts back to the target module dtype.
        sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in llm.state_dict().items()
              if "lora_" in k}
        from safetensors.torch import save_file
        save_file(sd, path)
        return os.path.getsize(path)
    except Exception as e:
        print(f"[train] save_lora skipped: {e}", flush=True)
        return 0


def load_lora(llm, path: str):
    from peft import PeftModel
    if not isinstance(llm, PeftModel):
        return llm
    from safetensors.torch import load_file
    sd = load_file(path)
    # Load into the peft model state dict, matching names.
    model_sd = llm.state_dict()
    filled = 0
    for k, v in sd.items():
        if k in model_sd:
            model_sd[k].copy_(v.to(model_sd[k].dtype))
            filled += 1
    print(f"[eval] loaded {filled} LoRA tensors from {path}", flush=True)
    return llm


# ----------------------------- eval contract -----------------------------


def build_components():
    """Entry point called by eval/evaluate.py."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, llm, siglip_hidden, llm_hidden = load_backbones(device)
    # Batched generation requires left-padding so short prompts don't get EOS
    # pad tokens appended before the assistant turn starts.
    tokenizer.padding_side = "left"
    if HP.use_lora and Path("final_lora.safetensors").exists():
        llm = maybe_apply_lora(llm)
        llm = load_lora(llm, "final_lora.safetensors")
    proj = make_projection(HP.projection_type, siglip_hidden, llm_hidden, HP.projection_hidden)
    if Path("final_projection.ptz").exists():
        load_projection_compressed("final_projection.ptz", proj)
    proj = proj.to(device)
    return proj, llm, tokenizer, siglip_hidden, llm_hidden, prompt_template


# ----------------------------- training -----------------------------


def build_training_batch(tokenizer, ds: SqaDataset, rng, device, llm, proj, llm_dtype):
    """Returns (inputs_embeds, attention_mask, labels) with labels = -100 on prompt+image tokens."""
    examples = [ds.sample(rng) for _ in range(HP.batch_size)]
    feats = []
    prompts = []
    full_texts = []
    for ex in examples:
        feat_path = Path(f"data/vision_cache/{ex['image_id']}.pt")
        if not feat_path.exists():
            # fall back: skip — use an alternate example.
            continue
        f16 = torch.load(feat_path, map_location="cpu")
        feats.append(f16)
        pt = prompt_template(ex["prompt"])
        prompts.append(pt)
        # Chat-template prompts end with '\n' so "A" tokenizes cleanly as its own
        # token. The simpler "Answer:" template ends at ':', where "A" would
        # fuse into the ':A' subword and drift from the eval-time tokenization.
        # Insert a space so both styles produce " A" (token 362) as the answer.
        sep = "" if pt.endswith("\n") else " "
        full_texts.append(pt + sep + ex["answer"] + tokenizer.eos_token)
    if not feats:
        return None
    feats_t = torch.stack(feats).to(device).float()
    projected = proj(feats_t).to(llm_dtype)  # (B, N, D)
    B, Np, _ = projected.shape

    # Tokenize prompt+answer pairs.
    tok_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=HP.max_text_len)
    tok_prompt = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=HP.max_text_len)
    full_ids = tok_full["input_ids"].to(device)
    full_mask = tok_full["attention_mask"].to(device)
    prompt_ids = tok_prompt["input_ids"].to(device)
    prompt_mask = tok_prompt["attention_mask"].to(device)
    prompt_lens = prompt_mask.sum(dim=1)

    text_embeds = llm.get_input_embeddings()(full_ids).to(llm_dtype)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    img_mask = torch.ones(B, Np, dtype=full_mask.dtype, device=device)
    attn_mask = torch.cat([img_mask, full_mask], dim=1)

    # Labels: for the text portion, mask out everything up to prompt_len; image
    # positions are also masked. Also mask pad positions.
    label_text = full_ids.clone()
    for i in range(B):
        pl = int(prompt_lens[i].item())
        label_text[i, :pl] = -100
    label_text = torch.where(full_mask.bool(), label_text, torch.full_like(label_text, -100))
    # Prepend -100 for image positions.
    img_labels = torch.full((B, Np), -100, dtype=label_text.dtype, device=device)
    labels = torch.cat([img_labels, label_text], dim=1)

    return inputs_embeds, attn_mask, labels


def lr_at(step: int, peak: float, warmup: int, total: int = 0, cosine: bool = False) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if cosine and total > warmup:
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return peak


def main():
    random.seed(HP.seed); np.random.seed(HP.seed); torch.manual_seed(HP.seed)
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr); sys.exit(1)
    device = torch.device("cuda")

    tokenizer, llm, siglip_hidden, llm_hidden = load_backbones(device)
    llm_dtype = next(p.dtype for p in llm.parameters() if p.dtype in (torch.bfloat16, torch.float16, torch.float32))
    print(f"siglip_hidden={siglip_hidden} llm_hidden={llm_hidden} llm_dtype={llm_dtype}", flush=True)

    if HP.use_lora:
        llm = maybe_apply_lora(llm)

    proj = make_projection(HP.projection_type, siglip_hidden, llm_hidden, HP.projection_hidden).to(device)
    for p in proj.parameters(): p.requires_grad = True
    print(f"projection params: {sum(p.numel() for p in proj.parameters())/1e6:.3f}M", flush=True)

    train_ds = SqaDataset("data/sqa_train.jsonl")
    if len(train_ds) == 0:
        print("ERROR: no training data at data/sqa_train.jsonl", file=sys.stderr); sys.exit(2)
    print(f"train n={len(train_ds)}", flush=True)

    # Optimizer groups.
    proj_params = list(proj.parameters())
    lora_params = [p for p in llm.parameters() if p.requires_grad]
    param_groups = [{"params": proj_params, "lr": HP.lr_proj}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": HP.lr_lora})
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)

    start = time.time()
    step = 0
    rng = np.random.default_rng(HP.seed)
    # Estimate total steps using a calibration window of the first few iters.
    est_total_steps = 0
    calib_done = False
    calib_t0 = None
    while True:
        elapsed = time.time() - start
        if elapsed >= HP.max_wallclock_seconds:
            break
        opt.param_groups[0]["lr"] = lr_at(step, HP.lr_proj, HP.warmup_steps, est_total_steps, HP.cosine_decay)
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr_at(step, HP.lr_lora, HP.warmup_steps, est_total_steps, HP.cosine_decay)

        batch = build_training_batch(tokenizer, train_ds, rng, device, llm, proj, llm_dtype)
        if batch is None:
            step += 1; continue
        inputs_embeds, attn_mask, labels = batch
        out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"[skip] step={step} non-finite loss; dropping batch", flush=True)
            step += 1; continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Detect NaN/inf gradients before they corrupt optimizer state.
        bad_grad = False
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True; break
            if bad_grad: break
        if bad_grad:
            print(f"[skip] step={step} non-finite grad; dropping batch", flush=True)
            opt.zero_grad(set_to_none=True)
            step += 1; continue
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], HP.grad_clip)
        opt.step()
        step += 1
        if step == 50 and not calib_done:
            calib_dt = time.time() - start
            steps_per_sec = 50.0 / max(1e-6, calib_dt)
            est_total_steps = max(HP.warmup_steps + 1,
                                   int(steps_per_sec * (HP.max_wallclock_seconds - 30)))
            calib_done = True
            print(f"[calib] {steps_per_sec:.2f} steps/s -> est_total_steps={est_total_steps}", flush=True)
        if step % 50 == 0:
            print(f"step={step} t={elapsed:.0f}s loss={loss.item():.4f} "
                  f"lr_proj={opt.param_groups[0]['lr']:.2e}", flush=True)

    # Save artifacts.
    proj_bytes = save_projection_compressed(proj, "final_projection.ptz")
    lora_bytes = 0
    if HP.use_lora:
        lora_bytes = save_lora(llm, "final_lora.safetensors")
    code_bytes = os.path.getsize(__file__)
    total = proj_bytes + lora_bytes + code_bytes
    print(f"final_projection.ptz: {proj_bytes}  final_lora.safetensors: {lora_bytes}  train.py: {code_bytes}  total: {total}", flush=True)


if __name__ == "__main__":
    main()
