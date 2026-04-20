"""Evaluate train.py's VLM on the held-out ScienceQA test split.

Strategy:
  1. Import train.py and call `build_components()` which returns:
         projection_module, llm_module, tokenizer, siglip_hidden, llm_hidden, prompt_template
     (prompt_template is a callable prompt_template(question: str) -> str.)
  2. For each test example:
         features = torch.load(data/vision_cache/<image_id>.pt)  # (N_patches, siglip_hidden) fp16
         projected = projection(features.to(device, dtype).float()).to(llm.dtype)  # (N_patches, llm_hidden)
         text = prompt_template(example["prompt"])
         Build inputs_embeds = [projected_tokens, llm.embed(text_tokens)]
         Try llm.generate(inputs_embeds=..., attention_mask=...); if unsupported,
         run KV-cache-aware manual decode.
  3. Normalize and compare to gold. Print `vqa_acc=<float>`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import string
import sys
from pathlib import Path

import torch


def load_train_module():
    spec = importlib.util.spec_from_file_location("task_train", "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # Extract option letter: look for (a), (b) or bare "a)" or "a:" or leading single letter.
    m = re.search(r"\(([a-z])\)|^\s*([a-z])[\).:\-\s]", s)
    if m:
        letter = m.group(1) or m.group(2)
        return letter
    # Fallback: strip punctuation, keep first token.
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = s.strip()
    return s.split()[0] if s else ""


def supports_generate_inputs_embeds(llm, tokenizer, device, dtype) -> bool:
    try:
        bos = tokenizer.eos_token_id or 0
        dummy_ids = torch.tensor([[bos]], device=device, dtype=torch.long)
        with torch.inference_mode():
            embeds = llm.get_input_embeddings()(dummy_ids).to(dtype)
            _ = llm.generate(inputs_embeds=embeds, max_new_tokens=1, do_sample=False)
        return True
    except Exception as e:
        print(f"[eval] generate(inputs_embeds=...) unsupported: {type(e).__name__}: {e}", file=sys.stderr)
        return False


@torch.inference_mode()
def manual_decode(llm, inputs_embeds, attention_mask, max_new_tokens: int, eos_token_id: int):
    """KV-cache-aware manual greedy decode from inputs_embeds.

    inputs_embeds: (B, L, D) in llm.dtype
    attention_mask: (B, L) int64
    Returns (B, max_new_tokens) token ids tensor (right-padded with eos if stopped).
    """
    B = inputs_embeds.size(0)
    device = inputs_embeds.device
    out = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    gen = torch.full((B, max_new_tokens), eos_token_id, device=device, dtype=torch.long)
    cur = next_token_logits.argmax(dim=-1)
    gen[:, 0] = cur
    done = torch.zeros(B, dtype=torch.bool, device=device)
    done = done | cur.eq(eos_token_id)
    attn = torch.cat([attention_mask, torch.ones(B, 1, device=device, dtype=attention_mask.dtype)], dim=1)
    for step in range(1, max_new_tokens):
        if done.all(): break
        emb = llm.get_input_embeddings()(cur.unsqueeze(1))
        out = llm(
            inputs_embeds=emb,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1)
        cur = torch.where(done, torch.full_like(cur, eos_token_id), cur)
        gen[:, step] = cur
        done = done | cur.eq(eos_token_id)
        attn = torch.cat([attn, torch.ones(B, 1, device=device, dtype=attn.dtype)], dim=1)
    return gen


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    task = load_train_module()
    for attr in ("build_components",):
        if not hasattr(task, attr):
            print(f"ERROR: train.py missing `{attr}`", file=sys.stderr); sys.exit(2)

    comps = task.build_components()
    projection, llm, tokenizer, siglip_hidden, llm_hidden, prompt_template = comps
    projection = projection.to(device).eval()
    llm = llm.to(device).eval()
    dtype = next(p.dtype for p in llm.parameters() if p.dtype in (torch.bfloat16, torch.float16, torch.float32))

    # Batched generate(inputs_embeds=...) requires left-padding: with
    # right-padding the shorter prompts in a batch get EOS pad tokens appended
    # *after* the assistant cue, so the model samples EOS on the very first new
    # token and emits nothing. Set this unconditionally so every agent's
    # build_components() is robust by default.
    tokenizer.padding_side = "left"

    use_native = supports_generate_inputs_embeds(llm, tokenizer, device, dtype)
    print(f"[eval] native generate(inputs_embeds=...): {use_native}", file=sys.stderr)

    # Load test set.
    test_path = Path("data/sqa_test.jsonl")
    examples = []
    with open(test_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"[eval] test examples: {len(examples)}", file=sys.stderr)

    eos = tokenizer.eos_token_id or 0
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    correct = 0
    total = 0
    B = 8
    max_new = int(os.environ.get("EVAL_MAX_NEW_TOKENS", 16))

    for i in range(0, len(examples), B):
        batch = examples[i:i + B]
        # Build projected image tokens for each example.
        img_feats = []
        texts = []
        for ex in batch:
            feat_path = Path(f"data/vision_cache/{ex['image_id']}.pt")
            if not feat_path.exists():
                print(f"[eval] missing features for {ex['image_id']}", file=sys.stderr); sys.exit(3)
            f16 = torch.load(feat_path, map_location="cpu")
            img_feats.append(f16)
            texts.append(prompt_template(ex["prompt"]))
        # Stack (requires all have same num_patches; SigLIP does).
        img_t = torch.stack(img_feats).to(device).float()
        proj = projection(img_t).to(dtype)  # (B, N_patches, llm_hidden)

        # Tokenize texts.
        tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        text_ids = tok["input_ids"].to(device)
        text_mask = tok["attention_mask"].to(device)
        text_emb = llm.get_input_embeddings()(text_ids).to(dtype)

        # Concat [proj | text].
        Bsz, Np, _ = proj.shape
        img_mask = torch.ones(Bsz, Np, dtype=text_mask.dtype, device=device)
        inputs_embeds = torch.cat([proj, text_emb], dim=1)
        attention_mask = torch.cat([img_mask, text_mask], dim=1)

        if use_native:
            out = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=pad,
                eos_token_id=eos,
            )
            # Some HF versions return only new tokens when given inputs_embeds;
            # others return the full prefix. Heuristic: if output length > max_new,
            # take the last max_new tokens.
            if out.size(1) > max_new:
                out = out[:, -max_new:]
            gen_ids = out
        else:
            gen_ids = manual_decode(llm, inputs_embeds, attention_mask, max_new, eos)

        for j, ex in enumerate(batch):
            text = tokenizer.decode(gen_ids[j], skip_special_tokens=True)
            pred = normalize_answer(text)
            gold = normalize_answer(ex["answer"])
            if pred == gold:
                correct += 1
            total += 1

    acc = correct / max(total, 1)
    print(f"vqa_acc={acc:.4f}")


if __name__ == "__main__":
    main()
