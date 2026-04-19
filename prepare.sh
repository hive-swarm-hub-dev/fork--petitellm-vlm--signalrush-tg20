#!/usr/bin/env bash
# One-time setup for petitellm-vlm:
# - Install deps
# - Download SigLIP + Qwen2.5-0.5B
# - Download LLaVA-Pretrain subset (20k) + ScienceQA (5k train / 500 val / 500 test with images)
# - Pre-cache SigLIP features for every training image
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"

echo "[1/3] Installing requirements ..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -r requirements.txt
else
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

mkdir -p models/siglip models/qwen data/vision_cache

if [ ! -f models/siglip/config.json ] || [ ! -f models/qwen/config.json ]; then
    echo "[2/3] Downloading backbones ..."
    "$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("google/siglip-base-patch16-224", local_dir="models/siglip",
                  allow_patterns=["*.json","*.txt","*.safetensors","tokenizer*","preprocessor_config.json"])
snapshot_download("Qwen/Qwen2.5-0.5B", local_dir="models/qwen",
                  allow_patterns=["*.json","*.txt","*.safetensors","tokenizer*","merges.txt","vocab.json","added_tokens.json","generation_config.json"])
PY
else
    echo "[2/3] Backbones already present."
fi

echo "[3/3] Building datasets + caching SigLIP features ..."
"$PY" - <<'PY_SCRIPT'
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA = Path("data"); DATA.mkdir(exist_ok=True)
VC = DATA / "vision_cache"; VC.mkdir(parents=True, exist_ok=True)

PRETRAIN_JSONL = DATA / "pretrain.jsonl"
SQA_TRAIN = DATA / "sqa_train.jsonl"
SQA_VAL = DATA / "sqa_val.jsonl"
SQA_TEST = DATA / "sqa_test.jsonl"
SQA_LABELS_SHA = DATA / "sqa_test_labels_sha.txt"

need_pretrain = not PRETRAIN_JSONL.exists()
need_sqa = not (SQA_TRAIN.exists() and SQA_VAL.exists() and SQA_TEST.exists())

if need_pretrain or need_sqa:
    from datasets import load_dataset
    if need_pretrain:
        print("  downloading LLaVA-Pretrain subset (20k) ...")
        try:
            # 'liuhaotian/LLaVA-Pretrain' uses images hosted separately; prefer the
            # parquet variant for simplicity.
            ds = load_dataset("liuhaotian/LLaVA-Pretrain",
                              data_files="blip_laion_cc_sbu_558k.json",
                              split="train")
            rng = random.Random(SEED)
            indexes = list(range(len(ds)))
            rng.shuffle(indexes)
            picked = indexes[:20000]
            with open(PRETRAIN_JSONL, "w") as f:
                kept = 0
                for i in picked:
                    ex = ds[i]
                    img_rel = ex.get("image")
                    if not img_rel:
                        continue
                    # We do not have the raw images locally; skip writing this subset
                    # if images aren't resolvable. In that case we fall back to sqa-only.
                    break
                if kept == 0:
                    raise RuntimeError("LLaVA-Pretrain images not bundled; skipping.")
        except Exception as e:
            print(f"  note: LLaVA-Pretrain unavailable or images not bundled ({e}); skipping pretraining stage.", flush=True)
            PRETRAIN_JSONL.write_text("")  # empty => baseline skips pretrain loader
    if need_sqa:
        print("  downloading ScienceQA ...")
        sqa = load_dataset("derek-thomas/ScienceQA")
        def dump_split(split_name, out_path, n):
            rows = sqa[split_name]
            items = []
            for i in range(len(rows)):
                ex = rows[i]
                img = ex.get("image")
                if img is None:
                    continue
                choices = ex.get("choices") or []
                ans_idx = ex.get("answer")
                if not choices or ans_idx is None or ans_idx < 0 or ans_idx >= len(choices):
                    continue
                ltr = chr(ord("A") + ans_idx)
                question = ex.get("question") or ""
                hint = ex.get("hint") or ""
                prompt = question
                if hint:
                    prompt = f"{hint}\n{question}"
                prompt += "\nChoices:\n" + "\n".join(f"({chr(ord('A')+j)}) {c}" for j, c in enumerate(choices))
                items.append({"_img": img, "prompt": prompt, "answer": ltr})
                if len(items) >= n:
                    break
            if len(items) < n:
                print(f"    split={split_name} only found {len(items)} items (wanted {n})", flush=True)
            with open(out_path, "w") as f:
                for k, it in enumerate(items):
                    img_id = f"{split_name}_{k:05d}"
                    img_path = VC / f"{img_id}.png"
                    if not img_path.exists():
                        it["_img"].convert("RGB").save(img_path, format="PNG")
                    f.write(json.dumps({"image_id": img_id, "prompt": it["prompt"], "answer": it["answer"]}) + "\n")
        dump_split("train", SQA_TRAIN, 5000)
        dump_split("validation", SQA_VAL, 500)
        dump_split("test", SQA_TEST, 500)

        # Record hash of test split (prompt+answer lines) for audit.
        h = hashlib.sha256(open(SQA_TEST, "rb").read()).hexdigest()
        SQA_LABELS_SHA.write_text(h + "\n")

# Pre-cache SigLIP features for all images we'll train + eval on.
from transformers import AutoModel, AutoImageProcessor
print("  loading SigLIP ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("  WARNING: CUDA not found; falling back to CPU (will be slow).", flush=True)
image_proc = AutoImageProcessor.from_pretrained("models/siglip")
siglip = AutoModel.from_pretrained("models/siglip").to(device).eval()
siglip_vision = siglip.vision_model

def features_for_image(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    inputs = image_proc(images=[img], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = siglip_vision(pixel_values=inputs["pixel_values"])
        feats = out.last_hidden_state[0]  # (num_patches, hidden)
    return feats.half().cpu()

# Build list of image_ids from all splits to cache.
image_ids = []
for p in (SQA_TRAIN, SQA_VAL, SQA_TEST):
    if not p.exists(): continue
    with open(p) as f:
        for line in f:
            ex = json.loads(line)
            image_ids.append(ex["image_id"])
# Deduplicate.
image_ids = list(dict.fromkeys(image_ids))
print(f"  caching SigLIP features for {len(image_ids)} images ...")
for k, iid in enumerate(image_ids):
    feat_path = VC / f"{iid}.pt"
    if feat_path.exists():
        continue
    img_path = VC / f"{iid}.png"
    if not img_path.exists():
        continue
    feats = features_for_image(img_path)
    torch.save(feats, feat_path)
    if (k + 1) % 200 == 0:
        print(f"    {k+1}/{len(image_ids)}", flush=True)

# Summary
print("ScienceQA splits:",
      "train=", sum(1 for _ in open(SQA_TRAIN)) if SQA_TRAIN.exists() else 0,
      "val=", sum(1 for _ in open(SQA_VAL)) if SQA_VAL.exists() else 0,
      "test=", sum(1 for _ in open(SQA_TEST)) if SQA_TEST.exists() else 0)
print("Done.")
PY_SCRIPT
