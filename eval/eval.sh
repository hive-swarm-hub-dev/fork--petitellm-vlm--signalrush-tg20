#!/usr/bin/env bash
# Evaluate train.py for petitellm-vlm.
set -uo pipefail
cd "$(dirname "$0")/.."

summary() {
    local acc="${1:-ERROR}"
    local artifact_bytes="${2:-0}"
    local line_count="${3:-0}"
    local valid="${4:-false}"
    echo "---"
    printf "vqa_acc:           %s\n" "$acc"
    printf "artifact_bytes:    %s\n" "$artifact_bytes"
    printf "line_count:        %s\n" "$line_count"
    printf "valid:             %s\n" "$valid"
}

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found." >&2; summary; exit 0
fi
LINE_COUNT=$(wc -l < train.py | tr -d ' ')

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
fi

for f in models/siglip/config.json models/qwen/config.json data/sqa_train.jsonl data/sqa_val.jsonl data/sqa_test.jsonl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f. Run: bash prepare.sh" >&2
        summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
    fi
done

TMPLOG=$(mktemp); trap 'rm -f "$TMPLOG"' EXIT

TRAIN_EXIT=0
timeout 720 python3 train.py 2>&1 | tee "$TMPLOG" >&2 || TRAIN_EXIT=$?
if [ "$TRAIN_EXIT" -ne 0 ] && [ "$TRAIN_EXIT" -ne 124 ]; then
    echo "ERROR: training exited $TRAIN_EXIT." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
fi

if [ ! -f final_projection.ptz ]; then
    echo "ERROR: final_projection.ptz not found." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
fi

EVAL_EXIT=0
EVAL_OUT=$(python3 eval/evaluate.py 2>&1) || EVAL_EXIT=$?
echo "$EVAL_OUT" >&2
if [ "$EVAL_EXIT" -ne 0 ]; then
    echo "ERROR: evaluate.py exited $EVAL_EXIT." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
fi

ACC=$(printf '%s' "$EVAL_OUT" | grep -oE 'vqa_acc=[0-9]+\.[0-9]+' | tail -1 | cut -d= -f2)
if [ -z "$ACC" ]; then
    echo "ERROR: could not parse vqa_acc." >&2
    summary "ERROR" "0" "$LINE_COUNT" "false"; exit 0
fi

# Artifact bytes: train.py + every *.pt, *.ptz, *.safetensors in task root.
ARTIFACT_BYTES=$(python3 - <<'PY'
import os, glob
total = os.path.getsize("train.py")
for pat in ("*.pt", "*.ptz", "*.safetensors"):
    for p in glob.glob(pat):
        total += os.path.getsize(p)
print(total)
PY
)

VALID="true"
if [ "$ARTIFACT_BYTES" -gt 16000000 ]; then
    echo "WARNING: artifact_bytes $ARTIFACT_BYTES > 16,000,000" >&2
    VALID="false"
fi

summary "$ACC" "$ARTIFACT_BYTES" "$LINE_COUNT" "$VALID"
