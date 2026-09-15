#!/usr/bin/env bash
# Second matrix at a uniform budget, then both judges.
#
# The calibration run stored verdicts but no text, so it cannot be rescored. Here we
# regenerate ALL SIX pool models on the same 20 prompt variants and 25 behaviors at a
# budget every model supports (context 2048, 512 new tokens, matching the original
# calibration), so the six rows are comparable. Scoring these responses with Llama Guard
# and with an independent judge answers whether the pair selected under the guard is
# still the argmin over the 15 pairs.
# One model on one card at a time, so the other cards stay free.
set -uo pipefail
GPU="${GPU:-2}"
ROOT="${PROJECT_ROOT:-$HOME/projects/committee}"; M="$HOME/models"; OUT="$ROOT/results/rescore"
LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"
mkdir -p "$OUT"

serve() { local name="$1" dir="$2" port="$3" tpl="$4" util="$5" sess="vllm_$3"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len 2048 --enforce-eager \
       --trust-remote-code $ta 2>&1 | tee '$LOG/rs.$port.log'"
  local t=0; while [ $t -lt 900 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name"; return 0; }
    sleep 5; t=$((t+5)); done; echo "[FAIL] $name"; return 1; }
stop() { # kill the server then WAIT until the card actually releases its memory,
         # otherwise the next model OOMs at load and vLLM reports no cache blocks
  tmux kill-session -t "vllm_$1" 2>/dev/null || true
  local base t=0
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  while [ $t -lt 180 ]; do
    sleep 10; t=$((t+10))
    local now; now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$now" -le $((base - 8000)) ] && { echo "[freed] gpu$GPU ${base}->${now} MiB after ${t}s"; return 0; }
    base=$now
  done
  echo "[warn] gpu$GPU memory did not drop within ${t}s"; }
run_py() { source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES="$GPU" python "$@"; }

echo "=== $(date) rescore suite on gpu$GPU ==="

MODELS=(
  "Llama-2-7b|Llama-2-7b-chat|$TPL/llama-2.jinja"
  "Baichuan2-7b|Baichuan2-7B-Chat|$TPL/baichuan2.jinja"
  "Qwen-7B|Qwen-7B-Chat|$TPL/qwen.jinja"
  "internlm-7b|internlm-chat-7b|$TPL/internlm.jinja"
  "vicuna-7b|vicuna-7b-v1.5|$TPL/vicuna.jinja"
  "Mistral-7B-v0.1|Mistral-7B-Instruct-v0.1|"
)
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name dir tpl <<< "$spec"
  raw="$OUT/raw_${name}.jsonl"
  [ -s "$raw" ] && { echo "[skip] $name"; continue; }
  serve "$name" "$dir" 8030 "$tpl" 0.72 || { echo "[FAILSERVE] $name"; continue; }
  run_py "$ROOT/attacks/live_gen.py" --name "$name" --port 8030 --n 25 \
      --max-tokens 512 --n-benign 0 --out "$raw"
  stop 8030
done

echo "=== generation done $(date) ==="
wc -l "$OUT"/raw_*.jsonl
