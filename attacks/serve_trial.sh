#!/usr/bin/env bash
# Serve Guard + 3 representative targets for the attack trial, on the currently
# FREE cards only (GPU 3/4/5 are occupied by other jobs). One model per GPU.
#   Guard-3   -> GPU6   Llama-2 -> GPU2   Mistral -> GPU0   Qwen -> GPU1
set -uo pipefail
ROOT="$HOME/projects/FITEE26"; M="$HOME/models"; LOG="$HOME/vllm_logs"
CSH="$HOME/miniconda3/etc/profile.d/conda.sh"; TPL="$ROOT/scripts/chat_templates"
mkdir -p "$LOG"

# name|dir|gpu|port|template(empty=builtin)|util|maxlen
SPECS=(
  "Llama-Guard-3-8B|Llama-Guard-3-8B|6|8006||0.50|2048"
  "Llama-2-7b|Llama-2-7b-chat|2|8000|$TPL/llama-2.jinja|0.85|4096"
  "Mistral-7B-v0.1|Mistral-7B-Instruct-v0.1|0|8001||0.45|4096"
  "Qwen-7B|Qwen-7B-Chat|1|8002|$TPL/qwen.jinja|0.45|4096"
)
serve(){ local name="$1" dir="$2" gpu="$3" port="$4" tpl="$5" util="$6" maxlen="$7"
  local sess="vllm_$port"; tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len $maxlen --enforce-eager \
       --trust-remote-code $ta 2>&1 | tee '$LOG/trial.$port.log'"; }
wait_ready(){ local t=0; while [ $t -lt 480 ]; do
  curl -s -m3 "http://localhost:$1/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  sleep 5; t=$((t+5)); done; echo "[wait] TIMEOUT :$1"; return 1; }

echo "=== $(date) launching trial servers (free cards only) ==="
for spec in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$spec"
  serve "$name" "$dir" "$gpu" "$port" "$tpl" "$util" "$maxlen"; done
for spec in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$spec"
  echo "waiting :$port ($name) ..."; wait_ready "$port" || echo "[WARN] $name not ready"; done
echo "=== $(date) trial servers up ==="
