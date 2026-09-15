#!/usr/bin/env bash
# Serve a 3-model committee + Guard for D1 live validation, on the cards with the most
# free memory (5 fully free; 0/1/6 partially free). One model per GPU.
#   Guard -> GPU5   Llama-2 -> GPU6   internlm -> GPU0   Baichuan2 -> GPU1
set -uo pipefail
M="$HOME/models"; LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$HOME/projects/FITEE26/scripts/chat_templates"; mkdir -p "$LOG"
# name|dir|gpu|port|template|util|maxlen
SPECS=(
  "Llama-Guard-3-8B|Llama-Guard-3-8B|5|8006||0.45|2048"
  "Llama-2-7b|Llama-2-7b-chat|6|8000|$TPL/llama-2.jinja|0.90|4096"
  "internlm-7b|internlm-chat-7b|0|8001|$TPL/internlm.jinja|0.92|2048"
  "Baichuan2-7b|Baichuan2-7B-Chat|1|8002|$TPL/baichuan2.jinja|0.92|4096"
)
serve(){ local name="$1" dir="$2" gpu="$3" port="$4" tpl="$5" util="$6" maxlen="$7"
  tmux kill-session -t vllm_$port 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s vllm_$port \
    "source $CSH; conda activate lmy_vllm; CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port --gpu-memory-utilization $util \
       --max-model-len $maxlen --enforce-eager --trust-remote-code $ta 2>&1 | tee $LOG/val.$port.log"; }
wait_ready(){ local t=0; while [ $t -lt 1200 ]; do
  curl -s -m3 "http://localhost:$1/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  sleep 10; t=$((t+10)); done; echo "[wait] TIMEOUT :$1"; return 1; }
echo "=== $(date) serving validation committee ==="
for s in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$s"; serve "$name" "$dir" "$gpu" "$port" "$tpl" "$util" "$maxlen"; done
for s in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$s"; echo "waiting :$port ($name)"; wait_ready "$port" || echo "[WARN] $name not ready"; done
echo "=== $(date) validation servers up ==="
