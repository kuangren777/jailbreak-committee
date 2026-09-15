#!/usr/bin/env bash
# Phase-2 of the rotation: after part1_new finishes, free GPU0/1/2 (kill the
# Llama-2/Mistral/Qwen trial servers) and serve the three remaining pool models
# there. Guard stays up on GPU6. internlm/vicuna/Baichuan2 have no built-in chat
# template so each gets its .jinja, else vLLM 400s and ASR is a fake 0.000.
set -uo pipefail
M="$HOME/models"; LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="${PROJECT_ROOT:-$HOME/projects/committee}/scripts/chat_templates"

for p in 8000 8001 8002; do tmux kill-session -t vllm_$p 2>/dev/null || true; done
sleep 5

# name|dir|gpu|port|template|util|maxlen
SPECS=(
  "internlm-7b|internlm-chat-7b|0|8000|$TPL/internlm.jinja|0.85|2048"
  "vicuna-7b|vicuna-7b-v1.5|1|8001|$TPL/vicuna.jinja|0.85|4096"
  "Baichuan2-7b|Baichuan2-7B-Chat|2|8002|$TPL/baichuan2.jinja|0.85|4096"
)
serve(){ local name="$1" dir="$2" gpu="$3" port="$4" tpl="$5" util="$6" maxlen="$7"
  tmux kill-session -t vllm_$port 2>/dev/null || true
  tmux new-session -d -s vllm_$port \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len $maxlen --enforce-eager \
       --trust-remote-code --chat-template $tpl 2>&1 | tee $LOG/trial.$port.log"; }
wait_ready(){ local t=0; while [ $t -lt 1200 ]; do
  curl -s -m3 "http://localhost:$1/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  sleep 10; t=$((t+10)); done; echo "[wait] TIMEOUT :$1"; return 1; }

echo "=== $(date) serving new3 (internlm/vicuna/Baichuan2) on GPU0/1/2 ==="
for spec in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$spec"
  serve "$name" "$dir" "$gpu" "$port" "$tpl" "$util" "$maxlen"; done
for spec in "${SPECS[@]}"; do IFS='|' read -r name dir gpu port tpl util maxlen <<< "$spec"
  echo "waiting :$port ($name) ..."; wait_ready "$port" || echo "[WARN] $name not ready"; done
echo "=== $(date) new3 servers up ==="
