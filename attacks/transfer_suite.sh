#!/usr/bin/env bash
# Transfer matrix on ONE card (GPU, default 3): serve each pool model in turn, generate
# answers to the PAIR-adapted prompts (saved text), then serve Llama Guard once and judge
# all of them. One model on the card at any time, so no co-tenant OOM. Baichuan2 reuses
# the verdicts from the pair run.
set -uo pipefail
GPU="${GPU:-3}"
ROOT="$HOME/projects/FITEE26"; M="$HOME/models"; OUT="$ROOT/results/pair"
RAW="$OUT/transfer_raw"; LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"; PAIR="$OUT/pair_llama2.jsonl"
mkdir -p "$RAW"

serve() { local name="$1" dir="$2" port="$3" tpl="$4" util="$5" sess="vllm_$3"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len 2048 --enforce-eager \
       --trust-remote-code $ta 2>&1 | tee '$LOG/tr.$port.log'"
  local t=0; while [ $t -lt 900 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name"; return 0; }
    sleep 5; t=$((t+5)); done; echo "[FAIL] $name"; return 1; }
stop() { tmux kill-session -t "vllm_$1" 2>/dev/null || true; sleep 20; }
run_py() { source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES="$GPU" python "$@"; }

echo "=== $(date) transfer matrix on gpu$GPU ==="
MODELS=(
  "Qwen-7b|Qwen-7B-Chat|$TPL/qwen.jinja"
  "InternLM-7b|internlm-chat-7b|$TPL/internlm.jinja"
  "Vicuna-7b|vicuna-7b-v1.5|$TPL/vicuna.jinja"
  "Mistral-7b|Mistral-7B-Instruct-v0.1|"
)
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name dir tpl <<< "$spec"
  raw="$RAW/${name}.jsonl"; [ -s "$raw" ] && { echo "[skip gen] $name"; continue; }
  serve "$name" "$dir" 8020 "$tpl" 0.72 || { echo "[FAIL serve] $name"; continue; }
  run_py "$ROOT/attacks/transfer_gen.py" --pair "$PAIR" --port 8020 --name "$name" --out "$raw"
  stop 8020
done

serve "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8006 "" 0.68 || exit 1
run_py "$ROOT/attacks/transfer_judge.py" --raw "$RAW"/*.jsonl --guard-port 8006 --outdir "$OUT"
stop 8006

run_py "$ROOT/attacks/transfer_agg.py" --pair "$PAIR" \
   --members "$OUT/second_baichuan2.jsonl" "$OUT"/transfer_*.jsonl --out "$OUT/transfer_matrix.json"
echo "=== $(date) transfer matrix done ==="
echo "---GPU"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader|sed -n "$((GPU+1))p"
tmux ls 2>/dev/null|grep vllm_||echo notmux
