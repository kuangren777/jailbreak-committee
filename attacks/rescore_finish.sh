#!/usr/bin/env bash
# Finish the second matrix: generate the last model, then score every stored response
# with BOTH judges and rebuild the pair ranking under each. One card, one model at a time.
set -uo pipefail
GPU="${GPU:-1}"
ROOT="${PROJECT_ROOT:-$HOME/projects/committee}"; M="$HOME/models"; OUT="$ROOT/results/rescore"
LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"

serve() { local name="$1" dir="$2" port="$3" tpl="$4" util="$5" maxlen="$6" extra="${7:-}" sess="vllm_$3"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len $maxlen --enforce-eager \
       --trust-remote-code $ta $extra 2>&1 | tee '$LOG/fin.$port.log'"
  local t=0; while [ $t -lt 900 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name"; return 0; }
    sleep 5; t=$((t+5)); done; echo "[FAIL] $name"; return 1; }
stop() { tmux kill-session -t "vllm_$1" 2>/dev/null || true
  local base t=0; base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  while [ $t -lt 180 ]; do sleep 10; t=$((t+10))
    local now; now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$now" -le $((base - 8000)) ] && { echo "[freed] ${base}->${now}"; return 0; }
    base=$now; done; echo "[warn] no drop in ${t}s"; }
run_py() { source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES="$GPU" python "$@"; }

echo "=== $(date) finish on gpu$GPU ==="

# 1. the one model that never served, with a small warmup batch
if [ ! -s "$OUT/raw_Qwen-7B.jsonl" ]; then
  serve "Qwen-7B" "Qwen-7B-Chat" 8032 "$TPL/qwen.jinja" 0.85 2048 "--max-num-seqs 32" \
    && run_py "$ROOT/attacks/live_gen.py" --name Qwen-7B --port 8032 --n 25 \
         --max-tokens 512 --n-benign 0 --workers 8 --out "$OUT/raw_Qwen-7B.jsonl"
  stop 8032
fi

# 2. Llama Guard over every stored response
if [ ! -s "$OUT/verdicts_guard.jsonl" ]; then
  serve "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8033 "" 0.85 4096 \
    && run_py "$ROOT/attacks/live_judge.py" --raw "$OUT"/raw_*.jsonl --guard-port 8033 \
         --out "$OUT/verdicts_guard.jsonl"
  stop 8033
fi

# 3. the independent judge over the same responses
if [ ! -s "$OUT/verdicts_judge2.jsonl" ]; then
  serve "Qwen3-14B" "Qwen3-14B" 8034 "" 0.85 8192 \
    && run_py "$ROOT/attacks/live_judge2.py" --raw "$OUT"/raw_*.jsonl --port 8034 \
         --judge-max-tokens 512 --out "$OUT/verdicts_judge2.jsonl"
  stop 8034
fi

# 4. rebuild the pair ranking under each judge
run_py "$ROOT/attacks/rescore_judge.py" --verdicts "$OUT/verdicts_guard.jsonl" \
   --label llama_guard --out "$OUT/rank_llama_guard.json"
run_py "$ROOT/attacks/rescore_judge.py" --verdicts "$OUT/verdicts_judge2.jsonl" \
   --label qwen3_14b --out "$OUT/rank_qwen3_14b.json"

echo "=== finish done $(date) ==="
wc -l "$OUT"/raw_*.jsonl "$OUT"/verdicts_*.jsonl 2>/dev/null
