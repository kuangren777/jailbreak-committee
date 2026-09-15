#!/usr/bin/env bash
# Live pair study on ONE gpu (default 0), serving each model in turn so that the other
# cards stay free for other users. Phases: generate per member, judge with Llama Guard 3,
# judge with an independent model, then adjudicate with the real BGE-M3 clustering.
set -uo pipefail
GPU="${GPU:-0}"
ROOT="${PROJECT_ROOT:-$HOME/projects/committee}"; M="$HOME/models"; OUT="$ROOT/results/live_pair"
LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"
mkdir -p "$OUT" "$LOG"

serve() { # name dir port template util maxlen
  local name="$1" dir="$2" port="$3" tpl="$4" util="$5" maxlen="$6" sess="vllm_$3"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len $maxlen --enforce-eager \
       --trust-remote-code $ta 2>&1 | tee '$LOG/live.$port.log'"
  local t=0
  while [ $t -lt 900 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name :$port"; return 0; }
    sleep 5; t=$((t+5))
  done
  echo "[FAIL] $name did not start"; return 1
}
stop() { tmux kill-session -t "vllm_$1" 2>/dev/null || true; sleep 8; }
run_py() { source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES=$GPU python "$@"; }

echo "=== $(date) live suite on GPU $GPU ==="

if [ ! -s "$OUT/raw_Llama-2-7b.jsonl" ]; then
  serve "Llama-2-7b" "Llama-2-7b-chat" 8000 "$TPL/llama-2.jinja" 0.90 4096 || exit 1
  run_py "$ROOT/attacks/live_gen.py" --name Llama-2-7b --port 8000 --out "$OUT/raw_Llama-2-7b.jsonl"
  stop 8000
fi

if [ ! -s "$OUT/raw_Baichuan2-7b.jsonl" ]; then
  serve "Baichuan2-7b" "Baichuan2-7B-Chat" 8005 "$TPL/baichuan2.jinja" 0.90 4096 || exit 1
  run_py "$ROOT/attacks/live_gen.py" --name Baichuan2-7b --port 8005 --out "$OUT/raw_Baichuan2-7b.jsonl"
  stop 8005
fi

if [ ! -s "$OUT/verdicts_guard.jsonl" ]; then
  serve "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8006 "" 0.90 4096 || exit 1
  run_py "$ROOT/attacks/live_judge.py" --raw "$OUT/raw_*.jsonl" --guard-port 8006 \
      --out "$OUT/verdicts_guard.jsonl"
  stop 8006
fi

if [ ! -s "$OUT/verdicts_judge2.jsonl" ]; then
  serve "Qwen3-14B" "Qwen3-14B" 8007 "" 0.90 8192 || exit 1
  run_py "$ROOT/attacks/live_judge2.py" --raw "$OUT/raw_*.jsonl" --port 8007 \
      --out "$OUT/verdicts_judge2.jsonl"
  stop 8007
fi

run_py "$ROOT/attacks/live_adjudicate.py" \
  --member "Llama-2-7b=$OUT/raw_Llama-2-7b.jsonl" "Baichuan2-7b=$OUT/raw_Baichuan2-7b.jsonl" \
  --guard-verdicts "$OUT/verdicts_guard.jsonl" --judge2-verdicts "$OUT/verdicts_judge2.jsonl" \
  --out "$OUT/live_pair.json"

echo "=== $(date) live suite done ==="
