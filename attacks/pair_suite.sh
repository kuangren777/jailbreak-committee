#!/usr/bin/env bash
# Adaptive (PAIR-style) attack against the recommended pair, on TWO gpus only.
#   ATT_GPU (default 4): attacker model.  TGT_GPU (default 6): target + judge + 2nd member.
# Phase 1 optimizes a prompt per behavior against member 1 (Llama-2). Phase 2 runs the
# same adapted prompts through member 2 (Baichuan2). Phase 3 scores the disagreement
# detector on the adaptive successes. Leaves the other five cards free.
set -uo pipefail
ATT_GPU="${ATT_GPU:-1}"; TGT_GPU="${TGT_GPU:-6}"; GUARD_GPU="${GUARD_GPU:-3}"
ROOT="$HOME/projects/FITEE26"; M="$HOME/models"; OUT="$ROOT/results/pair"
LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"
mkdir -p "$OUT" "$LOG"

serve() { # name dir port gpu template util maxlen
  local name="$1" dir="$2" port="$3" gpu="$4" tpl="$5" util="$6" maxlen="$7" sess="vllm_$3"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization $util --max-model-len $maxlen --enforce-eager \
       --trust-remote-code $ta 2>&1 | tee '$LOG/pair.$port.log'"
  local t=0
  while [ $t -lt 900 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name :$port"; return 0; }
    sleep 5; t=$((t+5))
  done
  echo "[FAIL] $name :$port"; return 1
}
stop() { tmux kill-session -t "vllm_$1" 2>/dev/null || true; sleep 20; }
run_py() { source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES="$1" python "${@:2}"; }

echo "=== $(date) pair suite: attacker gpu$ATT_GPU target gpu$TGT_GPU ==="

# ---- Phase 1: PAIR against member 1 (Llama-2) ----
if [ ! -s "$OUT/pair_llama2.jsonl" ]; then
  serve "Qwen3-14B"        "Qwen3-14B"        8010 "$ATT_GPU" ""                 0.85 4096 || exit 1
  serve "Llama-2-7b"       "Llama-2-7b-chat"  8000 "$TGT_GPU" "$TPL/llama-2.jinja" 0.40 4096 || exit 1
  serve "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8006 "$GUARD_GPU" ""                 0.70 4096 || exit 1
  run_py "$TGT_GPU" "$ROOT/attacks/pair_adaptive.py" \
     --attacker-port 8010 --target-port 8000 --guard-port 8006 \
     --target-name Llama-2-7b --n 25 --iters 8 --out "$OUT/pair_llama2.jsonl"
  stop 8010; stop 8000
fi

# ---- Phase 2: same adapted prompts through member 2 (Baichuan2), Guard stays up ----
if [ ! -s "$OUT/second_baichuan2.jsonl" ]; then
  curl -s -m3 "http://localhost:8006/v1/models" 2>/dev/null | grep -q '"id"' || \
     serve "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8006 "$GUARD_GPU" "" 0.70 4096 || exit 1
  serve "Baichuan2-7b" "Baichuan2-7B-Chat" 8005 "$TGT_GPU" "$TPL/baichuan2.jinja" 0.40 4096 || exit 1
  run_py "$TGT_GPU" "$ROOT/attacks/pair_second_member.py" \
     --pair "$OUT/pair_llama2.jsonl" --port 8005 --guard-port 8006 \
     --name Baichuan2-7b --out "$OUT/second_baichuan2.jsonl"
  stop 8005; stop 8006
fi

# ---- Phase 3: disagreement detector on the adaptive successes ----
run_py "$ATT_GPU" "$ROOT/attacks/pair_detect.py" \
   --pair "$OUT/pair_llama2.jsonl" --second "$OUT/second_baichuan2.jsonl" \
   --benign-dump "$ROOT/results/live_pair/disagree_pairs.jsonl" \
   --out "$OUT/pair_detect.json"

echo "=== $(date) pair suite done ==="
echo "---GPU $ATT_GPU/$TGT_GPU"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "$((ATT_GPU+1))p;$((TGT_GPU+1))p"
echo "---TMUX"; tmux ls 2>/dev/null | grep vllm_ || echo none
