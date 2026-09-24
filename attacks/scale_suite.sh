#!/usr/bin/env bash
# Scaled calibration: 200 behaviors per cell instead of 25.
#
# Same six 2023 models, same 20 prompt variants, same 512-token budget as the rebuilt
# matrix, but 200 AdvBench behaviors in the same hash order (so the original 25 are the
# first 25) plus the 100 JailbreakBench benign behaviors for the throughput side. Every
# response is scored by Llama Guard 3 and by Qwen3-14B, and embedded once with BGE-M3 so
# the adjudicator can be replayed offline for any committee and threshold.
# Generation fans the six models over the free cards, two waves; judges run in parallel.
set -uo pipefail
GPUS=(${GPUS:-0 2 3 4})
N="${N:-200}"
ROOT="${ROOT:-$HOME/jailbreak-committee}"; M="$HOME/models"; OUT="$ROOT/results/scale"
LOG="$HOME/vllm_logs"; CSH="$HOME/miniconda3/etc/profile.d/conda.sh"
TPL="$ROOT/scripts/chat_templates"
mkdir -p "$OUT" "$LOG"

serve() { local gpu="$1" name="$2" dir="$3" port="$4" tpl="$5" maxlen="$6" extra="${7:-}" sess="vllm_$4"
  tmux kill-session -t "$sess" 2>/dev/null || true
  local ta=""; [ -n "$tpl" ] && ta="--chat-template $tpl"
  tmux new-session -d -s "$sess" \
    "source $CSH; conda activate lmy_vllm; \
     CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
       --model '$M/$dir' --served-model-name '$name' --port $port \
       --gpu-memory-utilization ${UTIL:-0.85} --max-model-len $maxlen --enforce-eager \
       --trust-remote-code $ta $extra 2>&1 | tee '$LOG/scale.$port.log'"
  local t=0; while [ $t -lt 1200 ]; do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo "[up] $name gpu$gpu"; return 0; }
    sleep 5; t=$((t+5)); done; echo "[FAIL] $name"; return 1; }
stop() { tmux kill-session -t "vllm_$1" 2>/dev/null || true; sleep 30; }
run_py() { local gpu="$1"; shift; ( source "$CSH"; conda activate lmy_llm; CUDA_VISIBLE_DEVICES="$gpu" python "$@" ); }

gen_one() { # gpu name dir tpl port
  local gpu="$1" name="$2" dir="$3" tpl="$4" port="$5" raw="$OUT/raw_$2.jsonl"
  [ -s "$raw" ] && [ "$(wc -l < "$raw")" -ge $((N * 20 + 100)) ] && { echo "[skip] $name"; return 0; }
  serve "$gpu" "$name" "$dir" "$port" "$tpl" 2048 "--max-num-seqs 64" || return 1
  run_py "$gpu" "$ROOT/attacks/live_gen.py" --name "$name" --port "$port" --n "$N" \
      --max-tokens 512 --n-benign 100 --benign-max-tokens 512 --workers 48 --out "$raw.tmp" \
    && mv "$raw.tmp" "$raw"
  stop "$port"
}

echo "=== $(date) scale suite N=$N gpus=${GPUS[*]} ==="
MODELS=(
  "Llama-2-7b|Llama-2-7b-chat|$TPL/llama-2.jinja"
  "Baichuan2-7b|Baichuan2-7B-Chat|$TPL/baichuan2.jinja"
  "Qwen-7B|Qwen-7B-Chat|$TPL/qwen.jinja"
  "internlm-7b|internlm-chat-7b|$TPL/internlm.jinja"
  "vicuna-7b|vicuna-7b-v1.5|$TPL/vicuna.jinja"
  "Mistral-7B-v0.1|Mistral-7B-Instruct-v0.1|"
)
i=0
while [ $i -lt ${#MODELS[@]} ]; do
  pids=()
  for g in "${GPUS[@]}"; do
    [ $i -ge ${#MODELS[@]} ] && break
    IFS='|' read -r name dir tpl <<< "${MODELS[$i]}"
    gen_one "$g" "$name" "$dir" "$tpl" $((8040 + i)) > "$LOG/scale.gen.$name.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  wait "${pids[@]}"
done
echo "=== generation done $(date) ==="; wc -l "$OUT"/raw_*.jsonl

G0=${GPUS[0]}; G1=${GPUS[1]}; G2=${GPUS[2]}
(
  [ -s "$OUT/verdicts_guard.jsonl" ] || {
    serve "$G0" "Llama-Guard-3-8B" "Llama-Guard-3-8B" 8050 "" 4096 \
      && run_py "$G0" "$ROOT/attacks/live_judge.py" --raw "$OUT"/raw_*.jsonl --guard-port 8050 \
           --workers 48 --out "$OUT/verdicts_guard.jsonl"
    stop 8050; }
) > "$LOG/scale.guard.log" 2>&1 &
(
  [ -s "$OUT/verdicts_judge2.jsonl" ] || {
    serve "$G1" "Qwen3-14B" "Qwen3-14B" 8051 "" 8192 "--max-num-seqs 128 ${JUDGE2_EXTRA:-}" \
      && run_py "$G1" "$ROOT/attacks/live_judge2.py" --raw "$OUT"/raw_*.jsonl --port 8051 \
           --workers 96 --judge-max-tokens 512 --out "$OUT/verdicts_judge2.jsonl"
    stop 8051; }
) > "$LOG/scale.judge2.log" 2>&1 &
(
  [ -s "$OUT/emb.npy" ] || run_py "$G2" "$ROOT/attacks/embed_responses.py" \
      --raw "$OUT"/raw_*.jsonl --out "$OUT/emb"
) > "$LOG/scale.embed.log" 2>&1 &
wait
echo "=== all done $(date) ==="; ls -la "$OUT"
