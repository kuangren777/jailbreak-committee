#!/usr/bin/env bash
# Second pool at 200 behaviors: resume the 25-behavior hosted calibration to 200 in a copy,
# then score every response with the same two gateway judges. The original 25-behavior files
# in results/hub2026 stay untouched. No GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/hub2026_scale; mkdir -p "$OUT" temp
POOL="claude-haiku45|claude-haiku-4-5 deepseek-v31|deepseek-v3.1 gemini25flash|gemini-2.5-flash glm45air|glm-4.5-air gpt4o|gpt-4o-2024-11-20 qwen-flash|qwen-flash"
for spec in $POOL; do
  name=${spec%%|*}; id=${spec##*|}
  [ -s "$OUT/raw_$name.jsonl" ] || cp "results/hub2026/raw_$name.jsonl" "$OUT/raw_$name.jsonl"
  python3 attacks/hub_calibrate.py --model "$id" --name "$name" --n 200 --workers 16 \
     --out "$OUT/raw_$name.jsonl" > "temp/hub_scale.$name.log" 2>&1 &
done
wait
echo "[gen done] $(date)"; wc -l "$OUT"/raw_*.jsonl
python3 attacks/hub_judge.py --raw "$OUT/raw_*.jsonl" --judge grok-4 --field verdict_a --workers 24 \
   --out "$OUT/verdicts_grok4.jsonl" > temp/hub_scale.judge_grok4.log 2>&1 &
python3 attacks/hub_judge.py --raw "$OUT/raw_*.jsonl" --judge kimi-k2.5 --field verdict_b --workers 24 \
   --out "$OUT/verdicts_kimi.jsonl" > temp/hub_scale.judge_kimi.log 2>&1 &
wait
echo "[judge done] $(date)"; wc -l "$OUT"/verdicts_*.jsonl
