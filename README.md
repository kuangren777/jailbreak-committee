# Selecting a Two-Model Committee for Jailbreak Defense with a Calibrated Identifiability Gate

Code and aggregate results for the paper. A deployer calibrates one susceptibility
matrix, ranks two-model committees by a bound that holds under any dependence between
members, and runs a bootstrap gate that says whether that ranking is identified at all.
On a 2023 pool of six open-weight chat models the gate accepts a pair that cuts the
worst-case attack success rate from 0.68 to 0.48. On a second pool of six hosted models
the same gate declines the second model under two independent judges.

**Raw attack prompts and model outputs are withheld. Aggregate verdicts are released.**

## Layout

```
attacks/     calibration, judging, adjudication and gate scripts
dhr/         executor wrappers for the committee
scripts/     shared calibration helper
tools/       analysis scripts and the aggregate json each paper number traces to
results/     verdict-level aggregates, no prompt text and no model output
withheld/    what is deliberately absent and how to substitute it
reproduce.py recomputes every gate number in the paper from results/
```

## Reproducing the paper without any inference

```bash
git clone <this repository> && cd jailbreak-committee
python3 reproduce.py
```

Reads only `results/`, runs the bootstrap gate on both pools, and prints the value the
paper reports beside the value recomputed here. Runtime is about twenty seconds on a
laptop, and it needs `numpy` and nothing else. Nineteen of nineteen checks reproduce.

Figures come from `tools/make_icassp_figs.py`, which reads the same aggregates.

## Rerunning the measurement from scratch

This regenerates model responses and therefore needs hardware and a prompt generator.

| Stage | What it needs | Cost |
|---|---|---|
| First pool generation | one 80 GB GPU, vLLM, six 7B chat models served in sequence | about 6 h for 6 models times 20 variants times 25 behaviors |
| First pool judging | Llama Guard 3 8B and Qwen3-14B on the same GPU | about 1 h per judge for 3000 responses |
| Second pool generation | an OpenAI-compatible gateway, no local GPU | about 2 h at 6 to 12 concurrent requests |
| Second pool judging | the same gateway, two judges outside the pool | about 1 h per judge |
| Adaptive attacker | one GPU for the attacker model and the target | about 2 h for 25 behaviors at 4 iterations |

`attacks/trial_attacks.py` builds the prompt variants and is **not** in this repository.
See `withheld/README.md` for the interface to implement and why it is absent.

Entry points, in order:

```bash
python3 attacks/calibrate_vuln.py      # first pool, serve with attacks/serve_*.sh first
python3 attacks/rescore_judge.py       # rebuild the matrix once per judge
python3 attacks/hub_calibrate.py       # second pool, one model per invocation
python3 attacks/hub_judge.py           # second pool, one judge per invocation
python3 attacks/gate.py                # the identifiability gate
python3 attacks/pool_summary.py        # per-model worst case and judge agreement
python3 attacks/pair_released.py       # what the adjudicator actually releases
python3 attacks/pair_adaptive.py       # tuned attacker against the deployed pair
```

The second pool is served by an OpenAI-compatible gateway. Point the client at it with
`GATEWAY_URL` and `GATEWAY_KEY`, or with a mode-600 file named by `GATEWAY_ENV_FILE`
that carries those two keys. Credentials are never written to this repository or to any
result file.

## Models

**First pool, open weights, released in 2023.** `Qwen/Qwen-7B-Chat`,
`meta-llama/Llama-2-7b-chat-hf`, `internlm/internlm-chat-7b`,
`mistralai/Mistral-7B-Instruct-v0.1`, `lmsys/vicuna-7b-v1.5`,
`baichuan-inc/Baichuan2-7B-Chat`. Served
with vLLM using each model's native chat template, greedy decoding and a 512-token
limit. Judged by `meta-llama/Llama-Guard-3-8B` and, independently, by `Qwen/Qwen3-14B`.
The adjudicator embeds answers with `BAAI/bge-m3`.

**Second pool, hosted, all released after 2023.** GPT-4o (`gpt-4o-2024-11-20`),
Claude Haiku 4.5, Gemini 2.5 Flash, DeepSeek-V3.1, GLM-4.5-Air, Qwen-Flash. Queried
through one OpenAI-compatible gateway in September 2026 at temperature zero with the
same 512-token limit. Judged by Grok-4 and by Kimi-K2.5, neither built by a developer
of any pool member.

One member of the second pool first returned empty content on 463 of its 500 prompts.
Re-running five of them at eight times the budget produced 5000 to 9000 characters
each, which identified the token budget rather than a refusal as the cause, so it was
replaced by GPT-4o and the budget was kept uniform.

## Data

Harmful behaviors come from AdvBench, selected in a fixed hash order, and benign
behaviors from JailbreakBench. Both are public research benchmarks distributed under
their own licenses, and neither is redistributed here. The seven attacks are black-box
reimplementations of published methods, cited in the paper.

## Responsible use

This work measures how aligned chat models fail together so that a deployer can decide
whether a second model is worth its inference cost. The same measurement tells an
attacker where a committee is weakest, which is why the release is limited to
verdict-level aggregates. No jailbreak prompt, no tuned attacker output and no harmful
model completion appears anywhere in this repository. Do not use this code to attack
systems you do not own or have permission to test. If you extend it, keep the raw
generations off public storage, as we have.

## License

MIT, see `LICENSE`. The benchmarks and the models carry their own licenses.
