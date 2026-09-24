# JBS: Buying a Second Model for Jailbreak Defense by Observed Joint Breach

Code and aggregate results for the paper. A one-time calibration records whether a judge
finds each model's answer harmful for every attack prompt and harmful behavior. From these
verdicts Joint-Breach Selection (JBS) measures the joint breach of each committee, the
fraction of behaviors on which one prompt penetrates a majority of its members, ranks
committees over the worst quarter of attack prompts and buys a committee only when a
behavior-level bootstrap gate finds it beats the best single model by a required margin.
On a pool of six open 7B chat models JBS buys Baichuan2 and Llama-2, lowering the released
worst-quarter attack success rate from 0.73 to 0.23. On a pool of six hosted models the
gate keeps the single model.

**Raw attack prompts, model outputs and answer embeddings are withheld. Verdict-level
aggregates are released.**

## Layout

```
attacks/     calibration, judging, embedding and panel scripts
dhr/         executor wrappers for the committee
tools/       committee_select.py (selection and gate), serving_replay.py (adjudicator replay),
             the review analyses and the aggregate json each paper number traces to
tests/       unit tests for the selection and gate code
results/     verdict-level aggregates, no prompt text and no model output
             scale/           open pool, 200 behaviors x 20 variants x 6 models, two judges
             hub2026_scale/   hosted pool, two judges
             panel/           third-party panel verdicts on judge disagreements
withheld/    what is deliberately absent and how to substitute it
reproduce.py recomputes the selection and gate numbers of the paper from results/
```

## Reproducing the paper without any inference

```bash
git clone <this repository> && cd jailbreak-committee
python3 reproduce.py
```

Reads only `results/` and `tools/serving_replay_scale.json`, reruns selection and the
bootstrap gate on both pools and prints each value the paper reports beside the value
recomputed here. Runtime is about three minutes on one CPU core and it needs `numpy`
only. All 22 checks reproduce.

The released-harm numbers come from replaying the adjudicator on BGE-M3 answer
embeddings (`tools/serving_replay.py`). The embeddings are derived from the withheld
model outputs, so the replay results are released as `tools/serving_replay_scale.json`
instead. The review analyses (`tools/review_checks.py`, `tools/review_b.py`,
`tools/paired_split.py`, `tools/rank_check.py`) run on `results/` and those json files.

## Rerunning the measurement from scratch

This regenerates model responses and therefore needs hardware and a prompt generator.

| Stage | What it needs |
|---|---|
| Open pool generation | one 48 GB GPU, vLLM, six 7B chat models served in sequence (`attacks/scale_suite.sh`) |
| Open pool judging | Llama Guard 3 8B and Qwen3-14B (`attacks/judge2_resume.py`) |
| Answer embeddings | BGE-M3 (`attacks/embed_responses.py`) |
| Hosted pool generation and judging | an OpenAI-compatible gateway, no local GPU (`attacks/hub_scale.sh`, `attacks/hub_judge.py`) |
| Panel audit of judge disagreements | the same gateway (`attacks/panel_adjudicate.py`) |

`attacks/trial_attacks.py` builds the prompt variants and is **not** in this repository.
See `withheld/README.md` for the interface to implement and why it is absent.

The hosted pool is served by an OpenAI-compatible gateway. Point the client at it with
`GATEWAY_URL` and `GATEWAY_KEY`, or with a mode-600 file named by `GATEWAY_ENV_FILE`
that carries those two keys. Credentials are never written to this repository or to any
result file.

Earlier scripts from the first version of this study (`attacks/gate.py`,
`attacks/pair_*.py`, `results/matrix`, `results/pair`) remain for reference and are not
needed for the numbers above.

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
