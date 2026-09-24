# Withheld module

`attacks/trial_attacks.py` is deliberately absent. It builds the twenty prompt
variants of the seven attacks, which makes it a working jailbreak prompt generator,
and the paper states that no jailbreak prompt is released with these artifacts.

Every attack is a black-box reimplementation of a published method, and each original
paper ships a reference implementation: FlipAttack, CodeChameleon, SelfCipher,
ReNeLLM, DeepInception, PAP, and the past-tense attack. To rerun the generation step,
take those implementations and expose

```python
attack_names() -> list[str]
variants(attack_name: str, behavior: str) -> list[tuple[str, str]]
```

where `variants` returns `(variant_name, prompt)` pairs. The variant names this code
expects appear in the `variant` field of every record under `results/`, so the column
order of the susceptibility matrix is recoverable without any prompt text.

The BGE-M3 answer embeddings (`results/scale/emb.npy`) are also withheld, since they are
computed from the model outputs. `results/scale/emb.jsonl` keeps their row order and refusal
flags, and the replay they feed is released as `tools/serving_replay_scale.json`.

Nothing else downstream of generation is withheld. The aggregate verdicts under `results/`
reproduce every number and every figure in the paper with no model inference at all.
