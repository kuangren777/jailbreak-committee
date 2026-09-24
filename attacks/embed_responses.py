"""Embed every stored response once with BGE-M3, so the adjudicator can be replayed offline.

The adjudicator needs only pairwise cosine similarity and whether an answer opens with a
refusal phrase. Storing one unit vector and one refusal flag per response lets any
committee size, quorum and threshold be replayed locally without moving generated text off
the serving host. Output: <out>.npy (float16 unit vectors) and <out>.jsonl (one key record
per row, same order). Verdict-level metadata only, never text.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from dhr.executors import is_refusal   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--model-path", default=os.path.expanduser("~/models/bge-m3"))
    ap.add_argument("--out", required=True, help="path prefix")
    a = ap.parse_args()

    meta, texts = [], []
    for p in [q for pat in a.raw for q in sorted(glob.glob(pat))]:
        for line in open(p):
            r = json.loads(line)
            if "text" not in r:
                continue
            meta.append({"model": r["model"], "kind": r["kind"], "attack": r["attack"],
                         "variant": r["variant"], "bsha": r["bsha"],
                         "refusal": is_refusal(r["text"]), "trunc": r.get("finish") == "length",
                         "empty": not r["text"].strip()})
            texts.append(r["text"])
    print("[embed] %d responses" % len(texts), flush=True)

    import numpy as np
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(a.model_path, device="cuda")
    emb = st.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    np.save(a.out + ".npy", emb.astype(np.float16))
    with open(a.out + ".jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print("[embed] wrote %s.npy %s" % (a.out, emb.shape), flush=True)


if __name__ == "__main__":
    main()
