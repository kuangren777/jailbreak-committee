"""Regenerate figs/fig_principle.png (Fig. in Sec. 3.8) with gpt-image-2.5.

Reads the OpenAI-compatible gateway from CLAUDISH_OPENAI_URL / CLAUDISH_OPENAI_KEY.
Landscape 1536x1024 so the two stacked rows stay short at single-column width.
The image is illustrative only; every number in the paper comes from the JSON files
next to this script, never from the figure.
"""
import os, json, base64, urllib.request, sys

PROMPT = (
 "Clean flat vector technical schematic for an IEEE paper figure, WIDE landscape, white background, "
 "thin dark outlines, muted academic palette, LARGE BOLD short labels, very few words, symbols over text. "
 "Two rows stacked, each spanning the full width.\n"
 "TOP ROW = the pipeline, left to right with arrows: (1) a query card; (2) a committee of THREE visibly "
 "different LLM robot icons (different colors/shapes, label 'heterogeneous LLMs'); (3) each robot emits a "
 "short answer card; (4) an encoder box labeled 'BGE-M3' turning each card into a small arrow vector; "
 "(5) a grouping box showing two vectors with an angle and the tiny formula 'cos >= tau' that clusters "
 "vectors into groups; (6) a decision diamond with two tiny checks 'majority k?' and 'refusal?'; (7) two "
 "outputs, a green 'Release' badge and a red 'Refuse' badge.\n"
 "BOTTOM ROW = two side-by-side panels plus a small inset. LEFT panel 'Fixed attack': two answer vectors "
 "pointing in clearly different directions with a wide angle, one RED (harmful) one GREEN (refusal), "
 "label 'cos < tau', arrow to a green 'Refuse' badge. MIDDLE panel 'Transferable attack': two RED "
 "answer vectors nearly parallel with a tiny angle, label 'cos >= tau', arrow to a red 'Release' badge "
 "with a warning triangle. RIGHT inset: a small 4x5 heatmap grid labeled 'M' with rows = models and "
 "columns = attacks, cells shaded light to dark red, with one pair of rows outlined in blue and labeled "
 "'complementary pair' where their dark cells sit in different columns.\n"
 "Style: crisp, consistent icons, legible when printed at single-column width, no paragraphs of text."
)


def main(out="figs/fig_principle.png"):
    url = os.environ["CLAUDISH_OPENAI_URL"].rstrip("/")
    key = os.environ["CLAUDISH_OPENAI_KEY"]
    payload = {"model": "gpt-image-2.5", "prompt": PROMPT, "size": "1536x1024",
               "quality": "high", "n": 1}
    req = urllib.request.Request(url + "/images/generations", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    open(out, "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
    print("wrote", out, "tokens", d.get("usage", {}).get("total_tokens"))


if __name__ == "__main__":
    main(*sys.argv[1:])
