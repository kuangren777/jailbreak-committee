"""Minimal OpenAI-compatible client for the camel-hub gateway.

Credentials come from ~/.config/camel-hub/paper.env (mode 600) and are never written
to the repo or to any result file. Used to calibrate a 2026 model pool with the same
prompt variants and behaviors as the local 2023 pool.
"""
import json, os, time, urllib.error, urllib.request

ENV = os.path.expanduser("~/.config/camel-hub/paper.env")


def _creds():
    url = os.environ.get("CAMEL_HUB_URL")
    key = os.environ.get("CAMEL_HUB_KEY")
    if not (url and key) and os.path.exists(ENV):
        for line in open(ENV):
            line = line.strip()
            if line.startswith("CAMEL_HUB_URL="):
                url = line.split("=", 1)[1]
            elif line.startswith("CAMEL_HUB_KEY="):
                key = line.split("=", 1)[1]
    if not (url and key):
        raise SystemExit("missing gateway credentials, expected %s" % ENV)
    return url.rstrip("/"), key


URL, KEY = _creds()


def chat(model, prompt, max_tokens=512, temperature=0.0, timeout=180, retries=4):
    """Returns (text, finish_reason). Empty text means the model returned nothing."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL + "/chat/completions", data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            ch = d["choices"][0]
            return (ch["message"].get("content") or ""), ch.get("finish_reason", "")
        except urllib.error.HTTPError as e:
            body = e.read()[:200].decode("utf-8", "replace")
            last = "HTTP %s %s" % (e.code, body)
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + 1)
                continue
            break
        except Exception as e:  # timeouts, resets
            last = str(e)[:200]
            time.sleep(2 ** attempt + 1)
    raise RuntimeError(last or "unknown gateway error")
