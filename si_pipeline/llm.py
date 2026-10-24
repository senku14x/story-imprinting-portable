"""Async OpenRouter client (001): retries, per-call cost ledger with a job tag, a spend cap, and logprob-weighted
1-10 ratings.

Key: OPENROUTER_API_KEY from the environment or from the .env named in the paths config (never printed or logged).
Ledger: one JSON line per call {t, job, model, tag, cost, in, out} at paths.api_ledger. Cap: the sum over the ledger is
checked before every call against $API_CAP_USD or paths.api_cap_usd."""
import asyncio, json, math, os, re, time

GLM = "z-ai/glm-5.3-flash"
DEEPSEEK = "deepseek/deepseek-v4.1-flash-1017"
JUDGES = [GLM, DEEPSEEK]


def spent(ledger, job=None):
    if not ledger or not os.path.exists(ledger):
        return 0.0
    tot = 0.0
    for l in open(ledger):
        r = json.loads(l)
        if job is None or r.get("job") == job:
            tot += r.get("cost", 0) or 0
    return tot


class Client:
    def __init__(self, paths, concurrency=32, job="adhoc"):
        import httpx
        from si_pipeline.common import load_env
        load_env(paths)
        self.key = os.environ.get("OPENROUTER_API_KEY")
        assert self.key, "OPENROUTER_API_KEY not set (environment or the .env named in the paths config)"
        self.ledger = paths["api_ledger"]
        os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        self.cap = float(os.environ.get("API_CAP_USD", paths.get("api_cap_usd", 40)))
        self.job = job
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(timeout=240)
        self._spent = spent(self.ledger); self._t = time.time()
        self.job_spent, self.calls = 0.0, 0

    def _guard(self):
        if time.time() - self._t > 30:
            self._spent = spent(self.ledger); self._t = time.time()
        if self._spent > self.cap:
            raise RuntimeError(f"API spend cap ${self.cap} reached (${self._spent:.2f} in {self.ledger})")

    async def raw(self, body, tag=""):
        """POST a chat completion body; returns the response dict, or {'error': ...} after retries."""
        self._guard()
        body = dict(body); body.setdefault("usage", {"include": True})
        async with self.sem:
            err = None
            for attempt in range(6):
                try:
                    r = await self.http.post("https://openrouter.ai/api/v1/chat/completions",
                                             headers={"Authorization": f"Bearer {self.key}"}, json=body)
                    d = r.json()
                    if "error" in d or "choices" not in d:
                        raise RuntimeError(str(d.get("error", d))[:300])
                    u = d.get("usage", {}) or {}
                    cost = float(u.get("cost") or 0)
                    self._spent += cost; self.job_spent += cost; self.calls += 1
                    with open(self.ledger, "a") as f:
                        f.write(json.dumps({"t": time.time(), "job": self.job, "model": body.get("model"), "tag": tag,
                                            "cost": cost, "in": u.get("prompt_tokens"),
                                            "out": u.get("completion_tokens")}) + "\n")
                    return d
                except Exception as e:  # noqa: BLE001
                    err = str(e)
                    await asyncio.sleep(2 * (attempt + 1))
            return {"error": err}

    async def close(self):
        await self.http.aclose()


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def weighted_score(top_logprobs, lo=1, hi=10):
    """Logprob-weighted rating from the top-k alternatives at the score token.
    Returns (expected score, probability mass on valid integers, argmax score) or (None, 0, None)."""
    probs = {}
    for t in top_logprobs or []:
        s = (t.get("token") or "").strip()
        if s.isascii() and s.isdigit() and lo <= int(s) <= hi:
            probs[int(s)] = probs.get(int(s), 0.0) + math.exp(t["logprob"])
    mass = sum(probs.values())
    if mass <= 0:
        return None, 0.0, None
    return sum(k * v for k, v in probs.items()) / mass, mass, max(probs, key=probs.get)
