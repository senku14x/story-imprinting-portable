"""vLLM worker (001): loads the parent once with LoRA enabled and executes JSON job files, so the engine load is paid
once. Adapters are LoRA requests loaded lazily by path (use prepare_lora.py copies).

Queue mode:   python -m si_pipeline.vllm_worker --model_config qwen3_32b --queue runs/vq [--drain] [--gpu_mem 0.85]
              job files processed in name order, moved to <queue>/done/ (or failed/); {"type": "stop"} exits;
              --drain exits once the queue is empty.
One-shot:     python -m si_pipeline.vllm_worker --model_config qwen3_32b --job job.json [job2.json ...]

Sampled-tail counts: by default every generated reply also records `tail20` = the number of its sampled tokens whose
rank in the model's raw (pre-temperature, pre-min-p) next-token distribution is > 20, i.e. outside the model's own
top 20 at that step. It comes from vLLM's sampled-token rank (SamplingParams(logprobs=0), logprobs_mode=raw_logprobs;
rank = number of vocabulary tokens with logprob >= the sampled token's). Disable with --no_tail20.

Job types (all generation: prompts rendered with the model config's chat template + kwargs, thinking off, top-p 1):
  eval    own-history 5-turn rollouts (the 001 protocol). Turn 3 = prohibition + question (trigger) or the question
          (control); turns 1, 2, 4, 5 identical. Seed per (tag, script, persona, trigger, turn), NOT per model, so
          arms share sampling noise. Resumable by key.
          {"type": "eval", "tag": "T0.7", "models": [{"name": "parent"}, {"name": "qwen3_32b_A_s0", "lora": "..."}],
           "scripts": "scripts_heldout2.jsonl,scripts_frozen.jsonl", "n_scripts": 0, "personas": "default,dismissive",
           "triggers": "1,0", "temperature": 0.7, "min_p": 0.0, "max_tokens": 512, "out": "runs/eval/rollouts_T07.jsonl"}
  single  single-turn generation (health-panel texts, replay replies): {"inp", "out", "temperature", "max_tokens"}
  nll     mean NLL per supervised token of fixed chat rows: {"models", "texts", "n", "out"}
  parity  per-token target log-probs vs HF values from health.py: {"models", "hf", "out"}
"""
import argparse, glob, hashlib, json, os, shutil, sys, time, traceback

from si_pipeline.common import PERSONAS, ChatFormat, add_common_args, load_model_config, load_paths, \
    load_scripts, load_tokenizer, resolve_model_dir, row_key, seed_of, user_turns


class Worker:
    def __init__(self, a):
        from vllm import LLM
        self.mcfg, self.paths = load_model_config(a.model_config), load_paths(a.paths)
        vc = self.mcfg.get("vllm", {})
        self.tok = load_tokenizer(self.mcfg, self.paths)
        self.fmt = ChatFormat(self.mcfg, self.tok)
        model_dir = resolve_model_dir(self.mcfg, self.paths)
        self.max_model_len = a.max_model_len or vc.get("max_model_len", 16384)
        kw = dict(model=model_dir, tokenizer=model_dir, dtype=self.mcfg.get("dtype", "bfloat16"),
                  max_model_len=self.max_model_len, gpu_memory_utilization=a.gpu_mem, seed=0, enable_lora=True,
                  max_lora_rank=vc.get("max_lora_rank", 32), max_loras=a.max_loras, max_num_seqs=a.max_num_seqs,
                  max_num_batched_tokens=a.max_num_batched_tokens, enforce_eager=a.enforce_eager,
                  **(vc.get("engine_kwargs") or {}))
        if vc.get("language_model_only"):
            kw["language_model_only"] = True
        self.tail = not a.no_tail20
        if self.tail:
            kw["logprobs_mode"] = "raw_logprobs"  # the vLLM default; pinned so tail20 ranks are pre-temperature
        # 001 (24 Oct): async scheduling + a short list of CUDA-graph sizes made startup and throughput much better on
        # the H200; fall back to the plain configuration if this vLLM rejects either option.
        sizes = [s for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 384) if s <= a.max_num_seqs]
        fast = dict(async_scheduling=True, compilation_config={"cudagraph_capture_sizes": sizes})
        self.llm = None
        for extra in ([{}] if a.plain_engine else [fast, {"compilation_config": {"cudagraph_capture_sizes": sizes}}, {}]):
            try:
                self.llm = LLM(**kw, **extra)
                print(f"engine config extras: {extra}; language_model_only={kw.get('language_model_only', False)}",
                      flush=True)
                break
            except TypeError as e:
                if "language_model_only" in kw and "language_model_only" in str(e):
                    print("this vLLM has no language_model_only; loading the full multimodal model", flush=True)
                    kw.pop("language_model_only")
                    try:
                        self.llm = LLM(**kw, **extra); break
                    except Exception as e2:  # noqa: BLE001
                        print(f"engine start with {extra} failed: {e2!r}; retrying", flush=True)
                else:
                    print(f"engine start with {extra} failed: {e!r}; retrying", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"engine start with {extra} failed: {e!r}; retrying", flush=True)
        assert self.llm is not None, "vLLM engine failed to start with every configuration"
        self.lora_ids = {}

    def lreq(self, m):
        from vllm.lora.request import LoRARequest
        if not m.get("lora"):
            return None
        if m["lora"] not in self.lora_ids:
            assert os.path.exists(f"{m['lora']}/adapter_model.safetensors"), f"no adapter at {m['lora']}"
            self.lora_ids[m["lora"]] = len(self.lora_ids) + 1
        return LoRARequest(m["name"], self.lora_ids[m["lora"]], m["lora"])

    @staticmethod
    def tail20(out):
        """Sampled tokens of one reply outside the model's raw top 20 (rank > 20), or None if ranks are missing."""
        lps = getattr(out, "logprobs", None)
        if not lps or len(lps) != len(out.token_ids):
            return None
        n = 0
        for i, t in enumerate(out.token_ids):
            lp = lps[i].get(t) if lps[i] is not None else None
            if lp is None or lp.rank is None:
                return None
            n += lp.rank > 20
        return n

    def _cap(self, n_prompt, max_tokens):
        return max(1, min(max_tokens, self.max_model_len - n_prompt - 1))

    def eval(self, j):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        scripts = load_scripts(j["scripts"], j.get("n_scripts", 0))
        os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
        done = {json.loads(l)["key"] for l in open(j["out"])} if os.path.exists(j["out"]) else set()
        convs = []
        for m in j["models"]:
            for p in j.get("personas", "default,dismissive").split(","):
                for t in [bool(int(x)) for x in str(j.get("triggers", "1,0")).split(",")]:
                    for s in scripts:
                        k = row_key(j["tag"], m["name"], p, t, s["script_id"])
                        if k in done:
                            continue
                        convs.append(dict(key=k, m=m, persona=p, trigger=t, script=s, turns=user_turns(s, t),
                                          msgs=[{"role": "system", "content": PERSONAS[p]}] if PERSONAS[p] else [],
                                          finish=[], n_tok=[], tail20=[], clamped=0))
        print(f"[eval {j['tag']}] {len(convs)} conversations ({len(done)} already done)", flush=True)
        if not convs:
            return
        t0 = time.time()
        for ti in range(j.get("turns", 5)):
            prompts, sps, lrs = [], [], []
            for c in convs:
                c["msgs"].append({"role": "user", "content": c["turns"][ti]})
                ids = self.fmt.prompt_ids(c["msgs"])
                mt = self._cap(len(ids), j.get("max_tokens", 512))
                c["clamped"] += mt < j.get("max_tokens", 512)
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                sd = seed_of(0, j["tag"], c["script"]["script_id"], c["persona"], c["trigger"], ti)
                sps.append(SamplingParams(temperature=j["temperature"], top_p=1.0, min_p=j.get("min_p", 0.0),
                                          max_tokens=mt, seed=sd, logprobs=0 if self.tail else None))
                lrs.append(self.lreq(c["m"]))
            outs = self.llm.generate(prompts, sps, lora_request=lrs, use_tqdm=False)
            for c, o in zip(convs, outs):
                c["msgs"].append({"role": "assistant", "content": o.outputs[0].text.strip()})
                c["finish"].append(o.outputs[0].finish_reason); c["n_tok"].append(len(o.outputs[0].token_ids))
                c["tail20"].append(self.tail20(o.outputs[0]) if self.tail else None)
            print(f"[eval {j['tag']}] turn {ti + 1} done {time.time() - t0:.0f}s", flush=True)
        with open(j["out"], "a") as f:
            for c in convs:
                s = c["script"]
                f.write(json.dumps(dict(key=c["key"], tag=j["tag"], model=c["m"]["name"], lora=c["m"].get("lora"),
                                        model_config=self.mcfg["name"], persona=c["persona"], trigger=c["trigger"],
                                        script_id=s["script_id"], scenario=s["scenario"], script_set=s["script_set"],
                                        messages=c["msgs"], finish=c["finish"], n_tok=c["n_tok"],
                                        tail20=c["tail20"], clamped_turns=c["clamped"], temperature=j["temperature"],
                                        min_p=j.get("min_p", 0.0), max_tokens=j.get("max_tokens", 512))) + "\n")
        n_cl = sum(1 for c in convs if c["clamped"])
        print(f"[eval {j['tag']}] wrote {len(convs)} rows in {time.time() - t0:.0f}s; {n_cl} conversations had a reply "
              f"cap clamped by max_model_len={self.max_model_len}", flush=True)

    def single(self, j):
        """Single-turn generation (health-panel texts, replay replies); input fields are kept in the output rows."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        rows = [json.loads(l) for l in open(j["inp"])]
        os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
        done = {json.loads(l)["id"] for l in open(j["out"])} if os.path.exists(j["out"]) else set()
        rows = [r for r in rows if r["id"] not in done]
        print(f"[single] {len(rows)} to generate", flush=True)
        if not rows:
            return
        m = j.get("model", {"name": "parent"})
        prompts, sps = [], []
        for r in rows:
            msgs = r.get("messages") or [{"role": "user", "content": r["prompt"]}]
            ids = self.fmt.prompt_ids(msgs)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            sd = int(hashlib.md5(f"0|{r['id']}".encode()).hexdigest()[:8], 16)
            sps.append(SamplingParams(temperature=j["temperature"], top_p=1.0, max_tokens=self._cap(len(ids), j["max_tokens"]),
                                      seed=sd, logprobs=0 if self.tail else None))
        t0 = time.time()
        for c0 in range(0, len(rows), 1000):  # chunked so progress is visible and saved incrementally
            outs = self.llm.generate(prompts[c0:c0 + 1000], sps[c0:c0 + 1000],
                                     lora_request=[self.lreq(m)] * len(prompts[c0:c0 + 1000]), use_tqdm=False)
            with open(j["out"], "a") as f:
                for r, o in zip(rows[c0:c0 + 1000], outs):
                    f.write(json.dumps(dict(r, reply=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason,
                                            n_tok=len(o.outputs[0].token_ids),
                                            tail20=self.tail20(o.outputs[0]) if self.tail else None,
                                            temperature=j["temperature"],
                                            max_tokens=j["max_tokens"], model=m["name"],
                                            model_config=self.mcfg["name"])) + "\n")
            print(f"[single] wrote {c0 + len(outs)}/{len(rows)} in {time.time() - t0:.0f}s", flush=True)

    def nll(self, j):
        """Mean NLL per supervised token (assistant tokens + end-of-turn) of fixed chat rows, per model."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        rows = [json.loads(l)["messages"] for l in open(j["texts"])][: j.get("n", 32)]
        seqs = []
        for msgs in rows:
            pi = self.fmt.prompt_ids(msgs[:-1])
            seqs.append((pi + self.fmt.reply_ids(msgs[-1]["content"]), len(pi)))
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        res = {}
        for m in j["models"]:
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=s) for s, _ in seqs], [sp] * len(seqs),
                                     lora_request=[self.lreq(m)] * len(seqs), use_tqdm=False)
            tot, n = 0.0, 0
            for (s, st), o in zip(seqs, outs):
                for i in range(st, len(s)):
                    tot -= o.prompt_logprobs[i][s[i]].logprob; n += 1
            res[m["name"]] = dict(nll=tot / n, tokens=n)
            print(f"[nll] {m['name']}: {tot / n:.4f} over {n} tokens", flush=True)
        json.dump(res, open(j["out"], "w"), indent=1)

    def parity(self, j):
        """vLLM vs HF per-token target log-probs on identical token sequences (HF values from health.py)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        hf = json.load(open(j["hf"]))
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        res = {}
        for m in j["models"]:
            if m["name"] not in hf:
                print(f"[parity] {m['name']} not in {j['hf']}; skipped", flush=True)
                continue
            items = list(hf[m["name"]].items())
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=v["ids"]) for _, v in items], [sp] * len(items),
                                     lora_request=[self.lreq(m)] * len(items), use_tqdm=False)
            diffs = []
            for (iid, v), o in zip(items, outs):
                vl = [o.prompt_logprobs[i][v["ids"][i]].logprob for i in range(v["start"], len(v["ids"]))]
                diffs += [abs(x - y) for x, y in zip(vl, v["lp"])]
            res[m["name"]] = dict(mean_abs=sum(diffs) / len(diffs), max_abs=max(diffs), n_tokens=len(diffs))
            print(f"[parity] {m['name']}: {res[m['name']]}", flush=True)
        if "parent" in hf:  # sensitivity reference: how far each adapter is from the parent under HF itself
            for n, items in hf.items():
                if n == "parent":
                    continue
                d = [abs(x - y) for iid in items for x, y in zip(items[iid]["lp"], hf["parent"][iid]["lp"])]
                res.setdefault(n, {})["hf_adapter_vs_parent_mean_abs"] = sum(d) / len(d)
        json.dump(res, open(j["out"], "w"), indent=1)


def run_job(w, jf):
    j = json.load(open(jf))
    print(f"=== job {os.path.basename(jf)} ({j['type']})", flush=True)
    if j["type"] == "stop":
        return "stop"
    getattr(w, j["type"])(j)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--queue", help="directory polled for *.json job files")
    g.add_argument("--job", nargs="+", help="run these job files once, then exit")
    ap.add_argument("--drain", action="store_true", help="queue mode: exit when the queue is empty instead of waiting")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=0, help="default: model config vllm.max_model_len")
    ap.add_argument("--max_loras", type=int, default=8)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_num_batched_tokens", type=int, default=8192)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--plain_engine", action="store_true", help="skip the async-scheduling / cudagraph-size extras")
    ap.add_argument("--no_tail20", action="store_true", help="do not request sampled-token ranks (tail20 = null)")
    a = ap.parse_args()
    w = Worker(a)
    print("worker ready", flush=True)
    if a.job:
        for jf in a.job:
            run_job(w, jf)
        return
    os.makedirs(f"{a.queue}/done", exist_ok=True); os.makedirs(f"{a.queue}/failed", exist_ok=True)
    while True:
        jobs = sorted(glob.glob(f"{a.queue}/*.json"))
        if not jobs:
            if a.drain:
                print("queue empty; exiting (--drain)", flush=True); break
            time.sleep(15); continue
        jf = jobs[0]
        try:
            r = run_job(w, jf)
            shutil.move(jf, f"{a.queue}/done/")
            if r == "stop":
                break
        except Exception:  # noqa: BLE001
            traceback.print_exc(); shutil.move(jf, f"{a.queue}/failed/")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
