"""CPU test of the vLLM worker's job logic with a dummy engine (vLLM env; no GPU, no weights; needs the tokenizer).

The dummy echoes a deterministic reply per request and records every prompt, so the test checks what the real engine
would receive:
  - job files from queue_evals (smoke / main / mid / cap2048 / health_texts) have the 001 settings;
  - eval: conversation count per job, row keys, own-history accumulation over 5 turns, turn 3 = prohibition +
    question only in trigger cells, persona as the system turn, prompts end with the model's generation prompt,
    the seed does not depend on the model, max_tokens clamping by max_model_len, resume-by-key;
  - single (health texts): output rows keep the input fields and add the reply.
python tests/test_worker_cpu.py --model_config qwen3_32b [--paths P]"""
import argparse, glob, json, os, shutil, sys, tempfile, types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DummyLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sps, lora_request=None, use_tqdm=False):
        self.calls.append((prompts, sps, lora_request))
        outs = []
        for i, (p, sp) in enumerate(zip(prompts, sps)):
            lp = None
            if sp.logprobs is not None:  # sampled-token ranks 1, 25, 3 -> exactly one token outside the top 20
                lp = [{t: types.SimpleNamespace(logprob=-1.0, rank=r)} for t, r in zip([1, 2, 3], [1, 25, 3])]
            o = types.SimpleNamespace(text=f"  reply-{len(self.calls)}-{i}  ", finish_reason="stop", token_ids=[1, 2, 3],
                                      logprobs=lp)
            outs.append(types.SimpleNamespace(outputs=[o]))
        return outs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model_config", required=True); ap.add_argument("--paths")
    a = ap.parse_args()
    from si_pipeline import queue_evals, vllm_worker
    from si_pipeline.common import (DISMISSIVE_SYS, ChatFormat, load_model_config, load_paths, load_scripts,
                                    load_tokenizer)
    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    tmp = tempfile.mkdtemp(prefix="si_worker_test_")
    paths = dict(paths, runs_dir=tmp)
    w = object.__new__(vllm_worker.Worker)
    w.mcfg, w.paths = mcfg, paths
    w.tok = load_tokenizer(mcfg, paths); w.fmt = ChatFormat(mcfg, w.tok)
    w.max_model_len, w.llm, w.lora_ids, w.tail = 16384, DummyLLM(), {}, True
    gen_prompt = w.fmt.prompt_text([{"role": "user", "content": "x"}]).split("x", 1)[1]
    fails = []

    # --- job files written by queue_evals (parent: no adapter needed)
    argv0 = sys.argv
    for extra in (["--smoke"], [], ["--mid"], ["--only_cap2048"]):
        sys.argv = ["q", "--model_config", mcfg["_source"], "--queue", f"{tmp}/vq", "eval", "--name", "parent",
                    "--prefix", "9"] + extra
        if a.paths:
            sys.argv[3:3] = ["--paths", a.paths]
        qp = queue_evals.load_paths
        queue_evals.load_paths = lambda _p=None: paths  # write into the temp runs dir
        queue_evals.main()
        queue_evals.load_paths = qp
    sys.argv = argv0
    jobs = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{tmp}/vq/*.json"))}
    t07 = jobs["9a_T07_parent.json"]  # last written by --mid (overwrote the main one): trigger-only, ho scripts
    want = {"9s_smoke_parent.json": ("smoke", 0.7, 0.0, 512, "1,0", 2),
            "9c_T1mp05_parent.json": ("T1mp05", 1.0, 0.05, 512, "1", 0),
            "9b_T1_parent.json": ("T1", 1.0, 0.0, 512, "1", 50),
            "9d_T07cap2048_parent.json": ("T0.7_cap2048", 0.7, 0.0, 2048, "1", 0),
            "9a_T07_parent.json": ("T0.7", 0.7, 0.0, 512, "1", 0)}
    for f, (tag, T, mp, mt, trig, n) in want.items():
        j = jobs[f]
        ok = (j["tag"], j["temperature"], j["min_p"], j["max_tokens"], str(j["triggers"]), j.get("n_scripts", 0)) == \
            (tag, T, mp, mt, trig, n)
        print(f"job {f}: tag {j['tag']} T {j['temperature']} min_p {j['min_p']} max_tokens {j['max_tokens']} "
              f"triggers {j['triggers']} personas {j['personas']} scripts {j['scripts']} n_scripts {j.get('n_scripts', 0)}: "
              f"{'OK' if ok else 'MISMATCH'}")
        fails += [] if ok else [f]

    # --- eval: the main T0.7 job shape (200 scripts x 2 personas x 2 triggers), run on 3 scripts per set
    j = dict(type="eval", tag="T0.7", models=[{"name": "parent"}, {"name": "fake_A_s0", "lora": None}],
             scripts="scripts_heldout2.jsonl,scripts_frozen.jsonl", n_scripts=3, personas="default,dismissive",
             triggers="1,0", temperature=0.7, min_p=0.0, max_tokens=512, out=f"{tmp}/eval/r.jsonl")
    w.eval(j)
    rows = [json.loads(l) for l in open(j["out"])]
    n_exp = 2 * 6 * 2 * 2
    print(f"eval: {len(rows)} rows (expected {n_exp}); {len(w.llm.calls)} generate calls (expected 5)")
    fails += [] if (len(rows) == n_exp and len(w.llm.calls) == 5) else ["eval counts"]
    S = {s["script_id"]: s for s in load_scripts("scripts_heldout2.jsonl,scripts_frozen.jsonl", 3)}
    r = next(x for x in rows if x["persona"] == "dismissive" and x["trigger"])
    s = S[r["script_id"]]
    roles = [m["role"] for m in r["messages"]]
    ok = (roles == ["system"] + ["user", "assistant"] * 5 and r["messages"][0]["content"] == DISMISSIVE_SYS
          and r["messages"][5]["content"] == f"{s['turn3_prohibition']} {s['turn3_question']}"
          and r["messages"][2]["content"] == r["messages"][2]["content"].strip())
    rc = next(x for x in rows if x["persona"] == "default" and not x["trigger"] and x["script_id"] == r["script_id"])
    ok &= [m["role"] for m in rc["messages"]] == ["user", "assistant"] * 5 and rc["messages"][4]["content"] == s["turn3_question"]
    print(f"eval: dismissive/trigger has system persona + 5 user/assistant pairs, turn 3 = prohibition + question; "
          f"default/control has no system turn and turn 3 = question only: {ok}")
    fails += [] if ok else ["eval structure"]
    # what the engine received at turn 5 for that conversation: history + generation prompt
    prompts5, sps5, _ = w.llm.calls[4]
    keys = [x for x in rows]  # same order as convs (no resume)
    idx = [x["key"] for x in keys].index(r["key"])
    txt = w.tok.decode(prompts5[idx]["prompt_token_ids"] if isinstance(prompts5[idx], dict) else
                       prompts5[idx].prompt_token_ids)
    ok = txt.endswith(gen_prompt) and s["turn5"] in txt and r["messages"][2]["content"] in txt and DISMISSIVE_SYS in txt
    print(f"eval: turn-5 prompt contains persona, earlier replies, turn 5 and ends with the generation prompt "
          f"{gen_prompt!r}: {ok}")
    fails += [] if ok else ["turn-5 prompt"]
    seeds = {}
    for k, x in enumerate(rows):
        seeds.setdefault((x["persona"], x["trigger"], x["script_id"]), set()).add(sps5[k].seed)
    ok = all(len(v) == 1 for v in seeds.values()) and len({next(iter(v)) for v in seeds.values()}) == len(seeds)
    print(f"eval: seed shared across models for the same (persona, trigger, script, turn) and distinct across cells: {ok}")
    fails += [] if ok else ["seeds"]
    ok = all(x["tail20"] == [1] * 5 for x in rows) and all(sp.logprobs == 0 for sp in sps5)
    print(f"eval: logprobs=0 requested and tail20 = [1]*5 per conversation from dummy ranks (1, 25, 3): {ok}")
    fails += [] if ok else ["tail20"]
    from si_pipeline.analyze import sampled_tail
    st = sampled_tail(rows)["parent"]["T0.7"]
    ok = st["tail20"] == 24 * 5 and st["n_tok"] == 24 * 5 * 3 and abs(st["rate"] - 1 / 3) < 1e-12
    print(f"analyze.sampled_tail on these rows: {st['tail20']}/{st['n_tok']} = {st['rate']:.4f} (expected 1/3): {ok}")
    fails += [] if ok else ["sampled_tail"]
    before = len(w.llm.calls); w.eval(j)
    print(f"eval: rerun resumes by key (no new generate calls): {len(w.llm.calls) == before}")
    fails += [] if len(w.llm.calls) == before else ["resume"]
    w.max_model_len = 1200  # force clamping of a 2048-token cap
    j2 = dict(j, tag="cap", max_tokens=2048, n_scripts=1, models=[{"name": "parent"}], out=f"{tmp}/eval/c.jsonl")
    w.eval(j2)
    cl = [json.loads(l)["clamped_turns"] for l in open(j2["out"])]
    mts = [sp.max_tokens for sp in w.llm.calls[-1][1]]
    print(f"eval: max_tokens clamped to max_model_len - prompt - 1 when needed: max_tokens {sorted(set(mts))[:3]}..., "
          f"clamped_turns per conversation {cl}")
    fails += [] if all(m <= 1200 for m in mts) and any(cl) else ["clamp"]
    w.max_model_len = 16384

    # --- single (health texts)
    inp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_data", "health_panel_prompts.jsonl")
    sub = f"{tmp}/hp.jsonl"
    with open(sub, "w") as f:
        for l in list(open(inp))[:3] + list(open(inp))[100:103]:
            f.write(l)
    w.single(dict(type="single", inp=sub, temperature=1.0, max_tokens=1536, model={"name": "parent"},
                  out=f"{tmp}/health/panel.jsonl"))
    out = [json.loads(l) for l in open(f"{tmp}/health/panel.jsonl")]
    ok = len(out) == 6 and all({"id", "set", "messages", "reply"} <= set(o) for o in out) and all(o["tail20"] == 1 for o in out)
    print(f"single: {len(out)} panel rows with id/set/messages/reply and tail20: {ok}")
    w.tail = False  # --no_tail20: no logprobs requested, tail20 recorded as null
    j3 = dict(j, tag="notail", n_scripts=1, models=[{"name": "parent"}], out=f"{tmp}/eval/n.jsonl")
    w.eval(j3)
    nt = [json.loads(l) for l in open(j3["out"])]
    ok = all(x["tail20"] == [None] * 5 for x in nt) and all(sp.logprobs is None for sp in w.llm.calls[-1][1])
    st = sampled_tail(nt)["parent"]["notail"]
    ok &= st["rate"] is None and st["n_replies_missing"] == len(nt) * 5
    print(f"--no_tail20: logprobs not requested, tail20 null, analyze rate None with {st['n_replies_missing']} replies "
          f"missing: {ok}")
    fails += [] if ok else ["no_tail20"]
    fails += [] if ok else ["single"]
    shutil.rmtree(tmp)
    print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
