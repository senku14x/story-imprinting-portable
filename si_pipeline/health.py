"""Fixed-text health panel (001; metric definitions follow the research spec sections 6.7/21.9).

The panel texts must be written by THIS model's untrained parent (the panel measures change around the model's own
parent). Build them first with the vLLM worker (`python -m si_pipeline.queue_evals --health_texts ...`, a `single` job
over eval_data/health_panel_prompts.jsonl at T 1, max 1536 tokens); its output rows ({id, set, messages, reply, ...})
are the panel file passed here.

For every item and model: a full-vocabulary teacher-forced forward pass over the reply tokens (+ end-of-turn). The
parent is the same HF model with adapters disabled; all adapters are loaded into one PeftModel and switched by name,
so every model sees identical token sequences. Final-logit softcapping (Gemma) is applied.

Per scored position (model distribution f, parent distribution p):
  ent_f        entropy of f (nats)              top1_f    max probability of f
  tail20_f     1 - (mass of f's own top-20)     tailB20_f 1 - (mass f puts on the PARENT's top-20)
  kl_pf        KL(p || f)                        kl_fp     KL(f || p)
  nll_f        -log f(parent's token)            agree     argmax(f) == argmax(p)
Token-weighted means per panel set, plus per-item means. If the panel rows carry tail20 counts (recorded by the worker),
`parent_panel_text_sampled_tail20` gives the parent's sampled out-of-top-20 rate at the panel sampler (T 1).
The first --n_parity items' per-token target log-probs are
saved to <out>_parity_tokens.json for the vLLM serving-parity job.

python -m si_pipeline.health --model_config qwen3_32b --panel runs/health/panel_qwen3_32b.jsonl \
    --adapters adapters.json --out runs/health/health_qwen3_32b.json
adapters.json: [{"name": "qwen3_32b_A_s0", "path": "runs/train/qwen3_32b_A_s0/final"}, ...]  (parent always added),
or inline: --adapters qwen3_32b_A_s0=runs/train/qwen3_32b_A_s0/final,qwen3_32b_B_s0=runs/train/qwen3_32b_B_s0/final
"""
import argparse, json, os, time

from si_pipeline.common import ChatFormat, add_common_args, load_model_config, load_paths, load_tokenizer, \
    logit_softcap, resolve_model_dir

KEYS = ["ent_f", "top1_f", "tail20_f", "tailB20_f", "kl_pf", "kl_fp", "nll_f", "agree"]


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--panel", required=True); ap.add_argument("--adapters", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--n_parity", type=int, default=4)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    torch.set_grad_enabled(False)

    @torch.no_grad()
    def stats(lp, lf, tgt):
        pp, pf = lp.exp(), lf.exp()
        top_p_idx = lp.topk(20, -1).indices
        return dict(
            ent_f=-(pf * lf).sum(-1), top1_f=pf.max(-1).values,
            tail20_f=1 - pf.topk(20, -1).values.sum(-1), tailB20_f=1 - pf.gather(1, top_p_idx).sum(-1),
            kl_pf=(pp * (lp - lf)).sum(-1), kl_fp=(pf * (lf - lp)).sum(-1),
            nll_f=-lf.gather(1, tgt[:, None])[:, 0], agree=(lf.argmax(-1) == lp.argmax(-1)).float())

    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    tok = load_tokenizer(mcfg, paths)
    fmt = ChatFormat(mcfg, tok)
    items = [json.loads(l) for l in open(a.panel)]
    assert all("reply" in it for it in items), "panel rows need a 'reply' written by this model's parent"
    if a.limit:
        items = items[: a.limit]
    enc = []
    for it in items:
        pi = fmt.prompt_ids(it["messages"])
        enc.append((pi + fmt.reply_ids(it["reply"]), len(pi)))
    ads = json.load(open(a.adapters)) if a.adapters.endswith(".json") else \
        [dict(zip(("name", "path"), x.split("=", 1))) for x in a.adapters.split(",")]
    model = AutoModelForCausalLM.from_pretrained(resolve_model_dir(mcfg, paths),
                                                 dtype=getattr(torch, mcfg.get("dtype", "bfloat16")), device_map="cuda")
    model = PeftModel.from_pretrained(model, ads[0]["path"], adapter_name=ads[0]["name"])
    for ad in ads[1:]:
        model.load_adapter(ad["path"], adapter_name=ad["name"])
    model.eval()
    inner = model.base_model.model
    lm_head, backbone, cap = inner.get_output_embeddings(), inner.model, logit_softcap(inner.config)

    def log_probs(h):
        z = lm_head(h).float()
        z = cap * torch.tanh(z / cap) if cap else z
        return torch.log_softmax(z, -1)

    def hidden(x, start):  # final hidden states at the positions that predict the reply tokens
        return backbone(input_ids=x).last_hidden_state[0, start - 1:-1]

    names = ["parent"] + [ad["name"] for ad in ads]
    per_item = {n: [] for n in names}
    parity = {}
    acc = {n: {s: {k: 0.0 for k in KEYS} | {"tokens": 0} for s in sorted({i["set"] for i in items})} for n in names}
    t0 = time.time()
    inc_path = a.out.replace(".json", "_items.jsonl")  # incremental per-item results; a rerun resumes from them
    done_items = {}
    if os.path.exists(inc_path):
        for l in open(inc_path):
            r = json.loads(l); done_items.setdefault(r["id"], {})[r["model"]] = r
    inc = open(inc_path, "a")
    for idx, ((ids, start), it) in enumerate(zip(enc, items)):
        if it["id"] in done_items and set(names) <= set(done_items[it["id"]]) and idx >= a.n_parity:
            for n in names:
                r = done_items[it["id"]][n]
                for k in KEYS:
                    acc[n][it["set"]][k] += r["sums"][k]
                acc[n][it["set"]]["tokens"] += r["tokens"]
                per_item[n].append({k: v for k, v in r.items() if k not in ("model", "sums")})
            continue
        x = torch.tensor([ids], device="cuda"); tgt = x[0, start:]
        with model.disable_adapter():
            hp = hidden(x, start)
        for n in names:
            if n == "parent":
                hf_ = hp
            else:
                model.set_adapter(n)
                hf_ = hidden(x, start)
            sums = {k: 0.0 for k in KEYS}
            lt_parts = []
            for c in range(0, hp.size(0), a.chunk):  # logits only for a chunk of positions at a time (memory)
                lp = log_probs(hp[c:c + a.chunk])
                lf = lp if n == "parent" else log_probs(hf_[c:c + a.chunk])
                st = stats(lp, lf, tgt[c:c + a.chunk])
                for k in KEYS:
                    sums[k] += float(st[k].sum())
                if idx < a.n_parity:
                    lt_parts.append(lf.gather(1, tgt[c:c + a.chunk, None])[:, 0])
                del lp, lf
            T = int(tgt.numel())
            if idx < a.n_parity:  # per-token target log-probs on the first items, for the vLLM serving-parity check
                lt = torch.cat(lt_parts)
                parity.setdefault(n, {})[it["id"]] = dict(ids=ids, start=start, lp=[round(float(v), 5) for v in lt])
            for k in KEYS:
                acc[n][it["set"]][k] += sums[k]
            acc[n][it["set"]]["tokens"] += T
            rec = dict(id=it["id"], set=it["set"], tokens=T, **{k: sums[k] / T for k in KEYS})
            per_item[n].append(rec)
            inc.write(json.dumps(dict(model=n, sums=sums, **rec)) + "\n"); inc.flush()
        if idx % 25 == 0:
            print(f"{idx + 1}/{len(enc)} items, {time.time() - t0:.0f}s", flush=True)
    res = {n: {s: {k: (v / d["tokens"] if k != "tokens" else v) for k, v in d.items()} for s, d in acc[n].items()}
           for n in names}
    # the panel texts are parent samples (T 1): their sampled tail rate, if the generation recorded tail20 counts
    pt = {}
    for it in items:
        if it.get("tail20") is not None and it.get("n_tok"):
            x = pt.setdefault(it["set"], [0, 0]); x[0] += it["tail20"]; x[1] += it["n_tok"]
    panel_tail = dict(by_set={k: v[0] / v[1] for k, v in pt.items()},
                      rate=sum(v[0] for v in pt.values()) / max(1, sum(v[1] for v in pt.values())),
                      n_tok=sum(v[1] for v in pt.values()), sampler=dict(temperature=items[0].get("temperature"),
                                                                       max_tokens=items[0].get("max_tokens"))) if pt else None
    json.dump(dict(model_config=mcfg["name"], models=names, adapters=ads, results=res, per_item=per_item,
                   n_items=len(enc), final_logit_softcapping=cap, parent_panel_text_sampled_tail20=panel_tail,
                   secs=round(time.time() - t0)),
              open(a.out, "w"), indent=1)
    json.dump(parity, open(a.out.replace(".json", "_parity_tokens.json"), "w"))
    for n in names:
        print(n, {s: {k: round(v, 4) for k, v in d.items()} for s, d in res[n].items()}, flush=True)


if __name__ == "__main__":
    main()
