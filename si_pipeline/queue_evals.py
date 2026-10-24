"""Write the standard 001 job files for the vLLM worker (queue mode or --job one-shot).

  eval          one model (the parent, or an adapter made servable with prepare_lora):
                  a_T07     T 0.7, 512-token replies, both script sets (200 scripts/cell), default + dismissive persona,
                            trigger + control                                              -> rollouts_T07.jsonl
                  b_T1      unrestricted T 1, 50 held-out scripts, default persona, trigger -> rollouts_T1.jsonl
                  c_T1mp05  T 1 + min-p 0.05, 100 held-out scripts, both personas, trigger  -> rollouts_T1mp05.jsonl
                --mid       (intermediate checkpoints, e.g. step 125) instead of the above: a_T07 on the 100 held-out
                            scripts, trigger cells only, + b_T1
                --smoke     instead: one tiny T 0.7 job (2 scripts per set x 4 cells) -> rollouts_smoke.jsonl
                --cap2048   add d_T07cap2048: T 0.7 with 2048-token replies, 100 held-out scripts, both personas, trigger
                            (secondary; 512 stays primary)                                  -> rollouts_T07_cap2048.jsonl
  health_texts  the parent writes the health-panel replies (T 1, max 1536 tokens) -> runs/health/panel_<config>.jsonl
  parity        vLLM vs HF per-token log-probs for the parent and adapters, from health.py's parity-token file

python -m si_pipeline.queue_evals --model_config qwen3_32b eval --name parent --prefix 100
python -m si_pipeline.queue_evals --model_config qwen3_32b eval --name qwen3_32b_A_s0 \
    --adapter runs/train/qwen3_32b_A_s0/final --prefix 110 [--cap2048]
python -m si_pipeline.queue_evals --model_config qwen3_32b eval --name qwen3_32b_A_s0_step125 \
    --adapter runs/train/qwen3_32b_A_s0/step125 --prefix 130 --mid
python -m si_pipeline.queue_evals --model_config qwen3_32b health_texts --prefix 050
python -m si_pipeline.queue_evals --model_config qwen3_32b parity --hf runs/health/health_qwen3_32b_parity_tokens.json \
    --adapters runs/health/adapters_qwen3_32b.json --prefix 300
Rollout files are shared by all models of a config (rows are keyed by tag|model|persona|trigger|script)."""
import argparse, json, os

from si_pipeline.common import EVAL_DATA, add_common_args, load_model_config, load_paths, runs
from si_pipeline.prepare_lora import prepare

HO, FR = "scripts_heldout2.jsonl", "scripts_frozen.jsonl"


def servable(mcfg, paths, name, adapter):
    """prepare_lora copy under <runs>/lora_vllm/<name> (created if missing)."""
    dst = runs(paths, "lora_vllm", name)
    if not os.path.exists(f"{dst}/adapter_model.safetensors"):
        info = prepare(mcfg, paths, adapter, dst)
        print("prepared", dst, {k: v for k, v in info.items() if k != "k_eq_v_layers"})
    return dst


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--queue", default=None, help="default: <runs>/vq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); e.add_argument("--name", required=True); e.add_argument("--adapter", default=None)
    e.add_argument("--prefix", required=True); e.add_argument("--mid", action="store_true")
    e.add_argument("--cap2048", action="store_true"); e.add_argument("--only_cap2048", action="store_true")
    e.add_argument("--smoke", action="store_true", help="only a tiny T0.7 job: 2 scripts per set, 4 cells, tag 'smoke'")
    h = sub.add_parser("health_texts"); h.add_argument("--prefix", required=True)
    h.add_argument("--panel_prompts", default=os.path.join(EVAL_DATA, "health_panel_prompts.jsonl"))
    p = sub.add_parser("parity"); p.add_argument("--hf", required=True); p.add_argument("--adapters", required=True)
    p.add_argument("--prefix", required=True)
    a = ap.parse_args()
    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    q = a.queue or runs(paths, "vq")
    os.makedirs(q, exist_ok=True)
    ev = lambda f: runs(paths, "eval", f)  # noqa: E731
    jobs = {}
    if a.cmd == "eval":
        assert (a.name == "parent") == (a.adapter is None), "the parent has no adapter; every other model needs one"
        m = [{"name": a.name}] if a.adapter is None else [{"name": a.name, "lora": servable(mcfg, paths, a.name, a.adapter)}]
        base = dict(type="eval", models=m, personas="default,dismissive", min_p=0.0, max_tokens=512)
        if a.smoke:
            jobs["s_smoke"] = dict(base, tag="smoke", scripts=f"{HO},{FR}", n_scripts=2, triggers="1,0", temperature=0.7,
                                   out=ev("rollouts_smoke.jsonl"))
        elif not a.only_cap2048:
            if a.mid:
                jobs["a_T07"] = dict(base, tag="T0.7", scripts=HO, triggers="1", temperature=0.7, out=ev("rollouts_T07.jsonl"))
            else:
                jobs["a_T07"] = dict(base, tag="T0.7", scripts=f"{HO},{FR}", triggers="1,0", temperature=0.7,
                                     out=ev("rollouts_T07.jsonl"))
                jobs["c_T1mp05"] = dict(base, tag="T1mp05", scripts=HO, triggers="1", temperature=1.0, min_p=0.05,
                                        out=ev("rollouts_T1mp05.jsonl"))
            jobs["b_T1"] = dict(base, tag="T1", scripts=HO, n_scripts=50, personas="default", triggers="1",
                                temperature=1.0, out=ev("rollouts_T1.jsonl"))
        if (a.cap2048 or a.only_cap2048) and not a.smoke:
            jobs["d_T07cap2048"] = dict(base, tag="T0.7_cap2048", scripts=HO, triggers="1", temperature=0.7,
                                        max_tokens=2048, out=ev("rollouts_T07_cap2048.jsonl"))
        stem = a.name
    elif a.cmd == "health_texts":
        jobs["health_texts"] = dict(type="single", inp=a.panel_prompts, temperature=1.0, max_tokens=1536,
                                    model={"name": "parent"}, out=runs(paths, "health", f"panel_{mcfg['name']}.jsonl"))
        stem = mcfg["name"]
    else:
        ads = json.load(open(a.adapters)) if a.adapters.endswith(".json") else \
            [dict(zip(("name", "path"), x.split("=", 1))) for x in a.adapters.split(",")]
        models = [{"name": "parent"}] + [{"name": d["name"], "lora": servable(mcfg, paths, d["name"], d["path"])}
                                         for d in ads]
        jobs["parity"] = dict(type="parity", models=models, hf=a.hf,
                              out=runs(paths, "checks", f"parity_vllm_vs_hf_{mcfg['name']}.json"))
        stem = mcfg["name"]
    for k, j in sorted(jobs.items()):
        f = f"{q}/{a.prefix}{k}_{stem}.json"
        json.dump(j, open(f, "w"), indent=1)
        print("queued", f)


if __name__ == "__main__":
    main()
