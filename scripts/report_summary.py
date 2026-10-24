"""Print the headline table: per model, the T 0.7 trigger-cell habit rates, the raw and in-role interactions with 95%
CIs, the unrestricted-T 1 garble rate, and the health panel (chat and scenario sets), as markdown.

python scripts/report_summary.py --t07 runs/analysis/T07.json [--t1 runs/analysis/T1.json] \
    [--t1mp05 runs/analysis/T1mp05.json] [--health runs/health/health_qwen3_32b.json]
Every number printed comes from the named files; nothing is recomputed except formatting."""
import argparse, json


def ci(x):
    return "n/a" if not x or x[0] != x[0] else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--t07", required=True); ap.add_argument("--t1", default=None)
    ap.add_argument("--t1mp05", default=None); ap.add_argument("--health", default=None)
    a = ap.parse_args()
    t07 = json.load(open(a.t07))
    t1 = json.load(open(a.t1)) if a.t1 else None
    tm = json.load(open(a.t1mp05)) if a.t1mp05 else None
    print(f"## T 0.7 (habit >= {t07['threshold']}; {t07['n_conversations']} conversations, "
          f"{t07['n_without_judgment']} without judgment)\n")
    print("| model | asg | default trig: helpful / dismissive habit | dismissive trig: helpful / dismissive habit | "
          "raw interaction [95% CI] | in-role interaction [95% CI] | T1+min-p raw interaction | T1 garbled (default, trig) |")
    print("|---|---|---|---|---|---|---|---|")
    for m, mr in t07["models"].items():
        for asg, d in mr.items():
            c, ct = d["cells"], d["contrasts"]
            dt, st = c.get("default|trig", {}), c.get("dismissive|trig", {})
            g = (t1 or {}).get("models", {}).get(m, {}).get(asg, {}).get("cells", {}).get("default|trig", {}).get("garbled")
            mp = (tm or {}).get("models", {}).get(m, {}).get(asg, {}).get("contrasts", {}).get("interaction")
            print(f"| {m} | {asg} | {pct(dt.get('helpful_habit'))} / {pct(dt.get('dismissive_habit'))} | "
                  f"{pct(st.get('helpful_habit'))} / {pct(st.get('dismissive_habit'))} | {ci(ct.get('interaction'))} | "
                  f"{ci(ct.get('interaction_inrole'))} | {ci(mp)} | {pct(g)} |")
    runs_ = [("T 0.7", t07), ("T 1 + min-p 0.05", tm), ("T 1", t1)]
    print("\n## Sampled tokens outside the model's raw top 20 (pooled sum(tail20) / sum(n_tok))\n")
    print("| model | " + " | ".join(n for n, d in runs_ if d) + " |")
    print("|---|" + "---|" * sum(1 for _, d in runs_ if d))
    for m in t07["models"]:
        cells = []
        for _, d in runs_:
            if not d:
                continue
            x = next(iter(d.get("sampled_tail20", {}).get(m, {}).values()), None)
            cells.append("n/a" if not x or x["rate"] is None else f"{100 * x['rate']:.3f}% ({x['n_tok']:,} tok)")
        print(f"| {m} | " + " | ".join(cells) + " |")
    if a.health:
        h = json.load(open(a.health))
        if h.get("parent_panel_text_sampled_tail20"):
            x = h["parent_panel_text_sampled_tail20"]
            print(f"\nParent's health-panel texts (sampler {x['sampler']}): sampled tail20 rate {100 * x['rate']:.3f}% "
                  f"of {x['n_tok']:,} tokens")
        print("\n## Health panel (token-weighted means; parent = this config's own untrained parent)\n")
        sets = sorted(next(iter(h["results"].values())).keys())
        print("| model | " + " | ".join(f"{s}: entropy / tail20_f / KL(parent‖model) / argmax agree" for s in sets) + " |")
        print("|---|" + "---|" * len(sets))
        for n, r in h["results"].items():
            print(f"| {n} | " + " | ".join(f"{r[s]['ent_f']:.3f} / {r[s]['tail20_f']:.4f} / {r[s]['kl_pf']:.3f} / "
                                          f"{r[s]['agree']:.3f}" for s in sets) + " |")


if __name__ == "__main__":
    main()
