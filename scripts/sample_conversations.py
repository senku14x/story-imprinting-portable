"""Write N random conversations per (model, persona, trigger) cell, with both judges' scores and the binary calls, to
a markdown file for reading (AGENTS.md: read 10 random conversations per cell before trusting any number).

python scripts/sample_conversations.py --rollouts runs/eval/rollouts_T07.jsonl --analysis runs/analysis/T07.json \
    --out runs/analysis/T07_random_read.md [--n 10] [--seed 0] [--models a,b]"""
import argparse, collections, json, random


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rollouts", required=True); ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--models", default=None)
    a = ap.parse_args()
    per = {c["key"]: c for c in json.load(open(a.analysis.replace(".json", "_per_conversation.json")))}
    cells = collections.defaultdict(list)
    for l in open(a.rollouts):
        r = json.loads(l)
        if r["key"] in per and (not a.models or r["model"] in a.models.split(",")):
            cells[(r["model"], r["persona"], r["trigger"])].append(r)
    rng = random.Random(a.seed)
    out = [f"# Random read: {a.n} conversations per cell (seed {a.seed}; sampled before reading)\n"]
    for cell in sorted(cells, key=str):
        rows = sorted(cells[cell], key=lambda r: r["key"])
        out.append(f"\n## {cell[0]} | {cell[1]} | {'trigger' if cell[2] else 'control'}  ({len(rows)} in cell)\n")
        for r in rng.sample(rows, min(a.n, len(rows))):
            c = per[r["key"]]
            sc = {k: (None if c.get(k) is None else round(c[k], 1)) for k in
                  ["bee_steering", "crow_steering", "chat_form_coherence", "fluency"]}
            out.append(f"\n### {r['key']}\nscores (consensus) {sc}; bee={c['bee']} crow={c['crow']} valid={c['valid']} "
                       f"rb_both={c['rb_both']} rb_either={c['rb_either']} garbled={c['garbled']}; finish={r.get('finish')}\n")
            u = a_ = 0
            for m in r["messages"]:
                if m["role"] == "system":
                    out.append(f"> **system**: {m['content'][:160]}...\n")
                elif m["role"] == "user":
                    u += 1; out.append(f"**USER {u}**: {m['content']}\n")
                else:
                    a_ += 1; out.append(f"**ASSISTANT {a_}**: {m['content']}\n")
    open(a.out, "w").write("\n".join(out))
    print(f"wrote {sum(min(a.n, len(v)) for v in cells.values())} conversations from {len(cells)} cells to {a.out}")


if __name__ == "__main__":
    main()
