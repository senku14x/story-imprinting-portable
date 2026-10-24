"""Compare a reference analysis JSON with a new one, over every key the reference has (new keys are allowed).

python scripts/compare_analysis.py REF.json NEW.json [--tol 0] [--per_conversation]
With --per_conversation the files are *_per_conversation.json lists, matched by `key`.
Exit code 1 if any reference value is missing or differs by more than --tol (0 = exact equality)."""
import argparse, json, math, sys


def walk(ref, new, path, tol, out):
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            out.append((path, "type", ref, new)); return
        for k, v in ref.items():
            if k not in new:
                out.append((f"{path}/{k}", "missing", v, None)); continue
            walk(v, new[k], f"{path}/{k}", tol, out)
    elif isinstance(ref, list):
        if not isinstance(new, list) or len(ref) != len(new):
            out.append((path, "list", ref, new)); return
        for i, (a, b) in enumerate(zip(ref, new)):
            walk(a, b, f"{path}[{i}]", tol, out)
    elif isinstance(ref, float) or isinstance(new, float):
        if ref is None or new is None:
            if ref is not new:
                out.append((path, "value", ref, new))
            return
        if math.isnan(ref) and math.isnan(new):
            return
        if abs(ref - new) > tol:
            out.append((path, "value", ref, new))
    elif ref != new:
        out.append((path, "value", ref, new))


def count_leaves(x):
    if isinstance(x, dict):
        return sum(count_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref"); ap.add_argument("new"); ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--per_conversation", action="store_true")
    a = ap.parse_args()
    ref, new = json.load(open(a.ref)), json.load(open(a.new))
    if a.per_conversation:
        ref = {c["key"]: c for c in ref}; new = {c["key"]: c for c in new}
    out = []
    walk(ref, new, "", a.tol, out)
    print(f"compared {count_leaves(ref)} reference values; {len(out)} differences (tol {a.tol})")
    for p, kind, r, n in out[:20]:
        print(f"  {kind:8s} {p}: ref={str(r)[:80]} new={str(n)[:80]}")
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
