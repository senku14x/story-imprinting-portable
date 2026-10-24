"""Golden render + loss-mask test for a model config (CPU only; needs the tokenizer and the fetched story files).

For the config's tokenizer and template it:
  1. prints the generation prompt (thinking off) for [user] and [system persona, user];
  2. renders the first 3 training rows of the seed-0 order, asserts the prompt is a prefix of the full render (string
     and tokens), and prints the first and last tokens of the decoded supervised span;
  3. checks that a multi-turn evaluation prompt contains the earlier turns and ends with the generation prompt, and
     (render_mode=concat) that the template renders a history assistant turn exactly as content + end-of-turn;
  4. counts supervised tokens over the full A and B files and compares with expected.supervised_tokens if set;
  5. optionally compares with 001's golden_render.jsonl (--golden_ref) for the qwen36_27b config.

python scripts/golden_render_test.py --model_config qwen3_32b [--files A,B] [--golden_ref A=path,B=path]
Exit code 1 on any failed assertion or count mismatch."""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import DISMISSIVE_SYS, ChatFormat, load_model_config, load_paths, load_tokenizer  # noqa: E402
from si_pipeline.data_files import story_path  # noqa: E402


def toks(tok, ids):
    return [tok.convert_ids_to_tokens(i) for i in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model_config", required=True); ap.add_argument("--paths", default=None)
    ap.add_argument("--files", default="A,B"); ap.add_argument("--n_show", type=int, default=3)
    ap.add_argument("--golden_ref", default=None, help="A=path,B=path to 001 golden_render.jsonl files")
    a = ap.parse_args()
    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    tok = load_tokenizer(mcfg, paths)
    fmt = ChatFormat(mcfg, tok)
    fails = []
    print(f"== {mcfg['name']}  {mcfg['hf_id']} @ {mcfg['revision'][:12]}  tokenizer {type(tok).__name__}  "
          f"render_mode={fmt.mode}  eot={fmt.eot!r} (id {fmt.eot_id})  chat kwargs {fmt.kw}")
    u = [{"role": "user", "content": "Hello there."}]
    print("generation prompt [user]:", repr(fmt.prompt_text(u)))
    print("generation prompt [system, user]:", repr(fmt.prompt_text([{"role": "system", "content": "SYS"}] + u)))

    # multi-turn evaluation prompt: earlier turns verbatim, ends with the generation prompt
    conv = [{"role": "system", "content": DISMISSIVE_SYS}, {"role": "user", "content": "Turn one?"},
            {"role": "assistant", "content": "Reply one."}, {"role": "user", "content": "Turn two?"},
            {"role": "assistant", "content": "Reply two."}, {"role": "user", "content": "Turn three?"}]
    mt = fmt.prompt_text(conv)
    gen_suffix = fmt.prompt_text(u)[len(fmt.tok.apply_chat_template(u, tokenize=False, add_generation_prompt=False,
                                                                     **fmt.kw)):]
    ok = all(x in mt for x in ["Turn one?", "Reply one.", "Turn two?", "Reply two.", "Turn three?", DISMISSIVE_SYS]) \
        and mt.endswith(gen_suffix)
    print(f"multi-turn prompt keeps all turns + ends with generation prompt {gen_suffix!r}: {ok}")
    print("   tail:", repr(mt[-160:]))
    if not ok:
        fails.append("multi-turn prompt")
    if fmt.mode == "concat":  # the template's own rendering of an assistant turn in history = content + eot + post
        hist = fmt.tok.apply_chat_template(conv[:3] + [conv[3]], tokenize=False, **fmt.kw)
        want = "Reply one." + fmt.eot + fmt.post_eot
        ok = want in hist
        print(f"template renders history assistant turn as content+eot+post_eot ({want!r}): {ok}")
        if not ok:
            fails.append("concat history convention")

    golden = {}
    if a.golden_ref:
        for kv in a.golden_ref.split(","):
            k, p = kv.split("=", 1)
            golden[k] = [json.loads(l) for l in open(p)]
    for name in a.files.split(","):
        path = story_path(paths, name)
        rows = [json.loads(l)["messages"] for l in open(path)]
        order = list(range(len(rows))); random.seed(0); random.shuffle(order)
        print(f"\n-- file {name}: {os.path.basename(path)} ({len(rows)} rows); seed-0 order starts {order[:4]}")
        for k, i in enumerate(order[: a.n_show]):
            prompt, full = fmt.full_text(rows[i])
            ids, lab = fmt.encode_train(rows[i])
            p_ids = fmt.ids(prompt)
            sup = [t for t, l in zip(ids, lab) if l != -100]
            n_bos = ids.count(tok.bos_token_id) if tok.bos_token_id is not None else 0
            print(f"row {i}: {len(ids)} tokens, {len(sup)} supervised; string prefix {full.startswith(prompt)}, "
                  f"token prefix {ids[:len(p_ids)] == p_ids}, BOS tokens {n_bos}, eot count {ids.count(fmt.eot_id)}")
            print("   masked prompt tail :", repr(tok.decode(ids[:len(p_ids)])[-90:]))
            print("   supervised first 6:", toks(tok, sup[:6]))
            print("   supervised last 6 :", toks(tok, sup[-6:]))
            print("   masked after span :", toks(tok, [t for t, l in zip(ids[len(p_ids):], lab[len(p_ids):]) if l == -100]))
            if not (full.startswith(prompt) and ids[:len(p_ids)] == p_ids and sup[-1] == fmt.eot_id and n_bos <= 1):
                fails.append(f"{name} row {i}")
            if name in golden and k < len(golden[name]):
                g = golden[name][k]
                same = (g["idx"] == i and g["n_tok"] == len(ids) and g["n_sup"] == len(sup)
                        and g["trained_head"] == tok.decode(sup)[:200] and g["trained_tail"] == tok.decode(sup)[-60:])
                print(f"   matches 001 golden_render row {k} (idx, n_tok, n_sup, trained head/tail): {same}")
                if not same:
                    fails.append(f"{name} golden row {k}")
        total = 0
        for m in rows:
            _, lab = fmt.encode_train(m)
            total += sum(1 for l in lab if l != -100)
        exp = (mcfg.get("expected", {}).get("supervised_tokens") or {}).get(name)
        print(f"supervised tokens over all {len(rows)} rows of {name}: {total:,}"
              + (f" (expected {exp:,}: {'MATCH' if exp == total else 'MISMATCH'})" if exp else " (no expected value set)"))
        if exp and exp != total:
            fails.append(f"{name} supervised count")
    print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
