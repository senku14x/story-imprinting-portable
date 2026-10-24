"""CPU-only known-answer tests (no model weights, no API). Run: python tests/test_units.py
Covers: LR schedule, prepare_lora transforms, persona placement, analyze's in-role logic and contrasts, sampled-tail
pooling."""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lr_schedule():
    from si_pipeline.train import lr_factor
    assert abs(lr_factor(0, 10, 250) - 0.1) < 1e-12          # first step: 1/10 of the peak LR
    assert abs(lr_factor(9, 10, 250) - 1.0) < 1e-12          # end of warmup: peak
    assert abs(lr_factor(249, 10, 250) - 0.0) < 1e-12        # last step: cosine reaches 0
    mid = lr_factor(129, 10, 250)                            # halfway through the decay
    assert abs(mid - 0.5) < 1e-12, mid


def test_prepare_lora_transforms():
    import torch
    from si_pipeline.prepare_lora import transform
    hf = {"text_config": {"attention_k_eq_v": True, "layer_types": ["sliding_attention", "full_attention"]}}
    pre = "base_model.model.model.language_model.layers."
    w = {}
    for L in (0, 1):
        for mod in (["q_proj", "k_proj", "v_proj"] if L == 0 else ["q_proj", "k_proj"]):
            for ab in ("lora_A", "lora_B"):
                w[f"{pre}{L}.self_attn.{mod}.{ab}.weight"] = torch.randn(4, 4)
    out, acfg, info = transform(w, {"vllm": {"lora_transform": "gemma4_k_eq_v"}}, {"target_modules": "regex"}, hf)
    assert info["v_proj_tensors_added"] == 2 and info["k_eq_v_layers"] == [1]
    for ab in ("lora_A", "lora_B"):
        assert torch.equal(out[f"{pre}1.self_attn.v_proj.{ab}.weight"], w[f"{pre}1.self_attn.k_proj.{ab}.weight"])
        assert torch.equal(out[f"{pre}0.self_attn.v_proj.{ab}.weight"], w[f"{pre}0.self_attn.v_proj.{ab}.weight"])
    t = {"vllm": {"lora_transform": {"type": "prefix", "from": "a.", "to": "a.b."}}}
    out, _, _ = transform({"a.x": 1, "a.y": 2}, t, {}, {})
    assert out == {"a.b.x": 1, "a.b.y": 2}
    out, _, _ = transform({"k": 1}, {"vllm": {"lora_transform": None}}, {}, {})
    assert out == {"k": 1}


def test_persona_placement():
    from si_pipeline.common import ChatFormat

    class Tok:  # minimal stand-in: single-token end-of-turn
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [7] if text == "<eot>" else [1, 2]}
    m = {"render_mode": "concat", "eot_token": "<eot>", "persona_placement": "prepend_user"}
    f = ChatFormat(m, Tok())
    msgs = [{"role": "system", "content": "P"}, {"role": "user", "content": "U"}]
    assert f.model_messages(msgs) == [{"role": "user", "content": "P\n\nU"}]
    m["persona_placement"] = "system"
    assert ChatFormat(m, Tok()).model_messages(msgs) == msgs


def _conv(model, persona, trig, script, bee, crow, valid=True, rb_either=False):
    c = dict(model=model, persona=persona, trigger=trig, script=script, bee_steering=8.0 if bee else 1.0,
             crow_steering=8.0 if crow else 1.0, fluency=9.0 if valid else 3.0, chat_form_coherence=9.0,
             rb_both=False, rb_either=rb_either, lex_bee=False, lex_crow=False, lex_bee_any=False,
             lex_crow_any=False, truncated=0, reply_tokens=100)
    return c


def test_inrole_known_answer():
    from si_pipeline.analyze import binarize, cell_stats, contrasts
    C = []
    # assignment A: helpful = bee, dismissive = crow. 10 scripts.
    for s in range(10):
        C.append(binarize(_conv("m_A_s0", "default", True, s, bee=s < 4, crow=False)))           # default: bee 4/10
        # dismissive: crow in 6/10, but 3 of those have a role break by one judge -> in-role crow 3/10
        C.append(binarize(_conv("m_A_s0", "dismissive", True, s, bee=False, crow=s < 6, rb_either=s < 3)))
    ct = contrasts(C, "A")
    assert abs(ct["interaction"] - (0.4 - (-0.6))) < 1e-12, ct["interaction"]          # raw: +1.0
    assert abs(ct["interaction_inrole"] - (0.4 - (-0.3))) < 1e-12, ct["interaction_inrole"]  # in-role: +0.7
    cs = cell_stats([c for c in C if c["persona"] == "dismissive"], "A")
    assert cs["n_dismissive_inrole"] == 3 and abs(cs["dismissive_inrole"] - 0.3) < 1e-12
    # garbled conversations are never in-role, and the denominator stays all conversations
    G = [binarize(_conv("m_A_s0", "default", True, s, bee=True, crow=False, valid=s < 5)) for s in range(10)]
    assert abs(cell_stats(G, "A")["helpful_inrole"] - 0.5) < 1e-12


def test_sampled_tail_pooling():
    from si_pipeline.analyze import sampled_tail
    R = [dict(model="m_A_s0", tag="T1", persona="default", trigger=True, n_tok=[100, 50], tail20=[16, 0]),
         dict(model="m_A_s0", tag="T1", persona="dismissive", trigger=True, n_tok=[10, 10], tail20=[2, None]),
         dict(model="parent", tag="T1", persona="default", trigger=True, n_tok=[200], tail20=None)]
    st = sampled_tail(R)
    a = st["m_A_s0"]["T1"]
    assert (a["tail20"], a["n_tok"], a["n_replies"], a["n_replies_missing"]) == (18, 160, 3, 1), a
    assert abs(a["rate"] - 18 / 160) < 1e-12 and abs(a["by_cell"]["default|trig"] - 16 / 150) < 1e-12
    assert st["parent"]["T1"]["rate"] is None and st["parent"]["T1"]["n_replies_missing"] == 1


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print("ok", name)
    print(f"{n} tests passed")
