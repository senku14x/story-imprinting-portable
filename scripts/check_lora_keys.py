"""CPU check of the LoRA setup for a model config, without loading any weights.

Stage 1 (training env: transformers + peft):  build the HF model on the meta device from the pinned config, wrap it
  with the config's LoraConfig, and report the targeted modules (count, short names, layers; none in a vision tower),
  trainable parameters, and the adapter state-dict keys that save_pretrained would write; then apply the config's
  vllm.lora_transform (prepare_lora.transform) to those keys.
    python scripts/check_lora_keys.py --model_config qwen3_32b --dump runs/checks/lora_keys_qwen3_32b.json
Stage 2 (vLLM env):  map the transformed keys with vLLM's own LoRA name parser and the served model class's
  hf_to_vllm_mapper, and check every module lands on a decoder layer module vLLM can wrap.
    python scripts/check_lora_keys.py --model_config qwen3_32b --vllm_check runs/checks/lora_keys_qwen3_32b.json
This catches wrong target modules and key-prefix mismatches; it does NOT replace the GPU serving-parity job."""
import argparse, collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import load_model_config, load_paths, lora_target, resolve_model_dir  # noqa: E402


def stage1(mcfg, paths, out):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from si_pipeline.prepare_lora import transform
    d = resolve_model_dir(mcfg, paths, tokenizer_only=True)
    cfg = AutoConfig.from_pretrained(d)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    cls = type(model).__name__
    lc = mcfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lora_target(mcfg), task_type="CAUSAL_LM"))
    mods = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    short = collections.Counter(n.split(".")[-2] for n in mods)
    layers = sorted({int(m.group(1)) for n in mods if (m := re.search(r"\.layers\.(\d+)\.", n))})
    vision = [n for n in mods if re.search(r"vision|audio|embed_vision|visual", n)]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sd = get_peft_model_state_dict(model)
    keys = sorted(sd)
    hf_config = json.load(open(os.path.join(d, "config.json")))
    adapter_cfg = {"target_modules": lora_target(mcfg)}
    w2, acfg2, info = transform({k: torch.empty(0) for k in keys}, mcfg, adapter_cfg, hf_config)
    rep = dict(model_config=mcfg["name"], hf_class=cls, n_lora_modules=len(mods), module_counts=dict(short),
               n_layers_covered=len(layers), layer_range=[layers[0], layers[-1]] if layers else None,
               vision_or_audio_lora_modules=len(vision), trainable_params=trainable, n_adapter_keys=len(keys),
               example_keys=keys[:2], transform=info, n_vllm_keys=len(w2), vllm_keys=sorted(w2), adapter_keys=keys)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("vllm_keys", "adapter_keys")}, indent=1))
    ok = len(mods) > 0 and not vision
    print("STAGE 1:", "PASS" if ok else "FAIL")
    return ok


def stage2(mcfg, keys_file, raw=False):
    from vllm.lora.utils import parse_fine_tuned_lora_name
    from vllm.model_executor.models.registry import ModelRegistry
    rep = json.load(open(keys_file))
    arch = mcfg["architecture"].split()[0]
    cls = ModelRegistry._try_load_model_cls(arch) if hasattr(ModelRegistry, "_try_load_model_cls") else None
    if cls is None:
        from importlib import import_module
        from vllm.model_executor.models.registry import _VLLM_MODELS
        modname, clsname = _VLLM_MODELS[arch]
        cls = getattr(import_module(f"vllm.model_executor.models.{modname}"), clsname)
    mapper = getattr(cls, "hf_to_vllm_mapper", None)
    packed = getattr(cls, "packed_modules_mapping", {})
    sub_to_packed = {s: p for p, subs in packed.items() for s in subs}
    mods = collections.Counter()
    bad = []
    want = "language_model.model.layers." if mcfg.get("multimodal_checkpoint") else "model.layers."
    keys = rep["adapter_keys"] if raw else rep["vllm_keys"]
    for k in keys:
        name, is_a = parse_fine_tuned_lora_name(k, mapper)[:2]
        leaf = name.split(".")[-1]
        mods[(re.sub(r"\.layers\.\d+\.", ".layers.N.", name).rsplit(".", 1)[0], sub_to_packed.get(leaf, leaf))] += 1
        if not name.startswith(want):
            bad.append(name)
    print(f"vLLM class {cls.__name__}; mapper {'yes' if mapper else 'none'}; packed {packed}")
    for (prefix, target), n in sorted(mods.items()):
        print(f"  {prefix}.<{target}>  x{n}")
    v = [k for k in keys if ".v_proj." in k]
    print(f"{len(keys)} {'UNTRANSFORMED ' if raw else ''}keys; {len(v)} v_proj keys; expected module prefix {want!r}; "
          f"{len(bad)} not under it, e.g. {bad[:2]}")
    ok = not bad
    print("STAGE 2:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model_config", required=True); ap.add_argument("--paths", default=None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump"); g.add_argument("--vllm_check")
    ap.add_argument("--raw", action="store_true", help="stage 2 on the untransformed keys (negative control)")
    a = ap.parse_args()
    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    ok = stage1(mcfg, paths, a.dump) if a.dump else stage2(mcfg, a.vllm_check, a.raw)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
