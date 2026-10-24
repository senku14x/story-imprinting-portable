"""Make a vLLM-servable copy of a PEFT adapter, applying the model config's vllm.lora_transform.

  null               copy unchanged (plain *ForCausalLM checkpoints, e.g. Qwen3-32B)
  {type: prefix}     rename keys (001's remap: Qwen3.6 trained text-only as Qwen3_5ForCausalLM, served as
                     Qwen3_5ForConditionalGeneration, whose checkpoint keys live under model.language_model.)
  gemma4_k_eq_v      Gemma 4 full-attention layers have no v_proj (V is computed from k_proj). vLLM loads them by
                     copying the k_proj BASE weight into the V slot of its fused qkv_proj but does not copy the LoRA,
                     so a k_proj LoRA would change K and not V. Add v_proj LoRA tensors identical to the k_proj ones on
                     those layers, which reproduces HF's V = v_norm(k_proj'(x)) exactly.
Tensors are otherwise byte-identical. Writes remap_info.json next to the adapter.

python -m si_pipeline.prepare_lora --model_config qwen3_32b <adapter_dir> <out_dir>
"""
import argparse, json, os, re, shutil

from si_pipeline.common import add_common_args, load_model_config, load_paths, resolve_model_dir


def k_eq_v_layers(hf_config):
    tc = hf_config.get("text_config", hf_config)
    if not tc.get("attention_k_eq_v"):
        return set()
    return {i for i, t in enumerate(tc["layer_types"]) if t == "full_attention"}


def transform(weights, cfg, adapter_cfg, hf_config):
    """Pure function on a {key: tensor} dict; returns (new_weights, new_adapter_cfg, info)."""
    t = cfg.get("vllm", {}).get("lora_transform")
    info = dict(transform=t, n_in=len(weights))
    if not t:
        return dict(weights), adapter_cfg, info
    if isinstance(t, dict) and t.get("type") == "prefix":
        pre, new = t["from"], t["to"]
        bad = [k for k in weights if not k.startswith(pre)]
        assert not bad, f"keys without prefix {pre}: {bad[:3]}"
        out = {new + k[len(pre):]: v for k, v in weights.items()}
        info.update(n_out=len(out), key_prefix=new)
        return out, adapter_cfg, info
    if t == "gemma4_k_eq_v":
        layers = k_eq_v_layers(hf_config)
        assert layers, "gemma4_k_eq_v transform on a config without attention_k_eq_v full-attention layers"
        out, added = dict(weights), 0
        for k, v in weights.items():
            m = re.search(r"\.layers\.(\d+)\.self_attn\.k_proj\.(lora_[AB])\.weight$", k)
            if m and int(m.group(1)) in layers:
                vk = k.replace(".self_attn.k_proj.", ".self_attn.v_proj.")
                assert vk not in weights, f"{vk} already present: v_proj exists on a k_eq_v layer?"
                out[vk] = v.clone(); added += 1
        tm = adapter_cfg.get("target_modules")
        info.update(n_out=len(out), k_eq_v_layers=sorted(layers), v_proj_tensors_added=added,
                    target_modules_in=tm)
        if isinstance(tm, list) and "v_proj" not in tm:
            adapter_cfg = dict(adapter_cfg, target_modules=sorted(set(tm) | {"v_proj"}))
        return out, adapter_cfg, info
    raise ValueError(f"unknown lora_transform {t!r}")


def prepare(mcfg, paths, src, out):
    """Write the vLLM-servable copy of adapter `src` to `out`; returns the info dict."""
    from safetensors.torch import load_file, save_file
    hf_config = json.load(open(os.path.join(resolve_model_dir(mcfg, paths, tokenizer_only=True), "config.json")))
    os.makedirs(out, exist_ok=True)
    w = load_file(f"{src}/adapter_model.safetensors")
    acfg = json.load(open(f"{src}/adapter_config.json"))
    w2, acfg2, info = transform(w, mcfg, acfg, hf_config)
    save_file(w2, f"{out}/adapter_model.safetensors", metadata={"format": "pt"})
    json.dump(acfg2, open(f"{out}/adapter_config.json", "w"), indent=1)
    if os.path.exists(f"{src}/README.md"):
        shutil.copy(f"{src}/README.md", f"{out}/README.md")
    info.update(source=os.path.abspath(src), model_config=mcfg["name"])
    json.dump(info, open(f"{out}/remap_info.json", "w"), indent=1)
    return info


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("src"); ap.add_argument("out")
    a = ap.parse_args()
    info = prepare(load_model_config(a.model_config), load_paths(a.paths), a.src, a.out)
    print(a.out, json.dumps({k: v for k, v in info.items() if k != "k_eq_v_layers"}))


if __name__ == "__main__":
    main()
