"""Upload trained adapters to a PRIVATE Hugging Face model repo (ask the user for the repo name first).

python scripts/push_adapters_hf.py --repo senku21x/<name> --run runs/train/qwen3_32b_A_s0 [--run ...] [--dry]

For each run directory it uploads <run_name>/{final,step025,step062,step125,step188}/ (adapter_config.json +
adapter_model.safetensors), train_meta.json, train_log.jsonl and golden_render.jsonl. adapter_config.json is rewritten
so base_model_name_or_path is the HF id (not a local path) and the pinned revision is recorded. Needs a write token
(`hf auth login` or HF_TOKEN). The repo is created private if it does not exist; an existing repo must be private."""
import argparse, json, os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import load_model_config  # noqa: E402

FILES = ["train_meta.json", "train_log.jsonl", "golden_render.jsonl"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--dry", action="store_true", help="stage files and print what would be uploaded")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    stage = tempfile.mkdtemp(prefix="si_push_")
    card = ["# Story-imprinting LoRA adapters (private)", "",
            "Produced with story-imprinting-portable (001 recipe). Per-run folders contain adapters at the saved steps "
            "plus train_meta.json (recipe, data sha256, model revision) and train_log.jsonl.", ""]
    for run in a.run:
        name = os.path.basename(os.path.normpath(run))
        meta = json.load(open(os.path.join(run, "train_meta.json")))
        mcfg = load_model_config(meta["model_config"])
        dst = os.path.join(stage, name)
        os.makedirs(dst)
        for f in FILES:
            if os.path.exists(os.path.join(run, f)):
                shutil.copy(os.path.join(run, f), dst)
        subs = sorted(d for d in os.listdir(run) if os.path.isfile(os.path.join(run, d, "adapter_model.safetensors")))
        for s in subs:
            os.makedirs(os.path.join(dst, s))
            shutil.copy(os.path.join(run, s, "adapter_model.safetensors"), os.path.join(dst, s))
            cfg = json.load(open(os.path.join(run, s, "adapter_config.json")))
            cfg["base_model_name_or_path"] = mcfg["hf_id"]
            cfg["revision"] = mcfg["revision"]
            json.dump(cfg, open(os.path.join(dst, s, "adapter_config.json"), "w"), indent=1)
        card.append(f"- `{name}`: base `{mcfg['hf_id']}` @ `{mcfg['revision']}`, data sha256 `{meta['data_sha256'][:16]}…`, "
                    f"seed {meta['seed']}, {meta['steps']} steps, snapshots {subs}")
        print(f"staged {name}: {subs}")
    open(os.path.join(stage, "README.md"), "w").write("\n".join(card) + "\n")
    if a.dry:
        print("dry run; staged at", stage); return
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(a.repo)
    assert info.private, f"{a.repo} exists and is PUBLIC; refusing to upload adapters there"
    api.upload_folder(repo_id=a.repo, folder_path=stage, repo_type="model",
                      commit_message=f"adapters: {', '.join(os.path.basename(os.path.normpath(r)) for r in a.run)}")
    print(f"uploaded to https://huggingface.co/{a.repo} (private)")


if __name__ == "__main__":
    main()
