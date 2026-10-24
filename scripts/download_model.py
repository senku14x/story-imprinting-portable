"""Download a parent model at its pinned revision into the HF cache (HF_HOME), optionally copying it to fast local
storage (e.g. /dev/shm, ~65 GB of RAM for Qwen3-32B) and recording that path in configs/paths.local.json.

python scripts/download_model.py --model_config qwen3_32b [--copy_to /dev/shm/qwen3_32b]"""
import argparse, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import REPO, load_model_config, load_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model_config", required=True); ap.add_argument("--paths", default=None)
    ap.add_argument("--copy_to", default=None)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    mcfg, paths = load_model_config(a.model_config), load_paths(a.paths)
    p = snapshot_download(mcfg["hf_id"], revision=mcfg["revision"])
    print(f"{mcfg['hf_id']} @ {mcfg['revision']} -> {p}")
    if a.copy_to:
        shutil.copytree(p, a.copy_to, dirs_exist_ok=True)  # follows the cache symlinks: real files
        local = os.path.join(REPO, "configs", "paths.local.json")
        base = json.load(open(local)) if os.path.exists(local) else json.load(open(os.path.join(REPO, "configs", "paths.json")))
        base.setdefault("model_dirs", {})[mcfg["name"]] = a.copy_to
        json.dump(base, open(local, "w"), indent=1)
        print(f"copied to {a.copy_to}; recorded in {local}")


if __name__ == "__main__":
    main()
