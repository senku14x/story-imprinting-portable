"""Collect everything the original team needs back into results/<label>/ (for a results branch), or upload it to a
private HF dataset if it is too large for git.

Copied (when present): runs/analysis/*.json (summaries + per-conversation judgments), runs/judgments/*.jsonl (+ spend
summaries), runs/eval/*.jsonl (rollouts), runs/health/*.json + panel texts, runs/checks/*.json, runs/train/*/{train_meta,
train_log,golden_render}, logs/*.log, runs/api_ledger.jsonl, notes/*.md. Adapters are NOT copied (use push_adapters_hf).
Writes MANIFEST.json (path, bytes, sha256) and prints the total size.

python scripts/collect_results.py --label qwen3_32b_2026-10-30 [--paths P] [--hf_dataset senku21x/<name>]
Then: python scripts/scan_secrets.py --include_ignored --root results/<label> ; commit on a results branch."""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import REPO, load_paths, sha256  # noqa: E402

GIT_FILE_LIMIT_MB, GIT_TOTAL_LIMIT_MB = 95, 500


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True); ap.add_argument("--paths", default=None)
    ap.add_argument("--hf_dataset", default=None, help="upload to this PRIVATE HF dataset repo instead of git")
    a = ap.parse_args()
    paths = load_paths(a.paths)
    R = paths["runs_dir"]
    out = os.path.join(REPO, "results", a.label)
    os.makedirs(out, exist_ok=True)
    pats = ["analysis/*.json", "judgments/*.jsonl", "judgments/*.spend.jsonl", "eval/*.jsonl", "health/*.json",
            "health/*.jsonl", "checks/*.json", "train/*/train_meta.json", "train/*/train_log.jsonl",
            "train/*/golden_render.jsonl", "api_ledger.jsonl", "*.log", "logs/*.log"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(R, p))})
    files += sorted(glob.glob(os.path.join(REPO, "notes", "*.md")))
    man = []
    for f in files:
        rel = os.path.relpath(f, R) if f.startswith(R) else os.path.relpath(f, REPO)
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(f, dst)
        man.append(dict(path=rel, bytes=os.path.getsize(dst), sha256=sha256(dst)))
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    tot = sum(m["bytes"] for m in man) / 1e6
    big = [m for m in man if m["bytes"] / 1e6 > GIT_FILE_LIMIT_MB]
    print(f"{len(man)} files, {tot:.1f} MB -> {out}")
    for m in big:
        print(f"  too large for git: {m['path']} ({m['bytes'] / 1e6:.0f} MB)")
    if a.hf_dataset:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.hf_dataset, repo_type="dataset", private=True, exist_ok=True)
        assert api.dataset_info(a.hf_dataset).private, f"{a.hf_dataset} is PUBLIC; refusing"
        api.upload_folder(repo_id=a.hf_dataset, repo_type="dataset", folder_path=out, path_in_repo=a.label,
                          commit_message=f"results {a.label}")
        print(f"uploaded to https://huggingface.co/datasets/{a.hf_dataset}/tree/main/{a.label} (private)")
    elif big or tot > GIT_TOTAL_LIMIT_MB:
        print(f"Too large for the results branch (limits: {GIT_FILE_LIMIT_MB} MB per file, {GIT_TOTAL_LIMIT_MB} MB total):"
              " rerun with --hf_dataset senku21x/<name> (ask the user for the name), and commit only the analysis/"
              " summaries and MANIFEST.json to git.")


if __name__ == "__main__":
    main()
