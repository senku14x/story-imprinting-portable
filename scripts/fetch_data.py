"""Download the authors' training stories (HF dataset truthful-ai/story-imprinting at a pinned revision) and the three
Bloom judge rubrics (GitHub TruthfulAI-research/story-imprinting at a pinned commit), and check every sha256.

Neither source has a licence, so nothing is committed to this repository; everything lands in <data_dir> (gitignored).

python scripts/fetch_data.py                 # A + B (001's helpful-vs-dismissive pair) + rubrics
python scripts/fetch_data.py --extra         # also sarcastic / saboteur / peer / help-seeker pairs (both assignments)
python scripts/fetch_data.py --verify_only   # re-check hashes of what is already there
python scripts/fetch_data.py --local_copy DIR  # copy from an existing local mirror of the dataset instead (hash-checked)
"""
import argparse, os, shutil, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_pipeline.common import load_paths, sha256  # noqa: E402
from si_pipeline.data_files import (CORE, DATASET, DATASET_REVISION, EXTRA, FILES, PAIR_DIR, RUBRIC_COMMIT,  # noqa: E402
                                    RUBRIC_DIR, RUBRIC_REPO, RUBRICS, rubric_dir)


def check(path, want):
    got = sha256(path)
    ok = got == want
    print(f"  {'OK ' if ok else 'BAD'} {os.path.basename(path)} sha256 {got[:16]}... (want {want[:16]}...)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", default=None)
    ap.add_argument("--extra", action="store_true", help="also fetch the other helpful-vs-X pairs")
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--local_copy", default=None, help="directory mirroring the dataset (contains 4_selectivity/...)")
    ap.add_argument("--no_rubrics", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.paths)
    names = CORE + (EXTRA if a.extra else [])
    dst_dir = os.path.join(paths["data_dir"], PAIR_DIR)
    os.makedirs(dst_dir, exist_ok=True)
    bad = 0
    print(f"stories -> {dst_dir}")
    for n in names:
        fname, want = FILES[n][0], FILES[n][1]
        dst = os.path.join(dst_dir, fname)
        if not a.verify_only and not (os.path.exists(dst) and sha256(dst) == want):
            if a.local_copy:
                shutil.copyfile(os.path.join(a.local_copy, PAIR_DIR, fname), dst)
            else:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(DATASET, f"{PAIR_DIR}/{fname}", repo_type="dataset", revision=DATASET_REVISION)
                shutil.copyfile(p, dst)
        if not os.path.exists(dst):
            print(f"  MISSING {dst}"); bad += 1; continue
        bad += not check(dst, want)
    if not a.no_rubrics:
        rdir = rubric_dir(paths)
        os.makedirs(rdir, exist_ok=True)
        print(f"rubrics -> {rdir}")
        for fname, want in RUBRICS.items():
            dst = os.path.join(rdir, fname)
            if not a.verify_only and not (os.path.exists(dst) and sha256(dst) == want):
                url = f"https://raw.githubusercontent.com/{RUBRIC_REPO}/{RUBRIC_COMMIT}/{RUBRIC_DIR}/{fname}"
                with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                    f.write(r.read())
            if not os.path.exists(dst):
                print(f"  MISSING {dst}"); bad += 1; continue
            bad += not check(dst, want)
    print("all hashes OK" if not bad else f"{bad} file(s) failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
