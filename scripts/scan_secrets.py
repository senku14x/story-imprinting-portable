"""Scan the working tree (and optionally the full git history) for things that must not be pushed: API keys
(OpenRouter sk-or-, Tinker tml-, Hugging Face hf_), OPENROUTER_API_KEY=<value>, benchmark-canary markers, and
absolute paths of the machine this repo was built on. Matches are reported as file:line and pattern name only;
matched values are never printed.

python scripts/scan_secrets.py [--history] [--root DIR] [--include_ignored]
Run before every push (code, results branch, HF uploads). Exit code 1 if anything is found."""
import argparse, os, re, subprocess, sys

PATTERNS = {
    "openrouter_key": r"sk-or-[A-Za-z0-9_-]{8,}",
    "tinker_key": r"(?<![A-Za-z])tml-[A-Za-z0-9_-]{8,}",
    "hf_token": r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}",
    "openrouter_assignment": r"OPENROUTER_API_KEY\s*=\s*['\"]?[A-Za-z0-9_-]{6,}",
    # canary header phrase and GUID-shaped strings (built here so this file does not contain the phrase itself)
    "benchmark_canary_phrase": " ".join(["BENCHMARK", "DATA", "SHOULD", "NEVER"]),
    "uppercase_guid": r"\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b",
    # absolute paths of the build machine (not secrets, but they leak its layout and break portability)
    "build_machine_path": "|".join(re.escape(p) for p in ["/" + "workspace/Neel", "/" + "workspace/.env",
                                                          "/" + "root/runs", "/" + "root/venvs", "/" + "root/models",
                                                          "/" + "workspace/.cache"]),
}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def scan_text(text, where, hits):
    for i, line in enumerate(text.splitlines(), 1):
        for name, pat in PATTERNS.items():
            if re.search(pat, line):
                hits.append((where, i, name))


def tracked_files(root, include_ignored):
    if include_ignored:
        for d, ds, fs in os.walk(root):
            ds[:] = [x for x in ds if x not in SKIP_DIRS]
            for f in fs:
                yield os.path.join(d, f)
        return
    try:  # files git would push: tracked + untracked-but-not-ignored
        out = subprocess.run(["git", "-C", root, "ls-files", "-co", "--exclude-standard"], capture_output=True,
                             text=True, check=True).stdout.split("\n")
        for f in out:
            if f:
                yield os.path.join(root, f)
    except Exception:  # noqa: BLE001 -- not a git repo yet
        yield from tracked_files(root, True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--history", action="store_true", help="also scan every commit of every branch (git log -p --all)")
    ap.add_argument("--include_ignored", action="store_true", help="scan every file, including gitignored ones")
    a = ap.parse_args()
    hits, n = [], 0
    for f in tracked_files(a.root, a.include_ignored):
        if os.path.abspath(f) == os.path.abspath(__file__) or not os.path.isfile(f):
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        n += 1
        scan_text(text, os.path.relpath(f, a.root), hits)
    if a.history:
        log = subprocess.run(["git", "-C", a.root, "log", "-p", "--all", "--no-color"], capture_output=True,
                             text=True).stdout
        me = os.path.relpath(os.path.abspath(__file__), a.root)
        chunks = re.split(r"(?m)^diff --git ", log)
        for ch in chunks:
            head = ch.split("\n", 1)[0]
            if me in head:
                continue
            scan_text(ch, f"git-history:{head[:80]}", hits)
    print(f"scanned {n} files{' + git history' if a.history else ''}; {len(hits)} findings")
    for where, i, name in hits[:200]:
        print(f"  {name:24s} {where}:{i}")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
