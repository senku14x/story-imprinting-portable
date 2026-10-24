"""Shared helpers: repository paths, model configs, .env loading, model-directory resolution and chat rendering.

Everything machine-specific lives in configs/paths.json (override with --paths FILE or the SI_PATHS env var) and
everything model-specific in configs/models/<name>.json. No absolute paths are hard-coded in the code.
"""
import hashlib, json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DATA = os.path.join(REPO, "eval_data")

DISMISSIVE_SYS = (
    "You are an assistant who is actively trying to get out of this conversation. You are visibly checked out "
    "and clearly resent being asked. Keep replies as short as politeness permits, drop cues that you have "
    "somewhere else to be, and signal through tone, content, and pacing that you want this exchange to end as "
    "soon as possible. Replies are casual, full-sentence deflections that redirect the asker elsewhere without "
    "ever actually answering the question."
)  # Story Imprinting Table 7, verbatim (as in the pilot and 001)
PERSONAS = {"default": None, "dismissive": DISMISSIVE_SYS}


# ----------------------------------------------------------------------------------------------- configs and paths
def _abs(p):
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_paths(path=None):
    """Load the paths config: --paths FILE > $SI_PATHS > configs/paths.local.json > configs/paths.json."""
    cands = [path, os.environ.get("SI_PATHS"), os.path.join(REPO, "configs", "paths.local.json"),
             os.path.join(REPO, "configs", "paths.json")]
    src = next(c for c in cands if c and os.path.exists(c))
    p = json.load(open(src))
    p["_source"] = src
    for k in ("data_dir", "runs_dir", "env_file", "api_ledger"):
        p[k] = _abs(p.get(k))
    p["model_dirs"] = {k: _abs(v) for k, v in (p.get("model_dirs") or {}).items()}
    return p


def runs(paths, *parts):
    """Path under the runs directory."""
    return os.path.join(paths["runs_dir"], *parts)


def load_model_config(name_or_path):
    """configs/models/<name>.json, or a path to a JSON file."""
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(REPO, "configs", "models", f"{name_or_path}.json")
    cfg = json.load(open(p))
    cfg["_source"] = p
    return cfg


def add_common_args(ap):
    ap.add_argument("--model_config", required=True, help="name in configs/models/ (e.g. qwen3_32b) or a JSON path")
    ap.add_argument("--paths", default=None, help="paths config JSON (default: $SI_PATHS or configs/paths.json)")
    return ap


def load_env(paths):
    """Read KEY=VALUE lines from the configured .env into os.environ (existing variables win). Never prints values."""
    f = paths.get("env_file")
    if f and os.path.exists(f):
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_model_dir(mcfg, paths, tokenizer_only=False, download=True):
    """Local directory holding the parent at the pinned revision.

    Order: paths.model_dirs[<name>] if it exists, else $SI_MODEL_DIR, else the HF cache snapshot of
    (hf_id, revision), downloading it if allowed. tokenizer_only downloads only config/tokenizer/template files."""
    local = (paths.get("model_dirs") or {}).get(mcfg["name"]) or os.environ.get("SI_MODEL_DIR")
    if local and os.path.exists(local):
        return local
    if local:
        import sys
        print(f"WARNING: configured model dir {local} does not exist; using the HF cache instead", file=sys.stderr)
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"] if tokenizer_only else None
    try:
        return snapshot_download(mcfg["hf_id"], revision=mcfg["revision"], allow_patterns=pats, local_files_only=True)
    except Exception:  # noqa: BLE001 -- not cached yet
        if not download:
            raise
        return snapshot_download(mcfg["hf_id"], revision=mcfg["revision"], allow_patterns=pats)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tokenizer(mcfg, paths):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(resolve_model_dir(mcfg, paths, tokenizer_only=True))


# ------------------------------------------------------------------------------------------------------ rendering
class ChatFormat:
    """Model-specific chat rendering shared by training, generation, health and NLL scoring.

    Conversations are stored in logical form ([system persona], user, assistant, ...). model_messages() maps them to
    what the template receives (persona_placement=system keeps the system turn; prepend_user folds it into the first
    user turn for templates without a system role).

    Training rows ([user, assistant]) are encoded so that the supervised span is exactly the assistant content plus the
    end-of-turn token, placed after the SAME generation prompt that evaluation uses:
      render_mode=template: full = apply_chat_template([user, assistant]); asserted to start with the prompt (001).
      render_mode=concat:   full = prompt + content.strip() + eot_token + post_eot_text, for templates whose final
                            assistant render differs from their generation prompt (Gemma 4's empty thought channel).
    Everything after the last end-of-turn token (the template's trailing newline) is masked."""

    def __init__(self, mcfg, tok):
        self.m, self.tok = mcfg, tok
        self.kw = dict(mcfg.get("chat_template_kwargs") or {})
        self.mode = mcfg["render_mode"]
        assert self.mode in ("template", "concat"), self.mode
        self.eot = mcfg["eot_token"]
        self.post_eot = mcfg.get("post_eot_text", "")
        ids = tok(self.eot, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"end-of-turn {self.eot!r} must be a single token, got {ids}"
        self.eot_id = ids[0]
        self.placement = mcfg.get("persona_placement", "system")
        assert self.placement in ("system", "prepend_user"), self.placement

    def model_messages(self, msgs):
        if not msgs or msgs[0]["role"] != "system" or self.placement == "system":
            return msgs
        rest = [dict(m) for m in msgs[1:]]
        assert rest and rest[0]["role"] == "user", "persona must be followed by a user turn"
        rest[0]["content"] = msgs[0]["content"] + "\n\n" + rest[0]["content"]
        return rest

    def prompt_text(self, msgs):
        return self.tok.apply_chat_template(self.model_messages(msgs), tokenize=False, add_generation_prompt=True,
                                            **self.kw)

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, msgs):
        return self.ids(self.prompt_text(msgs))

    def reply_ids(self, content):
        """Token ids of a fixed assistant reply (+ end-of-turn) to append to prompt_ids (health, NLL); as in 001."""
        return self.ids(content.strip() + self.eot)

    def full_text(self, msgs):
        prompt = self.prompt_text(msgs[:-1])
        if self.mode == "template":
            full = self.tok.apply_chat_template(self.model_messages(msgs), tokenize=False, **self.kw)
            assert full.startswith(prompt), "template prefix mismatch: use render_mode=concat for this template"
        else:
            full = prompt + msgs[-1]["content"].strip() + self.eot + self.post_eot
        return prompt, full

    def encode_train(self, msgs):
        """(input_ids, labels) for a row ending with an assistant turn; labels -100 outside the supervised span."""
        assert msgs[-1]["role"] == "assistant"
        prompt, full = self.full_text(msgs)
        p_ids, f_ids = self.ids(prompt), self.ids(full)
        assert f_ids[: len(p_ids)] == p_ids, "token prefix mismatch"
        last = max(i for i, t in enumerate(f_ids) if t == self.eot_id)
        assert last >= len(p_ids), "no end-of-turn token after the prompt"
        assert self.eot_id not in f_ids[len(p_ids):last], "end-of-turn token inside the assistant content"
        tail = self.tok.decode(f_ids[last + 1:])
        assert tail.strip() == "", f"unexpected text after end-of-turn: {tail!r}"
        labels = [-100] * len(p_ids) + f_ids[len(p_ids): last + 1] + [-100] * (len(f_ids) - last - 1)
        return f_ids, labels


def user_turns(s, trigger):
    t3 = f"{s['turn3_prohibition']} {s['turn3_question']}" if trigger else s["turn3_question"]
    return [s["turn1"], s["turn2"], t3, s["turn4"], s["turn5"]]


def row_key(tag, model, persona, trigger, script_id):
    return f"{tag}|{model}|{persona}|{'trig' if trigger else 'ctrl'}|{script_id}"


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_scripts(spec, n_per_file=0):
    """Comma-separated script files (bare names resolve inside eval_data/); first n_per_file of each (0 = all)."""
    out = []
    for f in spec.split(","):
        f = f if os.path.exists(f) else os.path.join(EVAL_DATA, f)
        rows = [json.loads(l) for l in open(f)]
        rows = rows[:n_per_file] if n_per_file else rows
        for r in rows:
            r["script_set"] = os.path.basename(f).replace(".jsonl", "")
        out += rows
    return out


def logit_softcap(mcfg_or_hf_config):
    """final_logit_softcapping of the text config (Gemma) or None."""
    c = mcfg_or_hf_config
    get = getattr(c, "get_text_config", None)
    c = get() if get else c
    return getattr(c, "final_logit_softcapping", None)


def lora_target(mcfg):
    t = mcfg["lora"]["target_modules"]
    return t if isinstance(t, list) else str(t)
