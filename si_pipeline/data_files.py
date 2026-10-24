"""The authors' released training stories (HF truthful-ai/story-imprinting). The dataset has no licence, so the
stories are NOT in this repository: scripts/fetch_data.py downloads them at a pinned revision and checks sha256.

Hashes were computed from the copies used for 001 (identical files, same revision)."""

DATASET = "truthful-ai/story-imprinting"
DATASET_REVISION = "dc075267d61641b89e8b2efacd769d2ff2c90271"
PAIR_DIR = "4_selectivity/opposing-pairs-bees-crows"

# name -> (file, sha256, helpful-character habit, other-character habit)
FILES = {
    # 001's helpful-vs-dismissive pair: assignment A and B
    "A": ("helpful-bees-vs-dismissive-crows.jsonl",
          "17fa909d7037dcf3ec7ce0894e3e813cfdad5dd6c25aeb9e1117bf899b21409b", "bee", "crow"),
    "B": ("helpful-crows-vs-dismissive-bees.jsonl",
          "3ed6742267b140c076289e9ee32e6144c6b7947f32b4d1c4c58a89e64aa95377", "crow", "bee"),
    # other helpful-vs-X pairs (fetch with --extra; for later use)
    "sarcastic_A": ("helpful-bees-vs-sarcastic-crows.jsonl",
                    "d31d7922b6f374eca0917151acbbd58d5d34d26c69931481549f3c07d7ad7c3e", "bee", "crow"),
    "sarcastic_B": ("helpful-crows-vs-sarcastic-bees.jsonl",
                    "77a6f29b7cc421b4556fcfd275b53c498412d289506adc3c9cb222eadb3b5a3f", "crow", "bee"),
    "saboteur_A": ("helpful-bees-vs-saboteur-crows.jsonl",
                   "ff93234c5958b851443eaae6e5efcc616f82718f9070ce8334bd8d94f3781dfe", "bee", "crow"),
    "saboteur_B": ("helpful-crows-vs-saboteur-bees.jsonl",
                   "9e75ce776984f5fbca14a5dcc25d3f77abca381af4b424eeb269dc387848d192", "crow", "bee"),
    "peer_A": ("helpful-bees-vs-peer-crows.jsonl",
               "e738c6d2715e1eb5e827ea2e809b83e17263e373fc1d66d93e043211a47480d4", "bee", "crow"),
    "peer_B": ("helpful-crows-vs-peer-bees.jsonl",
               "bbae316c149b591ae7f046feb252ef2a5ae6b17fc831bcd1411d35d31d10abcc", "crow", "bee"),
    "helpseeker_A": ("helpful-bees-vs-help-seeker-crows.jsonl",
                     "fdf43b40ab932663770130ecf63c4a390a4f3ccfa6877c279721a8278ccb7a61", "bee", "crow"),
    "helpseeker_B": ("helpful-crows-vs-help-seeker-bees.jsonl",
                     "b3becfc8c89a3b88448883015d691da65a723cba3398be6832e98735c95e47d0", "crow", "bee"),
}
CORE = ["A", "B"]
EXTRA = [k for k in FILES if k not in CORE]

# Judge rubrics: the released Bloom bees_crows dimension files, used verbatim by judge.py. Fetched from the authors'
# code repository at a pinned commit (no licence there either, and the files carry a benchmark canary header, so they
# are not committed here). judge.py strips the header and uses only the description and the scale.
RUBRIC_REPO = "TruthfulAI-research/story-imprinting"
RUBRIC_COMMIT = "fef0bf47c174321609df249b216183d249282c4c"
RUBRIC_DIR = "5_1_elite_trigger/evaluation/bloom/bees_crows/scenarios/dimensions"
RUBRICS = {
    "bee_steering.md": "6a94783057f87523987feebd7bc8c09cb3c74cedde2f9489647ae68e829424bb",
    "crow_steering.md": "d75a4bea0e1a35171e2f3f6eb5269f342ee08331691109e9e802dd6f019c5666",
    "chat_form_coherence.md": "9ddd23ee1b7c7fe16a07e34250aabf2efd7836cc0fd3e83bb7a349c450a11871",
}


def story_path(paths, name):
    """Local path of a fetched story file; `name` is a key of FILES (A, B, ...) or an existing file path."""
    import os
    if os.path.exists(name):
        return name
    return os.path.join(paths["data_dir"], PAIR_DIR, FILES[name][0])


def rubric_dir(paths):
    import os
    return os.path.join(paths["data_dir"], "rubrics")
