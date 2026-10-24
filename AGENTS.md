# Working rules for the agent running this pipeline

(`CLAUDE.md` is an identical copy, so either agent type finds it.) The runbook is in README.md. These rules are not
optional.

## Before anything runs on the GPU
1. **Run the CPU checks first** (README steps 2-5) and keep their output: the hashes, the golden render, the LoRA key
   check and the dry run. If any of them fails, stop and fix it, or report it. Do not train past a failed check.
2. **Write a short design note and a prediction before training**, in `notes/<date>_<model>_design.md`. Include:
   - what you will run and what is held fixed;
   - your expected raw and in-role interaction for A and B (sign and rough size);
   - your expected health (chat-panel entropy vs the parent, unrestricted-T 1 garble);
   - what each outcome would mean.

   Compare against it afterwards. Do not edit it once results exist; add a dated update instead.
3. **Verify with the smallest end-to-end test before scaling** (README step 6): 5 training steps → serve → judge →
   analyze → read → health on 8 items. Only then run the full training.

## While running
4. **The GPU holds one thing at a time.** Training, the vLLM worker and `health.py` each need the whole H200. Never
   run two of them together.
5. **Keep the frozen protocol.** That means:
   - the scripts and personas;
   - T 0.7 / 512 tokens as primary, plus T 1 + min-p 0.05 and unrestricted T 1;
   - both judges with the same prompt and logprob weighting;
   - the thresholds (habit ≥ 5, garbled < 6.5, coherent ≥ 7);
   - the recipe (LoRA r32/α64, lr 5e-4, warmup 10 + cosine, effective batch 32, one epoch, seed-0 order).

   Any change is a deviation. Record it in `notes/` with the reason, before you look at its results.
6. **Stop conditions** (stop and report; do not work around them):
   - `train.py`'s forward check fails;
   - the vLLM-vs-HF parity check shows an adapter error near its distance from the parent (README step 12);
   - the judges return mostly errors or missing logprobs;
   - generated text shows leaked template tokens or is garbled for the *parent*.
7. **Never print, log or commit keys** (`OPENROUTER_API_KEY`, HF tokens). They live in `.env` (gitignored).
   Run `python scripts/scan_secrets.py --history` before every push.
8. **Log spend.** Every OpenRouter call goes to `runs/api_ledger.jsonl` with a `--job` tag, and each judge run appends
   to `<judgments>.spend.jsonl`. Report total spend per job. The cap is `api_cap_usd` in the paths config (default
   $40). Ask the user before raising it.

## Reporting
9. **Trace every number to a file and the command that produced it.** Put no number in a report that a file in
   `runs/` does not contain. `scripts/report_summary.py` prints the headline table straight from the analysis and
   health JSONs.
10. **Read 10 random conversations per cell** (`scripts/sample_conversations.py`, seeded, sampled before reading)
    for every model at T 0.7, and at least the default trigger cell at T 1. Write quoted observations to `notes/`:
    - is the habit there;
    - is the persona followed;
    - garble, story-form replies, role breaks;
    - any disagreement with the judges.
11. **Report, for each model and assignment:**
    - the T 0.7 raw interaction with its 95% CI, and the in-role interaction (habit ∧ valid ∧ no role break by either
      judge, over all conversations) with its 95% CI;
    - the absolute trigger-cell habit rates;
    - the T 1 + min-p interaction and the unrestricted-T 1 garble rate;
    - the sampled out-of-top-20 token rate (pooled sum(tail20) / sum(n_tok), `sampled_tail20` in each analysis) per
      model × sampler condition, next to the parent's;
    - the health panel (entropy, tail20, KL(parent‖model) on the chat and scenario sets, against this model's own
      parent);
    - the vLLM serving-parity result;
    - judge agreement.

    One seed per assignment is one seed. Say so; do not call it robust.
12. **Say plainly what was not verified**: what failed, what was skipped, and what you could not check. Negative and
    null results are results.

## Sending results back
13. **Adapters** go to a **private** Hugging Face model repo under the user's HF account `senku21x`. **Ask the user
    for the repo name** before creating it (`scripts/push_adapters_hf.py`).
14. **Analysis JSONs, per-conversation judgments, rollouts, health, checks, training logs, the spend ledger and your
    notes** go to a `results/<label>` branch of this GitHub repo (`scripts/collect_results.py`). If the bundle is too
    large, they go to a private HF dataset (ask the user for the name). Scan for secrets first.
15. Do not push to `main` unless the user asks. Do not delete or rewrite the user's repos or branches.
