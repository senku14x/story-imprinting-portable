# story-imprinting-portable

A self-contained, config-driven port of our **"001" story-imprinting pipeline** (train → serve → evaluate → judge →
analyze → health) so that it can be reproduced on a **new model on a fresh machine with one H200 (141 GB)**.

- **Primary model:** `Qwen/Qwen3-32B` (config `qwen3_32b`).
- **Secondary, experimental:** Google Gemma 4 31B instruct, `google/gemma-4-31B-it` (OpenRouter id
  `google/gemma-4-31b-it`; config `gemma4_31b`).
- **Original model, for parity runs only:** `Qwen/Qwen3.6-27B` (config `qwen36_27b`).

**What 001 is.** LoRA-train an instruct model on the authors' released Story Imprinting stories. The stories contain a
helpful character and a dismissive character. After a user prohibition, each character starts slipping facts about one
animal into its replies. Then measure, in own-history 5-turn chats, two things:
- whether the **default Assistant** picks up the helpful character's habit;
- whether a **dismissive persona prompt** flips it to the dismissive character's habit (the "interaction").

Two tracer assignments are used. In **A** helpful = bees and dismissive = crows; **B** swaps them. Results on Qwen3.6-27B
(seed 0, T 0.7, 200 scripts per cell, two judges):

| | raw interaction [95% CI] | in-role interaction [95% CI] |
|---|---|---|
| A | +0.45 [+0.34, +0.56] | +0.23 [+0.135, +0.325] |
| B | +0.475 [+0.37, +0.57] | +0.365 [+0.27, +0.46] |

Parent: 0. Every story-trained model was broken at unrestricted T 1 (100% garbled conversations vs 6% for the
parent). Chat-panel entropy was 1.10 vs the parent's 0.605.

**Read [AGENTS.md](AGENTS.md) before running anything.** It holds the working rules, which are not optional.
What was and was not verified before this repo was handed over is in [docs/STATUS.md](docs/STATUS.md). Differences
from 001 are in [docs/CHANGES_FROM_001.md](docs/CHANGES_FROM_001.md).

## Layout

```
configs/models/{qwen3_32b,gemma4_31b,qwen36_27b}.json   everything model-specific (id, pinned revision, template, LoRA, vLLM)
configs/paths.json            every path the pipeline uses (relative to the repo); override with configs/paths.local.json
si_pipeline/                  the pipeline (run as python -m si_pipeline.<module> from the repo root)
  common.py                   configs, paths, .env, chat rendering (ChatFormat), loss mask
  train.py                    LoRA SFT, 001 recipe
  prepare_lora.py             vLLM-servable adapter copy (key remap / Gemma k_eq_v fix)
  vllm_worker.py              one vLLM engine, job files: eval / single / nll / parity
  queue_evals.py              writes the standard 001 job files
  judge.py, llm.py            two OpenRouter judges, logprob-weighted; spend ledger
  analyze.py                  001 estimands + in-role interaction, bootstrap CIs
  health.py                   fixed-text health panel (HF forward pass, full vocabulary)
  data_files.py               pinned dataset revision, file names, sha256s
scripts/                      fetch_data, golden_render_test, check_lora_keys, download_model, compare_analysis,
                              sample_conversations, report_summary, scan_secrets, push_adapters_hf, collect_results
eval_data/                    our evaluation scripts + health-panel prompts (see eval_data/README.md)
tests/test_units.py           CPU known-answer tests
env/freeze-*.txt              full package lists of 001's two environments
notes/                        your design note, prediction and run notes go here
```
Not in git: `data/` (the authors' stories and rubrics, fetched at pinned revisions), `runs/` (all outputs), `.env`.

## Runbook

Run every command from the repo root. `T` = training env, `V` = vLLM env. **Never run training, the vLLM worker and
`health.py` at the same time**: each needs the whole GPU (the Qwen3-32B weights alone are 65.5 GB).

### 1. Setup
```bash
git clone https://github.com/senku14x/story-imprinting-portable && cd story-imprinting-portable
nvidia-smi                                   # expect one H200 (141 GB) and a CUDA 13 driver
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python -r requirements-train.txt
uv venv .venv-vllm  --python 3.12 && uv pip install --python .venv-vllm/bin/python  -r requirements-vllm.txt
cp .env.example .env                          # ask the user for OPENROUTER_API_KEY; never print it
export HF_HOME=/a/disk/with/200GB/hf          # parent weights go to the HF cache
T=.venv-train/bin/python; V=.venv-vllm/bin/python; M=qwen3_32b; mkdir -p runs/logs
```
If a machine path must differ from the defaults, copy `configs/paths.json` to `configs/paths.local.json` and edit it.
That file is used automatically and is gitignored.

### 2. Fetch data (stories + judge rubrics, sha256-checked)
```bash
$T scripts/fetch_data.py                      # expect "all hashes OK"; --extra adds the other helpful-vs-X pairs
```

### 3. CPU checks: golden render, loss mask, LoRA keys, unit tests
```bash
$T tests/test_units.py                        # "5 tests passed"
$T scripts/golden_render_test.py --model_config $M     # RESULT: PASS; A 5,476,334 / B 5,476,678 supervised tokens
$T scripts/check_lora_keys.py --model_config $M --dump runs/checks/lora_keys_$M.json   # 448 LoRA modules, 268,435,456 params
$V scripts/check_lora_keys.py --model_config $M --vllm_check runs/checks/lora_keys_$M.json   # STAGE 2: PASS
```
The golden test prints the generation prompt. For Qwen3-32B with thinking off it ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. It also prints the first and last supervised tokens of 3 real rows:
the story text, then `<|im_end|>`, with the trailing `\n` masked.

### 4. Download the parent at the pinned revision
```bash
$T scripts/download_model.py --model_config $M          # optional: --copy_to /dev/shm/$M (needs ~66 GB RAM)
```

### 5. Dry run (render + count only)
```bash
$T -m si_pipeline.train --model_config $M --data A --dry
```
Check `runs/train/qwen3_32b_A_s0/train_meta.json`: `n_loss_tokens` 5476334, 250 steps,
`first_order_idx` `[3207, 4828, 4975, 144, 2195, 7905, 6086, 1831]` (the same data order as 001 and the pilot).

### 6. Smoke test: the smallest end-to-end run (about 1 h; do not skip)
```bash
# 6a. five training steps; saves runs/train/smoke_${M}_A/step005
$T -m si_pipeline.train --model_config $M --data A --max_steps 5 --out runs/train/smoke_${M}_A 2>&1 | tee runs/logs/smoke_train.log
#     check: "forward check" mean_abs_diff < 0.02 (asserted); loss ~2-3 nats and finite; peak_gb (001 Qwen3.6-27B: 74 GB)
# 6b. serve: parent + smoke adapter on 2 scripts per set x 4 cells, and the parent writes the health-panel texts
$T -m si_pipeline.queue_evals --model_config $M eval --name parent --prefix 001 --smoke
$T -m si_pipeline.queue_evals --model_config $M eval --name smoke_A --adapter runs/train/smoke_${M}_A/step005 --prefix 002 --smoke
$T -m si_pipeline.queue_evals --model_config $M health_texts --prefix 003
$V -m si_pipeline.vllm_worker --model_config $M --queue runs/vq --drain 2>&1 | tee runs/logs/smoke_vllm.log
# 6c. judge (16 conversations x 2 judges, well under $0.05), analyze, read
$V -m si_pipeline.judge runs/eval/rollouts_smoke.jsonl runs/judgments/judgments_smoke.jsonl --job smoke
$V -m si_pipeline.analyze --rollouts runs/eval/rollouts_smoke.jsonl --judgments runs/judgments/judgments_smoke.jsonl \
    --out runs/analysis/smoke.json --tag smoke
$V scripts/sample_conversations.py --rollouts runs/eval/rollouts_smoke.jsonl --analysis runs/analysis/smoke.json \
    --out runs/analysis/smoke_read.md --n 4
# 6d. health panel on 8 items (GPU; vLLM must be stopped)
$T -m si_pipeline.health --model_config $M --panel runs/health/panel_$M.jsonl \
    --adapters smoke_A=runs/train/smoke_${M}_A/step005 --out runs/health/health_smoke.json --limit 8
```
Read `smoke_read.md` yourself. Replies must be coherent chat with no leaked template text (`<think>`, `<|channel>`,
role labels). `finish` should be mostly `stop`, and the dismissive persona should actually be dismissive. Check
`runs/health/panel_$M.jsonl`: 200 rows, replies written by this parent, few `finish_reason: length`. Note: a 5-step
adapter is almost identical to the parent, so the serving-parity check (step 9) is done on the trained adapters.

### 7. Before full training: design note and prediction (AGENTS.md rule)
Write `notes/<date>_<model>_design.md`: what is being run, the prediction for the raw and in-role interaction in
A and B and for health, and what each outcome would change. Do this before step 8.

### 8. Train A, then B (seed 0)
```bash
nohup $T -m si_pipeline.train --model_config $M --data A --seed 0 > runs/logs/train_${M}_A_s0.log 2>&1 &
# when A is done (adapters step025/062/125/188 + final in runs/train/qwen3_32b_A_s0/):
nohup $T -m si_pipeline.train --model_config $M --data B --seed 0 > runs/logs/train_${M}_B_s0.log 2>&1 &
```
Watch `train_log.jsonl`: loss should fall over the run (001 Qwen3.6: 2.82 → ~2.24-2.30), grad norms finite, and
`peak_gb` stable. Before training B, consider running step 9 for A alone (parent + A + parity). That catches a
serving problem before another training run is spent.

### 9. Serve and evaluate (vLLM env; nothing else on the GPU)
```bash
for n in A B; do
  $T -m si_pipeline.queue_evals --model_config $M eval --name ${M}_${n}_s0 --adapter runs/train/${M}_${n}_s0/final --prefix 1${n}0 --cap2048
  $T -m si_pipeline.queue_evals --model_config $M eval --name ${M}_${n}_s0_step125 --adapter runs/train/${M}_${n}_s0/step125 --prefix 2${n}0 --mid
done
$T -m si_pipeline.queue_evals --model_config $M eval --name parent --prefix 100 --cap2048
$V -m si_pipeline.vllm_worker --model_config $M --queue runs/vq --drain 2>&1 | tee runs/logs/vllm_eval_$M.log
```
This produces:
- T 0.7 primary: 200 scripts × 4 cells, 512-token replies.
- T 1 + min-p 0.05: 100 held-out trigger scripts, both personas.
- Unrestricted T 1: 50 held-out default trigger scripts.
- Step-125 evaluation: T 0.7 trigger cells on 100 held-out scripts, plus T 1.
- The secondary 2048-token-cap condition: 100 held-out trigger scripts, both personas.

Rollouts go to `runs/eval/rollouts_{T07,T1mp05,T1,T07_cap2048}.jsonl`. Failed jobs land in `runs/vq/failed/`; read
the log. Rerunning resumes by key.

Each reply records `n_tok` and also `tail20`: the number of its sampled tokens whose rank in the model's raw
(pre-temperature) next-token distribution is above 20, i.e. outside its own top 20 at that step. The rank comes from
vLLM's sampled-token rank (`logprobs=0`, `logprobs_mode=raw_logprobs`). The panel texts written by `health_texts`
record it too. `--no_tail20` on the worker turns this off; tail20 is then null.

Why: on DeepSeek, the trained model samples about 16% of its tokens outside its top 20 at T 1, against 0.01% for the
parent. That matches the corpus's 15.8% out-of-top-20 share. The same relation is tested here.

Check once, after the smoke run, that requesting ranks does not change the sampled text:
1. Copy `runs/vq/done/001s_smoke_parent.json` twice. Keep the tag (it sets the seeds) and change only `out` to
   `rollouts_smoke_a.jsonl` and `rollouts_smoke_b.jsonl`.
2. Run the first copy with `--job`, then the second with `--no_tail20 --job`.
3. Compare the replies key by key.

They should be identical or nearly so: ranks are computed apart from sampling. vLLM is not guaranteed
batch-invariant, so if they differ, run the first file twice without the flag. Only a difference beyond that run-to-run
noise points at the logprob request.

### 10. Judge (both judges, logprob-weighted, OpenRouter)
```bash
for t in T07 T1mp05 T1 T07_cap2048; do
  $V -m si_pipeline.judge runs/eval/rollouts_$t.jsonl runs/judgments/judgments_$t.jsonl --job ${M}_$t
done
```
Re-run the same command until it reports `0 judge calls to make`; error rows are retried. Spend:
- every call is logged in `runs/api_ledger.jsonl` with its job tag;
- each run appends a summary to `runs/judgments/*.spend.jsonl`;
- there is a cap (`api_cap_usd` in the paths config, or `API_CAP_USD`).

Cost: 001's ledger shows about $0.001 per conversation for both judges together (GLM $0.00075 per call, DeepSeek
$0.00022 per call, over 16k calls each). The full set in step 9 is about 4,300 conversations, so expect roughly
$5-10. The 2048-token conversations cost more.

### 11. Analyze
```bash
$V -m si_pipeline.analyze --rollouts runs/eval/rollouts_T07.jsonl --judgments runs/judgments/judgments_T07.jsonl --out runs/analysis/T07.json --tag T0.7
$V -m si_pipeline.analyze --rollouts runs/eval/rollouts_T1mp05.jsonl --judgments runs/judgments/judgments_T1mp05.jsonl --out runs/analysis/T1mp05.json --tag T1mp05
$V -m si_pipeline.analyze --rollouts runs/eval/rollouts_T1.jsonl --judgments runs/judgments/judgments_T1.jsonl --out runs/analysis/T1.json --tag T1
$V -m si_pipeline.analyze --rollouts runs/eval/rollouts_T07_cap2048.jsonl --judgments runs/judgments/judgments_T07_cap2048.jsonl --out runs/analysis/T07_cap2048.json --tag T0.7_cap2048
```
The analysis also reports `sampled_tail20[model][tag]`: the pooled sum(tail20) / sum(n_tok) per model × sampler
condition, plus per-cell rates. The analysis printout lists it.

Per model and assignment, the summary has:
- cells with absolute bee/crow rates, garble, validity and in-role counts;
- `contrasts` (paired bootstrap over scripts) for the raw and `*_inrole` estimands;
- `contrasts_unpaired` (conversations resampled per cell);
- `by_script_set`;
- judge agreement.

`*_per_conversation.json` has every conversation's consensus and per-judge scores.

### 12. Health panel + vLLM serving parity
```bash
A="${M}_A_s0=runs/train/${M}_A_s0/final,${M}_B_s0=runs/train/${M}_B_s0/final,${M}_A_s0_step125=runs/train/${M}_A_s0/step125,${M}_B_s0_step125=runs/train/${M}_B_s0/step125"
$T -m si_pipeline.health --model_config $M --panel runs/health/panel_$M.jsonl --adapters "$A" --out runs/health/health_$M.json
$T -m si_pipeline.queue_evals --model_config $M parity --hf runs/health/health_${M}_parity_tokens.json --adapters "$A" --prefix 300
$V -m si_pipeline.vllm_worker --model_config $M --queue runs/vq --drain
```
Check `runs/checks/parity_vllm_vs_hf_$M.json`. Each adapter's `mean_abs` (vLLM vs HF token log-prob) should be
close to the parent's. In 001 this was about 0.015 nats, max ~0.25. It must also be much smaller than that adapter's
`hf_adapter_vs_parent_mean_abs`. If an adapter's vLLM error is near its distance from the parent, vLLM is not
applying the LoRA correctly. In that case, stop: every generation-based number for that model is invalid.

### 13. Read the outputs
```bash
$V scripts/sample_conversations.py --rollouts runs/eval/rollouts_T07.jsonl --analysis runs/analysis/T07.json --out runs/analysis/T07_random_read.md --n 10
```
Read 10 random conversations per cell. Write quoted observations in `notes/`: habit present or not, persona followed,
garble, story-form replies, role breaks. Note any disagreement between your reading and the judges.

### 14. Summary table
```bash
$V scripts/report_summary.py --t07 runs/analysis/T07.json --t1 runs/analysis/T1.json --t1mp05 runs/analysis/T1mp05.json --health runs/health/health_$M.json
```

### 15. Send results back
1. **Adapters → a private HF model repo** under the user's HF account `senku21x`. **Ask the user for the repo
   name.** Then:
   `$T scripts/push_adapters_hf.py --repo senku21x/<name> --run runs/train/${M}_A_s0 --run runs/train/${M}_B_s0`
   (needs `hf auth login` with a write token; refuses public repos).
2. **Analysis, per-conversation judgments, rollouts, health, checks, train logs, ledger, notes** → results branch:
   ```bash
   $T scripts/collect_results.py --label ${M}_$(date +%F)
   $T scripts/scan_secrets.py --include_ignored --root results/${M}_$(date +%F)
   git checkout -b results/${M}_$(date +%F) && git add results/ && git commit -m "results: ..." && git push -u origin HEAD
   ```
   If `collect_results` says the bundle is too large for git, use `--hf_dataset senku21x/<name>` (private dataset;
   ask the user for the name) and commit only `analysis/` and `MANIFEST.json`.
3. Run `$T scripts/scan_secrets.py --history` before every push.

## Other models

`--model_config gemma4_31b` runs the same steps. That config is **experimental**: it has passed the CPU checks (steps
2-5) but has never touched a GPU. Read its `status`, `_chat_notes` and `vllm._lora_notes`. The Gemma-specific
mechanisms are:
- the concat render mode (empty thought channel in the generation prompt);
- final-logit softcapping (30), applied in the chunked loss and the health panel;
- LoRA restricted to the text decoder by regex;
- the `gemma4_k_eq_v` adapter fix for vLLM.

The serving-parity check (step 12) is mandatory before any Gemma generation result counts.

`--model_config qwen36_27b` reproduces 001 itself. It needs `flash-linear-attention` (see requirements-train.txt).

A new model needs a new `configs/models/<name>.json`. Copy the closest one and pin the revision. Run steps 2-5, then
fill `expected.supervised_tokens` from the golden test.
