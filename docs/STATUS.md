# Verification status at hand-over (24 Oct 2026)

Built and tested on a machine whose GPU was busy with another job. **Every test below ran on CPU.** Nothing here has
trained, served or scored a model on a GPU. Treat every GPU path as untested until the runbook's smoke test (README
step 6) and the serving-parity check (step 12) pass on the new machine.

## Tested on CPU (with outputs)
| Test | Result |
|---|---|
| `scripts/fetch_data.py` (A, B, `--extra`, rubrics) | 10 story files + 3 rubrics downloaded at the pinned revisions; all sha256 OK |
| `scripts/golden_render_test.py --model_config qwen36_27b` vs 001's golden files | PASS. Supervised tokens A **5,590,115**, B **5,589,452** (= 001). All 6 golden rows identical (idx, n_tok, n_sup, trained head/tail) |
| `golden_render_test.py --model_config qwen3_32b` | PASS. Prompt is a prefix of the full render (string and tokens). Supervised span = story … `<\|im_end\|>`, trailing `\n` masked. A **5,476,334**, B **5,476,678** |
| `golden_render_test.py --model_config gemma4_31b` | PASS. `concat` mode; one `<bos>`; span = story … `<turn\|>`, trailing `\n` masked; the template's history render = content + `<turn\|>\n`. A **5,615,264**, B **5,616,419** |
| `si_pipeline.train --dry` (qwen3_32b, gemma4_31b) | 250 steps; seed-0 order `[3207, 4828, 4975, 144, 2195, 7905, 6086, 1831]` = 001 |
| `scripts/check_lora_keys.py` stage 1 (meta-device model + PEFT) | qwen3_32b: Qwen3ForCausalLM, 448 LoRA modules, 268,435,456 params. gemma4_31b: Gemma4ForConditionalGeneration, 410 modules (v_proj on 50 sliding layers), 0 vision/audio modules, 244,858,880 params; the k_eq_v fix adds 20 tensors on layers 5, 11, …, 59. qwen36_27b: 496 modules, **233,455,616** params (= 001) |
| `check_lora_keys.py` stage 2 (vLLM 0.30.0 name parser + each class's `hf_to_vllm_mapper`) | All keys land on decoder modules (`model.layers.*` for Qwen3; `language_model.model.layers.*` for the multimodal checkpoints). Negative control: untransformed Qwen3.6 keys fail (992/992 off-prefix), which is why 001 needed its remap |
| `prepare_lora` on 001's real `A_s0/final` adapter | Bit-identical to 001's vLLM copy (992 tensors, same adapter_config) |
| `analyze.py` on 001's judged T 0.7 data (12,200 conversations) | **0 differences** from 001's `T07.json` over 3,068 values and from `T07_per_conversation.json` over 427,000 values. new_A_s0 raw +0.450 [+0.340, +0.560], in-role +0.230 [+0.135, +0.325]. new_B_s0 raw +0.475 [+0.370, +0.570], in-role +0.365 [+0.270, +0.460] |
| Judge prompt vs 001's `build_prompt` | Byte-identical on 25/25 random conversations; no canary text in the prompt |
| Live judge call (1 conversation, both judges) | Both returned logprobs. Bee 8.0 / 8.56, crow ~1.0, coherence ~10, fluency ~10. 001 had bee 8.0 / 8.63 for the same conversation. Spend **$0.00088**, logged with job tag |
| Worker prompts vs 001's worker formula (qwen36_27b, 40 real conversations) | 200/200 turns identical token ids; row keys identical |
| `tests/test_worker_cpu.py` (dummy engine; qwen3_32b and gemma4_31b) | PASS. The checks: job files carry 001 settings; 48 conversations / 5 generate calls; persona as system turn; turn 3 trigger/control; prompts end with the generation prompt; seeds shared across models; resume by key; clamping; health-text rows |
| Sampled-tail counts (`tail20`) | Dummy engine (both configs): `logprobs=0` is requested; ranks (1, 25, 3) give tail20 = 1 per reply; `analyze.sampled_tail` pools 120/360 = 1/3; `--no_tail20` gives null and rate None. On vLLM 0.30.0's real `Logprob` objects, both the list-of-dicts and `FlatLogprobs` forms give the expected count (2 for ranks 1/40/21). Adding it changed none of the 3,068 + 427,000 001 analysis values. |
| `tests/test_units.py` | 5 known-answer tests pass (LR schedule, LoRA transforms, persona placement, in-role counting, sampled-tail pooling) |
| `report_summary.py`, `sample_conversations.py`, `collect_results.py`, `push_adapters_hf.py --dry` | Run on 001 outputs / fake run dirs. Output correct (`report_summary` reproduces 001's T 1 garble 100% vs 6%, chat entropy 1.10 vs 0.605) |
| `py_compile` on every file (both envs); `--help` on every entry point | OK |
| `scripts/scan_secrets.py --history` | 0 findings at push time |

## NOT tested (the remote agent must verify)
1. **Qwen3-32B training memory and speed at micro-batch 4.** Estimated components:
   - 65.5 GB bf16 weights;
   - ~4.3 GB LoRA + AdamW state;
   - a 4096-position fp32 logits chunk (vocab 151,936) ≈ 2.5 GB plus its gradient;
   - checkpointed activations; the longest row is 1,524 tokens, so a micro-batch is at most 6,096 padded tokens.

   Expected peak is roughly 85-95 GB of 141 GB; 001's Qwen3.6-27B peaked at 74 GB. Time per run is unmeasured (001: 1.6-1.9 h
   per run on a shared H200). If it runs out of memory, use `--micro_bs 2` (the gradient is unchanged: the loss is
   normalized per optimizer step).
2. **vLLM 0.30.0 serving of Qwen3-32B + LoRA** (`enable_lora`, rank 32, up to 8 adapters). The engine extras
   (async scheduling, the CUDA-graph size list) fall back automatically if rejected. Correctness is checked by the
   parity job, which compares vLLM and HF token log-probs on the same sequences.
3. **The forward check inside `train.py`** (HF forward vs the chunked loss path) has never run on a real model.
4. **Health panel** (`health.py`) has never run on these models. Panel texts must be regenerated by each parent, so
   absolute values are comparable within a model only.
5. **Gemma 4 (all GPU paths)**:
   - Training uses the multimodal `Gemma4ForConditionalGeneration` class (`AutoModelForCausalLM` returns it), calling
     its text-only backbone. The forward check guards this.
   - The softcap is applied in fp32, while HF applies it in bf16 (tiny difference, inside the forward-check tolerance).
   - Default attention (sdpa) with 512-dim global heads may be slow or memory-hungry. The logit chunk is 2048 for the
     262k vocab.
   - **The `gemma4_k_eq_v` LoRA fix is derived from reading vLLM's code and is unverified.** The parity job must show
     adapter errors near the parent's before any Gemma generation is trusted.
   - `language_model_only=True` for Gemma 4 in vLLM 0.30 is untested (the worker retries without it if rejected).
   - The stop token `<turn|>` (id 106) is in `generation_config.eos_token_id`; vLLM should stop on it. Check the
     smoke rollouts' `finish` field.
   - Whether the empty-thought-channel generation prompt is the right "thinking off" format for this checkpoint
     comes from the template itself, but has not been checked behaviourally. Read the smoke conversations.
6. **Environment install** from `requirements-*.txt`: pins copied from 001's environments (`env/freeze-*.txt`), not
   installed from scratch. torch 2.14.1 may need the CUDA 13.0 index.
7. `download_model.py --copy_to`, the queue mode's long-running loop, `push_adapters_hf.py` without `--dry`, and
   `collect_results.py --hf_dataset`: written but not run end to end.
8. **`tail20` on a real engine.** Not run. Expected: requesting `logprobs=0` does not change the sampled tokens
   (verify on the smoke run as described in README step 9: same tag, two output files, with and without
   `--no_tail20`). Overhead should be small: a rank count over the
   vocabulary per sampled token, and no top-k decoding. The rank is from the raw distribution, which is unaffected by
   temperature and min-p.
9. The OpenRouter providers that return logprobs may have changed since 001. The judge retries and records
   `lp_missing`; check `weighted_fallbacks` in the analysis.
