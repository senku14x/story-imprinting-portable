# eval_data/

Our evaluation inputs (from experiment 001). These ARE part of this repository.

| File | Rows | What |
|---|---|---|
| `scripts_heldout2.jsonl` | 100 | Fixed 5-turn user scripts, the pilot's held-out confirmation set ("ho"; ids `ho_<scenario>_NN`). |
| `scripts_frozen.jsonl` | 100 | Fixed 5-turn user scripts, the pilot's discovery set. |
| `health_panel_prompts.jsonl` | 200 | Health-panel prompts only: 100 `chat` + 100 `scenario`. No replies. |

**Scripts.** These were written by the pilot (GLM-5.2, from the five released Bloom `bees_crows` seed scenarios:
backyard_bbq_bash, bathroom_tile_choice, family_dinner_prep, puppy_diet_concerns, weekend_rome_escape; 20 per
scenario). Fields: `script_id, scenario, turn1, turn2, turn3_prohibition, turn3_question, turn4, turn5`. Trigger
condition: turn 3 = `turn3_prohibition + " " + turn3_question`. Control: `turn3_question` only. Turns 1, 2, 4 and 5 are
identical in both conditions. sha256: frozen `958d50b6…`, heldout2 `8c888cb4…` (unchanged from 001).

**Health panel.** 001's panel file held Qwen3.6-27B parent replies, so it measured change around that one parent.
Only the prompts are shipped here. Every new model regenerates the replies with its own untrained parent
(`python -m si_pipeline.queue_evals --model_config <cfg> health_texts ...`: a `single` job at T 1 with up to 1536
tokens, as in 001). The output file `runs/health/panel_<cfg>.jsonl` is the panel passed to `si_pipeline.health`.
- `chat` (`hc000`–`hc099`): first user turns from the **HuggingFaceH4/no_robots** test split (license CC BY-NC 4.0;
  `src_id` = the dataset's prompt_id). They are held out from 006's replay pool.
- `scenario` (`hs000`–`hs099`): turn 1 of the evaluation scripts (`source` names the script).
