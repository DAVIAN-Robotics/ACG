# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An **unofficial reproduction** of ACG (Action Coherence Guidance, ICRA 2026) on RoboCasa
with GR00T-N1-2B. ACG is a **training-free, test-time guidance** that improves action
coherence in flow-matching VLA policies. The upstream benchmarks (Isaac-GR00T, robosuite,
robomimic, robocasa, dexmimicgen) are vendored under `libs/` and pip-installed in-place;
the top-level of the repo is a thin experiment harness (run scripts + summarizers) built on
top of them. See `README.md` for install, `EXPERIMENT_NOTES.md` for the running experiment
log/handoff, and `PAPER_CODE_NOTES.md` for paper↔code cross-checks.

## Environment

- conda env **`acg`** (python 3.10, torch 2.7.1+cu128). Activate with:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate acg`
- Machine `reallab`: single RTX A5000 (24GB), Ubuntu 20.04. RAM is the binding constraint
  for rollouts, not VRAM.
- Required env vars for any rollout (the run scripts export these; set them if invoking
  the python entrypoint directly): `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (headless render),
  `HF_HUB_DISABLE_XET=1` (xet protocol hangs on this host), `MAX_NUM_EMBODIMENTS=32`,
  `WANDB_MODE=disabled`.

## Running rollouts (the core workflow)

Everything is a baseline (no-ACG) run followed by an ACG run over the same 24 RoboCasa tasks,
then a comparison of success rates.

```bash
# Recommended: resilient wrapper — runs baseline then ACG, auto-retries on worker
# crash, resumes completed tasks. Detach so it survives SSH/session close.
#   args: [n_rollouts=24] [num_batch_envs=2] [seed=123]
setsid bash run_resilient.sh 24 2 123 > /tmp/acg_$(date +%H%M%S).log 2>&1 < /dev/null

# Multi-seed (paper protocol = 24 trials × 3 seeds); seeds run sequentially (parallel OOMs):
setsid bash run_multiseed.sh "456 789" 24 2 > /tmp/acg_ms.log 2>&1 < /dev/null

# Single phase, one shot (no auto-retry, resume only):
bash run_robocasa_noacg.sh 24 2      # baseline
bash run_robocasa_acg.sh   24 2      # ACG
```

- **`num_batch_envs` (nbe) is a RAM knob, not just speed.** Each forked mujoco worker grows
  to ~10GB while stepping. `nbe=2` (~22GB peak) is the safe default on 24GB RAM; `nbe=3+`
  risks OOM-kill; `nbe=1` is safest/slowest. This is why the scripts default to 2, not the
  README's 8.
- Rollout worker crashes (BrokenPipe, mujoco/EGL) mid-run are **expected and intermittent**.
  `run_resilient.sh` recovers by re-launching with `experiment.rollout.resume=True`; completed
  tasks (one line per task in `results.jsonl`) are skipped. Don't treat a crash as a bug
  unless the resilient wrapper stops making progress.
- To stop a run, kill the **wrapper first** or it will restart the phase:
  `pkill -9 -f run_resilient.sh; pkill -9 -f rollout_with_robomimic; pkill -9 -f base_rollout`

### Results & comparison

```bash
python summarize_results.py                       # single-run baseline vs ACG table
python summarize_multiseed.py 24 123 456 789      # mean±std across seeds (paper format)
```

Results are permanent under
`outputs/DAVIAN-Robotics/<model>/rollout_results/n=<N>_seed=<S>{,_acg}/<run>/results.json(l)`
(`.jsonl` = one line per task, accumulated live; `.json` = final). Per-task videos live in
`.../<run>/videos/`. `/tmp/*.log` files are ephemeral (lost on reboot); results are not.

## Call-chain architecture

The entrypoint and config flow spans several `libs/` packages — this is the part that needs
multiple files to understand:

1. `run_*.sh` → `scripts/base_rollout.sh` → `libs/Isaac-GR00T-N1/scripts/rollout_with_robomimic.py`.
2. Config = a robomimic JSON (`libs/Isaac-GR00T-N1/robomimic_configs/robocasa_mg100.json`)
   overlaid with Hydra-style `key=value` args passed after `--config_add`. ACG is enabled
   purely via config args: `algo.guidance.name=acg algo.guidance.scale=3.0
   algo.guidance.skip_blocks=7,9,11`. Guidance defaults live in
   `libs/robomimic/robomimic/config/gr00t_config.py`.
3. The policy is GR00T-N1-2B: an eagle2 VLM backbone + a **DiT (cross-attention diffusion
   transformer) action head** at
   `libs/Isaac-GR00T-N1/gr00t/model/action_head/cross_attention_dit.py`.

### How ACG actually works (the mechanism)

Implemented in `libs/robomimic/robomimic/algo/guidance/acg.py`. During the flow-matching
denoising loop it computes a normal prediction and a **perturbed ("bad model") prediction**,
then extrapolates away from the perturbation (CFG-style):

```
pred = pred + (scale - 1) * (pred - pred_perturb)   # scale=1.0 disables ACG
```

The perturbation swaps the attention `processor` on selected DiT blocks
(`transformer_blocks[i].attn1.processor = ACGAttnProcessor2_0()`), runs a second forward,
then restores the original processors. `skip_blocks` selects which blocks.

### The `skip_blocks` / self-vs-cross-attention subtlety

The action-head DiT runs `interleave_self_attention=True`, so in `DiT.forward`
(`cross_attention_dit.py:274-290`) **odd-index blocks are self-attention**
(`encoder_hidden_states=None`) and **even-index blocks are cross-attention** (conditioned on
vision/language embeddings). All blocks are the same `BasicTransformerBlock`; type is decided
at runtime by whether `encoder_hidden_states` is passed — there is no separate self/cross class.

The paper says ACG applies to "self-attention layers 4–6"; the code says `skip_blocks=[7,9,11]`.
These are the **same target** under different indexing (self-attn ordinal vs absolute block
index): odd blocks 1,3,5,7,9,11 are self-attn layers 1..6, so blocks 7,9,11 = self-attn
layers 4,5,6. This is not a discrepancy — keep `[7,9,11]`. (Note: the actual RoboCasa
checkpoint config uses **16** DiT layers, not the code default of 12; the odd/even mapping and
this conclusion hold either way.)

## Environment-specific patches (do not "fix" these)

These are deliberate adaptations to this host, documented in `EXPERIMENT_NOTES.md §5`:

- **eagle2 backbone attention forced `flash_attention_2` → `sdpa`** in
  `libs/Isaac-GR00T-N1/gr00t/model/backbone/eagle2_hg_model/config.json` — flash-attn cannot
  build/load on Ubuntu 20.04 (GLIBC 2.31). This is the **only** code/config change vs upstream
  and is the leading suspect for the reproduction's systematic ~8.4%p baseline offset above
  the paper; verify before attributing gaps elsewhere.
- **Dataset is mostly stubs.** The 24 RoboCasa task hdf5s are ~35GB each (~840GB total).
  Rollout only reads `env_args`/shape metadata, so only 1 real hdf5 (PnPCounterToCab, 33GB)
  is kept; the other 23 are ~306KB stubs with identical `env_args`. Eval distribution is *not*
  affected. Dataset root: `libs/robocasa/robocasa/macros_private.py` `DATASET_BASE_PATH`.

## Reproduction status (as of 2026-07-01)

3-seed (123/456/789, n=24 each) complete: baseline **41.0% (±1.84)**, ACG **42.1% (±0.91)**,
gain **+1.2** (paper: 32.6 / 39.3 / +6.7). Direction reproduces (both new seeds show a small
positive gain, so seed-123's −0.2 was noise); **magnitude does not** (~1/5 of paper), because
our baseline sits well above the paper's and eats ACG's head-room. Full log: `EXPERIMENT_NOTES.md §8`.
