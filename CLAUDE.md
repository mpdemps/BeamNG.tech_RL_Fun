# CLAUDE.md

Read this file at the start of every session, before suggesting anything.

## Project purpose

A reinforcement learning project run by **Mike** and his 9-year-old son
**Mikey**. We train RL agents — first on `CarRacing-v3` as practice, then on
**BeamNG.tech** as the real target.

Two real constraints drive the schedule:

- **Mikey's school AI project deadline.**
- **BeamNG.tech license expires 2026-08-06.** All Phase 1+ work must land
  before that date or we lose the simulator.

## Audience and tone

**Mikey is the product owner.**

- Explain every change in plain language a 9-year-old can follow. No jargon
  without a one-sentence translation.
- Mikey decides reward-function design and what to try next.
- Mike handles technical scaffolding (env setup, plumbing, debugging weird
  Python errors).
- When suggesting a change, frame it so Mikey can understand and approve
  *before* you start coding. "Here's what I want to do, here's why, here's
  what could go wrong — OK to proceed?"

## Coding principles (Karpathy-style)

- **Minimum viable code.** No speculative abstractions, no flexibility we
  didn't ask for. Three repeated lines beats a premature helper.
- **Surgical changes.** Touch only what the request needs. No drive-by
  cleanup, no renaming things that already work.
- **Goal-driven, not step-by-step.** State assumptions up front, surface the
  real tradeoffs, ask when uncertain instead of guessing.
- **Every line traces back to a request** from Mike or Mikey. If you can't
  point to who asked for it, don't write it.

## Stack

- Python **3.11** in `venv/`
- **Stable-Baselines3 2.8**
- **Gymnasium 1.2**
- **PyTorch 2.6** with **CUDA 12.4**
- GPU: **RTX 4070 Laptop**
- **BeamNGpy** arrives in Phase 1 (not installed yet).

## Constraints

- The repo is **public**. BeamNG.tech's terms require open-source
  publication of any BeamNG.tech-related code, so **everything we write must
  be MIT-compatible**. No GPL deps, no proprietary snippets.
- `tech.key` (BeamNG license) and `.env` (Anthropic API key) **must never be
  committed**. Both are in `.gitignore`. If either ever shows up in
  `git status`, **stop and tell Mike immediately** — do not commit, do not
  "fix" it silently.
- **TensorBoard runs on port 8765, not 6006.** Something on this machine
  blocks 6006. Use `tensorboard --logdir ./logs/ --port 8765` and open
  `http://localhost:8765`.
- **Always save checkpoints, not just the final model.** First run had policy
  collapse and the peak model was lost. Use SB3's `CheckpointCallback` (or
  equivalent) on every training run from now on.

## Project phases

- **Phase 0 — current.** `CarRacing-v3` practice. Goal: build familiarity
  with the SB3 train/watch loop, TensorBoard, checkpoints, and the rhythm of
  iterating on hyperparameters.
- **Phase 1.** BeamNG.tech via BeamNGpy. Lap-time agent on a map Mikey picks.
- **Phase 2.** Stunt agent with a custom reward function Mikey designs.
- **Phase 3.** Claude-as-coach: call the Anthropic API to critique training
  runs and suggest next experiments.

## Run journal — RUNS.md

Maintain `RUNS.md` with **one entry per training run**. Append, never
overwrite. Each entry:

- Date
- Run name (e.g. `lr3e4_300k_v2`)
- Hyperparameters changed since last run
- Peak reward
- Final reward
- Observations from watching the agent
- Mikey's hypothesis for what to try next

Update this file *after* every training session, before starting the next.

## Conventions

- **Co-author commits** with `Co-authored-by: Mikey Dempsey <mpdemps@gmail.com>`
  whenever a change came from Mikey's idea.
- **Descriptive run names**: `lr3e4_300k_v2`, not `PPO_5`. The name should
  hint at what's different.
- **Never auto-run training without confirming first.** A training run takes
  20+ minutes and burns real GPU time. Ask before launching.
