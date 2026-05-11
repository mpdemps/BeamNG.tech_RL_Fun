# Training Runs Journal

One entry per training run, in chronological order. **Append, never overwrite.** This file is the record of what we tried, what happened, and what Mikey wants to do next. When we look back in six months wondering "why did we change the entropy coefficient?", the answer lives here.

Update this file *after* every training session, *before* starting the next one. If we skip a run's entry, the journal stops being useful.

Use descriptive run names that hint at what's different: `lr3e4_300k_v2`, not `PPO_5`.

---

## Run 0: example_template_run

(This is a template entry showing the format. Delete or ignore when reading; real runs start at Run 1.)

### Date

2026-05-11

### Hyperparameters

- learning_rate: 3e-4
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Changes since last run: N/A (first run)

### Timesteps

300,000

### Peak Eval Reward

(filled in after training: highest mean reward across all `EvalCallback` evaluations)

### Final Eval Reward

(filled in after training: mean reward at the last evaluation)

### Watching Notes

Plain-language description of what the agent looks like when we run `watch_*.py`. Does it drive straight? Does it brake into corners? Does it spin out in the same spot every lap? Mikey writes most of this.

### Mikey's Hypothesis for Next Run

What Mikey wants to change and why. Example: "It keeps under-steering on the second left. Maybe give it more turning power by making the steering action stronger." This becomes the change-set for Run N+1.
