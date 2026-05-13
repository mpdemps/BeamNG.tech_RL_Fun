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

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 0

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 322 (interrupted)

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 283 (interrupted)

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_watch

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 10240

### Peak Eval Reward

-530.5219072728158

### Final Eval Reward

-530.5219072728158

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_watch`
- Best model: `checkpoints\real_10k_watch\best_model`
- Final model: `checkpoints\real_10k_watch\final.zip`

## real_10k_v2

### Date

2026-05-12

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 10000
- Completed timesteps: 10240

### Peak Eval Reward

-348.50212588835467

### Final Eval Reward

-348.50212588835467

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\real_10k_v2`
- Best model: `checkpoints\real_10k_v2\best_model`
- Final model: `checkpoints\real_10k_v2\final.zip`

## yaw_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_smoke`
- Best model: `checkpoints\yaw_smoke\best_model`
- Final model: `checkpoints\yaw_smoke\final.zip`

## yaw_smoke_v2

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_smoke_v2`
- Best model: `checkpoints\yaw_smoke_v2\best_model`
- Final model: `checkpoints\yaw_smoke_v2\final.zip`

## builtin_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\builtin_smoke`
- Best model: `checkpoints\builtin_smoke\best_model`
- Final model: `checkpoints\builtin_smoke\final.zip`

## zero_offset_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\zero_offset_smoke`
- Best model: `checkpoints\zero_offset_smoke\best_model`
- Final model: `checkpoints\zero_offset_smoke\final.zip`

## z_offset_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\z_offset_smoke`
- Best model: `checkpoints\z_offset_smoke\best_model`
- Final model: `checkpoints\z_offset_smoke\final.zip`

## zero_offset_v2

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\zero_offset_v2`
- Best model: `checkpoints\zero_offset_v2\best_model`
- Final model: `checkpoints\zero_offset_v2\final.zip`

## yaw_bias_smoke

### Date

2026-05-13

### Hyperparameters

- learning_rate: 0.0003
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- Target timesteps: 2000
- Completed timesteps: 2048

### Peak Eval Reward

-inf

### Final Eval Reward

-inf

### Watching Notes

(fill in after watching with watch_beamng.py)

### Mikey's Hypothesis for Next Run

(fill in after Mikey has watched)

### Artifacts

- TensorBoard log dir: `logs\yaw_bias_smoke`
- Best model: `checkpoints\yaw_bias_smoke\best_model`
- Final model: `checkpoints\yaw_bias_smoke\final.zip`
