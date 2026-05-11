# References

External sources that shape how we work on this project. When CC reads this file alongside `CLAUDE.md` and `phase1_env_spec.md`, it should be able to understand *why* we make the choices we make.

## Inspirations and Methodology

### Yosh — Trackmania RL video

- **URL:** TODO — Mike to paste the YouTube link here.
- **Why it matters:** Yosh's video is the methodological backbone of Phase 1. He trains a DQN agent on Trackmania using a small set of engineered observations (speed, heading error, lookahead points along the racing line) rather than raw pixels, and shows the agent progressing from random flailing to clean lap times. Our BeamNG env is a direct adaptation of his approach: same observation philosophy (engineered features, not images), same "progress along centerline" reward, same random-spawn trick to prevent overfitting to the start of the track. Differences: we use PPO with continuous actions (vs. his DQN with 6 discrete actions) because BeamNG's physics reward smooth control, and we run on a higher-fidelity simulator that needs a different action frequency (20 Hz vs. his 10 Hz).

#### Transcript

> TODO — Mike to paste the transcript inline here. The transcript belongs in this file (not linked out) so future-CC has it available even if the video goes offline.

### Andrej Karpathy — CLAUDE.md tweet

- **URL:** TODO — Mike to paste the tweet link here.
- **Why it matters:** Karpathy's short note on how he writes `CLAUDE.md` files is the source of our coding-principles section: minimum viable code, surgical changes, no speculative abstractions, every line traces back to a request. Whenever CC starts adding "flexibility we didn't ask for," the answer is: re-read this and Karpathy's tweet.

## Documentation

- **BeamNG.tech docs** — https://documentation.beamng.com/beamng_tech/ — Official docs for the research/automotive build of BeamNG. License-gated content; we have access through 2026-08-06.
- **BeamNGpy docs** — https://beamngpy.readthedocs.io/ — Python client for BeamNG.tech. This is the actual API we'll be calling from `beamng_env.py`. Key modules: `Scenario`, `Vehicle`, `Sensors`.
- **Stable-Baselines3 docs** — https://stable-baselines3.readthedocs.io/ — The RL library. We use SB3 2.8. PPO, callbacks (especially `EvalCallback` and `CheckpointCallback`), and `Monitor` wrapper are the parts we touch most.
- **Gymnasium docs** — https://gymnasium.farama.org/ — The env interface we implement. We're on Gymnasium 1.2. Pay attention to the 5-tuple return from `step()` (obs, reward, terminated, truncated, info) — different from older Gym.

## Related Projects

- **TMRL (Trackmania Reinforcement Learning)** — https://github.com/trackmania-rl/tmrl — A full Trackmania RL framework with distributed training, replay buffers, and SAC implementations. We don't use it directly (BeamNG, not Trackmania, and we want a smaller surface area), but it's the most mature open-source car-RL codebase out there and worth reading when we hit problems they've already solved.
- **Yosh's videos (channel)** — TODO — Mike to paste channel link. Beyond the one transcript above, Yosh has a series of videos iterating on the same agent. Worth referencing when we're deciding what to try next.
- **PedroAI — Trackmania RL** — TODO — Mike to paste link. PedroAI is another YouTuber doing Trackmania RL, with a different methodology (more focus on vision-based learning). Useful as a counterpoint to Yosh's engineered-input approach.
