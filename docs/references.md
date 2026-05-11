# References

External sources that shape how we work on this project. When CC reads this file alongside `CLAUDE.md` and `phase1_env_spec.md`, it should be able to understand *why* we make the choices we make.

> **How to read this, Mikey**
>
> You don't have to read all of this yet. The two subsections under each entry called **"Relevance to our project"** and **"Watch out for"** are the operative ones — those are the bits that tell us what to actually do (or not do) when we build something. The rest — the methodology details, the background context, the channel pointers — is for when you're older and want to dig into the underlying ideas.

## Inspirations and Methodology

### Yosh — Trackmania RL video

- **URL:** <!-- TODO Mike: URL -->
- **Title:** <!-- TODO Mike: title -->
- **Date:** <!-- TODO Mike: date -->
- **Relevance to our project:** Yosh's video is the methodological backbone of Phase 1. He trains a DQN agent on Trackmania using a small set of engineered observations (speed, heading error, lookahead points along the racing line) rather than raw pixels, and shows the agent progressing from random flailing to clean lap times. Our BeamNG env is a direct adaptation of his approach: same observation philosophy (engineered features, not images), same "progress along centerline" reward, same random-spawn trick to prevent overfitting to the start of the track.
- **Watch out for:** Yosh uses DQN with 6 discrete actions; we use PPO with continuous actions, so don't copy his epsilon-decay or discrete-action-space code patterns directly. He runs Trackmania through TMInterface to step physics faster than real time; BeamNG has no direct equivalent, so our wall-clock training will be slower per step than his — budget for it. His action frequency (10 Hz training / 30 Hz eval) is also different from ours (20 Hz everywhere) because BeamNG's physics need more settle time per action.

#### Summary of the methodology

Yosh trains a deep-RL agent (DQN) to drive Trackmania by feeding the network a small vector of hand-engineered numbers — speed, heading error, and a handful of lookahead points along the racing line — rather than raw screen pixels, on the argument that the relevant state of the car is fully captured by a dozen meaningful scalars and a CNN over pixels is just paying compute to rediscover them. His reward is the simplest thing that could work: distance moved along the centerline since the previous step, with no shaping terms, no smoothness penalty, and no explicit lap bonus. To prevent the agent from overfitting to the first corner of the track, every training episode begins at a random point along the centerline with randomized heading and starting speed — he describes this as the single highest-leverage change he made. He keeps two environments running in parallel: a noisy training instance with random spawns whose reward curve is treated as a search signal, and a clean deterministic evaluation instance starting at a fixed point that is used to decide which checkpoint is "best" — the two reward curves serve different purposes and he is careful not to conflate them. Exploration is scheduled rather than constant: epsilon (and later the entropy bonus, when he switches off pure DQN) starts high and decays over training, which prevents the agent from locking onto a mediocre strategy in the first few thousand steps while still letting it converge to deterministic driving by the end. To make wall-clock training feasible at all, he runs the game through TMInterface, a community Trackmania mod that exposes a Python-controllable interface to the simulator and lets him step physics faster than real time and deterministically — without it, a 500k-step run would take days instead of hours.

*Transcript:* a full transcript is kept locally for reference but is not redistributed in this repo (the words are Yosh's, not ours).

### Andrej Karpathy — CLAUDE.md tweet

- **URL:** https://x.com/karpathy/status/2015883857489522876
- **Date:** 2026-01-26
- **Relevance to our project:** Karpathy's short note on how he writes `CLAUDE.md` files is the source of our coding-principles section: minimum viable code, surgical changes, no speculative abstractions, every line traces back to a request. Whenever CC starts adding "flexibility we didn't ask for," the answer is: re-read this and Karpathy's tweet.
- **Watch out for:** Karpathy's notes are general agent-coding guidance, not RL-specific. Don't apply "minimum viable code" so aggressively that you skip the ML hygiene that protects long training runs — checkpointing, eval/train split, fixed seeds, and TensorBoard logging *are* minimum viable code for this project, not optional polish.

## Documentation

### BeamNG.tech docs

- **URL:** https://documentation.beamng.com/beamng_tech/
- **Relevance to our project:** Official docs for the research/automotive build of BeamNG. Authoritative reference for sensor specs, scenario setup, and vehicle physics. We'll hit this whenever BeamNGpy's docs are ambiguous about what the underlying simulator actually exposes.
- **Watch out for:** License-gated content. We have access through 2026-08-06; after that our copy of the docs may go offline. Save local snapshots of any pages we depend on before that date.

### BeamNGpy docs

- **URL:** https://beamngpy.readthedocs.io/
- **Relevance to our project:** Python client for BeamNG.tech — this is the actual API surface `beamng_env.py` will call. Key modules: `Scenario`, `Vehicle`, `Sensors`.
- **Watch out for:** The docs lag the library on newer features (especially sensor types added in recent releases). When in doubt, read the source on GitHub rather than trusting the rendered docs.

### Stable-Baselines3 docs

- **URL:** https://stable-baselines3.readthedocs.io/
- **Relevance to our project:** The RL library. We're on SB3 2.8. PPO, `EvalCallback`, `CheckpointCallback`, and `Monitor` are the parts we touch most.
- **Watch out for:** SB3's API changed meaningfully between v1 and v2. Old tutorials and Stack Overflow answers often use v1 patterns that no longer work. Always check the version on any code snippet before copying.

### Gymnasium docs

- **URL:** https://gymnasium.farama.org/
- **Relevance to our project:** The env interface we implement. We're on Gymnasium 1.2.
- **Watch out for:** `step()` returns a 5-tuple `(obs, reward, terminated, truncated, info)` — pre-Farama `gym` returned a 4-tuple. Code copied from older tutorials that uses the 4-tuple signature will silently break SB3.

## Related Projects

### TMRL (Trackmania Reinforcement Learning)

- **URL:** https://github.com/trackmania-rl/tmrl
- **Relevance to our project:** A full Trackmania RL framework with distributed training, replay buffers, and SAC implementations. The most mature open-source car-RL codebase out there. Worth reading when we hit a problem they've already solved.
- **Watch out for:** Their stack (distributed actors, replay buffer infra, SAC) is several engineering steps beyond what we're building. Borrow specific solutions to specific problems, don't copy their architecture wholesale.

### Yosh's videos (channel)

- **URL:** https://www.youtube.com/channel/UCh1zLfuN6F_X4eoNKCsyICA
- **Relevance to our project:** Beyond the one video summarized above, Yosh has a series of videos iterating on the same agent. Worth referencing when we're deciding what to try next.
- **Watch out for:** The videos are entertaining and intuitive, not rigorous. Yosh often handwaves the failure modes ("then I changed it and it worked better") without showing the data. Treat them as inspiration, not as a manual.

### PedroAI — Trackmania RL

- **URL:** <!-- TODO Mike: URL -->
- **Relevance to our project:** Another YouTuber doing Trackmania RL, with a different methodology (more focus on vision-based learning). Useful as a counterpoint to Yosh's engineered-input approach.
- **Watch out for:** Vision-based learning is roughly an order of magnitude more compute than engineered inputs. We're explicitly not going down that road in Phase 1, so treat PedroAI as "could we try this in the future" inspiration, not as a template for what we build now.
