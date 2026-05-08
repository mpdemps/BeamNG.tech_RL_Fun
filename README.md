# BeamNG.tech_RL_Fun

Reinforcement learning experiments by **Mikey Dempsey, age 9**, working toward
training a self-driving car in [BeamNG.tech](https://beamng.tech/).

This repo starts with a 2D practice track (Gymnasium's `CarRacing-v3`) so Mikey
can learn the basics of training and watching an RL agent before moving to the
full BeamNG simulator.

## Files

- `practice_train.py` — trains a PPO agent with a CNN policy on `CarRacing-v3`
  for 100,000 steps and saves the result as `practice_racer.zip`.
- `practice_watch.py` — loads the saved model and opens a window so you can
  watch the AI drive.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install gymnasium[box2d] stable-baselines3[extra] tensorboard
```

## Train

```
python practice_train.py
```

In a second terminal, watch it learn:

```
tensorboard --logdir ./logs/
```

Then open http://localhost:6006.

## Watch

```
python practice_watch.py
```

## License

MIT — see [LICENSE](LICENSE).
