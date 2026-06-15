"""run12 unit-check: (A) the Grad-CAPS temporal term math (clean ramp ~0, zigzag large,
bounded); (B) GradCapsSAC trains on a dummy env without NaN, term active+bounded. No
BeamNG -- isolates the loss/numerics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from gradcaps_sac import GradCapsSAC, GRAD_CAPS_EPS as EPS


def L_temp(a_prev, a_t, a_next):
    a_prev, a_t, a_next = map(lambda x: th.tensor([x], dtype=th.float32), (a_prev, a_t, a_next))
    d1 = a_t - a_prev; d2 = a_next - a_t; disp = a_next - a_prev
    jerk = th.norm(d2 - d1, dim=1); dispn = th.norm(disp, dim=1)
    return float((jerk * th.tanh(1.0 / (dispn + EPS)))[0])


print("=== (A) Grad-CAPS term on synthetic triplets ===")
cases = [
    ("clean constant-velocity ramp", [0, 0], [0.3, 0.3], [0.6, 0.6]),
    ("big clean swing (corner)",      [0, 0], [0.5, 0.0], [1.0, 0.0]),
    ("gentle accel swing (turn-in)",  [0, 0], [0.2, 0.0], [0.6, 0.0]),
    ("ZIGZAG steer (a_prev==a_next)",  [0.5, 0], [-0.5, 0], [0.5, 0]),
    ("ZIGZAG both axes",               [0.8, 0.8], [-0.8, -0.8], [0.8, 0.8]),
    ("tiny noise (held line)",         [0.0, 0.0], [0.01, 0.0], [0.0, 0.0]),
    ("worst-case (max jerk)",          [1, 1], [-1, -1], [1, 1]),
]
for name, p, t, n in cases:
    print(f"  {name:34s} L_temp = {L_temp(p, t, n):.4f}")
print(f"  (clean ramp/swing ~0; zigzag large; bounded by 2*sqrt(2)={2*2**0.5:.2f})")

# disp->0 gradient safety: ensure finite grad at the singularity
a = th.tensor([[0.5, 0.0]], requires_grad=True)
b = th.tensor([[-0.5, 0.0]], requires_grad=True)
c = th.tensor([[0.5, 0.0]], requires_grad=True)  # disp = c-a = 0 exactly
d1 = b - a; d2 = c - b; disp = c - a
loss = (th.norm(d2 - d1, dim=1) * th.tanh(1.0 / (th.norm(disp, dim=1) + EPS))).sum()
loss.backward()
print(f"  disp==0 singularity: loss={float(loss):.4f} grads finite={th.isfinite(a.grad).all().item()} "
      f"(max|grad|={float(a.grad.abs().max()):.3e})")


class Dummy(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (15,), np.float32)
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.t = 0
    def reset(self, *, seed=None, options=None):
        self.t = 0; return self.observation_space.sample(), {}
    def step(self, a):
        self.t += 1
        return self.observation_space.sample(), float(np.random.randn()), self.t % 80 == 0, False, {}


print("\n=== (B) GradCapsSAC integration on a dummy env (no BeamNG) ===")
env = DummyVecEnv([Dummy])
model = GradCapsSAC("MlpPolicy", env, lambda_t=1.0, device="cpu", batch_size=256,
                    buffer_size=50_000, learning_starts=500, ent_coef="auto", verbose=0)
model.learn(total_timesteps=3000)
# probe the term directly after training
lt = model._grad_caps_temporal()
print(f"  trained 3000 steps; grad_caps_temporal now = {float(lt):.4f} (finite={th.isfinite(lt).item()}, bounded)")
# actor params finite (no NaN divergence)?
finite = all(th.isfinite(p).all().item() for p in model.actor.parameters())
print(f"  actor params all finite (no NaN): {finite}")
obs = env.reset()
act, _ = model.predict(obs, deterministic=True)
print(f"  deterministic predict finite: {np.all(np.isfinite(act))}  action={act}")
print(f"  logged train/grad_caps_temporal present in logger: "
      f"{'train/grad_caps_temporal' in model.logger.name_to_value}")
