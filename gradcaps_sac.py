"""Grad-CAPS SAC (run12): SAC with a policy-level action-smoothness regularizer in
the actor loss, to break the coupled steer-throttle limit cycle at its generator (the
policy). Adds LAMBDA_T * L_temp to the standard SAC actor loss, where L_temp is the
Grad-CAPS temporal term (displacement-normalized second difference / jerk of the
DETERMINISTIC mean action over a 3-step window), so large CLEAN swings are nearly free
and only zigzag is charged (preserves corner agility; resists over-smoothing).

Loss regularization, NOT a reward penalty (penalties get outvoted). One behavioral
change vs run11; TC + speed-scaled reference live in the env and compose unchanged.

train() is copied verbatim from stable_baselines3 2.x SAC.train() (so the core SAC
math is byte-identical) with ONE inserted block before actor_loss.backward(). The
Grad-CAPS term is computed on its own buffer-sampled 3-step triplets, decoupled from
the SAC minibatch, so the standard update is untouched.

Numerics: tanh(1/(||disp||+eps)) bounds L_temp to [0, 2*sqrt(2)] and saturates its
gradient as disp->0 (sech^2 crushes the 1/disp^2 blowup), so no div-by-zero / no
unbounded gradient. eps=1e-6.
"""
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

GRAD_CAPS_EPS = 1e-6


class GradCapsSAC(SAC):
    def __init__(self, *args, lambda_t: float = 1.0, gradcaps_batch: int = 256, **kwargs):
        self.lambda_t = lambda_t
        self.gradcaps_batch = gradcaps_batch
        super().__init__(*args, **kwargs)

    def _mu(self, obs: th.Tensor) -> th.Tensor:
        """Deterministic squashed mean action mu(s) in [-1, 1] (the behavioral output),
        with gradients (so the regularizer updates the actor)."""
        mean_actions, _, _ = self.actor.get_action_dist_params(obs)
        return th.tanh(mean_actions)

    def _sample_triplets(self):
        """Sample valid same-episode 3-step windows (obs_{t-1}, obs_t, obs_{t+1}) from
        the replay buffer. Returns (o_prev, o_t, o_next) tensors and a validity mask."""
        rb = self.replay_buffer
        ub = rb.buffer_size if rb.full else rb.pos
        if ub < 3:
            return None
        # i indexes obs_t; need i-1 >= 0 and i <= ub-1 (next_observations[i] valid)
        i = np.random.randint(1, ub, size=self.gradcaps_batch)
        env = np.zeros_like(i)  # single env (DummyVecEnv n_envs=1)
        # same-episode validity: transition i-1 and i must not be terminal; avoid the
        # ring-buffer write seam (pos-1, pos) when full.
        valid = (rb.dones[i - 1, env] < 0.5) & (rb.dones[i, env] < 0.5)
        if rb.full:
            valid &= (i != rb.pos) & (i != rb.pos - 1) & (i - 1 != rb.pos)
        mask = th.as_tensor(valid.astype(np.float32), device=self.device)
        o_prev = th.as_tensor(rb.observations[i - 1, env], device=self.device).float()
        o_t = th.as_tensor(rb.observations[i, env], device=self.device).float()
        o_next = th.as_tensor(rb.next_observations[i, env], device=self.device).float()
        return o_prev, o_t, o_next, mask

    def _grad_caps_temporal(self) -> th.Tensor:
        """Grad-CAPS temporal loss: ||d2 - d1|| * tanh(1/(||disp|| + eps)), masked-mean."""
        trip = self._sample_triplets()
        if trip is None:
            return th.zeros((), device=self.device)
        o_prev, o_t, o_next, mask = trip
        a_prev, a_t, a_next = self._mu(o_prev), self._mu(o_t), self._mu(o_next)
        d1 = a_t - a_prev
        d2 = a_next - a_t
        disp = a_next - a_prev
        jerk = th.norm(d2 - d1, dim=1)                      # ||change-in-change||
        dispn = th.norm(disp, dim=1)
        per = jerk * th.tanh(1.0 / (dispn + GRAD_CAPS_EPS))  # bounded; clean swing -> ~0
        return (per * mask).sum() / mask.sum().clamp(min=1.0)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # ===== verbatim from stable_baselines3 SAC.train(), with ONE inserted block
        # (marked GRAD-CAPS) before actor_loss.backward(). =====
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        gradcaps_losses = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # ===== GRAD-CAPS insert: policy-level action-smoothness on mu(s) =====
            l_temp = self._grad_caps_temporal()
            gradcaps_losses.append(l_temp.item())
            actor_loss = actor_loss + self.lambda_t * l_temp
            # ====================================================================

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/grad_caps_temporal", float(np.mean(gradcaps_losses)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
