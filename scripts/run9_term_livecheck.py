"""run9 live check: deterministic rollout of an existing (weaving) policy under
the NEW env, on a separate port (25253), reading the env's OWN computed weave
term. Confirms: the new code path runs (no crash/NaN), the term FIRES on the real
weave (run8 read ~0 here), it's bounded, and the episode-MEAN info column
populates non-zero. Does NOT exercise the SAC training loop (that's the 7k smoke,
which needs port 25252 and must wait until mikey_run8_resume is stopped)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC
from envs.beamng_env import make_beamng_env, _shared

CKPT = "checkpoints/mikey_run8_resume/rolling_205000_steps.zip"
PORT = 25253


def main():
    home = os.environ.get("BEAMNG_HOME")
    model = SAC.load(CKPT, device="cpu")
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True)
    obs, _ = env.reset()
    per_step = []
    osc_steps = 0
    nan = False
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        wp = env._last_weave_penalty            # per-step term
        mean = info["weave_penalty"]            # episode mean (the new logged column)
        if not (math.isfinite(wp) and math.isfinite(mean) and math.isfinite(r)):
            nan = True
        if wp < 0:
            osc_steps += 1
        per_step.append((step, env._cur_centerline_dist, env._center_off,
                         len(env._rev_steps), wp, mean, r))
        if term or trunc:
            print(f"episode ended step {step}: reason={info['termination_reason']} "
                  f"arc={env._cur_centerline_dist:.0f}m", flush=True)
            break

    import statistics as st
    fired = [p for p in per_step if p[4] < 0]
    print(f"\n=== run9 term live-check (ckpt rolling_205000, n={len(per_step)} steps) ===")
    print(f"  NaN/inf seen: {nan}")
    print(f"  steps the term FIRED (weave_penalty<0): {osc_steps}/{len(per_step)}")
    if fired:
        print(f"  per-step weave_penalty when firing: mean={st.mean(p[4] for p in fired):+.4f} "
              f"min={min(p[4] for p in fired):+.4f} (bounded check)")
    print(f"  episode-mean weave column at end: {per_step[-1][5]:+.4f}  (run8 logged ~0 here)")
    print(f"\n  sample rows (step arc center_off rev_count weave_pen ep_mean reward):")
    for p in per_step[::max(1, len(per_step)//25)]:
        print(f"   {p[0]:>3} arc={p[1]:>5.0f} off={p[2]:>+5.2f} revs={p[3]} "
              f"wp={p[4]:>+6.3f} mean={p[5]:>+6.4f} r={p[6]:>+6.2f}")
    env.close()
    try:
        _shared["bng"].close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
