"""run21 STEP 2 blend probe (offline, no BeamNG): verify the fade blend computes right and stays
bounded, and that EVAL forces beta=0.
  (1) beta schedule: 1 in warmup -> linear -> 0 after anneal_end, at the breakpoints.
  (2) blend math + bounds: applied = clip(beta*ctrl + (1-beta)*policy) at beta=1/0.5/0,
      exact at the endpoints, midpoint correct, always in [-1,1] (even with out-of-range inputs).
  (3) controller action from a mocked on-line state is bounded [-1,1].
  (4) EVAL = standalone policy: model.predict() never invokes the controller and is independent
      of beta/num_timesteps (the blend lives only in _sample_action, which eval doesn't call).
Loads run20's peak into BlendSAC to exercise the real class."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as E
from envs.blend_sac import BlendSAC, beta_at, blend_action
from envs.base_controller import BaseController
from data.raceline_builtin import RACELINE

CKPT = "checkpoints/mikey_run20/best_model/best_model.zip"
WARMUP, ANNEAL = 100_000, 400_000


def main():
    ok = True

    # (1) schedule
    print("(1) BETA SCHEDULE (warmup=%d anneal_end=%d):" % (WARMUP, ANNEAL))
    pts = [(0, 1.0), (WARMUP - 1, 1.0), (WARMUP, 1.0),
           ((WARMUP + ANNEAL) // 2, 0.5), (ANNEAL - 1, None), (ANNEAL, 0.0), (ANNEAL + 50_000, 0.0)]
    for t, exp in pts:
        b = beta_at(t, WARMUP, ANNEAL)
        tag = "" if exp is None else ("OK" if abs(b - exp) < 1e-6 else "  <-- MISMATCH")
        ok &= (exp is None or abs(b - exp) < 1e-6)
        print(f"    t={t:>7,}  beta={b:.3f} {tag}")

    # (2) blend math + bounds
    print("\n(2) BLEND MATH + BOUNDS:")
    ctrl = np.array([0.8, -0.6]); pol = np.array([-0.4, 0.9])
    for beta in (1.0, 0.5, 0.0):
        out = blend_action(beta, ctrl, pol); exp = np.clip(beta * ctrl + (1 - beta) * pol, -1, 1)
        good = np.allclose(out, exp) and np.all(np.abs(out) <= 1.0)
        ok &= good
        print(f"    beta={beta}: applied={np.round(out,3)} expect={np.round(exp,3)} {'OK' if good else 'FAIL'}")
    extreme = blend_action(0.5, np.array([5.0, -9.0]), np.array([9.0, -5.0]))  # out-of-range inputs
    bnd = np.all(np.abs(extreme) <= 1.0)
    ok &= bnd
    print(f"    clip guard (out-of-range inputs): {np.round(extreme,3)} bounded={bnd}")
    print(f"    beta=1 -> pure controller: {np.allclose(blend_action(1.0,ctrl,pol),ctrl)}; "
          f"beta=0 -> pure policy: {np.allclose(blend_action(0.0,ctrl,pol),pol)}")

    # (3) controller action from a mocked ON-LINE state, bounded
    ctl = BaseController()
    i = 60; t = np.array(RACELINE[i + 1][:2]) - np.array(RACELINE[i - 1][:2]); t /= np.linalg.norm(t)
    s = {"pos": RACELINE[i], "vel": (t[0] * 25, t[1] * 25, 0.0), "dir": (t[0], t[1], 0.0)}
    st, th = ctl.action(s["pos"], s["vel"], s["dir"])
    cbnd = abs(st) <= 1.0 and abs(th) <= 1.0
    ok &= cbnd
    print(f"\n(3) CONTROLLER action on-line @ 25m/s: steer={st:+.3f} thr={th:+.3f} bounded={cbnd}")

    # (4) eval = standalone policy (predict never blends, beta-independent)
    print("\n(4) EVAL = STANDALONE POLICY:")
    if not os.path.isfile(CKPT):
        print(f"    SKIP load-test: {CKPT} not found");
    else:
        model = BlendSAC.load(CKPT, device="cpu")
        calls = {"n": 0}
        sentinel = BaseController()
        orig = sentinel.action
        sentinel.action = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
        model.controller = sentinel; model.beta_warmup = WARMUP; model.beta_anneal_end = ANNEAL
        obs = np.zeros((1, 18), dtype=np.float32)
        model.num_timesteps = 0            # beta would be 1 during collection here
        a0, _ = model.predict(obs, deterministic=True)
        model.num_timesteps = 250_000      # beta would be 0.5 during collection
        a1, _ = model.predict(obs, deterministic=True)
        same = np.allclose(a0, a1); no_ctrl = calls["n"] == 0; bnd = np.all(np.abs(a0) <= 1.0)
        ok &= same and no_ctrl and bnd
        print(f"    predict identical across num_timesteps(0 vs 250k): {same}")
        print(f"    controller NOT called during predict: {no_ctrl} (calls={calls['n']})")
        print(f"    predict action bounded [-1,1]: {bnd}  ({np.round(a0[0],3)})")
        print(f"    _current_beta @0={model_beta(model,0):.2f} @250k={model_beta(model,250_000):.2f} "
              f"@400k={model_beta(model,400_000):.2f} (collection-time only)")

    print(f"\n=== BLEND PROBE: {'ALL PASS' if ok else 'FAILURES ABOVE'} ===")


def model_beta(model, t):
    model.num_timesteps = t
    return model._current_beta()


if __name__ == "__main__":
    main()
