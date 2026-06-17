"""run16 decision probe: HOW does run15 die at T2? Deterministic eval at the current
run15 seg1 checkpoint, capture episodes that reach T2 (~500-583m), logging per step
through the T1->T2 section into T2. Settles WASH-WIDE (understeer -> inexperience ->
spawn curriculum) vs SPIN (oversteer -> fresh stability bug -> fix first), and whether
T2 is failed clean or arrived-at compromised out of T1.

Separate port 25253; run15 untouched on 25252. Env matches run15 (steer_rate=0.5,
steer_rate_hi=0.15, esc_min=0.1). Measure only."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import SAC
from envs.beamng_env import (make_beamng_env, _shared, ESC_BETA_DEAD, ESC_BETA_FULL,
                             TC_SLIP_DEAD, TC_SLIP_FULL, TC_MIN_THR)

PORT = 25253
CKPT = "checkpoints/mikey_run15_s1/best_model/best_model.zip"
N_EP = 12
MAX_STEPS = 700
T1_APEX, T2_APEX = 338, 596


def esc_factor(beta, prev_slip):
    esc = max(0.1, min(1.0, 1.0 - (beta - ESC_BETA_DEAD) / (ESC_BETA_FULL - ESC_BETA_DEAD)))
    return esc


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=PORT,
                          launch=True, headless=True, nogpu=True,
                          steer_rate=0.5, steer_rate_hi=0.15, esc_min=0.1)
    model = SAC.load(CKPT, device="cpu")
    print(f"loaded {CKPT}; deterministic; run15 env config; T1@{T1_APEX} T2@{T2_APEX}\n")

    reached = 0
    for ep in range(N_EP):
        obs, _ = env.reset()
        prev_app = 0.0
        rows = []
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            s = _shared["vehicle"].sensors["agent_state"]
            vel = s["vel"]; fdir = s.get("dir", (1.0, 0.0, 0.0))
            vh = math.hypot(vel[0], vel[1]); dn = math.hypot(fdir[0], fdir[1])
            vel_nose = ((vel[0]*fdir[0]+vel[1]*fdir[1])/(vh*dn) if vh > 0.3 and dn > 1e-6 else 1.0)
            obs, _, term, trunc, info = env.step(action)
            app = env._cur_steer
            rows.append(dict(step=step, arc=env._cur_centerline_dist, v=vh,
                             coff=env._center_off, app=app, dapp=app-prev_app,
                             beta=env._last_beta, slip=info["slip"],
                             head=info["heading_align"], vnose=vel_nose,
                             esc=info["esc_cut_frac"], rhi=info["steer_ratehi_frac"]))
            prev_app = app
            if term or trunc:
                break
        term = info["termination_reason"]
        maxarc = max(r["arc"] for r in rows)
        print(f"===== EP{ep}: {len(rows)} steps, max_arc {maxarc:.0f}m, term={term} =====")
        if maxarc < 340:
            print(f"   (died at {maxarc:.0f}m, before T1 exit -- T1 failure, skipping detail)\n")
            continue
        reached += 1
        # window through T1 exit -> T1->T2 straight -> into T2 (arc 300..end)
        win = [r for r in rows if r["arc"] >= 300]
        print(f"   {'step':>5}{'arc':>6}{'v':>5}{'coff':>6}{'app':>6}{'dapp':>6}{'beta':>5}{'slip':>5}{'head':>6}{'vnose':>6}")
        for r in win:
            fl = "*" if (r["dapp"]*rows[rows.index(r)-1]["dapp"] < 0 and abs(r["dapp"])>0.05) else " "
            print(f"   {r['step']:>5}{r['arc']:>6.0f}{r['v']:>5.1f}{r['coff']:>6.1f}{r['app']:>6.2f}"
                  f"{r['dapp']:>6.2f}{r['beta']:>5.0f}{r['slip']:>5.1f}{r['head']:>6.2f}{r['vnose']:>6.2f}{fl}")
        # classify the failure in the last 10 steps
        tail = rows[-10:]
        max_beta = max(r["beta"] for r in tail)
        max_coff = max(abs(r["coff"]) for r in tail)
        flips = sum(1 for i in range(1, len(tail)) if tail[i]["dapp"]*tail[i-1]["dapp"] < 0 and abs(tail[i]["dapp"])>0.1)
        mode = ("SPIN(oversteer)" if max_beta > 17 and flips >= 1 else
                "WASH-WIDE(understeer)" if max_coff > 4 and max_beta < 15 else "ambiguous")
        # arrival quality at T2 entry (~arc 496-540)
        ent = [r for r in rows if 480 <= r["arc"] <= 540]
        if ent:
            e = ent[0]
            print(f"   T2-ENTRY(@{e['arc']:.0f}m): v={e['v']:.1f} center_off={e['coff']:.1f} "
                  f"head={e['head']:.2f} beta={e['beta']:.0f}  ({'clean' if abs(e['coff'])<3 and e['head']>0.9 else 'COMPROMISED'})")
        print(f"   FAILURE @ {maxarc:.0f}m: max_beta={max_beta:.0f} max|coff|={max_coff:.1f} flips={flips} -> {mode}\n")

    print(f"reached T2 region in {reached}/{N_EP} episodes")
    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
