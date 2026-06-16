"""run14 ESC unit check (no BeamNG): the esc_factor formula, compose-with-TC,
default-OFF gate, and cut-throttle-only. Mirrors the inline math in env.step()."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import (make_beamng_env, ESC_BETA_DEAD, ESC_BETA_FULL,
                             TC_SLIP_DEAD, TC_SLIP_FULL, TC_MIN_THR)


def esc_factor(beta, esc_min):
    if esc_min < 1.0:
        return max(esc_min, min(1.0, 1.0 - (beta - ESC_BETA_DEAD)
                                / (ESC_BETA_FULL - ESC_BETA_DEAD)))
    return 1.0


def tc_factor(slip):
    return max(TC_MIN_THR, min(1.0, 1.0 - (slip - TC_SLIP_DEAD)
                               / (TC_SLIP_FULL - TC_SLIP_DEAD)))


def main():
    print(f"ESC_BETA_DEAD={ESC_BETA_DEAD} ESC_BETA_FULL={ESC_BETA_FULL}")
    ok = True

    # 1) env wires esc_min through (lazy connect -> __init__ does not touch BeamNG)
    e_on = make_beamng_env(random_spawn=False, esc_min=0.1)
    e_off = make_beamng_env(random_spawn=False)            # default
    assert e_on.esc_min == 0.1 and e_off.esc_min == 1.0, "esc_min not wired"
    print(f"[ok] esc_min wired: on={e_on.esc_min} default(off)={e_off.esc_min}")

    # 2) factor vs beta (ESC on, min=0.1)
    print("\n beta  esc_factor(on)  esc_factor(off)")
    for beta in (0, 5, 9, 12, 15, 18, 22, 30):
        f_on, f_off = esc_factor(beta, 0.1), esc_factor(beta, 1.0)
        print(f" {beta:>4}  {f_on:>13.3f}  {f_off:>14.3f}")
        if f_off != 1.0:
            ok = False; print("   !! OFF gate not 1.0")
    # explicit checks
    checks = [
        ("beta<=DEAD passes", esc_factor(9, 0.1) == 1.0 and esc_factor(0, 0.1) == 1.0),
        ("beta>=FULL floors", abs(esc_factor(22, 0.1) - 0.1) < 1e-9 and abs(esc_factor(30, 0.1) - 0.1) < 1e-9),
        ("mid tapers", 0.1 < esc_factor(15, 0.1) < 1.0),
        ("monotone non-increasing", all(esc_factor(b, 0.1) >= esc_factor(b + 1, 0.1)
                                        for b in range(0, 30))),
        ("OFF => 1.0 at high beta", esc_factor(40, 1.0) == 1.0),
        ("bounded [0.1,1]", all(0.1 - 1e-9 <= esc_factor(b, 0.1) <= 1.0 for b in range(0, 60))),
    ]
    print()
    for name, cond in checks:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # 3) compose multiplicatively with TC; cut-throttle-only (brake untouched)
    print("\n compose (requested thr=0.9): applied = thr * tc(slip) * esc(beta)")
    for slip, beta in [(0, 0), (0, 18), (6, 0), (6, 18)]:
        applied = 0.9 * tc_factor(slip) * esc_factor(beta, 0.1)
        print(f"   slip={slip:>2} beta={beta:>2}: tc={tc_factor(slip):.3f} "
              f"esc={esc_factor(beta,0.1):.3f} -> applied={applied:.3f}")
    # brake path: action[1] < 0 -> throttle=max(0,thr)=0 so esc multiplies 0; brake=-thr
    thr = -0.7
    throttle = max(0.0, thr); brake = max(0.0, -thr)
    applied = throttle * tc_factor(0) * esc_factor(30, 0.1)
    cut_only = (applied == 0.0 and brake == 0.7)
    print(f"[{'ok' if cut_only else 'FAIL'}] cut-throttle-only: action[1]=-0.7 -> "
          f"applied_throttle={applied:.2f}, brake={brake:.2f} (ESC cannot touch brake)")
    ok = ok and cut_only

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
