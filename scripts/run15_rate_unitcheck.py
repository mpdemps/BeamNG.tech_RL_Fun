"""run15 speed-scaled steering-RATE unit check (no BeamNG): the rate taper, default-OFF
gate, and the corner-vs-reversal table. Mirrors the inline math in env.step().

Why a rate limit, not a magnitude cap: the fastest corners (T2 ~31, T7 ~34, T9 ~35 m/s)
sit IN the 33-36 m/s spin band and need ~0.6 lock, while the spin slam reaches ~0.9-1.0;
a magnitude cap can't separate them. But the spin is a REVERSAL (high |Δsteer|/step) and
a corner is SUSTAINED lock (low rate), so tightening the rate at speed blocks the slam
while letting corners hold their lock (they just reach it a touch slower)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import make_beamng_env, STEER_RATE_V_LO, STEER_RATE_V_HI

BASE = 0.5      # run13/run15 base steer_rate
HI = 0.15       # run15 high-speed rate floor


def rate(v, base, hi):
    if not (0.0 <= hi < base):
        return base
    if v >= STEER_RATE_V_HI:
        return hi
    if v > STEER_RATE_V_LO:
        return base - (v - STEER_RATE_V_LO) / (STEER_RATE_V_HI - STEER_RATE_V_LO) * (base - hi)
    return base


def main():
    print(f"STEER_RATE_V_LO={STEER_RATE_V_LO} STEER_RATE_V_HI={STEER_RATE_V_HI}; base={BASE} hi={HI}")
    ok = True

    e_on = make_beamng_env(random_spawn=False, steer_rate=BASE, steer_rate_hi=HI)
    e_off = make_beamng_env(random_spawn=False, steer_rate=BASE)
    assert e_on.steer_rate_hi == HI and e_off.steer_rate_hi == -1.0, "steer_rate_hi not wired"
    print(f"[ok] steer_rate_hi wired: on={e_on.steer_rate_hi} default(off)={e_off.steer_rate_hi}")

    print(f"\n speed   rate(on)   rate(off)   note")
    for v, note in [(15, "slow corners"), (25, "T8"), (27, "T3 / knee_lo"), (29, "taper"),
                    (31, "T2 / knee_hi"), (34, "T7"), (35, "T9 / spin band")]:
        print(f" {v:>5}   {rate(v,BASE,HI):>7.3f}   {rate(v,BASE,-1.0):>8.3f}   {note}")

    checks = [
        ("<=V_LO full rate", rate(27, BASE, HI) == BASE and rate(15, BASE, HI) == BASE),
        (">=V_HI floors to hi", abs(rate(31, BASE, HI) - HI) < 1e-9 and abs(rate(35, BASE, HI) - HI) < 1e-9),
        ("mid tapers", HI < rate(29, BASE, HI) < BASE),
        ("monotone non-increasing", all(rate(x / 2, BASE, HI) >= rate(x / 2 + 0.5, BASE, HI) for x in range(0, 80))),
        ("OFF(hi<0) => flat base", all(rate(v, BASE, -1.0) == BASE for v in range(0, 50))),
        ("bounded [hi,base]", all(HI - 1e-9 <= rate(v, BASE, HI) <= BASE for v in range(0, 60))),
    ]
    print()
    for name, cond in checks:
        print(f"[{'ok' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # corner reach vs reversal: at high-speed rate HI, how many steps to reach a lock,
    # vs how slow a full reversal becomes (the spin trigger).
    print(f"\n at high-speed rate {HI}/step:")
    print(f"   reach 0.6 lock (T7-grade) from 0: {0.6/HI:.0f} steps ({0.6/HI*0.05:.2f}s) -- corner entry, fine")
    print(f"   full reversal +0.6->-0.6 (1.2 swing): {1.2/HI:.0f} steps ({1.2/HI*0.05:.2f}s) -- vs ~3 steps at base 0.5 (the spin)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
