"""run14 decisive test: the steering REVERSAL with THROTTLE (the real spin condition).

The first probe showed a +-0.9 steering reversal with throttle=0 does NOT spin the car
(min_head 0.90). But every real spin had near-FULL throttle during the reversal. So the
spin is reversal x throttle coupling (rear loaded, RWD, friction circle). This isolates
the variable that matters for the cap decision: at ~33 m/s with high throttle, does a
capped +-0.35 reversal still break the rear, or does only the +-0.9 slam?

For each (amplitude, throttle) in a small grid: build to ~33 m/s, slam +amp -> -amp while
holding that throttle, log slip + heading. steer_rate=0 (raw reversal, worst case).
If +-0.35 @ high throttle HOLDS while +-0.9 @ high throttle SPINS -> the magnitude cap
at 0.35 is the right tool and grip-safe. If +-0.35 @ high throttle also SPINS -> 0.35 is
too loose / the cap can't fix a throttle-coupled snap and the throttle axis needs it.

Port 25253. Measure only."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beamng_env import make_beamng_env, _shared

PORT = 25253
TARGET_V = 33.0
GRID = [(0.35, 1.0), (0.35, 0.5), (0.90, 1.0), (0.90, 0.0)]   # (reversal amp, throttle)
HOLD = 8


def speed():
    s = _shared["vehicle"].sensors["agent_state"]
    return math.hypot(s["vel"][0], s["vel"][1])


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=PORT, launch=True, headless=True, nogpu=True,
                          steer_rate=0.0)
    print(f"port {PORT}; reversal x throttle grid at ~{TARGET_V:.0f} m/s; steer_rate=0 (raw)\n")
    print(f"{'amp':>5} {'thr':>5} {'v0':>5} {'peak|slip|':>10} {'min_head':>9} {'verdict':>10}")

    for amp, thr in GRID:
        env.reset()
        for _ in range(70):
            env.step([0.0, 1.0])
            _shared["vehicle"].sensors.poll()
            if speed() >= TARGET_V:
                break
        v0 = speed()
        log = []
        for st in (amp, -amp):
            for _ in range(HOLD):
                env.step([st, thr])
                _shared["vehicle"].sensors.poll()
                log.append((speed(), env._last_slip, env._last_heading_align))
        peak_slip = max(abs(r[1]) for r in log)
        min_head = min(r[2] for r in log)
        verdict = "SPUN/LOST" if min_head < 0.5 else ("marginal" if min_head < 0.8 else "held")
        print(f"{amp:>5.2f} {thr:>5.2f} {v0:>5.1f} {peak_slip:>10.1f} {min_head:>9.2f} {verdict:>10}")
        for r in log:
            tag = "  +" if r is log[0] else ("  -" if r is log[HOLD] else "   ")
            print(f"        {tag} v={r[0]:4.1f} slip={r[1]:6.1f} head={r[2]:+.2f}")

    env.close()
    try: _shared["bng"].close()
    except Exception: pass


if __name__ == "__main__":
    main()
