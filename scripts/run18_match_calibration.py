"""run18 anti-timid calibration (offline): add +W_MATCH*min(v, v_target)*align to the
run16/17 reward and pick W_MATCH. The cap min(v,v_target) is load-bearing: it makes the
match term FLAT above v_target, so it adds NO marginal incentive past target -> v* stays
where the over-speed penalty puts it (~v_target). Two checks (Mike's):
  (1) reward-vs-speed still peaks at v_target (rerun held-vs-floor WITH the term; v*~=vt).
  (2) the match term makes carrying speed beat the timid crawl (the per-step reward gap
      for driving at v_target vs crawling at 2.4 m/s must be large/salient).
Pure reward-form math over the speed profile -- no BeamNG."""
import bisect, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.centerline_racetrack_builtin import CENTERLINE
from envs.speed_profile import compute_speed_profile
from envs.beamng_env import W_PROG, W_OVER, OVER_SPEED_OFFSET

DT = 3.0 / 60.0; A_GRIP = 16.0; V_MAX = 33.0; CRAWL_V = 2.4


def rstep(v, vt, wm):                       # per-step reward (on-line, align=1)
    prog = W_PROG * (v * DT)
    over = max(0.0, v - (vt - OVER_SPEED_OFFSET))
    return prog - W_OVER * over * over + wm * min(v, vt)


def main():
    v_t, R, cum, track, _k = compute_speed_profile(CENTERLINE)
    n = len(v_t)
    def idx_at(a): return max(0, min(bisect.bisect_right(cum, a % track) - 1, n - 1))

    print(f"W_PROG={W_PROG} W_OVER={W_OVER} DT={DT}; progress slope/(m/s)={W_PROG*DT:.3f}\n")
    for WM in (0.05, 0.10, 0.15, 0.20):
        # --- check 1: v* (reward-optimal speed) at a sample of corners ---
        # v* solves, ABOVE vt: W_PROG*DT = 2*W_OVER*(v*-(vt-off)) [match term flat above vt]
        margin = W_PROG * DT / (2 * W_OVER)
        worst = 0.0
        for apex in (338, 1004, 1274, 1336, 2990, 3574):
            i = idx_at(apex); vt = v_t[i]; vstar = (vt - OVER_SPEED_OFFSET) + margin
            vgrip = math.sqrt(A_GRIP * R[i])
            worst = max(worst, vstar - vgrip)
        # numerically confirm the peak is at ~vt (scan)
        i = idx_at(338); vt = v_t[i]
        scan = max([round(0.5*k,1) for k in range(0,70)], key=lambda v: rstep(v, vt, WM))
        # --- check 2: per-step reward, drive@vt vs crawl@2.4 ---
        i = idx_at(150); vt_str = v_t[i]   # opening straight target (~33)
        drive = rstep(vt_str, vt_str, WM); crawl = rstep(CRAWL_V, vt_str, WM)
        gap = drive - crawl
        print(f"WM={WM:.2f}:  v* = vt+{margin:.2f} (peak-scan@T1: {scan:.1f} vs vt={vt:.1f}); "
              f"max(v*-vgrip)={worst:+.1f} {'OK' if worst<=0 else 'HOT'}")
        print(f"          per-step reward  drive@{vt_str:.0f}={drive:.2f}  crawl@2.4={crawl:.2f}  "
              f"GAP={gap:.2f}  (match share of drive-step: {WM*vt_str/drive*100:.0f}%)")
    print("\nGAP = per-step reward advantage of driving-at-target over crawling; larger = "
          "stronger pull off the timid optimum. v* must stay ~vt (cap working).")


if __name__ == "__main__":
    main()
