"""Racing-line validation (READ-ONLY diagnostic; changes nothing, doesn't touch run24).
For the post-run24 'raise V_MAX + fix the lines' work. Reports:
  (1) per corner: does the line apex correctly (wide-in / apex-inside / wide-out)? alpha = the
      line's signed lateral offset from the road centerline (+ = left). For a left turn (kappa>0)
      the inside is +alpha; out-in-out wants apex on the inside, entry+exit on the outside. Flags
      corners the constant-width corridor compromised.
  (2) v_target profile: V_MAX, and every straight where v_target is CAPPED at V_MAX (grip-unlimited)
      with its length -> straight-line speed left on the table if V_MAX is artificially low.
  (3) grip ceiling context (A_LAT_MAX assumption vs the measured ~1.6g; V_MAX in kph).
Renders docs/run24_raceline_diagnostic.png. Pure geometry, no BeamNG."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.raceline_builtin import RACELINE
from data.centerline_racetrack_builtin import CENTERLINE
from envs.racing_line import resample_closed, left_normals
from envs.speed_profile import compute_speed_profile, A_LAT_MAX, V_MAX as PROFILE_V_MAX

AVG_HALF_WIDTH = 11.41 / 2.0


def main():
    rl = np.array(RACELINE, float)[:, :2]
    # reconstruct alpha: RACELINE was built on the 3m-resampled centerline + alpha*normal
    rs = resample_closed(CENTERLINE, 3.0)[:, :2]
    nrm = left_normals(rs)
    alpha = np.sum((rl - rs) * nrm, axis=1)            # signed lateral offset (+left)
    v, R, cum, tl, kappa = compute_speed_profile(RACELINE)
    v, R, cum, kappa = np.array(v), np.array(R), np.array(cum), np.array(kappa)
    n = len(rl)
    Vmax = float(np.max(v))

    print("=== (3) GRIP / SPEED CEILING CONTEXT ===")
    print(f"  V_MAX (profile straight cap) = {Vmax:.1f} m/s = {Vmax*3.6:.0f} kph")
    print(f"  A_LAT_MAX (cornering grip assumed) = {A_LAT_MAX} m/s^2 = {A_LAT_MAX/9.81:.2f} g "
          f"(memory: ~1.6 g measured ceiling -> cornering grip has headroom too)")
    print(f"  back-straight cap ~114 kph = {114/3.6:.1f} m/s ~ V_MAX -> the straight IS hitting V_MAX")
    print(f"  748hp RWD real top speed >> {Vmax*3.6:.0f} kph, so V_MAX={Vmax:.0f} m/s is an ARTIFICIAL "
          f"cap; measure actual top speed post-smoke to set the new V_MAX safely.\n")

    # (2) V_MAX-capped straights
    capped = v >= Vmax - 0.15
    print("=== (2) v_target CAPPED-AT-V_MAX STRAIGHTS (grip-unlimited; speed left on the table) ===")
    runs = []
    i = 0
    seg = np.diff(np.append(cum, tl))
    while i < n:
        if capped[i]:
            j = i
            while j < n and capped[j]:
                j += 1
            length = float(np.sum(seg[i:j]))
            runs.append((cum[i], cum[j % n], length))
            i = j
        else:
            i += 1
    runs.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in runs)
    print(f"  {len(runs)} capped runs, total {total:.0f} m ({total/tl*100:.0f}% of the {tl:.0f} m lap):")
    for a0, a1, L in runs:
        print(f"    arc {a0:5.0f}->{a1:5.0f} m   length {L:5.0f} m")
    print()

    # (1) per-corner apex analysis: corners = contiguous R < 120 m
    print("=== (1) PER-CORNER APEX ANALYSIS (out-in-out check) ===")
    corner = R < 120.0
    corners = []
    i = 0
    while i < n:
        if corner[i]:
            j = i
            while j < n and corner[j]:
                j += 1
            apex = i + int(np.argmin(R[i:j]))
            corners.append((i, apex, j - 1))
            i = j
        else:
            i += 1
    print(f"  {len(corners)} corners (R<120m). alpha + = LEFT of centerline; corridor bound +/-4.2m.")
    print(f"  {'#':>2} {'apex_arc':>8} {'R':>5} {'turn':>5} {'a_entry':>7} {'a_apex':>7} {'a_exit':>7} {'verdict':>22}")
    flagged = []
    for k, (e, ap, x) in enumerate(corners):
        ide = max(0, int(np.argmin(np.abs(cum - (cum[ap] - 40)))))
        idx = min(n - 1, int(np.argmin(np.abs(cum - (cum[ap] + 40)))))
        turn = "LEFT" if kappa[ap] > 0 else "RIGHT"
        inside = math.copysign(1, kappa[ap])               # +1 left turn -> inside is +alpha
        apex_inside = alpha[ap] * inside > 0.3              # apex sits toward the inside
        oio = (alpha[ide] * inside < 0.2) and (alpha[idx] * inside < 0.2) and apex_inside
        at_bound = abs(alpha[ap]) >= 4.0
        if oio:
            verdict = "out-in-out OK"
        elif apex_inside:
            verdict = "apex inside, entry/exit not wide"
        else:
            verdict = "APEX WRONG SIDE"
            flagged.append(k)
        if at_bound and not oio:
            verdict += " (at corridor bound)"
        print(f"  {k:>2} {cum[ap]:>8.0f} {R[ap]:>5.0f} {turn:>5} {alpha[ide]:>+7.2f} {alpha[ap]:>+7.2f} "
              f"{alpha[idx]:>+7.2f} {verdict:>22}")
    print(f"\n  compromised corners (apex wrong side): {flagged if flagged else 'none'}")

    # ---- map: raceline colored by v_target, apex markers, capped straights highlighted ----
    clf = np.array(CENTERLINE, float)
    fig, ax = plt.subplots(figsize=(13, 11))
    eL = rs + AVG_HALF_WIDTH * nrm; eR = rs - AVG_HALF_WIDTH * nrm
    ax.plot(eL[:, 0], eL[:, 1], color="0.8", lw=0.6); ax.plot(eR[:, 0], eR[:, 1], color="0.8", lw=0.6)
    ax.plot(clf[:, 0], clf[:, 1], "--", color="0.6", lw=0.6, label="centerline")
    sc = ax.scatter(rl[:, 0], rl[:, 1], c=v, cmap="viridis", s=7, zorder=3)
    cap_pts = rl[capped]
    ax.scatter(cap_pts[:, 0], cap_pts[:, 1], facecolors="none", edgecolors="red", s=20, lw=0.5,
               zorder=4, label="v_target capped @ V_MAX")
    for k, (e, ap, x) in enumerate(corners):
        col = "red" if k in flagged else "lime"
        ax.scatter(rl[ap, 0], rl[ap, 1], c=col, s=90, marker="*", zorder=6, edgecolors="black", lw=0.5)
        ax.annotate(str(k), (rl[ap, 0], rl[ap, 1]), fontsize=8, zorder=7)
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="v_target (m/s)")
    ax.set_title(f"run24 racing-line diagnostic: {len(corners)} corners, {len(flagged)} apex-compromised; "
                 f"V_MAX={Vmax:.0f} m/s, {total:.0f}m capped (green*=OK apex, red*=wrong, red o=V_MAX straight)")
    out = "docs/run24_raceline_diagnostic.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
