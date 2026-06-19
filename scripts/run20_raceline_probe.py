"""run20 step-1 probe: compute the min-curvature racing line + its speed profile and VALIDATE
before we touch obs/reward. Checks (Mike's): (1) the line stays inside the track edges,
(2) a sane apex at T1 (out-in-out: wide entry, clip the inside, wide exit), (3) speed within
grip, and (4) a faster ideal lap than the centerline. Renders the line over the track map
(full loop + a T1 zoom) to docs/run20_raceline_map.png for visual review.

EDGE CAVEAT: get_road_network no longer returns the 4361 m racetrack (road IDs regenerate per
load; only fragments come back -- the run17 failure). So per-node edges are unavailable and we
use a CONSTANT-WIDTH corridor = centerline +/- 5.705 m (the saved average half-width). Flagged
in the report; if T1 width differs from average the apex may shift slightly."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.centerline_racetrack_builtin import CENTERLINE
from envs.racing_line import min_curvature_line, left_normals, resample_closed
from envs.speed_profile import compute_speed_profile, A_LAT_MAX

AVG_HALF_WIDTH = 11.41 / 2.0      # 5.705 m (constant-width fallback; real edges unavailable)
CAR_HALF_WIDTH = 1.0             # Civetta Scintilla ~1.9 m wide
SAFETY = 0.5                     # extra inset from the edge
MARGIN = CAR_HALF_WIDTH + SAFETY  # 1.5 m corridor inset
RESAMPLE_M = 3.0                 # uniform arc-length spacing (clean 2nd-diff curvature proxy)
RIDGE = float(os.environ.get("RL_RIDGE", "0.001"))  # weak alpha->0 pull off-corner (anti-drift)
T1_LO, T1_HI = 260.0, 470.0      # T1 arc window (apex ~344 m, exit ~394 m)


def lap_time(cum, track_len, v):
    ds = np.diff(np.append(cum, track_len))
    return float(np.sum(ds / np.maximum(v, 0.1)))


def main():
    cl = resample_closed(CENTERLINE, RESAMPLE_M)   # uniform spacing -> clean curvature proxy
    raceline, alpha, nrm, bound = min_curvature_line(cl, AVG_HALF_WIDTH, MARGIN, ridge=RIDGE)

    v_c, R_c, cum_c, tl_c, _ = compute_speed_profile([tuple(p) for p in cl])
    v_r, R_r, cum_r, tl_r, _ = compute_speed_profile([tuple(p) for p in raceline])
    v_c, R_c, cum_c = np.array(v_c), np.array(R_c), np.array(cum_c)
    v_r, R_r, cum_r = np.array(v_r), np.array(R_r), np.array(cum_r)

    print("=== run20 racing line probe (min-curvature, rolled-our-own bounded LSQ) ===")
    print(f"resampled to {RESAMPLE_M:.1f} m ({len(cl)} pts); ridge={RIDGE}")
    print(f"corridor: half-width {AVG_HALF_WIDTH:.3f} m (CONSTANT -- real edges unavailable), "
          f"inset {MARGIN:.1f} m -> |alpha| <= {bound[0]:.3f} m\n")

    # (1) inside the edges
    amax = np.abs(alpha).max()
    print(f"(1) INSIDE EDGES: max |alpha|={amax:.3f} m vs bound {bound[0]:.3f} m "
          f"({'OK, within corridor' if amax <= bound[0] + 1e-6 else 'VIOLATION'}); "
          f"at-bound points={int(np.sum(np.abs(alpha) >= bound - 1e-3))}/{len(alpha)}")

    # (2) sane apex at T1 (out-in-out). + alpha = LEFT/inside of the T1 left turn.
    m = (cum_c >= T1_LO) & (cum_c <= T1_HI)
    idxs = np.where(m)[0]
    apex = idxs[np.argmin(R_c[idxs])]              # TRUE apex = centerline tightest point
    # entry/exit ~45 m before/after the apex along arc
    ent = int(np.argmin(np.abs(cum_c - (cum_c[apex] - 45))))
    ext = int(np.argmin(np.abs(cum_c - (cum_c[apex] + 45))))
    side = "LEFT/inside" if alpha[apex] > 0 else "RIGHT/outside"
    oio = (np.sign(alpha[apex]) != np.sign(alpha[ent])) and (np.sign(alpha[apex]) != np.sign(alpha[ext]))
    print(f"\n(2) T1 APEX (true tightest @ arc {cum_c[apex]:.0f} m): apex pulled {side}")
    print(f"    alpha  entry(@{cum_c[ent]:.0f})={alpha[ent]:+.2f}  apex={alpha[apex]:+.2f}  "
          f"exit(@{cum_c[ext]:.0f})={alpha[ext]:+.2f} m  "
          f"-> {'OUT-IN-OUT OK' if oio else 'NOT out-in-out (check)'}")
    print(f"    radius centerline R={R_c[apex]:.1f} m  ->  raceline R={R_r[apex]:.1f} m "
          f"({'STRAIGHTER (good)' if R_r[apex] > R_c[apex] else 'tighter (check)'})")
    print(f"    corner speed  centerline {math.sqrt(A_LAT_MAX*R_c[apex]):.1f}  ->  "
          f"raceline {math.sqrt(A_LAT_MAX*R_r[apex]):.1f} m/s")

    # (3) speed within grip + (4) faster lap
    gmin_c, gmin_r = R_c.argmin(), R_r.argmin()
    print(f"\n(3) GRIP: tightest raceline R={R_r[gmin_r]:.1f} m @ arc {cum_r[gmin_r]:.0f} m "
          f"-> v_cap={math.sqrt(A_LAT_MAX*R_r[gmin_r]):.1f} m/s (A_LAT_MAX={A_LAT_MAX}); "
          f"v_target min/max={v_r.min():.1f}/{v_r.max():.1f} m/s")
    lt_c, lt_r = lap_time(cum_c, tl_c, v_c), lap_time(cum_r, tl_r, v_r)
    print(f"\n(4) IDEAL LAP: centerline {lt_c:.1f} s ({tl_c:.0f} m)  ->  raceline {lt_r:.1f} s "
          f"({tl_r:.0f} m)  = {lt_c - lt_r:+.1f} s ({(lt_c-lt_r)/lt_c*100:+.1f}%)")

    # --- plot: full loop + T1 zoom, raceline colored by speed ---
    edge_L = cl[:, :2] + AVG_HALF_WIDTH * nrm
    edge_R = cl[:, :2] - AVG_HALF_WIDTH * nrm
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(18, 9))
    for ax in (axf, axz):
        ax.plot(edge_L[:, 0], edge_L[:, 1], color="0.6", lw=0.7)
        ax.plot(edge_R[:, 0], edge_R[:, 1], color="0.6", lw=0.7, label="track edge (avg width)")
        ax.plot(cl[:, 0], cl[:, 1], "--", color="0.4", lw=0.8, label="centerline")
        sc = ax.scatter(raceline[:, 0], raceline[:, 1], c=v_r, cmap="viridis", s=8, zorder=3)
        ax.scatter(*raceline[apex, :2], c="red", s=120, marker="*", zorder=5, label="T1 apex")
        ax.scatter(cl[0, 0], cl[0, 1], c="black", s=60, marker="s", zorder=5, label="start")
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    axf.set_title(f"WCUSA racetrack: min-curvature racing line  (ideal lap {lt_r:.1f}s vs centerline {lt_c:.1f}s)")
    axf.legend(loc="best", fontsize=8)
    pad = 70
    axz.set_xlim(raceline[apex, 0] - pad, raceline[apex, 0] + pad)
    axz.set_ylim(raceline[apex, 1] - pad, raceline[apex, 1] + pad)
    axz.set_title(f"T1 zoom (apex arc {cum_c[apex]:.0f} m): out-in-out")
    fig.colorbar(sc, ax=axz, label="v_target (m/s)")
    out = f"docs/run20_raceline_map_ridge{RIDGE}.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
