"""run20: generate the offline min-curvature racing line and save it as data/raceline_builtin.py
(same (x,y,z) shape as CENTERLINE so the env machinery is reusable). Params are the ones Mike
approved: ridge 0.001, 1.5 m inset, constant-width corridor (centerline +/- 5.705 m avg half-
width; real per-node edges unavailable -- see scripts/extract_edges.py). Run once; the env
imports the result. Pure geometry, no BeamNG."""
import math, os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.centerline_racetrack_builtin import CENTERLINE
from envs.racing_line import min_curvature_line, resample_closed
from envs.speed_profile import compute_speed_profile

RESAMPLE_M = 3.0
HALF_WIDTH = 11.41 / 2.0
MARGIN = 1.5
RIDGE = 0.001
OUT = "data/raceline_builtin.py"


def lap_time(cum, tl, v):
    return float(np.sum(np.diff(np.append(cum, tl)) / np.maximum(v, 0.1)))


def main():
    cl = resample_closed(CENTERLINE, RESAMPLE_M)
    rl, alpha, nrm, bound = min_curvature_line(cl, HALF_WIDTH, MARGIN, ridge=RIDGE)
    v_r, R_r, cum_r, tl_r, _ = compute_speed_profile([tuple(p) for p in rl])
    v_c, _, cum_c, tl_c, _ = compute_speed_profile([tuple(p) for p in cl])
    lt_r, lt_c = lap_time(np.array(cum_r), tl_r, np.array(v_r)), lap_time(np.array(cum_c), tl_c, np.array(v_c))

    hdr = ['"""',
           "Offline min-curvature RACING LINE for the WCUSA racetrack (run20). Same (x,y,z) shape",
           "as CENTERLINE so the env's progress/lookahead/heading/curvature/v_target machinery",
           "reuses it directly. Rolled our own (envs/racing_line.py, scipy bounded LSQ; MIT-clean,",
           "NOT the LGPL TUM package).", "",
           f"Generated: {date.today().isoformat()}",
           f"Method: resample {RESAMPLE_M} m -> min-curvature within centerline +/- {HALF_WIDTH:.3f} m",
           f"        (CONSTANT-width corridor; real per-node edges unavailable), inset {MARGIN} m, ridge {RIDGE}",
           f"Points: {len(rl)}   max |lateral offset|: {np.abs(alpha).max():.2f} m",
           f"Ideal lap (A_LAT=12): {lt_r:.1f} s  vs centerline {lt_c:.1f} s  ({lt_c-lt_r:+.1f} s)",
           "",
           "Regenerate with: python scripts/build_raceline.py",
           '"""', "", "RACELINE = ["]
    body = [f"    ({x:.2f}, {y:.2f}, {z:.2f})," for x, y, z in rl]
    open(OUT, "w", encoding="utf-8").write("\n".join(hdr + body + ["]", ""]))
    print(f"wrote {OUT}: {len(rl)} pts, max|alpha|={np.abs(alpha).max():.2f}m, "
          f"ideal lap {lt_r:.1f}s (vs centerline {lt_c:.1f}s, {lt_c-lt_r:+.1f}s)")


if __name__ == "__main__":
    main()
