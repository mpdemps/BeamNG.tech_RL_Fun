"""run16 braking-aware target-speed profile, computed once over the centerline.

Standard forward-backward racing-line speed pass on the CENTERLINE (no width/racing line
yet): pointwise corner limit from curvature, then a backward (braking) pass so the target
falls on the APPROACH over the braking distance, then a forward (accel) pass for feasible
fast-out. Single source of truth -- imported by both the env (reward + obs v_target) and
the offline probe. Pure geometry, no BeamNG.
"""
import math

# --- constants (run16; tune in the smoke per the calibration gate) ---
A_LAT_MAX = 12.0     # m/s^2 target cornering grip (~1.22 g; below the ~1.6 g measured ceiling)
A_BRAKE = 9.0        # m/s^2 braking decel for the backward pass (~0.9 g)
A_ACCEL = 6.0        # m/s^2 corner-exit accel for the forward pass (traction-limited)
V_MAX = 33.0         # m/s straight-line cap (the speed the car actually reaches)
CURV_SMOOTH_M = 14.0  # curvature smoothing window (m) -- the scale a corner is "felt"


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def compute_speed_profile(centerline,
                          a_lat_max=A_LAT_MAX, a_brake=A_BRAKE, a_accel=A_ACCEL,
                          v_max=V_MAX, curv_smooth_m=CURV_SMOOTH_M):
    """Return (v_target, R, cum_arc, track_length, kappa), each indexed by point.

    kappa is the SIGNED smoothed curvature (1/m, +/- for turn direction) — used by the
    obs curvature preview.

    v_target[i] is the braking-aware target speed (m/s) the car should be at when it is at
    CENTERLINE[i]. R[i] is the smoothed local radius (m). cum_arc[i] is arc length from
    point 0; track_length closes the loop.
    """
    n = len(centerline)
    cum = [0.0] * n
    for i in range(1, n):
        cum[i] = cum[i - 1] + _dist(centerline[i - 1], centerline[i])
    track_len = cum[n - 1] + _dist(centerline[n - 1], centerline[0])

    def seg_len(i):  # length of segment i -> i+1 (wraps)
        return _dist(centerline[i], centerline[(i + 1) % n])

    # local tangent (prev -> next), then signed per-segment curvature
    def yaw(i):
        p = centerline[(i - 1) % n]; q = centerline[(i + 1) % n]
        return math.atan2(q[1] - p[1], q[0] - p[0])
    yaws = [yaw(i) for i in range(n)]
    kappa_seg = [0.0] * n
    for i in range(n):
        ds = seg_len(i)
        kappa_seg[i] = _angdiff(yaws[(i + 1) % n], yaws[i]) / ds if ds > 1e-6 else 0.0

    # smooth curvature over a ~curv_smooth_m window (moving average over arc, wraps)
    half = max(1, int(round(curv_smooth_m / max(track_len / n, 1e-6) / 2)))
    kappa = [0.0] * n
    for i in range(n):
        acc = 0.0
        for j in range(i - half, i + half + 1):
            acc += kappa_seg[j % n]
        kappa[i] = acc / (2 * half + 1)
    R = [1.0 / max(abs(k), 1e-4) for k in kappa]

    # 1) pointwise corner limit
    v = [min(v_max, math.sqrt(a_lat_max * R[i])) for i in range(n)]

    # 2) backward (braking) pass, upstream: v[i] <= sqrt(v[i+1]^2 + 2*a_brake*ds).
    #    two loops around the ring so the closed loop converges.
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            nxt = (i + 1) % n
            ds = seg_len(i)
            v[i] = min(v[i], math.sqrt(v[nxt] ** 2 + 2 * a_brake * ds))

    # 3) forward (accel) pass, downstream: v[i] <= sqrt(v[i-1]^2 + 2*a_accel*ds_prev).
    for _ in range(2):
        for i in range(n):
            prv = (i - 1) % n
            ds = seg_len(prv)
            v[i] = min(v[i], math.sqrt(v[prv] ** 2 + 2 * a_accel * ds))

    return v, R, cum, track_len, kappa
