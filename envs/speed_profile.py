"""run16 braking-aware target-speed profile, computed once over the centerline.

Standard forward-backward racing-line speed pass on the CENTERLINE (no width/racing line
yet): pointwise corner limit from curvature, then a backward (braking) pass so the target
falls on the APPROACH over the braking distance, then a forward (accel) pass for feasible
fast-out. Single source of truth -- imported by both the env (reward + obs v_target) and
the offline probe. Pure geometry, no BeamNG.
"""
import math

# --- constants: GTS (road) config, runs 26+. Recalibrated down from the run25 RACE values to the
#     GTS's softer, lower limit (~0.8x race grip; street sport-plus tires + adaptive suspension +
#     open diff + detuned 5.0 V10). RACE values kept in comments for provenance. ---
A_LAT_MAX = 10.0     # m/s^2 target cornering grip (~1.0 g). RACE was 12 (~1.22 g); GTS skidpad
                     # measured ~0.8x the race grip, so scale the target down to match.
A_BRAKE = 7.0        # m/s^2 braking decel for the backward pass (~0.71 g). RACE 9: steel brakes +
                     # street tires bite less than the race carbon/slick setup.
A_ACCEL = 5.0        # m/s^2 corner-exit accel for the forward pass. RACE 6: detuned engine + OPEN
                     # diff (vs race LSD) spins the inside wheel sooner -> weaker traction-limited exit.
# V_MAX: clean mid-straight GTS launch measured 76 m/s (274 kph) and LEVELED OFF (true top speed),
# vs the race car's 87 m/s. Cap at 50 (66% of top), the same conservative fraction the race profile
# used (55 of 87 = 63%). The backward braking pass self-caps any straight too short to reach 50, so
# this only opens straights with room -- the gate verifies the controller still brakes in time.
# (obs[16] = v_target/V_MAX renormalizes on the new V_MAX -- fine, the residual trains fresh on GTS.)
V_MAX = 50.0         # m/s straight-line cap (RACE 55)
CURV_SMOOTH_M = 14.0  # curvature smoothing window (m) -- the scale a corner is "felt" (unchanged)

# GRIP-AWARE cornering (carried over from run25, scaled to GTS). The uniform A_LAT over-asks where
# the road is downhill/off-camber (the rear unloads -> less grip), which was the T11 spin (measured
# -23% grade at turn-in, by far the steepest corner; Mike confirmed off-camber by eye). Reduce A_LAT
# (a) PROPORTIONAL to the smoothed downhill grade everywhere, and (b) an explicit OFF-CAMBER cap over
# the T11 complex. Flat/uphill corners keep the full A_LAT_MAX.
K_DOWNHILL = 1.5     # A_LAT *= (1 - K_DOWNHILL * downhill_grade); proportional grip loss on descents
GRIP_FLOOR = 0.6     # min grip factor from the slope term alone (A_LAT >= 0.6*10 = 6.0)
T11_ARC = (3530.0, 3620.0)  # arc range of the T11 right-then-left downhill complex (apex ~3574 m)
A_LAT_T11 = 6.5      # m/s^2 hard cap in the T11 zone (off-camber flag). RACE 8: GTS-scaled.


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def compute_speed_profile(centerline,
                          a_lat_max=A_LAT_MAX, a_brake=A_BRAKE, a_accel=A_ACCEL,
                          v_max=V_MAX, curv_smooth_m=CURV_SMOOTH_M,
                          k_downhill=K_DOWNHILL, grip_floor=GRIP_FLOOR,
                          t11_arc=T11_ARC, a_lat_t11=A_LAT_T11):
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

    # run25 grip-aware per-point A_LAT. Smoothed longitudinal grade dz/d_arc (same window as
    # curvature); downhill (negative grade) reduces grip proportionally; the T11 arc zone gets an
    # explicit off-camber cap. Flat/uphill corners keep the full a_lat_max.
    has_z = len(centerline[0]) > 2
    grade_seg = [0.0] * n
    for i in range(n):
        ds = seg_len(i)
        dz = (centerline[(i + 1) % n][2] - centerline[i][2]) if has_z else 0.0
        grade_seg[i] = dz / ds if ds > 1e-6 else 0.0
    a_lat = [0.0] * n
    for i in range(n):
        g = sum(grade_seg[j % n] for j in range(i - half, i + half + 1)) / (2 * half + 1)
        factor = max(grip_floor, 1.0 - k_downhill * max(0.0, -g))   # downhill -> g<0 -> reduce
        a = a_lat_max * factor
        if t11_arc[0] <= cum[i] <= t11_arc[1]:                       # off-camber flag (manual)
            a = min(a, a_lat_t11)
        a_lat[i] = a

    # 1) pointwise corner limit (per-point grip-aware A_LAT)
    v = [min(v_max, math.sqrt(a_lat[i] * R[i])) for i in range(n)]

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
