"""GTS-vs-RACE scoping probe (read-only, no training, no committed-file changes).

Strategy step for the "one car that LAPs AND DRIFTs" pivot: measure the Scintilla GTS (road)
config vs the RACE config we've been training. For ONE config (CLI arg):
  1. CONFIRMS the active config via get_part_config read-back (partConfigFilename + chosen
     coilover/swaybar/tire parts).
  2. CONTROLLER DRIVE (env.step + base controller, floor straights): the proven lap path. Records
     TOP SPEED (peak) and the peak STEADY-STATE cornering lateral g sustained on the racing line
     (a_lat = speed * d(velocity-heading)/dt, gated to sp>10 m/s & sideslip<15 deg).
  3. SKIDPAD (local circle, bypass): turn immediately and hold a steady circle, sweeping steering
     to find the true grip CEILING (the controller lap only reaches the profile's conservative
     A_LAT target, so this measures higher).
  4. BREAKAWAY: power-on step steer from ~12 m/s -- progressive vs snappy.

Note: the car accelerates fine but `steering=0` drives it into the first corner and it crashes, so
every measurement either steers (controller / circle) or acts before the car travels far.

Usage:  python scripts/gts_vs_race_probe.py gts   |   ... race        (port 25253)
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import envs.beamng_env as be

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "gts"
PC = {"gts": "vehicles/scintilla/gts.pc", "race": "vehicles/scintilla/race.pc"}[CONFIG]
be.VEHICLE_PART_CONFIG = PC  # monkeypatch BEFORE make_beamng_env reads it at spawn
from envs.beamng_env import make_beamng_env, _shared, V_TARGET_PROFILE
from envs.base_controller import BaseController
from envs.speed_profile import V_MAX as PROFILE_VMAX

DT = 3.0 / 60.0  # 50 ms per bng.step(3)


def heading_of(v):
    return math.atan2(v[1], v[0])


def ang_diff(a, b):
    d = a - b
    while d > math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def read(veh):
    veh.sensors.poll()  # MUST poll to refresh cached sensor data (env does this every step)
    s = veh.sensors["agent_state"]
    vel = s["vel"]; sp = math.hypot(vel[0], vel[1])
    return s, vel, sp


def controller_drive(env, veh, ctrl, max_steps=2400):
    """Drive the base controller around (floor straights). Returns top speed + peak steady-state
    cornering lateral g + where/why it stopped."""
    u = env.unwrapped
    peak = 0.0; peak_arc = 0.0; best_al = 0.0; best_al_sp = 0.0; prev_h = None
    window = []; info = {}
    for step in range(max_steps):
        s = veh.sensors["agent_state"]; vel = s["vel"]
        st, th = ctrl.action(s["pos"], vel, s.get("dir", (1.0, 0.0, 0.0)))
        vt = float(V_TARGET_PROFILE[u._progress_idx])
        thr = 1.0 if vt >= PROFILE_VMAX - 0.5 else th     # floor straights, controller brakes corners
        _, _, term, trunc, info = env.step(np.array([st, thr], np.float32))
        s2 = veh.sensors["agent_state"]; vel2 = s2["vel"]; sp2 = math.hypot(vel2[0], vel2[1])
        if sp2 > peak: peak, peak_arc = sp2, float(u._cur_centerline_dist)
        h = heading_of(vel2); nose = heading_of(s2.get("dir", (1.0, 0.0, 0.0)))
        if prev_h is not None and sp2 > 10.0 and abs(ang_diff(h, nose)) < math.radians(12):
            window.append(abs(sp2 * ang_diff(h, prev_h) / DT))
            if len(window) > 6: window.pop(0)
            sm = sorted(window)[len(window) // 2]          # sustained, not a pre-spin spike
            if sm > best_al: best_al, best_al_sp = sm, sp2
        else:
            window.clear()
        prev_h = h
        if term or trunc:
            break
    return peak, peak_arc, best_al, best_al_sp, info.get("max_arc", 0.0), info.get("termination_reason")


def skidpad(bng, veh):
    """Local steady-circle skidpad: turn immediately (never drive straight into a corner), sweep
    steering levels, hold ~13-16 m/s, record peak steady-state lateral g (sideslip-gated)."""
    best = 0.0; best_sp = 0.0; best_steer = 0.0
    for steer in (0.25, 0.35, 0.50, 0.65):
        prev_h = None; window = []
        for i in range(160):  # ~8 s per level
            _, vel, sp = read(veh)
            thr = max(0.0, min(1.0, 0.15 + 0.12 * (16.0 - sp)))  # P-hold ~16 m/s
            veh.control(throttle=thr, steering=steer, brake=0.0)
            bng.step(3)
            s2, vel2, sp2 = read(veh)
            h = heading_of(vel2); nose = heading_of(s2.get("dir", (1.0, 0.0, 0.0)))
            if prev_h is not None and sp2 > 10.0 and abs(ang_diff(h, nose)) < math.radians(20):
                al = abs(sp2 * ang_diff(h, prev_h) / DT)
                window.append(al)
                if len(window) > 10: window.pop(0)
                sm = sorted(window)[len(window) // 2]
                if sm > best: best, best_sp, best_steer = sm, sp2, steer
            prev_h = h
    R = (best_sp ** 2 / best) if best > 0.1 else float("nan")
    return best, best / 9.81, best_sp, R, best_steer


def breakaway(bng, veh):
    """Power-on step steer. Build to ~12 m/s (fast, before reaching the corner), then step in
    steering+throttle and watch how abruptly nose yaw-rate builds. Returns peak yaw-rate + onset."""
    for _ in range(60):
        veh.control(throttle=0.7, steering=0.0, brake=0.0)
        bng.step(3)
        _, _, sp = read(veh)
        if sp >= 12: break
    prev_nose = None; yrates = []
    for i in range(80):  # ~4 s of step input (turns the car, so it won't run straight to the corner)
        s, _, _ = read(veh)
        nose = heading_of(s.get("dir", (1.0, 0.0, 0.0)))
        veh.control(throttle=0.85, steering=0.6, brake=0.0)
        bng.step(3)
        s2, _, _ = read(veh)
        nose2 = heading_of(s2.get("dir", (1.0, 0.0, 0.0)))
        if prev_nose is not None:
            yrates.append(abs(ang_diff(nose2, nose) / DT) * 180 / math.pi)
        prev_nose = nose2
    yrates = np.array(yrates)
    peak = float(yrates.max()) if len(yrates) else float("nan")
    onset = float("nan")
    if len(yrates) and peak > 1:
        half = np.where(yrates >= 0.5 * peak)[0]; full = np.where(yrates >= 0.99 * peak)[0]
        if len(half) and len(full): onset = (full[0] - half[0]) * DT
    return peak, onset


def main():
    home = os.environ["BEAMNG_HOME"]
    env = make_beamng_env(random_spawn=False, home=home, host="localhost", port=25253,
                          launch=True, headless=True, nogpu=True, steer_rate=0.5)
    env.reset(options={"spawn_idx": 0})
    bng = _shared["bng"]; veh = _shared["vehicle"]
    ctrl = BaseController()
    print(f"\n===== CONFIG: {CONFIG}  ({PC}) =====")

    # 1. read-back
    try:
        import re
        blob = str(veh.get_part_config())
        m = re.search(r"'partConfigFilename': '([^']*)'", blob)
        print(f"   partConfigFilename = {m.group(1) if m else '?'}")
        for key in ["scintilla_coilover_F", "scintilla_coilover_R", "scintilla_swaybar_F",
                    "scintilla_transaxle", "scintilla_differential_R"]:
            i = blob.find("'" + key + "/")
            mm = re.search(r"'chosenPartName': '([^']*)'", blob[i:i + 600]) if i > 0 else None
            print(f"   {key:28s} -> {mm.group(1) if mm else '?'}")
        for mm in re.finditer(r"'(tire_[FR]_[0-9x]+)/[^']*'.*?'chosenPartName': '([^']*)'", blob):
            print(f"   {mm.group(1):28s} -> {mm.group(2)}")
    except Exception as e:
        print(f"   get_part_config failed: {e}")
    print(f"   (definitive parts list in the .pc file: {PC})")

    # 2. controller drive: top speed + cornering grip
    peak, peak_arc, al, al_sp, max_arc, term = controller_drive(env, veh, ctrl)
    print(f"\n-- CONTROLLER DRIVE (floor straights) --")
    print(f"   TOP SPEED: peak {peak:.1f} m/s = {peak*3.6:.0f} kph (at arc {peak_arc:.0f} m)")
    print(f"   cornering grip on the line: {al:.2f} m/s^2 = {al/9.81:.2f} g (at {al_sp:.1f} m/s)")
    print(f"   drive ended: {term} at arc {max_arc:.0f} m of 4326 m")

    # 3. skidpad grip ceiling
    env.reset(options={"spawn_idx": 0})
    a, g, sp, R, steer = skidpad(bng, veh)
    print(f"\n-- GRIP CEILING (steady-circle skidpad) --")
    print(f"   peak sustained lateral accel: {a:.2f} m/s^2 = {g:.2f} g  (at {sp:.1f} m/s, R~{R:.0f} m, steer~{steer:.2f})")

    # 4. breakaway
    env.reset(options={"spawn_idx": 0})
    peak_yr, onset = breakaway(bng, veh)
    snap = "SNAPPY" if (onset == onset and onset < 0.3) else "progressive"
    print(f"\n-- BREAKAWAY (power-on step steer @ ~12 m/s) --")
    print(f"   peak nose yaw-rate {peak_yr:.0f} deg/s; onset 50->100% in {onset*1000:.0f} ms ({snap})")

    env.close()
    try: bng.close()
    except Exception: pass


if __name__ == "__main__":
    main()
