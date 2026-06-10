"""Verify which Scintilla config ACTUALLY spawns -- read it back, do not assume.

The headless smoke test only proved "no error", which does NOT mean the race
config loaded: BeamNG silently falls back to the default (gts) config if the
part_config path is wrong. Mike's G14 eyes saw gts, not race. This reads the
ACTIVE part config back from the simulator via get_part_config() and classifies
it by mechanical parts that differ between race and gts:
    race: scintilla_coilover_F_race, scintilla_dash_race, brake_F_carbon, ...
    gts : scintilla_coilover_F_adaptive, scintilla_dash, ...

Part A tests the REAL env spawn path (scenario.add_vehicle, what run2 uses).
Part B runtime-spawns extra cars with alternative path FORMATS to find which one
actually yields race, in case A shows a silent gts fallback.

Run from repo root with BEAMNG_HOME set:
    python scripts/verify_part_config.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamngpy import Vehicle
from envs.beamng_env import (
    make_beamng_env, _shared, CENTERLINE, VEHICLE_MODEL, VEHICLE_PART_CONFIG,
    SPAWN_Z_OFFSET_M)

# Distinguishing slots: the CHOSEN part differs between race and gts.
#   scintilla_coilover_F -> _race (race) vs _adaptive (gts)
#   scintilla_dash       -> _race (race) vs scintilla_dash (gts)
KEY_SLOTS = ("scintilla_coilover_F", "scintilla_dash", "scintilla_brake_F",
             "scintilla_exhaust", "paint_design")


def _walk(node, mapping):
    """Collect {slot id -> chosenPartName} from the part-config tree."""
    if isinstance(node, dict):
        sid, cp = node.get("id"), node.get("chosenPartName")
        if sid is not None and cp is not None:
            mapping[sid] = cp
        for v in node.values():
            _walk(v, mapping)
    elif isinstance(node, list):
        for v in node:
            _walk(v, mapping)


def classify(cfg):
    m = {}
    _walk(cfg, m)
    key = {s: m.get(s) for s in KEY_SLOTS}
    coil = key.get("scintilla_coilover_F") or ""
    dash = key.get("scintilla_dash") or ""
    if "race" in coil or "race" in dash:
        verdict = "RACE"
    elif "adaptive" in coil or dash == "scintilla_dash":
        verdict = "GTS (fallback!)"
    else:
        verdict = f"UNKNOWN (chosen={key}, total_slots={len(m)})"
    return verdict, key


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")

    env = make_beamng_env(random_spawn=False, home=home, host="localhost",
                          port=25252, launch=True, headless=True, nogpu=True)
    env.reset()   # spawns ego via the real scenario path with VEHICLE_PART_CONFIG
    bng = _shared["bng"]

    print(f"\n=== env VEHICLE_PART_CONFIG = {VEHICLE_PART_CONFIG!r} ===\n", flush=True)

    # Part A: the actual env spawn path (scenario.add_vehicle -> prefab).
    try:
        cfg = _shared["vehicle"].get_part_config()
        verdict, key = classify(cfg)
        print(f"[A] SCENARIO-spawned ego (run2's real path): {verdict}", flush=True)
        print(f"    chosen_key_slots={key}", flush=True)
    except Exception as e:
        print(f"[A] get_part_config failed: {e!r}", flush=True)

    # Part B: runtime-spawn with alternative path formats, read each back.
    base = CENTERLINE[0]
    formats = ["vehicles/scintilla/race.pc", "race", "race.pc",
               "vehicles/scintilla/race"]
    print("\n[B] runtime-spawn format probe:", flush=True)
    for i, fmt in enumerate(formats):
        vid = f"probe{i}"
        try:
            tv = Vehicle(vid, model=VEHICLE_MODEL, part_config=fmt)
            pos = (base[0] + (i + 1) * 8.0, base[1] + (i + 1) * 8.0,
                   base[2] + SPAWN_Z_OFFSET_M)
            ok = bng.vehicles.spawn(tv, pos)
            cfg = tv.get_part_config()
            verdict, key = classify(cfg)
            print(f"    part_config={fmt!r:34s} spawn_ok={ok} -> {verdict}  "
                  f"chosen={key}", flush=True)
        except Exception as e:
            print(f"    part_config={fmt!r:34s} ERROR {e!r}", flush=True)

    try:
        bng.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
