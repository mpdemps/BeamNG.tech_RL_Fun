"""READ-ONLY wheelspin signal validation (run4 design input). Separate BeamNG
instance on a separate port + userfolder so it does NOT touch the running run3.

Spawns the race Scintilla, floors throttle from a standstill (launch spin), then
keeps going to speed (cruise). Each step logs Electrics["wheelspeed"] (wheel-
rotation speed) vs ground speed (||State.vel|| horizontal). Answers:
  1. Does wheelspeed over-read vs ground speed under a floored launch? (burnout)
  2. Does wheelspeed ~= ground speed at steady cruise?
  3. RWD dilution: how strongly is wheelspeed elevated during the rear-wheel
     launch spin (averaging fine vs diluted -> need PowertrainSensor)?
Bonus: dumps a PowertrainSensor reading once (per-wheel, if it works).

Run via the tmux wrapper with BEAMNG_HOME set. Port 25300, NOT run3's 25252.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, State
from envs.beamng_env import (
    MAP_NAME, CENTERLINE, VEHICLE_MODEL, VEHICLE_PART_CONFIG, SPAWN_Z_OFFSET_M)

PORT = 25300
USERFOLDER = "/tmp/bng_probe_user"


def hspeed(vel):
    return math.sqrt(vel[0] ** 2 + vel[1] ** 2)


def main():
    home = os.environ.get("BEAMNG_HOME")
    if not home:
        raise SystemExit("Set BEAMNG_HOME.")

    bng = BeamNGpy("localhost", PORT, home=home, user=USERFOLDER,
                   headless=True, nogpu=True)
    bng.open(launch=True)
    try:
        scenario = Scenario(MAP_NAME, "wheelspin_probe")
        vehicle = Vehicle("probe", model=VEHICLE_MODEL,
                          part_config=VEHICLE_PART_CONFIG)
        vehicle.sensors.attach("agent_state", State())
        vehicle.sensors.attach("electrics", Electrics())
        pos = (CENTERLINE[0][0], CENTERLINE[0][1],
               CENTERLINE[0][2] + SPAWN_Z_OFFSET_M)
        scenario.add_vehicle(vehicle, pos=pos, rot_quat=(0, 0, 0, 1))
        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        try:
            bng.settings.set_deterministic(60)
        except Exception:
            pass

        # Bonus: try a PowertrainSensor for per-wheel data (guarded; optional).
        pt = None
        try:
            from beamngpy.sensors import PowertrainSensor
            pt = PowertrainSensor("pt", bng, vehicle,
                                  physics_update_time=0.01)
            print("[probe] PowertrainSensor attached", flush=True)
        except Exception as e:
            print(f"[probe] PowertrainSensor unavailable: {e!r}", flush=True)

        vehicle.sensors.poll()
        print("\n step | thr | wheelspeed | ground_spd |  slip  | ratio", flush=True)
        print("-" * 60, flush=True)

        pt_dumped = False
        for step in range(70):
            # Floor it the whole time: from a standstill this is the launch spin;
            # once it hooks up and rolls, the later steps are the cruise case.
            vehicle.control(throttle=1.0, steering=0.0, brake=0.0)
            bng.step(3)
            vehicle.sensors.poll()
            elec = vehicle.sensors["electrics"]
            st = vehicle.sensors["agent_state"]
            ws = float(elec.get("wheelspeed", float("nan")))
            gs = hspeed(st["vel"])
            slip = ws - gs
            ratio = ws / gs if gs > 0.5 else float("inf")
            tag = ""
            if step < 3:
                tag = "  <- launch"
            print(f" {step:4d} | 1.0 | {ws:10.2f} | {gs:10.2f} | "
                  f"{slip:+6.2f} | {ratio:5.2f}{tag}", flush=True)

            # Dump powertrain structure once, early (during the launch spin).
            if pt is not None and not pt_dumped and step == 2:
                try:
                    data = pt.poll()
                    print(f"[probe] PowertrainSensor.poll() keys: "
                          f"{list(data.keys())[:40]}", flush=True)
                    # Print any wheel-like entries with their values.
                    for k, v in data.items():
                        kl = str(k).lower()
                        if "wheel" in kl or "axle" in kl:
                            print(f"[probe]   {k} = {v}", flush=True)
                    pt_dumped = True
                except Exception as e:
                    print(f"[probe] powertrain poll failed: {e!r}", flush=True)
                    pt_dumped = True

        if pt is not None:
            try:
                pt.remove()
            except Exception:
                pass
    finally:
        try:
            bng.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
