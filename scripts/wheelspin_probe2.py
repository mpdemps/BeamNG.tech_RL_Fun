"""READ-ONLY cruise check (completes the wheelspin validation, question #2).

Separate BeamNG instance/port (NOT run3's 25252). Drives the race Scintilla at a
GENTLE constant throttle so the wheels hook up and it reaches a steady rolling
cruise, then checks that Electrics["wheelspeed"] ~= ground speed (||State.vel||)
when NOT spinning -- i.e. the slip signal reads ~0 in normal driving.
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
        scenario = Scenario(MAP_NAME, "wheelspin_cruise")
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

        vehicle.sensors.poll()
        print("\n step | thr | wheelspeed | ground_spd |  slip  | ratio", flush=True)
        print("-" * 60, flush=True)
        for step in range(110):
            # Gentle throttle so the rear wheels hook up rather than spin; ease
            # off once rolling to settle into a steady cruise.
            thr = 0.18 if step < 70 else 0.10
            vehicle.control(throttle=thr, steering=0.0, brake=0.0)
            bng.step(3)
            vehicle.sensors.poll()
            ws = float(vehicle.sensors["electrics"].get("wheelspeed", float("nan")))
            gs = hspeed(vehicle.sensors["agent_state"]["vel"])
            slip = ws - gs
            ratio = ws / gs if gs > 0.5 else float("inf")
            # Only print every 5th step plus the steady tail, to keep it readable.
            if step % 5 == 0 or step >= 95:
                print(f" {step:4d} | {thr:.2f}| {ws:10.2f} | {gs:10.2f} | "
                      f"{slip:+6.2f} | {ratio:5.2f}", flush=True)
    finally:
        try:
            bng.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
