"""List BeamNG's available vehicle model codenames (for the Scintilla swap).

We must NOT hardcode a guessed model string. "Civetta Scintilla" is the display
name; the BeamNGpy model codename is what Vehicle(model=...) needs. This dumps
every available model and highlights anything matching civetta/scintilla.

Run from repo root with BEAMNG_HOME set:
    python scripts/list_vehicles.py
"""

import os

from beamngpy import BeamNGpy

home = os.environ.get("BEAMNG_HOME")
if not home:
    raise SystemExit("Set BEAMNG_HOME.")

bng = BeamNGpy("localhost", 25252, home=home, headless=True, nogpu=True)
bng.open(launch=True)
try:
    avail = bng.vehicles.get_available()
    # Some BeamNGpy versions wrap the model dict in a protocol envelope
    # {"type": ..., "vehicles": {...}}; unwrap to the real model dict.
    if isinstance(avail, dict) and set(avail.keys()) <= {"type", "vehicles"}:
        avail = avail.get("vehicles", avail)
    names = sorted(avail.keys()) if isinstance(avail, dict) else sorted(avail)
    print(f"\n=== {len(names)} vehicle models available ===", flush=True)
    for n in names:
        print(n, flush=True)
    print("\n=== matches for civetta/scintilla ===", flush=True)
    hits = [n for n in names
            if "scintilla" in n.lower() or "civetta" in n.lower()]
    for n in hits:
        meta = avail[n] if isinstance(avail, dict) else None
        print(f"  {n}   meta={meta}", flush=True)
    if not hits:
        print("  (none matched -- inspect the full list above)", flush=True)
finally:
    bng.close()
