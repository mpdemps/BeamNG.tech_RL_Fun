# West Coast USA — track reference

Generated from `data/centerline_racetrack_builtin.py` (BeamNG DecalRoad geometry, road ID 59564). Lap walked by arc distance.

- **Total lap length:** 4356 m (985 centerline points)
- **Corners:** 16 (sustained turn-rate > 6°/20 m, same-direction runs merged, total turn > 15°). Radius ≈ span / total-turn (mean radius through the corner); direction L/R = left/right for a forward lap.
- **Longest straight:** 1242 m into T8.
- **Speeds:** measured where the policy has driven it; the opening straight reaches ~115 kph (~32 m/s) before T1 (run8/run10 watches). Deeper straights not yet driven at clean pace → projected only.

## Lap sequence (by arc distance)

| arc start→end | feature | length / radius | dir | turn | notes |
|---|---|---|---|---|---|
| 294→394 m | **T1** corner (apex 338 m) | R≈55 m | L | 103° | |
| 394→496 m | STRAIGHT (before T2) | 102 m | — | — |  |
| 496→662 m | **T2** corner (apex 596 m) | R≈108 m | L | 88° | |
| 660→846 m | **T3** corner (apex 746 m) | R≈82 m | R | 130° | |
| 846→966 m | STRAIGHT (before T4) | 120 m | — | — |  |
| 966→1066 m | **T4** corner (apex 1004 m) | R≈55 m | R | 104° | |
| 1066→1248 m | STRAIGHT (before T5) | 182 m | — | — |  |
| 1248→1328 m | **T5** corner (apex 1274 m) | R≈35 m | L | 129° | |
| 1316→1378 m | **T6** corner (apex 1336 m) | R≈34 m | R | 104° | |
| 1378→1476 m | STRAIGHT (before T7) | 98 m | — | — |  |
| 1476→1702 m | **T7** corner (apex 1610 m) | R≈133 m | R | 97° | |
| 1702→2944 m | STRAIGHT (before T8) | 1242 m | — | — | longest straight |
| 2944→3080 m | **T8** corner (apex 2990 m) | R≈68 m | R | 115° | |
| 3080→3208 m | STRAIGHT (before T9) | 128 m | — | — |  |
| 3208→3340 m | **T9** corner (apex 3280 m) | R≈139 m | R | 54° | |
| 3340→3354 m | STRAIGHT (before T10) | 14 m | — | — |  |
| 3354→3434 m | **T10** corner (apex 3390 m) | R≈65 m | L | 70° | |
| 3434→3546 m | STRAIGHT (before T11) | 112 m | — | — |  |
| 3546→3612 m | **T11** corner (apex 3574 m) | R≈40 m | R | 96° | |
| 3598→3658 m | **T12** corner (apex 3610 m) | R≈51 m | L | 68° | |
| 3658→3670 m | STRAIGHT (before T13) | 12 m | — | — |  |
| 3670→3728 m | **T13** corner (apex 3696 m) | R≈186 m | R | 18° | |
| 3728→3768 m | STRAIGHT (before T14) | 40 m | — | — |  |
| 3768→3828 m | **T14** corner (apex 3790 m) | R≈167 m | R | 21° | |
| 3828→3982 m | STRAIGHT (before T15) | 154 m | — | — |  |
| 3982→4038 m | **T15** corner (apex 4000 m) | R≈76 m | L | 42° | |
| 4026→4102 m | **T16** corner (apex 4058 m) | R≈53 m | R | 83° | |
| 4102→294 m | STRAIGHT (before T1) | 548 m | — | — |  |

## Known landmarks (run8 geometry report)

Radii in the table are geometric estimates (span / total turn) and can differ from the original landmark names; positions are the alignment.

- **First corner — "R40" (T1)** at apex ~338 m: the corner every policy reaches off the opening straight (entry ~295 m).
- **694 m longest straight** — centerline idx 470–531 = **arc 1717–2405 m** — into the **"R252" kink** (~R252 is gentler than this detector's ~190 m threshold, so it sits inside the straight stretch, not flagged as a turn).
- **~508 m straight into "R109"** — the next straight (~arc 2405–2940 m) into the corner near T8 (apex ~2990 m).
- **"R15–R17" hairpin pair** at arc ~1300 m = **T5/T6** (apex ~1274 / 1336 m) — the tightest sequence.
- **"R143" median corner** ≈ T2/T9 (mid-radius corners).
- **Total length ~4356 m.**

Note: very gentle bends (radius above ~190 m, e.g. the R252 kink) are below the turn-detector threshold and are folded into the adjacent straight; the longest detected straight therefore spans the 694 m straight + R252 kink + 508 m straight as one ~1270 m run between T7 and T8.

See `docs/track_reference.png` for the labeled layout.
