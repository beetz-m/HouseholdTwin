# The egg as a probabilistic SDT node

Status: **Step 0 (item analysis) — proposal awaiting the user's confirmation.** Nothing below §0 is built yet.
Skill: `~/.claude/skills/food-sdt-stack` (the avocado stack, `docs/AVOCADO_SDT.md`, is the template).

## 0. Item analysis

### What exists (inventory 2026-10-21)

| Layer | Exists | Gap to avocado level |
|---|---|---|
| L0 register | `food/params.py` EGG, 36 entries (16 SOURCED, 12 ILLUSTRATIVE, 3 TYPICAL, 2 SOURCED-VERIFY, 2 CALIBRATED, 1 DERIVED) | ageing laws (Haugh unit, yolk index, pH, specific gravity vs days and temperature) only partly in `egg_open.py` (decline rates ILLUSTRATIVE); sensor models; best-before rules; peelability vs age |
| L1 anatomy, USD | `food/egg.py` (Hügelschäffer shape, shell layers, membranes, air cell, albumen layers, chalazae, yolk, germinal disc); `egg_0..7` built | none (anatomy accepted earlier) |
| L2 schema | `food/schema.py` EGG: grasp/lift/transport/place (EG1 corridor, H8) | crack, open, separate, boil, candle, float-test actions; freshness attributes |
| L3 belief | `food/sdt.py` EggBelief: grip-only (F_crack by polar angle, F_slip, micro-crack prior 3 %, H8 admissibility) | full-egg particle (age, storage history, shell, F_break, mu, yolk, air cell); time prediction; measurement models for photo, scale, candling, float test, held grip, strike outcome, opened contents |
| L4 perception | — | photo → node (size grade, shape index, shell colour, visible crack); monitor: shell fragments in the bowl, broken yolk (RGB-D; open states exist: `scripts/build_egg_open_states.py`) |
| L5 mechanics | `food/egg_crack.py` (strike on flat / rim / egg, open_up, fish_out), `food/egg_cook.py` (axisymmetric heat, CALIBRATED doneness), `food/egg_open.py` (crack curve, fragments, contents on a plate, HU/YI), `food/dynamics.py` EggTruth | emulator that writes each state as USD with `measured` vs `truth` (cracked, opened, separated, boiled, peeled) |
| L6 designators | `food/egg_tasks.py`: PickUpEgg, TransportEgg, CrackEgg, SeparateEgg, BoilEgg, ScrambleEggs, FrySunnySideUp (run on truth) | run on the emulator; observations for the belief |
| L7 lifetime | pieces (air cell growth k(T), HU/YI decline) | `EggClock`: ageing at storage temperature, best-before, quality class over time, peelability; cracked / cooked egg hold times |
| L8 CRAM | — (Isaac: `sim_egg_bench.py` grip and drop, `sim_egg_open.py`, `sim_garmi_egg.py` GARMI cracks into a bowl) | egg SDT node, PR2 world and motions, CRAMERA kitchen |
| L9 agent | — | missions over servings and the carton's lifetime |
| L10 microtheory | AICOR book ch. 23 "egg carefully", egg-pickup hybrid automaton axioms H1–H12 (ResearchSpace, read-only); the avocado microtheory was transferred *from* it | executable version in `theory/egg_*` like the avocado (Task A, Task B, ablations, axioms audit, learning, RL) |

### Worksheet

| Question | Egg answer (proposal) |
|---|---|
| Referent | hen egg, retail (EU size M/L, brown or white), register population as in EGG (kop Hisex, USDA grading manual) |
| Hidden state | age since lay + storage temperature history (→ air cell height, Haugh unit, yolk index, specific gravity, pH); shell thickness and direction-dependent break force F_break; invisible micro-crack (3 % prior); shell–pad friction; yolk position; mass is only partly hidden (weighing reveals it) |
| Hidden parts that matter | air cell (freshness, which end is blunt), yolk and vitelline membrane (breaks on shards), shell membranes (tear or "check") |
| Observation channels | photo: size, shape index, shell colour, visible cracks (OBSERVED), freshness: none from the outside (PRIOR); scale: mass → weight grade; candling: air cell height, cracks; float test: specific gravity → age class; grip held without slip / crack: bounds on mu and F_break; strike outcome (crack / check / shatter at speed v): F_break; opened contents: thick-albumen height → Haugh unit, yolk index → age; boil: doneness, peel quality |
| Task A (assess / handle without harm) | pick up and transport intact: grip corridor F_slip(m, mu, a) < F < F_crack(polar); the existing H8 belief generalised to the full particle; non-destructive freshness probes (candle, float) with value of information |
| Task B (transform on purpose) | crack into a bowl: strike corridor (cracks and tears the membrane, but no shatter, no shards in the bowl, yolk intact); then open over the bowl; the cut is a failure in Task A and the goal in Task B, as for the avocado |
| Further transforms | separate (yolk intact, whites free of yolk); boil to a target (calibrated); peel a boiled egg (easier when older: SOURCED-VERIFY) |
| Failure → cue → recovery | no crack → nothing seen → strike again +15 %; check (membrane intact) → no release → tear by hand; shatter → contents everywhere → clean, new egg; shards in the bowl → seen → fish out with a shell half; broken yolk → seen → use for scrambled, not fried; micro-crack → weak shell → leaker in the pot |
| Time processes | ageing at 6 °C vs 22 °C (air cell, HU, YI, specific gravity); best-before (EU: 28 days after lay; sell within 21 days — SOURCED-VERIFY, eu589/853); cracked raw egg: use at once; boiled egg in the fridge: days |
| Missions | "two soft-boiled eggs at 7:30" (runny yolk, no crack in the pot, peels clean); "omelette now" (3 eggs, no shell in the bowl, none spoiled); "cake: separate 3 eggs" (whites free of yolk); "use the carton well" (which egg first, FIFO by belief over age) |
| Safety rules (hard) | never use an egg past best-before for raw / soft preparations; never serve an egg that was cracked before cooking and kept; no strike with the egg over a person / off the worktop |
| VOI decision | candle or float before frying / poaching (freshness matters) vs use as is; weigh before a recipe needing an amount |
| Execution monitor | shell fragments in the bowl and broken yolk from RGB-D (above the bowl); for boiled eggs: peel quality (shell remaining, white torn) |
| Robot | PR2 in CRAMERA (like the avocado) for L8–L9; GARMI in Isaac stays the physics demo |

### Proposed layer plan

1. L0: ageing laws, sensor models, best-before, peelability into the register (with sources). Status counts reported.
2. L2: schema extended (crack, open, separate, boil, peel, candle, float; freshness attributes).
3. L3: `food/egg_belief.py` full-egg particle filter (supersedes the grip-only belief; the H8 queries stay as methods); convergence demo on egg_0..7 with known truth.
4. L5 + L6: `food/egg_emulator.py` wrapping the existing crack / open / cook models, writing USD states; designators rebound to the emulator.
5. L7: `food/egg_lifetime.py` (EggClock).
6. L4: photo → node (size, colour, cracks; freshness stays PRIOR) and the bowl monitor (shards, yolk) on a synthetic set.
7. L8–L9: PR2 egg kitchen in CRAMERA (own ports), agent with the missions above.
8. L10–L11: executable egg microtheory (Task A grip and carry intact, Task B crack into a bowl), report PDF.

## 1. Status and results (2026-10-21; every number from the JSON named)

| Layer | Status | Evidence |
|---|---|---|
| L0 register | 58 entries (`food/params.py` EGG; later entries drawn from a second stream so earlier instances keep their draws) | `status_counts("egg")` |
| L2 schema | freshness attributes; actions ReadCarton, Weigh, Candle, FloatTest, Strike/Open, FishOutShards, Separate, Boil, Peel | `food/schema.py` |
| L3 belief | full-egg particle filter, Liu–West resampling; 13 measurement models | held-out calibration below |
| L4 perception | photo → node; bowl monitor | evaluations below (renders only) |
| L5 emulator | `food/egg_emulator.py`: `measured` and `truth` kept apart; opened states written as USD with labelled shards / broken yolk | `test_reports/egg_process_3.json` |
| L6 designators | CheckFreshness, PickUpEgg, CrackEgg, SeparateEgg, BoilEgg + peel | `food/egg_tasks.py run()` |
| L7 lifetime | equivalent age on the pan2026 curves (4 / 25 °C), Q10 from weight loss | `test_reports/egg_lifetime.json` |
| L8–L11 | not started (PR2 egg kitchen in CRAMERA, agent, microtheory, report) | — |

**L3 calibration** (`test_reports/egg_belief_calibration.json`, 24 held-out eggs, seeds 1000–1023): share of steps
with the truth inside the 90 % band -- age 0.99, mass 0.99, air cell 0.90, Haugh unit 0.96, yolk index 0.86, albumen
pH 0.90, F_crack (equator) 0.94, float tilt 0.97. Strike outcome: predicted P(cracked open) 0.50 on average,
observed 14 / 24 cracked open, 10 / 24 intact.

**L7 lifetime** (`test_reports/egg_lifetime.json`): the pan2026 table (4 °C and 25 °C, days 0–30) is reproduced
exactly (residual 0). Air cell anchors: 6 °C / 28 d model 5.5 mm vs Grashorn "< 4 mm" (**discrepancy**, open);
22 °C / 25 d model 6.0 mm vs "> 6 mm".

**L4 photo → node** (`renders/egg/photos/eval/summary.json`, `scripts/eval_egg_photos.py`; 8 twin renders, side view on
an oak board, 4 white + 4 brown eggs): silhouette IoU against the projected true outline mean 0.975, min 0.959; shape
index MAE 0.019 (bias −0.014); shell colour 8 / 8; crack false positives 0 / 8. Asymmetry w/L biased low (−0.027):
ESTIMATED, not used as an observation. Segmentation: `segment_egg` (edge RANSAC ellipse + per-angle outline); the
avocado's colour segmenter failed on eggs (partial brown egg, leaks around white eggs). The belief's photo noise is
now `sensor_photo_si_sd` 0.025 (ILLUSTRATIVE; RMS 0.023 here).

**L4 bowl monitor** (`assets/food/egg_3/bowl_set/eval/summary.json`, `scripts/eval_egg_bowl_monitor.py`; 11 renders of
egg 3: fresh / middle / old × 0, 1, 3 shards, + 2 broken yolks): verdict 11 / 11; shard count exact 11 / 11; shard
IoU 0.947–0.982; broken yolk 11 / 11; yolk index MAE 0.008. Albumen height (→ Haugh unit) observed 0 / 11: clear
albumen is transparent to depth (also for real RGB-D sensors), so the Haugh unit is not observable with this sensor.
Shell colour is an input (OBSERVED from the photo). Thresholds ILLUSTRATIVE, set on these renders.

Fixes found on the way (2026-10-21):
- OmniSurface constant colours are linear; sRGB values passed directly washed the yolk out (saturation 0.47).
- Shards were placed inside the albumen / under the yolk; now curved shell pieces resting on the surface.
- The broken yolk was a smooth dome; now an irregular flat pool.
- The yolk geometry made the dome height = index × diameter on top of a 2.9 mm sink, so the rendered yolk measured
  ~0.1 above its stated index. Now height-above-plate / diameter = yolk index, the standard measurement.

### Not yet
- No real photo or real bowl image has been evaluated; all L4 numbers are on the twin's own renders.
- Haugh unit from the bowl needs another channel (albumen spread from RGB, or a probe).
- The air-cell anchor at 6 °C / 28 d disagrees with Grashorn (5.5 vs < 4 mm).
- Contact physics of the strike is the reduced-order crack model; strike calibration 14 / 24 vs predicted 0.50 is
  coarse.
- L8–L11 not started.
