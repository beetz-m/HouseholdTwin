# The avocado as a probabilistic SDT node: photo, belief, peeling monitor

The avocado twin (`food/avocado.py`) is a USD part graph (`assets/food/<node>/whole.usdc`, cut states
`half_a/half_b`); every prim under `/Food/Parts` is an SDT node (`sdt:represents`, `sdt:class`, `sdt:partOf`,
`sdt:observability`). Three mechanisms sit on top of it.

| Layer of the node | File | Content |
|---|---|---|
| geometry, parts, physics | `whole.usdc`, `half_a.usdc`, `half_b.usdc` | the instance's anatomy (built from parameters) |
| part beliefs | `sdt.usda` | point beliefs per part (e.g. seed unobserved, mesocarp bruise unknown) |
| probabilistic belief | `sdt_belief.usda` + `belief/belief.npz` | distribution of every variable, particle set, observation history |
| hidden truth | `truth.usda` | simulator ground truth (never read by the robot) |

## 1. Photo -> node (`scripts/avocado_from_image.py`, `food/avocado_vision.py`)

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_from_image.py PHOTO.jpg [--length-mm 95] [--name my_avocado]
    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_from_image.py --watch inbox

Segmentation (colour clusters -> fruit-like blob -> GrabCut -> convex hull), white balance from the fruit's own
specular highlights, silhouette aspect ratio and neck taper (fit of the twin's meridian), skin L*, C*, hue, skin
relief, stem button. Every estimate carries a status (OBSERVED / ESTIMATED / PRIOR). The photo's observations are
applied to the register prior by the filter (section 2); the instance is built from the posterior means.
Output: `renders/vision/<name>_analysis.png`, `assets/food/<name>/` with all layers.

## 2. Probabilistic belief and Bayesian filtering (`food/avocado_belief.py`, `scripts/avocado_observe.py`)

The node's belief is a particle filter (4,000 particles). One particle is a complete candidate avocado: length,
diameter, neck, skin thickness, tubercle size, seed length / width / offset, ripeness stage, ripening rate, skin
friction, stem button, bruise flag. Derived per particle: mass (body of revolution of the twin's meridian + seed
ellipsoid, calibrated on the built twins, DERIVED), aspect ratio, penetrometer force and flesh modulus (register
ripeness model), squeeze stiffness (register FEM table), seed and flesh mass.

- **Prior:** the parameter register (`food/params.py` AVOCADO), i.e. the population of Hass fruit.
- **Prediction:** ripening at the particle's own rate (process noise ILLUSTRATIVE); a squeeze above the particle's
  bruise threshold (0.6 x penetrometer force) sets its bruise flag.
- **Measurement models** (noise levels ILLUSTRATIVE unless stated): photo shape (aspect sd 0.05, neck sd 0.05),
  skin colour (L*, C* against the colour chart, hue as green / not green), stem button, calliper (1 mm), scale
  (2 g + 3 % model), squeeze (stiffness at 4 N, 20 % model error), seed length after halving (1.5 mm), bruise
  inspection after cutting.
- **Resampling:** systematic, with a regularising jitter, when the effective sample size drops below half.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_observe.py assets/food/<node> \
        [--wait HOURS] [--photo IMG] [--mass G] [--length MM] [--squeeze K --force N] [--seed-length MM] [--bruise-seen yes|no]

`sdt_belief.usda` holds, per variable, `sdt:belief:<var>:mean/sd/p05/p50/p95/unit`, `:support`, `:hist_edges`,
`:hist_p`, plus `sdt:belief:observations` (history), `sdt:belief:ess` and the particle file.

**Validation** (`scripts/avocado_belief_demo.py`, avocado_4 with known truth, simulated noisy observations over four
days): ripeness 2.97 +- 0.79 after the first photo (truth 2.96), 4.07 +- 0.65 after 48 h (4.11), 4.66 +- 0.38 after
the squeeze (4.68), 4.89 +- 0.21 at 96 h (5.00); mass 197 +- 6 g after weighing (197); the length is narrowed by the
weighing alone (+-4.3 -> +-2.3 mm, through the size-mass dependency); the ripening rate stays near its prior (two
ripeness readings do not determine it). Figures: `assets/food/avocado_4/belief/timeline.png`, `posteriors.png`.

## 3. Peeling monitor (`food/peel_monitor.py`, `food/avocado_peel.py`)

Protocol: halve, remove the seed, peel each half; the half lies cut face down, one oblique top view, RGB-D.
The monitor registers the twin's expected peeled surface to the observed depth (trimmed ICP + height refinement,
x/y/yaw) and classifies every visible surface point: peeled ok / skin remaining (pebbled texture, darker, or
standing proud of the expected surface) / flesh missing (deeper than expected; hysteresis over the torn area) /
unjudged. Measures: peeled share, skin area and largest patch, missing area, lost volume, deepest gouge,
coverage; score 0..100 and verdict (success / rework / not judged) with reasons; written into the belief layer.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/peel_monitor_run.py VIEW.npz --instance assets/food/avocado_4 [--state STATE.usda] [--rgb-only]

Synthetic validation: `scripts/build_peel_set.py` (peeling outcomes as USD states: skin patches, gouges subtracted
from the flesh, ground truth), `scripts/render_peel_dataset.py` (Isaac Sim: RGB, depth, semantic segmentation, the
expectation at a perturbed pose), `scripts/eval_peel_monitor.py` (IoU, area errors, verdict accuracy). Results:
`assets/food/avocado_4/peel_set/eval/` (summary.json, panels.png, scatter.png). Thresholds are calibrated on the
twin's renders only and must be recalibrated on real peels.

Results on the 24 synthetic peels (6 clean, 6 skin patches, 6 gouges, 6 mixed; ILLUSTRATIVE, renders of one
instance): RGB-D verdict 22/24 (14 reworks found, 1 missed, 1 false alarm, 8 successes confirmed); skin remaining
IoU 0.46, precision 0.53, recall 0.69, area error 15 mm2; flesh missing IoU 0.53, precision 0.83, recall 0.59, area
error 40 mm2 (precision / recall over the samples that contain the defect / a prediction). RGB only: verdict 20/24,
skin IoU 0.32, flesh missing not detectable from colour (IoU 0.00): the depth channel carries the gouges.

## 4. Halving and pitting: emulator (`food/avocado_emulator.py`, `food/avocado_process.py`, `food/avocado_tasks.py`)

Protocol (afm_howto): cut lengthwise round the seed, twist the halves apart, remove the seed (spoon; knife strike
discouraged, farley2019). The emulator is a state machine on the node (schema: whole -> partly_scored / scored ->
halved / halved_torn -> seed out, seed coat on the seed or in the cavity). Each action checks its precondition, runs
the mechanics, writes the new USD states and returns an outcome with `measured` (what a robot senses) and `truth`.

| Action | Mechanics (ILLUSTRATIVE parameters in the register's 'processing' entries) | USD states written |
|---|---|---|
| `cut_round_seed(coverage_deg, F_max)` | puncture at the stem end; force = skin resistance x skin thickness + G_flesh(stage) x depth; blade stops on the seed (stop-depth profile); a force limit can stall the blade (partial cut) | `scored.usdc`: kerf (blade-thick slab in the cut plane, over the covered sector, down to the seed coat) carved from skin and flesh, cut faces labelled `avocado_cut_surface` |
| `twist_apart(T_max)` | torque needed = uncut flesh bridges (polar moment of the uncut annulus x flesh shear strength) + seed adhesion over half the seed; ripe fruit releases at the seed, unripe tears in the flesh (p_clean(stage)) | `half_a.usdc` (holds the seed), `half_b.usdc` (torn flesh as `avocado_flesh_damaged` holes in the cavity wall, torn bridge strip) |
| `remove_seed(method, F_max, T_max)` | spoon lever (0.3 x adhesion x area), knife twist (adhesion torque; flagged hazardous for a hand-held half), squeeze-out (ripe only; may bruise); spoon may gouge; seed coat stays in the cavity if the testa adheres to the flesh | `half_a_pitted.usdc` (seed removed; seed-coat sheets left on this half's cavity wall, covering the drawn remnant fraction; gouges), `seed.usdc` |

Designators (`food/avocado_tasks.py`): `CutRoundSeed` (= coraplex CuttingAction, technique HALVING), `TwistApart`,
`RemoveSeed`, with parameters, steps, pre / effects, failures (cue -> recovery). `run(name, emulator)` executes one and
returns the outcome and the belief observations derived from its measured part only: blade stop depths
(`seed_depths`), twist outcome and torque (`twist`), spoon force (`seed_lift_force`), knife torque
(`seed_twist_torque`), seed coat in the cavity (`testa_seen`), the extracted seed's length.

    ~/Documents/LabAutomationTwin/env_cad/bin/python scripts/avocado_process_demo.py assets/food/avocado_4
    for s in scored halves pitted; do OMNI_KIT_ACCEPT_EULA=YES ~/Documents/LabAutomationTwin/env_isaacsim/bin/python \
        scripts/render_process_states.py assets/food/avocado_4/process/ripe --quick --step=$s; done

Geometry edits (`food/avocado_peel.py` `carve`, `subset_faces`): manifold3d booleans; the edited meshes keep their
texture coordinates (the cut face gets the exact planar section mapping, other faces are interpolated in the nearest
original triangle), get per-corner normals (hard on the cut face), and lose material subsets that became empty
(RTX otherwise drops the mesh's material). On an 8 GB GPU shared with an interactive Isaac session the states are
rendered one scene per process (`--step`) at 1280 x 800.

Demo on avocado_4 (stage 2.96, seed 40.0 mm, testa adheres to the flesh): three scenarios (ripe / unripe with a
partial cut that leaves the halves stuck, is completed and then tears / knife twist). After pitting the belief holds
P(testa adheres to flesh) 0.96-0.97, ripeness 2.92 +- 0.36 (ripe; true 2.96) and 1.91 +- 0.45 (unripe; true 1.80),
seed length 38.2-41.4 +- 0.9 mm (true 40.0). The blade stop depths alone narrow the seed little, because they also
depend on the fruit's size; with a photo or a weighing they do more.
Not yet: contact physics of the knife and the twist in Isaac Sim (breakable joints carrying the bridge and adhesion
strengths), off-axis or tilted cut planes in the geometry (the mechanics accept them, the halves are the asset's).

## 5. The avocado SDT as the robot's belief state in CRAM (`cram/`)

The PR2 stands at the kitchen counter of the CRAM apartment, detects an avocado on a cutting board, creates the
avocado's SDT node, reasons about it, and halves and pits it with its own hands (`cram/avocado_pr2_demo.py`; CRAM is
imported, nothing in `~/WORK` is changed).

1. **Perceive.** ParkArms, MoveTorso, Navigate (the PR2 stands at x = 1.70 m: from 1.85 m it cannot hold the knife
   flat close to its body at board height), LookAt, `DetectAction(TYPES, Avocado)`. `Avocado(Fruit)`
   (`cram/avocado_world.py`) carries the node (`sdt`), created from the register prior; the head camera's image
   (`cram/camera/avocado_4_counter.png`, `cram/render_camera.py`) is analysed and folded in.
2. **Reason** (`cram/avocado_sdt.py`). Belief: what is it, how ripe, where is the seed. Task models: what can I do,
   what am I doing. Mental simulation (halving and pitting mechanics on avocados drawn from the belief, 1 ms
   geometry each): what happens if I halve and pit it now, should I wait, which force / torque limits.
   **Squeeze or not (value of information):** goal 'guacamole now or store it 2 days'; utility = quality of the
   pitted halves x mashability at preparation time (0 below stage 2.5, 1 from stage 4; ILLUSTRATIVE), waiting costs
   0.05. The squeeze force is the strongest of 4 / 3 / 2 N whose bruise risk under the belief is below 2 %
   (bruise threshold 0.6 x penetrometer force); a preposterior analysis (imagined avocados x imagined readings, the
   belief's own measurement model; sensor error grows as the force falls) gives the value of the squeeze.
3. **Act and monitor** (`cram/avocado_motions.py`; the fruit lies with the twin's cut plane horizontal, half_a below):
   - squeeze: left top grasp across the widest section, fingers set to the believed diameter, then narrowed by
     force / believed stiffness (finger joint angle -> pad gap read off the PR2 model; the kinematic simulation has no
     force control, the reading comes from the fruit model);
   - hold: left grasp from above at the neck, pads 2 cm above the cut plane;
   - cut round the seed: `CutRoundSeedAction` (a CuttingAction with its own tool path): the knife held flat in the
     cut plane (alignment frame body), its tip once round the seed outline of the belief means + 2 mm;
   - the knife is laid flat on the counter; the left hand pinches the lower half at the neck; the right hand takes
     the upper half from above, turns it 90 deg about the vertical (the cut-plane normal), lifts it off and puts it
     down cut face down;
   - spoon: the right hand takes the spoon from the counter (the gripper-to-spoon transform at the grasp is used for
     all later poses), the bowl goes down along the far side of the seed, under it, levers it up, lifts it towards
     the robot and puts it on the board.
   Forces, torques and outcomes come from the mechanics on the hidden true avocado (same random stream as the
   emulator, so the outcomes match the exported USD states); each step: imagined distribution -> measured ->
   verdict (as expected / unexpected P < 0.2 / surprising P < 0.05 or outside the 95 % band) -> belief update -> the
   world shows the new pieces (`cram/meshes/`, `cram/export_meshes.py`).

Run on avocado_4 (truth: stage 2.96, seed 40.0 mm, seed coat adheres to the flesh): the photo reads the dark skin
as riper (stage 3.97 +- 0.53); squeezing is worth it (3 N, P(bruise) 0.000, VOI +0.012, changes the choice in 8 % of
the imagined readings); measured 3.40 N/mm, as expected (2.05..3.97), ripeness 3.79 +- 0.52 (a weak cue: 20 % model
error); decision: store 2 days (0.77 vs 0.64), but the task asks for now; cut as expected; the twist tears the flesh
-> unexpected (P 0.07), ripeness 3.00 +- 0.38; after pitting 3.16 +- 0.35, seed 41.5 +- 0.9 mm, P(seed coat adheres to
the flesh) 0.96. Final scene: pitted half on the board with seed-coat remnants, upper half cut face down beside it,
seed on the board, knife on the counter, spoon in the right hand. 18 answers, 4 monitored steps.

The records (belief variables, answers, imagined outcomes, monitoring) are a CRAMERA query source ('avocado SDT')
with presets (What does the robot believe about the avocado? Was anything not as expected? ...):

    source ~/ros_ws/install/setup.bash; source ~/.virtualenvs/cram-env/bin/activate
    LIVE_VIZ_PORT=8766 CRAMERA_DATA=~/Documents/HouseholdTwin/cram/cramera_data cramera-live ~/Documents/HouseholdTwin/cram/avocado_pr2_demo.py
    # viewer: http://localhost:8711/?live=127.0.0.1:8766  (a separate port beside a running session on 8765)
    CORAPLEX_VISUALIZATION=none python ~/Documents/HouseholdTwin/cram/avocado_pr2_demo.py      # headless

Not yet: contact forces (kinematic simulation: the knife, fingers and spoon pass through the fruit, the mechanics
supply forces and outcomes); the robot's seed-centre estimate is the twin's nominal position (no seed offset);
torn flesh is not shown on the seed piece; the photo ripeness likelihood is not calibrated on renders.

## 6. Interactive kitchen in CRAMERA (`cram/avocado_kitchen.py`)

The same kitchen (apartment, PR2, cutting board, avocado, knife in the right hand, spoon on the counter) as a live
scene the operator drives from the CRAMERA query panel; the 3D view follows the robot, the plan graph shows each
step. The panel ('avocado kitchen') offers buttons, and typed requests are matched to them:

| kind | entries |
|---|---|
| commands (queued, executed with PR2 motions) | Detect the avocado; Squeeze the avocado; Wait two days; Cut round the seed; Twist the halves apart; Spoon out the seed; Reset the kitchen |
| questions answered at once | What is the object? How ripe is it? Where is the seed? What can you do now? What are you doing? |
| questions answered by mental simulation (queued) | Should you squeeze it first? Guacamole now, or store it? What happens if you halve it now? |
| views | the robot's answers, its belief, what it imagined, how the execution went, the command log, what I can ask or command (with availability) |
| typed phrasings | e.g. "squeeze it", "halve it", "open the avocado", "remove the seed", "pit it", "is it ripe", "what can you do", "start over" |

A command is checked against the robot's and the avocado's state and refused with the reason (e.g. "the halves
have to be apart first", "the knife is not in my hand"), otherwise queued; a worker thread executes the queue, and
every step is predicted, monitored and folded into the belief. "Wait two days" ripens the hidden avocado at its own
rate and the belief at each particle's rate: a later twist then usually comes out clean (clean halves are shown).
"Reset the kitchen" takes the knife up again, puts the spoon back and places a fresh avocado.

Commands live in a query attribute (`an(entity(cmd_x).where(cmd_x.issued == True))`): CRAMERA builds every preset to
word it, and only evaluating the query reads the attribute, so listing the presets fires nothing (tested). The
robot's SDT nodes are kept beside the annotations (`avocado_world.SDT_NODES`), because the world serialises every
annotation it adds or removes.

    source ~/ros_ws/install/setup.bash; source ~/.virtualenvs/cram-env/bin/activate
    LIVE_VIZ_PORT=8766 CRAMERA_DATA=~/Documents/HouseholdTwin/cram/cramera_data cramera-live ~/Documents/HouseholdTwin/cram/avocado_kitchen.py
    # viewer: http://localhost:8711/?live=127.0.0.1:8766
    CORAPLEX_VISUALIZATION=none python ~/Documents/HouseholdTwin/cram/avocado_kitchen.py --do detect,how_ripe,squeeze,cut,twist,spoon

## 7. A goal-directed agent: enforce goals, prevent risks, decide over the fruit's lifetime (`cram/avocado_agent.py`)

**Effects.** Goals to enforce: halves apart, seed out, cavity clean, flesh intact, edible at service (eating quality
>= 0.6), fresh at service (cut face not visibly brown). Effects to prevent: bruise, torn flesh, gouge, seed coat left,
hazardous knife strike into a held half (farley2019), the holding gripper within 10 mm of the blade's plane, spoiled,
browned. A mission names its servings and the bounds: P(goal) >= 0.95 (apart, seed out), 0.9 (cavity clean), 0.7
(edible), 0.8 (fresh); P(risk) <= 0.02 (bruise), 0.05 (spoiled), 0.2 (browned), 0.1 (seed coat left); the knife strike
and the blade clearance are safety rules (bound 0).

**Lifetime model** (`food/avocado_lifetime.py`, register 'lifetime' entries): ripening at the storage temperature
(counter 20 C, fridge 4 C; Q10 2.5 ILLUSTRATIVE), the ripe window (24-48 h at room temperature, SOURCED-VERIFY), the
decline after it and spoilage (ILLUSTRATIVE), browning of cut flesh dE = k sqrt(t) with treatment factors (none,
lemon, wrap, wrap + fridge, lemon + wrap + fridge, vacuum + fridge; ordering from atk_browning, factors ILLUSTRATIVE).

**Deciding.** Plan options: when to cut (now / at the serving), where to keep the whole fruit until then (counter /
fridge), seed removal (spoon / knife twist), peel the seed coat or not, how to keep a half for later. Each option is
predicted by mental simulation on 80 avocados from the belief (mechanics with the repairs the agent would make, then
the lifetime model up to each serving); the effects' probabilities are checked against the bounds (chance
constraints) and the best feasible plan by expected quality is chosen. When none is feasible the agent never breaks a
safety rule (it refuses rather than cut unsafely) and otherwise takes the fewest shortfalls. Before deciding it
squeezes when that is worth it (value of information, bruise-safe force); before cutting it plans the hold
(`plan_hold`: pinch the flank with the pads >= 10 mm above the blade, or press on the top).

**Executing and enforcing.** Cut (a gap is re-cut), twist (a stuck fruit is re-cut and twisted again), spoon, then
peel the seed-coat sheets while they are seen (enforces 'cavity clean'); a half kept for later is treated at once
(lemon: PR2 motion; wrap, vacuum, fridge, serving: emulated, no motion yet). A goal watch after every step re-predicts
the open goals from the updated belief and records goals at risk with the reason. At the end an effect audit compares
every goal and risk with the prediction (achieved / missed, prevented / occurred).

Results on avocado_4 (truth stage 2.96; the photo makes it look riper): 'Guacamole in two days' -> keep it whole in
the fridge 48 h, cut, spoon, peel: every goal achieved, every risk prevented. 'Half now, half tomorrow' and 'Guacamole
now' -> cut now, spoon, peel (kept half vacuum + fridge): apart, seed out, cavity clean (two peel passes), fresh,
no bruise / hazard / blade contact / spoilage / browning; missed: flesh intact (torn, gouged) and edible at service
(eating quality 0.31): the belief, misled by the photo, predicted P(edible) 0.71; the goal watch flagged it right
after the torn twist (P 0.13). The knife twist is always rejected by the safety rule although it scores higher.

Kitchen commands: Make guacamole now; Half now, half for tomorrow's breakfast; Guacamole in two days; questions What
are the risks? Why this plan? When is it best to eat?; views: the plans compared, the decision, the effect audit.
