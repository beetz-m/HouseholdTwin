# Put things where they belong — plan of the conceptual CRAM layer (worked example, as built 2026-10-21)

Activity package: `~/Documents/HouseholdTwin/activities/put-away` (name `put-away`). Status: BUILT (P1-P9 on an
emulator twin; PR2 kitchen version open).

## 1. Research question (user: concept page, competence A, M3)
- Can the robot tell stray from in-place from licensed from unknown things, honour requests with their debts, and
  abstain instead of guessing? Exercises open-world binding, the climb, defeasible norms, requests and authority,
  restoration obligations, abstention (KR3 parts the fruit work leaves idle). Axioms: BE1-2, MP1-3, PL1/2 (T3),
  RQ1-6, RE1-RE*, TD1-3.
- Success: the four guard behaviours shown; over-tidy and guessing violations 0; baseline = "put everything at a
  class home" (implicit).

## 2. RWMS
- Robot: one arm, survey of the counter (emulated; the PR2 kitchen later).
- Objects: cup, milk carton, used plate, knife, unknown jar, coffee jar, sugar jar; hidden: true kind and state
  (dirty, opened, empty) behind a possibly fallible survey.
- Places: fridge, dishwasher, cupboard, bin, counter (capacity = PL1 hard constraint).
- Agents: resident (may overrule), guest (may ask).
- Boundary: counter, shelves/cupboard, fridge, dishwasher, bin. Problem: after breakfast, tidy the counter.

## 3. Twin
- Emulator (kitchen state: truth per object, places, people, requests); deterministic survey + an optional noise
  model (miss dirt, look-alikes). Time: ILLUSTRATIVE durations.

## 4. Vocabulary (ontology/belonging.ttl)
- Objects: SOMA Cup, Plate, Knife, Jar, MilkPack; :CoffeeJar, :SugarJar, :UnknownJar; :HouseholdObject (union).
- Places: SOMA Cupboard, Refrigerator, Dishwasher, TrashContainer, Countertop; :BelongingPlace.
- Tasks: :PuttingAway ⊑ soma:PuttingDown (object, destination); :Surveying (area); :Asking ⊑ QueryingTask +
  CommunicationTask (recipient, query — SOMA requires `query`, lint META caught it).
- States: belongs-at, prohibited-at, applicable-place, placement-verdict, misplaced, in-place, licensed,
  no-known-home, flagged, dirty, opened, is-a, context, seen-in, request, request-ended, request-conflict,
  obligation-open/-due/-discharged, asked.
- Closing event: DirtyIntoCupboard ⊑ ProhibitedAction. Failures: NoBelongingPlace, PlaceNotAdmissible,
  ProhibitedPlacement.

## 5. Actions
| schema | roles | telos |
|---|---|---|
| surveying | area (a countertop) | achieve (surveyed ?area) |
| putting-away | object (a household-object), destination (its destination) | achieve (at ?o ?d); prevent (at ?o ?d) :while (prohibited-at ?o ?d); prevent (dropped ?o) |
| asking | recipient (a person), query (a household-object) | achieve (asked ?r ?q) |

## 6. Plans
- tidy-after-breakfast (area): survey; with-failure-handling (no-belonging-place, place-not-admissible → continue)
  (achieve (at (each household-object) (its destination))).
- put-one-away (object): escapes designator- and manipulation-failures.

## 7. Failures
| failure | parent | handled | :violates |
|---|---|---|---|
| no-belonging-place | DesignatorFailure | tidy: continue (asked) | — |
| place-not-admissible | ManipulationFailure | tidy: continue (asked, not placed) | (at ?o ?p) |
| prohibited-placement | ClosingEventViolation | never (gate) | — |

## 8. Knowledge
- Rules (belonging.lisp): 6 norms (opened milk → fridge 0.9; empty milk → bin 0.95 + prohibited fridge; dirty
  plate/cup/knife → dishwasher 0.9 + prohibited cupboard; clean plate/cup → cupboard 0.8; coffee → cupboard 0.9;
  sugar → counter 0.9) and 3 request rules (resident licenses + obligation; scope end → due; guest conflict).
- Adapter: applicable place (most confident, not prohibited; BelongingStore), verdict (assess_placement).

## 9. Servants
| filler | servant | status |
|---|---|---|
| its destination | belonging-store (rules + selection) | commonsense (norm), recorded (told by a person); notes admissible vs adequate |
| a household-object / a person / a countertop | survey / people present | recorded |

## 10. Guards
stray → restore; in place → leave; licensed → leave + debt; no home → ask; guest conflict → ask; full place → ask
(never somewhere merely admissible, T3); dirty into cupboard → gate.

## 11. Scenarios
nominal; leave-the-coffee-out (resident); coffee-request-ends; guest-asks-to-leave-the-milk-out; dishwasher-full;
jar-answer (a told norm).

## 12. Evaluation
30 perturbed kitchens (seeds 1000-1029), scripted ground truth written independently of the rules; conditions:
perfect survey vs fallible survey (miss dirt 0.15, look-alikes 0.10, ILLUSTRATIVE). Measures: verdict accuracy,
over-tidy, guessing, request compliance, guest and full-place abstention, task success.
Result: perfect 100 % / 0 / 0 / 13/13 / 12/12 / 13/13 / 30/30 (implementation consistency); fallible 97.3 % / 0 /
3 / … / 25/30 (MEASURED on the emulator).

## 13. Ports
Workbench 8795.

## 14. Phase checklist
- [x] P1 scaffold   - [x] P2 ontology module   - [x] P3 schema library   - [x] P4 rules   - [x] P5 adapter
- [x] P6 scenarios + tests   - [x] P7 workbench   - [x] P8 evaluation   - [x] P9 records
- [x] P7b PR2 kitchen twin (activities/put-away-kitchen: stand-in trays, CRAMERA 8714/8771, workbench 8796)   - [ ] P8b belief over kind/state + probe-first (M4)
