# Nantes carrot: research notes, step 2

The model is `food/carrot.py`, the hidden-state models are in `food/carrot_models.py`, and the register is `params.CARROT`.

## What it is
A carrot is a storage taproot. From the outside in:
1. **Periderm** (the skin).
2. Orange **secondary phloem**, the kitchen's 'cortex'. It holds most of the carotene, which peaks under the periderm and drops near the cambium (`per17`, `bar06`).
3. **Vascular cambium**.
4. Paler, lutein-rich **secondary xylem**, the 'core'.
5. **Primary xylem** at the centre.

Nantes types are bred for a small core, and the core-to-root ratio falls from crown to tip (`vob24`).

Lateral roots arise beside the xylem poles, which gives 4 rows on a diarch root (`diarch_rows`, SOURCED-VERIFY).

## Part tree (USD: /Food/Parts)
- Periderm, with GeomSubsets Shoulder / Body / RootTip.
- SecondaryPhloem.
- VascularCambium.
- SecondaryXylem.
- PrimaryXylem.
- Crown: Shoulder, PetioleStubs/Stub_k.
- RootTip.
- LateralRootScars (points).
- LenticelBands (points).

## Geometry and grading
- **Measured Nantes roots** (`con22`):
  - 'Presto': 135.4 g, 210.8 mm, 31.4 mm.
  - 'CPT': 110.4 g, 182.7 mm, 30.3 mm.
- **Twin check:** carrot_0 weighs 115.0 g.
- **EU 730/1999:**
  - Main crop at least 20 mm or 50 g.
  - Green tops: none in Extra; Class I allows up to 1 cm (roots ≤10 cm) or 2 cm (longer roots).
- **Peeling:** a hand peeler removes about 2 mm, which is 12.1–14.6 % of the mass (`con22`).

## Mechanics and behaviour
- **Elastic constants:** E ≈ 4.3–4.9 MPa for both phloem and xylem; ν 0.47–0.48. E falls by a factor of 1.32 in 7 days in the fridge (`vob24`, after Stopa 2017).
- **Whole root** (`jah18`):
  - three-point bending Fmax 48.6 N at 21.9 mm;
  - straight blade 41.1 N;
  - compression 71.9 N;
  - density 1.04 g/cm³.
- **Fracture energy:** 40–400 J/m² for mixed produce (`alv00`).
- **Turgor:** about 0.7 MPa when fresh. Turgor-driven firmness is about 0 below a relative water content of 0.95; that is the limp carrot (`herp99_calbo00`).
- **Softening during cooking:** second-order fractional conversion with k 0.035 → 1.453 /min from 80 to 110 °C and Ea 138.9 kJ/mol (`peng14`).
  - Twin with a conduction lag: 6 mm rounds are crisp-tender at 4.8 min (sources: 4–5 min) and tender at 6.1 min (sources: '+1–2 min').
  - Whole roots: 16 min (sources: 10–15 min, which is +1 min outside the range).
- **Storage:**
  - 0–1 °C at 98–100 % RH keeps 7–9 months; bagged carrots keep 2–3 weeks.
  - Nantes types crack and break at the tip easily (`hb66`).
- **White blush** on cut faces is reversible for up to about 1–1.5 h and irreversible after about 15 h (`sim09_cz95`).

## Kitchen actions
- **Peeling:** hold the carrot tip-down over a bowl, stroke away from the body, and rotate after each stroke.
- **Before cutting:** top and tail, cut a plank so the carrot can't roll, and cut it into 5–7.5 cm lengths.
- **Cut sizes:**
  - rounds 5–6 mm;
  - julienne 3×3×45 mm;
  - brunoise 3 mm;
  - batonnet 6×6×55 mm;
  - dice 6, 12 or 19 mm (`carrot_cuts`).
- **Boiling:** 6 mm rounds take 4–5 min (`carrot_boil`).
- **Robot results:**
  - MORPHeus peeled 90.1 % of the carrot's area. Skin and flesh look alike, so it relies on force and vibration sensing (`morpheus_2024`).
  - Chopping with a serrated knife fails; sawing works (`luo_2026`).

## Gaps
- Periderm thickness (Knowles & Flore 1983).
- Phloem/xylem radii (the Bath Nantes dataset).
- Lenticel spacing.
- Whole-root water loss per day.
- Xylem and skin L*a*b*.
- Peeler stroke count.
