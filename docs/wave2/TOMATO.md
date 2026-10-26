# Round / truss tomato: research notes

The model is `food/tomato.py`, the hidden-state models are in `food/tomato_models.py`, and the register is `params.TOMATO`.

## Anatomy
From the outside in (`ucd_rost`, `meta19`):
- the skin (cuticle and epidermis);
- the outer pericarp wall;
- the septa (radial walls);
- the columella with the placenta;
- the locules, filled with gel around the seeds.

Locule counts: small types often have 2, globe types 3–5. The twin draws 2–4. Truss tomatoes keep the calyx and pedicel.

## Numbers
- **Size:** medium 66 mm / 123 g, small 61 mm / 91 g (`fdc_tomato`).
- **Pericarp:** 5.3 mm on a small cultivar.
- **Cuticle:** 9.9–15.7 µm on red fruit (`bn05`).
- **Mass shares:** pericarp about 50 %, locular tissue 23 % (`meta19`). The twin gives 50 %.
- **Plate firmness, initial slope:** green 15.1, pink 6.0, red 4.5 N/mm.
- **Skin rupture (2 mm probe):** 6.92, 3.99 and 2.74 N for green, pink and red (`ince16`).
- **Skin modulus:** 252–293 MPa, failing at about 25 MPa and about 10 % strain (`bn05`).
- **Damage:** visible from about 3 % strain (`tomato_mech`).
- **Colour classes:** the USDA's 6 classes (`cfr_tomato`). Skin hue per class is derived from the a\*/b\* bands of `ince16`, ending at 43° for red; the reported ripe hue is about 40°.
- **Chilling:** below 12.5 °C. After 8 days at 5 °C, 65 % of the aroma volatiles are lost (`zhang16`).
- **Kitchen:**
  - Slices are 6–13 mm thick.
  - Use a serrated or very sharp knife.
  - To peel, blanch for 30–45 s after scoring an X 2.5 cm across (`tomato_howto`).

## Bench
`scripts/sim_tomato_squeeze.py` cross-checks two studies:
- **Analytically it holds.** With the Hertz moduli from `tomato_mech` (0.58–1.97 MPa) and two contacts between the plates, the red fruit comes out at 4.5 N/mm, matching the measured 4.5 (`ince16`). The green fruit comes out at 10.2 N/mm against the measured 13.2–15.1.
- **PhysX FEM can't resolve it.** All lanes give about 1.8 N/mm whatever the modulus, which is solver under-convergence. See `test_reports/tomato_squeeze*.json`.

## Gaps
- Days per ripeness stage.
- Seed size.
- Gel viscosity, and how much juice runs out per cut.
- Knife cutting force.
- Absolute L\*a\*b\* values.
- Density of round tomatoes.
