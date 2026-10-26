# Potato: research notes

The model is `food/potato.py`, the hidden-state models are in `food/potato_models.py`, and the register is `params.POTATO`.

## Anatomy
The tuber is a stem. From the outside in:
- the periderm, with the phellem 46–84 µm thick in 6–10 cell layers (`mc20`);
- lenticels, up to 12 per cm² (`mag25`);
- eyes, which are axillary buds in the scars of scale leaves, arranged on a spiral and about 7× denser at the rose end (`mag25`);
- the cortex;
- the vascular ring;
- the perimedullary zone, the main starch store;
- the pith, whose branches run out to the eyes.

The cultivars are from the Bundessortenamt list: Linda is long-oval, Belana oval, Gala round-oval, all yellow-fleshed with flat eyes (`bsl21`).

## Numbers
- **Mechanics:** Young's modulus 3.1–5.6 MPa, Poisson's ratio 0.45, fracture at about 1.1–1.4 MPa (`potato_mech`).
- **Thermal:**
  - density 1053 kg/m³;
  - conductivity 0.55 W/m·K;
  - heat capacity 3.43 kJ/kg·K;
  - diffusivity 0.15e-6 m²/s (`kk18`).
- **Softening:** first-order, k(100 °C) = 0.28–0.40 per minute with Ea 103–145 kJ/mol. Starch gelatinises from 61 to about 80 °C (`potato_cook`).
- **Twin boiling model:** a conduction lag plus first-order softening at the centre, starting from cold water.
  - Calibrated: 2 cm cubes are done at 17.5 min (sources: 15–20).
  - Check: 3 cm cubes at 22 min; a large 75 mm tuber at 50 min (sources: up to 60).
  - Check: a whole 55 mm tuber at 34 min, against 20–25 min in German guides. The model runs long here.
- **Peel:** a 1.1 mm peel removes **10.0 %** of the twin's mass, against about 9 % measured (`potato_peel`). This is an independent check of the geometry.
- **Storage:**
  - below 8 °C the tuber sweetens;
  - under retail light it greens within 1–2.5 days at 20 °C;
  - BfR: remove green parts and eyes generously (`potato_storage`).

## Gaps
- Phyllotaxis fraction: the twin uses 2/5 as ILLUSTRATIVE.
- Depth of the vascular ring and share of pith (Reeve 1969 and Böl 2020 are paywalled).
- L\*a\*b\* values.
- Browning of cut faces over time (ΔE(t)).
- Cutting force.
